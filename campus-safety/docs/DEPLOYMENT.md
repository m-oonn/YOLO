# Production Deployment Guide

This document provides instructions for deploying the YOLO Course Design system in a production environment.

## Prerequisites

- Ubuntu 20.04+ / Debian 11+ or CentOS 8+
- Docker 20.10+ and Docker Compose v2
- Nginx 1.18+
- SSL certificates (Let's Encrypt recommended)
- At least 4GB RAM and 20GB disk space

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Nginx (Reverse Proxy)                   │
│                    (SSL termination, :443)                   │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────┐      ┌─────────────────┐
│   Frontend      │      │   Backend       │
│   (Vue.js)      │      │   (FastAPI)     │
│   Port 8080     │      │   Port 8000     │
└─────────────────┘      └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   SQLite DB     │
                         │   (events.db)   │
                         └─────────────────┘
```

## Deployment Steps

### 1. Prepare the Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# API Security
API_KEY=your-secure-api-key-here

# Model Configuration
MODEL_PATH=models/yolov11x.pt

# Camera FPS
CAMERA_FPS=30
```

### 3. Nginx Configuration

Create `/etc/nginx/sites-available/yolo`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Frontend (Static files)
    location / {
        root /var/www/yolo/frontend/dist;
        try_files $uri $uri/ /index.html;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts for long-running detection
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # WebSocket Support
    location /api/detection/stream {
        proxy_pass http://127.0.0.1:8000/api/detection/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }

    # MJPEG Stream
    location /api/detection/stream.mjpg {
        proxy_pass http://127.0.0.1:8000/api/detection/stream.mjpg;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }

    # Health check (no auth required)
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    # Access/Error logs
    access_log /var/log/nginx/yolo_access.log;
    error_log /var/log/nginx/yolo_error.log;
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/yolo /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. SSL Certificate (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com
```

Auto-renewal is enabled by default. Verify with:

```bash
sudo certbot renew --dry-run
```

### 5. Systemd Service (Backend)

Create `/etc/systemd/system/yolo-backend.service`:

```ini
[Unit]
Description=YOLO Course Design Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/yolo
Environment="API_KEY=your-secure-api-key-here"
ExecStart=/var/www/yolo/venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo-backend
sudo systemctl start yolo-backend
sudo systemctl status yolo-backend
```

### 6. Firewall Configuration

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 7. Database Backup

Create `/etc/cron.daily/yolo-backup`:

```bash
#!/bin/bash
BACKUP_DIR="/var/backups/yolo"
mkdir -p $BACKUP_DIR
cp /var/www/yolo/events.db $BACKUP_DIR/events-$(date +%Y%m%d).db
find $BACKUP_DIR -name "events-*.db" -mtime +7 -delete
```

```bash
sudo chmod +x /etc/cron.daily/yolo-backup
```

### 8. Monitoring

Monitor system resources:

```bash
# CPU/Memory
watch -n 5 'free -h && echo "" && df -h'

# Docker logs
docker logs -f yolo-backend

# Nginx status
sudo nginx -T | grep "status_zone"
```

### 9. Log Rotation

Create `/etc/logrotate.d/yolo`:

```
/var/log/nginx/yolo_*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}

/var/www/yolo/outputs/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 0644 www-data www-data
}
```

## Security Checklist

- [ ] Change default API_KEY
- [ ] Enable SSL/TLS (Let's Encrypt)
- [ ] Configure firewall (UFW)
- [ ] Set up database backups
- [ ] Enable log rotation
- [ ] Review Nginx security headers
- [ ] Disable unused services
- [ ] Enable fail2ban for SSH

## Troubleshooting

### Backend won't start

```bash
# Check logs
sudo journalctl -u yolo-backend -f

# Verify environment
cd /var/www/yolo && source venv/bin/activate
python -c "from backend.main import app; print('OK')"
```

### Database corruption

```bash
# Restore from backup
sudo systemctl stop yolo-backend
cp /var/backups/yolo/events-20240101.db /var/www/yolo/events.db
sudo systemctl start yolo-backend
```

### SSL certificate issues

```bash
# Renew manually
sudo certbot renew --force-renewal

# Check certificate expiry
openssl s_client -connect your-domain.com:443 -servername your-domain.com 2>/dev/null | openssl x509 -noout -dates
```

## Updates

```bash
cd /var/www/yolo
git pull origin main

# Rebuild frontend
cd frontend && npm install && npm run build

# Restart backend
sudo systemctl restart yolo-backend
```
