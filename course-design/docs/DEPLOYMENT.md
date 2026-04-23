# Production Deployment Guide

This guide covers deploying the YOLO Real-time Object Detection & Behavior Analysis System in a production environment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [Docker Production Deployment](#docker-production-deployment)
4. [Nginx Reverse Proxy](#nginx-reverse-proxy)
5. [SSL/TLS Configuration](#ssltls-configuration)
6. [System Service Setup](#system-service-setup)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Hardening](#security-hardening)
9. [Backup & Recovery](#backup--recovery)
10. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | NVIDIA GPU with 4GB VRAM | NVIDIA GPU with 8GB+ VRAM (RTX 3060+) |
| Storage | 50 GB SSD | 200 GB SSD (for video storage) |

### Software Requirements

- **OS**: Ubuntu 22.04 LTS (recommended) or Debian 11+
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **Nginx**: 1.24+
- **CUDA**: 11.8+ (if using GPU)
- **nvidia-docker**: 2.0+ (if using GPU)

---

## Server Setup

### 1. Install Docker & Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin
```

### 2. Install NVIDIA Container Toolkit (GPU only)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. Clone and Configure

```bash
git clone https://github.com/m-oonn/YOLO.git
cd YOLO/course-design

# Copy environment template
cp .env.example .env

# Edit .env with production values
nano .env
```

### Production `.env` Example

```env
# Backend
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=info

# Model
MODEL_PATH=models/yolov8n.pt
DEVICE=0

# Database
DB_PATH=/data/yolo/events.db

# CORS (comma-separated origins)
ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Upload limits
MAX_UPLOAD_SIZE_MB=500

# API Key (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
API_KEY=your-secure-api-key-here
```

---

## Docker Production Deployment

### docker-compose.yml (Production)

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: yolo-backend
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - WORKERS=4
      - LOG_LEVEL=info
      - DB_PATH=/data/events.db
    volumes:
      - yolo-data:/data
      - yolo-models:/app/models
      - yolo-uploads:/app/uploads
      - yolo-logs:/app/logs
      - /dev/video0:/dev/video0  # Camera device (if needed)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  frontend:
    build:
      context: frontend
      dockerfile: Dockerfile
    container_name: yolo-frontend
    restart: unless-stopped
    expose:
      - "80"

  nginx:
    image: nginx:1.24-alpine
    container_name: yolo-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend

volumes:
  yolo-data:
  yolo-models:
  yolo-uploads:
  yolo-logs:
```

### Deploy

```bash
# Download YOLO model
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/yolov8n.pt

# Build and start
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend
```

---

## Nginx Reverse Proxy

### nginx.conf

```nginx
worker_processes auto;
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent"';
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
    limit_req_zone $binary_remote_addr zone=stream:10m rate=1r/s;

    # Upstream
    upstream backend {
        server 127.0.0.1:8000;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$host$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Frontend static files
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Increase timeout for long-running detection
            proxy_read_timeout 3600s;
            proxy_connect_timeout 60s;
        }

        # MJPEG video stream
        location /api/detection/stream {
            limit_req zone=stream burst=2 nodelay;
            proxy_pass http://backend;
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 3600s;
        }

        # WebSocket
        location /api/ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 86400s;
        }

        # Block access to sensitive files
        location ~ /\.(git|env) {
            deny all;
        }
    }
}
```

---

## SSL/TLS Configuration

### Using Let's Encrypt (Certbot)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot certonly --nginx -d your-domain.com

# Auto-renewal (already handled by certbot)
sudo crontab -e
# Add: 0 3 * * * certbot renew --quiet
```

---

## System Service Setup

### systemd Service (Non-Docker)

If not using Docker, create `/etc/systemd/system/yolo-detection.service`:

```ini
[Unit]
Description=YOLO Detection Service
After=network.target

[Service]
Type=simple
User=yolo
WorkingDirectory=/opt/yolo/course-design
Environment="PATH=/opt/yolo/venv/bin"
ExecStart=/opt/yolo/venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=append:/var/log/yolo/backend.log
StandardError=append:/var/log/yolo/error.log

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable yolo-detection
sudo systemctl start yolo-detection
sudo systemctl status yolo-detection
```

---

## Monitoring & Logging

### Docker Logging

```bash
# View logs
docker compose -f docker-compose.prod.yml logs -f backend

# Log rotation (add to /etc/docker/daemon.json)
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
```

### Health Check Endpoints

- `GET /api/health` — Backend health (returns `{"status": "healthy"}`)
- `GET /api/detection/status` — Detection pipeline status

### Prometheus Metrics (Future)

Consider adding `prometheus-fastapi-instrumentator` for metrics collection.

---

## Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

### 2. Disable Root Login

```bash
# Edit /etc/ssh/sshd_config
PermitRootLogin no
PasswordAuthentication no
sudo systemctl restart sshd
```

### 3. Fail2ban

```bash
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 4. Regular Updates

```bash
# Create /etc/apt/apt.conf.d/50unattended-upgrades
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Backup & Recovery

### Database Backup

```bash
# Backup script: /opt/yolo/scripts/backup.sh
#!/bin/bash
BACKUP_DIR="/opt/yolo/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_PATH="/var/lib/docker/volumes/yolo-data/_data/events.db"

mkdir -p $BACKUP_DIR
cp $DB_PATH $BACKUP_DIR/events_${DATE}.db
# Keep last 7 days
find $BACKUP_DIR -name "events_*.db" -mtime +7 -delete
```

### Cron Job (Daily Backup)

```bash
# crontab -e
0 2 * * * /opt/yolo/scripts/backup.sh
```

### Restore

```bash
docker compose -f docker-compose.prod.yml stop backend
cp /opt/yolo/backups/events_20250101_020000.db /var/lib/docker/volumes/yolo-data/_data/events.db
docker compose -f docker-compose.prod.yml start backend
```

---

## Performance Tuning

### GPU Memory Management

```python
# In production config, set device explicitly
DEVICE=0  # GPU 0

# For multi-GPU setups
DEVICE=0,1  # Use GPUs 0 and 1
```

### Worker Configuration

| Workers | Use Case |
|---------|----------|
| 1 | Single camera, low traffic |
| 2-4 | Multiple cameras, moderate traffic |
| 4+ | High traffic, multiple concurrent streams |

### Database Optimization

```sql
-- For large event databases, add indexes
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_s);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, timestamp_s);
```

### Nginx Buffer Tuning

```nginx
# Add to http block
client_max_body_size 500M;
client_body_buffer_size 128k;
proxy_buffer_size 128k;
proxy_buffers 4 256k;
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| GPU not detected | Check `nvidia-smi`, verify `nvidia-docker2` installed |
| Port already in use | Change `PORT` in `.env` or stop conflicting service |
| WebSocket disconnects | Check Nginx WebSocket proxy config |
| MJPEG stream lag | Reduce FPS in config, check network bandwidth |
| Database locked | Ensure only one process accesses the DB |
| OOM Killer | Increase RAM, reduce batch size, limit workers |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
docker compose -f docker-compose.prod.yml restart backend

# View debug logs
docker compose -f docker-compose.prod.yml logs -f --tail=100 backend
```

---

**Last Updated**: 2026-04-23
