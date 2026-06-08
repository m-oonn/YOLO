# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please follow these steps:

### For Security Researchers

1. **Do Not** create a public GitHub issue for the vulnerability
2. Go to the repository's **Security** tab (`Security` → **` Advisories`**)
3. Click **Report a vulnerability**
4. Fill out the vulnerability report form with:
   - Type of vulnerability
   - Full paths of source file(s) related to the vulnerability
   - Location of the affected source code (tag/commits)
   - Step-by-step instructions to reproduce
   - Proof-of-concept or exploit code (if possible)
   - Impact assessment

### What to Expect

- **Acknowledgment**: Within 48 hours, a maintainer will acknowledge receipt
- **Initial Assessment**: Within 7 days, we'll assess the vulnerability and determine severity
- **Fix Timeline**: For confirmed vulnerabilities:
  - Critical: Fix within 7 days
  - High: Fix within 30 days
  - Medium/Low: Fix within 90 days
- **Disclosure**: Credit will be given to reporters (unless anonymity requested)

## Security Best Practices

When deploying this project, please:

1. **Change Default Ports** - Don't expose services on default ports in production
2. **Enable Authentication** - Add API authentication before production deployment
3. **Limit CORS** - Configure allowed origins specifically, not with `*`
4. **Secure File Uploads** - Validate file types and sizes server-side
5. **Use HTTPS** - Enable TLS/SSL in production
6. **Environment Variables** - Never commit secrets to version control

## Known Limitations

- **No Built-in Authentication**: The API lacks authentication by default. Add JWT/API key auth before production use.
- **SQLite for Storage**: Not suitable for high-concurrency multi-node deployments. Consider PostgreSQL for scale.
- **YOLO Model Weights**: The YOLO model files are downloaded separately. Ensure you use official sources.

## Dependencies

This project uses third-party dependencies. Before deploying:

```bash
# Audit Python dependencies
pip audit

# Audit Node.js dependencies
cd frontend
npm audit
```

Update vulnerable packages promptly.
