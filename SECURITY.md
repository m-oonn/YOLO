# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within this project, please report it responsibly.

### Preferred Method: GitHub Security Advisories

1. Go to the repository's **Security** tab
2. Click **"Report a vulnerability"**
3. Fill in the description and submit

### Alternative: Private Email

If you cannot use GitHub Security Advisories, please contact the maintainers directly. Do NOT open a public issue.

### What to Include

- A clear description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Potential impact
- Any suggested fixes (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours of receipt
- **Assessment**: Within 5 business days
- **Fix timeline**: Depends on severity — critical issues will be prioritized

## Security Best Practices for Deployment

When deploying this system in production:

1. **Change default credentials** — update API keys and admin passwords
2. **Enable HTTPS** — never expose the API over plain HTTP in production
3. **Restrict CORS** — configure `backend/main.py` CORS origins to trusted domains only
4. **Use a reverse proxy** — place Nginx or similar in front of the FastAPI server
5. **Run as non-root user** — avoid running services with elevated privileges
6. **Keep dependencies updated** — regularly run `pip list --outdated` and `npm outdated`
7. **Enable rate limiting** — the system includes slowapi; configure appropriate limits
8. **Review file permissions** — ensure `events.db` and `outputs/` have restricted access

## Known Limitations

- **Authentication**: The system uses API key-based authentication, which is suitable for internal deployments but insufficient for public-facing services without additional layers
- **SQLite Storage**: SQLite is used for simplicity. For multi-user or high-throughput deployments, consider migrating to PostgreSQL
- **Model Security**: Downloaded pre-trained models may contain vulnerabilities. Only use models from trusted sources (Ultralytics, HuggingFace)

## Dependency Auditing

Audit dependencies regularly:

```bash
# Python
pip-audit
# or
safety check

# Node.js / Frontend
cd course-design/frontend
npm audit
```

## Disclosure Policy

We follow responsible disclosure:
- Security issues will be acknowledged and fixed promptly
- Credit will be given to reporters (unless anonymity is requested)
- Public disclosure will occur after a fix is available
