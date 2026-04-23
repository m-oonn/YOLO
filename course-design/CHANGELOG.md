# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI/CD pipeline with GitHub Actions
- Docker build and push workflow
- ESLint 9 flat config for frontend
- Prettier code formatting
- Ruff linting for Python
- Pre-commit hooks
- Additional integration tests
- `requirements-dev.txt` for development dependencies
- `pyproject.toml` for project configuration

### Changed
- Frontend `package.json` scripts updated with lint and format commands
- Test coverage threshold raised to 80%

### Security
- MIT License added
- SECURITY.md policy document added

## [1.0.0] - 2025-01-01

### Added
- Real-time object detection using YOLOv8/v11
- 5 behavior detection rules:
  - Running detection
  - Fall detection
  - Crowd detection
  - Intrusion detection
  - Fight detection
- ByteTrack multi-object tracking
- FastAPI backend with REST API
- Vue.js 3 frontend with Element Plus
- SQLite event storage
- MJPEG and WebSocket real-time streaming
- Docker and docker-compose deployment
- Unit tests for core modules
- YAML-based configuration system

[unreleased]: https://github.com/m-oonn/YOLO/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/YOUR_USERNAME/YOUR_REPO/releases/tag/v1.0.0
