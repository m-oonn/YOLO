# Changelog

All notable changes to the YOLO Detection System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Security module (`backend/security.py`) with path traversal prevention
- Authentication module (`backend/auth.py`) with API key support
- Comprehensive monitoring endpoint (`/api/detection/monitoring`)
- Locked dependencies (`requirements.lock.txt`)

### Changed
- Enhanced WebSocket to push status updates proactively
- Frontend reduced polling frequency when WebSocket is connected
- File upload validation now includes MIME type verification

### Fixed
- Path traversal vulnerabilities in model management
- File upload extension bypass vulnerabilities

## [1.0.0] - 2025-01-01

### Added
- Real-time object detection with YOLOv12
- Multi-object tracking with ByteTrack
- Behavior detection rules (running, fall detection, crowd gathering, intrusion, fighting)
- MLLM-powered scene understanding and alarm enhancement
- REST API with FastAPI
- Web-based monitoring dashboard
- Event storage and retrieval
- GPU acceleration support

[Unreleased]: https://github.com/yolo-course/design/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yolo-course/design/releases/tag/v1.0.0
