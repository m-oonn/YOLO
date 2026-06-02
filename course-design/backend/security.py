# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Security utilities for path validation, input sanitization, and common security checks.

This module provides essential security functions including:
- Path traversal prevention
- Input sanitization
- File upload validation
- Rate limiting

All functions are designed to be thread-safe unless noted otherwise.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants for file upload validation
ALLOWED_VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
ALLOWED_VIDEO_MIME_TYPES: set[str] = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/x-flv",
    "video/x-ms-wmv",
    "video/webm",
}
MAX_UPLOAD_SIZE_MB: int = 100
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def is_safe_path(base_dir: str | Path, target_path: str | Path) -> bool:
    """Check if target_path is safely within base_dir.

    Prevents path traversal attacks (e.g., ../../etc/passwd).

    Args:
        base_dir: The allowed base directory.
        target_path: The path to validate.

    Returns:
        True if target_path is within base_dir, False otherwise.

    Example:
        >>> is_safe_path("/app/uploads", "user123/file.mp4")
        True
        >>> is_safe_path("/app/uploads", "../../etc/passwd")
        False
    """
    base = Path(base_dir).resolve()
    try:
        target = (base / target_path).resolve()
        return target.is_relative_to(base)
    except (ValueError, OSError):
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing potentially dangerous characters.

    Args:
        filename: The original filename.

    Returns:
        A sanitized filename safe for filesystem operations.

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        '_etc_passwd'
        >>> sanitize_filename("video (1).mp4")
        'video_1_.mp4'
    """
    sanitized = re.sub(r"[^\w\-.]", "_", filename)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")

    if not sanitized or sanitized.startswith("."):
        sanitized = f"file_{sanitized}" if sanitized else "unnamed"

    return sanitized[: 255]


def validate_model_path(model_path: str, models_dir: str | Path) -> tuple[bool, str]:
    """Validate a model path to prevent path traversal.

    Args:
        model_path: The model path to validate.
        models_dir: The allowed models directory.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if not model_path:
        return False, "Model path cannot be empty"

    if ".." in model_path or model_path.startswith("/"):
        return False, "Invalid model path: path traversal not allowed"

    if not is_safe_path(models_dir, model_path):
        return False, "Invalid model path: outside allowed directory"

    full_path = Path(models_dir) / model_path
    if not full_path.exists():
        return False, f"Model file not found: {model_path}"

    allowed_extensions = {".pt", ".onnx", ".engine", ".tflite", ".pth"}
    if full_path.suffix.lower() not in allowed_extensions:
        return False, f"Invalid model file extension: {full_path.suffix}"

    return True, ""


def sanitize_config_value(value: Any, max_length: int = 1000) -> Any:
    """Sanitize configuration values to prevent injection attacks.

    Args:
        value: The value to sanitize.
        max_length: Maximum allowed string length.

    Returns:
        The sanitized value.
    """
    if isinstance(value, str):
        value = value[:max_length]
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        return value.strip()
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, list):
        return [sanitize_config_value(v, max_length) for v in value[:100]]
    elif isinstance(value, dict):
        return {k: sanitize_config_value(v, max_length) for k, v in value.items() if isinstance(k, str)}
    return value


def validate_upload_extension(filename: str) -> tuple[bool, str]:
    """Validate video file extension.

    Args:
        filename: The filename to validate.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if not filename:
        return False, "Filename cannot be empty"

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Invalid file extension: {ext}. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"

    return True, ""


def validate_upload_mime(content: bytes, mime_type: str | None = None) -> tuple[bool, str]:
    """Validate uploaded file content and MIME type.

    Args:
        content: The file content bytes.
        mime_type: Optional MIME type to validate.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if not content:
        return False, "Empty file content"

    if len(content) > 100 * 1024 * 1024:
        return False, "File too large (max 100MB)"

    import filetype

    kind = filetype.guess(content)
    if kind is None:
        if mime_type and mime_type.startswith("video/"):
            return True, ""
        return False, "Unable to determine file type"

    if kind.mime not in ALLOWED_VIDEO_MIME_TYPES:
        return False, f"File content is not a video (detected: {kind.mime})"

    return True, ""


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints.
    
    Thread-safe implementation with automatic cleanup of expired entries.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if a request from key is allowed under rate limits."""
        now = time.time()
        with self._lock:
            if key not in self._requests:
                self._requests[key] = []

            self._requests[key] = [t for t in self._requests[key] if now - t < self._window_seconds]

            if not self._requests[key]:
                del self._requests[key]
                self._requests[key] = []

            if len(self._requests[key]) >= self._max_requests:
                return False

            self._requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key in current window."""
        now = time.time()
        with self._lock:
            if key not in self._requests:
                return self._max_requests

            active_requests = [t for t in self._requests[key] if now - t < self._window_seconds]
            return max(0, self._max_requests - len(active_requests))

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        with self._lock:
            if key in self._requests:
                del self._requests[key]


_api_rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


def get_api_rate_limiter() -> RateLimiter:
    """Get the global API rate limiter instance."""
    return _api_rate_limiter
