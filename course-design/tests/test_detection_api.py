# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Detection API endpoints.

These tests verify core API behaviors including path traversal prevention,
file upload validation, and proper error handling.
"""

from unittest.mock import patch

import pytest


class TestSecurityPathTraversal:
    """Tests for path traversal prevention in model endpoints."""

    def test_blocks_parent_traversal_in_switch(self):
        """Model switch should block parent traversal attempts."""
        from backend.security import validate_model_path

        is_valid, error = validate_model_path("../../../etc/passwd", "/some/models/dir")
        assert not is_valid
        assert "path traversal" in error.lower() or "invalid" in error.lower()

    def test_blocks_absolute_paths(self):
        """Model switch should block absolute paths."""
        from backend.security import validate_model_path

        is_valid, error = validate_model_path("/etc/passwd", "/some/models/dir")
        assert not is_valid

    def test_allows_relative_paths_within_base(self):
        """Relative paths within models dir should be allowed."""
        from backend.security import validate_model_path

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_relative_to", return_value=True),
        ):
            is_valid, _ = validate_model_path("yolo12n.pt", "/project/models")
            assert is_valid


class TestSecurityFileUpload:
    """Tests for file upload validation."""

    @pytest.mark.parametrize(
        "ext", [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    )
    def test_allows_valid_video_extensions(self, ext):
        """Valid video extensions should be accepted."""
        from backend.security import validate_upload_extension

        is_valid, _ = validate_upload_extension(f"video{ext}")
        assert is_valid

    @pytest.mark.parametrize(
        "ext", [".exe", ".bat", ".sh", ".php", ".html", ".js", ".py"]
    )
    def test_rejects_dangerous_extensions(self, ext):
        """Dangerous file extensions should be rejected."""
        from backend.security import validate_upload_extension

        is_valid, error = validate_upload_extension(f"file{ext}")
        assert not is_valid
        assert "extension" in error.lower()


class TestSecuritySanitization:
    """Tests for input sanitization functions."""

    def test_sanitize_filename_removes_path_chars(self):
        """Filename sanitization should remove path separators."""
        from backend.security import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert "/" not in result
        assert "_" in result

    def test_sanitize_filename_preserves_extension(self):
        """Sanitization should preserve file extensions."""
        from backend.security import sanitize_filename

        result = sanitize_filename("malicious<script>.mp4")
        assert result.endswith(".mp4")

    def test_sanitize_config_removes_null_bytes(self):
        """Config sanitization should remove null bytes."""
        from backend.security import sanitize_config_value

        result = sanitize_config_value("test\x00value")
        assert "\x00" not in result


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_blocks_after_limit_exceeded(self):
        """Rate limiter should block requests after limit."""
        from backend.security import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)

        assert limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key")
        assert not limiter.is_allowed("test_key")

    def test_different_keys_independent(self):
        """Different rate limit keys should be independent."""
        from backend.security import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)

        assert limiter.is_allowed("key1")
        assert not limiter.is_allowed("key1")
        assert limiter.is_allowed("key2")

    def test_reset_clears_limit(self):
        """Reset should clear rate limit for a key."""
        from backend.security import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("test_key")
        limiter.is_allowed("test_key")
        assert not limiter.is_allowed("test_key")

        limiter.reset("test_key")
        assert limiter.is_allowed("test_key")


class TestSecurityConstants:
    """Tests for security configuration constants."""

    def test_allowed_video_extensions_defined(self):
        """ALLOWED_VIDEO_EXTENSIONS should be properly defined."""
        from backend.security import ALLOWED_VIDEO_EXTENSIONS

        assert ".mp4" in ALLOWED_VIDEO_EXTENSIONS
        assert ".avi" in ALLOWED_VIDEO_EXTENSIONS
        assert len(ALLOWED_VIDEO_EXTENSIONS) > 0

    def test_allowed_mime_types_defined(self):
        """ALLOWED_VIDEO_MIME_TYPES should be properly defined."""
        from backend.security import ALLOWED_VIDEO_MIME_TYPES

        assert "video/mp4" in ALLOWED_VIDEO_MIME_TYPES
        assert len(ALLOWED_VIDEO_MIME_TYPES) > 0
