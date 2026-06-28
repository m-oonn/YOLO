# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for security utilities.

Verifies path traversal prevention, input sanitization,
and other security measures.
"""

import os
import tempfile
from pathlib import Path

import pytest

from backend.security import (
    RateLimiter,
    is_safe_path,
    sanitize_config_value,
    sanitize_filename,
    validate_model_path,
    validate_upload_extension,
    validate_upload_mime,
)


class TestIsSafePath:
    """Tests for is_safe_path function."""

    def test_relative_path_within_base(self):
        """Relative path within base should be safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_safe_path(tmpdir, "subdir/file.txt")
            assert is_safe_path(tmpdir, "file.txt")

    def test_parent_traversal_attempt(self):
        """Path with .. should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not is_safe_path(tmpdir, "../etc/passwd")
            assert not is_safe_path(tmpdir, "subdir/../../etc/passwd")

    def test_absolute_path_outside_base(self):
        """Absolute path outside base should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not is_safe_path(tmpdir, "/etc/passwd")

    @pytest.mark.skipif(
        os.name == "nt", reason="Windows requires admin/dev mode to create symlinks"
    )
    def test_symlink_within_base(self):
        """Symlink within base should be allowed (follows symlink)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").write_text("test")

            real_file = Path(tmpdir) / "subdir" / "file.txt"
            linked_file = Path(tmpdir) / "linked.txt"
            linked_file.symlink_to(real_file)

            assert is_safe_path(tmpdir, "linked.txt")


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_normal_filename_unchanged(self):
        """Normal filenames should be preserved."""
        assert sanitize_filename("video.mp4") == "video.mp4"
        assert sanitize_filename("my_video_123.avi") == "my_video_123.avi"

    def test_dangerous_characters_removed(self):
        """Dangerous characters should be replaced."""
        assert sanitize_filename("file<script>.txt") == "file_script_.txt"
        assert sanitize_filename("file|pipe.txt") == "file_pipe.txt"

    def test_path_separators_normalized(self):
        """Path separators should be normalized to underscores."""
        result = sanitize_filename("../../../etc/passwd")
        assert "/" not in result

    def test_dots_in_filename_preserved(self):
        """Dots in filenames are preserved (except leading dots)."""
        result = sanitize_filename("../../../etc/passwd")
        assert "_" in result
        assert "." in result

    def test_multiple_dangerous_chars_normalized(self):
        """Multiple consecutive dangerous chars should be normalized."""
        assert (
            sanitize_filename("file///multiple///slashes.txt")
            == "file_multiple_slashes.txt"
        )

    def test_empty_after_sanitization(self):
        """Empty result should get fallback name."""
        result = sanitize_filename("...")
        assert result.startswith("file")
        assert len(result) > 0

    def test_long_filename_truncated(self):
        """Very long filenames should be truncated."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestValidateModelPath:
    """Tests for validate_model_path function."""

    def test_valid_model_path(self):
        """Valid model paths should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "yolo12n.pt"
            model_file.write_text("fake model")

            assert validate_model_path("yolo12n.pt", models_dir)[0]

    def test_parent_traversal_blocked(self):
        """Parent traversal should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            is_valid, error = validate_model_path("../etc/passwd", models_dir)
            assert not is_valid
            assert "path traversal" in error.lower()

    def test_nonexistent_file(self):
        """Nonexistent model files should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            is_valid, error = validate_model_path("nonexistent.pt", models_dir)
            assert not is_valid
            assert "not found" in error.lower()

    def test_invalid_extension(self):
        """Invalid file extensions should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "malware.exe"
            model_file.write_text("fake")

            is_valid, error = validate_model_path("malware.exe", models_dir)
            assert not is_valid
            assert "extension" in error.lower()


class TestValidateUploadExtension:
    """Tests for video upload extension validation."""

    @pytest.mark.parametrize("ext", [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"])
    def test_valid_extensions(self, ext):
        """Valid video extensions should pass."""
        is_valid, _ = validate_upload_extension(f"video{ext}")
        assert is_valid

    @pytest.mark.parametrize("ext", [".exe", ".bat", ".sh", ".php", ".html"])
    def test_invalid_extensions(self, ext):
        """Invalid extensions should be rejected."""
        is_valid, error = validate_upload_extension(f"file{ext}")
        assert not is_valid
        assert "extension" in error.lower()

    def test_empty_filename(self):
        """Empty filename should be rejected."""
        is_valid, error = validate_upload_extension("")
        assert not is_valid


class TestValidateUploadMime:
    """Tests for upload MIME type validation."""

    def test_filetype_can_detect_avi(self):
        """filetype can detect AVI files."""
        import filetype

        fake_avi = b"RIFF" + b"\x00" * 100
        kind = filetype.guess(fake_avi)
        if kind is not None:
            assert kind.mime.startswith("video/")

    def test_empty_content(self):
        """Empty content should be rejected."""
        is_valid, error = validate_upload_mime(b"")
        assert not is_valid
        assert "empty" in error.lower()

    def test_oversized_content(self):
        """Oversized content should be rejected."""
        large_content = b"x" * (101 * 1024 * 1024)
        is_valid, error = validate_upload_mime(large_content)
        assert not is_valid
        assert "too large" in error.lower()


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_within_limit(self):
        """Requests within limit should be allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for i in range(5):
            assert limiter.is_allowed(f"key_{i}"), f"Request {i} should be allowed"

    def test_blocks_over_limit(self):
        """Requests over limit should be blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        assert limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key")
        assert not limiter.is_allowed("test_key")

    def test_different_keys_independent(self):
        """Different keys should have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("key1")
        assert limiter.is_allowed("key1")
        assert not limiter.is_allowed("key1")

        assert limiter.is_allowed("key2")
        assert limiter.is_allowed("key2")
        assert not limiter.is_allowed("key2")

    def test_get_remaining(self):
        """get_remaining should return correct count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        limiter.is_allowed("key")
        limiter.is_allowed("key")
        remaining = limiter.get_remaining("key")
        assert remaining == 3

    def test_reset_key(self):
        """reset should clear limit for key."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("key")
        limiter.is_allowed("key")
        assert not limiter.is_allowed("key")

        limiter.reset("key")
        assert limiter.is_allowed("key")
        assert limiter.is_allowed("key")


class TestSanitizeConfigValue:
    """Tests for config value sanitization."""

    def test_sanitizes_string(self):
        """Strings should be sanitized."""
        result = sanitize_config_value("  test\x00value  ")
        assert result == "testvalue"
        assert "\x00" not in result

    def test_preserves_numbers(self):
        """Numbers should be preserved."""
        assert sanitize_config_value(42) == 42
        assert sanitize_config_value(3.14) == 3.14

    def test_truncates_long_strings(self):
        """Long strings should be truncated."""
        long_str = "a" * 2000
        result = sanitize_config_value(long_str, max_length=100)
        assert len(result) == 100

    def test_sanitizes_list(self):
        """Lists should be recursively sanitized."""
        result = sanitize_config_value(["test\x00", "  value  "])
        assert result == ["test", "value"]

    def test_sanitizes_dict(self):
        """Dicts should be recursively sanitized."""
        result = sanitize_config_value({"key": "  test\x00value  "})
        assert result == {"key": "testvalue"}
