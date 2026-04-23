"""Tests for API key authentication module."""

import os
import sys
from importlib import reload
from unittest.mock import patch

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def reset_api_key_env():
    """Reset API_KEY env var after each test."""
    original = os.environ.get("API_KEY")
    yield
    if original is None:
        os.environ.pop("API_KEY", None)
    else:
        os.environ["API_KEY"] = original


class TestGenerateApiKey:
    """Test API key generation."""

    def test_generate_api_key_returns_string(self):
        from backend.api.auth import generate_api_key

        key = generate_api_key()
        assert isinstance(key, str)
        assert len(key) > 32  # token_urlsafe(32) produces >32 chars

    def test_generate_api_key_unique(self):
        from backend.api.auth import generate_api_key

        key1 = generate_api_key()
        key2 = generate_api_key()
        assert key1 != key2


class TestGetApiKey:
    """Test API key extraction from header/query."""

    def test_from_header(self):
        from backend.api.auth import get_api_key

        result = get_api_key(header_key="header-key", query_key=None)
        assert result == "header-key"

    def test_from_query(self):
        from backend.api.auth import get_api_key

        result = get_api_key(header_key=None, query_key="query-key")
        assert result == "query-key"

    def test_header_takes_priority(self):
        from backend.api.auth import get_api_key

        result = get_api_key(header_key="header-key", query_key="query-key")
        assert result == "header-key"

    def test_both_none(self):
        from backend.api.auth import get_api_key

        result = get_api_key(header_key=None, query_key=None)
        assert result is None


class TestAuthenticateApi:
    """Test API key authentication."""

    def test_no_key_configured_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("API_KEY", None)
            reload(__import__("backend.api.auth", fromlist=["authenticate_api"]))
            from backend.api.auth import authenticate_api

            result = authenticate_api(api_key=None)
            assert result is None

    def test_valid_key_returns_key(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            reload(__import__("backend.api.auth", fromlist=["authenticate_api"]))
            from backend.api.auth import authenticate_api

            result = authenticate_api(api_key="secret123")
            assert result == "secret123"

    def test_invalid_key_raises_exception(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            reload(__import__("backend.api.auth", fromlist=["authenticate_api"]))
            from backend.api.auth import authenticate_api

            with pytest.raises(HTTPException) as exc_info:
                authenticate_api(api_key="wrong-key")
            assert exc_info.value.status_code == 401
            assert "Invalid or missing API key" in exc_info.value.detail

    def test_missing_key_raises_exception(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            reload(__import__("backend.api.auth", fromlist=["authenticate_api"]))
            from backend.api.auth import authenticate_api

            with pytest.raises(HTTPException) as exc_info:
                authenticate_api(api_key=None)
            assert exc_info.value.status_code == 401


class TestRequireApiKey:
    """Test require_api_key dependency."""

    def test_valid_key(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            reload(__import__("backend.api.auth", fromlist=["require_api_key"]))
            from backend.api.auth import require_api_key

            result = require_api_key(api_key="secret123")
            assert result == "secret123"

    def test_no_auth_configured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("API_KEY", None)
            reload(__import__("backend.api.auth", fromlist=["require_api_key"]))
            from backend.api.auth import require_api_key

            result = require_api_key(api_key=None)
            assert result == ""

    def test_missing_key_raises(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            reload(__import__("backend.api.auth", fromlist=["require_api_key"]))
            from backend.api.auth import require_api_key

            with pytest.raises(HTTPException) as exc_info:
                require_api_key(api_key=None)
            assert exc_info.value.status_code == 401


class TestAuthMiddlewareIntegration:
    """Test middleware public endpoint logic."""

    def test_public_prefixes_list(self):
        """Verify public prefixes are defined."""
        from backend import main

        assert "/" in main._PUBLIC_PREFIXES
        assert "/health" in main._PUBLIC_PREFIXES
        assert "/docs" in main._PUBLIC_PREFIXES
        assert "/api/detection/stream.mjpg" in main._PUBLIC_PREFIXES
        assert "/api/detection/stream" in main._PUBLIC_PREFIXES
