# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Simple API key authentication for FastAPI endpoints."""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# Read API key from environment; generate a random one if not set
_API_KEY: str | None = os.environ.get("API_KEY")

# List of endpoints that do NOT require authentication
PUBLIC_ENDPOINTS = {
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/detection/stream.mjpg",  # Video streams
}


def get_api_key(
    header_key: Annotated[str | None, Security(API_KEY_HEADER)] = None,
    query_key: Annotated[str | None, Security(API_KEY_QUERY)] = None,
) -> str | None:
    """Extract API key from header or query parameter."""
    return header_key or query_key


def authenticate_api(
    api_key: Annotated[str | None, Depends(get_api_key)] = None,
) -> str | None:
    """Verify API key if one is configured. Returns key if valid, None if auth disabled."""
    if not _API_KEY:
        # No API key configured — auth disabled (development mode)
        return None
    if api_key == _API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key. Provide via X-API-Key header or ?api_key= query param.",
        headers={"WWW-Authenticate": "APIKey"},
    )


def require_api_key(
    api_key: Annotated[str | None, Depends(authenticate_api)] = None,
) -> str:
    """Require a valid API key. Use on protected endpoints."""
    if not _API_KEY:
        # Auth disabled — allow access
        return ""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid API key.",
        )
    return api_key


def generate_api_key() -> str:
    """Generate a new secure API key. Returns the key for display to the admin."""
    return secrets.token_urlsafe(32)
