# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""FastAPI application entry point for YOLO course project."""

from __future__ import annotations

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import cameras, detection, events
from .store import close_store

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("outputs", "app.log")),
    ],
)
logger = logging.getLogger(__name__)


ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting YOLO Course Design API")
    yield
    logger.info("Shutting down YOLO Course Design API")
    close_store()


app = FastAPI(
    title="YOLO Course Design API",
    description="Real-time object detection and behavior analysis API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors with structured error response."""
    request_id = id(request)
    logger.exception(
        "[request_id=%s] Unhandled exception: %s - %s at %s %s",
        request_id,
        type(exc).__name__,
        exc,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "request_id": str(request_id),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Structured HTTP exception handler with request tracing."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "HTTPException",
            "status_code": exc.status_code,
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key Authentication Middleware ──────────────────────────
# Reads API_KEY from env. If set, all /api/* endpoints require
# a matching X-API-Key header (or ?api_key= query param).
# Public endpoints (health, docs, streams) are always accessible.
_API_KEY = os.environ.get("API_KEY")
_PUBLIC_PREFIXES = (
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/detection/stream.mjpg",
    "/api/detection/stream",
)


@app.middleware("http")
async def enforce_api_key(request: Request, call_next):
    """Reject requests with invalid/missing API key when API_KEY is configured."""
    if _API_KEY:
        path = request.url.path
        # Check if this is a public endpoint
        is_public = any(path.startswith(p) for p in _PUBLIC_PREFIXES)
        if not is_public:
            provided = request.headers.get("X-API-Key") or request.query_params.get(
                "api_key"
            )
            if not provided or provided != _API_KEY:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": "Invalid or missing API key. "
                        "Provide via X-API-Key header or ?api_key= query param.",
                    },
                    headers={"WWW-Authenticate": "APIKey"},
                )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, status code, client IP, and duration."""
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    log_kwargs = {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "client_ip": client_ip,
        "duration_ms": round(duration_ms, 2),
    }
    # user_agent may not exist in test environments
    try:
        if request.headers.get("user-agent"):
            log_kwargs["user_agent"] = request.headers.get("user-agent")
    except Exception:
        pass
    if 400 <= response.status_code < 500:
        logger.warning("Client error: %(method)s %(path)s -> %(status)d [%(client_ip)s] (%.2fms)", log_kwargs)
    elif response.status_code >= 500:
        logger.error("Server error: %(method)s %(path)s -> %(status)d [%(client_ip)s] (%.2fms)", log_kwargs)
    else:
        logger.info("%(method)s %(path)s -> %(status)d [%(client_ip)s] (%.2fms)", log_kwargs)
    return response


# API routes
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])


@app.get("/")
def root():
    return {
        "name": "YOLO Course Design API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
