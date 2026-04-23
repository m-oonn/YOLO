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

from fastapi import FastAPI, Request, status
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
    """Global exception handler for unhandled errors."""
    logger.exception("Unhandled exception: %s - %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
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
    """Log request method, path, status code, and duration."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "%s %s -> %d (%.2fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration * 1000,
    )
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
