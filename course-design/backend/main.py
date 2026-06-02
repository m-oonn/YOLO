# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application entry point for YOLO course project."""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file before any os.environ reads
except ImportError:
    # If dotenv not available, skip (we'll use defaults)
    pass

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import logging
import time as _time

_boot_ts = _time.perf_counter()

from .exceptions import YOLOException
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional slowapi import (skip if not available)
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from .limiter import app_limiter
    limiter = app_limiter
    has_slowapi = True
except ImportError:
    has_slowapi = False
    limiter = None

_t0 = _time.perf_counter()
from .api import cameras, detection, events  # noqa: E402
_t1 = _time.perf_counter()

from .api.alarms import router as alarms_router  # noqa: E402
from .api.archives import router as archives_router  # noqa: E402
from .api.config import router as config_router  # noqa: E402
from .api.mllm import router as mllm_router  # noqa: E402
from .api.mllm_config import router as mllm_config_router  # noqa: E402
from .logging_utils import clear_request_id, generate_request_id, setup_logging  # noqa: E402
from .store import close_store  # noqa: E402
_t2 = _time.perf_counter()

os.makedirs("outputs", exist_ok=True)

setup_logging(
    level=logging.INFO,
    log_file=os.path.join("outputs", "app.log"),
    json_format=os.environ.get("LOG_JSON", "0") == "1",
)
logger = logging.getLogger(__name__)

_boot_elapsed = (_time.perf_counter() - _boot_ts) * 1000
logger.info(
    "Backend module import timing: api-cameras/detection/events=%.0fms, "
    "remaining-routers=%.0fms, total=%.0fms",
    (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, _boot_elapsed,
)


ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting YOLO Course Design API")
    _prewarm_model()  # Always pre-warm YOLO model to reduce first-start latency
    yield
    logger.info("Shutting down YOLO Course Design API")
    try:
        from backend.alarm_singleton import close_alarm_engine
        close_alarm_engine()
    except Exception:
        pass
    close_store()


def _prewarm_model():
    """Pre-load YOLO model in background to warm CUDA JIT compiler cache.

    Runs on a daemon thread during app startup so the model and CUDA kernels
    are already hot when the user clicks 'start detection', reducing first-
    frame latency from ~5s to ~0.1s.
    """
    logger.info("Pre-warming YOLO model (background thread)...")
    def _warm():
        from core.config import load_config
        from ultralytics import YOLO
        import numpy as np
        try:
            cfg = load_config()
            model = YOLO(cfg.model_path)
            model.eval()
            model.predict(
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                imgsz=640, device="cuda" if cfg.device != "cpu" else "cpu",
                verbose=False, half=cfg.half,
            )
            logger.info("Model pre-warm complete (CUDA kernels compiled).")
        except Exception as e:
            logger.warning("Model pre-warm failed (non-fatal): %s", e)
    import threading
    t = threading.Thread(target=_warm, daemon=True)
    t.start()


app = FastAPI(
    title="YOLO Course Design API",
    description="Real-time object detection and behavior analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

if has_slowapi:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(YOLOException)
async def yolo_exception_handler(request: Request, exc: YOLOException):
    """Handler for YOLO application-specific exceptions."""
    from .logging_utils import get_request_id

    rid = get_request_id() or ""
    logger.warning(
        "[request_id=%s] YOLO exception: %s - %s at %s %s",
        rid,
        exc.code,
        exc.message,
        request.method,
        request.url.path,
    )
    status_map = {
        "PIPELINE_NOT_RUNNING": 400,
        "PIPELINE_ALREADY_RUNNING": 409,
        "MODEL_NOT_FOUND": 404,
        "INVALID_SOURCE": 400,
        "CONFIG_ERROR": 400,
    }
    status_code = status_map.get(exc.code, 500)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "request_id": rid,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors with structured error response."""
    from .logging_utils import get_request_id

    rid = get_request_id() or ""
    logger.exception(
        "[request_id=%s] Unhandled exception: %s - %s at %s %s",
        rid,
        type(exc).__name__,
        exc,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": rid,
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


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    """Inject a request_id into every request for tracing."""
    rid = request.headers.get("X-Request-ID") or generate_request_id()
    from .logging_utils import set_request_id
    set_request_id(rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        from .logging_utils import clear_request_id
        clear_request_id()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, status code, client IP, and duration."""
    start = _time.time()
    client_ip = request.client.host if request.client else "unknown"
    response = await call_next(request)
    duration_ms = (_time.time() - start) * 1000
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
        logger.warning(
            "%(method)s %(path)s -> %(status)d [%(client_ip)s] (%(duration_ms).2fms)"
            % log_kwargs
        )
    elif response.status_code >= 500:
        logger.error(
            "%(method)s %(path)s -> %(status)d [%(client_ip)s] (%(duration_ms).2fms)"
            % log_kwargs
        )
    else:
        logger.info(
            "%(method)s %(path)s -> %(status)d [%(client_ip)s] (%(duration_ms).2fms)"
            % log_kwargs
        )
    return response


# API routes
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])
app.include_router(alarms_router, prefix="/api/alarms", tags=["Alarms"])
app.include_router(archives_router, prefix="/api", tags=["Archives"])
app.include_router(mllm_router, prefix="/api/mllm", tags=["MLLM"])
app.include_router(mllm_config_router, prefix="/api/mllm", tags=["MLLM"])
app.include_router(config_router, prefix="/api/config", tags=["Config"])


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
