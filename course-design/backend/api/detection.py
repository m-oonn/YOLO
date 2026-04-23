# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Detection API endpoints with MJPEG stream for real-time video."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import suppress

import cv2
import filetype
import yaml
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.store import get_store
from core.config import load_config
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Global detection state
_pipeline: DetectionPipeline | None = None
_pipeline_thread: threading.Thread | None = None
_pipeline_lock = threading.Lock()
_latest_frame: bytes | None = None
_mjpeg_client_count: int = 0
_frame_ready = threading.Event()
_detection_active = False  # Track if detection thread is actively running


class DetectionStartRequest(BaseModel):
    source: str = "0"
    config: str = "configs/default.yaml"


class DetectionStatus(BaseModel):
    running: bool
    source: str | None = None
    fps: float = 0.0
    frame_count: int = 0
    elapsed_s: float = 0.0
    events_count: int = 0


class DetectionConfigRequest(BaseModel):
    config: str = "configs/default.yaml"


class SaveConfigRequest(BaseModel):
    model: dict
    rules: dict
    output: dict


def _resolve_config_path(config: str) -> str:
    """Resolve config file path relative to project root."""
    if os.path.exists(config):
        return config
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        config,
    )


def _run_detection(source: str, config_path: str):
    """Run detection pipeline in a separate thread."""
    global _pipeline, _latest_frame, _detection_active
    pipeline = None
    cap = None
    thread_name = threading.current_thread().name
    logger.info(f"Detection thread {thread_name} starting for source: {source}")

    try:
        cfg = load_config(config_path)
        store = get_store()
        pipeline = DetectionPipeline(cfg, store=store)
        with _pipeline_lock:
            _pipeline = pipeline

        pipeline.start()
        logger.info(f"Pipeline started for source: {source}")

        src = int(source) if source.isdigit() else source
        logger.info(f"Opening video source: {src}")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Cannot open source: {source}")
            # Update active flag since we failed to start
            with _pipeline_lock:
                _detection_active = False
            return

        logger.info(f"Video source opened successfully: {source}")
        target_frame_dt = 1.0 / max(1, cfg.camera_fps)

        frame_count = 0
        while pipeline.running:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of video stream")
                break

            timestamp = frame_start
            pipeline.process_frame(frame, timestamp)

            # Only encode JPEG when MJPEG clients are connected
            with _pipeline_lock:
                has_viewers = _mjpeg_client_count > 0
            if has_viewers:
                annotated = pipeline.annotate_frame(frame, pipeline._last_detections)
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _latest_frame = buffer.tobytes()
                _frame_ready.set()
            else:
                _latest_frame = None

            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count} frames from {source}")

            processing_time = time.time() - frame_start
            sleep_time = max(0, target_frame_dt - processing_time)
            time.sleep(sleep_time)

        logger.info(f"Detection loop ended for {source}, processed {frame_count} frames")

    except Exception as e:
        logger.error(f"Detection error in thread {thread_name}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up detection thread {thread_name}")
        try:
            if cap is not None:
                logger.info(f"Releasing video capture for {source}")
                cap.release()
                # Give the camera driver time to fully release resources
                time.sleep(0.5)
                logger.info(f"Video capture released for {source}")
        except Exception as e:
            logger.error(f"Error releasing video capture: {e}")

        try:
            with _pipeline_lock:
                if _pipeline:
                    logger.info("Closing pipeline")
                    _pipeline.close()
                    _pipeline = None
                _latest_frame = None
                _detection_active = False
                logger.info(f"Detection thread {thread_name} cleanup complete, active={_detection_active}")
        except Exception as e:
            logger.error(f"Error cleaning up pipeline: {e}")
            with _pipeline_lock:
                _detection_active = False


def _generate_mjpeg():
    """Generate MJPEG frames from the latest detection output."""
    global _mjpeg_client_count
    with _pipeline_lock:
        _mjpeg_client_count += 1
        logger.debug(f"MJPEG client connected (total: {_mjpeg_client_count})")
    boundary = "frame"
    try:
        while True:
            _frame_ready.wait(timeout=1.0)
            _frame_ready.clear()
            with _pipeline_lock:
                frame = _latest_frame
            if frame is not None:
                yield (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode()
                yield frame
                yield b"\r\n"
    except GeneratorExit:
        logger.debug("MJPEG stream client disconnected")
    finally:
        with _pipeline_lock:
            _mjpeg_client_count -= 1
            logger.debug(f"MJPEG client disconnected (total: {_mjpeg_client_count})")


@router.post("/start")
def start_detection(req: DetectionStartRequest):
    """Start detection on a camera or video source."""
    global _pipeline_thread, _detection_active

    with _pipeline_lock:
        if _detection_active:
            logger.warning("Detection already active, refusing new start")
            return {"status": "error", "message": "Detection already running"}

        # Ensure any previous thread is cleaned up
        if _pipeline_thread and _pipeline_thread.is_alive():
            logger.warning("Previous detection thread still alive, waiting for cleanup")
            # Try to stop the pipeline if it exists
            if _pipeline:
                _pipeline.stop()
            # Give it a moment to clean up
            _pipeline_thread.join(timeout=2.0)
            if _pipeline_thread.is_alive():
                logger.error("Previous detection thread failed to stop, cannot start new detection")
                return {"status": "error", "message": "Previous detection still stopping"}

    config_path = _resolve_config_path(req.config)

    with _pipeline_lock:
        _detection_active = True

    _pipeline_thread = threading.Thread(
        target=_run_detection,
        args=(req.source, config_path),
        daemon=True,
        name=f"DetectionThread-{req.source}",
    )
    _pipeline_thread.start()
    logger.info(f"Detection started on source: {req.source}")

    return {"status": "started", "source": req.source}


@router.post("/stop")
def stop_detection():
    """Stop the running detection."""
    global _detection_active

    with _pipeline_lock:
        if not _detection_active:
            logger.warning("Stop requested but no active detection")
            return {"status": "error", "message": "No active detection"}

        if _pipeline:
            logger.info("Stopping detection pipeline")
            _pipeline.stop()
            # Don't set _detection_active = False here, let _run_detection do it
        else:
            logger.warning("Pipeline is None but detection marked as active")
            _detection_active = False

    return {"status": "stopped", "message": "Detection stopping"}


@router.get("/status")
def get_detection_status():
    """Get current detection status."""
    with _pipeline_lock:
        if _pipeline:
            stats = _pipeline.get_stats()
            return DetectionStatus(
                running=stats["running"],
                fps=stats["fps"],
                frame_count=stats["frame_count"],
                elapsed_s=stats["elapsed_s"],
                events_count=stats["events"]["total_events"],
            )
    return DetectionStatus(running=False)


@router.post("/config")
def update_detection_config(req: DetectionConfigRequest):
    """Update detection config at runtime."""
    config_path = _resolve_config_path(req.config)
    with _pipeline_lock:
        if not _pipeline:
            return {"status": "error", "message": "No active detection"}
        try:
            new_cfg = load_config(config_path)
            _pipeline.cfg = new_cfg
            logger.info("Detection config updated")
            return {"status": "updated"}
        except Exception as e:
            logger.error(f"Config update failed: {e}")
            return {"status": "error", "message": str(e)}


@router.post("/save-config")
def save_detection_config(req: SaveConfigRequest):
    """Save config to YAML file and update runtime config."""
    config_path = _resolve_config_path("configs/default.yaml")

    yaml_data = {
        "model": {
            "path": req.model.get("path", "models/yolov11x.pt"),
            "imgsz": int(req.model.get("imgsz", 640)),
            "conf": float(req.model.get("conf", 0.35)),
            "iou": float(req.model.get("iou", 0.5)),
        },
        "camera": {"fps": 30},
        "rules": {
            "running": {
                "enabled": req.rules.get("running", {}).get("enabled", True),
                "speed_px_s": int(req.rules.get("running", {}).get("speed_px_s", 50)),
                "min_duration_s": 0.3,
                "debounce_s": float(req.rules.get("running", {}).get("debounce_s", 5.0)),
            },
            "fall": {
                "enabled": req.rules.get("fall", {}).get("enabled", True),
                "upright_aspect_min": float(req.rules.get("fall", {}).get("upright_aspect_min", 1.2)),
                "fallen_aspect_max": 1.0,
                "transition_window_s": 1.0,
                "debounce_s": float(req.rules.get("fall", {}).get("debounce_s", 5.0)),
            },
            "crowd": {
                "enabled": req.rules.get("crowd", {}).get("enabled", True),
                "min_people": int(req.rules.get("crowd", {}).get("min_people", 3)),
                "proximity_px": float(req.rules.get("crowd", {}).get("proximity_px", 200.0)),
                "debounce_s": float(req.rules.get("crowd", {}).get("debounce_s", 10.0)),
            },
            "intrusion": {
                "enabled": req.rules.get("intrusion", {}).get("enabled", True),
                "debounce_s": float(req.rules.get("intrusion", {}).get("debounce_s", 5.0)),
                "zones": req.rules.get("intrusion", {}).get("zones", [
                    {"name": "lab", "polygon": [[60, 60], [580, 60], [580, 340], [60, 340]]}
                ]),
            },
            "fight": {
                "enabled": req.rules.get("fight", {}).get("enabled", True),
                "distance_threshold": int(req.rules.get("fight", {}).get("distance_threshold", 150)),
                "movement_threshold": int(req.rules.get("fight", {}).get("movement_threshold", 30)),
                "min_duration_s": float(req.rules.get("fight", {}).get("min_duration_s", 0.5)),
                "debounce_s": float(req.rules.get("fight", {}).get("debounce_s", 5.0)),
            },
        },
        "output": {
            "directory": "outputs",
            "save_snapshots": req.output.get("save_snapshots", True),
            "view": True,
        },
    }

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to write config file: {e}")
        return {"status": "error", "message": f"Failed to write config file: {e}"}

    # Update runtime config if pipeline is active
    try:
        new_cfg = load_config(config_path)
        with _pipeline_lock:
            if _pipeline:
                _pipeline.cfg = new_cfg
                logger.info("Runtime config updated from saved file")
    except Exception as e:
        logger.warning(f"Config file saved, but runtime update failed: {e}")

    return {"status": "saved", "path": config_path}


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/x-flv",
    "video/x-ms-wmv",
}
MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for detection with size and type validation."""
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    if not file.filename:
        return {"status": "error", "message": "Invalid filename"}

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Invalid file extension. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        }

    content = await file.read()
    content_size = len(content)

    if content_size > MAX_UPLOAD_SIZE_BYTES:
        return {
            "status": "error",
            "message": f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB",
        }

    if content_size == 0:
        return {"status": "error", "message": "Empty file"}

    kind = filetype.guess(content)
    if kind is None:
        return {
            "status": "error",
            "message": "Unable to determine file type. The file may be corrupted or unsupported.",
        }

    if kind.mime not in ALLOWED_VIDEO_MIME_TYPES:
        return {
            "status": "error",
            "message": f"File content is not a video (detected: {kind.mime}). Only video files are allowed.",
        }

    dest = os.path.join(upload_dir, f"upload_{int(time.time())}{ext}")
    with open(dest, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded video saved to {dest} (size={content_size} bytes, type={kind.mime})")
    return {"status": "uploaded", "path": dest, "filename": file.filename}


@router.get("/stream.mjpg")
def mjpeg_stream():
    """MJPEG stream endpoint for real-time annotated video."""
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates (periodic push)."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            with suppress(Exception):
                await websocket.receive_text()

            with _pipeline_lock:
                if _pipeline and _pipeline.running:
                    stats = _pipeline.get_stats()
                    await websocket.send_json({"type": "status", "data": stats})

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
