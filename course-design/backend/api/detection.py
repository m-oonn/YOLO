# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Detection API endpoints with MJPEG stream for real-time video.

Security Improvements:
- Path traversal prevention
- Enhanced file upload validation
- Rate limiting on sensitive endpoints
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import filetype
import yaml
from fastapi import APIRouter, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.detection_manager import detection_manager
from backend.limiter import app_limiter as limiter
from backend.security import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    validate_model_path,
    validate_upload_extension,
    validate_upload_mime,
)
from core.config import load_config

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    timestamp: float
    uptime_s: float
    components: dict[str, str]


def _get_component_health() -> dict[str, str]:
    """Check health of individual components.

    Returns:
        Dictionary mapping component name to health status
    """
    components = {}

    try:
        import psutil
        components["cpu"] = "healthy"
        components["memory"] = "healthy"
    except Exception:
        components["system"] = "unknown"

    try:
        from core.gpu_manager import GPUManager
        gpu = GPUManager()
        gpu_status = gpu.get_status_dict()
        if gpu_status.get("gpu_available"):
            components["gpu"] = "healthy"
        else:
            components["gpu"] = "unavailable"
    except Exception:
        components["gpu"] = "unavailable"

    components["detection_manager"] = "healthy" if not detection_manager._detection_active else "running"
    components["pipeline"] = "healthy" if detection_manager.get_pipeline() else "inactive"

    return components


@router.get("/health", response_model=HealthStatus, tags=["health"])
def health_check():
    """Comprehensive health check endpoint.

    Returns:
        Health status of the service and its components
    """
    from datetime import datetime

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().timestamp(),
        uptime_s=time.time() - _start_time,
        components=_get_component_health(),
    )

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
_start_time = time.time()


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
    performance: dict | None = None


class DetectionConfigRequest(BaseModel):
    config: str = "configs/default.yaml"


class ModelSwitchRequest(BaseModel):
    model_path: str
    reload_pipeline: bool = True


class SaveConfigRequest(BaseModel):
    model: dict
    rules: dict
    output: dict
    mllm: dict = {}
    camera: dict = {}
    alarm: dict = {}


def _resolve_config_path(config: str) -> str:
    """Resolve config file path relative to project root.
    
    Prevents path traversal by rejecting paths containing '..' or
    absolute paths that resolve outside the project root.
    """
    if ".." in config:
        raise HTTPException(status_code=400, detail="Invalid config path: path traversal detected")
    resolved = Path(PROJECT_ROOT) / config
    if not str(resolved.resolve()).startswith(str(Path(PROJECT_ROOT).resolve())):
        raise HTTPException(status_code=400, detail="Invalid config path: path outside project root")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {config}")
    return str(resolved)


# ── Performance endpoint cache ────────────────────────────────

_perf_cache: dict = {}
_perf_cache_time: float = 0


def _get_system_stats() -> dict:
    """Cached system CPU and memory stats (1s TTL)."""
    global _perf_cache, _perf_cache_time
    now = time.time()
    if now - _perf_cache_time < 1.0:
        return _perf_cache
    import psutil

    _perf_cache = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
    }
    _perf_cache_time = now
    return _perf_cache


# ── GPU recommendations helper ────────────────────────────────

def _get_gpu_recommendations(gpu_mgr) -> list[str]:
    recs = []
    if not gpu_mgr.is_gpu_available:
        recs.append("未检测到GPU，当前使用CPU推理。安装CUDA版PyTorch可提升3-5倍帧率。")
        recs.append("推荐: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        info = gpu_mgr.gpu_info
        pressure = gpu_mgr.get_memory_pressure()

        if pressure in ("high", "critical"):
            recs.append(f"⚠️ GPU显存压力{pressure}！建议：1)关闭Ollama等占用GPU的程序 2)使用yolo12n.pt替代大模型 3)降低inference_scale至0.5")

        if info.total_memory_mb < 4000:
            recs.append(f"GPU显存较小({info.total_memory_mb:.0f}MB)，建议使用yolo12n.pt模型或降低推理缩放。")
        elif info.total_memory_mb < 6000:
            recs.append(f"GPU显存中等({info.total_memory_mb:.0f}MB)，建议使用yolo12s.pt模型，避免同时运行其他GPU程序。")

        if not info.supports_tensor_cores:
            recs.append("当前GPU不支持Tensor Cores，FP16加速效果有限。Volta及以上架构(Turing/Ampere/Ada)效果最佳。")
        if gpu_mgr.is_cuda:
            recs.append("GPU已就绪。确保配置中device='auto'或'cuda:0'以启用GPU加速。")
        elif gpu_mgr.is_mps:
            recs.append("Apple MPS已就绪。MPS不支持FP16半精度推理。")
    return recs


# ── Detection Control ─────────────────────────────────────────

@router.post("/start")
@limiter.limit("10/minute")
def start_detection(request: Request, req: DetectionStartRequest):
    """Start detection on a camera or video source."""
    config_path = _resolve_config_path(req.config)
    return detection_manager.start(req.source, config_path)


@router.post("/stop")
def stop_detection():
    """Stop the running detection."""
    return detection_manager.stop()


@router.get("/status")
def get_detection_status():
    """Get current detection status."""
    return detection_manager.get_status()


@router.post("/config")
def update_detection_config(req: DetectionConfigRequest):
    """Update detection config at runtime."""
    config_path = _resolve_config_path(req.config)
    return detection_manager.update_config(config_path)


# ── Model Management ──────────────────────────────────────────

@router.get("/models")
def list_available_models():
    """List all available YOLO models in the models directory.

    Security: Only lists files within the models directory,
    preventing directory traversal attacks.
    """
    available_models = []
    if MODELS_DIR.exists():
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith((".pt", ".onnx", ".engine", ".tflite")):
                model_path = os.path.join("models", filename)
                full_path = MODELS_DIR / filename
                try:
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    available_models.append({
                        "name": filename,
                        "path": model_path,
                        "size_mb": round(size_mb, 2),
                    })
                except OSError:
                    continue

    pipeline = detection_manager.get_pipeline()
    current_model = None
    if pipeline:
        current_model = getattr(pipeline, '_runtime', None).model_path if getattr(pipeline, '_runtime', None) else pipeline.cfg.model_path
    else:
        cfg_data = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        current_model = cfg_data.model_path

    return {
        "current": current_model,
        "available": available_models,
    }


@router.get("/gpu")
def get_gpu_status():
    """Get detailed GPU status and capabilities."""
    from core.gpu_manager import GPUManager
    gpu_mgr = GPUManager()
    status = gpu_mgr.get_status_dict()
    status["recommendations"] = _get_gpu_recommendations(gpu_mgr)

    try:
        import torch
        if torch.cuda.is_available():
            status["pytorch_memory"] = {
                "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 1),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
            }
    except Exception:
        pass

    return status


@router.get("/monitoring")
def get_monitoring_stats():
    """Get comprehensive monitoring data for dashboard.

    Returns combined stats from:
    - Detection pipeline
    - GPU manager
    - System resources
    - MLLM sidecar (if enabled)

    This endpoint is designed for monitoring dashboards and
    reduces the need for multiple API calls.
    """
    from core.gpu_manager import GPUManager

    gpu_mgr = GPUManager()
    pipeline = detection_manager.get_pipeline()

    response = {
        "timestamp": time.time(),
        "detection": {
            "running": detection_manager._detection_active,
            "source": detection_manager._current_source,
            "pipeline_stats": detection_manager.get_pipeline_stats() if pipeline else None,
            "mjpeg_clients": detection_manager.get_mjpeg_client_count(),
        },
        "gpu": gpu_mgr.get_status_dict(),
        "system": _get_system_stats(),
    }

    if pipeline and hasattr(pipeline, '_mllm_sidecar'):
        try:
            response["mllm"] = pipeline._mllm_sidecar.get_stats()
        except Exception:
            response["mllm"] = None

    return response


@router.get("/performance")
def get_performance_stats():
    """Get detailed performance metrics for the running detection pipeline."""
    from core.gpu_manager import GPUManager

    gpu_mgr = GPUManager()

    pipeline = detection_manager.get_pipeline()
    if not pipeline:
        return {
            "running": False,
            "system": _get_system_stats(),
            "gpu": gpu_mgr.get_status_dict(),
        }

    stats = pipeline.get_stats()
    perf = stats.get("performance", {})

    return {
        "running": True,
        "fps": stats["fps"],
        "avg_fps": stats.get("avg_fps", stats["fps"]),
        "min_fps": stats.get("min_fps", stats["fps"]),
        "frame_count": stats["frame_count"],
        "elapsed_s": stats["elapsed_s"],
        "pipeline": perf,
        "gpu": stats.get("gpu", gpu_mgr.get_status_dict()),
        "system": _get_system_stats(),
    }


@router.post("/models/switch")
def switch_model(req: ModelSwitchRequest):
    """Switch to a different model at runtime or update config for next start.

    Security: Validates model path to prevent path traversal attacks.
    """
    model_rel_path = req.model_path
    if model_rel_path.startswith("models/") or model_rel_path.startswith("models\\"):
        model_rel_path = model_rel_path[len("models/"):]

    is_valid, error_msg = validate_model_path(model_rel_path, MODELS_DIR)
    if not is_valid:
        logger.warning(f"Invalid model path attempt: {req.model_path} - {error_msg}")
        return {"status": "error", "message": f"Invalid model path: {error_msg}"}

    model_full_path = str(PROJECT_ROOT / req.model_path)

    pipeline = detection_manager.get_pipeline()
    if pipeline:
        return detection_manager.switch_model(req.model_path, model_full_path)

    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        if "model" not in yaml_data:
            yaml_data["model"] = {}
        yaml_data["model"]["path"] = req.model_path
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        logger.info("Model path updated in config for next start: %s", req.model_path)
        return {
            "status": "success",
            "message": f"Model path updated to {req.model_path}. Start detection to use the new model.",
            "new_model": req.model_path,
            "runtime_switch": False,
        }
    except Exception as e:
        logger.error("Failed to update model path in config: %s", e)
        return {"status": "error", "message": f"Failed to update config: {e}"}


class QualityControlRequest(BaseModel):
    """Request model for quality control."""

    quality: int = Field(..., ge=30, le=95, description="JPEG quality (30-95)")
    auto: bool = Field(default=False, description="Enable automatic quality adjustment")


@router.post("/quality")
def control_quality(req: QualityControlRequest):
    """Control MJPEG encoding quality dynamically.

    Allows real-time adjustment of video encoding quality based on:
    - Network bandwidth
    - Client count
    - CPU/GPU performance

    Args:
        req: Quality control request with target quality and auto flag

    Returns:
        Current quality settings and status
    """
    pipeline = detection_manager.get_pipeline()

    if not pipeline:
        return {
            "status": "error",
            "message": "No active detection pipeline",
        }

    if req.auto:
        optimal_quality = detection_manager.calculate_dynamic_quality()
        pipeline.set_jpeg_quality(optimal_quality)
        logger.info(f"Auto quality adjustment enabled, set to {optimal_quality}")
        return {
            "status": "success",
            "quality": optimal_quality,
            "auto": True,
            "message": f"Quality auto-adjusted to {optimal_quality}",
            "mjpeg_clients": detection_manager.get_mjpeg_client_count(),
        }

    pipeline.set_jpeg_quality(req.quality)
    logger.info(f"Manual quality adjustment to {req.quality}")

    return {
        "status": "success",
        "quality": req.quality,
        "auto": False,
        "message": f"Quality set to {req.quality}",
    }


@router.get("/quality")
def get_quality():
    """Get current MJPEG encoding quality settings.

    Returns:
        Current quality value and whether auto-adjust is enabled
    """
    pipeline = detection_manager.get_pipeline()

    if not pipeline:
        return {
            "status": "no_pipeline",
            "quality": 85,
            "mjpeg_clients": 0,
        }

    return {
        "status": "ok",
        "quality": pipeline.get_encoding_quality(),
        "mjpeg_clients": detection_manager.get_mjpeg_client_count(),
    }


# ── Configuration Persistence ─────────────────────────────────


@router.get("/config")
def get_detection_config():
    """Read saved config from YAML file and return it."""
    config_path = _resolve_config_path("configs/default.yaml")
    if not os.path.exists(config_path):
        return {"status": "not_found", "config": {}}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        return {"status": "ok", "config": yaml_data}
    except Exception as e:
        logger.error("Failed to read config file: %s", e)
        return {"status": "error", "message": str(e)}


@router.post("/save-config")
def save_detection_config(req: SaveConfigRequest):
    """Save config to YAML file and update runtime config."""
    config_path = _resolve_config_path("configs/default.yaml")

    try:
        import yaml as yaml_lib
        with open(config_path, "r", encoding="utf-8") as f:
            current_yaml = yaml_lib.safe_load(f) or {}
        current_mllm = current_yaml.get("mllm", {})
        if not isinstance(current_mllm, dict):
            current_mllm = {}
    except Exception:
        current_mllm = {}

    yaml_data = {
        "model": {
            "path": req.model.get("path", "models/yolo12s.pt"),
            "device": req.model.get("device", "cuda"),
            "imgsz": int(req.model.get("imgsz", 640)),
            "conf": float(req.model.get("conf", 0.25)),
            "iou": float(req.model.get("iou", 0.5)),
            "inference_scale": float(req.model.get("inference_scale", 1.0)),
            "jpeg_quality": int(req.model.get("jpeg_quality", 85)),
            "classes": req.model.get("classes"),
            "process_interval": int(req.model.get("process_interval", 1)),
        },
        "camera": {
            "url": req.camera.get("url", ""),
            "fps": int(req.camera.get("fps", 25)),
            "buffer_size": int(req.camera.get("buffer_size", 30)),
        },
        "alarm": {
            "enabled": req.alarm.get("enabled", True),
            "cooldown_s": float(req.alarm.get("cooldown_s", 30.0)),
            "sound_enabled": bool(req.alarm.get("sound_enabled", True)),
        },
        "rules": {
            "run": req.rules.get("run", {}).get("enabled", True),
            "fall": req.rules.get("fall", {}).get("enabled", True),
            "crowd": req.rules.get("crowd", {}).get("enabled", True),
            "intrusion": {
                "enabled": req.rules.get("intrusion", {}).get("enabled", True),
                "debounce_s": float(req.rules.get("intrusion", {}).get("debounce_s", 5.0)),
                "zones": req.rules.get("intrusion", {}).get("zones", []),
            },
            "fight": {
                "enabled": req.rules.get("fight", {}).get("enabled", True),
                "distance_threshold": int(req.rules.get("fight", {}).get("distance_threshold", 150)),
                "movement_threshold": int(req.rules.get("fight", {}).get("movement_threshold", 30)),
                "min_duration_s": float(req.rules.get("fight", {}).get("min_duration_s", 0.5)),
                "debounce_s": float(req.rules.get("fight", {}).get("debounce_s", 5.0)),
            },
        },
        "pose": {
            "enabled": True,
            "kp_threshold": 0.5,
            "smoothing_alpha": 0.3,
            "max_skeletons": 50,
            "process_interval": int(req.model.get("process_interval", 2)),
        },
        "output": {
            "directory": "outputs",
            "save_snapshots": req.output.get("save_snapshots", True),
            "view": True,
        },
        "mllm": {
            "enabled": req.mllm.get("enabled", current_mllm.get("enabled", False)),
            "model_type": req.mllm.get("model_type", current_mllm.get("model_type", "qwen2-vl-2b")),
            "model_path": req.mllm.get("model_path", current_mllm.get("model_path", "models/mllm/qwen2-vl-2b")),
            "inference_backend": req.mllm.get("inference_backend", current_mllm.get("inference_backend", "mock")),
            "shadow_mode": req.mllm.get("shadow_mode", current_mllm.get("shadow_mode", True)),
            "key_frame_interval": int(req.mllm.get("key_frame_interval", current_mllm.get("key_frame_interval", 15))),
            "max_new_tokens": int(req.mllm.get("max_new_tokens", current_mllm.get("max_new_tokens", 256))),
            "scene_description_enabled": req.mllm.get("scene_description_enabled", current_mllm.get("scene_description_enabled", True)),
            "alarm_enhance_enabled": req.mllm.get("alarm_enhance_enabled", current_mllm.get("alarm_enhance_enabled", True)),
            "enhancement_cooldown_s": float(req.mllm.get("enhancement_cooldown_s", current_mllm.get("enhancement_cooldown_s", 10.0))),
        },
    }

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        logger.info("Config saved to %s", config_path)
    except Exception as e:
        logger.error("Failed to write config file: %s", e)
        return {"status": "error", "message": f"Failed to write config file: {e}"}

    # Update runtime config if pipeline is active
    try:
        new_cfg = load_config(config_path)
        pipeline = detection_manager.get_pipeline()
        if pipeline:
            pipeline.update_config(new_cfg)
            logger.info("Runtime config updated from saved file")
    except Exception as e:
        logger.warning("Config file saved, but runtime update failed: %s", e)

    return {"status": "saved", "path": config_path}


# ── Video Upload ──────────────────────────────────────────────

@router.post("/upload")
@limiter.limit("5/minute")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """Upload a video file for detection with enhanced validation.

    Security improvements:
    - Extension validation
    - MIME type validation
    - File content validation using filetype
    - Size limit enforcement
    - Safe filename generation
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not file.filename:
        return {"status": "error", "message": "Invalid filename"}

    ext = os.path.splitext(file.filename)[1].lower()
    is_valid, error_msg = validate_upload_extension(file.filename)
    if not is_valid:
        return {"status": "error", "message": error_msg}

    content = await file.read()
    content_size = len(content)

    if content_size > MAX_UPLOAD_SIZE_BYTES:
        return {
            "status": "error",
            "message": f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB",
        }

    if content_size == 0:
        return {"status": "error", "message": "Empty file"}

    is_valid, error_msg = validate_upload_mime(content)
    if not is_valid:
        return {"status": "error", "message": error_msg}

    # Get MIME type for logging
    kind = filetype.guess(content)
    mime_type = kind.mime if kind else "unknown"

    import secrets
    safe_filename = f"upload_{int(time.time())}_{secrets.token_hex(4)}{ext}"
    dest = UPLOAD_DIR / safe_filename
    with open(dest, "wb") as f:
        f.write(content)

    logger.info(
        "Uploaded video saved to %s (size=%d bytes, type=%s)",
        dest, content_size, mime_type,
    )
    return {"status": "uploaded", "path": f"uploads/{safe_filename}", "filename": file.filename}


# ── Streaming ─────────────────────────────────────────────────

@router.get("/stream.mjpg")
def mjpeg_stream():
    """MJPEG stream endpoint for real-time annotated video."""
    return StreamingResponse(
        detection_manager.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates.

    This endpoint provides push-based status updates to reduce client polling.
    Sends updates at regular intervals and on significant state changes.

    Message types pushed:
    - status: Detection pipeline stats (FPS, frame count, etc.)
    - performance: Detailed performance metrics
    - events: Recent detection events
    - gpu: GPU status updates
    - mllm: MLLM status updates
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    from core.gpu_manager import GPUManager
    gpu_mgr = GPUManager()

    async def send_status_update():
        """Send comprehensive status update."""
        try:
            pipeline_stats = detection_manager.get_pipeline_stats()
            if pipeline_stats:
                await websocket.send_json({
                    "type": "status",
                    "data": pipeline_stats
                })

                if pipeline_stats.get("performance"):
                    await websocket.send_json({
                        "type": "performance",
                        "data": pipeline_stats["performance"]
                    })
            elif not detection_manager._detection_active:
                # 管道未运行且检测已停止时，通知前端停止状态
                status = detection_manager.get_status()
                msg = {
                    "type": "status",
                    "data": {
                        "running": False,
                        "fps": 0,
                        "frame_count": 0,
                        "elapsed_s": 0,
                        "events_count": 0,
                    }
                }
                if status.get("last_error"):
                    msg["data"]["last_error"] = status["last_error"]
                    msg["data"]["error_count"] = status.get("error_count", 1)
                await websocket.send_json(msg)
            else:
                # 检测已标记为active但管线尚未就绪（模型加载中）
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "running": True,
                        "state": "loading",
                        "fps": 0,
                        "frame_count": 0,
                        "elapsed_s": 0,
                        "events_count": 0,
                    }
                })
        except Exception as e:
            logger.debug("Error sending status update: %s", e)

    async def send_gpu_update():
        """Send GPU status update."""
        try:
            gpu_status = gpu_mgr.get_status_dict()
            await websocket.send_json({
                "type": "gpu",
                "data": gpu_status
            })
        except Exception:
            pass

    try:
        update_count = 0
        while True:
            # 使用超时方式监听客户端消息，确保不会永久阻塞
            # 从而允许定时发送状态更新到前端
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                pass  # 无客户端消息，继续发送定期更新
            except WebSocketDisconnect:
                break
            except Exception as ws_err:
                logger.error("WebSocket error (client may have disconnected): %s", ws_err)
                break

            update_count += 1

            if update_count % 2 == 0:
                await send_gpu_update()

            await send_status_update()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
