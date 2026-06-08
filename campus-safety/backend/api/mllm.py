# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""MLLM API endpoints for scene understanding status and control."""

from __future__ import annotations

import logging
from dataclasses import replace

from fastapi import APIRouter
from pydantic import BaseModel

from backend.detection_manager import detection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class MLLMEnableRequest(BaseModel):
    enabled: bool = True
    shadow_mode: bool = False


def _gpu_memory_stats() -> dict:
    try:
        from core.gpu_manager import GPUManager
        return GPUManager().get_status_dict()
    except Exception:
        return {}


@router.get("/status")
def get_mllm_status():
    pipeline = detection_manager.get_pipeline()
    gpu = _gpu_memory_stats()
    if pipeline and hasattr(pipeline, "_mllm_sidecar") and pipeline._mllm_sidecar:
        stats = pipeline._mllm_sidecar.get_stats()
        return {
            "available": True,
            "stats": stats,
            "gpu": gpu,
        }
    try:
        from core.config import load_config
        cfg = load_config()
        return {
            "available": True,
            "stats": {
                "enabled": cfg.mllm.enabled,
                "running": False,
                "model_loaded": False,
                "engine": {
                    "backend": cfg.mllm.inference_backend,
                    "loaded": False,
                },
                "total_inferences": 0,
                "scenes_described": 0,
                "alarms_enhanced": 0,
                "last_scene": None,
            },
            "gpu": gpu,
        }
    except Exception:
        pass
    return {
        "available": False,
        "stats": {"enabled": False, "running": False, "model_loaded": False},
        "gpu": gpu,
    }


@router.post("/enable")
def toggle_mllm(req: MLLMEnableRequest):
    pipeline = detection_manager.get_pipeline()
    if not pipeline:
        return {"status": "error", "message": "No active detection pipeline"}

    if req.enabled:
        try:
            ok = pipeline.enable_mllm(shadow_mode=req.shadow_mode)
            if not ok:
                return {"status": "error", "message": "MLLM failed to start"}
            return {
                "status": "success",
                "message": "MLLM enabled (Qwen2-VL-2B-Instruct loaded)",
                "gpu": _gpu_memory_stats(),
            }
        except Exception as e:
            logger.error("Failed to enable MLLM: %s", e)
            return {"status": "error", "message": str(e)}

    try:
        pipeline.disable_mllm()
        return {
            "status": "success",
            "message": "MLLM disabled, GPU memory released",
            "gpu": _gpu_memory_stats(),
        }
    except Exception as e:
        logger.error("Failed to disable MLLM: %s", e)
        return {"status": "error", "message": str(e)}
