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
    shadow_mode: bool = True


@router.get("/status")
def get_mllm_status():
    pipeline = detection_manager.get_pipeline()
    if pipeline and hasattr(pipeline, "_mllm_sidecar") and pipeline._mllm_sidecar:
        return {
            "available": True,
            "stats": pipeline._mllm_sidecar.get_stats(),
        }
    try:
        from core.config import load_config

        cfg = load_config()
        return {
            "available": True,
            "stats": {
                "enabled": cfg.mllm.enabled,
                "running": False,
                "engine": {
                    "backend": cfg.mllm.inference_backend,
                    "loaded": False,
                },
                "total_inferences": 0,
                "scenes_described": 0,
                "alarms_enhanced": 0,
                "last_scene": None,
            },
        }
    except Exception:
        pass
    return {
        "available": False,
        "stats": {"enabled": False, "running": False},
    }


@router.post("/enable")
def toggle_mllm(req: MLLMEnableRequest):
    pipeline = detection_manager.get_pipeline()
    if not pipeline:
        return {"status": "error", "message": "No active detection pipeline"}
    sidecar = getattr(pipeline, "_mllm_sidecar", None)
    if sidecar is None:
        return {"status": "error", "message": "MLLM sidecar not initialized"}

    if req.enabled:
        try:
            # Force enable in sidecar config (MLLMConfig is frozen, so replace)
            if (
                not sidecar._config.enabled
                or sidecar._config.shadow_mode != req.shadow_mode
            ):
                sidecar._config = replace(
                    sidecar._config, enabled=True, shadow_mode=req.shadow_mode
                )
            sidecar.initialize()
            return {"status": "success", "message": "MLLM sidecar enabled"}
        except Exception as e:
            logger.error("Failed to enable MLLM: %s", e)
            return {"status": "error", "message": str(e)}
    else:
        try:
            sidecar.shutdown()
            return {"status": "success", "message": "MLLM sidecar disabled"}
        except Exception as e:
            logger.error("Failed to disable MLLM: %s", e)
            return {"status": "error", "message": str(e)}
