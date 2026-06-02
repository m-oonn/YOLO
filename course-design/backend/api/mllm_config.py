# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""MLLM configuration API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.detection_manager import detection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pydantic Models ───────────────────────────────────────────

class MLLMConfigResponse(BaseModel):
    enabled: bool = False
    interval: int = Field(30, ge=5, le=300)
    enhance_alarms: bool = True
    model_type: str = "qwen2-vl-2b"
    inference_backend: str = "mock"
    shadow_mode: bool = True
    key_frame_interval: int = 15
    max_new_tokens: int = 256
    scene_description_enabled: bool = True
    alarm_enhance_enabled: bool = True
    enhancement_cooldown_s: float = 10.0


class MLLMConfigUpdateRequest(BaseModel):
    enabled: bool | None = None
    interval: int | None = Field(None, ge=5, le=300)
    enhance_alarms: bool | None = None
    model_type: str | None = None
    inference_backend: str | None = None
    shadow_mode: bool | None = None
    key_frame_interval: int | None = None
    max_new_tokens: int | None = None
    scene_description_enabled: bool | None = None
    alarm_enhance_enabled: bool | None = None
    enhancement_cooldown_s: float | None = None


# ── Helpers ───────────────────────────────────────────────────

def _get_mllm_config_from_pipeline() -> dict:
    """Extract MLLM config from active pipeline or default config."""
    pipeline = detection_manager.get_pipeline()
    if pipeline and hasattr(pipeline, "_mllm_sidecar") and pipeline._mllm_sidecar:
        try:
            cfg = pipeline._mllm_sidecar._config
            return {
                "enabled": cfg.enabled,
                "interval": getattr(cfg, "key_frame_interval", 15),
                "enhance_alarms": getattr(cfg, "alarm_enhance_enabled", True),
                "model_type": getattr(cfg, "model_type", "qwen2-vl-2b"),
                "inference_backend": getattr(cfg, "inference_backend", "mock"),
                "shadow_mode": getattr(cfg, "shadow_mode", True),
                "key_frame_interval": getattr(cfg, "key_frame_interval", 15),
                "max_new_tokens": getattr(cfg, "max_new_tokens", 256),
                "scene_description_enabled": getattr(cfg, "scene_description_enabled", True),
                "alarm_enhance_enabled": getattr(cfg, "alarm_enhance_enabled", True),
                "enhancement_cooldown_s": getattr(cfg, "enhancement_cooldown_s", 10.0),
            }
        except Exception as e:
            logger.warning("Failed to get MLLM config from pipeline: %s", e)

    # Fallback to YAML config
    try:
        from core.config import load_config
        cfg = load_config()
        mllm = cfg.mllm
        return {
            "enabled": mllm.enabled,
            "interval": getattr(mllm, "key_frame_interval", 15),
            "enhance_alarms": getattr(mllm, "alarm_enhance_enabled", True),
            "model_type": getattr(mllm, "model_type", "qwen2-vl-2b"),
            "inference_backend": getattr(mllm, "inference_backend", "mock"),
            "shadow_mode": getattr(mllm, "shadow_mode", True),
            "key_frame_interval": getattr(mllm, "key_frame_interval", 15),
            "max_new_tokens": getattr(mllm, "max_new_tokens", 256),
            "scene_description_enabled": getattr(mllm, "scene_description_enabled", True),
            "alarm_enhance_enabled": getattr(mllm, "alarm_enhance_enabled", True),
            "enhancement_cooldown_s": getattr(mllm, "enhancement_cooldown_s", 10.0),
        }
    except Exception:
        return MLLMConfigResponse().model_dump()


# ── Endpoints ─────────────────────────────────────────────────

@router.get("/config")
def get_mllm_config():
    """Get current MLLM configuration."""
    return {"status": "ok", "config": _get_mllm_config_from_pipeline()}


@router.post("/config")
def update_mllm_config(req: MLLMConfigUpdateRequest):
    """Update MLLM configuration."""
    pipeline = detection_manager.get_pipeline()

    # Update runtime config if pipeline is active
    if pipeline and hasattr(pipeline, "_mllm_sidecar") and pipeline._mllm_sidecar:
        try:
            sidecar = pipeline._mllm_sidecar
            cfg = sidecar._config

            # Build updated config dict
            updates = {}
            if req.enabled is not None:
                updates["enabled"] = req.enabled
            if req.interval is not None:
                updates["key_frame_interval"] = req.interval
            if req.enhance_alarms is not None:
                updates["alarm_enhance_enabled"] = req.enhance_alarms
            if req.model_type is not None:
                updates["model_type"] = req.model_type
            if req.inference_backend is not None:
                updates["inference_backend"] = req.inference_backend
            if req.shadow_mode is not None:
                updates["shadow_mode"] = req.shadow_mode
            if req.key_frame_interval is not None:
                updates["key_frame_interval"] = req.key_frame_interval
            if req.max_new_tokens is not None:
                updates["max_new_tokens"] = req.max_new_tokens
            if req.scene_description_enabled is not None:
                updates["scene_description_enabled"] = req.scene_description_enabled
            if req.alarm_enhance_enabled is not None:
                updates["alarm_enhance_enabled"] = req.alarm_enhance_enabled
            if req.enhancement_cooldown_s is not None:
                updates["enhancement_cooldown_s"] = req.enhancement_cooldown_s

            # MLLMConfig is frozen, use replace
            from dataclasses import replace
            sidecar._config = replace(cfg, **updates)

            # Re-initialize if enabling
            if req.enabled and not cfg.enabled:
                sidecar.initialize()
            elif not req.enabled and cfg.enabled:
                sidecar.shutdown()

            return {"status": "saved", "config": _get_mllm_config_from_pipeline()}
        except Exception as e:
            logger.error("Failed to update MLLM config: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to update MLLM config: {e}")

    # No active pipeline - update YAML config only
    try:
        import yaml
        from backend.api.config import CONFIG_PATH, _read_config, _write_config

        cfg = _read_config()
        if "mllm" not in cfg:
            cfg["mllm"] = {}

        mllm = cfg["mllm"]
        if req.enabled is not None:
            mllm["enabled"] = req.enabled
        if req.interval is not None:
            mllm["key_frame_interval"] = req.interval
        if req.enhance_alarms is not None:
            mllm["alarm_enhance_enabled"] = req.enhance_alarms
        if req.model_type is not None:
            mllm["model_type"] = req.model_type
        if req.inference_backend is not None:
            mllm["inference_backend"] = req.inference_backend
        if req.shadow_mode is not None:
            mllm["shadow_mode"] = req.shadow_mode
        if req.key_frame_interval is not None:
            mllm["key_frame_interval"] = req.key_frame_interval
        if req.max_new_tokens is not None:
            mllm["max_new_tokens"] = req.max_new_tokens
        if req.scene_description_enabled is not None:
            mllm["scene_description_enabled"] = req.scene_description_enabled
        if req.alarm_enhance_enabled is not None:
            mllm["alarm_enhance_enabled"] = req.alarm_enhance_enabled
        if req.enhancement_cooldown_s is not None:
            mllm["enhancement_cooldown_s"] = req.enhancement_cooldown_s

        _write_config(cfg)
        return {"status": "saved", "config": mllm}
    except Exception as e:
        logger.error("Failed to update MLLM config in YAML: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to update MLLM config: {e}")
