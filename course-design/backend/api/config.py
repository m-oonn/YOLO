# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Config API endpoints for rules, zones, and system settings management."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# ── Pydantic Models ───────────────────────────────────────────


class RuleConfig(BaseModel):
    enabled: bool = True
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class RulesUpdateRequest(BaseModel):
    running: RuleConfig | None = None
    falling: RuleConfig | None = None
    fighting: RuleConfig | None = None
    crowd: RuleConfig | None = None
    forbidden_zone: RuleConfig | None = None
    vehicle_zone: RuleConfig | None = None


class ZoneConfig(BaseModel):
    name: str
    type: str = Field(..., pattern="^(forbidden|attention|counting)$")
    coordinates: list[list[float]]


class ZonesUpdateRequest(BaseModel):
    zones: list[ZoneConfig]


class SettingsUpdateRequest(BaseModel):
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    iou: float | None = Field(None, ge=0.0, le=1.0)
    frame_skip: int | None = Field(None, ge=1, le=30)
    save_snapshots: bool | None = None
    record_video: bool | None = None


# ── Helpers ───────────────────────────────────────────────────


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to read config: {e}"
        ) from e


def _write_config(data: dict) -> None:
    try:
        os.makedirs(CONFIG_PATH.parent, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error("Failed to write config: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to write config: {e}"
        ) from e


# ── Endpoints ─────────────────────────────────────────────────


@router.get("/")
def get_config():
    """Get full system configuration."""
    return {"status": "ok", "config": _read_config()}


@router.post("/rules")
def update_rules(req: RulesUpdateRequest):
    """Update detection rules configuration."""
    cfg = _read_config()
    if "rules" not in cfg:
        cfg["rules"] = {}

    rules_mapping = {
        "running": "run",
        "falling": "fall",
        "fighting": "fight",
        "crowd": "crowd",
        "forbidden_zone": "intrusion",
        "vehicle_zone": "intrusion",
    }

    for field, rule_key in rules_mapping.items():
        rule_data = getattr(req, field)
        if rule_data is None:
            continue
        if rule_key not in cfg["rules"]:
            cfg["rules"][rule_key] = {}
        cfg["rules"][rule_key]["enabled"] = rule_data.enabled
        if rule_data.threshold is not None:
            cfg["rules"][rule_key]["threshold"] = rule_data.threshold

    _write_config(cfg)
    return {"status": "saved", "rules": cfg.get("rules", {})}


@router.post("/settings")
def update_settings(req: SettingsUpdateRequest):
    """Update system settings."""
    cfg = _read_config()

    if "model" not in cfg:
        cfg["model"] = {}
    if req.confidence is not None:
        cfg["model"]["conf"] = req.confidence
    if req.iou is not None:
        cfg["model"]["iou"] = req.iou
    if req.frame_skip is not None:
        cfg["model"]["process_interval"] = req.frame_skip

    if "output" not in cfg:
        cfg["output"] = {}
    if req.save_snapshots is not None:
        cfg["output"]["save_snapshots"] = req.save_snapshots

    if "camera" not in cfg:
        cfg["camera"] = {}
    if req.record_video is not None:
        cfg["camera"]["record"] = req.record_video

    _write_config(cfg)
    return {"status": "saved", "settings": cfg}


@router.post("/zones")
def update_zones(req: ZonesUpdateRequest):
    """Update detection zones configuration."""
    cfg = _read_config()
    if "rules" not in cfg:
        cfg["rules"] = {}
    if "intrusion" not in cfg["rules"]:
        cfg["rules"]["intrusion"] = {}

    cfg["rules"]["intrusion"]["zones"] = [
        {"name": z.name, "type": z.type, "coordinates": z.coordinates}
        for z in req.zones
    ]

    _write_config(cfg)
    return {"status": "saved", "zones_count": len(req.zones)}
