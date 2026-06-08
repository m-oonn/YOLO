# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【后端API路由】alarms.py — 报警管理端点
# 上游依赖：backend/alarm_singleton.py（AlarmEngine单例）
# 下游调用：前端 AlarmsView.vue 查询和处理报警
# 路由前缀：/api/alarms
# 核心端点：
#   GET /           — 分页查询报警（支持状态/级别/类型筛选）
#   POST /{id}/acknowledge — 确认报警（标记为已知悉）
#   POST /{id}/resolve     — 处理报警（标记为已解决）
#   DELETE /all             — 清空全部报警
#   GET /stats              — 报警统计（按状态/级别汇总）
# ──────────────────────────────────────────────────────────

"""Alarm API endpoints for querying, acknowledging, and managing alarms."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_engine():
    from backend.alarm_singleton import get_alarm_engine

    return get_alarm_engine()


@router.get("/")
def list_alarms(
    status: str | None = Query(None, description="Filter by status: active, acknowledged, resolved, escalated"),
    level: int | None = Query(None, description="Filter by level: 1=info, 2=warning, 3=critical"),
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Query alarms with optional filters and pagination."""
    engine = _get_engine()
    alarms, total = engine.get_alarms(
        status=status,
        level=level,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )
    return {
        "items": alarms,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/stats")
def alarm_stats():
    """Get alarm statistics."""
    engine = _get_engine()
    return engine.get_stats()


@router.get("/levels")
def alarm_levels():
    """List available alarm levels."""
    from core.alarm_engine import EVENT_LEVEL_LABELS, AlarmLevel

    return {
        "levels": [
            {"value": int(lv), "name": lv.name, "label": EVENT_LEVEL_LABELS.get(int(lv), "")}
            for lv in AlarmLevel
        ]
    }


@router.post("/{alarm_id}/acknowledge")
def acknowledge_alarm(alarm_id: int):
    """Acknowledge an active alarm."""
    engine = _get_engine()
    success = engine.acknowledge(alarm_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alarm not found or not in active/escalated state")
    return {"status": "acknowledged", "alarm_id": alarm_id}


@router.post("/{alarm_id}/resolve")
def resolve_alarm(alarm_id: int):
    """Resolve an alarm."""
    engine = _get_engine()
    success = engine.resolve(alarm_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alarm not found or already resolved")
    return {"status": "resolved", "alarm_id": alarm_id}


@router.post("/resolve-type/{event_type}")
def resolve_by_type(event_type: str):
    """Resolve all active alarms of a given event type."""
    engine = _get_engine()
    count = engine.store.resolve_by_type(event_type)
    return {"status": "resolved", "event_type": event_type, "count": count}


@router.get("/config")
def get_alarm_config():
    """Get current alarm configuration."""
    engine = _get_engine()
    cfg = engine.config
    return {
        "enabled": cfg.enabled,
        "suppress_window_s": cfg.suppress_window_s,
        "aggregate_window_s": cfg.aggregate_window_s,
        "escalate_after_s": cfg.escalate_after_s,
        "max_alarms_per_minute": cfg.max_alarms_per_minute,
        "level_overrides": cfg.level_overrides,
        "notifiers": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("smtp_pass",)}
            for k, v in cfg.notifiers.items()
        },
    }


@router.delete("/all")
def clear_all_alarms():
    """Delete ALL alarms from the database."""
    engine = _get_engine()
    count = engine.store.clear_all()
    return {"status": "cleared", "count": count}


@router.put("/config")
def update_alarm_config(
    enabled: bool | None = None,
    suppress_window_s: float | None = None,
    aggregate_window_s: float | None = None,
    escalate_after_s: float | None = None,
    max_alarms_per_minute: int | None = None,
):
    """Update alarm configuration parameters."""
    engine = _get_engine()
    cfg = engine.config
    if enabled is not None:
        cfg.enabled = enabled
    if suppress_window_s is not None:
        cfg.suppress_window_s = suppress_window_s
    if aggregate_window_s is not None:
        cfg.aggregate_window_s = aggregate_window_s
    if escalate_after_s is not None:
        cfg.escalate_after_s = escalate_after_s
    if max_alarms_per_minute is not None:
        cfg.max_alarms_per_minute = max_alarms_per_minute
    return {"status": "updated", "config": get_alarm_config()}
