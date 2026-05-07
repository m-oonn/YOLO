# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Event query API endpoints using shared EventsStore singleton."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.store import get_store

logger = logging.getLogger(__name__)

ALLOWED_SNAPSHOT_DIR = Path("outputs").resolve()

router = APIRouter()


@router.get("/")
def list_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    start_time: float | None = Query(None, description="Start timestamp"),
    end_time: float | None = Query(None, description="End timestamp"),
):
    """Query events with optional filters and pagination."""
    try:
        store = get_store()
        events, total = store.query_with_total(
            event_type=event_type,
            limit=limit,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
        )
        page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = (total + limit - 1) // limit if limit > 0 else 1
        return {
            "items": events,
            "total": total,
            "page": page,
            "page_size": limit,
            "total_pages": total_pages,
            "has_next": offset + limit < total,
            "has_prev": offset > 0,
        }
    except Exception as exc:
        logger.error("Failed to query events: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve events: {type(exc).__name__}",
        ) from exc


@router.get("/stats")
def event_stats():
    """Get event statistics."""
    store = get_store()
    return store.get_stats()


@router.get("/types")
def event_types():
    """List all available event types."""
    store = get_store()
    stats = store.get_stats()
    return {"event_types": list(stats.get("by_type", {}).keys())}


@router.delete("/")
def delete_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    before: float | None = Query(
        None, description="Delete events before this timestamp"
    ),
):
    """Delete events matching filters."""
    store = get_store()
    deleted = store.delete_events(event_type=event_type, before_timestamp=before)
    return {"status": "deleted", "count": deleted}


@router.delete("/all")
def clear_all_events():
    """Delete ALL events from the database."""
    store = get_store()
    deleted = store.clear_all()
    return {"status": "cleared", "count": deleted}


@router.get("/{event_id}/snapshot")
def event_snapshot(event_id: int):
    """Serve snapshot image for a given event."""
    store = get_store()
    with store._lock:
        cur = store.conn.cursor()
        cur.execute("SELECT snapshot_path FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
    if not row or not row["snapshot_path"]:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    path = row["snapshot_path"]
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(ALLOWED_SNAPSHOT_DIR)):
        logger.warning("Snapshot path traversal attempt: %s", path)
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Snapshot file not found on disk")
    return FileResponse(str(resolved), media_type="image/jpeg")
