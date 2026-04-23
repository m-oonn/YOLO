# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Event query API endpoints using shared EventsStore singleton."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.store import get_store

router = APIRouter()


@router.get("/")
def list_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    start_time: float | None = Query(None, description="Start timestamp"),
    end_time: float | None = Query(None, description="End timestamp"),
):
    """Query events with optional filters."""
    store = get_store()
    events, total = store.query_with_total(
        event_type=event_type,
        limit=limit,
        offset=offset,
        start_time=start_time,
        end_time=end_time,
    )
    return {"events": events, "total": total, "limit": limit, "offset": offset}


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
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Snapshot file not found on disk")
    return FileResponse(path, media_type="image/jpeg")
