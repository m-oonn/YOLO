# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Video archive API endpoints — list, stream, and manage event-triggered clips."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/archives", tags=["archives"])


class ClipInfo(BaseModel):
    clip_id: str
    event_type: str
    timestamp: float
    duration_s: float
    file_size_bytes: int
    event_count: int
    tags: list[str]


class ClipListResponse(BaseModel):
    clips: list[ClipInfo]
    total: int
    limit: int
    offset: int


def _get_recorder():
    from backend.detection_manager import detection_manager

    pipeline = detection_manager.get_pipeline()
    if pipeline is None:
        return None
    return getattr(pipeline, "_clip_recorder", None)


@router.get("", response_model=ClipListResponse)
def list_clips(
    event_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """List archived video clips with optional event type filter."""
    recorder = _get_recorder()
    if recorder is None:
        return ClipListResponse(clips=[], total=0, limit=limit, offset=offset)

    clips = recorder.get_clips(event_type=event_type, limit=limit, offset=offset)
    total = recorder.get_clip_count()

    return ClipListResponse(
        clips=[
            ClipInfo(
                clip_id=c.clip_id,
                event_type=c.event_type,
                timestamp=c.timestamp,
                duration_s=c.duration_s,
                file_size_bytes=c.file_size_bytes,
                event_count=c.event_count,
                tags=c.tags,
            )
            for c in clips
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{clip_id}")
def get_clip(clip_id: str):
    """Stream a video clip file with HTTP range support for seeking."""
    recorder = _get_recorder()
    if recorder is None:
        raise HTTPException(status_code=503, detail="Detection pipeline not running")

    clips = recorder.get_clips(limit=1000)
    matched = [c for c in clips if c.clip_id == clip_id]
    if not matched:
        raise HTTPException(status_code=404, detail="Clip not found")

    file_path = matched[0].file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip file missing")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=f"{clip_id}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.delete("/{clip_id}")
def delete_clip(clip_id: str):
    """Delete an archived video clip."""
    recorder = _get_recorder()
    if recorder is None:
        raise HTTPException(status_code=503, detail="Detection pipeline not running")

    ok = recorder.delete_clip(clip_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Clip not found")
    return {"status": "deleted", "clip_id": clip_id}
