# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Camera management API endpoints."""

from __future__ import annotations

import time
from fastapi import APIRouter

router = APIRouter()


def _list_camera_devices() -> list[dict]:
    """Detect available camera devices.

    Note: We do NOT open the camera here to avoid device lock on Windows.
    Instead, we return a list of potential camera IDs.
    The actual availability check happens when detection starts.
    """
    return [
        {
            "id": 0,
            "name": "Camera 0",
            "available": True,
            "resolution": "640x480",
            "fps": 30,
        }
    ]


@router.get("/")
def list_cameras():
    """List all available camera devices."""
    return {"devices": _list_camera_devices()}


@router.get("/{camera_id}")
def get_camera_info(camera_id: int):
    """Get information about a specific camera."""
    import cv2  # lazy import — cv2 is a heavy C extension
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {"id": camera_id, "available": False}
        # Give camera time to initialize
        time.sleep(0.3)
        ret, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # Give DirectShow time to release the device
        time.sleep(0.5)
        return {
            "id": camera_id,
            "name": f"Camera {camera_id}",
            "available": True,
            "resolution": f"{width}x{height}",
            "fps": round(fps) if fps > 0 else 30,
        }
    except Exception as e:
        return {"id": camera_id, "available": False, "error": str(e)}
