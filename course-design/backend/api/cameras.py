"""Camera management API endpoints."""

from __future__ import annotations

import cv2
from fastapi import APIRouter

router = APIRouter()


def _list_camera_devices() -> list[dict]:
    """Detect available camera devices. Uses CAP_DSHOW on Windows for fast enumeration."""
    devices = []
    for i in range(10):
        try:
            # CAP_DSHOW avoids lengthy backend enumeration on Windows
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                devices.append({
                    "id": i,
                    "name": f"Camera {i}",
                    "available": True,
                    "resolution": f"{width}x{height}" if width > 0 else "unknown",
                    "fps": round(fps) if fps > 0 else 30,
                })
                cap.release()
                break  # Stop after first available camera
        except Exception:
            continue
    return devices if devices else [
        {"id": 0, "name": "Camera 0", "available": True, "resolution": "640x480", "fps": 30}
    ]


@router.get("/")
def list_cameras():
    """List all available camera devices."""
    return {"devices": _list_camera_devices()}


@router.get("/{camera_id}")
def get_camera_info(camera_id: int):
    """Get information about a specific camera."""
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {"id": camera_id, "available": False}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return {
            "id": camera_id,
            "name": f"Camera {camera_id}",
            "available": True,
            "resolution": f"{width}x{height}",
            "fps": round(fps) if fps > 0 else 30,
        }
    except Exception as e:
        return {"id": camera_id, "available": False, "error": str(e)}
