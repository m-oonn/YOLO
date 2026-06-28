# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Timeout-safe camera open utilities for Windows and cross-platform use.

On Windows, cv2.VideoCapture() with the DirectShow backend can hang for
30-120+ seconds when the camera device is unavailable or locked. This
module provides timeout-wrapped alternatives that prevent the calling
thread from blocking indefinitely.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout per backend attempt (seconds)
DEFAULT_CAMERA_TIMEOUT = 3.0
# Short timeout for launcher pre-checks
QUICK_CHECK_TIMEOUT = 3.0
# Retry delay between backend attempts (seconds)
BACKEND_RETRY_DELAY = 0.3


def open_camera_with_timeout(
    source: int | str,
    backend: int | None = None,
    timeout_sec: float = DEFAULT_CAMERA_TIMEOUT,
) -> tuple[Any | None, str | None]:
    """Open a cv2.VideoCapture source with a hard timeout.

    Runs cv2.VideoCapture(source, backend) in a separate non-daemon thread.
    If the thread does not complete within *timeout_sec*, the caller receives
    an error and the hung thread is abandoned (it will be cleaned up on
    process exit).

    Args:
        source: Camera index (int) or video file path (str).
        backend: OpenCV backend constant (cv2.CAP_MSMF, cv2.CAP_DSHOW, etc.).
                 Pass None for OpenCV auto-selection.
        timeout_sec: Maximum seconds to wait for VideoCapture to return.

    Returns:
        Tuple of (VideoCapture | None, error_message | None).
        On success, returns (cap, None) where cap.isOpened() is True.
        On failure, returns (None, error_string).
    """
    import cv2

    result_container: list[Any] = [None]
    error_container: list[str | None] = [None]
    ready = threading.Event()

    def _open() -> None:
        """Open camera in background thread."""
        cap_local = None
        try:
            if backend is not None:
                cap_local = cv2.VideoCapture(source, backend)
            else:
                cap_local = cv2.VideoCapture(source)
            result_container[0] = cap_local
        except Exception as exc:
            error_container[0] = f"cv2.VideoCapture raised: {exc}"
            result_container[0] = None
        finally:
            ready.set()

    thread = threading.Thread(target=_open, name=f"cam-open-{source}", daemon=True)
    thread.start()

    finished = ready.wait(timeout=timeout_sec)
    if not finished:
        logger.warning(
            "Camera open timed out after %.1fs for source=%s backend=%s",
            timeout_sec,
            source,
            backend,
        )
        return (
            None,
            f"Camera open timed out after {timeout_sec:.0f}s. "
            f"The device may be unavailable, disconnected, or locked by another application.",
        )

    if error_container[0] is not None:
        logger.warning(
            "Camera open error for source=%s backend=%s: %s",
            source,
            backend,
            error_container[0],
        )
        return None, error_container[0]

    cap = result_container[0]
    if cap is None or not cap.isOpened():
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        return (
            None,
            f"cv2.VideoCapture returned but isOpened()=False for source={source} "
            f"backend={backend}. No camera at this index or device access denied.",
        )

    return cap, None


def try_open_camera_windows(
    source: int,
    timeout_per_backend: float = DEFAULT_CAMERA_TIMEOUT,
) -> tuple[Any | None, str | None]:
    """Open a camera on Windows, trying MSMF first, then DShow, then auto.

    Microsoft Media Foundation (MSMF) has better timeout behavior on
    Windows 10/11. DirectShow (DShow) is the traditional fallback.
    The final auto attempt lets OpenCV choose.

    Args:
        source: Integer camera index.
        timeout_per_backend: Timeout in seconds for each backend attempt.

    Returns:
        Tuple of (VideoCapture | None, error_message | None).
    """
    import cv2

    backends: list[tuple[int | None, str]] = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DShow"),
        (None, "auto"),
    ]

    last_error = None
    for idx, (backend, name) in enumerate(backends):
        logger.info(
            "Trying camera %d with backend %s (timeout=%.1fs)...",
            source,
            name,
            timeout_per_backend,
        )
        cap, err = open_camera_with_timeout(
            source, backend=backend, timeout_sec=timeout_per_backend
        )
        if cap is not None and cap.isOpened():
            logger.info("Camera %d opened successfully with %s backend", source, name)
            return cap, None
        if err:
            last_error = err
            logger.warning("Camera %d backend %s failed: %s", source, name, err)
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        # Small delay between backend attempts (not after last)
        if idx < len(backends) - 1:
            time.sleep(BACKEND_RETRY_DELAY)

    return None, last_error or f"All backends failed for camera index {source}"


def diagnose_camera_failure(source: str, system: str) -> str:
    """Return platform-specific troubleshooting steps for camera failures.

    Args:
        source: The source identifier that failed (e.g. "0", "file.mp4").
        system: platform.system() return value.

    Returns:
        Multi-line human-readable diagnostic message.
    """
    if system == "Windows":
        return (
            f"摄像头索引 {source} 无法打开。请依次检查:\n"
            f"  1. 摄像头是否已物理连接到电脑 (USB 或内置)\n"
            f"  2. Windows 设置 > 隐私和安全性 > 摄像头 > 摄像头访问 → 确保已开启\n"
            f"  3. 同上路径 > 允许应用访问你的摄像头 → 确保已开启\n"
            f"  4. 同上路径 > 允许桌面应用访问你的摄像头 → 确保已开启\n"
            f"  5. 设备管理器 (devmgmt.msc) > 照相机 > 检查驱动状态\n"
            f"  6. 关闭其他可能占用摄像头的程序 (Teams, Zoom, 浏览器等)\n"
            f"  7. 如使用杀毒软件，检查是否拦截了 Python 进程的摄像头访问"
        )
    elif system == "Darwin":
        return (
            f"Camera index {source} could not be opened. Please check:\n"
            f"  1. System Preferences > Security & Privacy > Camera\n"
            f"  2. Ensure Terminal / your IDE has camera permission\n"
            f"  3. Close other apps using the camera (FaceTime, Zoom, etc.)"
        )
    else:
        return (
            f"Video source {source} could not be opened. Please check:\n"
            f"  1. Device permissions (udev rules on Linux: user in 'video' group)\n"
            f"  2. Camera connection and drivers\n"
            f"  3. No other process is using the device"
        )


def quick_camera_check(
    source: int = 0,
    timeout_sec: float = QUICK_CHECK_TIMEOUT,
) -> tuple[bool, str]:
    """Lightweight camera availability check for launcher pre-flight.

    Opens the camera with a short timeout, verifies a frame can be read,
    then immediately releases. Safe to call before the detection pipeline
    is initialized.

    Args:
        source: Camera index to check (default 0).
        timeout_sec: Max time to wait for camera open.

    Returns:
        Tuple of (available: bool, message: str).
    """

    system = platform.system()
    cap = None

    try:
        if system == "Windows":
            cap, err = try_open_camera_windows(source, timeout_per_backend=timeout_sec)
        else:
            cap, err = open_camera_with_timeout(source, timeout_sec=timeout_sec)

        if cap is None or not cap.isOpened():
            return False, err or "Camera not available"

        # Try reading a frame to confirm the device is fully functional
        ret, _ = cap.read()
        if not ret:
            return (
                False,
                "Camera opened but cannot read frames. "
                "Device may be in use by another application.",
            )

        return True, "Camera is available and functional"

    except Exception as exc:
        return False, f"Camera check failed: {exc}"
    finally:
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
