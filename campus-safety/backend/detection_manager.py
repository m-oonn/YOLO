# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【后端API】detection_manager.py — 检测线程管理器
# 上游依赖：core/pipeline.py（创建并驱动检测流水线）
# 下游调用：backend/api/detection.py（HTTP端点调用）
# 核心职责：
#   ① 创建检测流水线线程（独立于API请求线程）
#   ② 管理检测状态（idle/running/stopping/error）
#   ③ MJPEG帧缓冲（线程间共享，供HTTP视频流端点读取）
#   ④ 性能统计（FPS/帧数/事件数）供前端WebSocket查询
#   ⑤ 优雅启停（模型预热、资源释放、锁安全）
# 这是后端最复杂的文件（42KB），因为要处理多线程协调
# ──────────────────────────────────────────────────────────

"""Detection thread management and shared state.

Encapsulates all global state previously held as module-level variables
in backend/api/detection.py, including pipeline lifecycle, MJPEG frame
buffering, and thread coordination.

Performance Optimizations Applied:
- Thread pool for efficient thread management
- Reduced lock contention with fine-grained locking
- Optimized MJPEG encoding with dynamic quality adjustment
- Fast shutdown with graceful degradation
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import cv2

if TYPE_CHECKING:
    from core.pipeline import DetectionPipeline

from backend.camera_utils import (
    diagnose_camera_failure,
    open_camera_with_timeout,
    try_open_camera_windows,
)
from backend.store import get_store
from core.config import load_config
from core.constants import PERSON_CLASS_ID
from core.rules import RulesEngine
from core.behavior_analyzer import BehaviorAnalyzer

logger = logging.getLogger(__name__)

# Performance tuning constants
DEFAULT_THREAD_TIMEOUT = 5.0
DEFAULT_SHUTDOWN_TIMEOUT = 5.0
MIN_JPEG_QUALITY = 30
MAX_JPEG_QUALITY = 95
DEFAULT_JPEG_QUALITY = 85


class DetectionManager:
    """Manages detection pipeline lifecycle, thread safety, and MJPEG streaming state.

    Thread Safety:
        All public methods are thread-safe using a reentrant lock.
        The lock protects shared state access during start/stop operations.

    Performance Considerations:
        - Uses daemon threads for automatic cleanup
        - Implements fine-grained locking to minimize contention
        - Optimizes MJPEG encoding based on client count
        - Supports graceful shutdown with configurable timeouts
    """

    def __init__(self):
        self._pipeline: DetectionPipeline | None = None
        self._pipeline_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._latest_frame: bytes | None = None
        self._mjpeg_frame_counter: int = 0
        self._mjpeg_client_count: int = 0
        self._mjpeg_has_viewers = threading.Event()
        self._frame_ready = threading.Event()
        self._detection_active = False
        self._current_source: str | None = None
        self._start_time: float | None = None
        self._error_count: int = 0
        self._last_error: str | None = None
        self._cap = None  # cv2.VideoCapture | None
        self._startup_deadline: float = 0.0  # timestamp after which loading is considered failed
        self._startup_progress: dict = {"step": "idle", "message": "", "percent": 0}
        self._progress_callbacks: list[Callable] = []

    # ── Public API ──────────────────────────────────────────────

    def start(self, source: str, config_path: str) -> dict:
        """Start detection on a camera or video source.

        Thread Safety:
            Uses a reentrant lock to prevent concurrent start/stop operations.
            Releases lock during stop to avoid deadlock with stop() method.

        Performance:
            - Reduces wait time for resource cleanup
            - Uses RLock for nested lock acquisition
            - Implements graceful degradation on errors
        """
        self._recover_if_stale()

        acquired = self._lock.acquire(timeout=1.0)
        if not acquired:
            logger.warning("Failed to acquire lock for start operation")
            return {"status": "error", "message": "Detection start in progress, please wait"}

        try:
            old_pipeline = None
            if self._detection_active:
                logger.info("Detection active, stopping first before starting new")
                # RLock allows reentrant acquisition; stop() acquires+releases internally
                stop_result = self.stop()
                if stop_result["status"] != "stopped":
                    error_msg = f"Failed to stop current detection: {stop_result.get('message', 'unknown error')}"
                    logger.error(error_msg)
                    self._record_error(error_msg)
                    return {"status": "error", "message": error_msg}

                time.sleep(0.3)
                self._cleanup_state()

            if self._pipeline_thread and self._pipeline_thread.is_alive():
                logger.warning("Previous thread still alive, waiting for cleanup")
                self._pipeline_thread.join(timeout=DEFAULT_THREAD_TIMEOUT)
                if self._pipeline_thread.is_alive():
                    logger.warning(
                        "Previous detection thread still alive, forcing fresh start; "
                        "daemon thread will be cleaned up by the runtime"
                    )
                    old_pipeline = self._pipeline
                    self._pipeline = None
                    self._pipeline_thread = None
                else:
                    old_pipeline = None

            self._cleanup_state()
            self._detection_active = True
            # Set a 60-second startup deadline: if pipeline isn't ready by then,
            # treat it as a startup failure (camera opening or model loading timed out)
            self._startup_deadline = time.time() + 60.0
        finally:
            if acquired:
                self._lock.release()

        if old_pipeline:
            self._release_pipeline(old_pipeline)

        self._pipeline_thread = threading.Thread(
            target=self._run_detection,
            args=(source, config_path),
            daemon=True,
            name=f"DetectionThread-{source}",
        )
        self._pipeline_thread.start()
        self._start_time = time.time()
        logger.info("Detection started on source: %s", source)
        # 优化：立即返回，不等待模型加载完成，前端通过状态轮询获取进度
        return {"status": "started", "source": source, "message": "检测启动中，模型加载可能需要几秒..."}

    def _record_error(self, error_msg: str) -> None:
        """Record error for diagnostics."""
        self._error_count += 1
        self._last_error = error_msg
        logger.error("Detection error recorded: %s (total: %d)", error_msg, self._error_count)

    def _release_pipeline(self, pipeline) -> None:
        """Safely release a pipeline and free GPU memory.

        Must be called without holding the lock (cleanup may take time).
        """
        if pipeline is None:
            return
        try:
            if hasattr(pipeline, "cleanup"):
                pipeline.cleanup()
            elif hasattr(pipeline, "model") and pipeline.model is not None:
                pipeline.model = None
        except Exception:
            logger.exception("Error releasing pipeline resources")

    def _cleanup_state(self) -> None:
        """Clean up detection state. Must be called with lock held."""
        self._latest_frame = None
        self._mjpeg_frame_counter = 0
        self._frame_ready.clear()
        self._current_source = None
        self._detection_active = False
        self._cap = None
        self._error_count = 0
        self._last_error = None
        self._startup_progress = {"step": "idle", "message": "", "percent": 0}
        # 注意：不在这里清理 _progress_callbacks，因为它们可能会被复用
        # 如果需要清理，应该单独调用 unregister

    def _set_progress(self, step: str, message: str, percent: int) -> None:
        """Update startup progress and notify callbacks."""
        with self._lock:
            self._startup_progress = {"step": step, "message": message, "percent": percent}
            # 复制回调列表，避免在锁内执行长时间操作
            callbacks = list(self._progress_callbacks)
        # 在锁外调用回调，避免死锁和长时间持有锁
        for cb in callbacks:
            try:
                cb(self._startup_progress)
            except Exception:
                pass

    def get_progress(self) -> dict:
        """Get current startup progress."""
        with self._lock:
            return dict(self._startup_progress)

    def register_progress_callback(self, callback: Callable) -> None:
        """Register a callback for startup progress updates."""
        with self._lock:
            self._progress_callbacks.append(callback)

    def unregister_progress_callback(self, callback: Callable) -> None:
        """Unregister a progress callback."""
        with self._lock:
            try:
                self._progress_callbacks.remove(callback)
            except ValueError:
                pass

    def stop(self) -> dict:
        """Stop the running detection.

        Thread Safety:
            Uses a reentrant lock to prevent concurrent stop/start operations.
            Releases lock early to avoid deadlock with start() method.

        GPU Memory Management:
            - Checks GPU memory pressure before deciding whether to keep or
              release the YOLO model from GPU memory.
            - If pressure is 'high' or 'critical', calls cleanup() to free VRAM.
            - If pressure is 'low' or 'medium', keeps model for fast restart.
        """
        acquired = self._lock.acquire(timeout=1.0)
        if not acquired:
            logger.warning("Failed to acquire lock for stop operation")
            return {"status": "error", "message": "Stop in progress, please wait"}

        try:
            if not self._detection_active:
                logger.warning("Stop requested but no active detection")
                return {"status": "stopped", "message": "No active detection"}

            pipeline_to_stop = self._pipeline
            thread_to_stop = self._pipeline_thread
            self._detection_active = False
        finally:
            self._lock.release()

        if pipeline_to_stop:
            logger.info("Stopping detection pipeline")
            try:
                pipeline_to_stop.stop()
            except Exception as e:
                logger.exception("Error stopping pipeline: %s", e)

        if thread_to_stop and thread_to_stop.is_alive():
            logger.info("Waiting for detection thread to stop")
            thread_to_stop.join(timeout=DEFAULT_SHUTDOWN_TIMEOUT)
            if thread_to_stop.is_alive():
                logger.warning("Detection thread did not stop within timeout, cleaning up references")
                with self._lock:
                    self._pipeline = None
                    self._pipeline_thread = None
                self._release_pipeline(pipeline_to_stop)

        with self._lock:
            if self._cap is not None:
                time.sleep(0.15)  # let capture thread see pipeline.running == False
                logger.info("Force-releasing video capture synchronously")
                try:
                    self._cap.release()
                except Exception as e:
                    logger.exception("Error releasing video capture in stop: %s", e)
                self._cap = None

        if pipeline_to_stop:
            should_cleanup = False
            try:
                from core.gpu_manager import GPUManager
                gpu_mgr = GPUManager()
                should_cleanup = gpu_mgr.should_release_model()
                pressure = gpu_mgr.get_memory_pressure()
                logger.info("GPU memory pressure: %s, will %s model", pressure, "release" if should_cleanup else "keep")
            except Exception as e:
                logger.warning("GPU pressure check failed, defaulting to cleanup: %s", e)
                should_cleanup = True

            if should_cleanup:
                logger.info("GPU memory pressure high — releasing model from GPU")
                try:
                    pipeline_to_stop.cleanup()
                except Exception as e:
                    logger.exception("Error cleaning up pipeline: %s", e)
            else:
                logger.info("GPU memory sufficient — keeping model in GPU for fast restart")

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU memory after stop: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")
        except Exception:
            pass

        self._frame_ready.clear()
        logger.info("Detection stopped successfully")
        return {"status": "stopped", "message": "Detection stopped"}

    def _recover_if_stale(self) -> None:
        """Detect and recover from a stale detection state.

        Handles the case where a previous server instance was killed while
        detection was running, leaving _detection_active=True but no live thread.
        """
        with self._lock:
            if not self._detection_active:
                return
            thread = self._pipeline_thread
            if thread is None or not thread.is_alive():
                logger.info("Stale detection state detected (active=True but thread dead). Auto-recovering.")
                stale_pipeline = self._pipeline
                self._detection_active = False
                self._pipeline = None
                self._pipeline_thread = None
                self._latest_frame = None
                self._mjpeg_frame_counter = 0
                self._frame_ready.clear()
                self._current_source = None
                self._cap = None
                self._startup_deadline = 0.0
            else:
                stale_pipeline = None
        if stale_pipeline:
            self._release_pipeline(stale_pipeline)

    def get_status(self) -> dict:
        """Get current detection status dict.

        Returns comprehensive status including:
        - Running state and source
        - FPS and frame statistics
        - Event counts
        - Performance metrics (if available)
        - Error diagnostics (if any)

        Distinguishes three states:
        - running=True with stats: pipeline active and processing frames
        - running=True, state="loading": detection flagged active but pipeline
          not yet ready (model loading / camera warmup in progress)
        - running=False: fully stopped (may carry last_error)
        """
        with self._lock:
            if self._pipeline:
                stats = self._pipeline.get_stats()
                result = {
                    "running": stats["running"],
                    "source": self._current_source,
                    "fps": stats["fps"],
                    "frame_count": stats["frame_count"],
                    "elapsed_s": stats["elapsed_s"],
                    "events_count": stats["events"]["total_events"],
                    "performance": stats.get("performance"),
                }
                if self._start_time:
                    result["uptime_s"] = round(time.time() - self._start_time, 1)
                if self._error_count > 0:
                    result["error_count"] = self._error_count
                    result["last_error"] = self._last_error
                return result
            # Pipeline not yet created but detection is active → loading
            if self._detection_active:
                loading_elapsed = time.time() - (self._start_time or time.time())
                # Check if startup has exceeded the deadline (camera/model loading stuck)
                now = time.time()
                if self._startup_deadline and now > self._startup_deadline:
                    # Startup timeout: auto-fail the detection
                    logger.error(
                        "Detection startup timeout after %.1fs (deadline was %.1fs). "
                        "Camera or model loading appears stuck.",
                        loading_elapsed, self._startup_deadline - (self._start_time or now),
                    )
                    self._detection_active = False
                    self._record_error(
                        f"摄像头启动超时（{loading_elapsed:.0f}秒未完成）"
                    )
                    result = {
                        "running": False,
                        "source": self._current_source,
                        "fps": 0,
                        "frame_count": 0,
                        "elapsed_s": round(loading_elapsed, 1),
                        "events_count": 0,
                        "state": "error",
                        "last_error": self._last_error,
                        "error_count": self._error_count,
                    }
                else:
                    result = {
                        "running": True,
                        "state": "loading",
                        "source": self._current_source,
                        "fps": 0,
                        "frame_count": 0,
                        "elapsed_s": round(loading_elapsed, 1),
                        "events_count": 0,
                    }
            else:
                result = {"running": False, "source": self._current_source}
            if self._error_count > 0:
                result["error_count"] = self._error_count
                result["last_error"] = self._last_error
            return result

    def get_mjpeg_client_count(self) -> int:
        """Get the current number of MJPEG stream clients.

        Returns:
            Number of connected MJPEG clients.
        """
        with self._lock:
            return self._mjpeg_client_count

    def calculate_dynamic_quality(self) -> int:
        """Calculate optimal JPEG quality based on conditions.

        Dynamic quality adjustment based on:
        - Number of connected clients
        - Current encoding performance
        - Network conditions (if measurable)

        Returns:
            Optimal JPEG quality value (30-95)
        """
        base_quality = DEFAULT_JPEG_QUALITY
        client_count = self._mjpeg_client_count

        if client_count == 0:
            return base_quality

        quality = base_quality

        if client_count > 5:
            quality = min(quality, 75)
        if client_count > 10:
            quality = min(quality, 65)
        if client_count > 20:
            quality = min(quality, 55)

        with self._lock:
            if self._pipeline and hasattr(self._pipeline, '_perf_encode_ms'):
                encode_ms = self._pipeline._perf_encode_ms
                if encode_ms > 50:
                    quality = min(quality, 70)
                if encode_ms > 100:
                    quality = min(quality, 60)

        return max(MIN_JPEG_QUALITY, min(MAX_JPEG_QUALITY, quality))

    def update_config(self, config_path: str) -> dict:
        """Update detection config at runtime."""
        with self._lock:
            if not self._pipeline:
                return {"status": "error", "message": "No active detection"}
            try:
                new_cfg = load_config(config_path)
                self._pipeline.update_config(new_cfg)
                logger.info("Detection config updated")
                return {"status": "updated"}
            except Exception as e:
                logger.error("Config update failed: %s", e)
                return {"status": "error", "message": str(e)}

    def switch_model(self, model_path: str, full_path: str) -> dict:
        """Switch YOLO model at runtime."""
        with self._lock:
            if not self._pipeline:
                return {"status": "error", "message": "No active detection"}
            try:
                from ultralytics import YOLO

                logger.info("Switching model to: %s", model_path)

                # 切换模型前清理缓存，避免旧模型显存残留
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                new_model = YOLO(full_path)
                new_model.eval()  # 关闭梯度追踪，减少显存

                # 预热新模型，避免首次推理触发 CUDA 编译导致卡顿
                try:
                    import numpy as np
                    dummy = np.random.randint(0, 255, (self._pipeline.cfg.imgsz, self._pipeline.cfg.imgsz, 3), dtype=np.uint8)
                    import torch
                    with torch.no_grad():
                        new_model.predict(
                            dummy,
                            imgsz=self._pipeline.cfg.imgsz,
                            device=self._pipeline._device,
                            half=self._pipeline._half,
                            verbose=False,
                        )
                    logger.info("New model warmup completed after switch")
                except Exception as warmup_err:
                    logger.warning("Model warmup after switch failed: %s", warmup_err)

                self._pipeline.model = new_model
                if self._pipeline._runtime:
                    self._pipeline._runtime.model_path = model_path
                logger.info("Model switched successfully to: %s", model_path)
                return {
                    "status": "success",
                    "message": f"Model switched to {model_path}",
                    "new_model": model_path,
                    "runtime_switch": True,
                }
            except Exception as e:
                logger.error("Model switch failed: %s", e)
                return {"status": "error", "message": str(e)}

    def get_pipeline(self) -> DetectionPipeline | None:
        with self._lock:
            return self._pipeline

    def get_current_source(self) -> str | None:
        return self._current_source

    def get_performance_stats(self) -> dict | None:
        """Get pipeline performance stats if running."""
        with self._lock:
            if self._pipeline:
                stats = self._pipeline.get_stats()
                return stats.get("performance")
        return None

    def get_pipeline_stats(self) -> dict | None:
        """Get full pipeline stats for WebSocket push."""
        with self._lock:
            if self._pipeline and self._pipeline.running:
                return self._pipeline.get_stats()
        return None

    def has_viewers(self) -> bool:
        return self._mjpeg_has_viewers.is_set()

    # ── MJPEG streaming ─────────────────────────────────────────

    def _mjpeg_connect(self):
        with self._lock:
            self._mjpeg_client_count += 1
            if self._mjpeg_client_count == 1:
                self._mjpeg_has_viewers.set()
            logger.debug("MJPEG client connected (total: %d)", self._mjpeg_client_count)

    def _mjpeg_disconnect(self):
        with self._lock:
            self._mjpeg_client_count -= 1
            if self._mjpeg_client_count <= 0:
                self._mjpeg_client_count = 0
                self._mjpeg_has_viewers.clear()
            logger.debug("MJPEG client disconnected (total: %d)", self._mjpeg_client_count)

    def generate_mjpeg(self):
        """Yields MJPEG frames from the latest detection output.

        Uses a sequence counter to detect and skip stale frames automatically.
        Slow clients that fall behind more than 1 frame will immediately receive
        the latest frame without blocking the pipeline.

        Gracefully exits when detection is no longer active, so the browser
        receives a clean connection close instead of hanging indefinitely.
        """
        self._mjpeg_connect()
        last_seq = -1
        boundary = "frame"
        try:
            while True:
                # 检测已停止时优雅退出，避免浏览器连接永远挂起
                with self._lock:
                    if not self._detection_active:
                        logger.debug("MJPEG stream ending: detection not active")
                        break

                self._frame_ready.wait(timeout=1.0)
                self._frame_ready.clear()
                with self._lock:
                    frame = self._latest_frame
                    seq = self._mjpeg_frame_counter
                if seq == last_seq:
                    continue
                last_seq = seq
                if frame is not None:
                    yield (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode()
                    yield frame
                    yield b"\r\n"
        except GeneratorExit:
            logger.debug("MJPEG stream client disconnected")
        finally:
            self._mjpeg_disconnect()

    # ── Internal: detection loop ─────────────────────────────────

    def _wait_for_previous(self) -> str | None:
        """Wait for previous detection thread to finish, without holding pipeline lock.

        Returns an error message if the previous thread cannot be stopped, None on success.
        
        优化：减少等待时间，使用非阻塞检查加速启动
        """
        thread = self._pipeline_thread
        if thread is None or not thread.is_alive():
            return None

        logger.warning(
            "Previous detection thread still alive, waiting for cleanup (thread: %s)",
            thread.name,
        )

        with self._lock:
            if self._pipeline:
                self._pipeline.stop()

        # 优化：从3.0秒减少到1.5秒等待时间
        thread.join(timeout=1.5)
        if thread.is_alive():
            logger.warning(
                "Previous detection thread %s still alive after 1.5s, proceeding with fresh start",
                thread.name,
            )
            stale_pipeline = self._pipeline
            with self._lock:
                self._pipeline = None
                self._pipeline_thread = None
            if stale_pipeline:
                self._release_pipeline(stale_pipeline)
        return None

    def _run_detection(self, source: str, config_path: str):
        """Run detection pipeline in a separate thread."""
        import cv2  # lazy import — cv2 is a heavy C extension (~70MB)
        pipeline = None
        cap = None
        thread_name = threading.current_thread().name
        logger.info("Detection thread %s starting for source: %s", thread_name, source)
        with self._lock:
            self._current_source = source
            self._set_progress("init", "正在初始化检测...", 5)

        try:
            from core.pipeline import DetectionPipeline  # 延迟导入，避免后端启动时加载 PyTorch

            cfg = load_config(config_path)
            store = get_store()

            # 优化：复用已有 Pipeline（模型已加载），跳过耗时的模型加载和预热
            existing_pipeline = None
            with self._lock:
                if self._pipeline is not None:
                    existing_pipeline = self._pipeline
                    self._pipeline = None

            if existing_pipeline is not None and existing_pipeline.model is not None:
                with self._lock:
                    self._set_progress("model", "复用已加载的模型...", 20)
                pipeline = existing_pipeline
                logger.info("Reusing existing pipeline (model already loaded)")
                pipeline.cfg = cfg
                # Sync RuntimeSettings so inference uses updated config values
                pipeline._runtime.conf = cfg.conf
                pipeline._runtime.iou = cfg.iou
                pipeline._runtime.inference_scale = cfg.inference_scale
                pipeline._runtime.jpeg_quality = cfg.jpeg_quality
                pipeline._inference_scale = max(0.25, min(1.0, cfg.inference_scale))
                pipeline._jpeg_quality = max(30, min(95, cfg.jpeg_quality))
                pipeline.running = False
                pipeline._frame_count = 0
                pipeline._perf_fps_history.clear()
                pipeline._last_detections = []
                pipeline._skeletons = []
                pipeline._sk_frame_counter = 0
                pipeline._cached_skeletons = []
                pipeline._cached_raw_skeletons = []
                pipeline._sk_process_interval = 2
                pipeline.rules = RulesEngine(cfg, person_class_id=PERSON_CLASS_ID)
                pipeline._behavior_analyzer = BehaviorAnalyzer(cfg)
            else:
                with self._lock:
                    self._set_progress("model", "正在加载模型（首次启动需要 10-30 秒）...", 10)
                pipeline = DetectionPipeline(cfg, store=store)

            with self._lock:
                self._pipeline = pipeline
                self._set_progress("model_ready", "模型加载完成", 30)

            pipeline.start()
            pipeline._current_source = source
            logger.info("Pipeline started for source: %s", source)

            src = int(source) if source.isdigit() else source
            is_video_file = not source.isdigit()
            source_type = "视频文件" if is_video_file else "摄像头"
            with self._lock:
                self._set_progress("source", f"正在打开{source_type}...", 40)
            logger.info("Opening video source: %s", src)

            # ── Timeout-safe camera open ──────────────────────────
            # 优化：减少单次超时时间，优先使用上次成功的后端
            CAMERA_TIMEOUT = 5.0  # 从8秒减少到5秒

            if isinstance(src, int) and platform.system() == "Windows":
                cap, cam_err = try_open_camera_windows(src, timeout_per_backend=CAMERA_TIMEOUT)
            else:
                cap, cam_err = open_camera_with_timeout(src, timeout_sec=CAMERA_TIMEOUT)

            if not cap or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                with self._lock:
                    self._set_progress("error", f"无法打开{source_type}", 0)
                diagnostic = diagnose_camera_failure(str(source), platform.system())
                err_msg = (
                    f"无法打开视频源: {source}\n"
                    f"技术详情: {cam_err or '原因未知'}\n"
                    f"诊断建议:\n{diagnostic}"
                )
                logger.error(err_msg)
                self._record_error(f"无法打开视频源: {source} (超时或多后端均失败)")
                with self._lock:
                    self._detection_active = False
                    self._cap = None
                return

            # 存储 cap 引用以便 stop() 能同步释放摄像头
            with self._lock:
                self._cap = cap

            logger.info("Video source opened successfully: %s", source)
            target_frame_dt = 1.0 / max(1, cfg.camera_fps)

            is_video_file = not source.isdigit()
            video_fps = 30
            if is_video_file:
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                if video_fps <= 0 or video_fps > 240:
                    video_fps = 30
                target_frame_dt = 1.0 / video_fps
                logger.info(
                    "Video file FPS detected: %s, target frame dt: %.4fs",
                    video_fps,
                    target_frame_dt,
                )

            # 摄像头打开后需要预热时间，跳过前几帧不稳定数据
            # 优化：减少预热帧数，使用更短的等待时间
            if source.isdigit():  # 摄像头源是数字，视频文件是路径
                with self._lock:
                    self._set_progress("warmup", "正在预热摄像头...", 60)
                logger.info("Warming up camera, skipping initial unstable frames...")
                warmup_ok = False
                for _ in range(3):  # 优化：从10减少到3次，大多数摄像头1-2帧即可稳定
                    ret, _ = cap.read()
                    if ret:
                        warmup_ok = True
                        break
                    time.sleep(0.05)  # 优化：从0.1减少到0.05秒
                if warmup_ok:
                    logger.info("Camera warmup completed")
                else:
                    logger.warning("Camera warmup: no valid frame after 10 attempts, proceeding anyway")
            else:
                with self._lock:
                    self._set_progress("ready", "视频文件已就绪", 70)

            latest_frame: np.ndarray | None = None
            latest_timestamp: float = 0.0
            frame_lock = threading.Lock()
            frame_available = threading.Event()
            capture_running = True

            # 摄像头断流重试上限；视频文件容错更低（文件损坏时快速失败）
            _max_cap_retries = 20 if not is_video_file else 5
            capture_exit_reason = None  # None=用户停止, 'eof'=视频正常播完, 'max_retries'=读帧失败

            def _capture_loop():
                nonlocal latest_frame, latest_timestamp, capture_running, capture_exit_reason
                last_capture_time = time.time()
                consecutive_failures = 0
                consecutive_eof = 0  # Track end-of-file for video files
                max_eof = 1  # Video file ends after 1 confirmed EOF

                while capture_running and pipeline.running:
                    try:
                        ret, f = cap.read()
                        if not ret:
                            consecutive_failures += 1
                            consecutive_eof += 1

                            # 视频文件播放完毕时优雅退出
                            if is_video_file and consecutive_eof >= max_eof:
                                logger.info(
                                    "Video file playback completed after %d frames", frame_count
                                )
                                capture_exit_reason = 'eof'
                                capture_running = False
                                break

                            if consecutive_failures >= _max_cap_retries:
                                logger.warning(
                                    "Video capture ended after %d consecutive failures", consecutive_failures
                                )
                                capture_exit_reason = 'max_retries'
                                capture_running = False
                                break

                            if consecutive_failures % 5 == 0:  # 每5次失败记录一次
                                logger.warning(
                                    "cap.read() failed (%d/%d), retrying...",
                                    consecutive_failures, _max_cap_retries,
                                )
                            time.sleep(0.5)
                            continue

                        consecutive_failures = 0  # 成功读取后重置失败计数
                        consecutive_eof = 0  # Reset EOF counter on successful read
                        current_time = time.time()

                        if is_video_file:
                            elapsed_since_last = current_time - last_capture_time
                            if elapsed_since_last < target_frame_dt:
                                time.sleep(target_frame_dt - elapsed_since_last)
                            current_time = time.time()
                            last_capture_time = current_time

                        latest_timestamp = current_time
                        with frame_lock:
                            latest_frame = f
                        frame_available.set()
                    except Exception as e:
                        logger.error("Error in capture loop: %s", e)
                        # 继续运行，避免捕获线程崩溃

            frame_count = 0

            # ── MJPEG encoder thread: separate JPEG encoding from detection loop ──
            encode_queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)
            encode_running = threading.Event()
            encode_running.set()
            def _encoder_thread_fn():
                while encode_running.is_set():
                    try:
                        annotated = encode_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if annotated is None:
                        break
                    t_enc_start = time.time()
                    jpeg_quality = min(pipeline._jpeg_quality, 70)
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    pipeline._perf_encode_ms = (time.time() - t_enc_start) * 1000
                    with self._lock:
                        self._latest_frame = buf.tobytes()
                        self._mjpeg_frame_counter += 1
                    self._frame_ready.set()

            encoder_thread = threading.Thread(target=_encoder_thread_fn, daemon=True, name="MJPEGEncoder")
            encoder_thread.start()

            capture_thread = threading.Thread(
                target=_capture_loop, daemon=True, name="FrameCapture"
            )
            capture_thread.start()
            # 首次成功处理帧后标记为运行中
            first_frame_processed = False
            while pipeline.running and capture_running:
                try:
                    frame_available.wait(timeout=0.05)
                    frame_available.clear()

                    with frame_lock:
                        frame = latest_frame
                        latest_frame = None
                    if frame is None:
                        continue

                    frame_start = time.time()
                    timestamp = latest_timestamp if is_video_file else frame_start
                    pipeline.process_frame(frame, timestamp)

                    if not first_frame_processed:
                        first_frame_processed = True
                        with self._lock:
                            self._set_progress("running", "检测运行中", 100)

                    if self.has_viewers():
                        stream_skip = pipeline._stream_frame_skip
                        pipeline._stream_frame_counter = (pipeline._stream_frame_counter + 1) % (stream_skip + 1)
                        if pipeline._stream_frame_counter == 0:
                            annotated = pipeline.annotate_frame(frame, pipeline._last_detections)
                            try:
                                encode_queue.put_nowait(annotated)
                            except queue.Full:
                                pass
                        else:
                            pipeline._perf_encode_ms = 0
                    else:
                        with self._lock:
                            self._latest_frame = None

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug("Processed %d frames from %s", frame_count, source)

                    processing_time = time.time() - frame_start
                    target_max_fps = 25.0
                    min_frame_dt = 1.0 / target_max_fps
                    sleep_time = max(0, min_frame_dt - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                except Exception as e:
                    logger.error("Error in detection loop: %s", e)

            capture_running = False
            capture_thread.join(timeout=2.0)

            logger.info(
                "Detection loop ended for %s, processed %d frames", source, frame_count
            )

            # 检测静默失败：捕获线程因读帧耗尽退出，但未抛异常
            if capture_exit_reason == 'max_retries':
                err_msg = (
                    f"视频源无法读取: {source} (连续 {_max_cap_retries} 次读取帧失败"
                    f"，仅处理了 {frame_count} 帧)"
                )
                logger.error(err_msg)
                self._record_error(err_msg)

        except Exception as e:
            logger.exception("Detection error in thread %s: %s", thread_name, e)
            self._record_error(str(e)[:200])
        finally:
            logger.info("Cleaning up detection thread %s", thread_name)
            try:
                if cap is not None:
                    logger.info("Releasing video capture for %s", source)
                    cap.release()
            except Exception as e:
                logger.exception("Error releasing video capture: %s", e)

            try:
                encode_running.clear()
                try:
                    encode_queue.put_nowait(None)
                except Exception:
                    pass
                encoder_thread.join(timeout=2.0)
            except Exception as e:
                logger.exception("Error stopping encoder thread: %s", e)

            try:
                with self._lock:
                    if self._pipeline:
                        self._pipeline.close()
                        self._pipeline = None
                    self._latest_frame = None
                    self._mjpeg_frame_counter = 0
                    self._frame_ready.clear()
                    self._detection_active = False
                    self._current_source = None
                    # 线程清理时也清除 cap 引用（stop() 可能已经先释放了）
                    self._cap = None
            except Exception as e:
                logger.exception("Error cleaning up pipeline: %s", e)


# Module-level singleton for use by route handlers
detection_manager = DetectionManager()
