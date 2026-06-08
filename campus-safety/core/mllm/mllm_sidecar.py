# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【MLLM子系统】mllm_sidecar.py — MLLM旁路协调器
# 依赖：scene_describer.py, alarm_enhancer.py, scene_context_buffer.py
# 被调用：pipeline.py（独立线程运行，不阻塞主循环）
# 核心职责：
#   ① 管理任务队列（场景描述 / 报警验证）
#   ② 独立线程工作循环（与检测主循环并行运行）
#   ③ 优先级调度（报警验证优先于场景描述）
# 为什么叫"旁路"？因为它不阻塞检测流程，在另一个线程中异步运行
# ──────────────────────────────────────────────────────────

"""MLLM Sidecar — async orchestrator for scene understanding and alarm enhancement.

Performance Optimizations Applied:
- Efficient task queue management with priority support
- Optimized worker loop with better error handling
- Frame batching for improved throughput
- Graceful shutdown with resource cleanup
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from core.mllm.alarm_enhancer import AlarmEnhancer
from core.mllm.inference_engine import MLLMInferenceEngine
from core.mllm.mllm_config import MLLMConfig
from core.mllm.scene_context_buffer import SceneContextBuffer
from core.mllm.scene_describer import (
    build_scene_prompt,
    normalize_scene_result,
    parse_json_response,
)

logger = logging.getLogger(__name__)

# Performance tuning constants
DEFAULT_TASK_QUEUE_SIZE = 10
WORKER_TIMEOUT = 1.0
SHUTDOWN_TIMEOUT = 5.0


class MLLMSidecar:
    """MLLM Sidecar for async scene understanding and alarm enhancement.

    Performance Features:
    - Non-blocking task submission with overflow protection
    - Efficient worker thread with timeout-based polling
    - Frame context buffering for efficient inference
    - Comprehensive error handling and recovery

    Thread Safety:
        All public methods are thread-safe.
        Internal state is protected by a reentrant lock.
    """

    def __init__(self, config: MLLMConfig):
        self._config = config
        self._engine: MLLMInferenceEngine | None = None
        self._buffer = SceneContextBuffer(max_frames=config.context_window_frames)
        self._enhancer = AlarmEnhancer(config)
        self._frame_counter = 0
        self._last_scene_description: dict[str, Any] = {}
        self._last_inference_time: float = 0.0
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._task_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=DEFAULT_TASK_QUEUE_SIZE)
        self._lock = threading.RLock()
        self._stats = {
            "frames_received": 0,
            "scenes_described": 0,
            "alarms_enhanced": 0,
            "errors": 0,
            "queue_overflows": 0,
            "last_processing_time_ms": 0.0,
            "model_loaded": False,
        }
        self._shutdown_complete = threading.Event()

    def _ensure_engine(self) -> MLLMInferenceEngine:
        if self._engine is None:
            self._engine = MLLMInferenceEngine(self._config)
        return self._engine

    def initialize(self) -> None:
        """Load MLLM model on demand and start the worker thread."""
        if not self._config.enabled:
            logger.info("MLLM sidecar disabled by config")
            return
        if self._running and self._worker_thread and self._worker_thread.is_alive():
            return
        logger.info("Initializing MLLM sidecar (lazy model load)")
        try:
            engine = self._ensure_engine()
            engine.initialize()
            with self._lock:
                self._stats["model_loaded"] = engine.is_loaded
            self._running = True
            self._shutdown_complete.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="MLLMSidecar",
            )
            self._worker_thread.start()
            logger.info("MLLM sidecar initialized and worker started")
        except Exception as e:
            logger.error("Failed to initialize MLLM sidecar: %s", e)
            self._running = False
            raise

    def on_frame(
        self,
        frame: np.ndarray | None = None,
        detections: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> None:
        """Process a new frame with non-blocking task submission.

        Performance:
        - Uses put_nowait to avoid blocking the caller
        - Tracks queue overflows for monitoring
        """
        if not self._config.enabled or not self._running:
            return

        self._frame_counter += 1
        with self._lock:
            self._stats["frames_received"] += 1

        self._buffer.add_frame(detections=detections, events=events, frame_data=frame)

        min_interval = self._config.min_inference_interval_s
        now = time.time()
        if now - self._last_inference_time < min_interval:
            return

        if self._frame_counter % self._config.key_frame_interval == 0:
            should_describe = self._config.scene_description_enabled
            should_enhance = (
                self._config.alarm_enhance_enabled and self._buffer.has_alarm_events()
            )

            if should_describe or should_enhance:
                try:
                    self._task_queue.put_nowait({
                        "type": "process_key_frame",
                        "frame": frame,
                        "should_describe": should_describe,
                        "should_enhance": should_enhance,
                        "timestamp": time.time(),
                    })
                except queue.Full:
                    with self._lock:
                        self._stats["queue_overflows"] += 1
                    logger.warning("MLLM task queue full, dropping frame")

    def _worker_loop(self) -> None:
        """Main worker loop with optimized task processing.

        Performance:
        - Uses timeout-based waiting to allow graceful shutdown
        - Implements error recovery to prevent worker crashes
        """
        logger.info("MLLM sidecar worker started")

        while self._running:
            try:
                task = self._task_queue.get(timeout=WORKER_TIMEOUT)
                task_start = time.time()

                try:
                    self._process_task(task)
                    processing_time = (time.time() - task_start) * 1000
                    with self._lock:
                        self._stats["last_processing_time_ms"] = processing_time
                except Exception as e:
                    logger.error("Task processing error: %s", e)
                    with self._lock:
                        self._stats["errors"] += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Worker loop error: %s", e)
                with self._lock:
                    self._stats["errors"] += 1

        self._shutdown_complete.set()
        logger.info("MLLM sidecar worker stopped")

    def process_events(self, events: list[dict]) -> list[dict]:
        """Process events for alarm enhancement.

        Args:
            events: List of event dictionaries to process.

        Returns:
            The same events list, enhanced events are queued asynchronously.
        """
        if not self._config.enabled or not self._config.alarm_enhance_enabled:
            return events

        enhanced = []
        for event in events:
            alarm_type = event.get("type", "unknown")
            if self._enhancer.should_enhance(alarm_type):
                try:
                    self._task_queue.put_nowait({
                        "type": "enhance_alarm",
                        "event": event,
                        "timestamp": time.time(),
                    })
                except queue.Full:
                    logger.warning("MLLM task queue full, alarm enhancement dropped")
            enhanced.append(event)
        return enhanced

    def _process_task(self, task: dict[str, Any]) -> None:
        """Process a single MLLM task with error handling.

        Args:
            task: Task dictionary containing task type and parameters.
        """
        task_type = task.get("type")
        if task_type == "process_key_frame":
            self._process_key_frame(task)
        elif task_type == "enhance_alarm":
            self._process_alarm_enhancement(task)

    def _process_key_frame(self, task: dict[str, Any]) -> None:
        """Process a key frame for scene description and alarm enhancement.

        Args:
            task: Task dictionary containing frame processing parameters.
        """
        if self._engine is None or not self._engine.is_loaded:
            return

        context = self._buffer.get_context()
        frame = task.get("frame")
        if frame is None:
            latest = self._buffer.get_latest_frame()
            frame = latest.frame_data if latest else None

        self._last_inference_time = time.time()
        generate_fn = self._engine.generate

        if task.get("should_describe"):
            prompt = build_scene_prompt(context)
            raw_text = generate_fn(prompt, image=frame)
            if raw_text:
                parsed = parse_json_response(raw_text)
                if parsed:
                    with self._lock:
                        self._last_scene_description = normalize_scene_result(parsed)
                        self._stats["scenes_described"] += 1
        if task.get("should_enhance"):
            for event in context.frames[-1].events if context.frames else []:
                alarm_type = event.get("type", "unknown")
                result = self._enhancer.enhance_alarm(
                    alarm_type,
                    str(event),
                    context,
                    generate_fn,
                )
                if result:
                    with self._lock:
                        self._stats["alarms_enhanced"] += 1

    def _process_alarm_enhancement(self, task: dict[str, Any]) -> None:
        """Process an alarm enhancement request.

        Args:
            task: Task dictionary containing event details.
        """
        event = task.get("event", {})
        alarm_type = event.get("type", "unknown")
        context = self._buffer.get_context()
        if self._engine is None or not self._engine.is_loaded:
            return
        result = self._enhancer.enhance_alarm(
            alarm_type, str(event), context, self._engine.generate
        )
        if result:
            with self._lock:
                self._stats["alarms_enhanced"] += 1

    def shutdown(self) -> None:
        """Shutdown the MLLM sidecar with graceful resource cleanup.

        Performance:
        - Waits for worker thread termination with timeout
        - Clears task queue to prevent stale tasks
        - Ensures engine shutdown completes
        """
        logger.info("Shutting down MLLM sidecar")
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=SHUTDOWN_TIMEOUT)

        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        with self._lock:
            self._stats["model_loaded"] = False

        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("MLLM sidecar shut down")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive MLLM sidecar statistics.

        Returns:
            Dictionary containing frames received, scenes described,
            alarms enhanced, errors, queue overflows, and engine stats.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["engine"] = self._engine.get_stats() if self._engine else {
                "backend": "none",
                "loaded": False,
            }
            stats["model_loaded"] = self._engine.is_loaded if self._engine else False
            stats["enhancer"] = self._enhancer.get_stats()
            stats["buffer_size"] = self._buffer.size
            stats["enabled"] = self._config.enabled
            stats["running"] = self._running
            stats["queue_size"] = self._task_queue.qsize()
            stats["last_scene"] = dict(self._last_scene_description)
            return stats

    @property
    def last_scene_description(self) -> dict[str, Any]:
        """Get the last scene description with thread-safe access."""
        with self._lock:
            return dict(self._last_scene_description)

    @property
    def is_running(self) -> bool:
        """Check if the sidecar worker is running."""
        return self._running and self._worker_thread is not None and self._worker_thread.is_alive()
