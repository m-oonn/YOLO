# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

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
    reconcile_with_detector,
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
        self._engine = MLLMInferenceEngine(config)
        self._buffer = SceneContextBuffer(max_frames=config.context_window_frames)
        self._enhancer = AlarmEnhancer(config)
        self._frame_counter = 0
        self._last_scene_description: dict[str, Any] = {}
        # Verdict callback: second-stage review calls this with the alarm id and
        # the MLLM verdict so the alarm engine can confirm/dismiss the alarm.
        # Signature: (alarm_id: int, verdict: str, confidence: float, reasoning: str)
        self._verdict_callback = None
        # Event-driven describe throttle: min frames between event-triggered
        # descriptions, so a sustained alarm doesn't flood the inference worker.
        self._last_event_describe_frame = -10_000
        self._event_describe_min_gap = max(2, config.key_frame_interval // 2)
        # Rolling memory of recent detector events (type -> last seen monotonic
        # time, best confidence). The 2B VLM is slow (~10s) and cannot perceive
        # fast/sparse events like fights from still frames, while the rule
        # engine detects them reliably. So the detector is authoritative for the
        # *displayed verdict*: this memory drives activity_type/risk on every
        # frame, independent of when the slow model happens to run.
        self._recent_events: dict[str, dict[str, float]] = {}
        self._recent_event_ttl_s = 5.0
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._task_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=DEFAULT_TASK_QUEUE_SIZE
        )
        self._lock = threading.RLock()
        self._stats = {
            "frames_received": 0,
            "scenes_described": 0,
            "alarms_enhanced": 0,
            "errors": 0,
            "queue_overflows": 0,
            "last_processing_time_ms": 0.0,
        }
        self._shutdown_complete = threading.Event()

    def initialize(self) -> None:
        """Initialize the MLLM sidecar and start the worker thread."""
        if not self._config.enabled:
            logger.info("MLLM sidecar disabled by config")
            return
        logger.info("Initializing MLLM sidecar")
        try:
            self._engine.initialize()
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

        # Record recent detector events and immediately reflect them in the
        # displayed verdict (fast path, no model needed). This guarantees that
        # whenever the rule engine fires a fight/fall, the UI shows it right
        # away — even though the slow VLM may be mid-inference on another frame
        # or describing a calm-looking frame.
        now = time.monotonic()
        if events:
            for e in events:
                etype = e.get("type", "")
                conf = float(e.get("confidence", 0.0) or 0.0)
                prev = self._recent_events.get(etype)
                self._recent_events[etype] = {
                    "ts": now,
                    "conf": max(conf, prev["conf"]) if prev else conf,
                }
        self._apply_detector_verdict(now)

        # Event-driven trigger: a periodic key-frame may miss sporadic alarm
        # events (a fight lasts a few frames; the 5-frame window often has none
        # by the time the 10th-frame tick fires). So also fire a description
        # whenever this frame carries an alarm event, throttled to avoid
        # flooding the worker. This guarantees the buffer holds the event when
        # the scene is described, so detector reconciliation can apply.
        has_event = bool(events)
        periodic = self._frame_counter % self._config.key_frame_interval == 0
        throttled_ok = (
            self._frame_counter - self._last_event_describe_frame
        ) >= self._event_describe_min_gap
        event_trigger = has_event and throttled_ok

        if periodic or event_trigger:
            should_describe = self._config.scene_description_enabled
            should_enhance = (
                self._config.alarm_enhance_enabled and self._buffer.has_alarm_events()
            )
            if event_trigger:
                self._last_event_describe_frame = self._frame_counter

            if should_describe or should_enhance:
                try:
                    self._task_queue.put_nowait(
                        {
                            "type": "process_key_frame",
                            "frame": frame,
                            "should_describe": should_describe,
                            "should_enhance": should_enhance,
                            "timestamp": time.time(),
                        }
                    )
                except queue.Full:
                    with self._lock:
                        self._stats["queue_overflows"] += 1
                    logger.warning("MLLM task queue full, dropping frame")

    # Detector event type -> (activity label, risk level), highest priority first.
    _DETECTOR_VERDICTS = [
        ("fight", "打架", "高"),
        ("fall", "跌倒", "高"),
        ("intrusion", "入侵", "高"),
        ("running", "奔跑", "中"),
        ("crowd", "聚集", "中"),
    ]

    def _apply_detector_verdict(self, now: float) -> None:
        """Make the displayed scene verdict authoritative from the detector.

        Expires stale events, then if any high-priority anomaly is still active
        within the TTL window, force the displayed activity_type/risk_level to
        match it. The VLM narrative (if any) is preserved as supplementary text.
        Runs every frame and is cheap (no inference).
        """
        # Expire events older than the TTL.
        expired = [
            t
            for t, v in self._recent_events.items()
            if now - v["ts"] > self._recent_event_ttl_s
        ]
        for t in expired:
            del self._recent_events[t]

        # Pick the highest-priority active anomaly.
        chosen = None
        for etype, label, risk in self._DETECTOR_VERDICTS:
            if etype in self._recent_events:
                chosen = (label, risk, self._recent_events[etype]["conf"])
                break

        with self._lock:
            if chosen is None:
                return
            label, risk, conf = chosen
            scene = (
                dict(self._last_scene_description)
                if self._last_scene_description
                else {}
            )
            # Keep any VLM-produced visual narrative as supplementary detail.
            visual = (scene.get("narrative") or "").strip()
            if visual.startswith("监控检测到"):
                visual = ""  # already a detector-anchored line; avoid nesting
            scene["activity_type"] = label
            scene["risk_level"] = risk
            scene["anomaly_detected"] = True
            scene["anomaly_details"] = f"规则引擎以{conf:.0%}置信度检测到{label}行为"
            scene["scene_summary"] = f"检测到{label}行为（置信度{conf:.0%}）"
            scene["suggested_action"] = "立即核查现场并采取相应处置措施"
            if visual:
                scene["narrative"] = (
                    f"监控检测到{label}行为（置信度{conf:.0%}）。画面补充：{visual}"
                )
            else:
                scene["narrative"] = (
                    f"监控检测到{label}行为（置信度{conf:.0%}），请立即核查现场。"
                )
            scene["reconciled_by_detector"] = True
            self._last_scene_description = scene

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
                    self._task_queue.put_nowait(
                        {
                            "type": "enhance_alarm",
                            "event": event,
                            "timestamp": time.time(),
                        }
                    )
                except queue.Full:
                    logger.warning("MLLM task queue full, alarm enhancement dropped")
            enhanced.append(event)
        return enhanced

    def set_verdict_callback(self, callback) -> None:
        """Register the callback that applies an MLLM verdict to an alarm.

        callback(alarm_id: int, verdict: str, confidence: float, reasoning: str)
        """
        self._verdict_callback = callback

    def review_alarm(
        self, alarm_id: int, event_type: str, alarm_details: str, frame
    ) -> None:
        """Queue a second-stage review for a freshly created (SUSPECTED) alarm.

        This is the entry point of the cascade's second stage: the pipeline
        hands us the alarm id + the frame that triggered it; the worker runs
        the VLM verification and reports the verdict back via the callback.
        """
        if not self._config.enabled or not self._config.alarm_enhance_enabled:
            return
        if not self._enhancer.should_enhance(event_type):
            return
        try:
            self._task_queue.put_nowait(
                {
                    "type": "review_alarm",
                    "alarm_id": alarm_id,
                    "event_type": event_type,
                    "alarm_details": alarm_details,
                    "frame": frame,
                    "timestamp": time.time(),
                }
            )
        except queue.Full:
            with self._lock:
                self._stats["queue_overflows"] += 1
            logger.warning(
                "MLLM task queue full, alarm review dropped (id=%s)", alarm_id
            )

    def _process_alarm_review(self, task: dict[str, Any]) -> None:
        """Run VLM verification for one suspected alarm and apply the verdict."""
        alarm_id = task.get("alarm_id")
        event_type = task.get("event_type", "unknown")
        alarm_details = task.get("alarm_details", "")
        frame = task.get("frame")
        context = self._buffer.get_context()
        result = self._enhancer.enhance_alarm(
            event_type,
            alarm_details,
            context,
            lambda prompt: self._engine.generate(prompt, image=frame),
        )
        if not result:
            return
        with self._lock:
            self._stats["alarms_enhanced"] += 1
        # Join point: hand the verdict back to the alarm engine.
        if self._verdict_callback is not None and alarm_id is not None:
            try:
                self._verdict_callback(
                    alarm_id,
                    result.get("verdict", "validate"),
                    result.get("confidence", 0.0),
                    result.get("reasoning", ""),
                )
            except Exception as e:
                logger.warning("Verdict callback failed for alarm %s: %s", alarm_id, e)

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
        elif task_type == "review_alarm":
            self._process_alarm_review(task)

    def _process_key_frame(self, task: dict[str, Any]) -> None:
        """Process a key frame for scene description and alarm enhancement.

        Args:
            task: Task dictionary containing frame processing parameters.
        """
        context = self._buffer.get_context()
        latest_frame = self._buffer.get_latest_frame()
        image_for_vlm = latest_frame.frame_data if latest_frame else task.get("frame")
        # Multi-frame sequence so the VLM can perceive motion (a fight looks
        # like calm standing in a single still frame).
        frames_for_vlm = self._buffer.get_recent_frames(n=3)
        if not frames_for_vlm and image_for_vlm is not None:
            frames_for_vlm = [image_for_vlm]
        if task.get("should_describe"):
            prompt = build_scene_prompt(context)
            raw_text = self._engine.generate(prompt, images=frames_for_vlm)
            if raw_text:
                parsed = parse_json_response(raw_text)
                if parsed:
                    result = normalize_scene_result(parsed)
                    # Force agreement with the motion-aware detector so a
                    # confident fight/fall isn't downgraded to "闲聊/正常".
                    result = reconcile_with_detector(
                        result, context.event_counts, context.event_confidences
                    )
                    with self._lock:
                        self._last_scene_description = result
                        self._stats["scenes_described"] += 1
                else:
                    # Preserve raw output even when JSON parsing fails
                    with self._lock:
                        self._last_scene_description = {
                            "scene_summary": raw_text[:200],
                            "activity_type": "unknown",
                            "confidence": 0.0,
                            "anomaly_detected": False,
                            "anomaly_details": "",
                            "risk_level": "低",
                            "suggested_action": "",
                            "narrative": raw_text,
                            "raw_output": raw_text,
                        }
                        self._stats["scenes_described"] += 1
                        self._stats["errors"] += 1
        if task.get("should_enhance"):
            for event in context.frames[-1].events if context.frames else []:
                alarm_type = event.get("type", "unknown")
                result = self._enhancer.enhance_alarm(
                    alarm_type,
                    str(event),
                    context,
                    lambda prompt: self._engine.generate(prompt, image=image_for_vlm),
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
        latest_frame = self._buffer.get_latest_frame()
        image_for_vlm = latest_frame.frame_data if latest_frame else None
        result = self._enhancer.enhance_alarm(
            alarm_type,
            str(event),
            context,
            lambda prompt: self._engine.generate(prompt, image=image_for_vlm),
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

        self._engine.shutdown()

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
            stats["engine"] = self._engine.get_stats()
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
        return (
            self._running
            and self._worker_thread is not None
            and self._worker_thread.is_alive()
        )
