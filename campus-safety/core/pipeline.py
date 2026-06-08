# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】pipeline.py — 主检测流水线（系统中枢）
# 上游依赖：rules.py, skeleton.py, behavior_analyzer.py,
#           alarm_engine.py, events_store.py, video_archiver.py
# 下游调用：backend/detection_manager.py 创建并驱动本流水线
# 核心职责：摄像头取帧 → YOLO推理 → ByteTrack跟踪 → 规则检测 →
#           骨架分析 → 报警生成 → 事件存储 → MJPEG视频流推送
# ──────────────────────────────────────────────────────────

"""Core detection pipeline: captures video, runs YOLO inference, applies rules, stores events."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import AppConfig, RuntimeSettings
from .constants import PERSON_CLASS_ID, get_class_name, get_detection_color
from .feature_indexer import FeatureIndexer
from .events_store import EventsStore
from .gpu_manager import GPUManager
from .rules import Detection, Event, RulesEngine
from .behavior_analyzer import BehaviorAnalyzer
from .skeleton import Skeleton, SkeletonExtractor, SkeletonRenderer
from .video_archiver import VideoClipRecorder
from .tensorrt_utils import (
    is_tensorrt_available,
    TRTModelWrapper,
    TensorRTConverter,
)

logger = logging.getLogger(__name__)


def _write_snapshot(path: str, image_data: np.ndarray) -> None:
    """Write snapshot image to disk in a background thread to avoid blocking detection."""
    try:
        cv2.imwrite(path, image_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception as e:
        logger.warning(f"Snapshot write failed: {e}")


def _index_snapshot_features(
    events_with_snapshots: list[tuple[int, str]],
    store: EventsStore,
) -> None:
    """Encode snapshot images with CLIP and update event feature_blob."""
    indexer = FeatureIndexer.instance()
    if not indexer.available:
        return
    for event_id, snapshot_path in events_with_snapshots:
        vec = indexer.encode_image(snapshot_path)
        if vec is not None:
            blob = indexer.vector_to_blob(vec)
            try:
                with store._lock:
                    cur = store.conn.cursor()
                    cur.execute(
                        "UPDATE events SET feature_blob = ? WHERE id = ?",
                        (sqlite3.Binary(blob), event_id),
                    )
                    store.conn.commit()
            except Exception as e:
                logger.debug("Feature indexing update failed: %s", e)


def _deduplicate_events(events: list[Event]) -> list[Event]:
    """Prefer skeleton events over bbox events for the same (event_type, track_id).

    Skeleton-based rules use multi-signal fusion (torso angle, head velocity,
    aspect ratio, hip displacement, angle rate) and perspective-calibrated
    speed estimation, producing far fewer false positives than bbox heuristics
    (raw pixel speed / simple aspect ratio).

    Events without a track_id (e.g. crowd density events) are always kept.
    """
    if not events:
        return events

    # Group by (event_type, track_id)
    groups: dict[tuple[str, int | None], list[Event]] = {}
    no_tid: list[Event] = []
    for evt in events:
        key = (evt.event_type, evt.track_id)
        if evt.track_id is None:
            no_tid.append(evt)
        else:
            groups.setdefault(key, []).append(evt)

    result: list[Event] = list(no_tid)

    for key, group in groups.items():
        skeleton_events = [
            e for e in group
            if e.extra.get("detection_method", "").startswith("skeleton")
        ]
        if skeleton_events:
            # Keep only skeleton events — discard bbox false positives
            result.extend(skeleton_events)
        else:
            result.extend(group)

    return result


class DetectionPipeline:
    """Main pipeline for real-time object detection and behavior analysis."""

    def __init__(
        self,
        cfg: AppConfig,
        store: EventsStore | None = None,
        preloaded_model: Any | None = None,
        skip_warmup: bool = False,
    ):
        t_init = time.perf_counter()
        self.cfg = cfg
        self.running = False
        self._frame_count = 0
        self._start_time = 0.0
        self._last_detections: list[Detection] = []
        self._alarm_engine = None
        self._skeletons: list[Skeleton] = []
        self._skeleton_enabled = cfg.pose.enabled
        self._current_source: str | None = None
        self._mllm_sidecar = None

        self._sk_process_interval = max(1, min(10, cfg.pose.process_interval))
        self._sk_frame_counter = 0
        self._cached_skeletons: list[Skeleton] = []
        self._cached_raw_skeletons: list[Any] = []

        self._inference_scale = max(0.25, min(1.0, cfg.inference_scale))
        self._jpeg_quality = max(30, min(95, cfg.jpeg_quality))

        self._perf_inference_ms: float = 0.0
        self._perf_skeleton_ms: float = 0.0
        self._perf_rules_ms: float = 0.0
        self._perf_annotate_ms: float = 0.0
        self._perf_encode_ms: float = 0.0
        self._perf_total_ms: float = 0.0
        self._perf_fps_history: list[float] = []
        self._last_cache_clear_frame: int = 0
        self._snapshot_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="snapshot"
        )

        # Resolve GPU device before any model loading
        t0 = time.perf_counter()
        self._gpu_mgr = GPUManager()
        self._device = self._gpu_mgr.resolve_device(cfg.device)
        self._half = cfg.half or self._gpu_mgr.should_use_half(self._device)
        self._use_gpu_preprocess = self._gpu_mgr.is_cuda and self._half
        logger.info(
            "Device resolved in %.0fms: device=%s half=%s cuda=%s",
            (time.perf_counter() - t0) * 1000,
            self._device,
            self._half,
            self._gpu_mgr.is_cuda,
        )

        if self._gpu_mgr.is_cuda:
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, 0)
                logger.info("CUDA memory fraction set to 80%% to prevent OOM")
            except Exception as e:
                logger.debug("Could not set CUDA memory fraction: %s", e)

        # TensorRT mode check
        self._use_tensorrt = cfg.tensorrt.enabled and is_tensorrt_available()
        self._trt_model = None
        self.model = None

        t0 = time.perf_counter()
        if self._use_tensorrt:
            logger.info(f"TensorRT mode enabled, loading engine for: {cfg.model_path}")
            try:
                converter = TensorRTConverter(config=cfg.tensorrt)
                engine_path = converter.convert(
                    model_path=cfg.model_path,
                    imgsz=cfg.imgsz,
                    force_rebuild=False,
                )
                self._trt_model = TRTModelWrapper(
                    engine_path=engine_path,
                    imgsz=cfg.imgsz,
                    conf=cfg.conf,
                    iou=cfg.iou,
                    classes=cfg.classes if cfg.classes else None,
                    device=self._device,
                )
                logger.info(
                    "TensorRT engine loaded in %.0fms: %s",
                    (time.perf_counter() - t0) * 1000,
                    engine_path,
                )
            except Exception as e:
                logger.warning(f"TensorRT initialization failed: {e}, falling back to PyTorch")
                self._use_tensorrt = False
                self._trt_model = None

        if not self._use_tensorrt:
            if preloaded_model is not None:
                self.model = preloaded_model
                logger.info(
                    "Reusing preloaded PyTorch model in %.0fms: %s",
                    (time.perf_counter() - t0) * 1000,
                    cfg.model_path,
                )
            else:
                logger.info(f"Loading PyTorch model: {cfg.model_path}")
                self.model = YOLO(cfg.model_path)
                self.model.eval()
                logger.info(
                    "PyTorch model loaded in %.0fms",
                    (time.perf_counter() - t0) * 1000,
                )

        # Secondary models: lazy-loaded on first use to speed up startup
        self._fall_model_path = cfg.fall_model_path
        self._fight_model_path = cfg.fight_model_path
        self._fall_model: YOLO | None = None
        self._fight_model: YOLO | None = None
        self._model_secondary_frame_skip = 3
        self._mllm_frame_skip = 5
        self._mllm_frame_counter = 0
        self._stream_frame_skip = 1
        self._stream_frame_counter = 0
        self._runtime = RuntimeSettings(cfg)

        if not skip_warmup:
            t0 = time.perf_counter()
            self._warmup_model()
            logger.info("Model warmup completed in %.0fms", (time.perf_counter() - t0) * 1000)
        else:
            logger.info("Skipping model warmup (preloaded model already warmed)")

        logger.info("Pipeline init total: %.0fms", (time.perf_counter() - t_init) * 1000)

        self.rules = RulesEngine(cfg, person_class_id=PERSON_CLASS_ID)

        # Skeleton analysis modules
        self._sk_extractor = SkeletonExtractor(kp_threshold=cfg.pose.kp_threshold)
        self._sk_renderer = SkeletonRenderer()
        self._behavior_analyzer = BehaviorAnalyzer(cfg)

        if store is not None:
            self.store = store
        else:
            db_path = os.path.join(cfg.output_dir, "events.db")
            self.store = EventsStore(db_path=db_path)

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.snapshots_dir, exist_ok=True)

        self._init_alarm_engine()
        if cfg.mllm.enabled:
            self._init_mllm_sidecar()
        else:
            logger.info("MLLM sidecar deferred (disabled in config)")

        self._clip_recorder = VideoClipRecorder(
            pre_seconds=8, post_seconds=4, clip_fps=15,
            output_dir=os.path.join(cfg.output_dir, "clips"),
        )
        logger.info("Video clip recorder initialized")

    def _ensure_fall_model(self) -> YOLO | None:
        if self._fall_model is not None:
            return self._fall_model
        self._fall_model = self._load_secondary_model(self._fall_model_path, "fall posture")
        return self._fall_model

    def _ensure_fight_model(self) -> YOLO | None:
        if self._fight_model is not None:
            return self._fight_model
        self._fight_model = self._load_secondary_model(self._fight_model_path, "fight detection")
        return self._fight_model

    @staticmethod
    def _load_secondary_model(path: str, name: str) -> YOLO | None:
        import os
        if not path or not os.path.exists(path):
            logger.info(f"Secondary model '{name}': not found ({path})")
            return None
        try:
            t0 = time.perf_counter()
            model = YOLO(path, task="detect")
            logger.info(
                "Secondary model '%s': loaded in %.0fms",
                name,
                (time.perf_counter() - t0) * 1000,
            )
            return model
        except Exception as e:
            logger.warning(f"Secondary model '{name}': load failed ({e})")
            return None

    def _classify_posture(self, frame, detections):
        """Run fall posture model on person bbox crops."""
        fall_model = self._ensure_fall_model()
        if fall_model is None or not detections:
            return []
        results = []
        for i, d in enumerate(detections):
            if d.class_id != PERSON_CLASS_ID:
                continue
            x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
            x2, y2 = min(frame.shape[1], int(d.x2)), min(frame.shape[0], int(d.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                preds = fall_model(crop, imgsz=640, verbose=False, device=self._device)
                if preds and preds[0].boxes and len(preds[0].boxes) > 0:
                    best = preds[0].boxes.conf.argmax()
                    results.append((
                        i,
                        fall_model.names.get(int(preds[0].boxes.cls[best].item()), "?"),
                        float(preds[0].boxes.conf[best].item()),
                    ))
            except Exception as e:
                logger.debug("Posture classification error: %s", e)
        return results

    def _detect_fight_model(self, frame):
        """Run fight model on full frame, return violence detections."""
        fight_model = self._ensure_fight_model()
        if fight_model is None:
            return []
        dets = []
        try:
            preds = fight_model(frame, imgsz=640, verbose=False, device=self._device)
            if preds and preds[0].boxes and len(preds[0].boxes) > 0:
                for box in preds[0].boxes:
                    cls_name = fight_model.names.get(int(box.cls.item()), "").lower()
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf.item())
                    dets.append({
                        "bbox": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                        "conf": conf,
                        "class": cls_name,
                    })
        except Exception as e:
            logger.debug("Fight model inference error: %s", e)
        return dets

    def _init_alarm_engine(self) -> None:
        try:
            from backend.alarm_singleton import get_alarm_engine
            self._alarm_engine = get_alarm_engine()
            logger.info("Alarm engine integrated into pipeline")
        except Exception as e:
            logger.warning("Alarm engine not available: %s", e)

    def _init_mllm_sidecar(self) -> None:
        """Initialize MLLM sidecar only when explicitly enabled."""
        if not self.cfg.mllm.enabled:
            return
        try:
            from core.mllm.mllm_sidecar import MLLMSidecar

            self._mllm_sidecar = MLLMSidecar(self.cfg.mllm)
            self._mllm_sidecar.initialize()
            logger.info("MLLM sidecar integrated into pipeline")
        except Exception as e:
            logger.warning("MLLM sidecar not available: %s", e)
            self._mllm_sidecar = None

    def enable_mllm(self, shadow_mode: bool | None = None) -> bool:
        """Lazy-enable MLLM at runtime (loads Qwen2-VL on demand)."""
        from dataclasses import replace

        cfg = self.cfg.mllm
        if shadow_mode is not None:
            cfg = replace(cfg, shadow_mode=shadow_mode)
        self.cfg = replace(self.cfg, mllm=replace(cfg, enabled=True))

        if self._mllm_sidecar is None:
            from core.mllm.mllm_sidecar import MLLMSidecar
            self._mllm_sidecar = MLLMSidecar(self.cfg.mllm)

        self._mllm_sidecar._config = self.cfg.mllm
        self._mllm_sidecar.initialize()
        return self._mllm_sidecar.is_running

    def disable_mllm(self) -> None:
        """Disable MLLM and fully release GPU memory."""
        from dataclasses import replace

        if self._mllm_sidecar is not None:
            self._mllm_sidecar.shutdown()
            self._mllm_sidecar = None
        self.cfg = replace(self.cfg, mllm=replace(self.cfg.mllm, enabled=False))
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("MLLM disabled and GPU memory released")

    def process_frame(self, frame: np.ndarray, timestamp_s: float) -> dict[str, Any]:
        """Process a single frame: detect, track, extract skeletons, apply rules."""
        t0 = time.time()
        self._frame_count += 1
        frame_idx = self._frame_count
        h, w = frame.shape[:2]

        if self._inference_scale < 1.0:
            new_w = int(w * self._inference_scale)
            new_h = int(h * self._inference_scale)
            inference_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            inference_frame = frame

        t_inf_start = time.time()
        if self._use_tensorrt and self._trt_model is not None:
            # TensorRT inference path
            trt_results = self._trt_model.predict(
                inference_frame,
                conf=self._runtime.conf,
                iou=self._runtime.iou,
                classes=self.cfg.classes if self.cfg.classes else None,
                verbose=False,
            )
            # Wrap TRT results to match Ultralytics API
            results = [self._wrap_trt_results(trt_results, inference_frame.shape[:2])]
        else:
            # PyTorch inference path
            with torch.no_grad():
                results = self.model.track(
                    inference_frame,
                    persist=True,
                    conf=self._runtime.conf,
                    iou=self._runtime.iou,
                    imgsz=self.cfg.imgsz,
                    classes=self.cfg.classes if self.cfg.classes else None,
                    tracker=f"{self.cfg.tracker}.yaml" if self.cfg.tracker else "bytetrack.yaml",
                    verbose=False,
                    device=self._device,
                    half=self._half,
                )
        self._perf_inference_ms = (time.time() - t_inf_start) * 1000

        # CUDA memory managed by PyTorch allocator; only clear on extreme pressure
        if self._gpu_mgr.is_cuda and frame_idx - self._last_cache_clear_frame >= 18000:
            try:
                torch.cuda.empty_cache()
                self._last_cache_clear_frame = frame_idx
            except Exception as e:
                logger.debug("CUDA cache clear skipped: %s", e)

        if self._inference_scale < 1.0 and results and results[0].boxes is not None:
            scale = 1.0 / self._inference_scale
            boxes = results[0].boxes
            if boxes.xyxy is not None:
                boxes.xyxy = boxes.xyxy * scale
                results[0].boxes.xyxy = boxes.xyxy

        # Extract detections
        detections = self._extract_detections(results)
        self._last_detections = detections

        # Apply bbox-based rules engine
        frame_events: list[Event] = []
        if detections:
            t_rules_start = time.time()
            frame_events = self.rules.update(detections, frame_idx, timestamp_s)
            self._perf_rules_ms = (time.time() - t_rules_start) * 1000

        # Extract and process skeletons (if pose enabled)
        self._skeletons = []
        if self._skeleton_enabled:
            try:
                t_skel_start = time.time()
                self._sk_frame_counter += 1
                should_process = self._sk_frame_counter % self._sk_process_interval == 0

                if should_process:
                    det_dicts: list[dict] = [
                        {"bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
                         "track_id": d.track_id}
                        for d in detections
                        if d.class_id == PERSON_CLASS_ID
                    ]
                    raw_skeletons = self._sk_extractor.extract(frame, det_dicts)
                    self._cached_raw_skeletons = raw_skeletons
                    sk_events, self._skeletons = self._behavior_analyzer.analyze(
                        raw_skeletons, timestamp_s, frame_idx, h, w,
                    )
                    self._cached_skeletons = self._skeletons
                    frame_events.extend(sk_events)
                    frame_events = _deduplicate_events(frame_events)
                else:
                    frame_events = list(frame_events)  # ensure mutable
            except Exception as e:
                logger.warning("Skeleton analysis error: %s", e)
                self._skeletons = self._cached_skeletons
            finally:
                self._perf_skeleton_ms = (time.time() - t_skel_start) * 1000

        # --- Secondary model inference (every N frames) ---
        # Posture model disabled: per-person GPU inference produces no fused events
        # and competes with main detector for GPU resources, causing frame drops
        if frame_idx % self._model_secondary_frame_skip == 0:
            for fd in self._detect_fight_model(frame):
                if fd["conf"] >= 0.3:
                    cls = fd["class"]
                    # Suspicious Activity 10-class model mapping
                    if cls in ("violence", "fighting", "assaulting"):
                        event_type = "fight"
                    elif cls in ("man_with_gun", "man_with_knife", "knife", "gun"):
                        event_type = "intrusion"
                    elif cls in ("kidnapping", "theft", "theaf_robbery", "terrorist_with_time_bomb"):
                        event_type = "intrusion"
                    else:
                        continue
                    frame_events.append(Event(
                        event_type=event_type, timestamp_s=timestamp_s,
                        frame_index=frame_idx,
                        confidence=fd["conf"], bbox=fd["bbox"],
                        extra={"detection_method": "model_suspicious",
                               "model_class": cls},
                    ))

        # Feed frame to clip recorder (ring buffer for event-triggered recording)
        try:
            self._clip_recorder.feed_frame(frame, timestamp_s)
        except Exception as e:
            logger.debug("Clip recorder feed error: %s", e)

        # Store events with optional skeleton data
        if frame_events:
            # Trigger video clip recording for each event type
            for event in frame_events:
                try:
                    self._clip_recorder.trigger_clip(event.event_type)
                except Exception as e:
                    logger.debug("Clip recorder trigger error: %s", e)
            batch: list[tuple[Event, str | None]] = []
            for event in frame_events:
                if event.source is None:
                    event.source = self._current_source

                snapshot = None
                if self.cfg.save_snapshots and event.bbox:
                    ts_ms = int(timestamp_s * 1000)
                    tid_part = (
                        f"_tid{event.track_id}"
                        if event.track_id is not None
                        else ""
                    )
                    zone_part = f"_{event.zone_name}" if event.zone_name else ""
                    fn = f"{event.event_type}{zone_part}{tid_part}_{ts_ms}.jpg"
                    snapshot = os.path.join(self.cfg.snapshots_dir, fn)
                    try:
                        crop_x1 = max(0, int(event.bbox.get("x1", 0)))
                        crop_y1 = max(0, int(event.bbox.get("y1", 0)))
                        crop_x2 = min(w, int(event.bbox.get("x2", w)))
                        crop_y2 = min(h, int(event.bbox.get("y2", h)))
                        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                            crop_data = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        else:
                            crop_data = frame.copy()
                        # 后台异步写入快照，避免阻塞检测线程
                        self._snapshot_executor.submit(
                            _write_snapshot, snapshot, crop_data
                        )
                    except Exception as e:
                        logger.warning(f"Snapshot prep error: {e}")
                        snapshot = None

                # Attach skeleton keypoints to event extra data
                if self._skeleton_enabled and event.track_id is not None:
                    matching = [s for s in self._skeletons if s.track_id == event.track_id]
                    if matching:
                        skel = matching[0]
                        event.extra.setdefault("skeleton_kps", [
                            {"x": kp.x, "y": kp.y, "conf": kp.confidence}
                            for kp in skel.keypoints
                        ])
                        event.extra["skeleton_avg_conf"] = skel.average_confidence

                batch.append((event, snapshot))
            self.store.record_batch(batch)

            # Submit feature indexing for snapshots in background
            indexed: list[tuple[int, str]] = []
            for event, snapshot in batch:
                if snapshot:
                    # Get the last inserted event id from record_batch
                    with self.store._lock:
                        cur = self.store.conn.cursor()
                        cur.execute(
                            "SELECT id FROM events WHERE snapshot_path = ? ORDER BY id DESC LIMIT 1",
                            (snapshot,),
                        )
                        row = cur.fetchone()
                        if row:
                            indexed.append((row["id"], snapshot))
            if indexed:
                self._snapshot_executor.submit(
                    _index_snapshot_features, indexed, self.store
                )

        if frame_events and self._alarm_engine:
            try:
                self._alarm_engine.process_events(frame_events)
            except Exception as e:
                logger.warning("Alarm processing error: %s", e)

        if self._mllm_sidecar is not None:
            self._mllm_frame_counter = (self._mllm_frame_counter + 1) % self._mllm_frame_skip
            if self._mllm_frame_counter == 0:
                try:
                    det_dicts = [
                        {"class_name": d.name, "confidence": d.conf, "bbox": (d.x1, d.y1, d.x2, d.y2)}
                        for d in detections
                    ]
                    event_dicts = [
                        {"type": e.event_type, "confidence": getattr(e, "confidence", 0)}
                        for e in frame_events
                    ]
                    self._mllm_sidecar.on_frame(
                        frame=frame, detections=det_dicts, events=event_dicts
                    )
                    scene = self._mllm_sidecar.last_scene_description or {}
                    narrative = scene.get("narrative", "") or scene.get("scene_summary", "")
                    if narrative and frame_events:
                        for evt in frame_events:
                            evt.extra.setdefault("mllm_narrative", narrative)
                            evt.extra.setdefault("mllm_risk_level", scene.get("risk_level", ""))
                            evt.extra.setdefault("mllm_action", scene.get("suggested_action", ""))
                except Exception as e:
                    logger.warning("MLLM sidecar frame error: %s", e)

        frame_result: dict[str, Any] = {
            "frame_index": frame_idx,
            "timestamp": timestamp_s,
            "detections": self._serialize_detections(detections),
            "events": frame_events,
        }
        if self._skeleton_enabled and self._skeletons:
            frame_result["skeleton_count"] = len(self._skeletons)
        self._perf_total_ms = (time.time() - t0) * 1000
        return frame_result

    def _wrap_trt_results(self, trt_results: list[dict], img_shape: tuple[int, int]):
        """Wrap TensorRT detection results to match Ultralytics Results API."""
        class _TRTResults:
            def __init__(self, detections, shape):
                self.boxes = _TRTBoxes(detections, shape)
                self.keypoints = None

        class _TRTBoxes:
            def __init__(self, detections, shape):
                import torch
                self._shape = shape
                if detections:
                    self.xyxy = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
                    self.conf = torch.tensor([d["conf"] for d in detections], dtype=torch.float32)
                    self.cls = torch.tensor([d["cls"] for d in detections], dtype=torch.float32)
                    self.id = None  # TensorRT doesn't do tracking natively
                else:
                    self.xyxy = torch.empty((0, 4), dtype=torch.float32)
                    self.conf = torch.empty(0, dtype=torch.float32)
                    self.cls = torch.empty(0, dtype=torch.float32)
                    self.id = None

            def __len__(self):
                return len(self.xyxy)

        return _TRTResults(trt_results, img_shape)

    def _extract_detections(self, results) -> list[Detection]:
        """Extract Detection objects from YOLO results with optimized GPU→CPU transfer."""
        detections = []
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        if len(boxes) == 0:
            return detections

        if self._gpu_mgr.is_cuda:
            try:
                with torch.no_grad():
                    all_data = torch.cat([
                        boxes.xyxy,
                        boxes.conf.unsqueeze(-1) if boxes.conf is not None else torch.full((len(boxes), 1), 1.0, device=boxes.xyxy.device),
                        boxes.cls.unsqueeze(-1) if boxes.cls is not None else torch.zeros((len(boxes), 1), device=boxes.xyxy.device),
                    ], dim=1)
                    ids_tensor = boxes.id if boxes.id is not None else None
                    np_data = all_data.cpu().numpy()
                    np_ids = ids_tensor.cpu().numpy() if ids_tensor is not None else None

                for i in range(len(np_data)):
                    row = np_data[i]
                    track_id = int(np_ids[i]) if np_ids is not None and np_ids[i] is not None else None
                    detections.append(
                        Detection(
                            track_id=track_id,
                            class_id=int(row[5]),
                            conf=float(row[4]),
                            x1=float(row[0]),
                            y1=float(row[1]),
                            x2=float(row[2]),
                            y2=float(row[3]),
                        )
                    )
                return detections
            except Exception as e:
                logger.debug("GPU detection extraction failed, falling back to CPU: %s", e)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        for i in range(len(xyxy)):
            track_id = int(ids[i]) if ids[i] is not None else None
            class_id = int(clss[i]) if i < len(clss) else 0
            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    conf=float(confs[i]),
                    x1=float(xyxy[i][0]),
                    y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]),
                    y2=float(xyxy[i][3]),
                )
            )
        return detections

    def _serialize_detections(
        self, detections: list[Detection]
    ) -> list[dict[str, Any]]:
        return [
            {
                "track_id": d.track_id,
                "class_id": d.class_id,
                "class_name": get_class_name(d.class_id),
                "confidence": d.conf,
                "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
            }
            for d in detections
        ]

    def annotate_frame(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and zone overlays on frame."""
        t_ann_start = time.time()
        out = frame

        if self.cfg.rules.intrusion.enabled:
            for zone in self.cfg.rules.intrusion.zones:
                pts = [(int(x), int(y)) for x, y in zone.polygon]
                if len(pts) >= 3:
                    cv2.polylines(out, [np.array(pts)], True, (255, 0, 0), 2)
                    cv2.putText(
                        out,
                        zone.name,
                        (pts[0][0], max(0, pts[0][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

        if self.cfg.rules.vehicle.enabled:
            for zone in self.cfg.rules.vehicle.zones:
                pts = [(int(x), int(y)) for x, y in zone.polygon]
                if len(pts) >= 3:
                    cv2.polylines(out, [np.array(pts)], True, (0, 165, 255), 2)
                    cv2.putText(
                        out,
                        f"[V] {zone.name}",
                        (pts[0][0], max(0, pts[0][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )

        for d in detections:
            color = get_detection_color(d.class_id)
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{get_class_name(d.class_id)} {d.conf:.2f}"
            if d.track_id is not None:
                label = f"ID:{d.track_id} {label}"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        elapsed = time.time() - self._start_time
        fps = self._frame_count / max(1e-6, elapsed)
        cv2.putText(
            out,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Skeleton analysis runs for behavior detection but not rendered
        # — bounding boxes are cleaner for surveillance display

        self._perf_annotate_ms = (time.time() - t_ann_start) * 1000
        return out

    def get_stats(self) -> dict[str, Any]:
        """Get current pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / max(1e-3, elapsed) if self._frame_count > 0 else 0.0
        
        # 确保 FPS 值是有限数字，避免 JSON 序列化错误
        if not (0 <= fps < 1e6):
            fps = 0.0

        self._perf_fps_history.append(fps)
        if len(self._perf_fps_history) > 60:
            self._perf_fps_history = self._perf_fps_history[-60:]

        avg_fps = sum(self._perf_fps_history) / len(self._perf_fps_history) if self._perf_fps_history else 0.0
        min_fps = min(self._perf_fps_history) if self._perf_fps_history else 0.0
        
        # 确保所有 FPS 值都是有限数字
        avg_fps = max(0.0, min(avg_fps, 1e6))
        min_fps = max(0.0, min(min_fps, 1e6))

        return {
            "running": self.running,
            "frame_count": self._frame_count,
            "elapsed_s": round(elapsed, 1),
            "fps": round(fps, 1),
            "avg_fps": round(avg_fps, 1),
            "min_fps": round(min_fps, 1),
            "events": self.store.get_stats(),
            "performance": {
                "inference_ms": round(self._perf_inference_ms, 1),
                "skeleton_ms": round(self._perf_skeleton_ms, 1),
                "rules_ms": round(self._perf_rules_ms, 1),
                "annotate_ms": round(self._perf_annotate_ms, 1),
                "encode_ms": round(self._perf_encode_ms, 1),
                "total_ms": round(self._perf_total_ms, 1),
                "inference_scale": self._inference_scale,
                "jpeg_quality": self._jpeg_quality,
                "frame_skip_interval": self._sk_process_interval,
                "device": self._device,
                "half_precision": self._half,
                "gpu_preprocess": self._use_gpu_preprocess,
                "tensorrt_enabled": self._use_tensorrt,
            },
            "gpu": self._gpu_mgr.get_status_dict(),
            "mllm": self._mllm_sidecar.get_stats() if self._mllm_sidecar else {"enabled": False},
        }

    def start(self) -> None:
        self.running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._perf_fps_history.clear()
        if self._gpu_mgr.is_cuda:
            self._gpu_mgr.optimize_memory()

    def stop(self) -> None:
        self.running = False

    def cleanup(self) -> None:
        """Clean up resources and free GPU memory."""
        self.running = False

        if self._use_tensorrt and self._trt_model is not None:
            try:
                self._trt_model = None
                logger.info("TensorRT model released")
            except Exception as e:
                logger.warning(f"Error releasing TensorRT model: {e}")

        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.cpu()
                del self.model
                self.model = None
            except Exception as e:
                logger.warning(f"Error releasing model: {e}")

        if hasattr(self, '_snapshot_executor') and self._snapshot_executor:
            try:
                self._snapshot_executor.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")

        if self._gpu_mgr.is_cuda:
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("GPU memory fully released (cleanup)")
            except Exception as e:
                logger.warning(f"Error clearing GPU cache: {e}")

        logger.info("Pipeline cleanup completed")

    def _warmup_model(self) -> None:
        try:
            if self._use_tensorrt and self._trt_model is not None:
                # TensorRT warmup
                dummy = np.random.randint(0, 255, (self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
                for _ in range(2):
                    self._trt_model.predict(dummy, verbose=False)
                logger.info("TensorRT warmup completed (2 dummy inferences)")
                return

            # 启用 cuDNN 自动调优，针对固定输入尺寸选择最快的卷积算法
            if self._device.startswith("cuda"):
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark mode enabled for device: %s", self._device)

            dummy = np.random.randint(0, 255, (self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
            for _ in range(2):  # 优化：从3次减少到2次，足够完成GPU预热
                self.model.predict(
                    dummy,
                    imgsz=self.cfg.imgsz,
                    device=self._device,
                    half=self._half,
                    verbose=False,
                )
            logger.info("Model warmup completed (2 dummy inferences)")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def update_config(self, new_cfg: AppConfig) -> None:
        """Update runtime config safely, handling frozen dataclass constraints.

        Uses dataclasses.replace() to create new frozen instances rather than
        mutating existing ones. Updates both the pipeline config and the MLLM
        sidecar config if active.
        """
        from dataclasses import replace
        self.cfg = replace(self.cfg,
            device=new_cfg.device,
            imgsz=new_cfg.imgsz,
            conf=new_cfg.conf,
            iou=new_cfg.iou,
            inference_scale=new_cfg.inference_scale,
            jpeg_quality=new_cfg.jpeg_quality,
            rules=new_cfg.rules,
            mllm=new_cfg.mllm,
        )
        # Sync RuntimeSettings and internal state so inference uses updated values
        self._runtime.conf = new_cfg.conf
        self._runtime.iou = new_cfg.iou
        self._runtime.inference_scale = new_cfg.inference_scale
        self._runtime.jpeg_quality = new_cfg.jpeg_quality
        self._inference_scale = max(0.25, min(1.0, new_cfg.inference_scale))
        self._jpeg_quality = max(30, min(95, new_cfg.jpeg_quality))
        if self._mllm_sidecar is not None:
            self._mllm_sidecar._config = new_cfg.mllm
        if new_cfg.pose.enabled != self._skeleton_enabled:
            self._skeleton_enabled = new_cfg.pose.enabled
        if new_cfg.mllm.enabled and self._mllm_sidecar is None:
            self._init_mllm_sidecar()
        elif not new_cfg.mllm.enabled and self._mllm_sidecar is not None:
            self._mllm_sidecar.shutdown()
            self._mllm_sidecar = None
        logger.info("Pipeline config updated at runtime")

    def set_jpeg_quality(self, quality: int) -> None:
        """Update JPEG encoding quality dynamically.

        This allows adjusting encoding quality based on network conditions
        or client count without restarting the pipeline.

        Args:
            quality: JPEG quality value (30-95)
        """
        old_quality = self._jpeg_quality
        self._jpeg_quality = max(30, min(95, quality))
        if old_quality != self._jpeg_quality:
            logger.debug("JPEG quality changed: %d -> %d", old_quality, self._jpeg_quality)

    def get_encoding_quality(self) -> int:
        """Get current JPEG encoding quality setting.

        Returns:
            Current JPEG quality value
        """
        return self._jpeg_quality

    def close(self) -> None:
        self.stop()
        self._snapshot_executor.shutdown(wait=False)
        if self._mllm_sidecar is not None:
            self._mllm_sidecar.shutdown()
        # Don't close self.store — it is a shared singleton managed by backend.store.
        # Closing it here would corrupt the global EventsStore connection for subsequent detections.
