# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Core detection pipeline: captures video, runs YOLO inference, applies rules, stores events."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import AppConfig
from .constants import PERSON_CLASS_ID, get_class_name, get_detection_color
from .events_store import EventsStore
from .rules import Detection, RulesEngine

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Main pipeline for real-time object detection and behavior analysis."""

    def __init__(self, cfg: AppConfig, store: EventsStore | None = None):
        self.cfg = cfg
        self.running = False
        self._frame_count = 0
        self._start_time = 0.0
        self._last_detections: list[Detection] = []

        # Load YOLO model
        logger.info(f"Loading model: {cfg.model_path}")
        self.model = YOLO(cfg.model_path)
        logger.info("Model loaded successfully")

        # Initialize rules engine
        self.rules = RulesEngine(cfg, person_class_id=PERSON_CLASS_ID)

        # Initialize event store (use shared instance if provided)
        if store is not None:
            self.store = store
        else:
            db_path = os.path.join(cfg.output_dir, "events.db")
            self.store = EventsStore(db_path=db_path)

        # Ensure output directories
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.snapshots_dir, exist_ok=True)

    def process_frame(self, frame: np.ndarray, timestamp_s: float) -> dict[str, Any]:
        """Process a single frame: detect, track, apply rules."""
        self._frame_count += 1
        frame_idx = self._frame_count

        # YOLO inference with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=self.cfg.classes if self.cfg.classes else None,
            verbose=False,
        )

        # Extract detections
        detections = self._extract_detections(results)
        self._last_detections = detections
        frame_events = []

        # Apply rules engine
        if detections:
            frame_events = self.rules.update(detections, frame_idx, timestamp_s)
            if frame_events:
                batch = []
                for event in frame_events:
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
                        cv2.imwrite(snapshot, frame)
                    batch.append((event, snapshot))
                self.store.record_batch(batch)

        return {
            "frame_index": frame_idx,
            "timestamp": timestamp_s,
            "detections": self._serialize_detections(detections),
            "events": frame_events,
        }

    def _extract_detections(self, results) -> list[Detection]:
        """Extract Detection objects from YOLO results."""
        detections = []
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = (
            boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
        )
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
        out = frame.copy()

        # Draw intrusion zones
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

        # Draw detections
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

        # Draw FPS counter
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

        return out

    def get_stats(self) -> dict[str, Any]:
        """Get current pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / max(1e-6, elapsed) if self._frame_count > 0 else 0
        return {
            "running": self.running,
            "frame_count": self._frame_count,
            "elapsed_s": round(elapsed, 1),
            "fps": round(fps, 1),
            "events": self.store.get_stats(),
        }

    def start(self) -> None:
        self.running = True
        self._start_time = time.time()
        self._frame_count = 0

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.stop()
        self.store.close()
