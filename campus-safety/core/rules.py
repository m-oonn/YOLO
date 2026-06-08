# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】rules.py — 6种行为检测规则引擎
# 上游依赖：config.py, constants.py, geometry.py, models.py
# 下游调用：pipeline.py 每帧调用 RulesEngine.apply()
# 核心职责：
#   ① 奔跑检测 — 速度阈值 + 持续时间
#   ② 跌倒检测 — 宽高比6因子打分（直立→倒地全过程）
#   ③ 聚集检测 — BFS空间聚类 + 透视补偿
#   ④ 入侵检测 — 多边形区域判断（4角+中心点）
#   ⑤ 打架检测 — 5因子打分（距离+IoU+接近+混乱+速度）
#   ⑥ 车辆闯入 — 机动车进入禁区检测
# 数据流：Detection[] 输入 → Event[] 输出
# ──────────────────────────────────────────────────────────

"""Rules engine for behavior detection from YOLO detections.

Supports: running, falling, crowd detection, intrusion, fight detection,
and vehicle intrusion detection.
Each rule is independently configurable via the YAML configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .constants import VEHICLE_CLASS_IDS, get_class_name
from .geometry import bbox_aspect_h_over_w, point_in_polygon
from .models import Detection, Event


def _bbox_intersects_polygon(d: Detection, polygon: list[list[float]]) -> bool:
    """Check if any part of a detection's bbox overlaps with a polygon zone.

    Tests the 4 corners + center point of the bbox against the polygon.
    More robust than center-only against partial intrusions (e.g. arm
    reaching into zone while body center is outside).
    """
    corners = [
        (d.x1, d.y1), (d.x2, d.y1), (d.x1, d.y2), (d.x2, d.y2),
        (d.cx, d.cy),
    ]
    return any(point_in_polygon(x, y, polygon) for x, y in corners)


class RulesEngine:
    """Applies behavior detection rules to a stream of detections."""

    def __init__(self, cfg: AppConfig, person_class_id: int = 0):
        self.cfg = cfg
        self.person_class_id = person_class_id

        # Track history buffers
        self._pos_hist: dict[int, deque[tuple[float, float, float]]] = {}
        self._aspect_hist: dict[int, deque[tuple[float, float]]] = {}

        # Running detection state
        self._running_start: dict[int, float | None] = {}
        self._running_last_emit: dict[int, float] = {}

        # Fall detection state
        self._fall_last_emit: dict[int, float] = {}
        self._fall_consecutive: dict[int, int] = {}  # track: consecutive fallen frames

        # Intrusion detection state
        self._intrusion_inside: dict[tuple[int, str], bool] = {}
        self._intrusion_start: dict[tuple[int, str], float] = {}
        self._intrusion_last_emit: dict[tuple[int, str], float] = {}

        # Crowd detection state
        self._crowd_start: float | None = None
        self._crowd_last_emit: float = -1e18

        # Fight detection state
        self._fight_start: dict[tuple[int, int], float | None] = {}
        self._fight_last_emit: dict[tuple[int, int], float] = {}

        # Vehicle intrusion detection state
        self._vehicle_inside: dict[tuple[int, str], bool] = {}
        self._vehicle_start: dict[tuple[int, str], float] = {}
        self._vehicle_last_emit: dict[tuple[int, str], float] = {}

        # Fallback track ID management (when ByteTrack IDs are unavailable)
        self._next_fallback_id: int = 1
        self._prev_fallback_dets: list[tuple[Detection, int]] = []

    def _make_event(
        self,
        event_type: str,
        t: float,
        frame_index: int,
        track_id: int | None = None,
        zone_name: str | None = None,
        conf: float | None = None,
        bbox: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Event:
        parts = [event_type]
        if zone_name:
            parts.append(f"in zone {zone_name}")
        if track_id is not None:
            parts.append(f"(track {track_id})")
        merged_extra: dict[str, Any] = {**(extra or {})}
        if "detection_method" not in merged_extra:
            merged_extra["detection_method"] = "bbox"
        return Event(
            event_type=event_type,
            timestamp_s=t,
            frame_index=frame_index,
            track_id=track_id,
            zone_name=zone_name,
            confidence=conf,
            bbox=bbox,
            extra=merged_extra,
            description=" ".join(parts),
        )

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        """Compute Intersection-over-Union of two bounding boxes."""
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1e-6, (a.x2 - a.x1) * (a.y2 - a.y1))
        area_b = max(1e-6, (b.x2 - b.x1) * (b.y2 - b.y1))
        return inter / (area_a + area_b - inter)

    def _resolve_tid(self, d: Detection) -> int:
        """Resolve track ID, using IOU matching as fallback when ByteTrack ID is absent."""
        if d.track_id is not None:
            return int(d.track_id)

        best_iou = 0.3
        best_id = None
        for prev_d, fid in self._prev_fallback_dets:
            iou = self._iou(d, prev_d)
            if iou > best_iou:
                best_iou = iou
                best_id = fid

        if best_id is not None:
            return best_id

        fid = self._next_fallback_id
        self._next_fallback_id += 1
        return fid

    def update(
        self, detections: list[Detection], frame_index: int, timestamp_s: float
    ) -> list[Event]:
        """Process new detections and return any triggered events."""
        events: list[Event] = []
        persons = [d for d in detections if d.class_id == self.person_class_id]

        # Update history for all person detections
        for d in persons:
            tid = self._resolve_tid(d)
            self._pos_hist.setdefault(tid, deque(maxlen=120)).append(
                (timestamp_s, d.cx, d.cy)
            )
            self._aspect_hist.setdefault(tid, deque(maxlen=120)).append(
                (timestamp_s, d.aspect_h_over_w)
            )

        # Apply each enabled rule
        if self.cfg.rules.running.enabled:
            events.extend(self._detect_running(persons, timestamp_s, frame_index))
        if self.cfg.rules.fall.enabled:
            events.extend(self._detect_fall(persons, timestamp_s, frame_index))
        if self.cfg.rules.crowd.enabled:
            events.extend(self._detect_crowd(persons, timestamp_s, frame_index))
        if self.cfg.rules.intrusion.enabled:
            events.extend(self._detect_intrusion(persons, timestamp_s, frame_index))
        if self.cfg.rules.fight.enabled:
            events.extend(self._detect_fight(persons, timestamp_s, frame_index))
        if self.cfg.rules.vehicle.enabled:
            vehicles = [d for d in detections if d.class_id in VEHICLE_CLASS_IDS]
            events.extend(self._detect_vehicle_intrusion(vehicles, timestamp_s, frame_index))

        # Save untracked person detections for next frame's IOU matching
        self._prev_fallback_dets = [
            (d, self._resolve_tid(d)) for d in persons if d.track_id is None
        ]

        return events

    @staticmethod
    def _px_to_kmh_calibration(bbox_h: float) -> float:
        """Compute px/s → km/h calibration factor from bbox height.

        Uses average person height (1.7 m) and bbox coverage (~90% of
        actual height) to estimate pixels-per-meter, then converts to
        a factor that transforms raw px/s speed into km/h.
        """
        person_height_px = max(bbox_h / 0.9, 30.0)
        px_per_m = person_height_px / 1.7
        return 3.6 / px_per_m

    def _detect_running(
        self, persons: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        for d in persons:
            tid = self._resolve_tid(d)
            hist = self._pos_hist.get(tid)
            if not hist or len(hist) < 2:
                continue

            _, cx_prev, cy_prev = hist[-2]
            _, cx_now, cy_now = hist[-1]
            dt = max(1e-6, t - hist[-2][0])
            speed_px_s = ((cx_now - cx_prev) ** 2 + (cy_now - cy_prev) ** 2) ** 0.5 / dt

            # Perspective correction: convert raw px/s to real-world km/h
            bbox_h = d.y2 - d.y1
            calib = self._px_to_kmh_calibration(bbox_h)
            speed_kmh = speed_px_s * calib

            # Use km/h threshold (12 km/h ≈ jogging) when calibration is
            # reliable (bbox_h > 60 px), fall back to legacy px/s threshold
            # for very distant persons where calibration is noisy.
            speed_ok = (
                speed_kmh >= 10.0
                if bbox_h > 60
                else speed_px_s >= self.cfg.rules.running.speed_px_s
            )

            if speed_ok:
                start = self._running_start.get(tid)
                if start is None:
                    self._running_start[tid] = t
                    continue
                if (t - start) >= self.cfg.rules.running.min_duration_s:
                    last_emit = self._running_last_emit.get(tid, -1e18)
                    if (t - last_emit) >= self.cfg.rules.running.debounce_s:
                        bbox = {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                        events.append(
                            self._make_event(
                                "running",
                                t,
                                frame_idx,
                                tid,
                                conf=d.conf,
                                bbox=bbox,
                                extra={
                                    "speed_px_s": float(speed_px_s),
                                    "speed_kmh": float(speed_kmh),
                                    "calibration_factor": float(calib),
                                    "bbox_height_px": float(bbox_h),
                                },
                            )
                        )
                        self._running_last_emit[tid] = t
                    self._running_start[tid] = None
            else:
                self._running_start[tid] = None
        return events

    def _detect_fall(
        self, persons: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        win = self.cfg.rules.fall.transition_window_s
        confirm_frames = getattr(self.cfg.rules.fall, 'confirm_frames', 6)
        for d in persons:
            tid = self._resolve_tid(d)
            hist = self._aspect_hist.get(tid)
            if not hist or len(hist) < 2:
                self._fall_consecutive[tid] = 0
                continue

            last_emit = self._fall_last_emit.get(tid, -1e18)
            if (t - last_emit) < self.cfg.rules.fall.debounce_s:
                continue

            t_min = t - win
            current_aspect = d.aspect_h_over_w

            # Count consecutive fallen frames
            if current_aspect <= self.cfg.rules.fall.fallen_aspect_max:
                self._fall_consecutive[tid] = self._fall_consecutive.get(tid, 0) + 1
            else:
                self._fall_consecutive[tid] = 0

            fallen_confirm = self._fall_consecutive.get(tid, 0) >= confirm_frames

            # --- Multi-signal scoring (replaces rigid AND gate) ---
            score = 0.0

            # Signal 1: Was upright recently (weight=2)
            upright_seen = any(
                aspect >= self.cfg.rules.fall.upright_aspect_min
                for ts, aspect in hist
                if ts >= t_min
            )
            if upright_seen:
                score += 2.0

            # Signal 2: Sustained fallen posture (weight=2)
            if fallen_confirm:
                score += 2.0

            # Signal 3: Current frame is horizontal (weight=1.5)
            if current_aspect <= self.cfg.rules.fall.fallen_aspect_max:
                score += 1.5

            # Signal 4: Rapid frame-to-frame transition (weight=1.5)
            recent = [(ts, asp) for ts, asp in hist if ts >= t_min]
            if len(recent) >= 2:
                max_delta = max(
                    abs(recent[k][1] - recent[k - 1][1])
                    for k in range(1, len(recent))
                )
                if max_delta >= self.cfg.rules.fall.min_aspect_change_rate:
                    score += 1.5

            # Signal 5: Significant aspect drop in window (weight=1)
            # Catches slow slides where per-frame delta is small but total change matters
            if len(recent) >= 2:
                first_asp = recent[0][1]
                total_drop = first_asp - current_aspect
                if total_drop >= 0.4:
                    score += 1.0

            # Signal 6: Pre-fall state awareness — was at least moderately
            # upright (aspect >= 1.0) recently? Catches sitting→lying falls.
            pre_fall = any(
                aspect >= 1.0
                for ts, aspect in hist
                if ts >= t_min
            )
            if pre_fall and fallen_confirm:
                score += 1.0

            # Threshold: 3.5+ triggers detection. Example passing combos:
            # upright(2) + fallen(2) = 4            classic standing fall
            # fallen(2) + current(1.5) = 3.5        sustained horizontal
            # upright(2) + current(1.5) + drop(1) = 4.5  slow slide
            # fallen(2) + pre_fall(1) + current(1.5) = 4.5  sitting→lying
            if score >= 3.5:
                bbox = {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                events.append(
                    self._make_event(
                        "fall",
                        t,
                        frame_idx,
                        tid,
                        conf=min(0.95, score / 6.0),
                        bbox=bbox,
                        extra={
                            "aspect_h_over_w": float(current_aspect),
                            "confirm_frames": self._fall_consecutive.get(tid, 0),
                            "fall_score": float(score),
                            "upright_seen": upright_seen,
                        },
                    )
                )
                self._fall_last_emit[tid] = t
                self._fall_consecutive[tid] = 0
        return events

    def _detect_crowd(
        self, persons: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        if len(persons) < 2:
            return events

        # Perspective compensation: scale proximity by average bbox height
        avg_h = sum((d.y2 - d.y1) for d in persons) / len(persons)
        ref_h = 200.0  # reference bbox height (~1.7m person at medium distance)
        scale = max(0.4, min(2.5, avg_h / ref_h))
        proximity = self.cfg.rules.crowd.proximity_px * scale

        # Spatial clustering: connect people within proximity_px of each other
        tracked = [(self._resolve_tid(p), p.cx, p.cy) for p in persons]
        n = len(tracked)
        visited = [False] * n
        max_cluster = 0
        for i in range(n):
            if visited[i]:
                continue
            cluster = 0
            stack = [i]
            visited[i] = True
            while stack:
                idx = stack.pop()
                cluster += 1
                _, cx_i, cy_i = tracked[idx]
                for j in range(n):
                    if visited[j]:
                        continue
                    _, cx_j, cy_j = tracked[j]
                    dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
                    if dist <= proximity:
                        visited[j] = True
                        stack.append(j)
            max_cluster = max(max_cluster, cluster)
        threshold_met = max_cluster >= self.cfg.rules.crowd.min_people

        if threshold_met:
            if self._crowd_start is None:
                self._crowd_start = t
            elapsed = t - self._crowd_start
            if elapsed >= self.cfg.rules.crowd.min_duration_s and (
                t - self._crowd_last_emit
            ) >= self.cfg.rules.crowd.debounce_s:
                self._crowd_last_emit = t
                self._crowd_start = None
                events.append(
                    self._make_event(
                        "crowd", t, frame_idx, extra={"people_count": max_cluster}
                    )
                )
        else:
            self._crowd_start = None
        return events

    def _detect_intrusion(
        self, persons: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        for zone in self.cfg.rules.intrusion.zones:
            for d in persons:
                tid = self._resolve_tid(d)
                # Check bbox corners + center for partial overlap with zone
                inside = _bbox_intersects_polygon(d, zone.polygon)
                key = (tid, zone.name)

                if inside and not self._intrusion_inside.get(key, False):
                    # Just entered the zone — start the duration clock
                    self._intrusion_start[key] = t
                elif inside:
                    # Already inside — check if min_duration_s has elapsed
                    start = self._intrusion_start.get(key)
                    if start is not None and (t - start) >= self.cfg.rules.intrusion.min_duration_s:
                        last_emit = self._intrusion_last_emit.get(key, -1e18)
                        if (t - last_emit) >= self.cfg.rules.intrusion.debounce_s:
                            bbox = {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                            events.append(
                                self._make_event(
                                    "intrusion",
                                    t,
                                    frame_idx,
                                    tid,
                                    zone_name=zone.name,
                                    conf=d.conf,
                                    bbox=bbox,
                                )
                            )
                            self._intrusion_last_emit[key] = t
                            self._intrusion_start.pop(key, None)
                else:
                    self._intrusion_start.pop(key, None)
                self._intrusion_inside[key] = inside
        return events

    def _detect_fight(
        self, persons: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        valid = []
        for d in persons:
            tid = self._resolve_tid(d)
            if len(self._pos_hist.get(tid, [])) >= 2:
                valid.append((tid, d))

        rule = self.cfg.rules.fight
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                tid1, d1 = valid[i]
                tid2, d2 = valid[j]
                key = tuple(sorted([tid1, tid2]))
                distance = ((d1.cx - d2.cx) ** 2 + (d1.cy - d2.cy) ** 2) ** 0.5

                # Factor 1: Proximity (weight=2) — close distance
                score = 0
                if distance <= rule.distance_threshold:
                    score += 2

                # Factor 2: Overlap/IoU (weight=2) — bboxes overlap
                if score > 0:
                    iou_val = self._iou(d1, d2)
                    if iou_val >= rule.iou_threshold:
                        score += 2

                # Factor 3: Approaching (weight=1) — closing distance
                if score > 0:
                    prev_dist = self._calc_pair_distance(tid1, tid2)
                    if prev_dist is not None and prev_dist > distance:
                        score += 1

                # Factor 4: Chaos (weight=1) — acceleration variance
                speed1 = self._calc_speed(tid1)
                speed2 = self._calc_speed(tid2)
                chaos1 = self._calc_chaos(tid1)
                chaos2 = self._calc_chaos(tid2)
                chaos = max(chaos1, chaos2)
                if chaos >= rule.chaos_threshold:
                    score += 1

                # Factor 5: Movement (weight=1) — high speed
                if speed1 >= rule.movement_threshold or speed2 >= rule.movement_threshold:
                    score += 1

                if score >= rule.required_score:
                    start = self._fight_start.get(key)
                    if start is None:
                        self._fight_start[key] = t
                        continue
                    if (t - start) >= rule.min_duration_s:
                        last_emit = self._fight_last_emit.get(key, -1e18)
                        if (t - last_emit) >= rule.debounce_s:
                            bbox = {
                                "x1": min(d1.x1, d2.x1),
                                "y1": min(d1.y1, d2.y1),
                                "x2": max(d1.x2, d2.x2),
                                "y2": max(d1.y2, d2.y2),
                            }
                            events.append(
                                self._make_event(
                                    "fight",
                                    t,
                                    frame_idx,
                                    conf=(d1.conf + d2.conf) / 2,
                                    bbox=bbox,
                                    extra={
                                        "track_ids": [tid1, tid2],
                                        "distance": float(distance),
                                        "iou": float(iou_val),
                                        "speeds": [float(speed1), float(speed2)],
                                        "chaos": float(chaos),
                                        "score": score,
                                    },
                                )
                            )
                            self._fight_last_emit[key] = t
                        self._fight_start[key] = None
                else:
                    self._fight_start[key] = None
        return events

    def _detect_vehicle_intrusion(
        self, vehicles: list[Detection], t: float, frame_idx: int
    ) -> list[Event]:
        events = []
        for zone in self.cfg.rules.vehicle.zones:
            for d in vehicles:
                tid = self._resolve_tid(d)
                inside = _bbox_intersects_polygon(d, zone.polygon)
                key = (tid, zone.name)

                if inside and not self._vehicle_inside.get(key, False):
                    self._vehicle_start[key] = t
                elif inside:
                    start = self._vehicle_start.get(key)
                    if start is not None and (t - start) >= self.cfg.rules.vehicle.min_duration_s:
                        last_emit = self._vehicle_last_emit.get(key, -1e18)
                        if (t - last_emit) >= self.cfg.rules.vehicle.debounce_s:
                            bbox = {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                            class_name = get_class_name(d.class_id) if d.class_id < 8 else "vehicle"
                            events.append(
                                self._make_event(
                                    "vehicle_intrusion",
                                    t,
                                    frame_idx,
                                    tid,
                                    zone_name=zone.name,
                                    conf=d.conf,
                                    bbox=bbox,
                                    extra={
                                        "vehicle_type": class_name,
                                        "class_id": d.class_id,
                                    },
                                )
                            )
                            self._vehicle_last_emit[key] = t
                            self._vehicle_start.pop(key, None)
                else:
                    self._vehicle_start.pop(key, None)
                self._vehicle_inside[key] = inside
        return events

    def _calc_speed(self, tid: int) -> float:
        hist = self._pos_hist.get(tid)
        if not hist or len(hist) < 2:
            return 0.0
        t_prev, cx_prev, cy_prev = hist[-2]
        t_now, cx_now, cy_now = hist[-1]
        dt = max(1e-6, t_now - t_prev)
        return ((cx_now - cx_prev) ** 2 + (cy_now - cy_prev) ** 2) ** 0.5 / dt

    def _calc_chaos(self, tid: int) -> float:
        """Compute motion chaos as variance of pixel acceleration (px/s^2)."""
        hist = self._pos_hist.get(tid)
        if not hist or len(hist) < 5:
            return 0.0
        speeds = []
        for k in range(1, len(hist)):
            t_prev, cx_prev, cy_prev = hist[k - 1]
            t_now, cx_now, cy_now = hist[k]
            dt = max(1e-6, t_now - t_prev)
            speed = ((cx_now - cx_prev) ** 2 + (cy_now - cy_prev) ** 2) ** 0.5 / dt
            speeds.append(speed)
        if len(speeds) < 2:
            return 0.0
        mean = sum(speeds) / len(speeds)
        variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
        return float(variance ** 0.5)

    def _calc_pair_distance(self, tid1: int, tid2: int) -> float | None:
        """Return the distance between two track IDs in the previous frame."""
        h1 = self._pos_hist.get(tid1)
        h2 = self._pos_hist.get(tid2)
        if not h1 or not h2 or len(h1) < 2 or len(h2) < 2:
            return None
        _, cx1, cy1 = h1[-2]
        _, cx2, cy2 = h2[-2]
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
