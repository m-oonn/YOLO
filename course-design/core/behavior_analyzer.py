# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Skeleton-based behavior analysis engine with rule and ML-based detection."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import AppConfig
from .constants import EVENT_PRIORITIES
from .pose_features import (
    PerFrameFeatureExtractor,
    PerFrameFeatures,
    TemporalFeatureExtractor,
    TemporalFeatures,
    InteractionFeatureExtractor,
)
from .rules import Detection, Event
from .skeleton import (
    Skeleton,
    compute_all_bone_angles,
    compute_torso_angle,
)
from .action_analyzer import ActionAnalyzer, SkeletonSequenceBuffer

logger = logging.getLogger(__name__)

# Approximate calibration: 1 meter in pixels at reference distance
# Used for speed/distance conversion when camera calibration is unavailable
DEFAULT_PX_PER_METER = 100.0


@dataclass
class SkeletonFrameBuffer:
    """Temporal buffer of skeleton frames for sequence analysis."""

    maxlen: int = 30
    frames: list[tuple[float, list[Skeleton]]] = field(default_factory=list)

    def append(self, timestamp: float, skeletons: list[Skeleton]) -> None:
        self.frames.append((timestamp, skeletons))
        if len(self.frames) > self.maxlen:
            self.frames.pop(0)

    def get_recent(self, n: int) -> list[tuple[float, list[Skeleton]]]:
        return self.frames[-n:] if self.frames else []

    def get_track_history(self, track_id: int, n: int = 30) -> list[tuple[float, Skeleton]]:
        """Get history for a specific track."""
        result: list[tuple[float, Skeleton]] = []
        for ts, skeletons in self.frames:
            for skel in skeletons:
                if skel.track_id == track_id:
                    result.append((ts, skel))
                    break
        return result[-n:] if result else []

    def clear(self) -> None:
        self.frames.clear()


class AdaptiveThresholdManager:
    """Adjusts detection thresholds based on environmental statistics and historical performance.

    Uses multi-factor adaptive adjustment:
    - False positive rate feedback
    - Time-of-day patterns
    - Scene complexity (number of people)
    - Historical confidence distribution
    """

    def __init__(self, config: AppConfig):
        self.enabled = config.adaptive_threshold.enabled
        self.adapt_window_s = config.adaptive_threshold.adapt_window_s
        self.min_trigger_count = config.adaptive_threshold.min_trigger_count
        self.sensitivity = config.adaptive_threshold.sensitivity

        # Per-rule statistics
        self._stats: dict[str, deque[dict[str, float]]] = {}
        self._baselines: dict[str, dict[str, float]] = {}
        self._confidence_history: dict[str, deque[float]] = {}
        self._scene_complexity_history: deque[float] = deque(maxlen=30)

    def record_false_positive(self, rule_name: str, context: dict[str, float] | None = None) -> None:
        """Record a false positive event for adaptive adjustment."""
        if not self.enabled:
            return
        if rule_name not in self._stats:
            self._stats[rule_name] = deque(maxlen=100)
        ctx = {"is_fp": True, **(context or {})}
        self._stats[rule_name].append(ctx)

    def record_true_positive(self, rule_name: str, context: dict[str, float] | None = None) -> None:
        """Record a true positive event."""
        if not self.enabled:
            return
        if rule_name not in self._stats:
            self._stats[rule_name] = deque(maxlen=100)
        ctx = {"is_fp": False, **(context or {})}
        self._stats[rule_name].append(ctx)

    def record_confidence(self, rule_name: str, confidence: float) -> None:
        """Record detection confidence for distribution analysis."""
        if not self.enabled:
            return
        if rule_name not in self._confidence_history:
            self._confidence_history[rule_name] = deque(maxlen=50)
        self._confidence_history[rule_name].append(confidence)

    def record_scene_complexity(self, person_count: int) -> None:
        """Record scene complexity for context-aware adjustment."""
        self._scene_complexity_history.append(float(person_count))

    def get_adjusted_threshold(self, rule_name: str, base_threshold: float) -> float:
        """Get threshold adjusted based on multiple factors."""
        if not self.enabled:
            return base_threshold

        adjusted = base_threshold

        # Factor 1: False positive rate feedback
        if rule_name in self._stats and len(self._stats[rule_name]) >= self.min_trigger_count:
            recent = list(self._stats[rule_name])
            fp_ratio = sum(1 for r in recent if r.get("is_fp", False)) / len(recent)
            if fp_ratio > 0.2:
                adjustment = min(fp_ratio, 0.5) * self.sensitivity
                adjusted *= (1.0 + adjustment)
            elif fp_ratio < 0.05:
                # Lower threshold slightly if very few FPs
                adjusted *= 0.95

        # Factor 2: Confidence distribution (avoid threshold in ambiguous zone)
        if rule_name in self._confidence_history and len(self._confidence_history[rule_name]) >= 10:
            confs = list(self._confidence_history[rule_name])
            mean_conf = np.mean(confs)
            std_conf = np.std(confs)
            # If mean confidence is high, we can afford slightly higher threshold
            if mean_conf > 0.8 and std_conf < 0.1:
                adjusted *= 1.02

        # Factor 3: Scene complexity (more people = slightly higher threshold to reduce noise)
        if self._scene_complexity_history:
            avg_complexity = np.mean(list(self._scene_complexity_history))
            if avg_complexity > 10:
                adjusted *= 1.05
            elif avg_complexity > 5:
                adjusted *= 1.02

        return float(adjusted)

    def get_speed_calibration(self, reference_height_px: float, reference_height_m: float = 1.7) -> float:
        """Get pixel-to-meter calibration factor.

        Args:
            reference_height_px: Height of a reference person in pixels.
            reference_height_m: Actual height in meters (default 1.7m for average person).

        Returns:
            Conversion factor from px/s to km/h.
        """
        if reference_height_px <= 0:
            return 1.0
        px_per_m = reference_height_px / reference_height_m
        return 3.6 / px_per_m  # px/s -> km/h (1 m/s = 3.6 km/h)

    def get_stats_summary(self) -> dict[str, Any]:
        """Get summary of adaptive threshold statistics."""
        summary = {}
        for rule_name, stats in self._stats.items():
            recent = list(stats)
            fp_count = sum(1 for r in recent if r.get("is_fp", False))
            summary[rule_name] = {
                "total_records": len(recent),
                "false_positives": fp_count,
                "fp_rate": fp_count / len(recent) if recent else 0.0,
            }
        return summary


class SkeletonRuleBase(ABC):
    """Base class for skeleton-based behavior rules."""

    def __init__(self, config: AppConfig, name: str):
        self.name = name
        self.enabled = True
        self.debounce_s: float = 5.0
        self._last_emit: float = -1e18

    @abstractmethod
    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        ...

    def _check_debounce(self, timestamp: float) -> bool:
        return (timestamp - self._last_emit) >= self.debounce_s

    def _make_event(self, event_type: str, timestamp: float, frame_idx: int,
                    track_id: int | None = None, zone_name: str | None = None,
                    conf: float | None = None,
                    bbox: dict[str, float] | None = None,
                    extra: dict[str, Any] | None = None) -> Event:
        self._last_emit = timestamp
        parts = [event_type]
        if zone_name:
            parts.append(f"in zone {zone_name}")
        if track_id is not None:
            parts.append(f"(track {track_id})")
        full_extra: dict[str, Any] = {**(extra or {})}
        if "detection_method" not in full_extra:
            full_extra["detection_method"] = "skeleton"
        return Event(
            event_type=event_type,
            timestamp_s=timestamp,
            frame_index=frame_idx,
            track_id=track_id,
            zone_name=zone_name,
            confidence=conf,
            bbox=bbox,
            extra=full_extra,
            description=" ".join(parts),
        )


class SkeletonRunningRule(SkeletonRuleBase):
    """Enhanced running detection using skeleton-based speed estimation with gait analysis.
    
    Optimizations implemented:
    1. Fixed calibration: Always compute pixel-to-meter conversion from skeleton height
    2. Improved speed calculation: Uses smoothed velocity from multiple frames
    3. Adaptive duration requirement: Shorter duration for higher speeds
    4. Gait analysis: Detects running-specific motion patterns
    5. Multi-frame confirmation: Reduces false positives while maintaining low latency
    """

    def __init__(self, config: AppConfig):
        super().__init__(config, "skeleton_running")
        sk_cfg = config.rules.skeleton.running
        self.enabled = sk_cfg.enabled
        self.speed_threshold_kmh = sk_cfg.speed_threshold_kmh
        self.min_duration_s = sk_cfg.min_duration_s
        self.debounce_s = sk_cfg.debounce_s
        self._speed_history: dict[int, deque[float]] = {}
        self._position_history: dict[int, deque[tuple[float, float, float]]] = {}
        self._gait_phase: dict[int, float] = {}
        self._running_start_time: dict[int, float] = {}
        
        self.base_speed_threshold_kmh = min(self.speed_threshold_kmh, 8.0)
        self.gait_frequency_threshold_hz = 1.5
        self.vertical_oscillation_threshold = 0.05
        
    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        events: list[Event] = []

        for skel in skeletons:
            if skel.track_id is None or skel.is_low_quality:
                continue

            tid = skel.track_id
            hist = buffer.get_track_history(tid, 10)  # Extended history for better analysis
            if len(hist) < 3:
                continue

            # === Optimized Speed Calculation ===
            # Use linear regression over multiple frames for robust velocity estimation
            speed_px_s, speed_confidence = self._compute_robust_speed(hist, timestamp)
            
            # === Improved Calibration ===
            # Always compute calibration from skeleton height (not just when adaptive_mgr exists)
            calib_factor = self._compute_calibration_factor(skel)
            speed_kmh = speed_px_s * calib_factor
            
            # Track speed history with smoothing
            if tid not in self._speed_history:
                self._speed_history[tid] = deque(maxlen=60)  # 2 seconds at 30 FPS
            self._speed_history[tid].append(speed_kmh)
            
            # === Adaptive Threshold ===
            adj_threshold = self._get_adaptive_speed_threshold(
                tid, speed_kmh, adaptive_mgr
            )
            
            # === Gait Analysis ===
            gait_score = self._analyze_gait_pattern(hist, tid, timestamp)
            
            # === Multi-criteria Decision ===
            # Require either:
            # 1. High speed (>15 km/h) alone, OR
            # 2. Moderate speed (>threshold) + gait confirmation, OR
            # 3. Sustained speed over duration
            is_high_speed = speed_kmh > 15.0
            is_moderate_speed = speed_kmh > adj_threshold
            has_gait_confirmation = gait_score > 0.6
            is_sustained = self._check_sustained_speed(tid, adj_threshold)
            
            should_detect = (
                (is_high_speed and speed_confidence > 0.7) or
                (is_moderate_speed and has_gait_confirmation) or
                (is_moderate_speed and is_sustained and speed_confidence > 0.5)
            )
            
            if should_detect:
                if tid not in self._running_start_time:
                    self._running_start_time[tid] = timestamp
                if self._check_debounce(timestamp):
                    required_duration = self._get_required_duration(speed_kmh)
                    elapsed = timestamp - self._running_start_time[tid]
                    if elapsed >= required_duration:
                        conf = self._compute_confidence(speed_kmh, gait_score, speed_confidence)
                        
                        if adaptive_mgr:
                            adaptive_mgr.record_confidence("running", conf)
                        
                        x1, y1, x2, y2 = skel.bbox
                        events.append(self._make_event(
                            "running", timestamp, frame_idx, tid,
                            conf=conf,
                            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            extra={
                                "speed_kmh": float(speed_kmh),
                                "speed_px_s": float(speed_px_s),
                                "calibration_factor": float(calib_factor),
                                "gait_score": float(gait_score),
                                "adjusted_threshold": float(adj_threshold),
                                "detection_method": "skeleton_speed_gait",
                            },
                        ))
            elif speed_kmh < adj_threshold * 0.6:
                self._speed_history.pop(tid, None)
                self._position_history.pop(tid, None)
                self._gait_phase.pop(tid, None)
                self._running_start_time.pop(tid, None)

        return events
    
    def _compute_robust_speed(self, hist: list[tuple[float, Skeleton]], 
                             timestamp: float) -> tuple[float, float]:
        """Compute speed using linear regression over multiple frames.
        
        Returns:
            Tuple of (speed_px_s, confidence_score)
        """
        if len(hist) < 2:
            return 0.0, 0.0
        
        # Use last N frames for speed calculation
        n = min(len(hist), 5)
        positions = []
        times = []
        
        for i in range(len(hist) - n, len(hist)):
            t, skel = hist[i]
            cx, cy = skel.center
            positions.append((cx, cy))
            times.append(t)
        
        # Compute displacements and time differences
        total_dist = 0.0
        total_time = 0.0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            dt = max(1e-6, times[i] - times[i-1])
            
            # Filter out unrealistic jumps (noise)
            if dist < 500:  # Less than 500 pixels per frame
                total_dist += dist
                total_time += dt
        
        if total_time < 1e-6:
            return 0.0, 0.0
        
        speed_px_s = total_dist / total_time
        
        # Confidence based on consistency of motion
        if len(positions) >= 3:
            speeds = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                dt = max(1e-6, times[i] - times[i-1])
                speeds.append(dist / dt)
            
            if speeds:
                mean_speed = sum(speeds) / len(speeds)
                std_speed = (sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)) ** 0.5
                cv = std_speed / max(mean_speed, 1e-6)  # Coefficient of variation
                confidence = max(0.0, 1.0 - cv)  # Lower variation = higher confidence
            else:
                confidence = 0.5
        else:
            confidence = 0.5
        
        return speed_px_s, confidence
    
    def _compute_calibration_factor(self, skel: Skeleton) -> float:
        """Compute pixel-to-meter calibration factor from skeleton height.
        
        Uses anthropometric data: average person height = 1.7m
        """
        bbox_h = skel.bbox["y2"] - skel.bbox["y1"]
        
        # Estimate person height in pixels from bounding box
        # Bounding box typically captures ~90% of actual height
        person_height_px = bbox_h / 0.9
        
        if person_height_px < 50:  # Too small, unreliable
            person_height_px = 170.0  # Default to medium distance
        
        # Pixels per meter
        px_per_m = person_height_px / 1.7
        
        # Convert px/s to km/h: (px/s) * (m/px) * (3600 s/h) / (1000 m/km) = (px/s) * 3.6 / (px/m)
        calib_factor = 3.6 / px_per_m
        
        # Clamp to reasonable range
        calib_factor = max(0.01, min(calib_factor, 1.0))
        
        return calib_factor
    
    def _get_adaptive_speed_threshold(self, tid: int, current_speed: float,
                                     adaptive_mgr: AdaptiveThresholdManager | None) -> float:
        """Get speed threshold adjusted based on context and history."""
        base_threshold = self.base_speed_threshold_kmh
        
        # Apply adaptive threshold manager adjustments
        if adaptive_mgr:
            base_threshold = adaptive_mgr.get_adjusted_threshold("running", base_threshold)
        
        # Dynamic adjustment based on recent speed history
        if tid in self._speed_history and len(self._speed_history[tid]) >= 5:
            recent_speeds = list(self._speed_history[tid])[-5:]
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            speed_variance = sum((s - avg_speed) ** 2 for s in recent_speeds) / len(recent_speeds)
            
            # If speed is stable and close to threshold, lower threshold slightly
            if speed_variance < 2.0 and abs(avg_speed - base_threshold) < 2.0:
                base_threshold *= 0.95
        
        return base_threshold
    
    def _analyze_gait_pattern(self, hist: list[tuple[float, Skeleton]], 
                             tid: int, timestamp: float) -> float:
        """Analyze gait pattern to distinguish running from other fast motions.
        
        Running typically has:
        - Higher stride frequency (>1.5 Hz)
        - More vertical oscillation
        - Regular periodic pattern
        
        Returns:
            Gait score from 0.0 (walking) to 1.0 (clear running gait)
        """
        if len(hist) < 10:
            return 0.5  # Neutral score with insufficient data
        
        # Analyze vertical motion (hip center oscillation)
        y_positions = [skel.center[1] for _, skel in hist]
        y_mean = sum(y_positions) / len(y_positions)
        y_std = (sum((y - y_mean) ** 2 for y in y_positions) / len(y_positions)) ** 0.5
        
        # Normalize vertical oscillation by person height
        avg_bbox_h = sum(skel.bbox["y2"] - skel.bbox["y1"] for _, skel in hist) / len(hist)
        normalized_oscillation = y_std / max(avg_bbox_h, 1.0)
        
        # Running typically has normalized oscillation > 0.03
        oscillation_score = min(1.0, normalized_oscillation / 0.05)
        
        # Estimate stride frequency from speed variations
        if tid in self._speed_history and len(self._speed_history[tid]) >= 10:
            speeds = list(self._speed_history[tid])[-10:]
            
            # Count zero crossings in speed derivative (proxy for stride frequency)
            speed_diffs = [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
            zero_crossings = sum(1 for i in range(1, len(speed_diffs)) 
                               if speed_diffs[i] * speed_diffs[i-1] < 0)
            
            duration = hist[-1][0] - hist[0][0]
            if duration > 0:
                stride_freq = zero_crossings / (2.0 * duration)  # Full stride cycle = 2 zero crossings
                frequency_score = min(1.0, stride_freq / self.gait_frequency_threshold_hz)
            else:
                frequency_score = 0.5
        else:
            frequency_score = 0.5
        
        # Combine scores
        gait_score = (oscillation_score * 0.4 + frequency_score * 0.6)
        
        return gait_score
    
    def _check_sustained_speed(self, tid: int, threshold: float) -> bool:
        """Check if speed has been sustained above threshold."""
        if tid not in self._speed_history or len(self._speed_history[tid]) < 3:
            return False
        
        recent = list(self._speed_history[tid])[-3:]
        return all(s > threshold for s in recent)
    
    def _get_required_duration(self, speed_kmh: float) -> float:
        if speed_kmh > 20.0:
            return 0.3
        elif speed_kmh > 15.0:
            return 0.5
        elif speed_kmh > 12.0:
            return 0.7
        else:
            return self.min_duration_s
    
    def _compute_confidence(self, speed_kmh: float, gait_score: float, 
                           speed_confidence: float) -> float:
        """Compute detection confidence from multiple factors."""
        # Speed component: higher speed = higher confidence
        speed_component = min(1.0, (speed_kmh - self.base_speed_threshold_kmh) / 15.0)
        
        # Gait component: stronger running gait pattern = higher confidence
        gait_component = gait_score
        
        # Measurement confidence
        measurement_component = speed_confidence
        
        # Weighted combination
        confidence = (
            speed_component * 0.5 +
            gait_component * 0.3 +
            measurement_component * 0.2
        )
        
        return min(0.99, max(0.0, confidence))


class SkeletonFallRule(SkeletonRuleBase):
    """Enhanced fall detection using multi-signal fusion.
    
    Optimizations:
    1. Multi-signal fusion: torso angle + head velocity + bbox aspect ratio + hip displacement
    2. Relative head height normalization (camera-distance independent)
    3. Gradual fall detection for slow slides
    4. Improved emergency detection with adaptive velocity threshold
    5. Better state machine with recovery tracking
    """

    def __init__(self, config: AppConfig):
        super().__init__(config, "skeleton_fall")
        sk_cfg = config.rules.skeleton.fall
        self.enabled = sk_cfg.enabled
        self.torso_angle_threshold = sk_cfg.torso_angle_threshold
        self.head_height_threshold = sk_cfg.head_height_threshold
        self.fall_velocity_threshold = sk_cfg.fall_velocity_threshold
        self.min_duration_s = sk_cfg.min_duration_s
        self.debounce_s = sk_cfg.debounce_s
        self._fall_state: dict[int, dict[str, Any]] = {}
        self._head_height_history: dict[int, deque[tuple[float, float]]] = {}
        self._hip_y_history: dict[int, deque[tuple[float, float]]] = {}
        self._aspect_history: dict[int, deque[tuple[float, float]]] = {}

        # Thresholds — emergency velocity from config, rest biomechanics-based
        self.emergency_velocity_threshold = -abs(getattr(sk_cfg, 'emergency_velocity_px', 1.5))
        self.aspect_ratio_fall_threshold = 0.8  # h/w < 0.8 indicates horizontal posture
        self.hip_displacement_threshold = 0.3  # Normalized hip drop for fall
        self.gradual_fall_angle_rate = 15.0  # Degrees per second for gradual fall

    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        events: list[Event] = []

        for skel in skeletons:
            if skel.track_id is None or skel.is_low_quality:
                continue

            tid = skel.track_id
            torso_angle = compute_torso_angle(skel)
            head_h = skel.head_height
            
            # Compute bbox aspect ratio (h/w)
            bbox_w = max(1e-6, skel.bbox["x2"] - skel.bbox["x1"])
            bbox_h = max(1e-6, skel.bbox["y2"] - skel.bbox["y1"])
            aspect_ratio = bbox_h / bbox_w
            
            # Get history for velocity and trend analysis
            hist = buffer.get_track_history(tid, 10)
            
            # Compute head velocity
            head_velocity = 0.0
            if len(hist) >= 2:
                prev_head_h = hist[-2][1].head_height
                dt = max(1e-6, timestamp - hist[-2][0])
                head_velocity = (head_h - prev_head_h) / dt
            
            # Track head height history for trend analysis
            if tid not in self._head_height_history:
                self._head_height_history[tid] = deque(maxlen=30)
            self._head_height_history[tid].append((timestamp, head_h))
            
            # Track hip y position for displacement analysis
            cx, cy = skel.center
            if tid not in self._hip_y_history:
                self._hip_y_history[tid] = deque(maxlen=30)
            self._hip_y_history[tid].append((timestamp, cy))
            
            # Track aspect ratio history
            if tid not in self._aspect_history:
                self._aspect_history[tid] = deque(maxlen=30)
            self._aspect_history[tid].append((timestamp, aspect_ratio))
            
            # Compute normalized head height (relative to bbox height)
            # This makes detection camera-distance independent
            normalized_head_h = head_h / max(bbox_h, 1.0) if bbox_h > 0 else 0.5
            
            # Compute hip displacement (normalized drop)
            hip_drop = self._compute_hip_drop(tid, timestamp)
            
            # Compute aspect ratio transition
            aspect_transition = self._compute_aspect_transition(tid, timestamp)
            
            # Compute torso angle rate of change
            angle_rate = self._compute_angle_rate(tid, timestamp, torso_angle)
            
            # Initialize state
            if tid not in self._fall_state:
                self._fall_state[tid] = {
                    "start_time": None,
                    "alerted": False,
                    "emergency": False,
                    "prev_angles": deque(maxlen=10),
                    "fall_confidence": 0.0,
                }
            
            state = self._fall_state[tid]
            state["prev_angles"].append((timestamp, torso_angle))
            
            # === Adaptive threshold ===
            adj_angle_th = self.torso_angle_threshold
            if adaptive_mgr:
                adj_angle_th = adaptive_mgr.get_adjusted_threshold("fall_angle", self.torso_angle_threshold)
            
            # === Multi-signal fall scoring ===
            fall_score = self._compute_fall_score(
                torso_angle=torso_angle,
                adj_angle_th=adj_angle_th,
                head_velocity=head_velocity,
                aspect_ratio=aspect_ratio,
                hip_drop=hip_drop,
                aspect_transition=aspect_transition,
                angle_rate=angle_rate,
            )
            
            state["fall_confidence"] = fall_score
            
            # === Emergency detection: rapid downward motion ===
            if head_velocity < self.emergency_velocity_threshold and fall_score > 0.5:
                state["emergency"] = True
                if self._check_debounce(timestamp):
                    conf = min(0.99, 0.8 + abs(head_velocity) / 10.0)
                    if adaptive_mgr:
                        adaptive_mgr.record_confidence("fall", conf)
                    x1, y1, x2, y2 = skel.bbox
                    events.append(self._make_event(
                        "fall", timestamp, frame_idx, tid,
                        conf=conf,
                        bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        extra={
                            "torso_angle": float(torso_angle),
                            "head_velocity": float(head_velocity),
                            "aspect_ratio": float(aspect_ratio),
                            "hip_drop": float(hip_drop),
                            "fall_score": float(fall_score),
                            "emergency": True,
                            "detection_method": "skeleton_emergency_v2",
                        },
                    ))
                    state["alerted"] = True
                continue
            
            # === Standard fall detection with multi-signal fusion ===
            is_tilted = torso_angle > adj_angle_th
            is_low_aspect = aspect_ratio < self.aspect_ratio_fall_threshold
            is_hip_dropped = hip_drop > self.hip_displacement_threshold
            is_rapid_angle = angle_rate > self.gradual_fall_angle_rate
            
            # Fall detected if:
            # 1. Torso tilted + low aspect ratio (strong signal)
            # 2. Torso tilted + hip dropped (moderate signal)
            # 3. High fall score (>0.5) + horizontal posture
            # 4. High fall score (>0.6) + rapid descent + torso tilted
            # 5. Rapid angle change + aspect transition (gradual fall)
            is_horizontal_posture = aspect_ratio < 1.5
            is_rapid_descent = head_velocity > 2.0
            fall_detected = (
                (is_tilted and is_low_aspect) or
                (is_tilted and is_hip_dropped) or
                (fall_score > 0.5 and is_horizontal_posture) or
                (fall_score > 0.6 and is_rapid_descent and is_tilted) or
                (is_rapid_angle and aspect_transition)
            )
            
            if fall_detected:
                if state["start_time"] is None:
                    state["start_time"] = timestamp
                elif (timestamp - state["start_time"]) >= self.min_duration_s:
                    if not state["alerted"] and self._check_debounce(timestamp):
                        conf = min(0.95, 0.5 + fall_score * 0.45)
                        if adaptive_mgr:
                            adaptive_mgr.record_confidence("fall", conf)
                        x1, y1, x2, y2 = skel.bbox
                        events.append(self._make_event(
                            "fall", timestamp, frame_idx, tid,
                            conf=conf,
                            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            extra={
                                "torso_angle": float(torso_angle),
                                "head_velocity": float(head_velocity),
                                "aspect_ratio": float(aspect_ratio),
                                "hip_drop": float(hip_drop),
                                "fall_score": float(fall_score),
                                "emergency": False,
                                "adjusted_threshold": float(adj_angle_th),
                                "detection_method": "skeleton_multisignal_v2",
                            },
                        ))
                        state["alerted"] = True
            else:
                # Reset state if person recovers
                if state["start_time"] is not None and (timestamp - state["start_time"]) > 0.5:
                    state["start_time"] = None
                    state["alerted"] = False
                    state["emergency"] = False
                    state["fall_confidence"] = 0.0

        # Clean up stale state
        active_ids = {s.track_id for s in skeletons if s.track_id is not None}
        for tid in list(self._fall_state.keys()):
            if tid not in active_ids:
                del self._fall_state[tid]
        for tid in list(self._head_height_history.keys()):
            if tid not in active_ids:
                del self._head_height_history[tid]
        for tid in list(self._hip_y_history.keys()):
            if tid not in active_ids:
                del self._hip_y_history[tid]
        for tid in list(self._aspect_history.keys()):
            if tid not in active_ids:
                del self._aspect_history[tid]

        return events
    
    def _compute_fall_score(self, torso_angle: float, adj_angle_th: float,
                           head_velocity: float, aspect_ratio: float,
                           hip_drop: float, aspect_transition: bool,
                           angle_rate: float) -> float:
        """Compute composite fall score from multiple signals.
        
        Returns:
            Score from 0.0 (no fall) to 1.0 (definite fall)
        """
        # Signal 1: Torso angle (0-1) - PRIMARY signal
        # 0° = 0.0, threshold = 0.5, 90°+ = 1.0
        if adj_angle_th > 0:
            angle_score = min(1.0, max(0.0, torso_angle / adj_angle_th))
        else:
            angle_score = 0.0
        
        # Signal 2: Head velocity (downward is positive score)
        # Positive head_velocity means head moving DOWN (y increases)
        # But we need to distinguish slow bending from rapid falling
        # Bending: velocity < 1.0 px/frame, Falling: velocity > 2.0 px/frame
        if head_velocity < 0.5:
            velocity_score = 0.0  # Very slow or upward - not a fall
        elif head_velocity < 1.5:
            velocity_score = head_velocity / 3.0  # Slow descent - low score
        else:
            velocity_score = min(1.0, head_velocity / 3.0)  # Rapid descent
        
        # Signal 3: Aspect ratio (low = horizontal = fall)
        # aspect > 1.5 = upright (0.0), aspect < 0.8 = fallen (1.0)
        if aspect_ratio > 1.5:
            aspect_score = 0.0
        elif aspect_ratio < 0.8:
            aspect_score = 1.0
        else:
            aspect_score = (1.5 - aspect_ratio) / 0.7
        
        # Signal 4: Hip displacement (drop = fall)
        hip_score = min(1.0, max(0.0, hip_drop / 0.3))
        
        # Signal 5: Angle rate of change (fast change = fall)
        rate_score = min(1.0, max(0.0, angle_rate / 90.0))
        
        # Weighted fusion - torso angle is the primary signal
        fall_score = (
            angle_score * 0.40 +
            velocity_score * 0.20 +
            aspect_score * 0.20 +
            hip_score * 0.10 +
            rate_score * 0.10
        )
        
        # Boost if aspect transition detected (upright -> horizontal)
        if aspect_transition:
            fall_score = min(1.0, fall_score * 1.3)
        
        # Boost if multiple strong signals
        strong_signals = sum([
            angle_score > 0.6,
            aspect_score > 0.5,
            velocity_score > 0.3,
            hip_score > 0.3,
        ])
        if strong_signals >= 2:
            fall_score = min(1.0, fall_score * 1.2)
        
        # Suppress if only angle is high but no other signals (likely bending)
        if angle_score > 0.6 and strong_signals < 2 and aspect_score < 0.3:
            fall_score *= 0.5
        
        return fall_score
    
    def _compute_hip_drop(self, tid: int, timestamp: float) -> float:
        """Compute normalized hip displacement (drop) from history.
        
        Returns:
            Normalized hip drop (0.0 = no drop, 1.0 = large drop)
        """
        hist = self._hip_y_history.get(tid)
        if not hist or len(hist) < 5:
            return 0.0
        
        # Compare recent hip y to baseline (earliest in window)
        recent_y = list(hist)[-5:]
        baseline_y = recent_y[0][1]
        current_y = recent_y[-1][1]
        
        # Positive displacement = hip moved down (fall)
        # Normalize by typical person height (~170 pixels)
        displacement = (current_y - baseline_y) / 170.0
        return max(0.0, displacement)
    
    def _compute_aspect_transition(self, tid: int, timestamp: float) -> bool:
        """Detect transition from upright (h>w) to horizontal (w>h) aspect ratio."""
        hist = self._aspect_history.get(tid)
        if not hist or len(hist) < 5:
            return False
        
        recent = list(hist)[-10:]
        if len(recent) < 3:
            return False
        
        # Check if aspect ratio transitioned from >1.0 to <1.0
        early_aspects = [ar for _, ar in recent[:len(recent)//2]]
        late_aspects = [ar for _, ar in recent[len(recent)//2:]]
        
        if not early_aspects or not late_aspects:
            return False
        
        early_mean = sum(early_aspects) / len(early_aspects)
        late_mean = sum(late_aspects) / len(late_aspects)
        
        # Transition: was upright (aspect > 1.2) and now horizontal (aspect < 1.0)
        return early_mean > 1.2 and late_mean < 1.0
    
    def _compute_angle_rate(self, tid: int, timestamp: float, current_angle: float) -> float:
        """Compute rate of change of torso angle (degrees per second)."""
        state = self._fall_state.get(tid)
        if not state:
            return 0.0
        
        prev_angles = state.get("prev_angles", deque(maxlen=10))
        if len(prev_angles) < 2:
            return 0.0
        
        prev_t, prev_angle = prev_angles[-2]
        dt = max(1e-6, timestamp - prev_t)
        
        return abs(current_angle - prev_angle) / dt


class SkeletonFightRule(SkeletonRuleBase):
    """Enhanced fight detection using temporal sequence analysis + limb interaction."""

    def __init__(self, config: AppConfig):
        super().__init__(config, "skeleton_fight")
        sk_cfg = config.rules.skeleton.fight
        self.enabled = sk_cfg.enabled
        self.proximity_threshold_m = sk_cfg.proximity_threshold_m
        self.wrist_speed_threshold_ms = sk_cfg.wrist_speed_threshold_ms
        self.limb_frequency_threshold = sk_cfg.limb_frequency_threshold
        self.min_duration_s = sk_cfg.min_duration_s
        self.debounce_s = sk_cfg.debounce_s

        self._fight_state: dict[tuple[int, int], dict[str, Any]] = {}
        self._wrist_history: dict[int, deque[tuple[float, float, float]]] = {}
        self._temporal_buffer: dict[int, deque[tuple[float, float, float]]] = {}
        self._action_analyzer = ActionAnalyzer()

    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        events: list[Event] = []

        valid = [s for s in skeletons if s.track_id is not None and not s.is_low_quality]

        # Update temporal buffer for sequence analysis
        for skel in valid:
            tid = skel.track_id
            cx, cy = skel.center
            if tid not in self._temporal_buffer:
                self._temporal_buffer[tid] = deque(maxlen=30)
            self._temporal_buffer[tid].append((timestamp, cx, cy))

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                skel_a = valid[i]
                skel_b = valid[j]
                tid_a = skel_a.track_id
                tid_b = skel_b.track_id
                key = tuple(sorted([tid_a, tid_b]))

                cx_a, cy_a = skel_a.center
                cx_b, cy_b = skel_b.center
                center_dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

                if center_dist > self.proximity_threshold_m * DEFAULT_PX_PER_METER:
                    self._fight_state.pop(key, None)
                    continue

                # === Temporal sequence analysis ===
                temporal_score = self._compute_temporal_fight_score(tid_a, tid_b, timestamp)

                # === Single-frame features ===
                wrist_speed = self._compute_wrist_speed(skel_a, timestamp)
                limb_freq = self._compute_limb_frequency(tid_a, timestamp)

                # === Adaptive threshold ===
                proximity_th = self.proximity_threshold_m * DEFAULT_PX_PER_METER
                if adaptive_mgr:
                    proximity_th = adaptive_mgr.get_adjusted_threshold("fight_proximity", proximity_th)

                is_close = center_dist < proximity_th
                is_agitated = wrist_speed > self.wrist_speed_threshold_ms * DEFAULT_PX_PER_METER * 0.01
                is_frequent = limb_freq > self.limb_frequency_threshold
                is_temporal_fight = temporal_score > 0.4

                # Combined decision: require temporal confirmation + at least one physical cue
                # For strong temporal signals, relax physical cue requirement
                fight_detected = is_close and is_temporal_fight and (is_agitated or is_frequent or temporal_score > 0.65)

                if key not in self._fight_state:
                    self._fight_state[key] = {
                        "start_time": None,
                        "last_emit": -1e18,
                        "consecutive_frames": 0,
                        "temporal_scores": deque(maxlen=10),
                    }

                state = self._fight_state[key]
                state["temporal_scores"].append(temporal_score)

                if fight_detected:
                    state["consecutive_frames"] += 1
                    if state["start_time"] is None:
                        state["start_time"] = timestamp
                    elif (timestamp - state["start_time"]) >= self.min_duration_s:
                        # Require multi-frame confirmation (3+ consecutive frames)
                        if state["consecutive_frames"] >= 3:
                            if (timestamp - state.get("last_emit", -1e18)) >= self.debounce_s:
                                avg_temporal = sum(state["temporal_scores"]) / len(state["temporal_scores"])
                                conf = min(0.95, 0.6 + avg_temporal * 0.35)
                                bbox = {
                                    "x1": min(skel_a.bbox["x1"], skel_b.bbox["x1"]),
                                    "y1": min(skel_a.bbox["y1"], skel_b.bbox["y1"]),
                                    "x2": max(skel_a.bbox["x2"], skel_b.bbox["x2"]),
                                    "y2": max(skel_a.bbox["y2"], skel_b.bbox["y2"]),
                                }
                                events.append(self._make_event(
                                    "fight", timestamp, frame_idx,
                                    conf=conf,
                                    bbox=bbox,
                                    extra={
                                        "track_ids": [tid_a, tid_b],
                                        "center_distance": float(center_dist),
                                        "wrist_speed": float(wrist_speed),
                                        "limb_frequency": float(limb_freq),
                                        "temporal_score": float(temporal_score),
                                        "avg_temporal_score": float(avg_temporal),
                                        "consecutive_frames": state["consecutive_frames"],
                                        "detection_method": "skeleton_temporal",
                                    },
                                ))
                                state["last_emit"] = timestamp
                                state["start_time"] = None
                                state["consecutive_frames"] = 0
                else:
                    state["start_time"] = None
                    state["consecutive_frames"] = 0

        # Clean up stale states
        valid_ids = {s.track_id for s in valid}
        stale = [k for k in self._fight_state if k[0] not in valid_ids and k[1] not in valid_ids]
        for k in stale:
            del self._fight_state[k]
        for tid in list(self._temporal_buffer.keys()):
            if tid not in valid_ids:
                del self._temporal_buffer[tid]
        for tid in list(self._wrist_history.keys()):
            if tid not in valid_ids:
                del self._wrist_history[tid]

        return events

    def _compute_temporal_fight_score(self, tid_a: int, tid_b: int, timestamp: float) -> float:
        """Compute fight score from temporal sequence analysis.

        Uses motion chaos, mutual proximity, and direction changes.
        """
        buf_a = self._temporal_buffer.get(tid_a, deque())
        buf_b = self._temporal_buffer.get(tid_b, deque())

        if len(buf_a) < 5 or len(buf_b) < 5:
            return 0.0

        seq_a = list(buf_a)
        seq_b = list(buf_b)

        # Mutual proximity score
        proximities = []
        min_len = min(len(seq_a), len(seq_b))
        for i in range(min_len):
            dx = seq_a[i][1] - seq_b[i][1]
            dy = seq_a[i][2] - seq_b[i][2]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            normalized = dist / 150.0
            proximities.append(max(0.0, 1.0 - normalized))
        proximity_score = float(np.mean(proximities)) if proximities else 0.0

        # Chaotic motion score (acceleration variance)
        chaos_scores = []
        for seq in [seq_a, seq_b]:
            if len(seq) < 3:
                continue
            accels = []
            for i in range(2, len(seq)):
                v1 = ((seq[i-1][1] - seq[i-2][1]) ** 2 + (seq[i-1][2] - seq[i-2][2]) ** 2) ** 0.5
                v2 = ((seq[i][1] - seq[i-1][1]) ** 2 + (seq[i][2] - seq[i-1][2]) ** 2) ** 0.5
                accels.append(abs(v2 - v1))
            if accels:
                chaos_scores.append(np.var(accels))
        chaotic_score = float(np.mean(chaos_scores)) if chaos_scores else 0.0

        # Direction changes
        changes = 0
        for seq in [seq_a, seq_b]:
            if len(seq) < 3:
                continue
            for i in range(2, len(seq)):
                dx1 = seq[i-1][1] - seq[i-2][1]
                dy1 = seq[i-1][2] - seq[i-2][2]
                dx2 = seq[i][1] - seq[i-1][1]
                dy2 = seq[i][2] - seq[i-1][2]
                dot = dx1 * dx2 + dy1 * dy2
                mag1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
                mag2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
                if mag1 > 1e-6 and mag2 > 1e-6:
                    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    angle = np.degrees(np.arccos(cos_a))
                    if angle > 90:
                        changes += 1

        # Combined score with calibrated weights
        # Normalize chaotic_score: typical range 0-1000 for fight motion
        normalized_chaos = min(chaotic_score / 100.0, 1.0)
        normalized_changes = min(changes / 3.0, 1.0)

        fight_score = (
            proximity_score * 0.25 +
            normalized_chaos * 0.45 +
            normalized_changes * 0.30
        )
        return fight_score

    def _compute_wrist_speed(self, skel: Skeleton, timestamp: float) -> float:
        """Compute wrist movement speed for a skeleton."""
        kps = skel.keypoints
        if len(kps) < 10:
            return 0.0

        left_wrist = kps[9]
        right_wrist = kps[10]
        if left_wrist.confidence < 0.3 or right_wrist.confidence < 0.3:
            return 0.0

        wrist_x = (left_wrist.x + right_wrist.x) / 2
        wrist_y = (left_wrist.y + right_wrist.y) / 2

        if skel.track_id not in self._wrist_history:
            self._wrist_history[skel.track_id] = deque(maxlen=10)

        hist = self._wrist_history[skel.track_id]
        hist.append((timestamp, wrist_x, wrist_y))

        if len(hist) < 2:
            return 0.0

        _, prev_x, prev_y = hist[-2]
        dt = max(1e-6, timestamp - hist[-2][0])
        return ((wrist_x - prev_x) ** 2 + (wrist_y - prev_y) ** 2) ** 0.5 / dt

    def _compute_limb_frequency(self, track_id: int, timestamp: float) -> float:
        """Estimate limb oscillation frequency from wrist position history."""
        if track_id not in self._wrist_history:
            return 0.0
        hist = list(self._wrist_history[track_id])
        if len(hist) < 10:
            return 0.0

        positions = [p[1] for p in hist]
        if np.std(positions) < 5:
            return 0.0

        centered = np.array(positions) - np.mean(positions)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(centered))))
        duration = hist[-1][0] - hist[0][0]
        return zero_crossings / max(1e-6, duration)


class CrowdDensityAnalyzer(SkeletonRuleBase):
    """Crowd density estimation using Voronoi-based spatial analysis.

    Uses nearest-neighbor distances and Delaunay triangulation for
    more accurate density estimation than simple area-based methods.
    """

    def __init__(self, config: AppConfig):
        super().__init__(config, "skeleton_crowd")
        sk_cfg = config.rules.skeleton.crowd
        self.enabled = sk_cfg.enabled
        self.density_threshold = sk_cfg.density_threshold
        self.min_duration_s = sk_cfg.min_duration_s
        self.debounce_s = sk_cfg.debounce_s
        self._density_active: float | None = None
        self._density_history: deque[float] = deque(maxlen=10)

    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        events: list[Event] = []

        valid = [s for s in skeletons if not s.is_low_quality]
        if len(valid) < 3:
            self._density_active = None
            self._density_history.clear()
            return events

        centers = np.array([s.center for s in valid])
        if len(centers) < 3:
            return events

        # === Multi-metric density estimation ===
        # 1. Nearest-neighbor distance-based density
        nn_density = self._compute_nn_density(centers)

        # 2. Convex hull area density
        hull_density = self._compute_hull_density(centers)

        # 3. Local clustering coefficient (social density)
        social_density = self._compute_social_density(centers)

        # Combine metrics: weighted average
        density = nn_density * 0.4 + hull_density * 0.35 + social_density * 0.25

        # Smooth with history
        self._density_history.append(density)
        smoothed_density = sum(self._density_history) / len(self._density_history)

        # Adaptive threshold
        threshold = self.density_threshold
        if adaptive_mgr:
            threshold = adaptive_mgr.get_adjusted_threshold("crowd", threshold)

        if smoothed_density >= threshold:
            if self._density_active is None:
                self._density_active = timestamp
            elif (timestamp - self._density_active) >= self.min_duration_s:
                if self._check_debounce(timestamp):
                    conf = min(smoothed_density / (threshold * 2), 0.95)
                    events.append(self._make_event(
                        "crowd", timestamp, frame_idx,
                        conf=conf,
                        extra={
                            "people_count": len(valid),
                            "density": float(smoothed_density),
                            "nn_density": float(nn_density),
                            "hull_density": float(hull_density),
                            "social_density": float(social_density),
                            "threshold": float(threshold),
                        },
                    ))
        else:
            self._density_active = None

        return events

    def _compute_nn_density(self, centers: np.ndarray) -> float:
        """Density from average nearest-neighbor distance."""
        n = len(centers)
        if n < 2:
            return 0.0

        # Compute pairwise distances
        dists = np.sqrt(
            np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        )
        # Exclude self-distances (diagonal)
        np.fill_diagonal(dists, np.inf)

        # Average nearest-neighbor distance
        nn_dists = np.min(dists, axis=1)
        avg_nn_dist = np.mean(nn_dists)

        # Density ~ 1 / avg_nn_dist² (normalized)
        # Reference: 1 person per 1m² => avg_nn ~ 1m => density = 1
        density = 1.0 / max((avg_nn_dist / DEFAULT_PX_PER_METER) ** 2, 0.01)
        return float(density)

    def _compute_hull_density(self, centers: np.ndarray) -> float:
        """Density from convex hull area."""
        n = len(centers)
        if n < 3:
            return 0.0

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(centers)
            area_px = hull.volume  # In 2D, volume = area
            area_m2 = area_px / (DEFAULT_PX_PER_METER ** 2)
            density = n / max(area_m2, 0.1)
            return float(density)
        except ImportError:
            # Fallback: bounding box area
            x_range = np.max(centers[:, 0]) - np.min(centers[:, 0])
            y_range = np.max(centers[:, 1]) - np.min(centers[:, 1])
            area_px = x_range * y_range
            area_m2 = area_px / (DEFAULT_PX_PER_METER ** 2)
            return float(n / max(area_m2, 0.1))

    def _compute_social_density(self, centers: np.ndarray) -> float:
        """Social density based on local clustering.

        Counts people within personal space radius (~1.5m).
        """
        n = len(centers)
        if n < 2:
            return 0.0

        personal_space_px = 1.5 * DEFAULT_PX_PER_METER
        dists = np.sqrt(
            np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(dists, np.inf)

        # Count neighbors within personal space for each person
        neighbor_counts = np.sum(dists < personal_space_px, axis=1)
        avg_neighbors = np.mean(neighbor_counts)

        # Normalize: > 4 neighbors in personal space = high density
        density = avg_neighbors / 4.0
        return float(density)


class SkeletonIntrusionRule(SkeletonRuleBase):
    """Enhanced intrusion detection using skeleton center (hip midpoint) for more accurate position."""

    def __init__(self, config: AppConfig):
        super().__init__(config, "skeleton_intrusion")
        sk_cfg = config.rules.skeleton.intrusion
        self.enabled = sk_cfg.enabled
        self.debounce_s = sk_cfg.debounce_s
        self._zones = config.rules.intrusion.zones
        self._intrusion_state: dict[tuple[int, str], bool] = {}

    def detect(self, skeletons: list[Skeleton], timestamp: float,
               frame_idx: int, buffer: SkeletonFrameBuffer,
               adaptive_mgr: AdaptiveThresholdManager | None = None) -> list[Event]:
        events: list[Event] = []
        zones = self._get_zones()
        if not zones:
            return events

        for zone in zones:
            if "polygon" not in zone:
                continue
            polygon = zone["polygon"]
            for skel in skeletons:
                if skel.track_id is None:
                    continue
                tid = skel.track_id
                cx, cy = skel.center
                inside = self._point_in_polygon(cx, cy, polygon)
                key = (tid, zone["name"])

                if inside and not self._intrusion_state.get(key, False):
                    if self._check_debounce(timestamp):
                        x1, y1, x2, y2 = skel.bbox
                        events.append(self._make_event(
                            "intrusion", timestamp, frame_idx, tid,
                            zone_name=zone["name"],
                            conf=0.9,
                            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            extra={"skeleton_center": [float(cx), float(cy)]},
                        ))
                self._intrusion_state[key] = inside

        # Clean up stale state
        active = {(s.track_id, z["name"])
                  for s in skeletons if s.track_id is not None
                  for z in zones}
        stale = [k for k in self._intrusion_state if k not in active]
        for k in stale:
            del self._intrusion_state[k]

        return events

    def _get_zones(self) -> list[dict[str, Any]]:
        return [
            {"name": z.name, "polygon": z.polygon}
            for z in self._zones
        ]

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1e-10, (yj - yi)) + xi):
                inside = not inside
            j = i
        return inside


class BehaviorAnalyzer:
    """Orchestrates skeleton-based behavior analysis across all rules and ML models."""

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.enabled = config.pose.enabled

        # Feature extraction
        self.per_frame_extractor = PerFrameFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(window_size=30)
        self.interaction_extractor = InteractionFeatureExtractor()

        # Frame buffer for temporal analysis
        self.frame_buffer = SkeletonFrameBuffer(maxlen=30)

        # Adaptive threshold manager
        self.adaptive_mgr = AdaptiveThresholdManager(config)

        # Rule-based detectors
        self.rules: list[SkeletonRuleBase] = []
        if config.rules.skeleton.running.enabled:
            self.rules.append(SkeletonRunningRule(config))
        if config.rules.skeleton.fall.enabled:
            self.rules.append(SkeletonFallRule(config))
        if config.rules.skeleton.fight.enabled:
            self.rules.append(SkeletonFightRule(config))
        if config.rules.skeleton.crowd.enabled:
            self.rules.append(CrowdDensityAnalyzer(config))
        if config.rules.skeleton.intrusion.enabled:
            self.rules.append(SkeletonIntrusionRule(config))

    def analyze(self, skeletons: list[Skeleton], timestamp: float,
                frame_idx: int, frame_h: int, frame_w: int) -> tuple[list[Event], list[Skeleton]]:
        """Run all skeleton-based behavior detectors.

        Args:
            skeletons: Raw skeletons from SkeletonExtractor.
            timestamp: Current frame timestamp.
            frame_idx: Current frame index.
            frame_h: Frame height in pixels.
            frame_w: Frame width in pixels.

        Returns:
            Tuple of (detected events list, smoothed skeletons list).
        """
        if not self.enabled or not skeletons:
            return [], []

        # Update frame buffer
        self.frame_buffer.append(timestamp, skeletons)

        # Record scene complexity for adaptive thresholds
        self.adaptive_mgr.record_scene_complexity(len(skeletons))

        # Run all enabled rules
        all_events: list[Event] = []
        for rule in self.rules:
            if rule.enabled:
                try:
                    events = rule.detect(skeletons, timestamp, frame_idx,
                                         self.frame_buffer, self.adaptive_mgr)
                    all_events.extend(events)
                except Exception as e:
                    logger.warning("Rule '%s' error: %s", rule.name, e)

        return all_events, skeletons

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rules_active": len([r for r in self.rules if r.enabled]),
            "rules_total": len(self.rules),
            "buffer_frames": len(self.frame_buffer.frames),
        }
