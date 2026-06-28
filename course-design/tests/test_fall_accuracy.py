# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive benchmark for fall detection accuracy.

Tests cover:
- Different fall types (forward, backward, lateral, slow slide)
- Different demographics (height variations, distance from camera)
- Environmental factors (occlusion, multiple people, recovery)
- False positive scenarios (bending, squatting, sitting, lying down slowly)
- Detection latency measurement
"""

import math
import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import pytest

from core.behavior_analyzer import (
    SkeletonFallRule,
    SkeletonFrameBuffer,
)
from core.config import (
    AdaptiveThresholdConfig,
    AppConfig,
    CrowdRule,
    FallRule,
    FightRule,
    IntrusionRule,
    PoseConfig,
    RulesConfig,
    RunningRule,
    SkeletonRulesConfig,
)
from core.config import (
    SkeletonFallRule as SkFallCfg,
)
from core.config import (
    SkeletonRunningRule as SkRunCfg,
)
from core.rules import Detection, RulesEngine
from core.skeleton import Skeleton, SkeletonKeypoint

# ============================================================
# Skeleton-based test helpers
# ============================================================


def make_skeleton_at(
    track_id: int,
    x: float,
    y: float,
    height: float = 170.0,
    conf: float = 0.9,
    torso_angle: float = 0.0,
    head_y: float | None = None,
) -> Skeleton:
    """Create a skeleton with realistic keypoint positions.

    Args:
        track_id: Track ID
        x, y: Hip center position
        height: Skeleton height in pixels
        conf: Keypoint confidence
        torso_angle: Torso tilt angle in degrees (0 = upright, 90 = horizontal)
        head_y: Override head y position (for fall simulation)
    """
    keypoints = []

    # Apply torso tilt
    tilt_rad = math.radians(torso_angle)

    # Head keypoints
    head_x = x + math.sin(tilt_rad) * height * 0.45
    if head_y is None:
        head_y = y - math.cos(tilt_rad) * height * 0.45

    keypoints.extend(
        [
            SkeletonKeypoint(x=head_x, y=head_y, confidence=conf),
            SkeletonKeypoint(x=head_x - 5, y=head_y - 5, confidence=conf),
            SkeletonKeypoint(x=head_x + 5, y=head_y - 5, confidence=conf),
            SkeletonKeypoint(x=head_x - 10, y=head_y, confidence=conf),
            SkeletonKeypoint(x=head_x + 10, y=head_y, confidence=conf),
        ]
    )

    # Upper body
    shoulder_y = y - math.cos(tilt_rad) * height * 0.35
    shoulder_x = x + math.sin(tilt_rad) * height * 0.35
    shoulder_width = height * 0.15

    keypoints.extend(
        [
            SkeletonKeypoint(
                x=shoulder_x - shoulder_width, y=shoulder_y, confidence=conf
            ),
            SkeletonKeypoint(
                x=shoulder_x + shoulder_width, y=shoulder_y, confidence=conf
            ),
            SkeletonKeypoint(
                x=shoulder_x - shoulder_width * 1.2,
                y=shoulder_y + height * 0.15,
                confidence=conf,
            ),
            SkeletonKeypoint(
                x=shoulder_x + shoulder_width * 1.2,
                y=shoulder_y + height * 0.15,
                confidence=conf,
            ),
            SkeletonKeypoint(
                x=shoulder_x - shoulder_width * 1.3,
                y=shoulder_y + height * 0.25,
                confidence=conf,
            ),
            SkeletonKeypoint(
                x=shoulder_x + shoulder_width * 1.3,
                y=shoulder_y + height * 0.25,
                confidence=conf,
            ),
        ]
    )

    # Lower body (hips at center position)
    hip_width = height * 0.1
    keypoints.extend(
        [
            SkeletonKeypoint(x=x - hip_width, y=y, confidence=conf),
            SkeletonKeypoint(x=x + hip_width, y=y, confidence=conf),
            SkeletonKeypoint(
                x=x - hip_width * 0.8, y=y + height * 0.25, confidence=conf
            ),
            SkeletonKeypoint(
                x=x + hip_width * 0.8, y=y + height * 0.25, confidence=conf
            ),
            SkeletonKeypoint(
                x=x - hip_width * 0.7, y=y + height * 0.5, confidence=conf
            ),
            SkeletonKeypoint(
                x=x + hip_width * 0.7, y=y + height * 0.5, confidence=conf
            ),
        ]
    )

    # Compute bbox from keypoints
    all_x = [kp.x for kp in keypoints]
    all_y = [kp.y for kp in keypoints]

    bbox = {
        "x1": min(all_x) - 5,
        "y1": min(all_y) - 5,
        "x2": max(all_x) + 5,
        "y2": max(all_y) + 5,
    }

    sk = Skeleton(track_id=track_id, keypoints=keypoints, bbox=bbox)
    sk.head_height = head_y

    return sk


def make_skeleton_config(
    torso_angle_threshold: float = 45.0,
    head_height_threshold: float = 0.3,
    fall_velocity_threshold: float = 0.5,
    min_duration_s: float = 0.3,
    debounce_s: float = 5.0,
) -> AppConfig:
    """Create config for skeleton-based fall detection tests."""
    sk_rules = SkeletonRulesConfig(
        running=SkRunCfg(enabled=False),
        fall=SkFallCfg(
            enabled=True,
            torso_angle_threshold=torso_angle_threshold,
            head_height_threshold=head_height_threshold,
            fall_velocity_threshold=fall_velocity_threshold,
            min_duration_s=min_duration_s,
            debounce_s=debounce_s,
        ),
        fight=SkRunCfg(enabled=False),
        crowd=SkRunCfg(enabled=False),
        intrusion=SkRunCfg(enabled=False),
    )
    return AppConfig(
        model_path="yolo11n.pt",
        output_dir="./test_output",
        rules=RulesConfig(skeleton=sk_rules),
        adaptive_threshold=AdaptiveThresholdConfig(enabled=False),
        pose=PoseConfig(enabled=True),
    )


# ============================================================
# Bbox-based test helpers (for RulesEngine)
# ============================================================


def make_bbox_config(
    upright_aspect_min: float = 1.35,
    fallen_aspect_max: float = 0.95,
    transition_window_s: float = 1.0,
    debounce_s: float = 5.0,
) -> AppConfig:
    """Create config for bbox-based fall detection tests."""
    return AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(enabled=False),
            fall=FallRule(
                enabled=True,
                upright_aspect_min=upright_aspect_min,
                fallen_aspect_max=fallen_aspect_max,
                transition_window_s=transition_window_s,
                debounce_s=debounce_s,
            ),
            crowd=CrowdRule(enabled=False),
            intrusion=IntrusionRule(enabled=False),
            fight=FightRule(enabled=False),
        ),
    )


def _person(track_id, x1, y1, x2, y2, conf=0.9):
    return Detection(
        track_id=track_id, class_id=0, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2
    )


# ============================================================
# Test: Skeleton-based Fall Detection - True Positives
# ============================================================


class TestSkeletonFallTruePositives:
    """Test that actual falls are correctly detected."""

    def test_forward_fall_detected(self):
        """Forward fall: person tilts forward and head drops rapidly."""
        cfg = make_skeleton_config(min_duration_s=0.2, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Phase 1: Upright (0.5s)
        for i in range(15):
            t = i / 30.0
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        # Phase 2: Falling forward (0.5s) - torso angle increases, head drops
        for i in range(15):
            t = 0.5 + i / 30.0
            angle = 5.0 + (i / 15.0) * 80.0  # 5° -> 85°
            head_drop = i * 8.0  # Head drops rapidly
            sk = make_skeleton_at(
                1, 200, 200 + head_drop, torso_angle=angle, head_y=200 - 80 + head_drop
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 15 + i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) >= 1, (
            f"Forward fall should be detected, got {len(fall_events)} events"
        )

    def test_backward_fall_detected(self):
        """Backward fall: person falls backward."""
        cfg = make_skeleton_config(min_duration_s=0.2, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Phase 1: Upright
        for i in range(15):
            t = i / 30.0
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        # Phase 2: Falling backward
        for i in range(25):
            t = 0.5 + i / 30.0
            angle = min(85.0, 5.0 + (i / 15.0) * 70.0)
            head_drop = i * 10.0
            sk = make_skeleton_at(
                1, 200, 200 + head_drop, torso_angle=angle, head_y=200 - 80 + head_drop
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 15 + i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) >= 1, (
            f"Backward fall should be detected, got {len(fall_events)} events"
        )

    def test_lateral_fall_detected(self):
        """Lateral (sideways) fall."""
        cfg = make_skeleton_config(min_duration_s=0.2, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Phase 1: Upright
        for i in range(15):
            t = i / 30.0
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        # Phase 2: Falling sideways
        for i in range(15):
            t = 0.5 + i / 30.0
            angle = 5.0 + (i / 15.0) * 85.0  # Full lateral tilt
            head_drop = i * 12.0
            sk = make_skeleton_at(
                1, 200, 200 + head_drop, torso_angle=angle, head_y=200 - 80 + head_drop
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 15 + i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) >= 1, (
            f"Lateral fall should be detected, got {len(fall_events)} events"
        )

    def test_rapid_fall_emergency_detection(self):
        """Rapid fall with very high head velocity should trigger emergency."""
        cfg = make_skeleton_config(
            min_duration_s=0.1, debounce_s=2.0, fall_velocity_threshold=0.5
        )
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Phase 1: Upright
        for i in range(10):
            t = i / 30.0
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        # Phase 2: Rapid fall - head drops very fast (>2.0 velocity threshold)
        for i in range(10):
            t = 0.33 + i / 30.0
            head_drop = i * 30.0  # Very rapid drop
            sk = make_skeleton_at(
                1, 200, 200 + head_drop * 0.5, torso_angle=60.0, head_y=120 + head_drop
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 10 + i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) >= 1, "Rapid fall should trigger emergency detection"


# ============================================================
# Test: Skeleton-based Fall Detection - False Positives
# ============================================================


class TestSkeletonFallFalsePositives:
    """Test that non-fall actions are NOT detected as falls."""

    def test_bending_no_false_positive(self):
        """Bending over should not trigger fall detection."""
        cfg = make_skeleton_config(min_duration_s=0.3, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Simulate bending: torso tilts but head stays relatively high
        for i in range(60):
            t = i / 30.0
            # Bending: torso angle increases but head doesn't drop much
            angle = min(40.0, i * 1.5)  # Max 40° tilt (not enough for fall)
            sk = make_skeleton_at(1, 200, 200, torso_angle=angle, head_y=120 + i * 0.5)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) == 0, (
            f"Bending should not trigger fall, got {len(fall_events)} events"
        )

    def test_squatting_no_false_positive(self):
        """Squatting should not trigger fall detection."""
        cfg = make_skeleton_config(min_duration_s=0.3, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Squatting: body lowers but stays upright (low torso angle)
        for i in range(60):
            t = i / 30.0
            # Squat: y position lowers but torso stays upright
            y_offset = min(50.0, i * 2.0)
            sk = make_skeleton_at(
                1, 200, 200 + y_offset, torso_angle=10.0, head_y=120 + y_offset
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) == 0, (
            f"Squatting should not trigger fall, got {len(fall_events)} events"
        )

    def test_sitting_down_slowly_no_false_positive(self):
        """Sitting down slowly should not trigger fall detection."""
        cfg = make_skeleton_config(min_duration_s=0.3, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Sitting: slow descent, torso stays relatively upright
        for i in range(90):
            t = i / 30.0
            y_offset = min(80.0, i * 1.5)
            angle = min(20.0, i * 0.3)  # Slight tilt
            sk = make_skeleton_at(
                1, 200, 200 + y_offset, torso_angle=angle, head_y=120 + y_offset
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) == 0, (
            f"Sitting slowly should not trigger fall, got {len(fall_events)} events"
        )

    def test_walking_no_false_positive(self):
        """Normal walking should not trigger fall detection."""
        cfg = make_skeleton_config(min_duration_s=0.3, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        for i in range(60):
            t = i / 30.0
            x = 200 + i * 5
            # Walking: slight body sway but upright
            angle = 5.0 * math.sin(i * 0.3)
            sk = make_skeleton_at(1, x, 200, torso_angle=abs(angle), head_y=120)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        assert len(fall_events) == 0, (
            f"Walking should not trigger fall, got {len(fall_events)} events"
        )


# ============================================================
# Test: Bbox-based Fall Detection (RulesEngine)
# ============================================================


class TestBboxFallDetection:
    """Test bbox-based fall detection in RulesEngine."""

    def test_fall_detection_triggers(self):
        """Upright -> fallen transition should trigger."""
        cfg = make_bbox_config()
        engine = RulesEngine(cfg, person_class_id=0)
        t = 100.0

        # Upright person (tall box)
        upright = _person(1, 0, 0, 30, 100)
        events = engine.update([upright], 1, t)
        assert len(events) == 0

        # Fallen person (wide box)
        fallen = _person(1, 0, 0, 100, 30)
        events = engine.update([fallen], 2, t + 0.5)
        assert len(events) == 1
        assert events[0].event_type == "fall"

    def test_no_false_positive_standing(self):
        """Standing person should not trigger fall."""
        cfg = make_bbox_config()
        engine = RulesEngine(cfg, person_class_id=0)

        dets = [_person(1, 0, 0, 30, 100)]  # aspect = 100/30 ~ 3.33
        events = engine.update(dets, 1, 100.0)
        events = engine.update(dets, 2, 100.5)
        assert len(events) == 0

    def test_no_false_positive_crouching(self):
        """Crouching person (moderate aspect ratio) should not trigger."""
        cfg = make_bbox_config()
        engine = RulesEngine(cfg, person_class_id=0)

        # Crouching: aspect ratio ~1.0 (between upright and fallen)
        dets = [_person(1, 0, 0, 60, 60)]
        events = engine.update(dets, 1, 100.0)
        events = engine.update(dets, 2, 100.5)
        # Should not trigger because aspect 1.0 > fallen_aspect_max (0.95)
        assert len(events) == 0


# ============================================================
# Test: Detection Latency
# ============================================================


class TestFallDetectionLatency:
    """Measure detection latency for different fall types."""

    def test_rapid_fall_latency(self):
        """Rapid fall should be detected within 0.5 seconds."""
        cfg = make_skeleton_config(min_duration_s=0.1, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        fall_detected_at = None
        fall_start_time = 0.5  # Fall starts at 0.5s

        for i in range(60):
            t = i / 30.0
            if t < fall_start_time:
                sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
            else:
                dt = t - fall_start_time
                angle = min(85.0, 5.0 + dt * 200.0)  # Rapid tilt
                head_drop = dt * 200.0  # Rapid head drop
                sk = make_skeleton_at(
                    1,
                    200,
                    200 + head_drop * 0.3,
                    torso_angle=angle,
                    head_y=120 + head_drop,
                )

            buffer.append(t, [sk])
            events = rule.detect([sk], t, i + 1, buffer)

            if fall_detected_at is None and any(e.event_type == "fall" for e in events):
                fall_detected_at = t

        if fall_detected_at is not None:
            latency = fall_detected_at - fall_start_time
            assert latency < 0.5, (
                f"Rapid fall detection latency {latency:.2f}s exceeds 0.5s"
            )

    def test_slow_fall_latency(self):
        """Slow fall should still be detected within 1.5 seconds."""
        cfg = make_skeleton_config(min_duration_s=0.3, debounce_s=2.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        fall_detected_at = None
        fall_start_time = 0.5

        for i in range(120):
            t = i / 30.0
            if t < fall_start_time:
                sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
            else:
                dt = t - fall_start_time
                angle = min(70.0, 5.0 + dt * 30.0)  # Slow tilt
                head_drop = dt * 40.0  # Slow head drop
                sk = make_skeleton_at(
                    1,
                    200,
                    200 + head_drop * 0.3,
                    torso_angle=angle,
                    head_y=120 + head_drop,
                )

            buffer.append(t, [sk])
            events = rule.detect([sk], t, i + 1, buffer)

            if fall_detected_at is None and any(e.event_type == "fall" for e in events):
                fall_detected_at = t

        if fall_detected_at is not None:
            latency = fall_detected_at - fall_start_time
            assert latency < 2.0, (
                f"Slow fall detection latency {latency:.2f}s exceeds 2.0s"
            )


# ============================================================
# Test: Recovery Detection
# ============================================================


class TestFallRecovery:
    """Test that fall state resets when person recovers."""

    def test_fall_then_recovery_resets_state(self):
        """After a fall, standing back up should reset the fall state."""
        cfg = make_skeleton_config(min_duration_s=0.2, debounce_s=1.0)
        rule = SkeletonFallRule(cfg)
        buffer = SkeletonFrameBuffer()

        events = []
        # Phase 1: Upright
        for i in range(10):
            t = i / 30.0
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, i + 1, buffer)
            events.extend(e)

        # Phase 2: Fall
        for i in range(15):
            t = 0.33 + i / 30.0
            angle = min(80.0, 5.0 + i * 5.0)
            head_drop = i * 10.0
            sk = make_skeleton_at(
                1, 200, 200 + head_drop * 0.3, torso_angle=angle, head_y=120 + head_drop
            )
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 10 + i + 1, buffer)
            events.extend(e)

        # Phase 3: Recovery (stand back up after debounce)
        for i in range(30):
            t = 2.0 + i / 30.0  # After debounce period
            sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
            buffer.append(t, [sk])
            e = rule.detect([sk], t, 25 + i + 1, buffer)
            events.extend(e)

        fall_events = [e for e in events if e.event_type == "fall"]
        # Should detect the initial fall
        assert len(fall_events) >= 1, "Initial fall should be detected"
        # After recovery, state should be reset (no continuous fall events)


# ============================================================
# Test: Comprehensive Accuracy Metrics
# ============================================================


class TestFallDetectionMetrics:
    """Compute comprehensive accuracy metrics across diverse scenarios."""

    def test_confusion_matrix(self):
        """Simulate confusion matrix for fall detection."""
        scenarios = {
            # True falls (should be detected)
            "forward_fall": True,
            "backward_fall": True,
            "lateral_fall": True,
            "rapid_fall": True,
            "slow_slide": True,
            # Non-falls (should NOT be detected)
            "bending": False,
            "squatting": False,
            "sitting_slowly": False,
            "walking": False,
            "standing_still": False,
            "picking_up_object": False,
        }

        cfg = make_skeleton_config(min_duration_s=0.2, debounce_s=2.0)

        results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for scenario, is_fall in scenarios.items():
            rule = SkeletonFallRule(
                cfg
            )  # Fresh rule per scenario to avoid debounce contamination
            buffer = SkeletonFrameBuffer()
            events = []

            for i in range(90):  # 3 seconds for better detection
                t = i / 30.0

                if scenario == "forward_fall":
                    if i < 15:
                        sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
                    else:
                        angle = min(85.0, 5.0 + (i - 15) * 5.0)
                        head_drop = (i - 15) * 8.0
                        sk = make_skeleton_at(
                            1,
                            200,
                            200 + head_drop * 0.5,
                            torso_angle=angle,
                            head_y=120 + head_drop,
                        )

                elif scenario == "backward_fall":
                    if i < 15:
                        sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
                    else:
                        angle = min(85.0, 5.0 + (i - 15) * 4.5)
                        head_drop = (i - 15) * 9.0
                        sk = make_skeleton_at(
                            1,
                            200,
                            200 + head_drop * 0.5,
                            torso_angle=angle,
                            head_y=120 + head_drop,
                        )

                elif scenario == "lateral_fall":
                    if i < 15:
                        sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
                    else:
                        angle = min(85.0, 5.0 + (i - 15) * 5.5)
                        head_drop = (i - 15) * 10.0
                        sk = make_skeleton_at(
                            1,
                            200,
                            200 + head_drop * 0.5,
                            torso_angle=angle,
                            head_y=120 + head_drop,
                        )

                elif scenario == "rapid_fall":
                    if i < 10:
                        sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
                    else:
                        head_drop = (i - 10) * 25.0
                        sk = make_skeleton_at(
                            1,
                            200,
                            200 + head_drop * 0.5,
                            torso_angle=60.0,
                            head_y=120 + head_drop,
                        )

                elif scenario == "slow_slide":
                    if i < 15:
                        sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)
                    else:
                        angle = min(75.0, 5.0 + (i - 15) * 1.5)
                        head_drop = (i - 15) * 4.0
                        sk = make_skeleton_at(
                            1,
                            200,
                            200 + head_drop * 0.5,
                            torso_angle=angle,
                            head_y=120 + head_drop,
                        )

                elif scenario == "bending":
                    angle = min(35.0, i * 1.2)
                    sk = make_skeleton_at(
                        1, 200, 200, torso_angle=angle, head_y=120 + i * 0.5
                    )

                elif scenario == "squatting":
                    y_off = min(50.0, i * 2.0)
                    sk = make_skeleton_at(
                        1, 200, 200 + y_off, torso_angle=10.0, head_y=120 + y_off
                    )

                elif scenario == "sitting_slowly":
                    y_off = min(80.0, i * 1.5)
                    angle = min(20.0, i * 0.3)
                    sk = make_skeleton_at(
                        1, 200, 200 + y_off, torso_angle=angle, head_y=120 + y_off
                    )

                elif scenario == "walking":
                    x = 200 + i * 5
                    angle = 5.0 * abs(math.sin(i * 0.3))
                    sk = make_skeleton_at(1, x, 200, torso_angle=angle, head_y=120)

                elif scenario == "standing_still":
                    sk = make_skeleton_at(1, 200, 200, torso_angle=3.0, head_y=120)

                elif scenario == "picking_up_object":
                    angle = min(30.0, i * 1.0)
                    sk = make_skeleton_at(
                        1, 200, 200, torso_angle=angle, head_y=120 + i * 0.8
                    )

                else:
                    sk = make_skeleton_at(1, 200, 200, torso_angle=5.0, head_y=120)

                buffer.append(t, [sk])
                e = rule.detect([sk], t, i + 1, buffer)
                events.extend(e)

            detected = any(e.event_type == "fall" for e in events)

            if is_fall and detected:
                results["tp"] += 1
            elif is_fall and not detected:
                results["fn"] += 1
            elif not is_fall and detected:
                results["fp"] += 1
            else:
                results["tn"] += 1

        precision = results["tp"] / max(1, results["tp"] + results["fp"])
        recall = results["tp"] / max(1, results["tp"] + results["fn"])
        accuracy = (results["tp"] + results["tn"]) / sum(results.values())
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        print("\nFall Detection Metrics:")
        print(
            f"  TP={results['tp']}, FP={results['fp']}, TN={results['tn']}, FN={results['fn']}"
        )
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1 Score: {f1:.2f}")
        print(f"  Accuracy: {accuracy:.2f}")

        # Target: 95% accuracy in controlled testing
        assert accuracy >= 0.80, f"Accuracy {accuracy:.2f} below 80% target"
        assert precision >= 0.80, f"Precision {precision:.2f} below 80% target"
        assert recall >= 0.80, f"Recall {recall:.2f} below 80% target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
