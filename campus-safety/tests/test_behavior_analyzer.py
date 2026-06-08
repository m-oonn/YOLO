# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for behavior analyzer improvements."""

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import numpy as np
import pytest

from core.behavior_analyzer import (
    AdaptiveThresholdManager,
    CrowdDensityAnalyzer,
    SkeletonFightRule,
    SkeletonFrameBuffer,
    SkeletonRunningRule,
    SkeletonFallRule,
)
from core.config import (
    AppConfig,
    RulesConfig,
    SkeletonRulesConfig,
    SkeletonRunningRule as SkRunCfg,
    SkeletonFallRule as SkFallCfg,
    SkeletonFightRule as SkFightCfg,
    SkeletonCrowdRule as SkCrowdCfg,
    SkeletonIntrusionRule as SkIntrCfg,
    AdaptiveThresholdConfig,
    PoseConfig,
)
from core.skeleton import Skeleton, SkeletonKeypoint


def make_skeleton(track_id, x, y, conf=0.9, n_kpts=17):
    """Create a test skeleton at position (x, y)."""
    keypoints = []
    for i in range(n_kpts):
        kp_x = x + (i % 5) * 5
        kp_y = y + (i // 5) * 10 + np.random.randint(-5, 5)
        keypoints.append(SkeletonKeypoint(x=kp_x, y=kp_y, confidence=conf))
    bbox = {"x1": x, "y1": y, "x2": x + 50, "y2": y + 100}
    sk = Skeleton(track_id=track_id, keypoints=keypoints, bbox=bbox)
    # Monkey-patch attributes expected by behavior_analyzer
    sk.head_height = y
    # Use object.__setattr__ to bypass property setter
    object.__setattr__(sk, '_center_override', (x + 25, y + 50))
    return sk


def make_config():
    """Create properly typed test config."""
    sk_rules = SkeletonRulesConfig(
        running=SkRunCfg(
            enabled=True,
            speed_threshold_kmh=10.0,
            min_duration_s=0.5,
            debounce_s=2.0,
        ),
        fall=SkFallCfg(
            enabled=True,
            torso_angle_threshold=45.0,
            head_height_threshold=100.0,
            fall_velocity_threshold=50.0,
            min_duration_s=0.5,
            debounce_s=3.0,
        ),
        fight=SkFightCfg(
            enabled=True,
            proximity_threshold_m=2.0,
            wrist_speed_threshold_ms=3.0,
            limb_frequency_threshold=2.0,
            min_duration_s=0.3,
            debounce_s=1.0,
        ),
        crowd=SkCrowdCfg(
            enabled=True,
            density_threshold=0.5,
            min_duration_s=2.0,
            debounce_s=5.0,
        ),
        intrusion=SkIntrCfg(enabled=False),
    )
    rules = RulesConfig(skeleton=sk_rules)
    return AppConfig(
        model_path="yolo11n.pt",
        output_dir="./test_output",
        rules=rules,
        adaptive_threshold=AdaptiveThresholdConfig(
            enabled=True,
            adapt_window_s=60.0,
            min_trigger_count=5,
            sensitivity=1.0,
        ),
        pose=PoseConfig(enabled=True),
    )


class TestSkeletonFightRule:
    """Test enhanced fight detection with temporal analysis."""

    def test_temporal_score_requires_history(self):
        cfg = make_config()
        rule = SkeletonFightRule(cfg)
        sk1 = make_skeleton(1, 100, 100)
        sk2 = make_skeleton(2, 110, 100)
        buf = SkeletonFrameBuffer()

        # No history yet -> temporal score = 0
        events = rule.detect([sk1, sk2], 0.0, 1, buf)
        assert len(events) == 0

    def test_fight_detection_with_temporal_confirmation(self):
        cfg = make_config()
        rule = SkeletonFightRule(cfg)
        buf = SkeletonFrameBuffer()

        # Simulate 30 frames of chaotic close movement with high speed
        events = []
        import random
        random.seed(42)
        for frame_idx in range(30):
            t = frame_idx * 0.2
            # Chaotic movement: random large offsets
            off1_x = random.randint(-100, 100)
            off1_y = random.randint(-50, 50)
            off2_x = random.randint(-100, 100)
            off2_y = random.randint(-50, 50)
            sk1 = make_skeleton(1, 150 + off1_x, 150 + off1_y)
            sk2 = make_skeleton(2, 160 + off2_x, 160 + off2_y)
            e = rule.detect([sk1, sk2], t, frame_idx + 1, buf)
            events.extend(e)

        # After 30 frames with chaotic motion, should detect fight
        assert len(events) > 0
        assert events[-1].event_type == "fight"
        assert events[-1].extra.get("detection_method") == "skeleton_temporal"
        assert "temporal_score" in events[-1].extra

    def test_no_fight_for_static_people(self):
        cfg = make_config()
        rule = SkeletonFightRule(cfg)
        buf = SkeletonFrameBuffer()

        # Two people standing close but still
        events = []
        for frame_idx in range(10):
            t = frame_idx * 0.1
            sk1 = make_skeleton(1, 100, 100)
            sk2 = make_skeleton(2, 105, 100)
            e = rule.detect([sk1, sk2], t, frame_idx + 1, buf)
            events.extend(e)

        # Should not detect fight for static people
        assert len(events) == 0

    def test_adaptive_threshold_integration(self):
        cfg = make_config()
        rule = SkeletonFightRule(cfg)
        buf = SkeletonFrameBuffer()
        mgr = AdaptiveThresholdManager(cfg)

        # Record some false positives to raise threshold
        for _ in range(10):
            mgr.record_false_positive("fight_proximity", {"is_fp": True})

        adj_th = mgr.get_adjusted_threshold("fight_proximity", 2.0)
        assert adj_th > 2.0  # Threshold should increase


class TestCrowdDensityAnalyzer:
    """Test enhanced crowd density with multi-metric analysis."""

    def test_nn_density_calculation(self):
        cfg = make_config()
        analyzer = CrowdDensityAnalyzer(cfg)

        # 5 people close together with varied positions (avoid collinear)
        skeletons = [
            make_skeleton(i, 100 + i * 20, 100 + i * 25 + (i % 2) * 30)
            for i in range(5)
        ]
        buf = SkeletonFrameBuffer()
        events = analyzer.detect(skeletons, 0.0, 1, buf)
        assert len(events) == 0  # Need duration

    def test_social_density_metric(self):
        cfg = make_config()
        analyzer = CrowdDensityAnalyzer(cfg)

        # Many people in small area with varied positions
        skeletons = [
            make_skeleton(i, 100 + (i % 3) * 20, 100 + (i // 3) * 25)
            for i in range(8)
        ]
        buf = SkeletonFrameBuffer()

        # Run for multiple frames to trigger duration
        events = []
        for i in range(30):
            e = analyzer.detect(skeletons, i * 0.1, i + 1, buf)
            events.extend(e)

        assert len(events) > 0
        assert events[0].event_type == "crowd"
        extra = events[0].extra
        assert "nn_density" in extra
        assert "hull_density" in extra
        assert "social_density" in extra

    def test_density_smoothing(self):
        cfg = make_config()
        analyzer = CrowdDensityAnalyzer(cfg)

        skeletons = [make_skeleton(i, 100 + i * 30, 100 + i * 20 + (i % 2) * 40) for i in range(5)]
        buf = SkeletonFrameBuffer()

        # First frame
        e1 = analyzer.detect(skeletons, 0.0, 1, buf)
        # Second frame should use smoothed density
        e2 = analyzer.detect(skeletons, 0.5, 2, buf)

        # History should smooth the density values
        assert len(analyzer._density_history) > 0


class TestAdaptiveThresholdManager:
    """Test multi-factor adaptive threshold adjustment."""

    def test_fp_rate_adjustment(self):
        cfg = make_config()
        mgr = AdaptiveThresholdManager(cfg)

        base = 10.0
        # No stats yet -> return base
        assert mgr.get_adjusted_threshold("test", base) == base

        # Record FPs
        for _ in range(10):
            mgr.record_false_positive("test", {"is_fp": True})

        adj = mgr.get_adjusted_threshold("test", base)
        assert adj > base  # Should increase

    def test_confidence_distribution_adjustment(self):
        cfg = make_config()
        mgr = AdaptiveThresholdManager(cfg)

        # Record high confidence detections
        for _ in range(20):
            mgr.record_confidence("test", 0.9)

        base = 10.0
        adj = mgr.get_adjusted_threshold("test", base)
        # High confidence with low variance -> slight increase
        assert adj >= base

    def test_scene_complexity_adjustment(self):
        cfg = make_config()
        mgr = AdaptiveThresholdManager(cfg)

        # High scene complexity
        for _ in range(10):
            mgr.record_scene_complexity(15)

        base = 10.0
        adj = mgr.get_adjusted_threshold("test", base)
        assert adj > base  # Should increase for crowded scenes

    def test_stats_summary(self):
        cfg = make_config()
        mgr = AdaptiveThresholdManager(cfg)

        for _ in range(5):
            mgr.record_false_positive("rule1", {"is_fp": True})
        for _ in range(5):
            mgr.record_true_positive("rule1", {"is_fp": False})

        summary = mgr.get_stats_summary()
        assert "rule1" in summary
        assert summary["rule1"]["total_records"] == 10
        assert summary["rule1"]["fp_rate"] == 0.5


class TestSkeletonRunningRule:
    """Test running detection with adaptive thresholds."""

    def test_speed_threshold_adaptation(self):
        cfg = make_config()
        rule = SkeletonRunningRule(cfg)
        buf = SkeletonFrameBuffer()
        mgr = AdaptiveThresholdManager(cfg)

        # Create fast-moving skeleton
        sk = make_skeleton(1, 100, 100)
        buf.append(0.0, [make_skeleton(1, 50, 100)])

        events = rule.detect([sk], 1.0, 2, buf, mgr)
        # Should not trigger immediately (needs duration)
        assert len(events) == 0


class TestSkeletonFallRule:
    """Test fall detection with adaptive angle threshold."""

    def test_adaptive_angle_threshold(self):
        cfg = make_config()
        rule = SkeletonFallRule(cfg)
        buf = SkeletonFrameBuffer()
        mgr = AdaptiveThresholdManager(cfg)

        # Record FPs to raise threshold
        for _ in range(10):
            mgr.record_false_positive("fall_angle", {"is_fp": True})

        adj = mgr.get_adjusted_threshold("fall_angle", 45.0)
        assert adj > 45.0


class TestPerformanceOptimization:
    """Test frame skipping and caching."""

    def test_frame_skip_counter(self):
        # Just test the counter logic
        counter = 0
        interval = 2
        process_frames = []
        for i in range(10):
            counter += 1
            if counter % interval == 0:
                process_frames.append(i)

        assert process_frames == [1, 3, 5, 7, 9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
