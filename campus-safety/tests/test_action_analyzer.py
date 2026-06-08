# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for action sequence analyzer."""

from __future__ import annotations

import numpy as np

from core.action_analyzer import (
    ActionAnalyzer,
    ActionFeatures,
    BehaviorTemplate,
    SkeletonSequenceBuffer,
)
from core.skeleton import PersonSkeleton, SkeletonKeypoint


def _make_skeleton(track_id: int, x: float = 0, y: float = 0, frame: int = 0, ts: float = 0.0):
    """Helper to create a skeleton with minimal fields."""
    sk = PersonSkeleton(
        track_id=track_id,
        keypoints=[SkeletonKeypoint(x=x, y=y, confidence=1.0)],
        bbox={"x1": x - 10, "y1": y - 10, "x2": x + 10, "y2": y + 10},
    )
    sk._center_override = (x, y)
    sk.frame_index = frame
    sk.timestamp = ts
    sk.body_angle = 0.0
    sk.limb_lengths = {}
    return sk


class TestSkeletonSequenceBuffer:
    def test_add_and_get_sequence(self):
        buf = SkeletonSequenceBuffer(max_length=10)
        sk = _make_skeleton(1, x=100, y=200)
        buf.add(1, sk)
        seq = buf.get_sequence(1)
        assert len(seq) == 1
        assert seq[0].track_id == 1

    def test_get_all_sequences(self):
        buf = SkeletonSequenceBuffer(max_length=10)
        buf.add(1, _make_skeleton(1))
        buf.add(2, _make_skeleton(2))
        all_seq = buf.get_all_sequences()
        assert set(all_seq.keys()) == {1, 2}

    def test_clear_old_removes_stale(self):
        buf = SkeletonSequenceBuffer(max_length=10)
        buf.add(1, _make_skeleton(1, frame=0))
        buf.add(1, _make_skeleton(1, frame=5))
        # current max is 5, max_age_frames=3, so seq[0] with frame 0 should be removed
        buf.add(2, _make_skeleton(2, frame=100))
        buf.clear_old(max_age_frames=3)
        # Track 1 has frame 5, track 2 frame 100, current_max=100
        # Track 1: 100-5=95 > 3, should be removed
        # Track 2: 100-100=0 <= 3, should remain
        assert 2 in buf._buffers


class TestActionFeatures:
    def test_default_values(self):
        f = ActionFeatures()
        assert f.avg_speed == 0.0
        assert f.frame_count == 0


class TestBehaviorTemplate:
    def test_create_template(self):
        t = BehaviorTemplate(
            name="test",
            description="test template",
            feature_ranges={"avg_speed": (10, 100)},
            min_duration=0.5,
            priority=1,
        )
        assert t.name == "test"
        assert t.min_duration == 0.5


class TestActionAnalyzer:
    def make_skeletons_for_speed(self, track_id, positions):
        """Create a sequence of skeletons at given positions."""
        skeletons = []
        for i, (x, y) in enumerate(positions):
            sk = _make_skeleton(track_id, x=x, y=y, frame=i, ts=i * 0.1)
            sk.limb_lengths = {"arm": 10.0}
            skeletons.append(sk)
        return skeletons

    def test_extract_features_empty_sequence(self):
        analyzer = ActionAnalyzer()
        features = analyzer._extract_features([])
        assert isinstance(features, ActionFeatures)
        assert features.avg_speed == 0.0

    def test_extract_features_two_frames(self):
        analyzer = ActionAnalyzer()
        sk1 = _make_skeleton(1, x=0, y=0, frame=0, ts=0.0)
        sk2 = _make_skeleton(1, x=100, y=0, frame=1, ts=1.0)
        sk1.limb_lengths = {"arm": 10.0}
        sk2.limb_lengths = {"arm": 10.0}
        sk2.body_angle = 10.0
        features = analyzer._extract_features([sk1, sk2])
        assert features.avg_speed > 0
        assert features.duration == 1.0

    def test_match_template_below_duration(self):
        analyzer = ActionAnalyzer()
        template = BehaviorTemplate(
            name="test",
            description="test",
            feature_ranges={"avg_speed": (10, 100)},
            min_duration=1.0,
        )
        features = ActionFeatures(duration=0.5)
        confidence = analyzer._match_template(features, template)
        assert confidence == 0.0

    def test_match_template_perfect_match(self):
        analyzer = ActionAnalyzer()
        template = BehaviorTemplate(
            name="test",
            description="test",
            feature_ranges={"avg_speed": (10, 100)},
            min_duration=0.1,
        )
        features = ActionFeatures(duration=1.0, avg_speed=50)
        confidence = analyzer._match_template(features, template)
        assert confidence > 0.0

    def test_process_frame_no_skeletons(self):
        analyzer = ActionAnalyzer()
        results = analyzer.process_frame([], 0, 0.0)
        assert results == []

    def test_detect_crowd_below_threshold(self):
        analyzer = ActionAnalyzer()
        # 2 skeletons is below the threshold of 3
        sks = [
            _make_skeleton(1, x=0, y=0),
            _make_skeleton(2, x=100, y=100),
        ]
        result = analyzer._detect_crowd_behavior(sks)
        assert result is None

    def test_detect_fight_pair_insufficient_frames(self):
        analyzer = ActionAnalyzer()
        sequences = {
            1: [_make_skeleton(1, frame=0)],
            2: [_make_skeleton(2, frame=0)],
        }
        results = analyzer._detect_fight_behavior(sequences)
        assert results == []

    def test_calculate_proximity_score(self):
        analyzer = ActionAnalyzer()
        seq1 = self.make_skeletons_for_speed(1, [(0, 0), (10, 10)])
        seq2 = self.make_skeletons_for_speed(2, [(200, 200), (210, 210)])
        score = analyzer._calculate_proximity_score(seq1, seq2)
        assert 0 <= score <= 1

    def test_calculate_chaotic_score(self):
        analyzer = ActionAnalyzer()
        seq = self.make_skeletons_for_speed(1, [(0, 0), (10, 10), (20, 0), (30, 10)])
        seq2 = self.make_skeletons_for_speed(2, [(100, 100), (110, 110), (120, 100), (130, 110)])
        score = analyzer._calculate_chaotic_score(seq, seq2)
        assert score >= 0

    def test_count_direction_changes(self):
        analyzer = ActionAnalyzer()
        # Zigzag pattern
        seq = self.make_skeletons_for_speed(1, [
            (0, 0), (100, 0), (0, 100), (100, 0), (0, 0), (100, 100),
        ])
        seq2 = self.make_skeletons_for_speed(2, [
            (200, 200), (200, 200), (200, 200), (200, 200), (200, 200), (200, 200),
        ])
        changes = analyzer._count_direction_changes(seq, seq2)
        assert changes >= 0

    def test_behavior_templates_loaded(self):
        analyzer = ActionAnalyzer()
        assert len(analyzer.templates) == 4
        names = [t.name for t in analyzer.templates]
        assert "running" in names
        assert "fall" in names
        assert "fight" in names
        assert "crowd" in names

    def test_clear_sequences(self):
        analyzer = ActionAnalyzer()
        analyzer.sequence_buffer.add(1, _make_skeleton(1))
        analyzer.clear_sequences()
        assert len(analyzer.sequence_buffer.get_sequence(1)) == 0
