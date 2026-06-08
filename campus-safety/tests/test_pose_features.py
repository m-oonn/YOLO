# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pose feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

COURSE_DIR = str(Path(__file__).resolve().parent.parent)
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

from core.constants import NUM_SKELETON_KEYPOINTS
from core.skeleton import Skeleton, SkeletonKeypoint as Keypoint
from core.pose_features import (
    PerFrameFeatures,
    PerFrameFeatureExtractor,
    TemporalFeatures,
    TemporalFeatureExtractor,
    InteractionFeatures,
    InteractionFeatureExtractor,
    normalize_features,
)


class TestPerFrameFeatures:
    def test_to_vector_shape(self):
        feat = PerFrameFeatures(
            track_id=1, torso_angle=45.0, center_x=0.5, center_y=0.5,
            head_height=0.2, body_span=0.3, avg_kp_confidence=0.8,
            valid_kp_ratio=0.9, skeleton_quality=0.85,
        )
        vec = feat.to_vector()
        # 8 base + 6 bone angles = 14
        assert vec.shape == (14,)
        # Torso angle normalized
        assert vec[0] == pytest.approx(45.0 / 180.0)

    def test_default_values(self):
        feat = PerFrameFeatures(track_id=1)
        vec = feat.to_vector()
        assert np.all(vec >= 0) and np.all(vec <= 1.1)


class TestPerFrameFeatureExtractor:
    def _make_valid_skel(self) -> Skeleton:
        kps = [Keypoint(100, 200, 0.9) for _ in range(NUM_SKELETON_KEYPOINTS)]
        kps[5] = Keypoint(120, 150, 0.9)   # left_shoulder
        kps[6] = Keypoint(180, 150, 0.9)   # right_shoulder
        kps[11] = Keypoint(140, 300, 0.9)  # left_hip
        kps[12] = Keypoint(160, 300, 0.9)  # right_hip
        sk = Skeleton(track_id=1, keypoints=kps, bbox={"x1": 100, "y1": 150, "x2": 200, "y2": 350})
        sk.average_confidence = 0.9
        sk.valid_count = 17
        return sk

    def test_extract_valid(self):
        extractor = PerFrameFeatureExtractor()
        skel = self._make_valid_skel()
        feat = extractor.extract(skel, 640, 480)
        assert feat.track_id == 1
        assert feat.skeleton_quality > 0
        assert feat.center_x > 0
        assert feat.center_y > 0

    def test_extract_low_quality(self):
        extractor = PerFrameFeatureExtractor()
        kps = [Keypoint(0, 0, 0.0) for _ in range(5)]
        skel = Skeleton(track_id=1, keypoints=kps, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        skel.average_confidence = 0.1
        skel.valid_count = 0
        feat = extractor.extract(skel, 640, 480)
        assert feat.skeleton_quality == 0.0

    def test_extract_empty(self):
        extractor = PerFrameFeatureExtractor()
        skel = Skeleton(track_id=1, keypoints=[], bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        skel.average_confidence = 0.0
        skel.valid_count = 0
        feat = extractor.extract(skel, 640, 480)
        assert feat.skeleton_quality == 0.0


class TestTemporalFeatures:
    def test_to_vector_shape(self):
        feat = TemporalFeatures()
        vec = feat.to_vector()
        # 34 + 34 + 1 + 1 + 1 = 71
        assert vec.shape == (71,)

    def test_is_empty(self):
        feat = TemporalFeatures()
        assert feat.is_empty() == True

    def test_not_empty(self):
        feat = TemporalFeatures(velocities=np.ones(NUM_SKELETON_KEYPOINTS * 2))
        assert feat.is_empty() == False


class TestTemporalFeatureExtractor:
    def test_insufficient_history(self):
        extractor = TemporalFeatureExtractor(window_size=30)
        feat = PerFrameFeatures(track_id=1, center_x=0.5, center_y=0.5)
        result = extractor.extract(1, 0.0, feat)
        assert result.is_empty() is True

    def test_two_frames(self):
        extractor = TemporalFeatureExtractor(window_size=30)
        f1 = PerFrameFeatures(track_id=1, center_x=0.3, center_y=0.3)
        f2 = PerFrameFeatures(track_id=1, center_x=0.5, center_y=0.5)
        extractor.extract(1, 0.0, f1)
        result = extractor.extract(1, 1.0, f2)
        assert result.is_empty() is False
        assert result.motion_energy > 0

    def test_cleanup(self):
        extractor = TemporalFeatureExtractor(window_size=30)
        extractor._history[1] = []
        extractor._history[2] = []
        extractor.cleanup({1})
        assert 1 in extractor._history
        assert 2 not in extractor._history

    def test_reset_track(self):
        extractor = TemporalFeatureExtractor(window_size=30)
        extractor._history[1] = []
        extractor.reset_track(1)
        assert 1 not in extractor._history


class TestInteractionFeatures:
    def test_to_vector_shape(self):
        feat = InteractionFeatures()
        vec = feat.to_vector()
        assert vec.shape == (5,)

    def test_contact_flag(self):
        feat = InteractionFeatures(contact_detected=True)
        vec = feat.to_vector()
        assert vec[3] == 1.0


class TestInteractionFeatureExtractor:
    def _make_skel(self, track_id, kps):
        sk = Skeleton(track_id=track_id, keypoints=kps, bbox={"x1": 0, "y1": 0, "x2": 200, "y2": 200})
        sk.average_confidence = 0.9
        sk.valid_count = 17
        return sk

    def test_extract_distant(self):
        kps_a = [Keypoint(0, 0, 0.9) for _ in range(NUM_SKELETON_KEYPOINTS)]
        kps_b = [Keypoint(500, 500, 0.9) for _ in range(NUM_SKELETON_KEYPOINTS)]
        skel_a = self._make_skel(1, kps_a)
        skel_b = self._make_skel(2, kps_b)
        feat = InteractionFeatureExtractor.extract(skel_a, skel_b, 640, 480)
        assert feat.center_distance > 0
        assert feat.contact_detected == False

    def test_extract_close(self):
        kps_a = [Keypoint(100, 100, 0.9) for _ in range(NUM_SKELETON_KEYPOINTS)]
        kps_b = [Keypoint(105, 105, 0.9) for _ in range(NUM_SKELETON_KEYPOINTS)]
        skel_a = self._make_skel(1, kps_a)
        skel_b = self._make_skel(2, kps_b)
        feat = InteractionFeatureExtractor.extract(skel_a, skel_b, 640, 480)
        assert feat.center_distance < 10
        assert feat.contact_detected == True


class TestNormalizeFeatures:
    def test_clamps_values(self):
        feat = PerFrameFeatures(
            track_id=1, torso_angle=200.0, center_x=1.5, center_y=-0.5,
        )
        normalized = normalize_features(feat)
        assert normalized.torso_angle == 180.0
        assert normalized.center_x == 1.0
        assert normalized.center_y == 0.0

    def test_preserves_valid_values(self):
        feat = PerFrameFeatures(
            track_id=1, torso_angle=45.0, center_x=0.5, center_y=0.5,
        )
        normalized = normalize_features(feat)
        assert normalized.torso_angle == 45.0
        assert normalized.center_x == 0.5
