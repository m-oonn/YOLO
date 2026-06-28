# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Feature extraction from skeleton data for behavior analysis."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .constants import NUM_SKELETON_KEYPOINTS
from .skeleton import Skeleton, compute_all_bone_angles, compute_torso_angle

logger = logging.getLogger(__name__)


@dataclass
class PerFrameFeatures:
    """Per-frame features extracted from a single skeleton."""

    track_id: int | None
    torso_angle: float = 0.0  # torso inclination from vertical (degrees)
    bone_angles: dict[str, float] = field(default_factory=dict)
    center_x: float = 0.0  # body center x (normalized)
    center_y: float = 0.0  # body center y (normalized)
    head_height: float = 0.0  # nose y (normalized)
    body_span: float = 0.0  # shoulder_width / height_ratio
    avg_kp_confidence: float = 0.0
    valid_kp_ratio: float = 0.0  # fraction of valid keypoints
    skeleton_quality: float = 0.0  # overall quality score

    def to_vector(self) -> np.ndarray:
        """Convert to fixed-length feature vector (14-dim)."""
        base = np.array(
            [
                self.torso_angle / 180.0,  # normalize to [0, 1]
                self.center_x,
                self.center_y,
                self.head_height,
                self.body_span,
                self.avg_kp_confidence,
                self.valid_kp_ratio,
                self.skeleton_quality,
            ]
        )
        # Add bone angles (up to 6)
        bone_keys = [
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "left_hip_angle",
            "right_hip_angle",
        ]
        bone_vec = np.array([self.bone_angles.get(k, 90.0) / 180.0 for k in bone_keys])
        return np.concatenate([base, bone_vec])  # 14-dim


@dataclass
class TemporalFeatures:
    """Temporal features from a sliding window of skeleton history."""

    velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_SKELETON_KEYPOINTS * 2)
    )
    accelerations: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_SKELETON_KEYPOINTS * 2)
    )
    torso_angle_velocity: float = 0.0
    motion_energy: float = 0.0
    gait_frequency: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (72-dim)."""
        return np.concatenate(
            [
                self.velocities,  # 34-dim
                self.accelerations,  # 34-dim
                [self.torso_angle_velocity],
                [self.motion_energy],
                [self.gait_frequency],
            ]
        )

    def is_empty(self) -> bool:
        return bool(np.all(self.velocities == 0))


@dataclass
class InteractionFeatures:
    """Pairwise interaction features between two skeletons."""

    center_distance: float = 0.0
    min_limb_distance: float = 0.0
    orientation_diff: float = 0.0
    contact_detected: bool = False
    relative_speed: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.center_distance,
                self.min_limb_distance,
                self.orientation_diff / 180.0,
                1.0 if self.contact_detected else 0.0,
                self.relative_speed,
            ]
        )


class PerFrameFeatureExtractor:
    """Extracts per-frame features from skeleton data."""

    def extract(
        self, skel: Skeleton, frame_w: int = 640, frame_h: int = 480
    ) -> PerFrameFeatures:
        """Compute per-frame features for a single skeleton."""
        kps = skel.keypoints
        valid_kps = [kp for kp in kps if kp.is_valid()]
        if not valid_kps or skel.is_low_quality:
            return PerFrameFeatures(track_id=skel.track_id)

        # Normalize positions
        cx, cy = skel.center
        nx = cx / frame_w if frame_w else 0
        ny = cy / frame_h if frame_h else 0
        head_h = skel.head_height / frame_h if frame_h else 0

        # Body span (shoulder width / height)
        span = 0.0
        if len(kps) >= 6:
            ls = kps[5]
            rs = kps[6]
            if ls.is_valid() and rs.is_valid():
                sw = abs(rs.x - ls.x)
                bbox_h = (
                    (skel.bbox["y2"] - skel.bbox["y1"])
                    if skel.bbox["y2"] > skel.bbox["y1"]
                    else 1
                )
                span = sw / bbox_h

        # Bone angles
        bone_angles = compute_all_bone_angles(skel)

        # Quality metrics
        avg_conf = float(np.mean([kp.confidence for kp in kps]))
        valid_ratio = len(valid_kps) / len(kps)
        quality = avg_conf * valid_ratio

        return PerFrameFeatures(
            track_id=skel.track_id,
            torso_angle=compute_torso_angle(skel),
            bone_angles=bone_angles,
            center_x=nx,
            center_y=ny,
            head_height=head_h,
            body_span=span,
            avg_kp_confidence=avg_conf,
            valid_kp_ratio=valid_ratio,
            skeleton_quality=quality,
        )


class TemporalFeatureExtractor:
    """Extracts temporal features from a history of per-frame features."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        # Per-track history: track_id -> deque of (timestamp, PerFrameFeatures)
        self._history: dict[int, deque[tuple[float, PerFrameFeatures]]] = {}

    def extract(
        self, track_id: int, timestamp: float, frame_features: PerFrameFeatures
    ) -> TemporalFeatures:
        """Compute temporal features from recent history."""
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.window_size)
        self._history[track_id].append((timestamp, frame_features))

        hist = self._history[track_id]
        if len(hist) < 2:
            return TemporalFeatures()

        # Compute keypoint velocities and accelerations from skeleton data
        # (simplified: use center position for motion energy)
        curr_ts, curr_feat = hist[-1]
        prev_ts, prev_feat = hist[-2]
        dt = max(1e-6, curr_ts - prev_ts)

        vel_center_x = (curr_feat.center_x - prev_feat.center_x) / dt
        vel_center_y = (curr_feat.center_y - prev_feat.center_y) / dt
        motion_energy = (vel_center_x**2 + vel_center_y**2) ** 0.5

        torso_vel = (
            (curr_feat.torso_angle - prev_feat.torso_angle) / dt if dt > 0 else 0
        )

        # Approximate velocities for all keypoints
        velocities = np.zeros(NUM_SKELETON_KEYPOINTS * 2)
        accelerations = np.zeros(NUM_SKELETON_KEYPOINTS * 2)

        if len(hist) >= 3:
            _, prev2_feat = hist[-3]
            vel_prev_x = (prev_feat.center_x - prev2_feat.center_x) / max(
                1e-6, prev_ts - hist[-3][0]
            )
            vel_prev_y = (prev_feat.center_y - prev2_feat.center_y) / max(
                1e-6, prev_ts - hist[-3][0]
            )
            accel_x = (vel_center_x - vel_prev_x) / dt
            accel_y = (vel_center_y - vel_prev_y) / dt
            accelerations[0] = accel_x
            accelerations[1] = accel_y

        velocities[0] = vel_center_x
        velocities[1] = vel_center_y

        # Gait frequency detection (simplified: head bobbing frequency)
        gait_freq = 0.0
        if len(hist) >= 10:
            head_heights = [feat.head_height for _, feat in hist]
            if len(head_heights) >= 10:
                # Simple zero-crossing frequency estimate
                centered = np.array(head_heights) - np.mean(head_heights)
                zero_crossings = np.sum(np.abs(np.diff(np.signbit(centered))))
                gait_freq = (
                    zero_crossings / (hist[-1][0] - hist[0][0])
                    if (hist[-1][0] - hist[0][0]) > 0
                    else 0
                )

        return TemporalFeatures(
            velocities=velocities,
            accelerations=accelerations,
            torso_angle_velocity=torso_vel,
            motion_energy=motion_energy,
            gait_frequency=gait_freq,
        )

    def cleanup(self, active_tracks: set[int]) -> None:
        """Remove history for tracks no longer active."""
        stale = set(self._history.keys()) - active_tracks
        for tid in stale:
            del self._history[tid]

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)


class InteractionFeatureExtractor:
    """Extracts pairwise interaction features between skeletons."""

    @staticmethod
    def extract(
        skel_a: Skeleton, skel_b: Skeleton, frame_w: int = 640, frame_h: int = 480
    ) -> InteractionFeatures:
        """Compute interaction features between two skeletons."""
        cx_a, cy_a = skel_a.center
        cx_b, cy_b = skel_b.center
        center_dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

        # Minimum limb distance
        min_limb_dist = float("inf")
        limb_indices = [
            (7, 9),  # left arm
            (8, 10),  # right arm
            (5, 7),  # left upper arm
            (6, 8),  # right upper arm
        ]
        for i, j in limb_indices:
            if (
                i < len(skel_a.keypoints)
                and j < len(skel_a.keypoints)
                and i < len(skel_b.keypoints)
                and j < len(skel_b.keypoints)
            ) and (
                skel_a.keypoints[i].is_valid()
                and skel_a.keypoints[j].is_valid()
                and skel_b.keypoints[i].is_valid()
                and skel_b.keypoints[j].is_valid()
            ):
                p1 = np.array([skel_a.keypoints[i].x, skel_a.keypoints[i].y])
                p2 = np.array([skel_a.keypoints[j].x, skel_a.keypoints[j].y])
                p3 = np.array([skel_b.keypoints[i].x, skel_b.keypoints[i].y])
                p4 = np.array([skel_b.keypoints[j].x, skel_b.keypoints[j].y])
                d = np.min(
                    [
                        np.linalg.norm(p1 - p3),
                        np.linalg.norm(p1 - p4),
                        np.linalg.norm(p2 - p3),
                        np.linalg.norm(p2 - p4),
                    ]
                )
                min_limb_dist = min(min_limb_dist, d)

        if min_limb_dist == float("inf"):
            min_limb_dist = center_dist

        # Orientation difference (simplified: torso angles difference)
        orient_a = compute_torso_angle(skel_a)
        orient_b = compute_torso_angle(skel_b)
        orient_diff = abs(orient_a - orient_b)

        # Contact detection
        contact = bool(min_limb_dist < 30)  # pixels

        return InteractionFeatures(
            center_distance=center_dist,
            min_limb_distance=min_limb_dist,
            orientation_diff=orient_diff,
            contact_detected=contact,
            relative_speed=0.0,  # computed externally with temporal info
        )


def normalize_features(features: PerFrameFeatures) -> PerFrameFeatures:
    """Ensure feature values are in reasonable ranges (returns new instance)."""
    return PerFrameFeatures(
        track_id=features.track_id,
        torso_angle=max(0.0, min(180.0, features.torso_angle)),
        center_x=max(0.0, min(1.0, features.center_x)),
        center_y=max(0.0, min(1.0, features.center_y)),
        head_height=max(0.0, min(1.0, features.head_height)),
        body_span=max(0.0, min(2.0, features.body_span)),
        avg_kp_confidence=max(0.0, min(1.0, features.avg_kp_confidence)),
        valid_kp_ratio=max(0.0, min(1.0, features.valid_kp_ratio)),
        skeleton_quality=max(0.0, min(1.0, features.skeleton_quality)),
    )
