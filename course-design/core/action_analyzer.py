# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Action sequence analyzer for skeleton-based behavior recognition.

Analyzes temporal sequences of skeleton data to extract motion features
and classify behaviors using statistical pattern matching.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .skeleton import PersonSkeleton

logger = logging.getLogger(__name__)


@dataclass
class ActionFeatures:
    """Extracted features from a skeleton sequence."""
    # Motion features
    avg_speed: float = 0.0
    max_speed: float = 0.0
    speed_variance: float = 0.0
    acceleration_variance: float = 0.0
    
    # Posture features
    avg_body_angle: float = 0.0
    body_angle_variance: float = 0.0
    min_body_angle: float = 0.0
    
    # Limb features
    limb_movement_variance: float = 0.0
    left_right_asymmetry: float = 0.0
    
    # Spatial features
    trajectory_length: float = 0.0
    trajectory_complexity: float = 0.0
    bounding_box_variance: float = 0.0
    
    # Temporal features
    duration: float = 0.0
    frame_count: int = 0
    
    # Fight-specific features
    mutual_proximity_score: float = 0.0
    chaotic_motion_score: float = 0.0
    rapid_direction_changes: int = 0


@dataclass
class BehaviorTemplate:
    """Template for a specific behavior pattern."""
    name: str
    description: str
    feature_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_duration: float = 0.5
    priority: int = 1


class SkeletonSequenceBuffer:
    """Buffer for storing skeleton sequences over time."""
    
    def __init__(self, max_length: int = 90):
        self.max_length = max_length
        self._buffers: dict[int, deque[PersonSkeleton]] = {}
    
    def add(self, track_id: int, skeleton: PersonSkeleton) -> None:
        """Add a skeleton to the track's sequence."""
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.max_length)
        self._buffers[track_id].append(skeleton)
    
    def get_sequence(self, track_id: int) -> list[PersonSkeleton]:
        """Get the skeleton sequence for a track."""
        return list(self._buffers.get(track_id, deque()))
    
    def get_all_sequences(self) -> dict[int, list[PersonSkeleton]]:
        """Get all track sequences."""
        return {tid: list(seq) for tid, seq in self._buffers.items()}
    
    def clear_old(self, max_age_frames: int = 300) -> None:
        """Clear sequences older than max_age_frames."""
        current_max = max(
            (seq[-1].frame_index for seq in self._buffers.values() if seq),
            default=0
        )
        to_remove = []
        for tid, seq in self._buffers.items():
            if seq and (current_max - seq[-1].frame_index) > max_age_frames:
                to_remove.append(tid)
        for tid in to_remove:
            del self._buffers[tid]


class ActionAnalyzer:
    """Analyzes skeleton sequences to detect behaviors.
    
    Uses statistical feature extraction and template matching
    to classify behaviors from temporal skeleton data.
    """
    
    def __init__(self):
        self.sequence_buffer = SkeletonSequenceBuffer(max_length=90)
        self.templates = self._load_behavior_templates()
    
    def _load_behavior_templates(self) -> list[BehaviorTemplate]:
        """Load predefined behavior templates."""
        return [
            BehaviorTemplate(
                name="running",
                description="Fast movement with consistent direction",
                feature_ranges={
                    "avg_speed": (50.0, 500.0),
                    "speed_variance": (0.0, 200.0),
                    "avg_body_angle": (0.0, 30.0),
                    "limb_movement_variance": (10.0, 100.0),
                },
                min_duration=0.5,
                priority=2,
            ),
            BehaviorTemplate(
                name="fall",
                description="Body angle drops significantly",
                feature_ranges={
                    "avg_body_angle": (45.0, 90.0),
                    "body_angle_variance": (100.0, 1000.0),
                    "min_body_angle": (0.0, 30.0),
                    "acceleration_variance": (50.0, 500.0),
                },
                min_duration=0.3,
                priority=3,
            ),
            BehaviorTemplate(
                name="fight",
                description="Chaotic mutual movement between two persons",
                feature_ranges={
                    "chaotic_motion_score": (50.0, 1000.0),
                    "rapid_direction_changes": (3, 50),
                    "mutual_proximity_score": (0.5, 1.0),
                    "limb_movement_variance": (50.0, 300.0),
                },
                min_duration=0.5,
                priority=3,
            ),
            BehaviorTemplate(
                name="crowd",
                description="High density of people in area",
                feature_ranges={
                    "frame_count": (1, 1000),
                },
                min_duration=1.0,
                priority=2,
            ),
        ]
    
    def process_frame(
        self,
        skeletons: list[PersonSkeleton],
        frame_index: int,
        timestamp: float,
    ) -> list[dict[str, Any]]:
        """Process skeletons from a frame and detect behaviors.
        
        Args:
            skeletons: List of detected person skeletons
            frame_index: Current frame index
            timestamp: Current timestamp
            
        Returns:
            List of detected behaviors with confidence scores
        """
        # Add skeletons to sequence buffer
        for sk in skeletons:
            sk.frame_index = frame_index
            sk.timestamp = timestamp
            if sk.track_id is not None:
                self.sequence_buffer.add(sk.track_id, sk)
        
        # Extract features and detect behaviors
        behaviors = []
        
        # Analyze individual behaviors (running, fall)
        for track_id, sequence in self.sequence_buffer.get_all_sequences().items():
            if len(sequence) < 5:
                continue
            
            features = self._extract_features(sequence)
            
            # Check each template
            for template in self.templates:
                if template.name in ("running", "fall"):
                    confidence = self._match_template(features, template)
                    if confidence > 0.6:
                        behaviors.append({
                            "behavior": template.name,
                            "track_id": track_id,
                            "confidence": confidence,
                            "features": features,
                            "description": template.description,
                        })
        
        # Analyze pair behaviors (fight)
        fight_behaviors = self._detect_fight_behavior(
            self.sequence_buffer.get_all_sequences()
        )
        behaviors.extend(fight_behaviors)
        
        # Analyze crowd behavior
        crowd_behavior = self._detect_crowd_behavior(skeletons)
        if crowd_behavior:
            behaviors.append(crowd_behavior)
        
        return behaviors
    
    def _extract_features(self, sequence: list[PersonSkeleton]) -> ActionFeatures:
        """Extract features from a skeleton sequence."""
        features = ActionFeatures()
        
        if len(sequence) < 2:
            return features
        
        # Extract positions over time
        positions = []
        angles = []
        speeds = []
        accelerations = []
        limb_vars = []
        
        for i, sk in enumerate(sequence):
            center = sk.get_center()
            positions.append(center)
            angles.append(sk.body_angle)
            
            if i > 0:
                dt = max(1e-6, sk.timestamp - sequence[i-1].timestamp)
                dx = center[0] - positions[i-1][0]
                dy = center[1] - positions[i-1][1]
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
                
                if i > 1:
                    accel = (speed - speeds[-2]) / dt
                    accelerations.append(abs(accel))
            
            # Calculate limb movement variance
            if sk.limb_lengths:
                limb_vars.append(np.var(list(sk.limb_lengths.values())))
        
        # Calculate features
        if speeds:
            features.avg_speed = float(np.mean(speeds))
            features.max_speed = float(np.max(speeds))
            features.speed_variance = float(np.var(speeds))
        
        if accelerations:
            features.acceleration_variance = float(np.var(accelerations))
        
        if angles:
            features.avg_body_angle = float(np.mean(angles))
            features.body_angle_variance = float(np.var(angles))
            features.min_body_angle = float(np.min(angles))
        
        if limb_vars:
            features.limb_movement_variance = float(np.mean(limb_vars))
        
        # Trajectory features
        if len(positions) > 1:
            total_dist = 0.0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_dist += np.sqrt(dx**2 + dy**2)
            features.trajectory_length = float(total_dist)
            
            # Trajectory complexity (displacement vs path length)
            if total_dist > 0:
                start = positions[0]
                end = positions[-1]
                displacement = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                features.trajectory_complexity = float(displacement / total_dist)
        
        features.duration = sequence[-1].timestamp - sequence[0].timestamp
        features.frame_count = len(sequence)
        
        return features
    
    def _match_template(self, features: ActionFeatures, template: BehaviorTemplate) -> float:
        """Match features against a behavior template.
        
        Returns confidence score between 0 and 1.
        """
        if features.duration < template.min_duration:
            return 0.0
        
        scores = []
        for feature_name, (min_val, max_val) in template.feature_ranges.items():
            value = getattr(features, feature_name, 0.0)
            
            # Calculate how well the value fits in the range
            if value < min_val:
                score = max(0.0, 1.0 - (min_val - value) / max(min_val, 1.0))
            elif value > max_val:
                score = max(0.0, 1.0 - (value - max_val) / max(max_val, 1.0))
            else:
                # Value is within range - calculate center proximity
                center = (min_val + max_val) / 2
                half_range = (max_val - min_val) / 2
                if half_range > 0:
                    score = 1.0 - abs(value - center) / half_range
                else:
                    score = 1.0
            
            scores.append(score)
        
        if not scores:
            return 0.0
        
        # Weight by number of matching features
        return float(np.mean(scores)) * (len(scores) / len(template.feature_ranges))
    
    def _detect_fight_behavior(
        self, sequences: dict[int, list[PersonSkeleton]]
    ) -> list[dict[str, Any]]:
        """Detect fight behavior between pairs of persons."""
        behaviors = []
        track_ids = list(sequences.keys())
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid1, tid2 = track_ids[i], track_ids[j]
                seq1 = sequences[tid1]
                seq2 = sequences[tid2]
                
                if len(seq1) < 5 or len(seq2) < 5:
                    continue
                
                # Calculate mutual features
                proximity_score = self._calculate_proximity_score(seq1, seq2)
                chaotic_score = self._calculate_chaotic_score(seq1, seq2)
                direction_changes = self._count_direction_changes(seq1, seq2)
                
                # Combined fight score
                fight_score = (
                    proximity_score * 0.3 +
                    min(chaotic_score / 500.0, 1.0) * 0.4 +
                    min(direction_changes / 10.0, 1.0) * 0.3
                )
                
                if fight_score > 0.6:
                    behaviors.append({
                        "behavior": "fight",
                        "track_ids": [tid1, tid2],
                        "confidence": float(fight_score),
                        "features": {
                            "proximity_score": float(proximity_score),
                            "chaotic_score": float(chaotic_score),
                            "direction_changes": direction_changes,
                        },
                        "description": "Mutual chaotic movement detected",
                    })
        
        return behaviors
    
    def _calculate_proximity_score(
        self, seq1: list[PersonSkeleton], seq2: list[PersonSkeleton]
    ) -> float:
        """Calculate how close two persons are over time."""
        proximities = []
        min_len = min(len(seq1), len(seq2))
        
        for i in range(min_len):
            center1 = seq1[i].get_center()
            center2 = seq2[i].get_center()
            dist = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
            
            # Normalize by average body height (approx 150 pixels)
            normalized_dist = dist / 150.0
            proximity = max(0.0, 1.0 - normalized_dist)
            proximities.append(proximity)
        
        return float(np.mean(proximities)) if proximities else 0.0
    
    def _calculate_chaotic_score(
        self, seq1: list[PersonSkeleton], seq2: list[PersonSkeleton]
    ) -> float:
        """Calculate motion chaos score for two persons."""
        chaos_scores = []
        
        for seq in [seq1, seq2]:
            if len(seq) < 3:
                continue
            
            # Calculate acceleration variance
            accelerations = []
            for i in range(2, len(seq)):
                center_i = seq[i].get_center()
                center_i1 = seq[i-1].get_center()
                center_i2 = seq[i-2].get_center()
                
                v1 = np.sqrt((center_i1[0]-center_i2[0])**2 + (center_i1[1]-center_i2[1])**2)
                v2 = np.sqrt((center_i[0]-center_i1[0])**2 + (center_i[1]-center_i1[1])**2)
                accel = abs(v2 - v1)
                accelerations.append(accel)
            
            if accelerations:
                chaos_scores.append(np.var(accelerations))
        
        return float(np.mean(chaos_scores)) if chaos_scores else 0.0
    
    def _count_direction_changes(
        self, seq1: list[PersonSkeleton], seq2: list[PersonSkeleton]
    ) -> int:
        """Count rapid direction changes."""
        changes = 0
        
        for seq in [seq1, seq2]:
            if len(seq) < 3:
                continue
            
            for i in range(2, len(seq)):
                center_i = seq[i].get_center()
                center_i1 = seq[i-1].get_center()
                center_i2 = seq[i-2].get_center()
                
                dx1 = center_i1[0] - center_i2[0]
                dy1 = center_i1[1] - center_i2[1]
                dx2 = center_i[0] - center_i1[0]
                dy2 = center_i[1] - center_i1[1]
                
                # Calculate angle between consecutive movements
                dot = dx1*dx2 + dy1*dy2
                mag1 = np.sqrt(dx1**2 + dy1**2)
                mag2 = np.sqrt(dx2**2 + dy2**2)
                
                if mag1 > 1e-6 and mag2 > 1e-6:
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    if angle > 90:  # Significant direction change
                        changes += 1
        
        return changes
    
    def _detect_crowd_behavior(
        self, skeletons: list[PersonSkeleton]
    ) -> dict[str, Any] | None:
        """Detect crowd behavior from current frame skeletons."""
        if len(skeletons) < 3:
            return None
        
        # Calculate density
        centers = [sk.get_center() for sk in skeletons]
        if not centers:
            return None
        
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area < 1:
            area = 1
        
        density = len(skeletons) / (area / 10000)  # persons per 100x100 pixel area
        
        if density > 0.5:  # Threshold for crowd
            return {
                "behavior": "crowd",
                "track_ids": [sk.track_id for sk in skeletons if sk.track_id is not None],
                "confidence": min(density, 1.0),
                "features": {"density": float(density), "count": len(skeletons)},
                "description": f"High density crowd: {len(skeletons)} persons",
            }
        
        return None
    
    def clear_sequences(self) -> None:
        """Clear all stored sequences."""
        self.sequence_buffer = SkeletonSequenceBuffer(max_length=self.sequence_buffer.max_length)
