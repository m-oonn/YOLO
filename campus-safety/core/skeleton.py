# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】skeleton.py — 人体骨架提取（COCO 17关键点）
# 依赖：YOLO pose 模型（yolo11n-pose.pt）
# 被调用：pipeline.py 周期性调用（每 process_interval 帧）
# 核心职责：
#   ① YOLO姿态估计 → 提取17个关键点（鼻/眼/肩/肘/腕/髋/膝/踝）
#   ② 关键点去噪（低置信度过滤、遮挡处理）
#   ③ 骨架平滑（指数移动平均 EMA）
#   ④ 关键点归一化（相对于bbox的坐标）
# 数据流：YOLO keypoints → Skeleton对象（17个Point）→ behavior_analyzer
# ──────────────────────────────────────────────────────────

"""Skeleton extraction module for human pose estimation.

Extracts 17-point COCO-format skeleton keypoints from YOLO detection results.
Provides robust keypoint extraction with occlusion handling and confidence scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-point skeleton definition
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections for visualization and feature extraction
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


@dataclass
class SkeletonKeypoint:
    """A single keypoint with position and confidence."""
    x: float
    y: float
    confidence: float
    visible: bool = True

    def is_valid(self, threshold: float = 0.3) -> bool:
        """Check if keypoint is valid (visible and above confidence threshold)."""
        return self.visible and self.confidence >= threshold


@dataclass
class PersonSkeleton:
    """Complete skeleton for a single person."""
    track_id: int | None
    keypoints: list[SkeletonKeypoint]
    bbox: dict[str, float]
    timestamp: float = 0.0
    frame_index: int = 0
    body_angle: float = 0.0
    aspect_ratio: float = 0.0
    limb_lengths: dict[str, float] = field(default_factory=dict)
    head_height: float = 0.0

    def get_keypoint(self, name: str) -> SkeletonKeypoint | None:
        """Get keypoint by name."""
        if name not in COCO_KEYPOINTS:
            return None
        idx = COCO_KEYPOINTS.index(name)
        return self.keypoints[idx] if idx < len(self.keypoints) else None

    def is_valid(self, min_visible: int = 5) -> bool:
        """Check if skeleton has enough visible keypoints."""
        visible_count = sum(1 for kp in self.keypoints if kp.visible and kp.confidence > 0.3)
        return visible_count >= min_visible

    @property
    def is_low_quality(self) -> bool:
        """Check if skeleton quality is too low for reliable analysis."""
        visible_count = sum(1 for kp in self.keypoints if kp.visible and kp.confidence > 0.3)
        return visible_count < 5

    @property
    def center(self) -> tuple[float, float]:
        """Get skeleton center point (hip center or bbox center)."""
        # Allow override for testing
        if hasattr(self, '_center_override'):
            return self._center_override
        return self.get_center()

    def get_center(self) -> tuple[float, float]:
        """Get skeleton center point (hip center)."""
        left_hip = self.get_keypoint("left_hip")
        right_hip = self.get_keypoint("right_hip")
        if left_hip and right_hip and left_hip.visible and right_hip.visible:
            return ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        # Fallback to bbox center
        return (
            (self.bbox["x1"] + self.bbox["x2"]) / 2,
            (self.bbox["y1"] + self.bbox["y2"]) / 2,
        )


# Backward-compatible alias used by pose_features.py and other modules
Skeleton = PersonSkeleton


class SkeletonExtractor:
    """Extracts skeleton keypoints from YOLO detection results.
    
    Uses a lightweight pose estimation model or heuristic-based
    keypoint estimation from bounding boxes.
    """
    
    def __init__(self, use_pose_model: bool = False, pose_model_path: str | None = None, kp_threshold: float = 0.3):
        self.use_pose_model = use_pose_model
        self.pose_model = None
        self.kp_threshold = kp_threshold
        
        if use_pose_model and pose_model_path:
            try:
                from ultralytics import YOLO
                self.pose_model = YOLO(pose_model_path)
                logger.info(f"Loaded pose model: {pose_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pose model: {e}, using heuristic extraction")
                self.use_pose_model = False
    
    def extract(self, frame: np.ndarray, detections: list[dict]) -> list[PersonSkeleton]:
        """Extract skeletons from frame and detections.
        
        Args:
            frame: Current video frame
            detections: List of detection dicts with bbox and track_id
            
        Returns:
            List of PersonSkeleton objects
        """
        skeletons = []
        
        if self.use_pose_model and self.pose_model is not None:
            # Use YOLO pose model for accurate keypoint extraction
            skeletons = self._extract_with_pose_model(frame, detections)
        else:
            # Use heuristic-based extraction from bounding boxes
            skeletons = self._extract_heuristic(frame, detections)
        
        # Post-process: handle occlusions and estimate missing keypoints
        skeletons = [self._post_process(sk) for sk in skeletons]
        
        return skeletons
    
    def _extract_with_pose_model(
        self, frame: np.ndarray, detections: list[dict]
    ) -> list[PersonSkeleton]:
        """Extract keypoints using YOLO pose model."""
        skeletons = []
        
        try:
            results = self.pose_model(frame, verbose=False)
            
            for result in results:
                if result.keypoints is None:
                    continue
                
                keypoints_data = result.keypoints.data.cpu().numpy()
                
                for i, kpts in enumerate(keypoints_data):
                    # Match with detection by IoU
                    matched_detection = self._match_detection(result.boxes[i], detections)
                    
                    if matched_detection is None:
                        continue
                    
                    # Convert to SkeletonKeypoint list
                    keypoint_list = []
                    for j, (x, y, conf) in enumerate(kpts[:17]):
                        keypoint_list.append(SkeletonKeypoint(
                            x=float(x),
                            y=float(y),
                            confidence=float(conf),
                            visible=conf > 0.3
                        ))
                    
                    skeleton = PersonSkeleton(
                        track_id=matched_detection.get("track_id"),
                        keypoints=keypoint_list,
                        bbox=matched_detection["bbox"],
                    )
                    skeletons.append(skeleton)
        except Exception as e:
            logger.warning(f"Pose model extraction failed: {e}, falling back to heuristic")
            return self._extract_heuristic(frame, detections)
        
        return skeletons
    
    def _extract_heuristic(
        self, frame: np.ndarray, detections: list[dict]
    ) -> list[PersonSkeleton]:
        """Extract approximate keypoints from bounding boxes using heuristics.
        
        Uses anthropometric ratios to estimate keypoint positions.
        """
        skeletons = []
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            w = x2 - x1
            h = y2 - y1
            
            # Estimate keypoints based on anthropometric ratios
            keypoints = []
            
            # Head (nose) - top center
            keypoints.append(SkeletonKeypoint(x1 + w * 0.5, y1 + h * 0.08, 0.6))
            # Eyes
            keypoints.append(SkeletonKeypoint(x1 + w * 0.4, y1 + h * 0.06, 0.5))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.6, y1 + h * 0.06, 0.5))
            # Ears
            keypoints.append(SkeletonKeypoint(x1 + w * 0.25, y1 + h * 0.08, 0.4))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.75, y1 + h * 0.08, 0.4))
            
            # Shoulders
            keypoints.append(SkeletonKeypoint(x1 + w * 0.25, y1 + h * 0.22, 0.7))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.75, y1 + h * 0.22, 0.7))
            
            # Elbows
            keypoints.append(SkeletonKeypoint(x1 + w * 0.15, y1 + h * 0.38, 0.5))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.85, y1 + h * 0.38, 0.5))
            
            # Wrists
            keypoints.append(SkeletonKeypoint(x1 + w * 0.1, y1 + h * 0.52, 0.4))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.9, y1 + h * 0.52, 0.4))
            
            # Hips
            keypoints.append(SkeletonKeypoint(x1 + w * 0.3, y1 + h * 0.55, 0.7))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.7, y1 + h * 0.55, 0.7))
            
            # Knees
            keypoints.append(SkeletonKeypoint(x1 + w * 0.25, y1 + h * 0.75, 0.6))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.75, y1 + h * 0.75, 0.6))
            
            # Ankles
            keypoints.append(SkeletonKeypoint(x1 + w * 0.2, y1 + h * 0.95, 0.5))
            keypoints.append(SkeletonKeypoint(x1 + w * 0.8, y1 + h * 0.95, 0.5))
            
            skeleton = PersonSkeleton(
                track_id=det.get("track_id"),
                keypoints=keypoints,
                bbox=bbox,
            )
            skeletons.append(skeleton)
        
        return skeletons
    
    def _match_detection(self, box, detections: list[dict]) -> dict | None:
        """Match pose result box with detection by IoU."""
        best_iou = 0.3
        best_det = None
        
        box_xyxy = box.xyxy.cpu().numpy()[0]
        bx1, by1, bx2, by2 = box_xyxy
        
        for det in detections:
            bbox = det["bbox"]
            dx1, dy1, dx2, dy2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # Calculate IoU
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx2, dx2)
            iy2 = min(by2, dy2)
            
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_box = (bx2 - bx1) * (by2 - by1)
            area_det = (dx2 - dx1) * (dy2 - dy1)
            union = area_box + area_det - inter
            
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        
        return best_det
    
    def _post_process(self, skeleton: PersonSkeleton) -> PersonSkeleton:
        """Post-process skeleton: estimate missing keypoints, calculate features."""
        # Estimate occluded/missing keypoints before feature calculation
        skeleton = self._estimate_occluded_keypoints(skeleton)

        # Calculate body angle with ground
        skeleton.body_angle = self._calculate_body_angle(skeleton)

        # Calculate aspect ratio
        bbox = skeleton.bbox
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        skeleton.aspect_ratio = h / max(w, 1e-6)

        # Calculate limb lengths
        skeleton.limb_lengths = self._calculate_limb_lengths(skeleton)

        return skeleton

    def _estimate_occluded_keypoints(self, skeleton: PersonSkeleton) -> PersonSkeleton:
        """Estimate missing keypoints using anthropometric constraints.

        Uses visible keypoints and body part relationships to estimate
        occluded keypoints. Improves robustness in crowded scenes.
        """
        kps = skeleton.keypoints
        if len(kps) < 17:
            return skeleton

        bbox_h = skeleton.bbox.get("y2", 0) - skeleton.bbox.get("y1", 0)
        scale = max(0.5, min(3.0, bbox_h / 200.0)) if bbox_h > 0 else 1.0
        wrist_offset = int(30 * scale)
        ankle_offset = int(40 * scale)

        # Estimate missing eyes from nose and ears
        nose = kps[0]
        left_eye = kps[1]
        right_eye = kps[2]
        left_ear = kps[3]
        right_ear = kps[4]

        if not left_eye.visible and nose.visible and left_ear.visible:
            # Eye is approximately between nose and ear
            kps[1] = SkeletonKeypoint(
                x=(nose.x + left_ear.x) / 2,
                y=(nose.y + left_ear.y) / 2,
                confidence=0.4,
                visible=True,
            )
        if not right_eye.visible and nose.visible and right_ear.visible:
            kps[2] = SkeletonKeypoint(
                x=(nose.x + right_ear.x) / 2,
                y=(nose.y + right_ear.y) / 2,
                confidence=0.4,
                visible=True,
            )

        # Estimate missing ears from eyes
        if not left_ear.visible and left_eye.visible:
            # Ear is typically to the left of left eye
            kps[3] = SkeletonKeypoint(
                x=left_eye.x - abs(nose.x - left_eye.x) * 0.8,
                y=left_eye.y,
                confidence=0.3,
                visible=True,
            )
        if not right_ear.visible and right_eye.visible:
            kps[4] = SkeletonKeypoint(
                x=right_eye.x + abs(nose.x - right_eye.x) * 0.8,
                y=right_eye.y,
                confidence=0.3,
                visible=True,
            )

        # Estimate missing wrists from elbows
        left_elbow = kps[7]
        right_elbow = kps[8]
        left_wrist = kps[9]
        right_wrist = kps[10]

        if not left_wrist.visible and left_elbow.visible:
            # Wrist typically extends downward from elbow
            kps[9] = SkeletonKeypoint(
                x=left_elbow.x,
                y=left_elbow.y + wrist_offset,
                confidence=0.3,
                visible=True,
            )
        if not right_wrist.visible and right_elbow.visible:
            kps[10] = SkeletonKeypoint(
                x=right_elbow.x,
                y=right_elbow.y + wrist_offset,
                confidence=0.3,
                visible=True,
            )

        # Estimate missing knees from hips and ankles
        left_hip = kps[11]
        right_hip = kps[12]
        left_knee = kps[13]
        right_knee = kps[14]
        left_ankle = kps[15]
        right_ankle = kps[16]

        if not left_knee.visible and left_hip.visible and left_ankle.visible:
            kps[13] = SkeletonKeypoint(
                x=(left_hip.x + left_ankle.x) / 2,
                y=(left_hip.y + left_ankle.y) / 2,
                confidence=0.4,
                visible=True,
            )
        if not right_knee.visible and right_hip.visible and right_ankle.visible:
            kps[14] = SkeletonKeypoint(
                x=(right_hip.x + right_ankle.x) / 2,
                y=(right_hip.y + right_ankle.y) / 2,
                confidence=0.4,
                visible=True,
            )

        # Estimate missing ankles from knees
        if not left_ankle.visible and left_knee.visible:
            kps[15] = SkeletonKeypoint(
                x=left_knee.x,
                y=left_knee.y + ankle_offset,
                confidence=0.3,
                visible=True,
            )
        if not right_ankle.visible and right_knee.visible:
            kps[16] = SkeletonKeypoint(
                x=right_knee.x,
                y=right_knee.y + ankle_offset,
                confidence=0.3,
                visible=True,
            )

        skeleton.keypoints = kps
        return skeleton

    def _calculate_body_angle(self, skeleton: PersonSkeleton) -> float:
        """Calculate body angle with ground plane (degrees).
        
        Uses shoulder-hip line to estimate body orientation.
        """
        left_shoulder = skeleton.get_keypoint("left_shoulder")
        right_shoulder = skeleton.get_keypoint("right_shoulder")
        left_hip = skeleton.get_keypoint("left_hip")
        right_hip = skeleton.get_keypoint("right_hip")
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
        
        # Use torso center line
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_x = (left_hip.x + right_hip.x) / 2
        
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        
        if abs(dy) < 1e-6:
            return 90.0
        
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        return angle

    def _calculate_limb_lengths(self, skeleton: PersonSkeleton) -> dict[str, float]:
        """Calculate lengths of major body limbs."""
        lengths = {}
        
        limb_pairs = [
            ("left_upper_arm", "left_shoulder", "left_elbow"),
            ("left_lower_arm", "left_elbow", "left_wrist"),
            ("right_upper_arm", "right_shoulder", "right_elbow"),
            ("right_lower_arm", "right_elbow", "right_wrist"),
            ("left_thigh", "left_hip", "left_knee"),
            ("left_shin", "left_knee", "left_ankle"),
            ("right_thigh", "right_hip", "right_knee"),
            ("right_shin", "right_knee", "right_ankle"),
            ("torso", "left_shoulder", "left_hip"),
        ]
        
        for name, kp1_name, kp2_name in limb_pairs:
            kp1 = skeleton.get_keypoint(kp1_name)
            kp2 = skeleton.get_keypoint(kp2_name)
            if kp1 and kp2 and kp1.visible and kp2.visible:
                length = np.sqrt((kp2.x - kp1.x) ** 2 + (kp2.y - kp1.y) ** 2)
                lengths[name] = float(length)
        
        return lengths

    def draw_skeleton(
        self, frame: np.ndarray, skeleton: PersonSkeleton, color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """Draw skeleton on frame for visualization."""
        out = frame.copy()
        
        # Draw connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(skeleton.keypoints) and end_idx < len(skeleton.keypoints):
                kp1 = skeleton.keypoints[start_idx]
                kp2 = skeleton.keypoints[end_idx]
                if kp1.visible and kp2.visible:
                    pt1 = (int(kp1.x), int(kp1.y))
                    pt2 = (int(kp2.x), int(kp2.y))
                    cv2.line(out, pt1, pt2, color, 2)
        
        # Draw keypoints
        for i, kp in enumerate(skeleton.keypoints):
            if kp.visible:
                pt = (int(kp.x), int(kp.y))
                cv2.circle(out, pt, 3, color, -1)
                # Draw keypoint name for debugging
                if i < len(COCO_KEYPOINTS):
                    cv2.putText(
                        out, COCO_KEYPOINTS[i], (pt[0] + 5, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                    )
        
        return out


def compute_torso_angle(skeleton: PersonSkeleton) -> float:
    """Compute torso angle for fall detection."""
    extractor = SkeletonExtractor()
    return extractor._calculate_body_angle(skeleton)


class SkeletonRenderer:
    """Renders skeletons on video frames for visualization."""

    def __init__(self):
        self.extractor = SkeletonExtractor()

    def render(self, frame: np.ndarray, skeletons: list[PersonSkeleton]) -> np.ndarray:
        """Render all skeletons on the frame."""
        out = frame.copy()
        for skeleton in skeletons:
            out = self.extractor.draw_skeleton(out, skeleton)
        return out


def compute_all_bone_angles(skeleton: PersonSkeleton) -> dict[str, float]:
    """Compute all bone angles for behavior analysis."""
    angles = {}
    
    # Define bone pairs and their expected directions
    bone_defs = [
        ("left_upper_arm", "left_shoulder", "left_elbow"),
        ("right_upper_arm", "right_shoulder", "right_elbow"),
        ("left_thigh", "left_hip", "left_knee"),
        ("right_thigh", "right_hip", "right_knee"),
        ("left_shin", "left_knee", "left_ankle"),
        ("right_shin", "right_knee", "right_ankle"),
    ]
    
    for name, start_name, end_name in bone_defs:
        start_kp = skeleton.get_keypoint(start_name)
        end_kp = skeleton.get_keypoint(end_name)
        if start_kp and end_kp and start_kp.visible and end_kp.visible:
            dx = end_kp.x - start_kp.x
            dy = end_kp.y - start_kp.y
            angle = np.degrees(np.arctan2(dy, dx))
            angles[name] = float(angle)
    
    return angles
