# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Constants and helper functions for detection."""

# ── Performance Tuning ─────────────────────────────────────────────────────────

VACUUM_EVENT_COUNT = 1000
EVENT_FLUSH_BATCH = 50
MJPEG_CHUNK_SIZE = 8192
STREAM_TIMEOUT_S = 5.0
EVENT_STORE_CACHE_SIZE_MB = 8
STREAM_RESPONSE_TIMEOUT_S = 300

# ── COCO Classes ─────────────────────────────────────────────────────────────

COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

EVENT_TYPES: list[str] = ["running", "fall", "crowd", "intrusion", "fight", "vehicle_intrusion"]

PERSON_CLASS_ID: int = 0

VEHICLE_CLASS_IDS: list[int] = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

# COCO 17-keypoint skeleton definition
SKELETON_KEYPOINTS: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NUM_SKELETON_KEYPOINTS: int = 17

# Skeleton connections (bone index pairs for drawing and angle computation)
SKELETON_BONES: list[tuple[int, int]] = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12),
    # Lower body
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

# Bone groups for angle calculation (parent, child, grandchild for joint angles)
SKELETON_ANGLE_GROUPS: list[tuple[str, int, int, int]] = [
    ("left_elbow", 5, 7, 9),     # left_shoulder → left_elbow → left_wrist
    ("right_elbow", 6, 8, 10),   # right_shoulder → right_elbow → right_wrist
    ("left_knee", 11, 13, 15),   # left_hip → left_knee → left_ankle
    ("right_knee", 12, 14, 16),  # right_hip → right_knee → right_ankle
    ("left_hip_angle", 5, 11, 13),  # left_shoulder → left_hip → left_knee
    ("right_hip_angle", 6, 12, 14), # right_shoulder → right_hip → right_knee
]

# Priority levels for events
EVENT_PRIORITIES: dict[str, str] = {
    "fight": "CRITICAL",
    "fall": "CRITICAL",
    "intrusion": "WARNING",
    "crowd": "WARNING",
    "running": "INFO",
    "vehicle_intrusion": "WARNING",
}

DEFAULT_PRIORITY: str = "INFO"

# HSV-based color palette covering all 80 COCO classes for visual distinction
_COCO_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # 0  person
    (0, 255, 255),  # 1  bicycle
    (0, 165, 255),  # 2  car
    (255, 255, 0),  # 3  motorcycle
    (0, 0, 255),  # 4  airplane
    (255, 0, 0),  # 5  bus
    (128, 0, 128),  # 6  train
    (0, 255, 128),  # 7  truck
    (255, 128, 0),  # 8  boat
    (128, 128, 255),  # 9  traffic light
    (255, 0, 255),  # 10 fire hydrant
    (0, 128, 255),  # 11 stop sign
    (128, 255, 0),  # 12 parking meter
    (255, 128, 128),  # 13 bench
    (128, 0, 0),  # 14 bird
    (0, 128, 128),  # 15 cat
    (128, 128, 0),  # 16 dog
    (0, 0, 128),  # 17 horse
    (255, 255, 128),  # 18 sheep
    (128, 255, 255),  # 19 cow
    (255, 128, 255),  # 20 elephant
    (128, 64, 64),  # 21 bear
    (64, 128, 64),  # 22 zebra
    (64, 64, 128),  # 23 giraffe
    (192, 192, 128),  # 24 backpack
    (128, 192, 192),  # 25 umbrella
    (192, 128, 192),  # 26 handbag
    (64, 64, 64),  # 27 tie
    (192, 192, 192),  # 28 suitcase
    (64, 128, 128),  # 29 frisbee
    (128, 64, 128),  # 30 skis
    (128, 128, 64),  # 31 snowboard
    (64, 192, 64),  # 32 sports ball
    (192, 64, 64),  # 33 kite
    (64, 64, 192),  # 34 baseball bat
    (192, 64, 192),  # 35 baseball glove
    (64, 192, 192),  # 36 skateboard
    (192, 192, 64),  # 37 surfboard
    (64, 128, 192),  # 38 tennis racket
    (192, 64, 128),  # 39 bottle
    (128, 64, 192),  # 40 wine glass
    (128, 192, 64),  # 41 cup
    (64, 192, 128),  # 42 fork
    (192, 128, 64),  # 43 knife
    (64, 128, 0),  # 44 spoon
    (128, 64, 0),  # 45 bowl
    (192, 128, 0),  # 46 banana
    (64, 0, 128),  # 47 apple
    (128, 0, 64),  # 48 sandwich
    (0, 128, 64),  # 49 orange
    (0, 64, 128),  # 50 broccoli
    (128, 0, 192),  # 51 carrot
    (192, 0, 64),  # 52 hot dog
    (64, 0, 192),  # 53 pizza
    (192, 0, 128),  # 54 donut
    (128, 192, 0),  # 55 cake
    (64, 192, 0),  # 56 chair
    (0, 192, 64),  # 57 couch
    (0, 64, 192),  # 58 potted plant
    (192, 64, 0),  # 59 bed
    (64, 128, 255),  # 60 dining table
    (128, 64, 255),  # 61 toilet
    (64, 255, 128),  # 62 tv
    (128, 255, 64),  # 63 laptop
    (255, 64, 128),  # 64 mouse
    (255, 128, 64),  # 65 remote
    (64, 255, 192),  # 66 keyboard
    (192, 64, 255),  # 67 cell phone
    (192, 255, 64),  # 68 microwave
    (64, 192, 255),  # 69 oven
    (255, 64, 192),  # 70 toaster
    (255, 192, 64),  # 71 sink
    (192, 255, 128),  # 72 refrigerator
    (128, 192, 255),  # 73 book
    (64, 128, 192),  # 74 clock
    (192, 128, 255),  # 75 vase
    (128, 255, 192),  # 76 scissors
    (255, 192, 128),  # 77 teddy bear
    (192, 192, 255),  # 78 hair drier
    (192, 128, 128),  # 79 toothbrush
]


def get_class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


def get_detection_color(class_id: int) -> tuple[int, int, int]:
    if 0 <= class_id < len(_COCO_COLORS):
        return _COCO_COLORS[class_id]
    return (255, 255, 255)
