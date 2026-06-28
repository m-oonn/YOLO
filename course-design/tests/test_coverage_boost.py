"""Additional unit tests to improve coverage."""

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import yaml

# ── Constants ─────────────────────────────────────────────────────────


class TestConstants:
    def test_get_class_name_valid(self):
        from core.constants import get_class_name

        assert get_class_name(0) == "person"
        assert get_class_name(79) == "toothbrush"

    def test_get_class_name_out_of_range(self):
        from core.constants import get_class_name

        assert get_class_name(-1) == "class_-1"
        assert get_class_name(80) == "class_80"
        assert get_class_name(100) == "class_100"

    def test_get_detection_color_valid(self):
        from core.constants import get_detection_color

        color = get_detection_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_get_detection_color_out_of_range(self):
        from core.constants import get_detection_color

        assert get_detection_color(-1) == (255, 255, 255)
        assert get_detection_color(80) == (255, 255, 255)
        assert get_detection_color(999) == (255, 255, 255)

    def test_coco_classes_count(self):
        from core.constants import COCO_CLASSES

        assert len(COCO_CLASSES) == 80
        assert "person" in COCO_CLASSES

    def test_event_types_complete(self):
        from core.constants import EVENT_TYPES

        expected = {
            "running",
            "fall",
            "crowd",
            "intrusion",
            "fight",
            "vehicle_intrusion",
            "suspicious",
        }
        assert set(EVENT_TYPES) == expected

    def test_person_class_id(self):
        from core.constants import PERSON_CLASS_ID

        assert PERSON_CLASS_ID == 0

    def test_coco_colors_count(self):
        from core.constants import _COCO_COLORS

        assert len(_COCO_COLORS) == 80


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_load_default_config(self):
        from core.config import load_config

        cfg = load_config()
        assert ".pt" in cfg.model_path or ".onnx" in cfg.model_path
        assert cfg.conf == 0.25
        assert cfg.camera_fps == 30
        assert cfg.rules.running.enabled is True

    def test_load_custom_config(self, tmp_path):
        from core.config import load_config

        data = {
            "model": {"path": "custom.pt", "conf": 0.5, "imgsz": 320, "device": "cpu"},
            "camera": {"fps": 15},
            "output": {"directory": "/tmp/out", "save_snapshots": False, "view": False},
            "rules": {
                "running": {"enabled": True, "speed_px_s": 500},
                "fall": {"enabled": True},
                "crowd": {"enabled": True, "min_people": 8},
                "intrusion": {"enabled": False},
                "fight": {"enabled": True, "distance_threshold": 30},
            },
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_config(str(f))
        assert cfg.model_path == "custom.pt"
        assert cfg.conf == 0.5
        assert cfg.imgsz == 320
        assert cfg.camera_fps == 15
        assert cfg.output_dir == "/tmp/out"
        assert cfg.save_snapshots is False
        assert cfg.view is False
        assert cfg.device == "cpu"
        assert cfg.rules.running.speed_px_s == 500
        assert cfg.rules.crowd.min_people == 8

    def test_load_empty_yaml_returns_defaults(self, tmp_path):
        from core.config import load_config

        f = tmp_path / "empty.yaml"
        f.write_text("---\n")
        cfg = load_config(str(f))
        assert "yolov" in cfg.model_path
        assert cfg.conf == 0.25

    def test_load_config_with_intrusion_zones(self, tmp_path):
        from core.config import load_config

        data = {
            "model": {},
            "rules": {
                "intrusion": {
                    "enabled": True,
                    "zones": [
                        {
                            "name": "zone1",
                            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                }
            },
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_config(str(f))
        assert cfg.rules.intrusion.enabled is True
        assert len(cfg.rules.intrusion.zones) == 1
        assert cfg.rules.intrusion.zones[0].name == "zone1"

    def test_snapshots_dir(self):
        from core.config import load_config

        cfg = load_config()
        assert "snapshots" in cfg.snapshots_dir

    def test_model_config_defaults(self, tmp_path):
        from core.config import load_config

        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump({"model": {}}))
        cfg = load_config(str(f))
        assert "yolov" in cfg.model_path
        assert cfg.conf == 0.25
        assert cfg.iou == 0.45


# ── Geometry ──────────────────────────────────────────────────────────


class TestGeometry:
    def test_point_in_polygon_center(self):
        from core.geometry import point_in_polygon

        assert point_in_polygon(5, 5, [(0, 0), (10, 0), (10, 10), (0, 10)]) is True

    def test_point_in_polygon_outside(self):
        from core.geometry import point_in_polygon

        assert point_in_polygon(15, 15, [(0, 0), (10, 0), (10, 10), (0, 10)]) is False

    def test_point_in_polygon_concave(self):
        from core.geometry import point_in_polygon

        poly = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
        assert point_in_polygon(2, 2, poly) is True
        assert point_in_polygon(8, 8, poly) is False

    def test_point_in_polygon_empty(self):
        from core.geometry import point_in_polygon

        assert point_in_polygon(0, 0, []) is False


# ── Detection API helpers ─────────────────────────────────────────────


class TestDetectionHelpers:
    def test_allowed_video_extensions(self):
        from backend.security import ALLOWED_VIDEO_EXTENSIONS

        assert ".mp4" in ALLOWED_VIDEO_EXTENSIONS
        assert ".avi" in ALLOWED_VIDEO_EXTENSIONS
        assert len(ALLOWED_VIDEO_EXTENSIONS) == 7

    def test_allowed_video_mime_types(self):
        from backend.security import ALLOWED_VIDEO_MIME_TYPES

        assert "video/mp4" in ALLOWED_VIDEO_MIME_TYPES

    def test_max_upload_size(self):
        from backend.api.detection import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB

        assert MAX_UPLOAD_SIZE_MB == 100
        assert MAX_UPLOAD_SIZE_BYTES == 100 * 1024 * 1024
