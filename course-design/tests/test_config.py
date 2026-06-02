"""Tests for configuration loading."""

import os
import sys
import tempfile

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

from core.config import AppConfig, load_config


def _write_yaml(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(content)
    return tmp.name


def test_load_minimal_config():
    yaml_content = """
model:
  path: "models/yolov11x.pt"
  imgsz: 640
  conf: 0.35
  iou: 0.5
"""
    path = _write_yaml(yaml_content)
    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert cfg.model_path == "models/yolov11x.pt"
    assert cfg.imgsz == 640
    assert cfg.conf == 0.35
    assert cfg.iou == 0.5
    assert cfg.device == "auto"
    os.unlink(path)


def test_load_full_config():
    yaml_content = """
model:
  path: "models/yolov12x.pt"
  imgsz: 320
  conf: 0.5
  iou: 0.4
device: "cuda:0"
camera:
  fps: 15
rules:
  running:
    enabled: true
    speed_px_s: 100
    min_duration_s: 0.5
  fall:
    enabled: false
  crowd:
    enabled: true
    min_people: 5
  intrusion:
    enabled: true
    zones:
      - name: "zone1"
        polygon: [[0,0],[100,0],[100,100],[0,100]]
  fight:
    enabled: true
    distance_threshold: 200
output:
  directory: "custom_outputs"
  save_snapshots: false
  view: false
"""
    path = _write_yaml(yaml_content)
    cfg = load_config(path)
    assert cfg.model_path == "models/yolov12x.pt"
    assert cfg.imgsz == 320
    assert cfg.conf == 0.5
    assert cfg.iou == 0.4
    assert cfg.device == "auto"  # device is read from model section, not top-level
    assert cfg.camera_fps == 15
    assert cfg.output_dir == "custom_outputs"
    assert cfg.save_snapshots is False
    assert cfg.view is False

    # Rules
    assert cfg.rules.running.enabled is True
    assert cfg.rules.running.speed_px_s == 100
    assert cfg.rules.fall.enabled is False
    assert cfg.rules.crowd.min_people == 5
    assert cfg.rules.intrusion.enabled is True
    assert len(cfg.rules.intrusion.zones) == 1
    assert cfg.rules.intrusion.zones[0].name == "zone1"
    assert cfg.rules.fight.distance_threshold == 200
    os.unlink(path)


def test_missing_model_section_uses_defaults():
    """When model section is missing entirely, defaults should be used."""
    yaml_content = """
output:
  directory: "outputs"
"""
    path = _write_yaml(yaml_content)
    cfg = load_config(path)
    assert cfg.model_path == "models/yolov11x.pt"
    assert cfg.imgsz == 640
    assert cfg.conf == 0.25
    os.unlink(path)


def test_empty_config_uses_defaults():
    yaml_content = "{}"
    path = _write_yaml(yaml_content)
    cfg = load_config(path)
    assert cfg.model_path == "models/yolov11x.pt"
    assert cfg.imgsz == 640
    assert cfg.device == "auto"
    os.unlink(path)


def test_snapshots_dir_property():
    import os

    cfg = AppConfig(output_dir="outputs")
    expected = os.path.join("outputs", "snapshots")
    assert cfg.snapshots_dir == expected


def test_config_crowd_proximity_default():
    """CrowdRule should have a default proximity_px."""
    cfg = AppConfig()
    assert cfg.rules.crowd.proximity_px == 180.0


def test_config_debounce_defaults():
    """All rules should have debounce_s defaults."""
    cfg = AppConfig()
    assert cfg.rules.running.debounce_s == 5.0
    assert cfg.rules.fall.debounce_s == 5.0
    assert cfg.rules.crowd.debounce_s == 8.0
    assert cfg.rules.intrusion.debounce_s == 5.0
    assert cfg.rules.fight.debounce_s == 5.0
