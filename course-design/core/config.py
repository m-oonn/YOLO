# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Configuration loading and dataclasses for detection engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: list[list[float]]


@dataclass(frozen=True)
class RunningRule:
    enabled: bool = False
    speed_px_s: float = 380
    min_duration_s: float = 0.6
    debounce_s: float = 5.0


@dataclass(frozen=True)
class FallRule:
    enabled: bool = False
    upright_aspect_min: float = 1.35
    fallen_aspect_max: float = 0.95
    transition_window_s: float = 1.0
    debounce_s: float = 5.0


@dataclass(frozen=True)
class CrowdRule:
    enabled: bool = False
    min_people: int = 10
    proximity_px: float = 200.0
    debounce_s: float = 10.0


@dataclass(frozen=True)
class IntrusionRule:
    enabled: bool = False
    zones: list[Zone] = field(default_factory=list)
    debounce_s: float = 5.0


@dataclass(frozen=True)
class FightRule:
    enabled: bool = False
    distance_threshold: float = 50
    movement_threshold: float = 100
    min_duration_s: float = 1.0
    debounce_s: float = 5.0


@dataclass(frozen=True)
class RulesConfig:
    running: RunningRule = field(default_factory=RunningRule)
    fall: FallRule = field(default_factory=FallRule)
    crowd: CrowdRule = field(default_factory=CrowdRule)
    intrusion: IntrusionRule = field(default_factory=IntrusionRule)
    fight: FightRule = field(default_factory=FightRule)


@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/yolov11x.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    classes: list[int] = field(default_factory=list)
    camera_fps: int = 30
    output_dir: str = "outputs"
    save_snapshots: bool = True
    view: bool = True
    rules: RulesConfig = field(default_factory=RulesConfig)

    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.output_dir, "snapshots")


def load_config(path: str | None = None) -> AppConfig:
    """Load YAML configuration file into AppConfig dataclass.

    If *path* is None, loads from ``configs/default.yaml`` relative to
    the project root.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "default.yaml",
        )
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Guard against empty YAML files (yaml.safe_load returns None)
    if raw is None:
        raw = {}

    model_cfg = raw.get("model") or {}
    camera_cfg = raw.get("camera") or {}
    output_cfg = raw.get("output") or {}
    rules_raw = raw.get("rules", {})

    # Parse rules
    def _parse_rule(key: str, cls, **defaults):
        data = rules_raw.get(key, {})
        return cls(**{**defaults, **data})

    running = _parse_rule("running", RunningRule)
    fall = _parse_rule("fall", FallRule)
    crowd = _parse_rule("crowd", CrowdRule)
    fight = _parse_rule("fight", FightRule)

    intrusion_raw = rules_raw.get("intrusion", {})
    zones = []
    for z in intrusion_raw.get("zones", []) or []:
        zones.append(
            Zone(
                name=str(z["name"]),
                polygon=[[float(x), float(y)] for x, y in z["polygon"]],
            )
        )
    intrusion = IntrusionRule(
        enabled=bool(intrusion_raw.get("enabled", False)),
        zones=zones,
        debounce_s=float(intrusion_raw.get("debounce_s", 5.0)),
    )

    return AppConfig(
        model_path=str(model_cfg.get("path", "models/yolov11x.pt")),
        device=str(raw.get("device", "auto")),
        imgsz=int(model_cfg.get("imgsz", 640)),
        conf=float(model_cfg.get("conf", 0.35)),
        iou=float(model_cfg.get("iou", 0.5)),
        classes=model_cfg.get("classes", []),
        camera_fps=int(camera_cfg.get("fps", 30)),
        output_dir=str(output_cfg.get("directory", "outputs")),
        save_snapshots=bool(output_cfg.get("save_snapshots", True)),
        view=bool(output_cfg.get("view", True)),
        rules=RulesConfig(
            running=running,
            fall=fall,
            crowd=crowd,
            intrusion=intrusion,
            fight=fight,
        ),
    )
