# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Configuration loading and dataclasses for detection engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from core.mllm.mllm_config import MLLMConfig, TensorRTConfig


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
class PoseConfig:
    enabled: bool = True
    model_path: str = "models/yolov11n-pose.pt"
    kp_threshold: float = 0.5
    smoothing_alpha: float = 0.3
    max_skeletons: int = 50
    process_interval: int = 2


@dataclass(frozen=True)
class SkeletonRunningRule:
    enabled: bool = True
    speed_threshold_kmh: float = 15.0
    min_duration_s: float = 0.5
    debounce_s: float = 5.0


@dataclass(frozen=True)
class SkeletonFallRule:
    enabled: bool = True
    torso_angle_threshold: float = 45.0
    head_height_threshold: float = 0.3
    fall_velocity_threshold: float = 0.5
    min_duration_s: float = 0.3
    debounce_s: float = 10.0


@dataclass(frozen=True)
class SkeletonFightRule:
    enabled: bool = True
    proximity_threshold_m: float = 1.5
    wrist_speed_threshold_ms: float = 2.0
    limb_frequency_threshold: float = 3.0
    min_duration_s: float = 0.5
    debounce_s: float = 5.0


@dataclass(frozen=True)
class SkeletonCrowdRule:
    enabled: bool = True
    density_threshold: float = 4.0
    min_duration_s: float = 5.0
    debounce_s: float = 10.0


@dataclass(frozen=True)
class SkeletonIntrusionRule:
    enabled: bool = True
    debounce_s: float = 5.0


@dataclass(frozen=True)
class SkeletonRulesConfig:
    running: SkeletonRunningRule = field(default_factory=SkeletonRunningRule)
    fall: SkeletonFallRule = field(default_factory=SkeletonFallRule)
    fight: SkeletonFightRule = field(default_factory=SkeletonFightRule)
    crowd: SkeletonCrowdRule = field(default_factory=SkeletonCrowdRule)
    intrusion: SkeletonIntrusionRule = field(default_factory=SkeletonIntrusionRule)


@dataclass(frozen=True)
class SequenceModelConfig:
    enabled: bool = False
    model_path: str = "models/behavior_lstm.onnx"
    sequence_length: int = 16
    inference_device: str = "cpu"
    confidence_threshold: float = 0.7
    shadow_mode: bool = True


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    enabled: bool = True
    adapt_window_s: int = 3600
    min_trigger_count: int = 10
    sensitivity: float = 0.8


@dataclass(frozen=True)
class PriorityAlertConfig:
    escalation_enabled: bool = False
    escalation_delay_s: int = 60


@dataclass(frozen=True)
class RulesConfig:
    running: RunningRule = field(default_factory=RunningRule)
    fall: FallRule = field(default_factory=FallRule)
    crowd: CrowdRule = field(default_factory=CrowdRule)
    intrusion: IntrusionRule = field(default_factory=IntrusionRule)
    fight: FightRule = field(default_factory=FightRule)
    skeleton: SkeletonRulesConfig = field(default_factory=SkeletonRulesConfig)


@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/yolov11x.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    classes: list[int] = field(default_factory=list)
    camera_fps: int = 30
    inference_scale: float = 1.0
    jpeg_quality: int = 80
    output_dir: str = "outputs"
    save_snapshots: bool = True
    view: bool = True
    rules: RulesConfig = field(default_factory=RulesConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    sequence_model: SequenceModelConfig = field(default_factory=SequenceModelConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    priority_alert: PriorityAlertConfig = field(default_factory=PriorityAlertConfig)
    mllm: MLLMConfig = field(default_factory=MLLMConfig)
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)

    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.output_dir, "snapshots")


class RuntimeSettings:
    """Mutable runtime settings that can be modified during pipeline execution.

    This class exists because AppConfig is a frozen dataclass and cannot be
    modified at runtime. RuntimeSettings tracks overrides applied after the
    pipeline was initialized (e.g., model hot-swapping).
    """

    def __init__(self, cfg: AppConfig):
        self._model_path: str = cfg.model_path
        self._device: str = cfg.device
        self._inference_scale: float = cfg.inference_scale
        self._jpeg_quality: int = cfg.jpeg_quality
        self._conf: float = cfg.conf
        self._iou: float = cfg.iou

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, value: str) -> None:
        self._model_path = value

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

    @property
    def inference_scale(self) -> float:
        return self._inference_scale

    @inference_scale.setter
    def inference_scale(self, value: float) -> None:
        self._inference_scale = value

    @property
    def jpeg_quality(self) -> int:
        return self._jpeg_quality

    @jpeg_quality.setter
    def jpeg_quality(self, value: int) -> None:
        self._jpeg_quality = value

    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, value: float) -> None:
        self._conf = value

    @property
    def iou(self) -> float:
        return self._iou

    @iou.setter
    def iou(self, value: float) -> None:
        self._iou = value


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
        # 处理 YAML 中布尔值简写（如 fall: true  →  {"enabled": true}）
        if not isinstance(data, dict):
            data = {"enabled": bool(data)}
        return cls(**{**defaults, **data})

    # "run" is the YAML key; "running" is the class name
    running = _parse_rule("run", RunningRule)
    fall = _parse_rule("fall", FallRule)
    crowd = _parse_rule("crowd", CrowdRule)
    fight = _parse_rule("fight", FightRule)

    intrusion_raw = rules_raw.get("intrusion", {})
    if not isinstance(intrusion_raw, dict):
        intrusion_raw = {}
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

    # Parse skeleton rules
    skeleton_raw = rules_raw.get("skeleton", {})
    def _parse_sk_rule(key: str, cls, **defaults):
        data = skeleton_raw.get(key, {})
        return cls(**{**defaults, **data})

    sk_running = _parse_sk_rule("running", SkeletonRunningRule)
    sk_fall = _parse_sk_rule("fall", SkeletonFallRule)
    sk_fight = _parse_sk_rule("fight", SkeletonFightRule)
    sk_crowd = _parse_sk_rule("crowd", SkeletonCrowdRule)
    sk_intrusion = _parse_sk_rule("intrusion", SkeletonIntrusionRule)

    # Parse pose config
    pose_raw = raw.get("pose", {})
    pose = PoseConfig(
        enabled=bool(pose_raw.get("enabled", True)),
        model_path=str(pose_raw.get("model_path", "models/yolov11n-pose.pt")),
        kp_threshold=float(pose_raw.get("kp_threshold", 0.5)),
        smoothing_alpha=float(pose_raw.get("smoothing_alpha", 0.3)),
        max_skeletons=int(pose_raw.get("max_skeletons", 50)),
        process_interval=int(pose_raw.get("process_interval", 2)),
    )

    # Parse sequence model config
    seq_raw = raw.get("sequence_model", {})
    sequence_model = SequenceModelConfig(
        enabled=bool(seq_raw.get("enabled", False)),
        model_path=str(seq_raw.get("model_path", "models/behavior_lstm.onnx")),
        sequence_length=int(seq_raw.get("sequence_length", 16)),
        inference_device=str(seq_raw.get("inference_device", "cpu")),
        confidence_threshold=float(seq_raw.get("confidence_threshold", 0.7)),
        shadow_mode=bool(seq_raw.get("shadow_mode", True)),
    )

    # Parse adaptive threshold config
    at_raw = raw.get("adaptive_threshold", {})
    adaptive_threshold = AdaptiveThresholdConfig(
        enabled=bool(at_raw.get("enabled", True)),
        adapt_window_s=int(at_raw.get("adapt_window_s", 3600)),
        min_trigger_count=int(at_raw.get("min_trigger_count", 10)),
        sensitivity=float(at_raw.get("sensitivity", 0.8)),
    )

    # Parse priority alert config
    pa_raw = raw.get("priority_alert", {})
    priority_alert = PriorityAlertConfig(
        escalation_enabled=bool(pa_raw.get("escalation_enabled", False)),
        escalation_delay_s=int(pa_raw.get("escalation_delay_s", 60)),
    )

    # Parse MLLM config
    mllm_raw = raw.get("mllm", {})
    mllm = MLLMConfig(
        enabled=bool(mllm_raw.get("enabled", False)),
        model_type=str(mllm_raw.get("model_type", "qwen2-vl-2b")),
        model_path=str(mllm_raw.get("model_path", "Qwen/Qwen2-VL-2B-Instruct")),
        inference_backend=str(mllm_raw.get("inference_backend", "auto")),
        device=str(mllm_raw.get("device", "auto")),
        half_precision=bool(mllm_raw.get("half_precision", True)),
        max_new_tokens=int(mllm_raw.get("max_new_tokens", 256)),
        temperature=float(mllm_raw.get("temperature", 0.3)),
        key_frame_interval=int(mllm_raw.get("key_frame_interval", 15)),
        context_window_frames=int(mllm_raw.get("context_window_frames", 5)),
        min_frame_size=int(mllm_raw.get("min_frame_size", 224)),
        max_frame_size=int(mllm_raw.get("max_frame_size", 512)),
        scene_description_enabled=bool(mllm_raw.get("scene_description_enabled", True)),
        scene_confidence_threshold=float(mllm_raw.get("scene_confidence_threshold", 0.5)),
        alarm_enhance_enabled=bool(mllm_raw.get("alarm_enhance_enabled", True)),
        enhancement_cooldown_s=float(mllm_raw.get("enhancement_cooldown_s", 10.0)),
        shadow_mode=bool(mllm_raw.get("shadow_mode", True)),
    )

    # Parse TensorRT config
    trt_raw = mllm_raw.get("tensorrt", {}) if isinstance(mllm_raw.get("tensorrt"), dict) else {}
    tensorrt = TensorRTConfig(
        enabled=bool(trt_raw.get("enabled", False)),
        precision=str(trt_raw.get("precision", "fp16")),
        engine_dir=str(trt_raw.get("engine_dir", "models/trt_engines")),
        onnx_dir=str(trt_raw.get("onnx_dir", "models/onnx")),
        max_batch_size=int(trt_raw.get("max_batch_size", 1)),
        workspace_gb=float(trt_raw.get("workspace_gb", 2.0)),
        dla_core=int(trt_raw.get("dla_core", -1)),
        optimization_level=int(trt_raw.get("optimization_level", 3)),
    )

    # Resolve device: check YAML model section first, then env var override
    device = str(model_cfg.get("device", "auto"))
    _yolo_device = os.environ.get("YOLO_DEVICE")
    if _yolo_device:
        device = _yolo_device

    return AppConfig(
        model_path=str(model_cfg.get("path", "models/yolov11x.pt")),
        device=device,
        imgsz=int(model_cfg.get("imgsz", 640)),
        conf=float(model_cfg.get("conf", 0.35)),
        iou=float(model_cfg.get("iou", 0.5)),
        classes=model_cfg.get("classes", []),
        camera_fps=int(camera_cfg.get("fps", 30)),
        inference_scale=float(model_cfg.get("inference_scale", 1.0)),
        jpeg_quality=int(model_cfg.get("jpeg_quality", 80)),
        output_dir=str(output_cfg.get("directory", "outputs")),
        save_snapshots=bool(output_cfg.get("save_snapshots", True)),
        view=bool(output_cfg.get("view", True)),
        rules=RulesConfig(
            running=running,
            fall=fall,
            crowd=crowd,
            intrusion=intrusion,
            fight=fight,
            skeleton=SkeletonRulesConfig(
                running=sk_running,
                fall=sk_fall,
                fight=sk_fight,
                crowd=sk_crowd,
                intrusion=sk_intrusion,
            ),
        ),
        pose=pose,
        sequence_model=sequence_model,
        adaptive_threshold=adaptive_threshold,
        priority_alert=priority_alert,
        mllm=mllm,
        tensorrt=tensorrt,
    )
