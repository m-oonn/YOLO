# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】config.py — 配置中枢（YAML → 冻结数据类）
# 上游依赖：core/mllm/mllm_config.py
# 下游调用：所有模块都通过 AppConfig 读取配置参数
# 核心职责：
#   ① 读取 configs/default.yaml
#   ② 解析为冻结数据类 AppConfig（不可变，线程安全）
#   ③ 定义所有规则参数（奔跑/跌倒/聚集/打架/车辆/骨架）
#   ④ RuntimeSettings 支持运行时热更新（模型路径/阈值）
# 提示：修改任何参数前，先看 default.yaml 中有没有对应项
# ──────────────────────────────────────────────────────────

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
    speed_px_s: float = 350
    min_duration_s: float = 0.4
    debounce_s: float = 5.0


@dataclass(frozen=True)
class FallRule:
    enabled: bool = False
    upright_aspect_min: float = 1.10
    fallen_aspect_max: float = 0.95
    transition_window_s: float = 2.0
    debounce_s: float = 5.0
    confirm_frames: int = 6
    min_aspect_change_rate: float = 0.10


@dataclass(frozen=True)
class CrowdRule:
    enabled: bool = False
    min_people: int = 4
    proximity_px: float = 180.0
    debounce_s: float = 8.0
    min_duration_s: float = 1.5


@dataclass(frozen=True)
class IntrusionRule:
    enabled: bool = False
    zones: list[Zone] = field(default_factory=list)
    debounce_s: float = 5.0
    min_duration_s: float = 1.0


@dataclass(frozen=True)
class FightRule:
    enabled: bool = False
    distance_threshold: float = 150
    movement_threshold: float = 30
    iou_threshold: float = 0.05
    chaos_threshold: float = 60.0
    min_duration_s: float = 0.6
    debounce_s: float = 5.0
    required_score: int = 3


@dataclass(frozen=True)
class VehicleRule:
    enabled: bool = False
    zones: list[Zone] = field(default_factory=list)
    debounce_s: float = 10.0
    min_duration_s: float = 2.0


@dataclass(frozen=True)
class PoseConfig:
    enabled: bool = True
    model_path: str = "models/yolo11s-pose.pt"
    kp_threshold: float = 0.4
    smoothing_alpha: float = 0.4
    max_skeletons: int = 50
    process_interval: int = 1


@dataclass(frozen=True)
class SkeletonRunningRule:
    enabled: bool = True
    speed_threshold_kmh: float = 10.0
    min_duration_s: float = 0.4
    debounce_s: float = 5.0


@dataclass(frozen=True)
class SkeletonFallRule:
    enabled: bool = True
    torso_angle_threshold: float = 40.0
    head_height_threshold: float = 0.3
    fall_velocity_threshold: float = 0.4
    emergency_velocity_px: float = 1.2
    min_duration_s: float = 0.2
    debounce_s: float = 8.0


@dataclass(frozen=True)
class SkeletonFightRule:
    enabled: bool = True
    proximity_threshold_m: float = 2.0
    wrist_speed_threshold_ms: float = 1.5
    limb_frequency_threshold: float = 2.5
    min_duration_s: float = 0.4
    debounce_s: float = 5.0


@dataclass(frozen=True)
class SkeletonCrowdRule:
    enabled: bool = True
    density_threshold: float = 3.5
    min_duration_s: float = 3.0
    debounce_s: float = 8.0


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
    vehicle: VehicleRule = field(default_factory=VehicleRule)
    skeleton: SkeletonRulesConfig = field(default_factory=SkeletonRulesConfig)


@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/yolov11x.pt"
    fall_model_path: str = "models/best.pt"
    fight_model_path: str = "models/suspicious_activity_nano.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    half: bool = False
    classes: list = field(default_factory=list)
    tracker: str = "botsort"
    camera_fps: int = 30
    inference_scale: float = 1.0
    jpeg_quality: int = 85
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

    running = _parse_rule("running", RunningRule)
    fall = _parse_rule("fall", FallRule)
    crowd = _parse_rule("crowd", CrowdRule)
    fight = _parse_rule(
        "fight", FightRule,
        distance_threshold=150.0, movement_threshold=30.0,
        iou_threshold=0.05, chaos_threshold=60.0,
        required_score=3,
    )

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

    # Parse vehicle rule
    vehicle_raw = rules_raw.get("vehicle", {})
    if not isinstance(vehicle_raw, dict):
        vehicle_raw = {}
    vehicle_zones = []
    for z in vehicle_raw.get("zones", []) or []:
        vehicle_zones.append(
            Zone(
                name=str(z["name"]),
                polygon=[[float(x), float(y)] for x, y in z["polygon"]],
            )
        )
    vehicle = VehicleRule(
        enabled=bool(vehicle_raw.get("enabled", False)),
        zones=vehicle_zones,
        debounce_s=float(vehicle_raw.get("debounce_s", 10.0)),
        min_duration_s=float(vehicle_raw.get("min_duration_s", 2.0)),
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
        model_path=str(pose_raw.get("model_path", "models/yolo11s-pose.pt")),
        kp_threshold=float(pose_raw.get("kp_threshold", 0.4)),
        smoothing_alpha=float(pose_raw.get("smoothing_alpha", 0.4)),
        max_skeletons=int(pose_raw.get("max_skeletons", 50)),
        process_interval=int(pose_raw.get("process_interval", 1)),
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
        model_type=str(mllm_raw.get("model_type", "qwen2-vl-2b-instruct")),
        model_path=str(mllm_raw.get("model_path", "Qwen/Qwen2-VL-2B-Instruct")),
        inference_backend=str(mllm_raw.get("inference_backend", "auto")),
        device=str(mllm_raw.get("device", "auto")),
        half_precision=bool(mllm_raw.get("half_precision", True)),
        use_flash_attention=bool(mllm_raw.get("use_flash_attention", True)),
        max_new_tokens=int(mllm_raw.get("max_new_tokens", 128)),
        temperature=float(mllm_raw.get("temperature", 0.2)),
        key_frame_interval=int(mllm_raw.get("key_frame_interval", 30)),
        min_inference_interval_s=float(mllm_raw.get("min_inference_interval_s", 3.0)),
        context_window_frames=int(mllm_raw.get("context_window_frames", 5)),
        min_frame_size=int(mllm_raw.get("min_frame_size", 224)),
        max_frame_size=int(mllm_raw.get("max_frame_size", 384)),
        scene_description_enabled=bool(mllm_raw.get("scene_description_enabled", True)),
        scene_confidence_threshold=float(mllm_raw.get("scene_confidence_threshold", 0.5)),
        alarm_enhance_enabled=bool(mllm_raw.get("alarm_enhance_enabled", True)),
        enhancement_cooldown_s=float(mllm_raw.get("enhancement_cooldown_s", 10.0)),
        shadow_mode=bool(mllm_raw.get("shadow_mode", False)),
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
        fall_model_path=str(model_cfg.get("fall_model_path", "models/best.pt")),
        fight_model_path=str(model_cfg.get("fight_model_path", "models/suspicious_activity_nano.pt")),
        device=device,
        imgsz=int(model_cfg.get("imgsz", 640)),
        conf=float(model_cfg.get("conf", 0.25)),
        iou=float(model_cfg.get("iou", 0.45)),
        half=bool(model_cfg.get("half", False)),
        classes=model_cfg.get("classes", []),
        tracker=str(model_cfg.get("tracker", "botsort")),
        camera_fps=int(camera_cfg.get("fps", 30)),
        inference_scale=float(model_cfg.get("inference_scale", 1.0)),
        jpeg_quality=int(model_cfg.get("jpeg_quality", 85)),
        output_dir=str(output_cfg.get("directory", "outputs")),
        save_snapshots=bool(output_cfg.get("save_snapshots", True)),
        view=bool(output_cfg.get("view", True)),
        rules=RulesConfig(
            running=running,
            fall=fall,
            crowd=crowd,
            intrusion=intrusion,
            fight=fight,
            vehicle=vehicle,
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
