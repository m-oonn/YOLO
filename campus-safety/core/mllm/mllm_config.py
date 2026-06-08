# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【MLLM子系统】mllm_config.py — MLLM配置数据类
# 依赖：仅标准库（dataclasses）
# 被调用：core/config.py（作为 AppConfig 的子配置）
# 核心职责：
#   ① MLLMConfig — 模型路径/后端/精度/温度等参数
#   ② TensorRTConfig — TensorRT加速配置（引擎目录/精度/批大小）
# 这两个配置统一由 configs/default.yaml 的 mllm: 段控制
# ──────────────────────────────────────────────────────────

"""MLLM and TensorRT configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MLLMConfig:
    enabled: bool = False
    model_type: str = "qwen2-vl-2b-instruct"
    model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    inference_backend: str = "auto"
    device: str = "auto"
    half_precision: bool = True
    use_flash_attention: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.2
    key_frame_interval: int = 30
    min_inference_interval_s: float = 3.0
    context_window_frames: int = 5
    min_frame_size: int = 224
    max_frame_size: int = 384
    scene_description_enabled: bool = True
    scene_confidence_threshold: float = 0.5
    alarm_enhance_enabled: bool = True
    enhancement_cooldown_s: float = 10.0
    shadow_mode: bool = False


@dataclass(frozen=True)
class TensorRTConfig:
    enabled: bool = False
    precision: str = "fp16"
    engine_dir: str = "models/trt_engines"
    onnx_dir: str = "models/onnx"
    max_batch_size: int = 1
    workspace_gb: float = 2.0
    dla_core: int = -1
    optimization_level: int = 3
