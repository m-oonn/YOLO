# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""MLLM and TensorRT configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLLMConfig:
    enabled: bool = False
    model_type: str = "qwen2-vl-2b"
    model_path: str = "models/mllm/qwen2-vl-2b"
    inference_backend: str = "mock"
    device: str = "auto"
    half_precision: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.3
    key_frame_interval: int = 15
    context_window_frames: int = 5
    min_frame_size: int = 224
    max_frame_size: int = 512
    scene_description_enabled: bool = True
    scene_confidence_threshold: float = 0.5
    alarm_enhance_enabled: bool = True
    enhancement_cooldown_s: float = 10.0
    shadow_mode: bool = True


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
