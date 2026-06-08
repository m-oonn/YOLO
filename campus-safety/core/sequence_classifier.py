# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】sequence_classifier.py — LSTM时序行为分类
# 依赖：ONNX Runtime
# 被调用：behavior_analyzer.py（可选，基于骨架序列分类）
# 核心职责：
#   ① 加载 ONNX 格式的 LSTM 模型（行为分类）
#   ② 输入：连续16帧的骨架特征序列
#   ③ 输出：行为类别 + 置信度
# 当前状态：实验性功能，默认禁用（shadow_mode=True）
# ──────────────────────────────────────────────────────────

"""Lightweight sequence-based behavior classification using LSTM/ONNX."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from .config import SequenceModelConfig

logger = logging.getLogger(__name__)


class FeatureSequenceBuffer:
    """Maintains a sliding window of feature vectors for sequence classification."""

    def __init__(self, maxlen: int = 16):
        self.maxlen = maxlen
        self._buffers: dict[int, deque[np.ndarray]] = {}

    def add(self, track_id: int, feature_vector: np.ndarray) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.maxlen)
        self._buffers[track_id].append(feature_vector)

    def get_sequence(self, track_id: int) -> np.ndarray | None:
        """Get the feature sequence as a fixed-size array [T, D].

        Returns None if insufficient history.
        """
        buf = self._buffers.get(track_id)
        if buf is None or len(buf) < self.maxlen:
            return None
        return np.array(buf)  # [T, D]

    def is_ready(self, track_id: int) -> bool:
        return track_id in self._buffers and len(self._buffers[track_id]) >= self.maxlen

    def cleanup(self, active_tracks: set[int]) -> None:
        stale = set(self._buffers.keys()) - active_tracks
        for tid in stale:
            del self._buffers[tid]

    def reset(self, track_id: int | None = None) -> None:
        if track_id is not None:
            self._buffers.pop(track_id, None)
        else:
            self._buffers.clear()

    def __len__(self) -> int:
        return len(self._buffers)


class LSTMBranch:
    """Lightweight LSTM classifier for behavior sequence analysis.

    Designed for ONNX Runtime deployment with minimal parameters (<50K).
    Architecture: TemporalConv → LSTM(64→32) → FC(32→16→3)
    """

    def __init__(self, input_dim: int = 86, hidden_dim: int = 32,
                 num_classes: int = 3, num_layers: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Dummy model state (in production, loaded from ONNX)
        self._loaded = False

        # Class labels
        self.class_names = ["normal", "fight", "other_abnormal"]

    def load_onnx(self, model_path: str) -> bool:
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if self._device == 'cuda'
                else ['CPUExecutionProvider']
            )
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._loaded = True
            logger.info("LSTM model loaded from %s (providers: %s)", model_path, providers)
            return True
        except ImportError:
            logger.warning("onnxruntime not installed, using dummy classifier")
            return False
        except Exception as e:
            logger.warning("Failed to load ONNX model: %s", e)
            return False

    def predict(self, sequence: np.ndarray) -> tuple[int, float]:
        """Predict behavior class from feature sequence.

        Args:
            sequence: Feature sequence array [T, D].

        Returns:
            Tuple of (class_id, confidence).
        """
        if self._loaded:
            try:
                input_name = self._session.get_inputs()[0].name
                probs = self._session.run(None, {input_name: sequence[np.newaxis, ...].astype(np.float32)})[0]
                class_id = int(np.argmax(probs[0]))
                confidence = float(probs[0][class_id])
                return class_id, confidence
            except Exception as e:
                logger.warning("ONNX inference error: %s", e)

        # Dummy fallback: use simple energy-based heuristic
        motion_energy = float(np.mean(np.abs(np.diff(sequence[:, :2], axis=0))))
        if motion_energy > 0.5:
            return 1, min(motion_energy, 0.9)  # fight
        elif motion_energy > 0.2:
            return 2, motion_energy  # other abnormal
        return 0, 1.0 - motion_energy  # normal

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class ModelInferenceEngine:
    """Inference engine that manages model lifecycle and shadow mode."""

    def __init__(self, config: SequenceModelConfig):
        self.config = config
        self.enabled = config.enabled
        self.shadow_mode = config.shadow_mode
        self.confidence_threshold = config.confidence_threshold
        self.device = config.inference_device

        self.classifier = LSTMBranch()
        self.buffer = FeatureSequenceBuffer(maxlen=config.sequence_length)

        if self.enabled and config.model_path:
            self.classifier.load_onnx(config.model_path)

    def infer(self, track_id: int, features: np.ndarray
              ) -> tuple[int, float, str | None]:
        """Run inference on feature sequence for a track.

        Args:
            track_id: Target track ID.
            features: Current frame feature vector [D].

        Returns:
            Tuple of (class_id, confidence, event_type or None).
        """
        self.buffer.add(track_id, features)
        if not self.buffer.is_ready(track_id):
            return 0, 0.0, None

        sequence = self.buffer.get_sequence(track_id)
        if sequence is None:
            return 0, 0.0, None

        class_id, confidence = self.classifier.predict(sequence)

        if class_id == 0 or confidence < self.confidence_threshold:
            return class_id, confidence, None

        # Map class to event type
        event_map = {1: "fight", 2: "running"}
        event_type = event_map.get(class_id)

        if event_type and self.shadow_mode:
            logger.info("Shadow mode: detected %s (conf=%.2f) for track %d",
                        event_type, confidence, track_id)
            return class_id, confidence, None  # Don't trigger in shadow mode

        return class_id, confidence, event_type

    def cleanup(self, active_tracks: set[int]) -> None:
        self.buffer.cleanup(active_tracks)

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "shadow_mode": self.shadow_mode,
            "model_loaded": self.classifier.is_loaded,
            "buffered_tracks": len(self.buffer),
            "device": self.device,
        }
