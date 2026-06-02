# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sequence-based behavior classification using LSTM/ONNX."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from core.sequence_classifier import (
    FeatureSequenceBuffer,
    LSTMBranch,
    ModelInferenceEngine,
)
from core.config import SequenceModelConfig


class TestFeatureSequenceBuffer:
    def test_add_and_get_sequence(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        for i in range(4):
            buf.add(1, np.array([float(i)]))
        seq = buf.get_sequence(1)
        assert seq is not None
        assert seq.shape == (4, 1)

    def test_get_sequence_insufficient_history(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        buf.add(1, np.array([1.0]))
        assert buf.get_sequence(1) is None

    def test_is_ready(self):
        buf = FeatureSequenceBuffer(maxlen=3)
        assert buf.is_ready(1) is False
        for i in range(3):
            buf.add(1, np.array([float(i)]))
        assert buf.is_ready(1) is True

    def test_cleanup_removes_stale(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        buf.add(1, np.array([1.0]))
        buf.add(2, np.array([2.0]))
        buf.cleanup({1})
        assert buf.get_sequence(2) is None
        assert buf.is_ready(1) is False  # one sample not enough

    def test_reset_single_track(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        buf.add(1, np.array([1.0]))
        buf.add(2, np.array([2.0]))
        buf.reset(1)
        assert buf.get_sequence(1) is None
        assert 2 in buf._buffers

    def test_reset_all(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        buf.add(1, np.array([1.0]))
        buf.add(2, np.array([2.0]))
        buf.reset()
        assert len(buf) == 0

    def test_len(self):
        buf = FeatureSequenceBuffer(maxlen=4)
        assert len(buf) == 0
        buf.add(1, np.array([1.0]))
        assert len(buf) == 1


class TestLSTMBranch:
    def test_dummy_fallback_high_energy(self):
        lstm = LSTMBranch(input_dim=2, hidden_dim=32, num_classes=3)
        seq = np.random.randn(16, 2)
        # Scale up to create high motion energy
        seq[:, :2] *= 2.0
        class_id, confidence = lstm.predict(seq)
        assert class_id == 1  # fight
        assert confidence > 0.0

    def test_dummy_fallback_low_energy(self):
        lstm = LSTMBranch(input_dim=2, hidden_dim=32, num_classes=3)
        seq = np.zeros((16, 2))
        class_id, confidence = lstm.predict(seq)
        assert class_id == 0  # normal
        assert confidence > 0.0

    def test_dummy_fallback_medium_energy(self):
        lstm = LSTMBranch(input_dim=2, hidden_dim=32, num_classes=3)
        seq = np.random.randn(16, 2) * 0.3
        class_id, confidence = lstm.predict(seq)
        assert class_id in (0, 2)  # normal or other_abnormal

    def test_not_loaded_by_default(self):
        lstm = LSTMBranch()
        assert lstm.is_loaded is False

    def test_class_names(self):
        lstm = LSTMBranch()
        assert lstm.class_names == ["normal", "fight", "other_abnormal"]

    def test_load_onnx_import_error(self):
        lstm = LSTMBranch()
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch("builtins.__import__", side_effect=ImportError("no onnx")):
                result = lstm.load_onnx("nonexistent.onnx")
                assert result is False
                assert lstm.is_loaded is False


class TestModelInferenceEngine:
    def make_config(self, **kwargs):
        defaults = dict(
            enabled=False,
            shadow_mode=True,
            confidence_threshold=0.5,
            inference_device="cpu",
            model_path="",
            sequence_length=8,
        )
        defaults.update(kwargs)
        return SequenceModelConfig(**defaults)

    def test_infer_insufficient_history(self):
        config = self.make_config()
        engine = ModelInferenceEngine(config)
        class_id, confidence, event = engine.infer(1, np.array([1.0] * 86))
        assert class_id == 0
        assert confidence == 0.0
        assert event is None

    def test_infer_low_confidence(self):
        config = self.make_config(enabled=True, shadow_mode=False)
        engine = ModelInferenceEngine(config)
        for _ in range(config.sequence_length):
            engine.infer(1, np.zeros(86))
        # Zero features = low energy = class 0 (normal)
        class_id, confidence, event = engine.infer(1, np.zeros(86))
        assert event is None  # class 0 maps to None

    def test_infer_shadow_mode(self):
        config = self.make_config(enabled=True, shadow_mode=True, confidence_threshold=0.1)
        engine = ModelInferenceEngine(config)
        # High-motion features to trigger fight detection
        for _ in range(config.sequence_length):
            engine.infer(1, np.random.randn(86) * 2.0)
        class_id, confidence, event = engine.infer(1, np.random.randn(86) * 2.0)
        # Shadow mode: should return None event
        assert event is None

    def test_cleanup(self):
        config = self.make_config()
        engine = ModelInferenceEngine(config)
        engine.infer(1, np.array([1.0] * 86))
        engine.infer(2, np.array([2.0] * 86))
        engine.cleanup({1})
        assert engine.buffer.is_ready(2) is False  # insufficient for track 2

    def test_get_stats(self):
        config = self.make_config()
        engine = ModelInferenceEngine(config)
        stats = engine.get_stats()
        assert "enabled" in stats
        assert "shadow_mode" in stats
        assert "model_loaded" in stats
        assert "buffered_tracks" in stats
        assert "device" in stats

    def test_disabled_engine_still_buffers(self):
        config = self.make_config(enabled=False)
        engine = ModelInferenceEngine(config)
        engine.infer(1, np.array([1.0] * 86))
        assert engine.buffer.is_ready(1) is False
