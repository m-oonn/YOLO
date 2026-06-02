# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for archives API and TensorRT VLM backend."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ── Archives API ────────────────────────────────────────────────

class TestArchivesAPI:
    def test_list_empty_when_no_pipeline(self):
        with patch("backend.api.archives._get_recorder", return_value=None):
            from backend.api.archives import list_clips
            result = list_clips()
            assert result.total == 0

    def test_list_with_params(self):
        mock = MagicMock()
        mock.get_clips.return_value = []
        mock.get_clip_count.return_value = 0
        with patch("backend.api.archives._get_recorder", return_value=mock):
            from backend.api.archives import list_clips
            result = list_clips(event_type="fall", limit=10, offset=5)
            mock.get_clips.assert_called_once_with(event_type="fall", limit=10, offset=5)
            assert result.total == 0

    def test_delete_success(self):
        mock = MagicMock()
        mock.delete_clip.return_value = True
        with patch("backend.api.archives._get_recorder", return_value=mock):
            from backend.api.archives import delete_clip
            assert delete_clip("c1")["status"] == "deleted"

    def test_delete_not_found(self):
        mock = MagicMock()
        mock.delete_clip.return_value = False
        with patch("backend.api.archives._get_recorder", return_value=mock):
            from backend.api.archives import delete_clip
            from fastapi import HTTPException
            with pytest.raises(HTTPException):
                delete_clip("no")


# ── TensorRT backend ────────────────────────────────────────────

class TestTensorRTCapabilities:
    def test_has_torch_compile(self):
        from core.mllm.inference_engine import TensorRTVLMBackend
        assert TensorRTVLMBackend._has_torch_compile()

    def test_has_torch_tensorrt_default_false(self):
        from core.mllm.inference_engine import TensorRTVLMBackend
        assert not TensorRTVLMBackend._has_torch_tensorrt()

    def test_has_onnxruntime_trt_default_false(self):
        from core.mllm.inference_engine import TensorRTVLMBackend
        assert not TensorRTVLMBackend._has_onnxruntime_trt()


class TestTensorRTInit:
    def test_attrs(self):
        from core.mllm.inference_engine import TensorRTVLMBackend
        from core.mllm.mllm_config import MLLMConfig
        b = TensorRTVLMBackend(MLLMConfig())
        assert b.backend_name.startswith("tensorrt")
        assert not b.is_loaded

    def test_find_engine_none(self):
        from core.mllm.inference_engine import TensorRTVLMBackend
        with tempfile.TemporaryDirectory() as d:
            import os as _os
            cwd = _os.getcwd()
            try:
                _os.chdir(d)
                assert TensorRTVLMBackend._find_trt_engine() is None
            finally:
                _os.chdir(cwd)


class TestMockVLM:
    def test_generate(self):
        from core.mllm.inference_engine import MockVLMBackend
        from core.mllm.mllm_config import MLLMConfig
        b = MockVLMBackend(MLLMConfig())
        b.load()
        r = b.generate("test")
        assert "scene_summary" in r or "verdict" in r
        r2 = b.generate("告警验证 test")
        assert "verdict" in r2
        b.unload()
        assert not b.is_loaded


class TestInferenceEngine:
    def test_resolve_mock(self):
        from core.mllm.inference_engine import MLLMInferenceEngine, MockVLMBackend
        from core.mllm.mllm_config import MLLMConfig
        e = MLLMInferenceEngine(MLLMConfig(inference_backend="mock"))
        assert isinstance(e._resolve_backend(), MockVLMBackend)

    def test_resolve_tensorrt(self):
        from core.mllm.inference_engine import MLLMInferenceEngine, TensorRTVLMBackend
        from core.mllm.mllm_config import MLLMConfig
        e = MLLMInferenceEngine(MLLMConfig(inference_backend="tensorrt"))
        assert isinstance(e._resolve_backend(), TensorRTVLMBackend)

    def test_unloaded_generate(self):
        from core.mllm.inference_engine import MLLMInferenceEngine
        from core.mllm.mllm_config import MLLMConfig
        assert MLLMInferenceEngine(MLLMConfig(inference_backend="mock")).generate("x") == ""

    def test_init_mock(self):
        from core.mllm.inference_engine import MLLMInferenceEngine
        from core.mllm.mllm_config import MLLMConfig
        e = MLLMInferenceEngine(MLLMConfig(inference_backend="mock"))
        e.initialize()
        assert e._backend and e._backend.is_loaded
        assert e.get_stats()["backend"] == "mock"
        e.shutdown()

    def test_stats_no_backend(self):
        from core.mllm.inference_engine import MLLMInferenceEngine
        from core.mllm.mllm_config import MLLMConfig
        s = MLLMInferenceEngine(MLLMConfig()).get_stats()
        assert s["backend"] == "none"
        assert not s["loaded"]


# ── DetectionManager loading state ──────────────────────────────

class TestDetectionManagerLoading:
    def test_status_returns_loading_state(self):
        from backend.detection_manager import DetectionManager
        mgr = DetectionManager()
        mgr._detection_active = True
        mgr._pipeline = None
        mgr._current_source = "0"
        status = mgr.get_status()
        assert status["running"] is True
        assert status["state"] == "loading"
        assert status["fps"] == 0

    def test_status_returns_error(self):
        from backend.detection_manager import DetectionManager
        mgr = DetectionManager()
        mgr._detection_active = False
        mgr._pipeline = None
        mgr._error_count = 1
        mgr._last_error = "camera open failed"
        status = mgr.get_status()
        assert status["running"] is False
        assert status["last_error"] == "camera open failed"
        assert status["error_count"] == 1
