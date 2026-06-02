# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU manager singleton and GPU detection logic."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from core.gpu_manager import GPUInfo, GPUManager, GPUPerformanceSnapshot


class TestGPUInfo:
    def test_default_values(self):
        info = GPUInfo()
        assert info.available is False
        assert info.name == ""
        assert info.total_memory_mb == 0.0


class TestGPUPerformanceSnapshot:
    def test_default_values(self):
        snap = GPUPerformanceSnapshot()
        assert snap.timestamp == 0.0
        assert snap.memory_allocated_mb == 0.0


class TestGPUManagerSingleton:
    def setup_method(self):
        GPUManager.reset()

    def test_singleton_pattern(self):
        g1 = GPUManager()
        g2 = GPUManager()
        assert g1 is g2

    def test_initial_state(self):
        manager = GPUManager()
        # Should always be in a valid state
        assert manager.device_name in ("cpu", "cuda:0", "mps")
        assert isinstance(manager.is_gpu_available, bool)
        assert isinstance(manager.is_cuda, bool)
        assert isinstance(manager.is_mps, bool)
        assert isinstance(manager.is_jetson, bool)

    def test_initialized_flag(self):
        manager = GPUManager()
        assert manager._initialized is True


class TestGPUManagerResolution:
    def setup_method(self):
        GPUManager.reset()

    def test_resolve_device_auto_returns_active_device(self):
        manager = GPUManager()
        assert manager.resolve_device("auto") == manager.device_name

    def test_resolve_device_custom_passthrough(self):
        manager = GPUManager()
        assert manager.resolve_device("cuda:0") == "cuda:0"
        assert manager.resolve_device("cpu") == "cpu"
        assert manager.resolve_device("mps") == "mps"

    def test_should_use_half_cpu_returns_false(self):
        manager = GPUManager()
        assert manager.should_use_half("cpu") is False

    def test_should_use_half_cuda_no_gpu(self):
        manager = GPUManager()
        with patch.object(manager._gpu_info, "supports_half", True):
            assert manager.should_use_half("cuda:0") is True


class TestGPUManagerCUDA:
    def setup_method(self):
        GPUManager.reset()

    def test_cuda_detection(self):
        # Create a mock torch with CUDA available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.name = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value.major = 8
        mock_torch.cuda.get_device_properties.return_value.minor = 0
        mock_torch.version.cuda = "12.1"

        GPUManager.reset()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-initialize will use mocked torch
            manager = GPUManager.__new__(GPUManager)
            manager._initialized = False
            manager.__init__()
            assert manager.device_name == "cuda:0"
            assert manager.is_gpu_available is True
            assert manager._gpu_info.name == "Test GPU"
        GPUManager.reset()

    def test_status_dict_keys(self):
        manager = GPUManager()
        status = manager.get_status_dict()
        expected_keys = {
            "gpu_available", "gpu_name", "gpu_total_memory_mb",
            "gpu_used_memory_mb", "gpu_reserved_memory_mb",
            "gpu_compute_capability", "cuda_version",
            "supports_half", "supports_tensor_cores", "device",
            "gpu_utilization_pct", "gpu_temperature_c", "gpu_power_w",
            "is_jetson", "jetson_model", "jetson_memory_type",
            "jetson_effective_vram_gb",
        }
        assert set(status.keys()) == expected_keys


class TestGPUManagerJetson:
    def setup_method(self):
        GPUManager.reset()

    def test_jetson_memory_type_dedicated_when_not_jetson(self):
        manager = GPUManager()
        assert manager.jetson_memory_type == "dedicated"

    def test_jetson_effective_vram_zero_when_not_jetson(self):
        manager = GPUManager()
        assert manager.jetson_effective_vram_gb == 0.0


class TestGPUManagerMPS:
    def setup_method(self):
        GPUManager.reset()

    def test_not_mps_on_cpu(self):
        manager = GPUManager()
        assert manager.is_mps is False


class TestGPUManagerProperties:
    def setup_method(self):
        GPUManager.reset()

    def test_gpu_info_property(self):
        manager = GPUManager()
        assert isinstance(manager.gpu_info, GPUInfo)

    def test_supports_half_precision(self):
        manager = GPUManager()
        assert isinstance(manager.supports_half_precision, bool)

    def test_supports_tensor_cores(self):
        manager = GPUManager()
        assert isinstance(manager.supports_tensor_cores, bool)


class TestGPUManagerPerformanceSnapshot:
    def setup_method(self):
        GPUManager.reset()

    def test_get_performance_snapshot(self):
        manager = GPUManager()
        snap = manager.get_performance_snapshot()
        assert isinstance(snap, GPUPerformanceSnapshot)
        assert isinstance(snap.timestamp, float)
