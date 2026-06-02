# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Jetson platform utilities and optimizations.

This module provides:
- Jetson device detection and identification
- Jetson-specific GPU memory management (unified memory)
- DLA (Deep Learning Accelerator) configuration
- Jetson power mode detection and recommendations
- Jetson-optimized TensorRT settings

Supported Jetson platforms:
- Jetson Nano
- Jetson TX2 / TX2 NX
- Jetson Xavier NX
- Jetson AGX Xavier
- Jetson Orin NX / Orin Nano / AGX Orin
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JetsonInfo:
    """Jetson device information."""
    is_jetson: bool = False
    model: str = ""
    soc: str = ""  # System on Chip (e.g., Tegra X1, Xavier, Orin)
    cuda_arch: str = ""  # CUDA compute capability
    max_gpu_clock_mhz: int = 0
    has_dla: bool = False
    num_dla_cores: int = 0
    unified_memory: bool = True
    total_memory_mb: int = 0
    recommended_batch_size: int = 1
    recommended_precision: str = "fp16"
    recommended_workspace_mb: int = 512


class JetsonManager:
    """Manages Jetson-specific configurations and optimizations."""

    # Jetson SoC to CUDA architecture mapping
    JETSON_SOC_MAP: dict[str, str] = {
        "tegra210": "5.3",   # Jetson Nano (Maxwell)
        "tegra186": "6.2",   # Jetson TX2 (Pascal)
        "tegra194": "7.2",   # Jetson Xavier (Volta)
        "tegra234": "8.7",   # Jetson Orin (Ampere)
    }

    # Jetson model detection patterns
    JETSON_MODEL_PATTERNS: dict[str, dict[str, Any]] = {
        "nano": {
            "soc": "tegra210",
            "cuda_arch": "5.3",
            "has_dla": False,
            "num_dla_cores": 0,
            "max_gpu_clock": 921,
            "recommended_precision": "fp16",
            "recommended_workspace_mb": 256,
        },
        "tx2": {
            "soc": "tegra186",
            "cuda_arch": "6.2",
            "has_dla": False,
            "num_dla_cores": 0,
            "max_gpu_clock": 1300,
            "recommended_precision": "fp16",
            "recommended_workspace_mb": 512,
        },
        "xavier": {
            "soc": "tegra194",
            "cuda_arch": "7.2",
            "has_dla": True,
            "num_dla_cores": 2,
            "max_gpu_clock": 1377,
            "recommended_precision": "fp16",
            "recommended_workspace_mb": 1024,
        },
        "orin": {
            "soc": "tegra234",
            "cuda_arch": "8.7",
            "has_dla": True,
            "num_dla_cores": 2,
            "max_gpu_clock": 1300,
            "recommended_precision": "fp16",
            "recommended_workspace_mb": 2048,
        },
    }

    def __init__(self):
        self.info = self._detect_jetson()

    def _detect_jetson(self) -> JetsonInfo:
        """Detect Jetson device and return information."""
        info = JetsonInfo()

        # Check device tree model
        model_path = "/proc/device-tree/model"
        if not os.path.exists(model_path):
            return info

        try:
            with open(model_path, "r") as f:
                model_str = f.read().strip().rstrip("\x00")
        except Exception:
            return info

        if "jetson" not in model_str.lower() and "tegra" not in model_str.lower():
            return info

        info.is_jetson = True
        info.model = model_str
        info.unified_memory = True

        # Identify specific model
        model_lower = model_str.lower()
        for pattern, specs in self.JETSON_MODEL_PATTERNS.items():
            if pattern in model_lower:
                info.soc = specs["soc"]
                info.cuda_arch = specs["cuda_arch"]
                info.has_dla = specs["has_dla"]
                info.num_dla_cores = specs["num_dla_cores"]
                info.max_gpu_clock_mhz = specs["max_gpu_clock"]
                info.recommended_precision = specs["recommended_precision"]
                info.recommended_workspace_mb = specs["recommended_workspace_mb"]
                break

        # Get total memory
        info.total_memory_mb = self._get_total_memory_mb()

        # Adjust recommendations based on memory
        if info.total_memory_mb > 0:
            if info.total_memory_mb < 4096:  # 4GB
                info.recommended_batch_size = 1
                info.recommended_workspace_mb = min(info.recommended_workspace_mb, 256)
            elif info.total_memory_mb < 8192:  # 8GB
                info.recommended_batch_size = 1
                info.recommended_workspace_mb = min(info.recommended_workspace_mb, 512)
            else:  # 16GB+
                info.recommended_batch_size = 2
                info.recommended_workspace_mb = min(info.recommended_workspace_mb, 1024)

        logger.info(
            f"Jetson detected: {info.model} (SoC: {info.soc}, "
            f"CUDA arch: {info.cuda_arch}, Memory: {info.total_memory_mb}MB, "
            f"DLA: {info.num_dla_cores} cores)"
        )

        return info

    def _get_total_memory_mb(self) -> int:
        """Get total system memory in MB."""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
        return 0

    def get_power_mode(self) -> str:
        """Get current Jetson power mode."""
        if not self.info.is_jetson:
            return ""
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "NV Power Mode" in line:
                        return line.split(":")[-1].strip()
        except Exception:
            pass
        return "unknown"

    def set_max_performance(self) -> bool:
        """Set Jetson to maximum performance mode.

        Returns:
            True if successful
        """
        if not self.info.is_jetson:
            return False
        try:
            # Set maximum power mode
            subprocess.run(
                ["nvpmodel", "-m", "0"],
                capture_output=True,
                timeout=10,
                check=True,
            )
            # Enable jetson_clocks for maximum performance
            subprocess.run(
                ["jetson_clocks"],
                capture_output=True,
                timeout=30,
            )
            logger.info("Jetson set to maximum performance mode")
            return True
        except Exception as e:
            logger.warning(f"Could not set max performance: {e}")
            return False

    def get_tensorrt_config(self) -> dict[str, Any]:
        """Get Jetson-optimized TensorRT configuration."""
        if not self.info.is_jetson:
            return {}

        config = {
            "precision": self.info.recommended_precision,
            "max_batch_size": self.info.recommended_batch_size,
            "workspace_mb": self.info.recommended_workspace_mb,
            "use_dla": self.info.has_dla,
            "dla_core": 0 if self.info.has_dla else -1,
        }

        # Jetson Nano/TX2: use smaller workspace
        if self.info.soc in ("tegra210", "tegra186"):
            config["precision"] = "fp16"
            config["workspace_mb"] = 256

        return config

    def get_jetpack_version(self) -> str:
        """Get JetPack version."""
        if not self.info.is_jetson:
            return ""
        try:
            # Try reading from apt list
            result = subprocess.run(
                ["dpkg-query", "--showformat=${Version}', "--show", "nvidia-jetpack"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Alternative: check L4T version
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                return f.read().strip()
        except Exception:
            pass

        return "unknown"

    def get_temperature(self) -> dict[str, float]:
        """Get Jetson thermal zone temperatures."""
        temps = {}
        if not self.info.is_jetson:
            return temps

        try:
            thermal_path = "/sys/class/thermal"
            if os.path.exists(thermal_path):
                for zone in os.listdir(thermal_path):
                    if zone.startswith("thermal_zone"):
                        zone_path = os.path.join(thermal_path, zone)
                        try:
                            with open(os.path.join(zone_path, "type"), "r") as f:
                                zone_type = f.read().strip()
                            with open(os.path.join(zone_path, "temp"), "r") as f:
                                temp_millidegrees = int(f.read().strip())
                                temps[zone_type] = temp_millidegrees / 1000.0
                        except Exception:
                            pass
        except Exception:
            pass

        return temps

    def is_throttling(self) -> bool:
        """Check if Jetson is thermal throttling."""
        temps = self.get_temperature()
        for zone, temp in temps.items():
            if "AO" in zone or "PLL" in zone:
                continue
            if temp > 80:  # 80°C threshold
                logger.warning(f"Jetson thermal warning: {zone} = {temp:.1f}°C")
                if temp > 95:
                    return True
        return False

    def get_status_dict(self) -> dict[str, Any]:
        """Get complete Jetson status for API response."""
        if not self.info.is_jetson:
            return {"is_jetson": False}

        return {
            "is_jetson": True,
            "model": self.info.model,
            "soc": self.info.soc,
            "cuda_arch": self.info.cuda_arch,
            "total_memory_mb": self.info.total_memory_mb,
            "has_dla": self.info.has_dla,
            "num_dla_cores": self.info.num_dla_cores,
            "power_mode": self.get_power_mode(),
            "jetpack_version": self.get_jetpack_version(),
            "temperatures": self.get_temperature(),
            "is_throttling": self.is_throttling(),
            "recommended_config": {
                "precision": self.info.recommended_precision,
                "batch_size": self.info.recommended_batch_size,
                "workspace_mb": self.info.recommended_workspace_mb,
            },
        }


def is_jetson() -> bool:
    """Quick check if running on Jetson."""
    model_path = "/proc/device-tree/model"
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "r") as f:
            model = f.read().lower()
        return "jetson" in model or "tegra" in model
    except Exception:
        return False


def get_jetson_manager() -> JetsonManager:
    """Get or create JetsonManager singleton."""
    if not hasattr(get_jetson_manager, "_instance"):
        get_jetson_manager._instance = JetsonManager()
    return get_jetson_manager._instance


def setup_jetson_environment() -> None:
    """Setup environment variables optimized for Jetson.

    Call this before importing PyTorch/TensorRT on Jetson.
    """
    if not is_jetson():
        return

    # CUDA settings
    os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # PyTorch settings for Jetson
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # TensorRT settings
    os.environ.setdefault("TRT_DEPRECATED_GIE_LIBRARY", "1")

    # OpenCV settings
    os.environ.setdefault("OPENCV_CUDA", "1")

    logger.info("Jetson environment variables configured")
