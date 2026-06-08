# ──────────────────────────────────────────────────────────
# 【核心引擎】gpu_manager.py — GPU内存管理
# 依赖：torch
# 被调用：pipeline.py（定期清理GPU缓存）
# 核心职责：
#   ① 监控GPU显存使用率
#   ② 到达阈值时自动清理 torch CUDA 缓存
#   ③ 避免长时间运行导致显存溢出（OOM）
# ──────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    available: bool = False
    name: str = ""
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    cuda_version: str = ""
    supports_half: bool = False
    supports_tensor_cores: bool = False
    device_index: int = 0
    is_jetson: bool = False
    jetson_model: str = ""


@dataclass
class GPUPerformanceSnapshot:
    timestamp: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_usage_w: float = 0.0


import threading

class GPUManager:
    _instance: GPUManager | None = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> GPUManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
        self._gpu_info = GPUInfo()
        self._torch_available = False
        self._cuda_available = False
        self._device_name = "cpu"
        self._is_jetson = False
        self._jetson_model = ""
        self._detect_jetson()
        self._detect_gpu()

    def _detect_jetson(self) -> None:
        try:
            model_path = "/proc/device-tree/model"
            if os.path.exists(model_path):
                with open(model_path, "r") as f:
                    model_str = f.read().strip().rstrip("\x00")
                if "jetson" in model_str.lower() or "tegra" in model_str.lower():
                    self._is_jetson = True
                    self._jetson_model = model_str
                    logger.info(f"Jetson device detected: {model_str}")
        except Exception:
            pass

    def _detect_gpu(self) -> None:
        try:
            import torch
            self._torch_available = True
            if torch.cuda.is_available():
                self._cuda_available = True
                props = torch.cuda.get_device_properties(0)
                total_mem = getattr(props, "total_memory", None) or getattr(props, "total_global_mem", None) or getattr(props, "total_mem", 0)
                self._gpu_info = GPUInfo(
                    available=True,
                    name=props.name,
                    total_memory_mb=round(total_mem / (1024**2), 1),
                    compute_capability=(props.major, props.minor),
                    supports_half=True,
                    supports_tensor_cores=props.major >= 7,
                    device_index=0,
                    is_jetson=self._is_jetson,
                    jetson_model=self._jetson_model,
                )
                self._device_name = "cuda:0"
                try:
                    self._gpu_info.cuda_version = torch.version.cuda or ""
                except Exception:
                    pass
                logger.info(
                    f"GPU detected: {props.name} "
                    f"({self._gpu_info.total_memory_mb:.0f}MB, "
                    f"CC {props.major}.{props.minor}, "
                    f"CUDA {self._gpu_info.cuda_version})"
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._gpu_info = GPUInfo(
                    available=True,
                    name="Apple MPS",
                    supports_half=False,
                    supports_tensor_cores=False,
                )
                self._device_name = "mps"
                logger.info("Apple Metal Performance Shaders (MPS) detected")
            else:
                logger.info("No GPU detected, using CPU")
        except ImportError:
            logger.info("PyTorch not installed, CPU-only mode")
        except Exception as e:
            logger.warning(f"GPU detection error: {e}")

    @property
    def gpu_info(self) -> GPUInfo:
        return self._gpu_info

    @property
    def is_gpu_available(self) -> bool:
        return self._gpu_info.available

    @property
    def is_cuda(self) -> bool:
        return self._cuda_available

    @property
    def is_mps(self) -> bool:
        return self._device_name == "mps"

    @property
    def is_jetson(self) -> bool:
        return self._is_jetson

    @property
    def jetson_model(self) -> str:
        return self._jetson_model

    @property
    def jetson_memory_type(self) -> str:
        if self._is_jetson:
            return "unified"
        return "dedicated"

    @property
    def jetson_effective_vram_gb(self) -> float:
        if self._is_jetson and self._gpu_info.total_memory_mb > 0:
            return round(self._gpu_info.total_memory_mb / 1024 * 0.7, 1)
        return 0.0

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def supports_half_precision(self) -> bool:
        return self._gpu_info.supports_half

    @property
    def supports_tensor_cores(self) -> bool:
        return self._gpu_info.supports_tensor_cores

    def resolve_device(self, config_device: str = "auto") -> str:
        if config_device == "auto":
            return self._device_name
        if config_device.startswith("cuda") and not self._cuda_available:
            logger.warning("CUDA requested (%s) but not available, falling back to CPU", config_device)
            return "cpu"
        if config_device == "mps" and self._device_name != "mps":
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return config_device

    def should_use_half(self, device: str) -> bool:
        if device.startswith("cuda"):
            return self._gpu_info.supports_half
        return False

    def get_performance_snapshot(self) -> GPUPerformanceSnapshot:
        snap = GPUPerformanceSnapshot()
        if not self._cuda_available:
            return snap
        try:
            import torch
            import time as _time
            snap.timestamp = _time.time()
            snap.memory_allocated_mb = round(
                torch.cuda.memory_allocated(0) / (1024**2), 1
            )
            snap.memory_reserved_mb = round(
                torch.cuda.memory_reserved(0) / (1024**2), 1
            )
            total_mem = getattr(torch.cuda.get_device_properties(0), "total_memory", None) or getattr(torch.cuda.get_device_properties(0), "total_global_mem", None) or getattr(torch.cuda.get_device_properties(0), "total_mem", 1)
            snap.gpu_utilization_pct = round(snap.memory_allocated_mb / (total_mem / (1024**2)) * 100, 1) if total_mem > 0 else 0
        except Exception:
            pass

        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                snap.gpu_utilization_pct = util.gpu
            except Exception:
                pass
            try:
                snap.gpu_temperature_c = nvmlDeviceGetTemperature(handle, 0)
            except Exception:
                pass
            try:
                snap.gpu_power_usage_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            pass

        return snap

    def optimize_memory(self) -> None:
        if not self._cuda_available:
            return
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("GPU cache emptied and IPC collected")
        except Exception:
            pass

    def get_memory_pressure(self) -> str:
        """Return GPU memory pressure level: 'low', 'medium', 'high', 'critical'.

        Uses nvidia-smi for accurate total VRAM usage (includes other processes
        like Ollama, WeChat, etc.) rather than just PyTorch-allocated memory.
        """
        if not self._cuda_available:
            return "low"

        total_mb = self._gpu_info.total_memory_mb
        if total_mb <= 0:
            return "low"

        used_mb = self._get_total_gpu_memory_used_mb()
        if used_mb <= 0:
            alloc_mb = 0
            try:
                import torch
                alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
            except Exception:
                pass
            used_mb = alloc_mb

        pct = (used_mb / total_mb) * 100
        if pct >= 90:
            return "critical"
        if pct >= 75:
            return "high"
        if pct >= 55:
            return "medium"
        return "low"

    def should_release_model(self) -> bool:
        """Decide whether the YOLO model should be released from GPU memory.

        Returns True when GPU memory pressure is high enough that keeping
        the model resident would risk OOM or cause significant slowdowns.
        """
        pressure = self.get_memory_pressure()
        if pressure in ("critical", "high"):
            logger.warning(
                "GPU memory pressure is %s — recommending model release", pressure
            )
            return True
        return False

    def _get_total_gpu_memory_used_mb(self) -> float:
        """Get total GPU memory used by ALL processes via nvidia-smi.

        This is more accurate than torch.cuda.memory_allocated() which only
        tracks PyTorch allocations. On a shared GPU (e.g. with Ollama running),
        the actual usage can be much higher.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().split("\n")[0].strip())
        except Exception:
            pass
        return 0.0

    def get_status_dict(self) -> dict:
        info = self._gpu_info
        perf = self.get_performance_snapshot()
        pressure = self.get_memory_pressure()
        total_used_mb = self._get_total_gpu_memory_used_mb()
        return {
            "gpu_available": info.available,
            "gpu_name": info.name,
            "gpu_total_memory_mb": info.total_memory_mb,
            "gpu_used_memory_mb": perf.memory_allocated_mb,
            "gpu_total_used_mb": round(total_used_mb, 1),
            "gpu_reserved_memory_mb": perf.memory_reserved_mb,
            "gpu_memory_pressure": pressure,
            "gpu_memory_usage_pct": round((total_used_mb / info.total_memory_mb) * 100, 1) if info.total_memory_mb > 0 and total_used_mb > 0 else 0,
            "gpu_compute_capability": f"{info.compute_capability[0]}.{info.compute_capability[1]}",
            "cuda_version": info.cuda_version,
            "supports_half": info.supports_half,
            "supports_tensor_cores": info.supports_tensor_cores,
            "device": self._device_name,
            "gpu_utilization_pct": perf.gpu_utilization_pct,
            "gpu_temperature_c": perf.gpu_temperature_c,
            "gpu_power_w": perf.gpu_power_usage_w,
            "is_jetson": self._is_jetson,
            "jetson_model": self._jetson_model,
            "jetson_memory_type": self.jetson_memory_type,
            "jetson_effective_vram_gb": self.jetson_effective_vram_gb,
        }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._initialized = False
