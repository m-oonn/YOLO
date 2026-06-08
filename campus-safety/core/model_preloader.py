# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Singleton YOLO model preloader — load once at app startup, reuse on detection start."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelPreloader:
    """Thread-safe singleton that preloads and warms the main YOLO model."""

    _instance: ModelPreloader | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> ModelPreloader:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_constructed", False):
            return
        self._constructed = True
        self._lock = threading.RLock()
        self._model: Any = None
        self._model_path: str | None = None
        self._device: str = "cpu"
        self._half: bool = False
        self._imgsz: int = 640
        self._ready = threading.Event()
        self._loading = False
        self._error: str | None = None
        self._timings: dict[str, float] = {}

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._model is not None

    @property
    def timings(self) -> dict[str, float]:
        with self._lock:
            return dict(self._timings)

    def start_background(self, config_path: str | None = None) -> None:
        """Start model loading on a daemon thread (non-blocking app startup)."""
        with self._lock:
            if self._loading or self.is_ready:
                return
            self._loading = True

        def _load() -> None:
            try:
                self.load_sync(config_path)
            except Exception as e:
                with self._lock:
                    self._error = str(e)
                logger.exception("Background model preload failed: %s", e)
                self._ready.set()

        threading.Thread(target=_load, daemon=True, name="ModelPreloader").start()

    def load_sync(self, config_path: str | None = None) -> Any:
        """Load and warm up the YOLO model synchronously. Returns the model instance."""
        from core.config import load_config
        from core.gpu_manager import GPUManager
        from ultralytics import YOLO
        import torch

        cfg = load_config(config_path)
        model_path = cfg.model_path

        with self._lock:
            if self._model is not None and self._model_path == model_path:
                return self._model

        t_total = time.perf_counter()
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        gpu_mgr = GPUManager()
        device = gpu_mgr.resolve_device(cfg.device)
        half = cfg.half or gpu_mgr.should_use_half(device)
        timings["device_resolve_ms"] = (time.perf_counter() - t0) * 1000

        if device.startswith("cuda"):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        t0 = time.perf_counter()
        model = YOLO(model_path)
        model.eval()
        timings["model_load_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dummy = np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8)
        with torch.no_grad():
            for _ in range(2):
                model.predict(
                    dummy,
                    imgsz=cfg.imgsz,
                    device=device,
                    half=half,
                    verbose=False,
                )
        timings["warmup_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        with self._lock:
            self._model = model
            self._model_path = model_path
            self._device = device
            self._half = half
            self._imgsz = cfg.imgsz
            self._timings = timings
            self._loading = False
            self._error = None

        self._ready.set()
        logger.info(
            "Model preloaded: path=%s device=%s half=%s "
            "(load=%.0fms warmup=%.0fms total=%.0fms)",
            model_path,
            device,
            half,
            timings["model_load_ms"],
            timings["warmup_ms"],
            timings["total_ms"],
        )
        return model

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the model is ready or timeout. Returns True if ready."""
        return self._ready.wait(timeout=timeout) and self._model is not None

    def get_model(self, model_path: str | None = None) -> Any | None:
        """Return the preloaded model if path matches (or any if path is None)."""
        with self._lock:
            if self._model is None:
                return None
            if model_path is not None and self._model_path != model_path:
                return None
            return self._model

    def get_device_info(self) -> tuple[str, bool, int]:
        """Return (device, half, imgsz) from preload."""
        with self._lock:
            return self._device, self._half, self._imgsz

    def release(self) -> None:
        """Release the preloaded model and free GPU memory."""
        with self._lock:
            if self._model is not None:
                try:
                    self._model = None
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            self._model_path = None
            self._ready.clear()
            self._loading = False


def get_model_preloader() -> ModelPreloader:
    return ModelPreloader()
