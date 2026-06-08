# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】model_registry.py — 模型文件查找与路径解析
# 依赖：仅标准库（pathlib）
# 被调用：pipeline.py, download_models.py
# 核心职责：
#   ① 在 models/ 目录中查找 .pt 模型文件
#   ② 解析 HuggingFace 缓存目录（models--org--repo/snapshots/）
#   ③ 列出所有已下载模型（find_yolo_model / find_hf_model）
#   ④ 按优先级解析模型（如先找专用模型，回退到通用模型）
# ──────────────────────────────────────────────────────────

"""Model registry for managing downloaded detection models.

Resolves model paths from the models/ directory, supporting both
HuggingFace-cached and directly-downloaded models.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Finds and resolves model files in the project's models/ directory."""

    def __init__(self, models_root: str | Path = "models"):
        self._root = Path(models_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def find_yolo_model(self, name: str) -> str | None:
        """Find a YOLO .pt model file by name. Returns absolute path or None."""
        candidates = [
            self._root / name,
            self._root / f"{name}.pt",
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        return None

    def find_hf_model(self, repo_id: str) -> str | None:
        """Find a HuggingFace-cached model directory by repo ID. Returns path or None."""
        cache_dir = self._root / f"models--{repo_id.replace('/', '--')}"
        if cache_dir.is_dir():
            # Find the snapshots subdirectory
            snapshots = cache_dir / "snapshots"
            if snapshots.is_dir():
                dirs = list(snapshots.iterdir())
                if dirs:
                    return str(dirs[0].resolve())
        return None

    def find_clip_model(self) -> str | None:
        """Find any available CLIP model (Chinese first, then OpenAI)."""
        for repo in [
            "OFA-Sys/chinese-clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
        ]:
            path = self.find_hf_model(repo)
            if path:
                return path
        return None

    def list_downloaded_models(self) -> dict[str, str]:
        """List all downloaded models with their paths."""
        found: dict[str, str] = {}

        # Check for YOLO models
        for pattern in ["*.pt"]:
            for p in self._root.glob(pattern):
                found[p.name] = str(p.resolve())

        # Check for HF models
        for d in self._root.iterdir():
            if d.is_dir() and d.name.startswith("models--"):
                repo = d.name[8:].replace("--", "/")
                snapshots = d / "snapshots"
                if snapshots.is_dir():
                    contents = list(snapshots.iterdir())
                    if contents:
                        found[f"HF:{repo}"] = str(contents[0].resolve())

        return found

    def resolve_vehicle_model(self) -> str | None:
        """Find the best available vehicle detection model."""
        # Priority: dedicated vehicle model > COCO pretrained
        for candidate in [
            "vehicle_detection_yolov10",  # from HF
            "yolo12s.pt",  # COCO pretrained (80-class)
            "yolov8n.pt",
        ]:
            path = self.find_yolo_model(candidate)
            if path:
                return path
        return None

    def resolve_fight_model(self) -> str | None:
        """Find the best available fight detection model."""
        for candidate in [
            "fight_detection_yolov8",  # from HF (Musawer14)
            "fight_yolov8",  # existing pretrained
            "suspicious_nano",  # existing nano model
        ]:
            path = self.find_yolo_model(candidate)
            if path:
                return path
        # Check HF cache
        hf = self.find_hf_model("Musawer14/fight_detection_yolov8")
        if hf:
            # Look for .pt files inside
            for p in Path(hf).glob("*.pt"):
                return str(p)
        return None

    def resolve_pose_model(self) -> str | None:
        """Find the best available pose estimation model."""
        for candidate in [
            "yolo11s-pose.pt",
            "yolo11n-pose.pt",
            "yolov8n-pose.pt",
        ]:
            path = self.find_yolo_model(candidate)
            if path:
                return path
        return None
