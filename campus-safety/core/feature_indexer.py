# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】feature_indexer.py — CLIP图文检索索引
# 依赖：transformers（Chinese CLIP / OpenAI CLIP）
# 被调用：events_store.py（事件存储时提取特征）
# 核心职责：
#   ① 用中文 CLIP 模型（OFA-Sys）对事件快照提取特征向量
#   ② 支持文本搜索图片（如"打架"→找到打架事件的快照）
#   ③ 自动检测HF Mirror（国内网络加速下载）
#   ④ 多模型回退（Chinese CLIP → OpenAI CLIP）
# ──────────────────────────────────────────────────────────

"""CLIP-based feature indexing for text-to-image event search.

Supports both OpenAI CLIP and Chinese CLIP (OFA-Sys/chinese-clip-vit-base-patch16).
Chinese CLIP is preferred for Chinese-language queries.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Preferred models in order: Chinese CLIP for Chinese queries, OpenAI CLIP as fallback
_MODEL_CANDIDATES = [
    "OFA-Sys/chinese-clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
]


class FeatureIndexer:
    """Indexes event snapshot images with CLIP for natural-language search.

    Uses Chinese CLIP (OFA-Sys) by default for better Chinese-language matching.
    Falls back to OpenAI CLIP, then to SQLite LIKE text search.
    """

    _instance: FeatureIndexer | None = None

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._available = False
        self._load_attempted = False

    @classmethod
    def instance(cls) -> FeatureIndexer:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._available

    @property
    def model_name(self) -> str:
        return self._model_name or "unknown"

    def _try_load(self) -> None:
        self._load_attempted = True

        # Set HF Mirror for China network if not already set
        if not os.environ.get("HF_ENDPOINT"):
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from transformers import CLIPModel, CLIPProcessor

        candidates = [self._model_name] if self._model_name else _MODEL_CANDIDATES
        candidates = [c for c in candidates if c]

        for name in candidates:
            try:
                self._model = CLIPModel.from_pretrained(name)
                self._processor = CLIPProcessor.from_pretrained(name)
                self._model_name = name
                self._available = True
                logger.info("CLIP model loaded: %s", name)
                return
            except Exception as e:
                logger.debug("CLIP model %s not available: %s", name, e)

        logger.warning("No CLIP model available, falling back to text search")

    def encode_image(self, image_path: str) -> np.ndarray | None:
        """Encode an image file to a 512-d feature vector."""
        if not self.available or not os.path.isfile(image_path):
            return None
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            outputs = self._model.get_image_features(**inputs)
            vec = outputs.detach().cpu().numpy().flatten()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)
        except Exception as e:
            logger.debug("CLIP image encoding failed: %s", e)
            return None

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode a query text to a 512-d feature vector."""
        if not self.available:
            return None
        try:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            outputs = self._model.get_text_features(**inputs)
            vec = outputs.detach().cpu().numpy().flatten()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)
        except Exception as e:
            logger.debug("CLIP text encoding failed: %s", e)
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def search(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Rank candidates by CLIP cosine similarity to query_text.

        Each candidate dict must have either ``feature_blob`` (bytes of
        float32 array) or ``snapshot_path`` (path to image file).
        Results are returned sorted by descending similarity.
        """
        query_vec = self.encode_text(query_text)
        if query_vec is None:
            return candidates[:top_k]

        scored: list[tuple[float, dict[str, Any]]] = []
        for c in candidates:
            cand_vec = None
            if c.get("feature_blob"):
                try:
                    cand_vec = np.frombuffer(c["feature_blob"], dtype=np.float32)
                except Exception:
                    pass
            if cand_vec is None and c.get("snapshot_path"):
                cand_vec = self.encode_image(c["snapshot_path"])
            if cand_vec is not None:
                scored.append((self.cosine_similarity(query_vec, cand_vec), c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def vector_to_blob(self, vec: np.ndarray) -> bytes:
        return vec.tobytes()

    def blob_to_vector(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)
