#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify MLLM lazy load/unload and config (Task 2 acceptance)."""

from __future__ import annotations

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)


def main() -> int:
    from core.mllm.mllm_config import MLLMConfig
    from core.mllm.mllm_sidecar import MLLMSidecar
    from core.mllm.inference_engine import MLLMInferenceEngine, MockVLMBackend

    print("=" * 60)
    print("Task 2 MLLM Verification")
    print("=" * 60)

    cfg = MLLMConfig(enabled=True, inference_backend="mock")
    assert cfg.model_path == "Qwen/Qwen2-VL-2B-Instruct", f"unexpected model: {cfg.model_path}"
    print(f"OK model_path = {cfg.model_path}")

    sidecar = MLLMSidecar(cfg)
    assert sidecar._engine is None, "engine should not load at init"
    print("OK lazy init: engine is None before initialize()")

    sidecar.initialize()
    assert sidecar._engine is not None and sidecar._engine.is_loaded
    print("OK mock model loaded on demand")

    sidecar.shutdown()
    assert sidecar._engine is None
    print("OK model fully unloaded after shutdown")

    engine = MLLMInferenceEngine(MLLMConfig(inference_backend="auto"))
    backend = engine._resolve_backend()
    assert backend.__class__.__name__ == "PyTorchVLMBackend"
    print("OK auto backend resolves to PyTorchVLMBackend")

    mock = MLLMInferenceEngine(MLLMConfig(inference_backend="mock"))
    mock.initialize()
    result = mock.generate("校园监控场景")
    assert "scene_summary" in result or result
    mock.shutdown()
    print("OK mock inference works")

    print("\nOverall: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
