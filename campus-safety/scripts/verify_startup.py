#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify detection startup performance (Task 1 acceptance).

Measures:
  1. Model preloader timing
  2. DetectionPipeline init with/without preloaded model
  3. First-frame inference latency (synthetic frame)

Usage:
    cd campus-safety
    python scripts/verify_startup.py
    python scripts/verify_startup.py --quick   # skip real model if unavailable
"""

from __future__ import annotations

import argparse
import os
import sys
import time

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def verify_preloader(config_path: str | None) -> dict:
    from core.model_preloader import get_model_preloader

    preloader = get_model_preloader()
    preloader.release()

    t0 = time.perf_counter()
    try:
        preloader.load_sync(config_path)
        elapsed = time.perf_counter() - t0
        ok = preloader.is_ready
        return {
            "ok": ok,
            "elapsed_s": round(elapsed, 3),
            "timings": preloader.timings,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "timings": {},
            "error": str(e),
        }


def verify_pipeline_init(config_path: str | None, use_preload: bool) -> dict:
    from core.config import load_config
    from core.model_preloader import get_model_preloader
    from core.pipeline import DetectionPipeline
    from unittest.mock import patch
    import numpy as np

    cfg = load_config(config_path)
    if not os.path.exists(cfg.model_path):
        return {"ok": False, "skipped": True, "reason": f"model not found: {cfg.model_path}"}

    preloaded = None
    skip_warmup = False
    if use_preload:
        preloader = get_model_preloader()
        if not preloader.is_ready:
            preloader.load_sync(config_path)
        preloaded = preloader.get_model(cfg.model_path)
        skip_warmup = preloaded is not None

    t0 = time.perf_counter()
    try:
        pipeline = DetectionPipeline(
            cfg,
            preloaded_model=preloaded,
            skip_warmup=skip_warmup,
        )
        init_ms = (time.perf_counter() - t0) * 1000

        # First frame
        t1 = time.perf_counter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.start()
        pipeline.process_frame(frame, time.time())
        first_frame_ms = (time.perf_counter() - t1) * 1000
        pipeline.stop()
        pipeline.close()

        return {
            "ok": True,
            "skipped": False,
            "use_preload": use_preload,
            "init_ms": round(init_ms, 1),
            "first_frame_ms": round(first_frame_ms, 1),
            "total_ms": round(init_ms + first_frame_ms, 1),
        }
    except Exception as e:
        return {
            "ok": False,
            "skipped": False,
            "use_preload": use_preload,
            "error": str(e),
        }


def verify_mock_pipeline() -> dict:
    """Fast unit-style check without real weights."""
    from core.config import AppConfig, RulesConfig, RunningRule, FallRule, CrowdRule, IntrusionRule, FightRule
    from core.pipeline import DetectionPipeline
    from unittest.mock import MagicMock, patch
    import numpy as np

    cfg = AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(),
            fall=FallRule(),
            crowd=CrowdRule(),
            intrusion=IntrusionRule(),
            fight=FightRule(),
        ),
    )
    t0 = time.perf_counter()
    with patch("core.pipeline.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_instance.track.return_value = []
        mock_yolo.return_value = mock_instance
        pipeline = DetectionPipeline(cfg, preloaded_model=mock_instance, skip_warmup=True)
        pipeline.start()
        pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), 1.0)
        pipeline.stop()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"ok": True, "mock_init_and_frame_ms": round(elapsed_ms, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify detection startup performance")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--quick", action="store_true", help="Only run mock test")
    parser.add_argument("--target-s", type=float, default=3.0, help="Target startup seconds")
    args = parser.parse_args()

    _banner("Task 1 Startup Verification")
    results: dict = {"target_s": args.target_s, "passed": True}

    mock = verify_mock_pipeline()
    results["mock"] = mock
    print(f"Mock pipeline init+frame: {mock['mock_init_and_frame_ms']}ms — OK")

    if args.quick:
        print("\nQuick mode: skipping real model tests.")
        print(json_dumps(results))
        return 0

    _banner("Model Preloader")
    preload = verify_preloader(args.config)
    results["preloader"] = preload
    if preload["ok"]:
        print(f"Preload total: {preload['elapsed_s']}s")
        for k, v in preload.get("timings", {}).items():
            print(f"  {k}: {v:.1f}ms")
    else:
        print(f"Preload FAILED: {preload.get('error')}")
        results["passed"] = False

    _banner("Pipeline Init (no preload)")
    cold = verify_pipeline_init(args.config, use_preload=False)
    results["cold_start"] = cold
    if cold.get("skipped"):
        print(f"SKIPPED: {cold.get('reason')}")
    elif cold.get("ok"):
        print(f"Init: {cold['init_ms']}ms, first frame: {cold['first_frame_ms']}ms, total: {cold['total_ms']}ms")
    else:
        print(f"FAILED: {cold.get('error')}")
        results["passed"] = False

    _banner("Pipeline Init (with preload)")
    warm = verify_pipeline_init(args.config, use_preload=True)
    results["warm_start"] = warm
    if warm.get("skipped"):
        print(f"SKIPPED: {warm.get('reason')}")
    elif warm.get("ok"):
        total_s = warm["total_ms"] / 1000
        print(f"Init: {warm['init_ms']}ms, first frame: {warm['first_frame_ms']}ms, total: {warm['total_ms']}ms")
        if total_s > args.target_s:
            print(f"WARNING: warm start {total_s:.2f}s exceeds target {args.target_s}s")
            results["passed"] = False
        else:
            print(f"PASS: warm start within {args.target_s}s target")
    else:
        print(f"FAILED: {warm.get('error')}")
        results["passed"] = False

    _banner("Summary")
    status = "PASS" if results["passed"] else "FAIL"
    print(f"Overall: {status}")
    return 0 if results["passed"] else 1


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
