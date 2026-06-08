#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""YOLO12 Model Upgrade Script

This script helps download and configure YOLO12 models,
providing a seamless upgrade from YOLOv8/YOLO11.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

COURSE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COURSE_DIR))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


YOLO12_MODELS = {
    "yolo12n": {"name": "YOLO12-Nano", "description": "Smallest and fastest", "params": "~5.4M"},
    "yolo12s": {"name": "YOLO12-Small", "description": "Best balance (RECOMMENDED)", "params": "~9.3M"},
    "yolo12m": {"name": "YOLO12-Medium", "description": "Higher accuracy", "params": "~20.1M"},
    "yolo12l": {"name": "YOLO12-Large", "description": "High accuracy", "params": "~28.7M"},
    "yolo12x": {"name": "YOLO12-XLarge", "description": "Highest accuracy", "params": "~61.8M"},
}


def download_model(model_name: str, models_dir: Path) -> bool:
    """Download YOLO model using ultralytics library."""
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / f"{model_name}.pt"

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        response = input(f"\n⚠️  Model exists: {output_path} ({size_mb:.1f} MB)\n   Re-download? (y/N): ")
        if response.lower() != 'y':
            logger.info(f"Using existing model")
            return True

    logger.info(f"\nDownloading {model_name}...")

    try:
        from ultralytics import YOLO
        import shutil
        home = Path.home()

        model = YOLO(model_name + ".pt")

        cache_locations = [
            home / ".cache" / "ultralytics" / f"{model_name}.pt",
            home / ".cache" / "ultralytics" / f"{model_name.replace('-', '')}.pt",
        ]

        for src in cache_locations:
            if src.exists():
                shutil.copy(src, output_path)
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Saved: {output_path} ({size_mb:.1f} MB)")
                return True

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Ready: {output_path} ({size_mb:.1f} MB)")
            return True

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def update_config(model_name: str) -> bool:
    """Update configuration to use new model."""
    config_path = COURSE_DIR / "configs" / "default.yaml"

    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        old = config.get('model', {}).get('path', 'unknown')
        if 'model' not in config:
            config['model'] = {}
        config['model']['path'] = f"models/{model_name}.pt"

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ Config updated: {old} -> models/{model_name}.pt")
        return True

    except Exception as e:
        logger.error(f"Config update failed: {e}")
        return False


def benchmark_model(model_path: str, num_runs: int = 50) -> Optional[dict]:
    """Quick benchmark of a model."""
    try:
        import time
        import numpy as np
        from ultralytics import YOLO

        logger.info(f"Benchmarking {model_path}...")

        model = YOLO(model_path)
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for _ in range(5):
            _ = model.predict(dummy, verbose=False)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.predict(dummy, verbose=False)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        return {"mean_ms": times.mean(), "std_ms": times.std(), "fps": 1000 / times.mean()}

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO12 Model Upgrade Tool")
    parser.add_argument("--list", "-l", action="store_true", help="Show available models")
    parser.add_argument("--model", "-m", choices=list(YOLO12_MODELS.keys()), help="Model to download")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    parser.add_argument("--update-config", "-c", action="store_true", help="Update config")
    parser.add_argument("--models-dir", default="models", help="Models directory")

    args = parser.parse_args()

    if args.list or not args.model:
        print("\nAvailable YOLO12 Models:")
        print("-" * 50)
        for key, info in YOLO12_MODELS.items():
            rec = " ⭐" if key == "yolo12s" else ""
            print(f"  {key:<10} - {info['name']}{rec}")
            print(f"            {info['description']} ({info['params']})")
            print()
        if not args.model:
            return 0

    models_dir = COURSE_DIR / args.models_dir
    if download_model(args.model, models_dir):
        print(f"\n✅ {args.model} downloaded!")

        if args.benchmark:
            result = benchmark_model(str(models_dir / f"{args.model}.pt"))
            if result:
                print(f"\n📊 Results: {result['mean_ms']:.2f}ms, {result['fps']:.1f} FPS")

        if args.update_config:
            update_config(args.model)

        print("\n🎉 Done! Update configs/default.yaml manually if needed.")
    else:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
