#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""One-click model download script for all required detection models.

Downloads models from HuggingFace (via hf-mirror.com for China networks),
ModelScope, or Ultralytics. Run before first launch:

    python scripts/download_models.py          # download all models
    python scripts/download_models.py --list   # list available models
    python scripts/download_models.py --only clip,pose  # download specific models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ModelScope IDs for models also available on ModelScope (faster in China)
_MODELSCOPE_MAP = {
    "OFA-Sys/chinese-clip-vit-base-patch16": "damo/multi-modal_clip-vit-base-patch16_zh",
}

MODELS = {
    "chinese-clip": {
        "name": "Chinese CLIP (text-image retrieval, Chinese-optimized)",
        "source": "modelscope",
        "repo": "damo/multi-modal_clip-vit-base-patch16_zh",
        "hf_repo": "OFA-Sys/chinese-clip-vit-base-patch16",
        "size": "~400MB",
        "required": True,
    },
    "clip": {
        "name": "OpenAI CLIP (English fallback text-image)",
        "source": "huggingface",
        "repo": "openai/clip-vit-base-patch32",
        "size": "~600MB",
        "required": False,
    },
    "yolo-pose": {
        "name": "YOLO11n-Pose (skeleton keypoints)",
        "source": "ultralytics",
        "repo": "yolo11n-pose.pt",
        "size": "~6MB",
        "required": True,
    },
    "vehicle-yolo": {
        "name": "Vehicle Detection YOLOv10 (optional, COCO covers this)",
        "source": "huggingface",
        "repo": "hanungaddi/vehicle_detection_yolov10",
        "size": "~6MB",
        "required": False,
    },
    "fight-yolo": {
        "name": "Fight Detection YOLOv8 (optional, pretrained covers this)",
        "source": "huggingface",
        "repo": "Musawer14/fight_detection_yolov8",
        "size": "~6MB",
        "required": False,
    },
}


def download_huggingface(repo: str) -> str | None:
    """Download from HuggingFace (via HF Mirror for China)."""
    try:
        from huggingface_hub import snapshot_download

        print(f"  Downloading {repo}...")
        path = snapshot_download(repo, cache_dir=str(MODELS_DIR))
        print(f"  -> {path}")

        # Copy .pt files to pretrained for YOLO models
        import shutil
        pt_dir = MODELS_DIR / "pretrained"
        pt_dir.mkdir(exist_ok=True)
        for f in Path(path).glob("*.pt"):
            dest = pt_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f"  -> copied {f.name} to models/pretrained/")

        return path
    except Exception as e:
        print(f"  HF download failed: {e}")
        return None


def download_modelscope(repo: str) -> str | None:
    """Download from ModelScope (China domestic)."""
    try:
        from modelscope import snapshot_download

        print(f"  Downloading from ModelScope: {repo}...")
        path = snapshot_download(repo, cache_dir=str(MODELS_DIR / "modelscope"))
        print(f"  -> {path}")
        return path
    except Exception as e:
        print(f"  ModelScope download failed: {e}")
        return None


def download_ultralytics(repo: str) -> str | None:
    """Download YOLO model via Ultralytics."""
    try:
        from ultralytics import YOLO

        dest = MODELS_DIR / repo
        if dest.exists():
            print(f"  Already exists: {dest}")
            return str(dest)

        print(f"  Downloading {repo}...")
        model = YOLO(repo)
        dest = MODELS_DIR / repo
        print(f"  -> {dest}")
        return str(dest)
    except Exception as e:
        print(f"  Ultralytics download failed: {e}")
        return None


def download_model(key: str, info: dict) -> bool:
    """Download a single model. Returns True if successful."""
    print(f"\n[{key}] {info['name']} ({info['size']})")

    source = info["source"]
    repo = info["repo"]

    if source == "modelscope":
        result = download_modelscope(repo)
        if result is None:
            # Fallback to HuggingFace
            hf_repo = info.get("hf_repo")
            if hf_repo:
                print("  ModelScope failed, trying HF Mirror fallback...")
                result = download_huggingface(hf_repo)
        return result is not None

    elif source == "huggingface":
        result = download_huggingface(repo)
        if result is None:
            # Try ModelScope if mapped
            ms_repo = _MODELSCOPE_MAP.get(repo)
            if ms_repo:
                print("  Trying ModelScope fallback...")
                result = download_modelscope(ms_repo)
        return result is not None

    elif source == "ultralytics":
        return download_ultralytics(repo) is not None

    return False


def list_models():
    """Print all available models."""
    print("Available models:")
    for key, info in MODELS.items():
        tag = "[required]" if info["required"] else "[optional]"
        print(f"  {key:20s} {info['size']:>8s}  {tag}")
        print(f"  {'':20s} {info['name']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download detection models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated model keys to download (e.g. 'clip,pose')",
    )
    parser.add_argument(
        "--skip-optional", action="store_true", help="Skip optional models"
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    keys = [k for k in MODELS]
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        invalid = set(keys) - set(MODELS)
        if invalid:
            print(f"Unknown models: {invalid}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)

    if args.skip_optional:
        keys = [k for k in keys if MODELS[k]["required"]]

    print(f"Downloading {len(keys)} model(s) to {MODELS_DIR}")
    print("=" * 50)

    ok = 0
    fail = 0
    for key in keys:
        if download_model(key, MODELS[key]):
            ok += 1
        else:
            fail += 1

    print()
    print("=" * 50)
    print(f"Done: {ok} downloaded, {fail} failed")

    if fail > 0 and not os.environ.get("HF_ENDPOINT"):
        print("\nTip: For China networks, set the HF mirror:")
        print('  set HF_ENDPOINT=https://hf-mirror.com')


if __name__ == "__main__":
    main()
