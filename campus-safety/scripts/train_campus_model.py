#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Train YOLO model on campus safety dataset.

This script fine-tunes a pre-trained YOLO model on the campus safety dataset
for optimized person detection in campus environments.

Usage:
    python scripts/train_campus_model.py --model yolo11n.pt --epochs 100
    python scripts/train_campus_model.py --model yolo11s.pt --epochs 50 --imgsz 640
    python scripts/train_campus_model.py --resume  # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on campus safety dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model to fine-tune (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/campus_safety.yaml",
        help="Dataset YAML config (default: datasets/campus_safety.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: cuda device index or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="models/training",
        help="Project directory for saving results (default: models/training)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="campus_safety",
        help="Experiment name (default: campus_safety)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N layers (default: 10)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (default: 0.01)",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (default: 1.0)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="Mixup augmentation probability (default: 0.1)",
    )
    parser.add_argument(
        "--hsv-h",
        type=float,
        default=0.015,
        help="HSV-Hue augmentation (default: 0.015)",
    )
    parser.add_argument(
        "--hsv-s",
        type=float,
        default=0.7,
        help="HSV-Saturation augmentation (default: 0.7)",
    )
    parser.add_argument(
        "--hsv-v",
        type=float,
        default=0.4,
        help="HSV-Value augmentation (default: 0.4)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs (default: 10)",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate training plots (default: True)",
    )
    return parser.parse_args()


def validate_dataset(data_path: str) -> bool:
    """Validate dataset configuration file exists."""
    if not os.path.exists(data_path):
        logger.error(f"Dataset config not found: {data_path}")
        logger.error("Please ensure the campus safety dataset is prepared.")
        return False
    return True


def validate_base_model(model_path: str) -> bool:
    """Validate base model exists or can be downloaded."""
    if os.path.exists(model_path):
        return True

    # Check if it's an official Ultralytics model that can be auto-downloaded
    official_models = [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    ]
    if model_path in official_models:
        logger.info(f"Base model {model_path} will be auto-downloaded by Ultralytics")
        return True

    logger.error(f"Base model not found: {model_path}")
    return False


def main() -> int:
    args = parse_args()

    # Validate inputs
    if not validate_dataset(args.data):
        return 1
    if not validate_base_model(args.model):
        return 1

    # Import ultralytics here to avoid slow startup if just checking args
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Please run: pip install ultralytics")
        return 1

    # Create project directory
    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Campus Safety YOLO Training")
    print("=" * 60)
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 60)

    # Load model
    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Training arguments
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "close_mosaic": args.close_mosaic,
        "save_period": args.save_period,
        "seed": args.seed,
        "plots": args.plots,
        "exist_ok": True,
    }

    if args.resume:
        train_args["resume"] = True
        logger.info("Resuming from last checkpoint")

    # Start training
    logger.info("Starting training...")
    try:
        results = model.train(**train_args)

        # Print final metrics
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Results saved to: {args.project}/{args.name}")
        print("=" * 60)

        # Export to ONNX for TensorRT conversion
        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        if best_model_path.exists():
            print("\nExporting best model to ONNX...")
            try:
                best_model = YOLO(str(best_model_path))
                best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
                print(f"ONNX export complete: {best_model_path.with_suffix('.onnx')}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
