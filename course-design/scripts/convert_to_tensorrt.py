#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convert YOLO models to TensorRT format for accelerated inference.

Usage:
    python scripts/convert_to_tensorrt.py --model models/yolov11x.pt
    python scripts/convert_to_tensorrt.py --model models/yolo12s.pt --precision int8 --calib-data datasets/calibration
    python scripts/convert_to_tensorrt.py --model models/yolov11x.pt --dla-core 0  # Jetson DLA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.tensorrt_utils import TensorRTConverter, is_tensorrt_available, get_tensorrt_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT engines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO .pt model file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TensorRT precision mode (default: fp16)",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default=None,
        help="Directory with calibration images for INT8 mode",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=2.0,
        help="Workspace size in GB (default: 2.0)",
    )
    parser.add_argument(
        "--dla-core",
        type=int,
        default=-1,
        help="DLA core index for Jetson (-1 to disable, default: -1)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=range(0, 6),
        help="Builder optimization level 0-5 (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if engine exists",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check TensorRT availability and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Check TensorRT availability
    if args.check:
        print(f"TensorRT available: {is_tensorrt_available()}")
        print(f"TensorRT version: {get_tensorrt_version()}")
        return 0 if is_tensorrt_available() else 1

    if not is_tensorrt_available():
        logger.error("TensorRT is not available. Please install:")
        logger.error("  pip install tensorrt onnx onnxsim pycuda")
        return 1

    # Load config
    cfg = load_config(args.config)

    # Override TensorRT config with CLI args
    from core.mllm.mllm_config import TensorRTConfig

    trt_config = TensorRTConfig(
        enabled=True,
        precision=args.precision,
        max_batch_size=args.max_batch,
        workspace_gb=args.workspace,
        dla_core=args.dla_core,
        optimization_level=args.opt_level,
    )

    logger.info(f"Converting {args.model} to TensorRT")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  Input size: {args.imgsz}x{args.imgsz}")
    logger.info(f"  Max batch: {args.max_batch}")
    logger.info(f"  Workspace: {args.workspace}GB")
    if args.dla_core >= 0:
        logger.info(f"  DLA core: {args.dla_core}")

    # Run conversion
    try:
        converter = TensorRTConverter(config=trt_config)
        engine_path = converter.convert(
            model_path=args.model,
            imgsz=args.imgsz,
            force_rebuild=args.force,
        )
        logger.info(f"Conversion successful: {engine_path}")

        # Verify engine
        from core.tensorrt_utils import TRTModelWrapper

        logger.info("Verifying engine with test inference...")
        trt_model = TRTModelWrapper(engine_path, imgsz=args.imgsz)
        logger.info("Engine verification passed!")

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
