#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Setup script for NVIDIA Jetson deployment.

This script configures the system for optimal performance on Jetson devices:
- Sets maximum performance mode
- Converts models to TensorRT with Jetson-optimized settings
- Configures environment variables
- Verifies DLA availability

Usage:
    sudo python scripts/setup_jetson.py --model models/yolov11x.pt
    sudo python scripts/setup_jetson.py --model models/yolov11x.pt --dla-core 0
    python scripts/setup_jetson.py --check
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.jetson_utils import get_jetson_manager, is_jetson, setup_jetson_environment
from core.tensorrt_utils import TensorRTConverter, is_tensorrt_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup and optimize for NVIDIA Jetson deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model to convert to TensorRT",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--dla-core",
        type=int,
        default=-1,
        help="DLA core to use (-1 for GPU only, 0/1 for DLA)",
    )
    parser.add_argument(
        "--max-performance",
        action="store_true",
        default=True,
        help="Set Jetson to maximum performance mode (default: True)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Jetson environment and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Check if running on Jetson
    if not is_jetson():
        logger.error("This script is designed for NVIDIA Jetson devices only.")
        logger.error("Current platform does not appear to be a Jetson.")
        return 1

    # Setup environment
    setup_jetson_environment()

    # Get Jetson manager
    jm = get_jetson_manager()
    info = jm.info

    print("=" * 60)
    print("NVIDIA Jetson Setup")
    print("=" * 60)
    print(f"Model: {info.model}")
    print(f"SoC: {info.soc}")
    print(f"CUDA Architecture: {info.cuda_arch}")
    print(f"Total Memory: {info.total_memory_mb} MB")
    print(f"DLA Cores: {info.num_dla_cores}")
    print(f"Power Mode: {jm.get_power_mode()}")
    print(f"JetPack: {jm.get_jetpack_version()}")
    print("=" * 60)

    if args.check:
        print(f"\nTensorRT Available: {is_tensorrt_available()}")
        print(f"Temperatures: {jm.get_temperature()}")
        print(f"Throttling: {jm.is_throttling()}")
        return 0

    # Set maximum performance
    if args.max_performance:
        print("\nSetting maximum performance mode...")
        if jm.set_max_performance():
            print("Maximum performance mode enabled")
        else:
            print("Warning: Could not set maximum performance mode")
            print("You may need to run with sudo")

    # Convert model to TensorRT if specified
    if args.model:
        if not is_tensorrt_available():
            logger.error("TensorRT is not available. Please install TensorRT for Jetson.")
            return 1

        print(f"\nConverting model to TensorRT: {args.model}")

        # Get Jetson-optimized config
        trt_config = jm.get_tensorrt_config()
        print(f"Jetson-optimized config: {trt_config}")

        # Override DLA core if specified
        if args.dla_core >= 0 and info.has_dla:
            trt_config["dla_core"] = args.dla_core
            print(f"Using DLA core: {args.dla_core}")

        from core.mllm.mllm_config import TensorRTConfig

        config = TensorRTConfig(
            enabled=True,
            precision=trt_config.get("precision", "fp16"),
            max_batch_size=trt_config.get("max_batch_size", 1),
            workspace_gb=trt_config.get("workspace_mb", 512) / 1024,
            dla_core=trt_config.get("dla_core", -1),
            optimization_level=3,
        )

        try:
            converter = TensorRTConverter(config=config)
            engine_path = converter.convert(
                model_path=args.model,
                imgsz=args.imgsz,
                force_rebuild=False,
            )
            print(f"\nConversion successful!")
            print(f"Engine saved to: {engine_path}")

            # Verify engine
            from core.tensorrt_utils import TRTModelWrapper

            print("\nVerifying engine...")
            trt_model = TRTModelWrapper(engine_path, imgsz=args.imgsz)
            print("Engine verification passed!")

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return 1

    print("\n" + "=" * 60)
    print("Jetson setup complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
