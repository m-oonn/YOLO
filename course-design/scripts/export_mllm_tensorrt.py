#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI script for exporting MLLM models to ONNX and TensorRT."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export MLLM model to ONNX/TensorRT")
    parser.add_argument(
        "--model",
        default="qwen2-vl-2b",
        choices=["qwen2-vl-2b", "smolvlm-500m", "florence-2"],
        help="Model type to export",
    )
    parser.add_argument(
        "--backend",
        default="onnx",
        choices=["onnx", "tensorrt", "all"],
        help="Export backend",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TensorRT precision",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download model, skip export",
    )

    args = parser.parse_args()

    from core.mllm.export_utils import (
        build_tensorrt_engine,
        download_mllm_model,
        export_to_onnx,
    )

    logger.info(f"Processing model: {args.model}")

    model_dir = download_mllm_model(args.model, os.path.join(args.output_dir, "mllm"))

    if args.download_only:
        logger.info("Download complete (--download-only specified)")
        return

    if args.backend in ("onnx", "all"):
        onnx_path = os.path.join(args.output_dir, "onnx", f"{args.model}_vision.onnx")
        try:
            export_to_onnx(model_dir, onnx_path)
        except Exception as e:
            logger.warning(f"ONNX export skipped: {e}")

    if args.backend in ("tensorrt", "all"):
        onnx_path = os.path.join(args.output_dir, "onnx", f"{args.model}_vision.onnx")
        engine_path = os.path.join(
            args.output_dir,
            "trt_engines",
            f"{args.model}_vision_{args.precision}.engine",
        )
        try:
            build_tensorrt_engine(
                onnx_path,
                engine_path,
                precision=args.precision,
            )
        except Exception as e:
            logger.warning(f"TensorRT build skipped: {e}")

    logger.info("Export complete")


if __name__ == "__main__":
    main()
