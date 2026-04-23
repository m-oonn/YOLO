#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Command-line detection runner for testing without the web interface."""

import argparse
import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import logging
import time

import cv2

from core.config import AppConfig, load_config
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection pipeline")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: camera index (0) or file path")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Configuration file path")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save output to video file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = DetectionPipeline(cfg)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    pipeline.start()
    print(f"Detection started. Source: {args.source}")
    print("Press 'q' to quit.")

    try:
        while pipeline.running:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()
            result = pipeline.process_frame(frame, timestamp)
            annotated = pipeline.annotate_frame(frame, pipeline._last_detections)

            if video_writer:
                video_writer.write(annotated)

            if cfg.view:
                try:
                    cv2.imshow("YOLO Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    logger.warning("No display available, disabling view mode")
                    cfg_dict = {
                        "model_path": cfg.model_path, "device": cfg.device,
                        "imgsz": cfg.imgsz, "conf": cfg.conf, "iou": cfg.iou,
                        "classes": cfg.classes, "camera_fps": cfg.camera_fps,
                        "output_dir": cfg.output_dir,
                        "save_snapshots": cfg.save_snapshots, "view": False,
                        "rules": cfg.rules,
                    }
                    cfg = AppConfig(**cfg_dict)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        pipeline.close()
        cv2.destroyAllWindows()
        print(f"Detection stopped. Processed {pipeline._frame_count} frames.")


if __name__ == "__main__":
    main()
