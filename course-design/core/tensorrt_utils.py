# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""TensorRT model conversion and inference utilities for YOLO acceleration.

This module provides:
- ONNX export from PyTorch YOLO models
- TensorRT engine building with configurable precision (FP32/FP16/INT8)
- TensorRT inference wrapper compatible with Ultralytics YOLO API
- Automatic engine caching and reuse

Requirements:
    pip install tensorrt onnx onnxsim onnxruntime-gpu

For INT8 calibration:
    pip install pycuda
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Lazy imports for TensorRT to avoid hard dependency
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not installed. GPU acceleration will use PyTorch only.")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


@dataclass
class TRTInferenceResult:
    """Unified result format matching Ultralytics YOLO output."""
    boxes: np.ndarray | None = None
    scores: np.ndarray | None = None
    classes: np.ndarray | None = None
    masks: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    orig_img: np.ndarray | None = None
    orig_shape: tuple[int, int] | None = None
    path: str = ""


# Only define TensorRT-dependent classes when TensorRT is available
if TRT_AVAILABLE:
    class TensorRTLogger(trt.ILogger):
        """TensorRT logger that maps to Python logging."""

        def __init__(self, level: int = trt.Logger.WARNING):
            super().__init__()
            self.level = level

        def log(self, severity: trt.Logger.Severity, msg: str) -> None:
            if severity == trt.Logger.INTERNAL_ERROR:
                logger.error(f"[TRT] {msg}")
            elif severity == trt.Logger.ERROR:
                logger.error(f"[TRT] {msg}")
            elif severity == trt.Logger.WARNING:
                logger.warning(f"[TRT] {msg}")
            elif severity == trt.Logger.INFO:
                logger.info(f"[TRT] {msg}")
            else:
                logger.debug(f"[TRT] {msg}")

    class TensorRTInferenceSession:
        """TensorRT inference session for YOLO models.

        Handles engine loading/creation, memory management, and inference.
        Compatible with the Ultralytics YOLO detection pipeline.
        """

        def __init__(
            self,
            engine_path: str,
            input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
            device_id: int = 0,
        ):
            if not TRT_AVAILABLE:
                raise RuntimeError("TensorRT is not installed")
            if not PYCUDA_AVAILABLE:
                raise RuntimeError("pycuda is required for TensorRT inference")

            self.engine_path = engine_path
            self.input_shape = input_shape
            self.device_id = device_id
            self.logger = TensorRTLogger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)

            self.engine: trt.ICudaEngine | None = None
            self.context: trt.IExecutionContext | None = None
            self.stream: Any = cuda.Stream()

            # CUDA buffers
            self.inputs: list[dict] = []
            self.outputs: list[dict] = []
            self.bindings: list[int] = []

            self._load_or_build_engine()

        def _load_or_build_engine(self) -> None:
            """Load existing engine or build from ONNX."""
            if os.path.exists(self.engine_path):
                logger.info(f"Loading TensorRT engine: {self.engine_path}")
                with open(self.engine_path, "rb") as f:
                    self.engine = self.runtime.deserialize_cuda_engine(f.read())
            else:
                raise FileNotFoundError(
                    f"TensorRT engine not found: {self.engine_path}. "
                    "Run model conversion first."
                )

            self.context = self.engine.create_execution_context()
            self._allocate_buffers()

        def _allocate_buffers(self) -> None:
            """Allocate GPU memory for inputs and outputs."""
            self.inputs = []
            self.outputs = []
            self.bindings = []

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize

                # Allocate device memory
                device_mem = cuda.mem_alloc(size)
                self.bindings.append(int(device_mem))

                buffer = {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "size": size,
                    "device": device_mem,
                }

                if mode == trt.TensorIOMode.INPUT:
                    self.inputs.append(buffer)
                else:
                    self.outputs.append(buffer)

            logger.info(
                f"TensorRT buffers allocated: {len(self.inputs)} inputs, "
                f"{len(self.outputs)} outputs"
            )

        def infer(self, image: np.ndarray) -> list[np.ndarray]:
            """Run inference on a preprocessed image.

            Args:
                image: NCHW formatted float32 array, normalized to [0, 1]

            Returns:
                List of output arrays matching the model outputs
            """
            if self.context is None:
                raise RuntimeError("TensorRT context not initialized")

            # Copy input to GPU
            cuda.memcpy_htod_async(
                self.inputs[0]["device"], image.ravel(), self.stream
            )

            # Set input shape (for dynamic batch)
            self.context.set_input_shape(
                self.inputs[0]["name"], self.inputs[0]["shape"]
            )

            # Execute
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Copy outputs from GPU
            outputs = []
            for out in self.outputs:
                host_mem = np.empty(out["shape"], dtype=out["dtype"])
                cuda.memcpy_dtoh_async(host_mem, out["device"], self.stream)
                outputs.append(host_mem)

            self.stream.synchronize()
            return outputs

        def __del__(self):
            """Cleanup CUDA resources."""
            try:
                if hasattr(self, "stream") and self.stream:
                    self.stream.synchronize()
            except Exception:
                pass

    class YOLOInt8Calibrator(trt.IInt8EntropyCalibrator2):
        """INT8 calibrator for YOLO models using entropy calibration."""

        def __init__(self, image_paths: list[str], batch_size: int = 1):
            super().__init__()
            self.image_paths = image_paths
            self.batch_size = batch_size
            self.current_index = 0
            self.cache_file = "int8_calibration.cache"

            # Allocate device memory for batch
            self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self.current_index >= len(self.image_paths):
                return None

            batch_images = []
            for i in range(self.batch_size):
                idx = self.current_index + i
                if idx >= len(self.image_paths):
                    break
                img = self._preprocess(self.image_paths[idx])
                batch_images.append(img)

            if not batch_images:
                return None

            batch = np.stack(batch_images, axis=0)
            cuda.memcpy_htod(self.device_input, batch.ravel())
            self.current_index += self.batch_size
            return [int(self.device_input)]

        def _preprocess(self, image_path: str) -> np.ndarray:
            import cv2
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            return img

        def read_calibration_cache(self) -> bytes | None:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
else:
    # Placeholder classes when TensorRT is not available
    class TensorRTLogger:
        pass

    class TensorRTInferenceSession:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorRT is not installed")

    class YOLOInt8Calibrator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorRT is not installed")


def export_yolo_to_onnx(
    model_path: str,
    output_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """Export Ultralytics YOLO model to ONNX format.

    Args:
        model_path: Path to .pt model file
        output_path: Path for output .onnx file
        imgsz: Input image size
        opset: ONNX opset version
        simplify: Whether to simplify ONNX graph with onnxsim
        dynamic: Whether to enable dynamic batch/axes

    Returns:
        Path to exported ONNX file
    """
    from ultralytics import YOLO

    logger.info(f"Exporting {model_path} to ONNX...")
    model = YOLO(model_path)

    export_args = {
        "format": "onnx",
        "imgsz": imgsz,
        "opset": opset,
        "simplify": simplify,
        "dynamic": dynamic,
    }

    # Ultralytics export
    model.export(**export_args)

    # Find the exported file
    default_onnx = model_path.replace(".pt", ".onnx")
    if os.path.exists(default_onnx):
        if output_path != default_onnx:
            os.rename(default_onnx, output_path)
        logger.info(f"ONNX export complete: {output_path}")
        return output_path

    raise RuntimeError(f"ONNX export failed: {default_onnx} not found")


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_gb: float = 2.0,
    dla_core: int = -1,
    optimization_level: int = 3,
    calibration_images: list[str] | None = None,
) -> str:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for .engine file
        precision: "fp32", "fp16", or "int8"
        max_batch_size: Maximum batch size
        workspace_gb: Workspace size in GB
        dla_core: DLA core index (-1 to disable)
        optimization_level: Builder optimization level (0-5)
        calibration_images: Image paths for INT8 calibration

    Returns:
        Path to built engine file
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed")
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx is required for engine building")

    logger.info(f"Building TensorRT engine: precision={precision}, batch={max_batch_size}")

    logger_obj = TensorRTLogger(trt.Logger.INFO)
    builder = trt.Builder(logger_obj)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger_obj)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            for err in errors:
                logger.error(f"ONNX parse error: {err}")
            raise RuntimeError("ONNX parsing failed")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    config.builder_optimization_level = optimization_level

    # Precision configuration
    precision = precision.lower()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 mode enabled")
        if calibration_images:
            calibrator = YOLOInt8Calibrator(
                calibration_images, max_batch_size
            )
            config.int8_calibrator = calibrator
    elif precision == "fp32":
        logger.info("FP32 mode (default)")
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # DLA configuration (Jetson)
    if dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        logger.info(f"DLA core {dla_core} enabled")

    # Build engine
    logger.info("Building engine (this may take several minutes)...")
    start = time.time()
    engine = builder.build_engine(network, config)
    elapsed = time.time() - start

    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    # Save engine
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    logger.info(f"Engine built in {elapsed:.1f}s: {engine_path}")
    logger.info(f"Engine size: {os.path.getsize(engine_path) / (1024**2):.1f} MB")

    return engine_path


class TRTModelWrapper:
    """Wrapper that presents a Ultralytics-like API for TensorRT engines.

    This allows the DetectionPipeline to use TensorRT without code changes.
    """

    def __init__(
        self,
        engine_path: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        classes: list[int] | None = None,
        device: str = "cuda:0",
    ):
        self.engine_path = engine_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.device = device

        self._session: TensorRTInferenceSession | None = None
        self._load_engine()

    def _load_engine(self) -> None:
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        self._session = TensorRTInferenceSession(
            self.engine_path,
            input_shape=(1, 3, self.imgsz, self.imgsz),
        )
        logger.info(f"TensorRT model loaded: {self.engine_path}")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image to NCHW format."""
        # Resize
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        # HWC to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess(
        self, outputs: list[np.ndarray], orig_shape: tuple[int, int]
    ) -> list[dict]:
        """Convert TensorRT outputs to Detection objects."""
        # YOLOv8/v11 output format: [batch, 84, 8400] where 84 = 4(box) + 80(classes)
        # or [batch, 56, 8400] for pose (4 + 80 + 3*17/80)
        predictions = outputs[0]  # [1, 84, 8400]
        if predictions.ndim == 3:
            predictions = predictions[0]  # [84, 8400]
            predictions = np.transpose(predictions, (1, 0))  # [8400, 84]

        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores >= self.conf
        predictions = predictions[mask]
        scores = scores[mask]

        if len(predictions) == 0:
            return []

        # Extract boxes and classes
        boxes = predictions[:, :4]  # xywh center format
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Convert xywh to xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Scale to original image
        scale_x = orig_shape[1] / self.imgsz
        scale_y = orig_shape[0] / self.imgsz
        xyxy[:, [0, 2]] *= scale_x
        xyxy[:, [1, 3]] *= scale_y

        # Filter by class
        if self.classes:
            class_mask = np.isin(class_ids, self.classes)
            xyxy = xyxy[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]

        # Simple NMS (can be optimized with cv2.dnn.NMSBoxes)
        keep = self._nms(xyxy, scores, self.iou)

        detections = []
        for i in keep:
            detections.append({
                "box": xyxy[i].tolist(),
                "conf": float(scores[i]),
                "cls": int(class_ids[i]),
            })

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]

        return keep

    def predict(
        self,
        source: str | np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        classes: list[int] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        """Run detection on an image ( Ultralytics-compatible API )."""
        if self._session is None:
            raise RuntimeError("TensorRT session not initialized")

        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        classes = classes if classes is not None else self.classes

        # Load image
        if isinstance(source, str):
            img = cv2.imread(source)
            orig_shape = img.shape[:2]
        else:
            img = source
            orig_shape = img.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(img)

        # Inference
        t0 = time.time()
        outputs = self._session.infer(input_tensor)
        t1 = time.time()

        if verbose:
            logger.info(f"TensorRT inference: {(t1 - t0) * 1000:.1f}ms")

        # Postprocess
        detections = self._postprocess(outputs, orig_shape)
        return detections

    def track(
        self,
        source: str | np.ndarray,
        tracker: str = "bytetrack",
        persist: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        """Run detection + tracking (fallback to detection without tracker integration)."""
        # For TensorRT, tracking is done externally via ByteTrack/BoT-SORT
        # This is a simplified version - full integration would require tracker state
        return self.predict(source, verbose=verbose, **kwargs)


class TensorRTConverter:
    """High-level converter for YOLO → ONNX → TensorRT pipeline."""

    def __init__(self, config: Any | None = None):
        self.config = config

    def convert(
        self,
        model_path: str,
        imgsz: int = 640,
        force_rebuild: bool = False,
    ) -> str:
        """Full conversion pipeline: YOLO .pt → ONNX → TensorRT .engine.

        Returns:
            Path to the TensorRT engine file
        """
        model_name = Path(model_path).stem
        engine_dir = Path(self.config.engine_dir) if self.config else Path("models/trt_engines")
        onnx_dir = Path(self.config.onnx_dir) if self.config else Path("models/onnx")
        engine_dir.mkdir(parents=True, exist_ok=True)
        onnx_dir.mkdir(parents=True, exist_ok=True)

        engine_path = str(engine_dir / f"{model_name}_{imgsz}.engine")

        # Check if engine already exists
        if os.path.exists(engine_path) and not force_rebuild:
            logger.info(f"Using existing engine: {engine_path}")
            return engine_path

        # Step 1: Export to ONNX
        onnx_path = str(onnx_dir / f"{model_name}_{imgsz}.onnx")
        if not os.path.exists(onnx_path) or force_rebuild:
            export_yolo_to_onnx(
                model_path=model_path,
                output_path=onnx_path,
                imgsz=imgsz,
                simplify=True,
                dynamic=False,
            )

        # Step 2: Build TensorRT engine
        build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=self.config.precision if self.config else "fp16",
            max_batch_size=self.config.max_batch_size if self.config else 1,
            workspace_gb=self.config.workspace_gb if self.config else 2.0,
            dla_core=self.config.dla_core if self.config else -1,
            optimization_level=self.config.optimization_level if self.config else 3,
        )

        return engine_path


def is_tensorrt_available() -> bool:
    """Check if TensorRT and all dependencies are available."""
    return TRT_AVAILABLE and ONNX_AVAILABLE and PYCUDA_AVAILABLE


def get_tensorrt_version() -> str:
    """Get TensorRT version string."""
    if TRT_AVAILABLE:
        return trt.__version__
    return "not installed"
