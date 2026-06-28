# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model export utilities — download, ONNX export, TensorRT engine build."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "qwen2-vl-2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "vlm",
        "estimated_size_gb": 4.0,
    },
    "qwen2-vl-7b": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "vlm",
        "estimated_size_gb": 16.0,
    },
    "smolvlm-500m": {
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "type": "vlm",
        "estimated_size_gb": 1.2,
    },
    "florence-2": {
        "hf_id": "microsoft/Florence-2-base",
        "type": "vlm",
        "estimated_size_gb": 1.0,
    },
}


def download_mllm_model(model_type: str, output_dir: str = "models/mllm") -> str:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[model_type]
    hf_id = info["hf_id"]
    target_dir = os.path.join(output_dir, model_type)

    os.makedirs(target_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading {hf_id} to {target_dir}")
        path = snapshot_download(
            repo_id=hf_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to: {path}")
        return path
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def export_to_onnx(
    model_dir: str,
    output_path: str = "models/onnx/vision_encoder.onnx",
    opset: int = 17,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import torch
        from transformers import AutoModel

        logger.info(f"Loading model from {model_dir} for ONNX export")
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
        )
        logger.info(f"ONNX model exported to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.info(
            "Note: VLM ONNX export is complex. Consider using PyTorch backend instead."
        )
        raise


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str = "models/trt_engines/vision_encoder.engine",
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_gb: float = 2.0,
    dla_core: int = -1,
) -> str:
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    trtexec_path = _find_trtexec()
    if not trtexec_path:
        raise FileNotFoundError(
            "trtexec not found. Install TensorRT and ensure trtexec is in PATH."
        )

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--maxBatch={max_batch_size}",
        f"--workspace={int(workspace_gb * 1024)}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")

    if dla_core >= 0:
        cmd.extend([f"--useDLACore={dla_core}", "--allowGPUFallback"])

    logger.info(f"Building TensorRT engine: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)
        logger.info(f"TensorRT engine built: {engine_path}")
        return engine_path
    except subprocess.CalledProcessError as e:
        logger.error(f"TensorRT build failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("trtexec not found")
        raise


def _find_trtexec() -> str | None:
    try:
        result = subprocess.run(
            ["which", "trtexec"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    common_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        "/opt/nvidia/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p

    return None
