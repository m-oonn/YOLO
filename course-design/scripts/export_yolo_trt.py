"""
Export YOLO detection model to TensorRT engine for Jetson deployment.

Usage (run ON the Jetson device):
    python scripts/export_yolo_trt.py --model models/yolo12s.pt --precision fp16

Output: models/trt_engines/yolo12s_fp16.engine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def export_to_onnx(model_path: str, output: str, imgsz: int = 640, opset: int = 17) -> str:
    """Export YOLO .pt to ONNX."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    print(f"[ONNX] {onnx_path}")
    return str(onnx_path)


def build_trt_engine(
    onnx_path: str,
    output: str,
    precision: str = "fp16",
    workspace_gb: int = 2,
) -> str:
    """Build TensorRT engine from ONNX using trtexec."""
    import subprocess
    import shutil

    trtexec = shutil.which("trtexec")
    if not trtexec:
        for candidate in [
            "/usr/src/tensorrt/bin/trtexec",
            "/opt/nvidia/tensorrt/bin/trtexec",
        ]:
            if Path(candidate).exists():
                trtexec = candidate
                break
    if not trtexec:
        raise FileNotFoundError(
            "trtexec not found. Install TensorRT: "
            "sudo apt install tensorrt tensorrt-dev"
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output}",
        f"--{precision}" if precision != "fp32" else "",
        f"--workspace={workspace_gb * 1024}",
        "--optShapes=images:1x3x640x640",
        "--minShapes=images:1x3x640x640",
        "--maxShapes=images:4x3x640x640",
    ]
    cmd = [c for c in cmd if c]

    print(f"[TRT] Building engine: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_mb = Path(output).stat().st_size / 1024**2
    print(f"[TRT] Engine built: {output} ({size_mb:.1f} MB)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT")
    parser.add_argument("--model", default="models/yolo12s.pt", help="YOLO model path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace", type=int, default=2, help="TRT workspace GB")
    args = parser.parse_args()

    model_name = Path(args.model).stem
    onnx_path = str(ROOT / f"models/onnx/{model_name}.onnx")
    engine_path = str(ROOT / f"models/trt_engines/{model_name}_{args.precision}.engine")

    print(f"Exporting {args.model} → {engine_path}")
    print(f"  Precision: {args.precision}")
    print(f"  Input size: {args.imgsz}")

    # Step 1: ONNX
    onnx_out = export_to_onnx(args.model, onnx_path, args.imgsz)
    print(f"  ONNX done: {onnx_out} ({Path(onnx_out).stat().st_size / 1024**2:.1f} MB)")

    # Step 2: TensorRT
    engine_out = build_trt_engine(onnx_out, engine_path, args.precision, args.workspace)

    print(f"\n Done. Engine ready for deployment: {engine_out}")
    print(f" Use in detection pipeline: device='cuda', engine='{engine_out}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
