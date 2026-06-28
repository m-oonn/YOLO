"""
Campus Safety Detection Model Training Script

Trains a YOLO model for campus safety scenarios:
  - person:     Basic person detection
  - running:    Running behavior
  - fighting:   Fighting/violence behavior
  - fallen:     Fallen person
  - crowd:      Crowd gathering

Usage:
  # Quick test with COCO128 (auto-downloaded, ~7MB)
  python scripts/train_campus_safety.py --quick

  # Full training with COCO 2017 person class
  python scripts/train_campus_safety.py --dataset coco2017 --epochs 100

  # Custom dataset
  python scripts/train_campus_safety.py --dataset /path/to/data.yaml

Requirements:
  pip install ultralytics

GPU Memory Guide (RTX 4060 Laptop 8GB):
  yolo12n: batch=16, ~1.5GB VRAM
  yolo12s: batch=8,  ~2.5GB VRAM
  yolo12m: batch=4,  ~4.0GB VRAM
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_SIZE_MAP = {
    "yolo12n": 16,
    "yolo12s": 8,
    "yolo12m": 4,
    "yolo12l": 2,
    "yolo12x": 1,
}


def check_ultralytics():
    try:
        import ultralytics

        print(f"Ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Please install: pip install ultralytics")
        return False


def check_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024**2)
            print(f"GPU: {props.name} ({total_mb:.0f}MB)")
            return total_mb
        else:
            print("No CUDA GPU detected, using CPU")
            return 0
    except Exception:
        return 0


def resolve_dataset(args):
    if args.dataset == "coco128" or args.quick:
        print("Using COCO128 (auto-downloaded by Ultralytics, ~7MB)")
        return "coco128.yaml"
    elif args.dataset == "coco2017":
        data_yaml = ROOT / "datasets" / "campus_safety" / "data.yaml"
        if data_yaml.exists():
            print(f"Using COCO 2017 converted dataset: {data_yaml}")
            return str(data_yaml)
        else:
            print(f"COCO 2017 dataset not yet converted ({data_yaml} not found)")
            print("Run: python scripts/prepare_coco_yolo.py")
            print("Falling back to COCO128 for now...")
            return "coco128.yaml"
    else:
        custom_path = Path(args.dataset)
        if custom_path.exists():
            print(f"Using custom dataset: {custom_path}")
            return str(custom_path)
        else:
            print(f"ERROR: Dataset not found: {args.dataset}")
            return None


def train(args):
    from datetime import datetime

    from ultralytics import YOLO

    model_name = args.model
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"

    # Resume mode: load from checkpoint
    if args.resume:
        if args.weights and Path(args.weights).exists():
            model_path = args.weights
            print(f"Resuming from checkpoint: {model_path}")
        else:
            # Find latest checkpoint automatically
            checkpoint_dir = ROOT / "outputs" / "train" / "campus_safety" / "weights"
            candidates = ["last.pt", "best.pt"]
            model_path = None
            for c in candidates:
                p = checkpoint_dir / c
                if p.exists():
                    model_path = str(p)
                    break
            if model_path is None:
                print("ERROR: No checkpoint found to resume from.")
                print("Please train first or specify --weights path")
                return 1
            print(f"Resuming from latest checkpoint: {model_path}")

        model = YOLO(model_path)
    else:
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

    data_yaml = resolve_dataset(args)
    if data_yaml is None:
        return 1

    vram_mb = check_gpu()

    batch = args.batch
    if batch <= 0:
        base_model = args.model.replace(".pt", "")
        batch = BATCH_SIZE_MAP.get(base_model, 8)
        if vram_mb > 0 and vram_mb < 6000:
            batch = max(1, batch // 2)
            print(f"Low VRAM ({vram_mb:.0f}MB), reduced batch to {batch}")

    epochs = 5 if args.quick else args.epochs

    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {args.device}")
    if args.resume:
        print(f"Resume timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "device": args.device,
        "project": str(ROOT / "runs" / "detect" / "outputs" / "train"),
        "name": "campus_safety",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "augment": False,
        "auto_augment": None,
        "erasing": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "deterministic": False,
        "save": True,
        "save_period": 10,
        "workers": 0,
        "cache": False,
    }

    if vram_mb > 0:
        train_kwargs["amp"] = False
        train_kwargs["half"] = False
        print("AMP disabled (fixes 0xC0000005 crash on Windows + PyTorch 2.7+)")

    results = model.train(**train_kwargs)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model: {results.best}")
    print(f"Results: {results.results_dict}")
    print("=" * 60)
    return results


def validate(args):
    from ultralytics import YOLO

    model_path = (
        args.weights
        or ROOT / "outputs" / "train" / "campus_safety" / "weights" / "best.pt"
    )
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    model = YOLO(str(model_path))
    data_yaml = ROOT / "datasets" / "campus_safety" / "data.yaml"
    if not data_yaml.exists():
        data_yaml = "coco128.yaml"

    results = model.val(
        data=str(data_yaml) if isinstance(data_yaml, Path) else data_yaml
    )
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return 0


def export_model(args):
    from ultralytics import YOLO

    model_path = (
        args.weights
        or ROOT / "outputs" / "train" / "campus_safety" / "weights" / "best.pt"
    )
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    model = YOLO(str(model_path))
    formats = args.format.split(",")
    for fmt in formats:
        print(f"Exporting to {fmt} ...")
        model.export(format=fmt, imgsz=args.imgsz)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Campus Safety Detection Training")
    parser.add_argument(
        "--model", default="yolo12n", help="Model name (yolo12n, yolo12s, yolo12m)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--device", default="0", help="Device (cpu, 0, 0,1,2,3)")
    parser.add_argument(
        "--dataset",
        default="coco2017",
        help="Dataset: coco128, coco2017, or /path/to/data.yaml",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test: 5 epochs with COCO128"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to trained weights for val/export or resume",
    )
    parser.add_argument(
        "--format", default="onnx", help="Export formats (onnx,engine,openvino)"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "val", "export"],
        help="Operation mode",
    )
    args = parser.parse_args()

    if not check_ultralytics():
        return 1

    os.makedirs(ROOT / "outputs" / "train", exist_ok=True)

    if args.mode == "train":
        return train(args)
    elif args.mode == "val":
        return validate(args)
    elif args.mode == "export":
        return export_model(args)

    return 0


if __name__ == "__main__":
    exit(main())
