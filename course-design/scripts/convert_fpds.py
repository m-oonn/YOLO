"""
Convert FPDS dataset to YOLO format.

FPDS format (absolute pixel coords):
    class x1 y1 x2 y2
    class: 1=fallen, -1=non-fallen

YOLO format (normalized):
    class x_center y_center width height
    all values in [0,1]

Unified class mapping:
    -1 (non-fallen) → 0 (person)
     1 (fallen)     → 4 (fallen)

Output: datasets/fpds_yolo/{images,labels}/{train,val}/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "fpds_yolo"

TRAIN_SPLITS = ["split1", "split2", "split3", "split10", "split11"]
VAL_SPLITS = ["split12", "split13"]

CLASS_MAP = {-1: 0, 1: 4}  # non-fallen→person, fallen→fallen


def convert_one(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path) -> bool:
    """Convert one image+label pair to YOLO format. Returns True on success."""
    from PIL import Image

    try:
        img = Image.open(img_path)
        w, h = img.size
    except Exception:
        return False

    lines: list[str] = []
    for raw in lbl_path.read_text().strip().splitlines():
        if not raw.strip():
            continue
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls_old = int(parts[0])
        cls_new = CLASS_MAP.get(cls_old)
        if cls_new is None:
            continue
        x1, x2, y1, y2 = map(int, parts[1:5])  # FPDS uses x1 x2 y1 y2 order

        # Convert to YOLO normalized format
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Clamp to [0,1]
        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        bw = max(0, min(1, bw))
        bh = max(0, min(1, bh))

        if bw > 0 and bh > 0:
            lines.append(f"{cls_new} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.write_text("\n".join(lines) + "\n")
        shutil.copy2(img_path, out_img)
        return True
    return False


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)

    total = 0
    total_skipped = 0

    for split_name in TRAIN_SPLITS:
        split_dir = RAW / "train" / split_name
        if not split_dir.exists():
            continue
        out_img_dir = OUT / "images" / "train"
        out_lbl_dir = OUT / "labels" / "train"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for png in sorted(split_dir.glob("*.png")):
            txt = split_dir / (png.stem + ".txt")
            if not txt.exists():
                total_skipped += 1
                continue
            new_name = f"fpds_{split_name}_{png.name}"
            if convert_one(
                png,
                txt,
                out_img_dir / new_name,
                out_lbl_dir / (new_name.replace(".png", ".txt")),
            ):
                total += 1
        print(f"  {split_name}: done")

    for split_name in VAL_SPLITS:
        split_dir = RAW / "valid" / split_name
        if not split_dir.exists():
            continue
        out_img_dir = OUT / "images" / "val"
        out_lbl_dir = OUT / "labels" / "val"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for png in sorted(split_dir.glob("*.png")):
            txt = split_dir / (png.stem + ".txt")
            if not txt.exists():
                total_skipped += 1
                continue
            new_name = f"fpds_{split_name}_{png.name}"
            if convert_one(
                png,
                txt,
                out_img_dir / new_name,
                out_lbl_dir / (new_name.replace(".png", ".txt")),
            ):
                total += 1
        print(f"  {split_name}: done")

    # Count classes
    person = 0
    fallen = 0
    for lbl in OUT.rglob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.startswith("0 "):
                person += 1
            elif line.startswith("4 "):
                fallen += 1

    # Write data.yaml
    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"# FPDS Fallen People Dataset (YOLO format)\n"
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 6\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: fire\n"
        f"  2: smoke\n"
        f"  3: weapon\n"
        f"  4: fallen\n"
        f"  5: violence\n"
    )

    print(f"\nConverted: {total} images ({total_skipped} skipped)")
    print(f"Person labels: {person}, Fallen labels: {fallen}")
    print(f"Output: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
