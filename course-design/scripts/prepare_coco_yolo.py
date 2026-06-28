"""
COCO 2017 to YOLO format converter for campus safety detection.

Extracts person-related annotations from COCO and converts them to YOLO format.
Uses symlinks instead of copying images to save disk space (~19GB saved).

Classes:
  0: person      (base person detection)

Usage:
  python scripts/prepare_coco_yolo.py
  python scripts/prepare_coco_yolo.py --copy  # copy instead of symlink
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

COCO_PERSON_ID = 1

ROOT = Path(__file__).resolve().parent.parent
COCO_ROOT = ROOT / "datasets" / "coco"
OUTPUT_ROOT = ROOT / "datasets" / "campus_safety"


def load_annotations(split: str):
    ann_file = COCO_ROOT / "annotations" / f"instances_{split}2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    print(f"Loading {ann_file} ...")
    with open(ann_file, encoding="utf-8") as f:
        return json.load(f)


def convert_to_yolo(
    coco_ann,
    img_dir: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    split: str,
    use_symlink: bool = True,
):
    images = {img["id"]: img for img in coco_ann["images"]}
    anns = defaultdict(list)
    for ann in coco_ann["annotations"]:
        if ann["category_id"] == COCO_PERSON_ID:
            anns[ann["image_id"]].append(ann)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    total_persons = 0

    for img_id, img_info in images.items():
        img_anns = anns.get(img_id, [])
        if not img_anns:
            continue

        img_w, img_h = img_info["width"], img_info["height"]
        src_img = img_dir / img_info["file_name"]
        if not src_img.exists():
            skipped += 1
            continue

        dst_img = out_img_dir / img_info["file_name"]
        if not dst_img.exists():
            if use_symlink:
                try:
                    os.symlink(str(src_img.resolve()), str(dst_img))
                except OSError:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        label_file = out_label_dir / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            for ann in img_anns:
                x, y, w, h = ann["bbox"]
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                total_persons += 1
        processed += 1

        if processed % 10000 == 0:
            print(f"  {split}: processed {processed} images ...")

    print(
        f"  {split}: {processed} images with person annotations, {total_persons} person boxes, {skipped} skipped"
    )
    return processed


def create_data_yaml():
    yaml_content = f"""# Campus Safety Detection Dataset
# Converted from COCO 2017 (person class only)
# Original: https://cocodataset.org/
# License: Creative Commons Attribution 4.0 License

path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Created {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="COCO 2017 to YOLO format converter")
    parser.add_argument(
        "--copy", action="store_true", help="Copy images instead of symlinks"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("COCO 2017 to YOLO Format Converter")
    print("=" * 60)

    if not COCO_ROOT.exists():
        print(f"ERROR: COCO directory not found: {COCO_ROOT}")
        print("Please download COCO 2017 dataset first.")
        return 1

    use_symlink = not args.copy
    if use_symlink:
        print("Mode: symlink (saves ~19GB disk space)")
    else:
        print("Mode: copy (requires ~19GB additional disk space)")

    total = 0
    for split in ["train", "val"]:
        print(f"\nProcessing {split} split ...")
        coco_ann = load_annotations(split)
        img_dir = COCO_ROOT / f"{split}2017"
        out_img_dir = OUTPUT_ROOT / "images" / split
        out_label_dir = OUTPUT_ROOT / "labels" / split
        count = convert_to_yolo(
            coco_ann, img_dir, out_img_dir, out_label_dir, split, use_symlink
        )
        total += count

    create_data_yaml()

    print("\n" + "=" * 60)
    print(f"Conversion complete! Total: {total} images with person annotations")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
