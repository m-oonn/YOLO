"""
Merge 4 campus safety datasets into one unified YOLO dataset.

Class mapping (unified):
    0: person      (from fight NonViolence)
    1: fire        (from fire/fire)
    2: smoke       (from fire/smoke)
    3: weapon      (from weapon/Weapon)
    4: fallen      (from fall)
    5: violence    (from fight/Violence)

Output: datasets/campus_safety_full/images/{train,val}/ + labels/{train,val}/

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --weapon-limit 8000
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "campus_safety_full"

# Source structure per dataset: (root, [train_dirs...], [val_dirs...])
# Each entry: (path_to_images, path_to_labels, prefix_for_uniqueness)
SOURCES: list[dict] = []


def gather_sources() -> None:
    """Scan raw datasets and populate SOURCES list."""
    # fire: data/train, data/valid
    fire_root = RAW / "fire" / "data"
    if fire_root.exists():
        SOURCES.append(
            {
                "name": "fire",
                "train_img": fire_root / "train" / "images",
                "train_lbl": fire_root / "train" / "labels",
                "val_img": fire_root / "valid" / "images",
                "val_lbl": fire_root / "valid" / "labels",
                "remap": {0: 1, 1: 2},  # fire->1, smoke->2
            }
        )

    # weapon: dataset_merged/train, val, test
    wp_root = RAW / "weapon" / "dataset_merged"
    if wp_root.exists():
        SOURCES.append(
            {
                "name": "weapon",
                "train_img": wp_root / "train" / "images",
                "train_lbl": wp_root / "train" / "labels",
                "val_img": wp_root / "val" / "images",
                "val_lbl": wp_root / "val" / "labels",
                "test_img": wp_root / "test" / "images",
                "test_lbl": wp_root / "test" / "labels",
                "remap": {0: 3},  # Weapon->3
            }
        )

    # fall: images/train, images/val and labels/train, labels/val
    fall_root = RAW / "fall" / "fall_dataset"
    if fall_root.exists():
        SOURCES.append(
            {
                "name": "fall",
                "train_img": fall_root / "images" / "train",
                "train_lbl": fall_root / "labels" / "train",
                "val_img": fall_root / "images" / "val",
                "val_lbl": fall_root / "labels" / "val",
                "remap": {0: 4},  # fall->4 (fallen)
            }
        )

    # fight: violance-nonviolance.v7i.yolov8/train, valid, test
    fight_root = RAW / "fight" / "violance-nonviolance.v7i.yolov8"
    if fight_root.exists():
        SOURCES.append(
            {
                "name": "fight",
                "train_img": fight_root / "train" / "images",
                "train_lbl": fight_root / "train" / "labels",
                "val_img": fight_root / "valid" / "images",
                "val_lbl": fight_root / "valid" / "labels",
                "test_img": fight_root / "test" / "images",
                "test_lbl": fight_root / "test" / "labels",
                "remap": {0: 0, 1: 5},  # NonViolence->person(0), Violence->5
            }
        )


def copy_and_remap(
    src_img: Path,
    src_lbl: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    remap: dict[int, int],
    prefix: str,
    limit: int = 0,
) -> int:
    """Copy images and remap labels. Returns count of copied items."""
    if not src_img.exists() or not src_lbl.exists():
        print(f"  WARN: missing {src_img} or {src_lbl}")
        return 0

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        [f for f in src_img.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")]
    )
    if limit and limit < len(img_files):
        random.seed(42)
        img_files = random.sample(img_files, limit)

    count = 0
    for img in img_files:
        label_name = img.stem + ".txt"
        lbl = src_lbl / label_name
        if not lbl.exists():
            # Try with additional suffixes (some datasets have .rf.xxx in names)
            candidates = list(src_lbl.glob(img.stem + "*.txt"))
            if candidates:
                lbl = candidates[0]
            else:
                continue

        new_name = f"{prefix}_{img.name}"
        shutil.copy2(img, dst_img_dir / new_name)

        # Remap labels
        new_lines: list[str] = []
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(float(parts[0]))
            new_cls = remap.get(old_cls)
            if new_cls is None:
                continue  # skip classes not in mapping
            new_lines.append(f"{new_cls} " + " ".join(parts[1:]))

        if new_lines:
            (dst_lbl_dir / new_name).with_suffix(".txt").write_text(
                "\n".join(new_lines) + "\n"
            )
        count += 1

    return count


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Merge campus safety datasets")
    parser.add_argument(
        "--weapon-limit",
        type=int,
        default=8000,
        help="Max weapon training images (downsample to balance)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    # Clean output
    if OUT.exists():
        shutil.rmtree(OUT)

    gather_sources()

    train_img = OUT / "images" / "train"
    train_lbl = OUT / "labels" / "train"
    val_img = OUT / "images" / "val"
    val_lbl = OUT / "labels" / "val"

    total_train = 0
    total_val = 0

    for src in SOURCES:
        name = src["name"]
        remap = src["remap"]
        limit = args.weapon_limit if name == "weapon" else 0
        print(f"\n[{name}] remap={remap}" + (f" limit={limit}" if limit else ""))

        # Train
        if "train_img" in src:
            n = copy_and_remap(
                src["train_img"],
                src["train_lbl"],
                train_img,
                train_lbl,
                remap,
                name,
                limit=limit,
            )
            print(f"  train: {n} images")
            total_train += n

        # Val
        if "val_img" in src:
            n = copy_and_remap(
                src["val_img"],
                src["val_lbl"],
                val_img,
                val_lbl,
                remap,
                name,
            )
            print(f"  val:   {n} images")
            total_val += n

        # Test → add to val (no separate test set)
        if "test_img" in src:
            n = copy_and_remap(
                src["test_img"],
                src["test_lbl"],
                val_img,
                val_lbl,
                remap,
                name,
            )
            print(f"  test→val: {n} images")
            total_val += n

        # Flat (fall dataset)
        if "flat_img" in src:
            # Split 80/20 for train/val
            imgs = sorted(
                [
                    f
                    for f in src["flat_img"].iterdir()
                    if f.suffix.lower() in (".jpg", ".png", ".jpeg")
                ]
            )
            random.shuffle(imgs)
            split = int(len(imgs) * 0.8)
            train_imgs = imgs[:split]
            val_imgs = imgs[split:]

            # Train split
            (train_img / name).mkdir(parents=True, exist_ok=True)
            for img in train_imgs:
                new_name = f"{name}_{img.name}"
                shutil.copy2(img, train_img / new_name)
                # Find matching label
                lbl = src["flat_lbl"] / (img.stem + ".txt")
                if lbl.exists():
                    new_lines: list[str] = []
                    for line in lbl.read_text().strip().splitlines():
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        if not parts:
                            continue
                        old_cls = int(float(parts[0]))
                        new_cls = remap.get(old_cls)
                        if new_cls is None:
                            continue
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                    if new_lines:
                        (train_lbl / new_name).with_suffix(".txt").write_text(
                            "\n".join(new_lines) + "\n"
                        )

            # Val split
            for img in val_imgs:
                new_name = f"{name}_{img.name}"
                shutil.copy2(img, val_img / new_name)
                lbl = src["flat_lbl"] / (img.stem + ".txt")
                if lbl.exists():
                    new_lines = []
                    for line in lbl.read_text().strip().splitlines():
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        if not parts:
                            continue
                        old_cls = int(float(parts[0]))
                        new_cls = remap.get(old_cls)
                        if new_cls is None:
                            continue
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                    if new_lines:
                        (val_lbl / new_name).with_suffix(".txt").write_text(
                            "\n".join(new_lines) + "\n"
                        )

            print(f"  train: {len(train_imgs)} images (80/20 split)")
            print(f"  val:   {len(val_imgs)} images")
            total_train += len(train_imgs)
            total_val += len(val_imgs)

    # Write data.yaml
    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"# Campus Safety Full Detection Dataset\n"
        f"# Merged from: fire, weapon, fall, fight\n\n"
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 6\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: fire\n"
        f"  2: smoke\n"
        f"  3: weapon\n"
        f"  4: fallen\n"
        f"  5: violence\n"
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Train: {total_train} images")
    print(f"Val:   {total_val} images")
    print(f"Total: {total_train + total_val} images")
    print("Classes: person, fire, smoke, weapon, fallen, violence")
    print(f"Output: {OUT}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
