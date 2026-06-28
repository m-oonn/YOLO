"""
Download campus safety datasets from Kaggle.

Uses short cache path to avoid Windows MAX_PATH issues.
Downloads sequentially: fire → weapon → fall → fight

Usage:
  python scripts/download_datasets.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Use a SHORT cache path to avoid Windows MAX_PATH (260 char) limit
# during kagglehub archive extraction
CACHE = Path("e:/kaggle_cache")
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(CACHE)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

import kagglehub

DATASETS = [
    (
        "fire",
        "sinchanashivanand/indoor-fire-and-smoke-detection-with-yolov8",
        "Fire+Smoke",
    ),
    (
        "weapon",
        "alinoorqureshi/weapon-detection-yolo-optimized",
        "Weapon (pistol+knife)",
    ),
    ("fall", "uttejkumarkandagatla/fall-detection-dataset", "Fall Detection"),
    (
        "fight",
        "musawerhussain/thisisithefinalkillerofthefightdetectiondatasetan",
        "Fight/Violence",
    ),
]


def extract_archive(archive: Path, dest: Path) -> int:
    """Extract zip archive to dest via short temp path. Returns file count."""
    tmp = Path(tempfile.mkdtemp(dir=str(RAW)))
    count = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.namelist():
            try:
                zf.extract(member, path=str(tmp))
                count += 1
            except Exception:
                pass
    for item in tmp.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.rmtree(tmp)
    return count


def download_one(name: str, kaggle_id: str, desc: str) -> Path:
    """Download dataset via kagglehub, extract to RAW/<name>/."""
    dest = RAW / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[{name}] {desc}")
    print(f"  Kaggle: {kaggle_id}")
    print(f"{'=' * 60}")

    # Let kagglehub download & extract (short cache avoids long paths)
    try:
        cache_path = kagglehub.dataset_download(kaggle_id)
        print(f"  Extracted: {cache_path}")
        # Copy from cache to raw
        src = Path(cache_path)
        file_count = 0
        for item in src.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
            file_count += 1
        print(f"  Copied {file_count} items to {dest}")
    except Exception as exc:
        print(f"  Kagglehub failed: {exc}")
        print("  Trying manual extraction from archive...")
        # Find archive in cache
        owner, dataset = kaggle_id.split("/")
        archive = CACHE / "datasets" / owner / dataset / "1.archive"
        if not archive.exists():
            # Alternative path
            archive = CACHE / "datasets" / kaggle_id / "1.archive"
        if archive.exists():
            count = extract_archive(archive, dest)
            print(f"  Manually extracted {count} files to {dest}")
        else:
            print("  ERROR: No archive found")
            return dest

    # Count
    imgs = (
        len(list(dest.rglob("*.jpg")))
        + len(list(dest.rglob("*.png")))
        + len(list(dest.rglob("*.jpeg")))
    )
    lbls = len(list(dest.rglob("*.txt")))
    size_mb = (
        sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024 / 1024
    )
    print(f"  Result: {imgs} images, {lbls} labels, {size_mb:.0f}MB")
    return dest


def main() -> int:
    for name, kaggle_id, desc in DATASETS:
        try:
            download_one(name, kaggle_id, desc)
        except Exception as exc:
            print(f"[FATAL] {name}: {exc}")
            return 1

    print(f"\n{'=' * 60}")
    print("ALL DOWNLOADS COMPLETE")
    print(f"{'=' * 60}")
    for name, _, _ in DATASETS:
        d = RAW / name
        imgs = (
            len(list(d.rglob("*.jpg")))
            + len(list(d.rglob("*.png")))
            + len(list(d.rglob("*.jpeg")))
        )
        print(f"  {name}: {imgs} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
