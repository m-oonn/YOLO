#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify project integrity for Task 3 (GitHub readiness)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "config.example.yaml",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "configs/default.yaml",
    "models/README.md",
    "models/.gitkeep",
    "outputs/.gitkeep",
]

FORBIDDEN_TRACKED = [
    ".env",
    "outputs/events.db",
]


def main() -> int:
    print("=" * 60)
    print("Task 3 Project Integrity Verification")
    print("=" * 60)
    passed = True

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        ok = path.exists()
        print(f"{'OK' if ok else 'MISSING':4} {rel}")
        passed = passed and ok

    print("\n--- Import smoke test ---")
    try:
        from backend.store import get_store, close_store
        from core.config import load_config
        cfg = load_config()
        assert cfg.model_path
        close_store()
        print("OK   core imports")
    except Exception as e:
        print(f"FAIL imports: {e}")
        passed = False

    print("\n--- Sensitive path check ---")
    for rel in FORBIDDEN_TRACKED:
        if (ROOT / rel).exists():
            print(f"WARN {rel} exists locally (should be gitignored)")

    print("\n--- No absolute paths in configs ---")
    yaml_text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    if "E:" in yaml_text or "C:\\" in yaml_text:
        print("FAIL absolute Windows paths in default.yaml")
        passed = False
    else:
        print("OK   configs use relative paths")

    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
