"""
Real-time Training Progress Monitor.

Continuously reads the results.csv written by Ultralytics during training
and displays a live-updating dashboard with loss curves and mAP.

Usage:
  # Auto-watch default run:
  python scripts/monitor_training.py

  # Specify a custom results CSV:
  python scripts/monitor_training.py --csv runs/detect/coco2017_train/results.csv

  # Refresh interval:
  python scripts/monitor_training.py --interval 10

  # Console-only (no matplotlib):
  python scripts/monitor_training.py --no-plot

The dashboard updates every N seconds and shows:
  - Training loss curves (box / cls / dfl)
  - Validation mAP (mAP50 and mAP50-95)
  - Console summary of recent epochs
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS = str(ROOT / "runs" / "detect" / "coco2017_train" / "results.csv")


def console_table(rows, max_rows=25):
    """Print a compact console summary of recent epochs."""
    os.system("cls" if os.name == "nt" else "clear")
    print(
        "╔══════════════════════════════════════════════════════════════════════════╗"
    )
    print("║        YOLO Training Progress Monitor                                  ║")
    print(
        f"║        {datetime.now():%Y-%m-%d %H:%M:%S}                                   ║"
    )
    print(
        "╠══════════════════════════════════════════════════════════════════════════╣"
    )

    if not rows:
        print(
            "║  Waiting for training data ...                                           ║"
        )
        print(
            "╚══════════════════════════════════════════════════════════════════════════╝"
        )
        return

    print(
        "║ Epoch │ box_loss │ cls_loss │ dfl_loss │ Val_box │ Val_cls │  mAP50   │ mAP50-95║"
    )
    print(
        "╟───────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼──────────╢"
    )

    recent = rows[-max_rows:]
    for r in recent:
        epoch = r.get("epoch", "?")
        box = _f(r, "train/box_loss")
        cls = _f(r, "train/cls_loss")
        dfl = _f(r, "train/dfl_loss")
        vbox = _f(r, "val/box_loss")
        vcls = _f(r, "val/cls_loss")
        m50 = _f(r, "metrics/mAP50(B)")
        m95 = _f(r, "metrics/mAP50-95(B)")
        print(
            f"║ {epoch:>5} │ {box:>8} │ {cls:>8} │ {dfl:>8} │ {vbox:>7} │ {vcls:>7} │ {m50:>8} │ {m95:>8} ║"
        )

    print(
        "╚══════════════════════════════════════════════════════════════════════════╝"
    )
    print(f"  {len(rows)} epochs recorded  |  Ctrl+C to exit")


def _f(row, key):
    """Format a metric value for display."""
    try:
        v = float(row.get(key, -1))
        return f"{v:.4f}" if v >= 0 else "   ·"
    except (ValueError, TypeError):
        return "   ·"


def matplotlib_dashboard(rows):
    """Render a matplotlib figure with loss + mAP subplots."""
    try:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed - run: pip install matplotlib")
        return

    epochs = [int(r.get("epoch", 0)) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("YOLO COCO 2017 Training Progress", fontsize=13, fontweight="bold")

    # Panel 1: Training Loss
    ax = axes[0]
    for key, color, label in [
        ("train/box_loss", "#2196F3", "Box Loss"),
        ("train/cls_loss", "#4CAF50", "Cls Loss"),
        ("train/dfl_loss", "#FF9800", "DFL Loss"),
    ]:
        vals = _col(rows, key)
        if vals:
            ax.plot(epochs[: len(vals)], vals, color=color, label=label, linewidth=1.2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Validation mAP
    ax = axes[1]
    for key, color, label in [
        ("metrics/mAP50(B)", "#E91E63", "mAP50"),
        ("metrics/mAP50-95(B)", "#9C27B0", "mAP50-95"),
    ]:
        vals = _col(rows, key)
        if vals:
            ax.plot(epochs[: len(vals)], vals, color=color, label=label, linewidth=1.5)
    ax.set_title("Validation mAP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Precision / Recall
    ax = axes[2]
    for key, color, label in [
        ("metrics/precision(B)", "#00BCD4", "Precision"),
        ("metrics/recall(B)", "#FF5722", "Recall"),
    ]:
        vals = _col(rows, key)
        if vals:
            ax.plot(epochs[: len(vals)], vals, color=color, label=label, linewidth=1.2)
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(rows[-1].get("_csv_dir", ".")) / "live_dashboard.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard -> {out_path}")


def _col(rows, key):
    """Extract a numeric column, filtering invalid values."""
    vals = []
    for r in rows:
        try:
            v = float(r.get(key, -1))
            vals.append(v if v >= 0 else float("nan"))
        except (ValueError, TypeError):
            vals.append(float("nan"))
    # forward-fill nan for continuous lines
    last = None
    for i, v in enumerate(vals):
        if not math.isnan(v):
            last = v
        vals[i] = last
    # strip leading Nones/nans
    first_valid = 0
    for i, v in enumerate(vals):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            first_valid = i
            break
    else:
        return None
    return vals[first_valid:]


def monitor(csv_path: str, interval: float = 10.0, show_plot: bool = True):
    """Main loop: watch CSV and refresh display."""
    csv_path = Path(csv_path).resolve()
    print(f"Monitoring: {csv_path}")
    print(f"Refresh interval: {interval}s")
    print("Press Ctrl+C to stop.\n")

    last_n = 0
    try:
        while True:
            if not csv_path.exists():
                print(f"\rWaiting for {csv_path.name} ...  ", end="", flush=True)
                time.sleep(interval)
                continue

            try:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception as e:
                print(f"\rRead error: {e}", flush=True)
                time.sleep(interval)
                continue

            if not rows:
                time.sleep(interval)
                continue

            # Inject dir for save
            for r in rows:
                r["_csv_dir"] = str(csv_path.parent)

            if len(rows) != last_n:
                console_table(rows)
                last_n = len(rows)

            if show_plot and len(rows) >= 2:
                matplotlib_dashboard(rows)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO training progress monitor"
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_RESULTS,
        help=f"Path to results CSV (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Console-only mode, skip matplotlib dashboard",
    )
    args = parser.parse_args()

    monitor(args.csv, args.interval, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
