from __future__ import annotations

"""
COCO 2017 Training Script with Auto-Resume and Progress Visualization.

Key features:
  - Auto-resume: automatically detects and resumes from last checkpoint
  - Time-limited training: stops gracefully after --max-time hours
  - Multi-level tqdm progress bars with resume-aware initialization
  - Training state persistence (JSON) for monitoring between runs
  - Post-training plots from results.csv

Usage:
  # First run - starts from scratch
  python scripts/train_coco2017.py --epochs 100

  # Subsequent runs - auto-resumes from last checkpoint
  python scripts/train_coco2017.py --epochs 100

  # Force fresh start (ignores existing checkpoints)
  python scripts/train_coco2017.py --epochs 100 --fresh

  # Time-limited training (auto-stops after 3.5 hours)
  python scripts/train_coco2017.py --epochs 100 --max-time 3.5

  # Quick test
  python scripts/train_coco2017.py --quick --epochs 3

Requirements:
  pip install ultralytics tensorboard matplotlib tqdm
"""
import argparse
import contextlib
import csv
import ctypes
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE_MAP = {
    "yolo12n": 16,
    "yolo12s": 4,
    "yolo12m": 4,
    "yolo12l": 2,
    "yolo12x": 1,
}

DEFAULT_PROJECT = str(ROOT / "runs" / "detect")
DEFAULT_NAME = "coco2017_train"
STATE_FILE = "training_state.json"

# Windows kernel32 flags to prevent sleep during long-running training
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002  # keeps display on too


class InsomniaBlocker:
    """Prevent Windows from sleeping while training. Restores on exit / crash."""

    def __init__(self) -> None:
        self._active = False

    def __enter__(self) -> InsomniaBlocker:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
            )
            self._active = True
            print("[INSOMNIA] System sleep disabled while training")
        return self

    def __exit__(self, *args: object) -> None:
        if self._active:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            self._active = False
            print("[INSOMNIA] System sleep restored")


# ---------------------------------------------------------------------------
# Training state persistence
# ---------------------------------------------------------------------------
def load_training_state(run_dir: Path) -> dict:
    """Load persisted training state, or return empty dict."""
    path = run_dir / STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_training_state(run_dir: Path, state: dict) -> None:
    """Persist training state as JSON for crash-recovery visibility."""
    path = run_dir / STATE_FILE
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint detection
# ---------------------------------------------------------------------------
def detect_checkpoint(run_dir: Path, target_epochs: int) -> dict:
    """Check if a checkpoint exists and extract resume info.

    Returns dict with keys:
      resume: bool          - whether to resume
      checkpoint_path: str  - path to last.pt if available
      start_epoch: int      - 0-indexed epoch that was LAST completed
      completed_epochs: int - number of epochs already finished
    """
    last_pt = run_dir / "weights" / "last.pt"
    if not last_pt.exists():
        return {
            "resume": False,
            "checkpoint_path": "",
            "start_epoch": -1,
            "completed_epochs": 0,
        }

    # Try to extract epoch from checkpoint
    start_epoch = -1
    try:
        import torch

        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        start_epoch = ckpt.get("epoch", -1)  # 0-indexed, the last COMPLETED epoch
    except Exception:
        # Fallback: infer from results.csv
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            try:
                with open(results_csv) as f:
                    reader = csv.DictReader(f)
                    completed = sum(1 for _ in reader)
                if completed > 0:
                    start_epoch = completed - 1  # results.csv rows are 1-indexed epochs
            except Exception:
                pass
        if start_epoch < 0:
            start_epoch = 0  # assume at least epoch 0 was done

    completed_epochs = start_epoch + 1
    checkpoint_path = str(last_pt)
    print(
        f"[DETECT] Found checkpoint: epoch {completed_epochs}/{target_epochs} completed"
    )
    return {
        "resume": True,
        "checkpoint_path": checkpoint_path,
        "start_epoch": start_epoch,
        "completed_epochs": completed_epochs,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def check_env() -> bool:
    """Verify environment: ultralytics installed, GPU available."""
    try:
        import ultralytics

        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return False

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024**2)
            print(f"[OK] GPU: {props.name} ({total_mb:.0f} MB)")
        else:
            print("[WARN] No CUDA GPU - training will be slow on CPU")
    except Exception:
        print("[WARN] Could not detect GPU")

    return True


def resolve_dataset(args: argparse.Namespace) -> str | None:
    """Return the dataset config path to use."""
    if args.dataset == "coco128" or args.quick:
        print("[INFO] Using coco128 (quick mode, auto-download by Ultralytics)")
        return "coco128.yaml"

    if args.dataset == "coco2017":
        yaml_path = ROOT / "datasets" / "campus_safety" / "data.yaml"
        if yaml_path.exists():
            print(f"[INFO] COCO 2017 dataset: {yaml_path}")
            return str(yaml_path)
        print("[WARN] COCO 2017 not converted yet.")
        print("  Run first: python scripts/prepare_coco_yolo.py")
        print("  Falling back to coco128.yaml...")
        return "coco128.yaml"

    custom = Path(args.dataset)
    if custom.exists():
        return str(custom)
    print(f"[ERROR] Dataset not found: {args.dataset}")
    return None


def resolve_model(args: argparse.Namespace) -> str | None:
    """Return the model checkpoint path (non-resume path only)."""
    if args.weights and Path(args.weights).exists():
        return args.weights
    model = args.model
    if not model.endswith(".pt"):
        model = f"{model}.pt"
    return model


def auto_batch(model_name: str, gpu_mb: float) -> int:
    """Pick a batch size appropriate for the GPU."""
    base = model_name.replace(".pt", "")
    batch = BATCH_SIZE_MAP.get(base, 8)
    if 0 < gpu_mb < 6000:
        batch = max(1, batch // 2)
        print(f"[INFO] Low VRAM ({gpu_mb:.0f} MB) -> batch={batch}")
    return batch if gpu_mb > 0 else 4


# ---------------------------------------------------------------------------
# Multi-level training progress callback (tqdm bars + console summary)
# ---------------------------------------------------------------------------
class TrainingProgressCallback:
    """Real-time progress visualization with dual tqdm bars and time-limit support.

    Displays two simultaneous progress bars during training:
      1. Epoch bar — overall progress including previously completed epochs
      2. Batch bar — current-epoch detail with live loss

    Auto-stop via time limit: if --max-time is set, training stops gracefully
    after the current epoch completes and before the time limit is exceeded.
    """

    def __init__(
        self,
        total_epochs: int,
        completed_epochs: int = 0,
        max_time_seconds: float = 0.0,
        run_dir: Path | None = None,
    ):
        self.total_epochs = total_epochs
        self.completed_epochs = (
            completed_epochs  # epochs already finished (from prior runs)
        )
        self.max_time_seconds = max_time_seconds
        self.run_dir = run_dir

        self.current_epoch = 0
        self._start_time = 0.0
        self._best_map50 = -1.0
        self._best_map95 = -1.0
        self._history: list[dict] = []
        self._epoch_pbar: object | None = None
        self._batch_pbar: object | None = None
        self._time_limit_reached = False

    # -- Callback interface --------------------------------------------------

    def on_train_start(self, trainer: object) -> None:
        """Initialize epoch-level progress bar, resume-aware."""
        from tqdm import tqdm as _tqdm

        self._start_time = time.time()

        self._epoch_pbar = _tqdm(
            total=self.total_epochs,
            initial=self.completed_epochs,
            desc="Progress",
            unit="epoch",
            position=0,
            leave=True,
            ncols=120,
            mininterval=0.5,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} epochs "
                "[{elapsed}<{remaining}] "
                "{postfix}"
            ),
        )

        if self.completed_epochs > 0:
            self._epoch_pbar.set_postfix(
                {"status": f"resumed @ epoch {self.completed_epochs}"}
            )
            _tqdm.write(
                f"  >>> Auto-resumed from epoch {self.completed_epochs}/{self.total_epochs} <<<"
            )

    def on_train_epoch_start(self, trainer: object) -> None:
        """Create batch-level progress bar for the upcoming epoch."""
        from tqdm import tqdm as _tqdm

        # Check time limit before starting new epoch
        if self.max_time_seconds > 0:
            elapsed = time.time() - self._start_time
            if elapsed > self.max_time_seconds:
                _tqdm.write(
                    f"\n  ⏰ Time limit ({self.max_time_seconds / 3600:.1f}h) reached. "
                    f"Stopping after epoch {self.current_epoch}..."
                )
                with contextlib.suppress(Exception):
                    trainer.stop_training = True
                self._time_limit_reached = True
                return

        try:
            n_batches = len(trainer.train_loader)
        except Exception:
            n_batches = 0

        if n_batches > 0:
            self._batch_pbar = _tqdm(
                total=n_batches,
                desc=f"  Epoch {trainer.epoch + 1}/{self.total_epochs}",
                unit="bat",
                position=1,
                leave=False,
                ncols=120,
                mininterval=0.1,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} batches "
                    "[{elapsed}<{remaining}, {rate_fmt}] "
                    "{postfix}"
                ),
            )

    def on_train_batch_end(self, trainer: object) -> None:
        """Update batch bar with current loss after each batch."""
        if self._batch_pbar is None:
            return

        try:
            loss_val = trainer.loss_items.detach().cpu().mean().item()
        except Exception:
            loss_val = 0.0

        self._batch_pbar.set_postfix({"loss": f"{loss_val:.3f}"})
        self._batch_pbar.update(1)

        # Periodic GPU memory sampling (every 200 batches)
        if self._batch_pbar.n % 200 == 0 and self.run_dir:
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_log = self.run_dir / "gpu_memory.log"
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(gpu_log, "a", encoding="utf-8") as gf:
                        gf.write(
                            f"{ts} | epoch={self.current_epoch + 1} "
                            f"batch={self._batch_pbar.n} | "
                            f"alloc={allocated:.2f}G reserved={reserved:.2f}G "
                            f"total={total:.2f}G | loss={loss_val:.3f}\n"
                        )
            except Exception:
                pass

    def on_fit_epoch_end(self, trainer: object) -> None:
        """Close batch bar, update epoch bar with validation metrics, save state."""
        if self._batch_pbar is not None:
            self._batch_pbar.close()
            self._batch_pbar = None

        epoch = trainer.epoch + 1  # 1-indexed for display
        self.current_epoch = epoch

        rd = self._extract_metrics(trainer)
        mAP50 = rd.get("metrics/mAP50(B)", 0.0)
        mAP95 = rd.get("metrics/mAP50-95(B)", 0.0)
        box_loss = rd.get(
            "train/box_loss",
            self._history[-1].get("box_loss", 0.0) if self._history else 0.0,
        )

        is_best = mAP50 > 0 and mAP50 > self._best_map50
        if is_best:
            self._best_map50 = mAP50
            self._best_map95 = mAP95

        if self._epoch_pbar is not None:
            postfix = {"mAP50": f"{mAP50:.3f}"}
            if mAP95 > 0:
                postfix["mAP95"] = f"{mAP95:.3f}"
            if box_loss > 0:
                postfix["loss"] = f"{box_loss:.2f}"
            self._epoch_pbar.set_postfix(postfix)
            self._epoch_pbar.update(1)

        self._history.append(
            {
                "epoch": epoch,
                "mAP50": mAP50,
                "mAP50-95": mAP95,
                "box_loss": box_loss,
            }
        )

        elapsed = time.time() - self._start_time
        remaining = self.total_epochs - epoch
        new_epochs = epoch - self.completed_epochs  # epochs trained in this session
        eta = (elapsed / new_epochs) * remaining if new_epochs > 0 else 0
        best_flag = "  ★ NEW BEST" if is_best else ""

        summary = (
            f"  ── Epoch {epoch:3d}/{self.total_epochs} │ "
            f"mAP50={mAP50:.4f} │ "
            f"mAP50-95={mAP95:.4f} │ "
            f"loss={box_loss:.4f} │ "
            f"⏱ {elapsed / 60:.1f}m elapsed │ "
            f"ETA {eta / 60:.1f}m"
            f"{best_flag}"
        )

        from tqdm import tqdm as _tqdm

        _tqdm.write(summary)

        # Persist state after each epoch for crash recovery
        if self.run_dir:
            state = {
                "total_epochs": self.total_epochs,
                "completed_epochs": epoch,
                "best_map50": self._best_map50,
                "best_map95": self._best_map95,
                "elapsed_seconds": round(elapsed, 1),
                "updated_at": datetime.now().isoformat(),
            }
            try:
                import torch

                if torch.cuda.is_available():
                    state["gpu_alloc_gb"] = round(
                        torch.cuda.memory_allocated(0) / 1024**3, 2
                    )
                    state["gpu_reserved_gb"] = round(
                        torch.cuda.memory_reserved(0) / 1024**3, 2
                    )
                    state["gpu_max_alloc_gb"] = round(
                        torch.cuda.max_memory_allocated(0) / 1024**3, 2
                    )
                    torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass
            save_training_state(self.run_dir, state)

        # Check time limit: stop after this epoch if exceeded
        if self.max_time_seconds > 0 and elapsed > self.max_time_seconds:
            _tqdm.write(
                f"\n  ⏰ Time limit ({self.max_time_seconds / 3600:.1f}h) reached. "
                f"Stopping gracefully..."
            )
            with contextlib.suppress(Exception):
                trainer.stop_training = True
            self._time_limit_reached = True

    def on_train_end(self, trainer: object) -> None:
        """Clean up progress bars and print final summary."""
        if self._batch_pbar is not None:
            self._batch_pbar.close()
            self._batch_pbar = None
        if self._epoch_pbar is not None:
            self._epoch_pbar.close()
            self._epoch_pbar = None

        if self._history:
            self._print_final_summary()

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _extract_metrics(trainer: object) -> dict:
        """Safely extract metrics dictionary from trainer object."""
        rd: dict = {}
        try:
            metrics = getattr(trainer, "metrics", None)
            if metrics is None:
                return rd
            if hasattr(metrics, "results_dict"):
                rd = dict(metrics.results_dict)
            elif isinstance(metrics, dict):
                rd = dict(metrics)
        except Exception:
            pass
        return rd

    def _print_final_summary(self) -> None:
        """Print formatted training summary after completion."""
        total_time = time.time() - self._start_time
        h = self._history
        best_idx = max(range(len(h)), key=lambda i: h[i].get("mAP50", 0.0))

        print()
        print("=" * 62)
        print(
            "  TRAINING COMPLETE"
            + (" (time limit reached)" if self._time_limit_reached else "")
        )
        print("=" * 62)
        print(f"  Total epochs:       {self.total_epochs}")
        print(
            f"  Completed:          {len(h)} (started from epoch {self.completed_epochs + 1})"
        )
        print(
            f"  Total time:         {total_time / 60:.1f} min "
            f"({total_time / 3600:.2f} h)"
        )
        if len(h) > 0:
            print(f"  Avg per epoch:      {total_time / len(h):.1f}s")
        print(
            f"  Best mAP50:         {self._best_map50:.4f}  "
            f"(epoch {h[best_idx]['epoch']})"
        )
        print(f"  Best mAP50-95:      {self._best_map95:.4f}")
        if h:
            print(f"  Final mAP50:        {h[-1]['mAP50']:.4f}")
            print(f"  Final mAP50-95:     {h[-1]['mAP50-95']:.4f}")
        if self._time_limit_reached:
            print(f"  Next run will auto-resume from epoch {self.current_epoch}")
        print("=" * 62)

    # -- Context manager (optional) -----------------------------------------

    def __enter__(self) -> TrainingProgressCallback:
        return self

    def __exit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# TensorBoard launcher
# ---------------------------------------------------------------------------
def start_tensorboard(logdir: str) -> None:
    """Launch TensorBoard in a background subprocess."""
    import subprocess

    logdir = Path(logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Starting TensorBoard -> http://localhost:6006")
    print(f"       Logdir: {logdir}")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--port",
            "6006",
            "--bind_all",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Post-training visualization generator
# ---------------------------------------------------------------------------
def generate_training_plots(run_dir: Path) -> None:
    """Generate training progress plots from results.csv after training."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print("[WARN] No results.csv found, skipping plot generation")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plots")
        return

    print("\n[PLOTS] Generating training progress charts...")

    rows: list[dict] = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    epochs = list(range(1, len(rows) + 1))

    def col(key: str) -> list[float]:
        vals: list[float] = []
        for r in rows:
            try:
                v = float(r.get(key, float("nan")))
                vals.append(v if not math.isnan(v) else float("nan"))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        return vals

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLO COCO 2017 Training Progress", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax = axes[0][0]
    for k, color, label in [
        ("train/box_loss", "#2196F3", "Box Loss"),
        ("train/cls_loss", "#4CAF50", "Cls Loss"),
        ("train/dfl_loss", "#FF9800", "DFL Loss"),
    ]:
        vals = col(k)
        if vals and not all(math.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=color, label=label, linewidth=1.2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- mAP ---
    ax = axes[0][1]
    for k, color, label in [
        ("metrics/mAP50(B)", "#E91E63", "mAP50"),
        ("metrics/mAP50-95(B)", "#9C27B0", "mAP50-95"),
    ]:
        vals = col(k)
        if vals and not all(math.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=color, label=label, linewidth=1.5)
    ax.set_title("Validation mAP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Precision / Recall ---
    ax = axes[1][0]
    for k, color, label in [
        ("metrics/precision(B)", "#00BCD4", "Precision"),
        ("metrics/recall(B)", "#FF5722", "Recall"),
    ]:
        vals = col(k)
        if vals and not all(math.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=color, label=label, linewidth=1.2)
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- LR schedule ---
    ax = axes[1][1]
    for k in ["lr/pg0", "lr/pg1", "lr/pg2"]:
        vals = col(k)
        if vals and not all(math.isnan(v) for v in vals):
            ax.plot(epochs, vals, label=k, linewidth=1.0, alpha=0.7)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = run_dir / "training_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTS] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Crash diagnostics
# ---------------------------------------------------------------------------
def dump_crash_info(run_dir: Path, exc: Exception) -> str:
    """Log full crash context: traceback, GPU state, disk space."""
    crash_path = run_dir / "crash_info.log"
    lines: list[str] = []
    lines.append(f"=== CRASH at {datetime.now().isoformat()} ===")
    lines.append(f"Exception: {type(exc).__name__}: {exc}")
    lines.append(f"\nTraceback:\n{traceback.format_exc()}")

    try:
        import torch

        if torch.cuda.is_available():
            lines.append("\n--- GPU State ---")
            lines.append(
                f"allocated:  {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            )
            lines.append(
                f"reserved:   {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            )
            lines.append(
                f"max_alloc:  {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB"
            )
            lines.append(f"device:     {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    try:
        import shutil

        usage = shutil.disk_usage(str(run_dir))
        free_gb = usage.free / 1024**3
        lines.append(f"disk free:  {free_gb:.1f} GB")
    except Exception:
        pass

    crash_path.write_text("\n".join(lines), encoding="utf-8")
    return str(crash_path)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> object:
    """Run training with auto-resume and progress visualization."""
    import torch
    from ultralytics import YOLO

    # --- dataset ---
    data_yaml = resolve_dataset(args)
    if data_yaml is None:
        return 1

    # --- batch size ---
    gpu_mb = 0.0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    batch = args.batch if args.batch > 0 else auto_batch(args.model, gpu_mb)
    epochs = args.epochs

    # --- output directory ---
    name = args.name if args.name else DEFAULT_NAME
    run_dir = Path(DEFAULT_PROJECT) / name
    os.makedirs(run_dir, exist_ok=True)

    # --- auto-resume detection ---
    resume = False
    completed_epochs = 0
    if not args.fresh:
        ckpt_info = detect_checkpoint(run_dir, epochs)
        if ckpt_info["resume"]:
            # Check if training is already complete
            if ckpt_info["completed_epochs"] >= epochs:
                print(
                    f"[DONE] Training already completed "
                    f"({ckpt_info['completed_epochs']}/{epochs} epochs)."
                )
                print("  Use --fresh to restart, or --epochs to train more.")
                # Still show plots for existing results
                generate_training_plots(run_dir)
                return None
            resume = True
            completed_epochs = ckpt_info["completed_epochs"]
            print(f"[RESUME] Auto-resuming from epoch {completed_epochs}/{epochs}")
            model = YOLO(ckpt_info["checkpoint_path"])
        else:
            model_path = resolve_model(args)
            if model_path is None:
                return 1
            print(f"[INFO] No checkpoint found. Starting fresh from {model_path}")
            model = YOLO(model_path)
    else:
        model_path = resolve_model(args)
        if model_path is None:
            return 1
        print(f"[INFO] Fresh start requested. Loading: {model_path}")
        model = YOLO(model_path)

    # --- load previous training state for display ---
    prev_state = load_training_state(run_dir)
    if prev_state and not args.fresh:
        print(
            f"[STATE] Previous run: {prev_state.get('completed_epochs', '?')} epochs, "
            f"best mAP50={prev_state.get('best_map50', 0):.4f}"
        )

    # --- TensorBoard (optional) ---
    if args.tensorboard:
        start_tensorboard(str(run_dir / "tensorboard"))

    print("=" * 60)
    print(f"Dataset:     {data_yaml}")
    print(f"Model:       {args.model}")
    print(f"Epochs:      {epochs}")
    print(f"Image size:  {args.imgsz}")
    print(f"Batch size:  {batch}")
    print(f"Device:      {args.device}")
    print(f"Output dir:  {run_dir}")
    print(f"Resume:      {resume} (completed: {completed_epochs})")
    print(f"Fresh start: {args.fresh}")
    if args.max_time > 0:
        print(f"Max time:    {args.max_time:.1f}h")
    print("=" * 60)

    # --- max time in seconds ---
    max_time_s = args.max_time * 3600.0 if args.max_time > 0 else 0.0

    # --- train kwargs ---
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "device": args.device,
        "project": DEFAULT_PROJECT,
        "name": name,
        "exist_ok": True,
        "resume": resume,
        "pretrained": not resume,
        "optimizer": "auto",
        "save": True,
        "save_period": 1,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "verbose": False,
        "plots": False,
    }

    if gpu_mb > 0 and not args.amp:
        train_kwargs["amp"] = False
        train_kwargs["half"] = False
        print("[INFO] AMP/half disabled (use --amp to enable)")

    # --- register multi-level progress callback ---
    progress = TrainingProgressCallback(
        total_epochs=epochs,
        completed_epochs=completed_epochs,
        max_time_seconds=max_time_s,
        run_dir=run_dir,
    )
    model.add_callback("on_train_start", progress.on_train_start)
    model.add_callback("on_train_epoch_start", progress.on_train_epoch_start)
    model.add_callback("on_train_batch_end", progress.on_train_batch_end)
    model.add_callback("on_fit_epoch_end", progress.on_fit_epoch_end)
    model.add_callback("on_train_end", progress.on_train_end)

    print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if resume:
        print(f"Continuing from epoch {completed_epochs + 1}...")
    print("Live progress bars will appear below...")
    print("-" * 60)

    interrupted = False

    with InsomniaBlocker():
        try:
            t0 = time.perf_counter()
            results = model.train(**train_kwargs)
            elapsed = time.perf_counter() - t0
        except (KeyboardInterrupt, SystemExit):
            interrupted = True
            results = None
            elapsed = 0
            print("\n[INTERRUPT] Training interrupted. Checkpoint saved at last epoch.")
        except Exception as exc:
            crash_file = dump_crash_info(run_dir, exc)
            print(f"\n[CRASH] Training crashed: {type(exc).__name__}: {exc}")
            print(f"  Full diagnostic written to: {crash_file}")
            raise

    # --- post-training plots ---
    if not interrupted:
        generate_training_plots(run_dir)

    if results is not None:
        print(f"\nElapsed: {elapsed / 60:.1f} min  |  Results: {run_dir}")
        best_path = run_dir / "weights" / "best.pt"
        if best_path.exists():
            print(f"Best model:  {best_path}")
        print(f"Plots:       {run_dir / 'training_results.png'}")
        print("TensorBoard:  tensorboard --logdir " + str(run_dir))

    if interrupted:
        print("\nTo resume, simply run the same command again (auto-resume).")

    return results


def validate(args: argparse.Namespace) -> int:
    """Run validation on a trained model."""
    from ultralytics import YOLO

    model_path = args.weights
    if not model_path or not Path(model_path).exists():
        auto = Path(DEFAULT_PROJECT) / DEFAULT_NAME / "weights" / "best.pt"
        if auto.exists():
            model_path = str(auto)
        else:
            print(f"[ERROR] Model not found: {args.weights or auto}")
            return 1

    model = YOLO(model_path)
    data_yaml = resolve_dataset(args)
    if data_yaml is None:
        return 1

    results = model.val(data=data_yaml)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train YOLO on COCO 2017 with auto-resume and progress visualization"
    )
    parser.add_argument(
        "--model",
        default="yolo12n",
        help="Model (yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100, early stop via --patience)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--device", default="0", help="Device: cpu, 0, 0,1,2,3")
    parser.add_argument(
        "--dataset",
        default="coco2017",
        help="Dataset: coco2017, coco128, or path/to/data.yaml",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test: use coco128 with given epochs"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh start (ignore existing checkpoints)",
    )
    parser.add_argument(
        "--weights", default=None, help="Path to model weights (for val mode)"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard alongside training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="DataLoader workers (0=none, 1-2=ok on Windows)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without mAP improvement)",
    )
    parser.add_argument(
        "--name", default=None, help="Output run name (default: coco2017_train)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP mixed precision (faster but may be unstable on Windows)",
    )
    parser.add_argument(
        "--cache", action="store_true", help="Cache images in RAM (needs 20+ GB)"
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=0.0,
        help="Max training time in hours (0=unlimited). "
        "Training stops gracefully after current epoch.",
    )
    parser.add_argument(
        "--mode", default="train", choices=["train", "val"], help="Operation mode"
    )
    args = parser.parse_args()

    if not check_env():
        return 1

    run_dir = Path(DEFAULT_PROJECT) / DEFAULT_NAME
    os.makedirs(run_dir, exist_ok=True)

    if args.mode == "train":
        result = train(args)
        return 0 if result is not None else 0
    elif args.mode == "val":
        return validate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
