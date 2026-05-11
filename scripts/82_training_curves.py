"""
82_training_curves.py — Plot training & validation loss curves for v2 / v5 / v9.

Reads Lightning's CSVLogger output (one metrics.csv per run, possibly multiple
version_* dirs per run) from a local mirror of:
    ~/orcd/scratch/diffmm/checkpoints/<run>/lightning_logs/version_*/metrics.csv

By default looks under either:
    ~/orcd/scratch/diffmm/checkpoints/  (if you mounted scratch locally)
    or  ./checkpoints_lightning/        (after a manual rsync)

Outputs:
    results/report_figures/fig_b_training_curves.{pdf,png}
    results/report_figures/training_curves_summary.csv

Run:
    python scripts/82_training_curves.py
    python scripts/82_training_curves.py --runs intc_v2 intc_v5 intc_v9 --models v2 v5 v9
    python scripts/82_training_curves.py --logs-root ~/Desktop/scratch_pull/checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_LOGS_ROOTS = [
    Path.home() / "orcd" / "scratch" / "diffmm" / "checkpoints",
    ROOT / "checkpoints_lightning",
]

DEFAULT_RUNS = ["intc_v2", "intc_v5", "intc_v9"]
DEFAULT_LABELS = ["v2", "v5", "v9"]

MODEL_COLORS = {"v2": "#1f77b4", "v5": "#2ca02c", "v9": "#9467bd"}


def find_metrics_csvs(logs_root: Path, run: str) -> list[Path]:
    """Find all version_*/metrics.csv under <logs_root>/<run>/lightning_logs/."""
    base = logs_root / run / "lightning_logs"
    if not base.exists():
        return []
    return sorted(base.glob("version_*/metrics.csv"))


def concat_runs(metrics_paths: list[Path]) -> pd.DataFrame:
    """Concat multiple version_*/metrics.csv files in chronological order
    (later versions are resumes / continuations)."""
    frames = []
    offset = 0
    for p in metrics_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  WARN: skipped {p}: {e}", file=sys.stderr)
            continue
        if df.empty or "step" not in df.columns:
            continue
        # Distinct step namespace per version (resumes restart counter)
        df = df.copy()
        df["global_step"] = df["step"].fillna(0).astype(int) + offset
        offset = int(df["global_step"].max()) + 1
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="Lightning run dir names")
    ap.add_argument("--models", nargs="+", default=DEFAULT_LABELS,
                    help="Display labels (one per run)")
    ap.add_argument("--logs-root", type=Path, default=None,
                    help="Override logs root (defaults probe scratch + local mirror)")
    args = ap.parse_args()

    if len(args.runs) != len(args.models):
        ap.error("--runs and --models must have the same length")

    roots = [args.logs_root] if args.logs_root else DEFAULT_LOGS_ROOTS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary_rows: list[dict] = []

    for run, label in zip(args.runs, args.models):
        paths: list[Path] = []
        for r in roots:
            paths = find_metrics_csvs(r, run)
            if paths:
                print(f"[{label}] found {len(paths)} version dir(s) under {r}")
                break
        if not paths:
            print(f"[{label}] no metrics.csv found in {[str(r) for r in roots]}")
            continue
        df = concat_runs(paths)
        if df.empty:
            print(f"[{label}] empty metrics — skipped")
            continue

        color = MODEL_COLORS.get(label, "gray")

        # Train loss (probably named "train/loss_step" or "train_loss" or just "loss")
        train_col = next((c for c in ["train/loss_step", "train_loss",
                                       "train/loss", "loss"] if c in df.columns),
                         None)
        if train_col:
            sub = df.dropna(subset=[train_col])
            axes[0].plot(sub["global_step"], sub[train_col],
                         color=color, label=label, alpha=0.7, linewidth=1.2)

        # Val loss (probably "val/loss" or "val_loss")
        val_col = next((c for c in ["val/loss", "val_loss",
                                     "val/loss_epoch"] if c in df.columns),
                       None)
        if val_col:
            sub = df.dropna(subset=[val_col])
            axes[1].plot(sub["global_step"], sub[val_col],
                         color=color, marker="o", markersize=4,
                         label=label, linewidth=1.5)

        summary_rows.append({
            "model": label,
            "n_steps": int(df["global_step"].max()),
            "n_epochs": int(df["epoch"].max()) if "epoch" in df.columns else np.nan,
            "best_val": float(df[val_col].min()) if val_col else np.nan,
            "final_train": float(df[train_col].dropna().iloc[-1])
                              if train_col and df[train_col].notna().any() else np.nan,
        })

    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train loss (log)")
    axes[0].set_title("Training loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("val loss (log)")
    axes[1].set_title("Validation loss (per epoch)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss — v2 / v5 / v9",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    pdf = OUT / "fig_b_training_curves.pdf"
    png = OUT / "fig_b_training_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = OUT / "training_curves_summary.csv"
        df_sum.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper")
    main()
