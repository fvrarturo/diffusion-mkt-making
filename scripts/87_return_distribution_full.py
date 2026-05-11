"""
87_return_distribution_full.py — Replace the sparse Fig 2.3 (3 tail thresholds)
with a proper TRADES-Fig-3 style full return-distribution comparison:

  Panel A: Log-y density of mid_return for real + each synth (overlaid lines).
           Heavy tails are immediately visible — flatter slope on log-log =
           heavier tails. Real INTC's kurtosis ~755 should appear as a much
           wider, flatter density than (e.g.) v9's kurtosis ~129.

  Panel B: Q-Q plot of synth quantiles vs real quantiles for mid_return.
           A perfectly-fitting model lies on the diagonal. Heavy-tailed
           generators bow away from the diagonal in the tails.

Cluster-bound (loads ~M-row return arrays from val + synth parquets).

Run on cluster login node:
    python scripts/87_return_distribution_full.py \
        --real-dir data/processed/val \
        --synth-base $HOME/orcd/scratch/diffmm \
        --models v2 v5 v9 v3_e9_noclip \
        --out results/trades_style

Includes v3_e9_noclip by default to surface the scale-exploded tail in the
density plot — it's the single most informative contrast point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent

# Same SYNTH_LAYOUT as scripts 83 / 86. Keep in sync.
SYNTH_LAYOUT = {
    "v2":          "synthetic_v2_stitched/INTC",
    "v2_noclip":   "synthetic_v2_noclip_stitched/INTC",
    "v2_remapped": "synthetic_v2_remapped/INTC",
    "v3_e9":       "synthetic_v3_e9_stitched/INTC",
    "v3_e9_noclip":"synthetic_v3_e9_noclip_stitched/INTC",
    "v3_e19":      "synthetic_v3_e19_stitched/INTC",
    "v3p5":        "synthetic_v3p5_stitched/INTC",
    "v3p5_noclip": "synthetic_v3p5_noclip_stitched/INTC",
    "v4":          "synthetic_v4_stitched/INTC",
    "v4_noclip":   "synthetic_v4_noclip_stitched/INTC",
    "v5":          "synthetic_v5_stitched/INTC",
    "v6":          "synthetic_v6_stitched/INTC",
    "v7":          "synthetic_v7_stitched/INTC",
    "v7_b":        "synthetic_v7_b_stitched/INTC",
    "v8":          "synthetic/intc_v8/INTC",
    "v8_b":        "synthetic/intc_v8_b/INTC",
    "v9":          "synthetic/intc_v9/INTC",
    "v9_b":        "synthetic/intc_v9_b/INTC",
    "v10_cascade": "synthetic/intc_v10_cascade/INTC",
    "v11":         "synthetic/intc_v11/INTC",
}

MODEL_COLOR = {
    "real": "#000000", "v2": "#1f77b4", "v5": "#2ca02c",
    "v9": "#9467bd", "v7_b": "#ff7f0e", "v8": "#bcbd22",
    "v2_noclip": "#d62728", "v3_e9_noclip": "#d62728", "v4_noclip": "#d62728",
    "v2_remapped": "#e377c2",
}


def _color(model: str) -> str:
    return MODEL_COLOR.get(model, "#888")


def load_returns(real_dir: Path, ticker: str, max_files: int,
                  max_n: int = 5_000_000) -> np.ndarray:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))[:max_files]
    if not paths:
        sys.exit(f"FATAL: no real parquets at {real_dir}")
    print(f"loading real: {len(paths)} {ticker} parquets")
    chunks = []
    n = 0
    for p in paths:
        d = pl.read_parquet(p, columns=["mid_return"]).to_pandas()
        v = d["mid_return"].dropna().values
        chunks.append(v)
        n += len(v)
        if n >= max_n:
            break
    arr = np.concatenate(chunks)
    if len(arr) > max_n:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=max_n, replace=False)
    print(f"  {len(arr):,} real return events")
    return arr


def load_synth_returns(synth_base: Path, model: str, max_files: int,
                        max_n: int = 5_000_000) -> np.ndarray:
    rel = SYNTH_LAYOUT.get(model)
    if rel is None:
        return np.array([])
    base = synth_base / rel
    if not base.exists():
        return np.array([])
    paths = sorted(base.glob("**/*.parquet"))[:max_files]
    if not paths:
        return np.array([])
    print(f"  [{model}] loading {len(paths)} parquets")
    chunks = []
    n = 0
    for p in paths:
        try:
            d = pl.read_parquet(p, columns=["mid_return"]).to_pandas()
            v = d["mid_return"].dropna().values
            chunks.append(v)
            n += len(v)
            if n >= max_n:
                break
        except Exception as e:
            print(f"    WARN: {p.name}: {e}")
    if not chunks:
        return np.array([])
    arr = np.concatenate(chunks)
    if len(arr) > max_n:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=max_n, replace=False)
    print(f"  {model}: {len(arr):,} return events")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path,
                    default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--models", nargs="+",
                    default=["v2", "v5", "v9", "v3_e9_noclip"])
    ap.add_argument("--max-files-real", type=int, default=10)
    ap.add_argument("--max-files-synth", type=int, default=200)
    ap.add_argument("--max-n-points", type=int, default=2_000_000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "trades_style")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

    real_ret = load_returns(args.real_dir, args.ticker,
                              args.max_files_real, args.max_n_points)
    synth_returns: dict[str, np.ndarray] = {}
    for m in args.models:
        arr = load_synth_returns(args.synth_base, m,
                                  args.max_files_synth, args.max_n_points)
        if len(arr):
            synth_returns[m] = arr

    # ── Panel A: log-y density (kernel density via histogram) ────────────────
    # Use a wide symmetric range covering the union of the data, with log-y so
    # heavy tails read as "less-steep slope".
    all_data = np.concatenate([real_ret] + list(synth_returns.values()))
    all_data = all_data[np.isfinite(all_data)]
    if len(all_data) == 0:
        sys.exit("FATAL: all data empty")
    p_lo, p_hi = np.percentile(all_data, [0.01, 99.99])
    edge = max(abs(p_lo), abs(p_hi)) * 1.1
    bins = np.linspace(-edge, edge, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    sources = [("real", real_ret)] + list(synth_returns.items())
    for label, arr in sources:
        if len(arr) < 100:
            continue
        h, edges = np.histogram(arr, bins=bins, density=True)
        c = (edges[:-1] + edges[1:]) / 2
        nz = h > 0
        kurt = scipy_stats.kurtosis(arr[np.isfinite(arr)])
        ax.plot(c[nz], h[nz], color=_color(label),
                linewidth=2.5 if label == "real" else 1.5,
                label=f"{label}  (kurt={kurt:.0f})", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("mid_return")
    ax.set_ylabel("density (log)")
    ax.set_title("A. Full Return Density (log-y)\n"
                 "Heavier tails ↔ flatter wings", pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel B: Q-Q plot of synth vs real for the synth set (excluding real) ─
    ax = axes[1]
    qs = np.linspace(0.001, 0.999, 199)
    real_q = np.quantile(real_ret, qs)
    for label, arr in synth_returns.items():
        if len(arr) < 100:
            continue
        synth_q = np.quantile(arr, qs)
        ax.plot(real_q, synth_q, color=_color(label),
                linewidth=1.6, label=label, alpha=0.9, marker="o", markersize=2.5)
    # Diagonal reference line
    diag = np.linspace(real_q.min(), real_q.max(), 100)
    ax.plot(diag, diag, "k--", linewidth=1, alpha=0.7,
            label="y = x  (perfect match)")
    ax.set_xlabel("real quantile")
    ax.set_ylabel("synth quantile")
    ax.set_title("B. Q-Q Plot (synth vs real)\n"
                 "Heavy-tail miss ↔ bowing away from diagonal in extremes",
                 pad=8)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Mid-Return Distribution — Real INTC vs Selected Generators",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    pdf = args.out / "fig_return_distribution_full.pdf"
    png = args.out / "fig_return_distribution_full.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")

    # Persist per-model summary
    rows = []
    for label, arr in [("real", real_ret)] + list(synth_returns.items()):
        if len(arr) < 100:
            continue
        rows.append({
            "model": label,
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "skew": float(scipy_stats.skew(arr)),
            "kurtosis": float(scipy_stats.kurtosis(arr)),
            "p001":  float(np.quantile(arr, 0.001)),
            "p01":   float(np.quantile(arr, 0.01)),
            "p99":   float(np.quantile(arr, 0.99)),
            "p999":  float(np.quantile(arr, 0.999)),
            "max":   float(np.max(np.abs(arr))),
            "frac_zero": float(np.mean(arr == 0)),
        })
    if rows:
        pd.DataFrame(rows).to_csv(
            args.out / "return_distribution_stats.csv", index=False)
        print(f"wrote {args.out / 'return_distribution_stats.csv'}")


if __name__ == "__main__":
    main()
