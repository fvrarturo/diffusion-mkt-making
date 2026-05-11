"""
86_intraday_lob_volume.py — Reproduce TRADES Fig 4 (and improve on it):
intraday level-1 bid/ask queue size as a function of session time, with the
5-95 percentile envelope across days/tapes.

Key design call: real INTC has an intraday clock (ts_ns) so we bucket by
minute-of-session. Synthetic tapes do NOT have meaningful intraday
timestamps — they are 256-event windows stitched together. We therefore
plot the synth equivalent as bid/ask mean ± envelope across windows,
indexed by EVENT POSITION within the window. The contrast itself is a
finding: real has a clear U-shape and bid/ask asymmetry, while every synth
generator we've tested is uniform across the session because no model has
a time-of-day mechanism.

4 panels (real, v2, v5, v9 — adjustable via --models). Each panel shows:
  - Solid line: mean(bid_sz) and mean(ask_sz) per bucket
  - Shaded band: 5-95 percentile envelope per bucket
  - Annotation: bid/ask asymmetry ratio (mean_bid / mean_ask)

Reads:
  data/processed/val/INTC_*.parquet                      (real, 10 days)
  $HOME/orcd/scratch/diffmm/synthetic_<model>_stitched/INTC/<regime>/*.parquet
  (or .../synthetic/intc_<model>/INTC/<regime>/*.parquet for v8/v9 etc.)

Writes:
  results/trades_style/fig_intraday_lob_volume.{pdf,png}
  results/trades_style/intraday_volume_stats.csv

Cluster-bound (needs val parquets). Run on the cluster login node:
    python scripts/86_intraday_lob_volume.py \
        --real-dir data/processed/val \
        --synth-base $HOME/orcd/scratch/diffmm \
        --models v2 v5 v9 \
        --out results/trades_style
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

ROOT = Path(__file__).resolve().parent.parent

# Same SYNTH_LAYOUT used by script 83. Keep in sync.
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

MODEL_COLOR = {"real": "#000000", "v2": "#1f77b4", "v5": "#2ca02c",
               "v9": "#9467bd", "v7_b": "#ff7f0e", "v8": "#bcbd22"}


# ─── Loaders ───────────────────────────────────────────────────────────────
def load_real(real_dir: Path, ticker: str, max_files: int) -> pd.DataFrame:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))[:max_files]
    if not paths:
        sys.exit(f"FATAL: no real parquets at {real_dir}")
    print(f"loading real: {len(paths)} {ticker} parquets")
    frames = [pl.read_parquet(p, columns=[
        "ts_ns", "bid_sz", "ask_sz", "event_type"
    ] if "ts_ns" in pl.read_parquet_schema(p) else None
    ).to_pandas() for p in paths]
    return pd.concat(frames, ignore_index=True)


def load_synth(synth_base: Path, model: str, max_files: int) -> pd.DataFrame:
    rel = SYNTH_LAYOUT.get(model)
    if rel is None:
        return pd.DataFrame()
    base = synth_base / rel
    if not base.exists():
        return pd.DataFrame()
    paths = sorted(base.glob("**/*.parquet"))[:max_files]
    if not paths:
        return pd.DataFrame()
    print(f"  [{model}] loading {len(paths)} parquets")
    frames = []
    for p in paths:
        try:
            d = pl.read_parquet(p).to_pandas()
            # Mark each tape with a unique window_id so we can group when
            # computing per-window event-position curves.
            d["_window_id"] = p.stem
            frames.append(d)
        except Exception as e:
            print(f"    WARN: {p.name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── Bucketing ─────────────────────────────────────────────────────────────
def real_per_minute_stats(df: pd.DataFrame, n_buckets: int = 26) -> pd.DataFrame:
    """Bucket real events by minute-of-session ET (09:30-16:00 = 390 min →
    n_buckets=26 → 15-min buckets). Returns per-bucket bid/ask mean + p05/p95."""
    if "ts_ns" not in df.columns:
        return pd.DataFrame()
    ts = pd.to_datetime(df["ts_ns"].astype("int64"), unit="ns", utc=True) \
            .dt.tz_convert("America/New_York")
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    # Trading session is 9:30 - 16:00; we drop pre/post session.
    in_session = (minute_of_day >= 9 * 60 + 30) & (minute_of_day <= 16 * 60)
    df = df.loc[in_session].copy()
    minute = (minute_of_day[in_session] - (9 * 60 + 30))   # 0..390
    bins = np.linspace(0, 390, n_buckets + 1).astype(int)
    df["_bucket"] = np.clip(np.searchsorted(bins, minute, side="right") - 1,
                              0, n_buckets - 1)
    out = df.groupby("_bucket").agg(
        bid_mean=("bid_sz", "mean"), bid_p05=("bid_sz", lambda x: np.percentile(x, 5)),
        bid_p95=("bid_sz", lambda x: np.percentile(x, 95)),
        ask_mean=("ask_sz", "mean"), ask_p05=("ask_sz", lambda x: np.percentile(x, 5)),
        ask_p95=("ask_sz", lambda x: np.percentile(x, 95)),
        n=("bid_sz", "size"),
    ).reset_index()
    out["bucket_min"] = bins[out["_bucket"]] + (390 // n_buckets) // 2
    return out


def synth_per_event_position(df: pd.DataFrame, window_size: int = 256) -> pd.DataFrame:
    """For synth (no real intraday clock), bucket by event position within the
    256-event window, aggregated across all windows. Tells us whether the
    generator produces ANY non-uniform pattern across the synthesized session."""
    if df.empty or "_window_id" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["_pos"] = df.groupby("_window_id").cumcount()
    df = df[df["_pos"] < window_size]
    n_buckets = 26
    df["_bucket"] = (df["_pos"] // (window_size // n_buckets)).clip(0, n_buckets - 1)
    out = df.groupby("_bucket").agg(
        bid_mean=("bid_sz", "mean"), bid_p05=("bid_sz", lambda x: np.percentile(x, 5)),
        bid_p95=("bid_sz", lambda x: np.percentile(x, 95)),
        ask_mean=("ask_sz", "mean"), ask_p05=("ask_sz", lambda x: np.percentile(x, 5)),
        ask_p95=("ask_sz", lambda x: np.percentile(x, 95)),
        n=("bid_sz", "size"),
    ).reset_index()
    out["bucket_min"] = (out["_bucket"] * (390 // n_buckets)).astype(int)
    return out


# ─── Plotting ──────────────────────────────────────────────────────────────
def plot_panel(ax, stats: pd.DataFrame, title: str, *, color: str,
                show_xlabel: bool = False, show_ylabel: bool = False) -> dict:
    if stats.empty:
        ax.set_visible(False)
        return {}
    x = stats["bucket_min"]
    # Bid (solid) and Ask (dashed)
    ax.plot(x, stats["bid_mean"], color=color, linewidth=2, label="bid mean")
    ax.fill_between(x, stats["bid_p05"], stats["bid_p95"],
                    color=color, alpha=0.18, label="bid 5-95")
    ax.plot(x, stats["ask_mean"], color=color, linewidth=2, linestyle="--",
            label="ask mean")
    ax.fill_between(x, stats["ask_p05"], stats["ask_p95"],
                    color=color, alpha=0.10, hatch="//",
                    label="ask 5-95")
    asym = float(stats["bid_mean"].mean() / max(stats["ask_mean"].mean(), 1e-9))
    bid_total = float(stats["bid_mean"].sum())
    ask_total = float(stats["ask_mean"].sum())
    ax.set_title(f"{title}\n"
                 f"$\\langle\\mathrm{{bid}}\\rangle/\\langle\\mathrm{{ask}}\\rangle$ = {asym:.3f}",
                 fontsize=11)
    if show_xlabel:
        ax.set_xlabel("session minute (0 = 09:30 ET)")
    if show_ylabel:
        ax.set_ylabel("level-1 queue size (shares)")
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    return {"asymmetry_bid_over_ask": asym,
            "bid_total": bid_total, "ask_total": ask_total,
            "n_buckets": int(len(stats)), "n_events": int(stats["n"].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--models", nargs="+", default=["v2", "v5", "v9"])
    ap.add_argument("--max-files-real", type=int, default=10)
    ap.add_argument("--max-files-synth", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "trades_style")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

    # Load
    real_df = load_real(args.real_dir, args.ticker, args.max_files_real)
    synth_dfs = {m: load_synth(args.synth_base, m, args.max_files_synth)
                 for m in args.models}

    # Compute stats
    real_stats = real_per_minute_stats(real_df)
    synth_stats = {m: synth_per_event_position(df) for m, df in synth_dfs.items()}

    # Layout: 1 row × (1 + N) panels (real + N synth). Cap at 4 panels for layout.
    panels = [("real", real_stats)] + \
             [(m, synth_stats[m]) for m in args.models if not synth_stats[m].empty]
    n = min(len(panels), 4)
    panels = panels[:n]
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    summary_rows = []
    for i, (label, stats) in enumerate(panels):
        ax = axes[i]
        info = plot_panel(ax, stats,
                           title=f"{label}" + (
                               "  (intraday minute)" if label == "real"
                               else "  (event position)"
                           ),
                           color=MODEL_COLOR.get(label, "gray"),
                           show_xlabel=True, show_ylabel=(i == 0))
        if info:
            summary_rows.append({"model": label, **info})

    fig.suptitle("Level-1 LOB Queue Size — Real Has Intraday U-Shape, "
                 "Synth Is Uniform Across the Synthesized Window",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    pdf = args.out / "fig_intraday_lob_volume.pdf"
    png = args.out / "fig_intraday_lob_volume.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")

    # Persist per-model summary CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.out / "intraday_volume_stats.csv",
                                            index=False)
        print(f"wrote {args.out / 'intraday_volume_stats.csv'}")


if __name__ == "__main__":
    main()
