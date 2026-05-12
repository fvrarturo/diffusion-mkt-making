"""
91_book_resilience.py — Event-study of book queue dynamics around trade
events (the "does the synthetic book behave like a real book" figure).

Defines t=0 as each trade event. For all trades, computes the average
queue size on the depleted side at events t-W .. t+W around the trade.
Plots one curve per source (real, v2, v5, v9) on the same axes with
±1σ bands. Inspired by the LOB resiliency literature
(Bouchaud et al. 2009, Large 2007).

What the reader should see:
  real — the depleted side drops sharply at t=0, then recovers over
         10-20 events (queue replenishment).
  v2/v5 — because ~55% of events are trades, the surrounding window is
          contaminated; the dip-and-recovery pattern is smeared / absent.
  v9   — trade fraction matches real (~5%); inter-trade spacing is
         correct, so we can ask whether v9 actually reproduces the
         resilience pattern.

Cluster-bound: needs the raw stitched parquets at the same paths as
script 87 / 89.

Run on cluster login:
    python scripts/91_book_resilience.py \
        --real-dir data/processed/val \
        --synth-base ~/orcd/scratch/diffmm \
        --models v2 v5 v9 \
        --window 30 --max-trades 5000 \
        --out results/report_figures
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

SYNTH_LAYOUT = {
    "v2": "synthetic_v2_stitched/INTC",
    "v5": "synthetic_v5_stitched/INTC",
    "v9": "synthetic/intc_v9/INTC",
}

SOURCE_COLOR = {
    "real": "#000000",
    "v2":   "#1f77b4",
    "v5":   "#2ca02c",
    "v9":   "#9467bd",
}


def _gather_paths_real(real_dir: Path, ticker: str, max_files: int) -> list[Path]:
    return sorted(real_dir.glob(f"{ticker}_*.parquet"))[:max_files]


def _gather_paths_synth(synth_base: Path, model: str, max_files: int) -> list[Path]:
    rel = SYNTH_LAYOUT.get(model)
    if rel is None:
        return []
    base = synth_base / rel
    if not base.exists():
        return []
    # Balanced across regimes when subdirs exist
    regimes = ["base", "high_vol", "toxic", "thin"]
    regime_paths = [d for d in (base / r for r in regimes) if d.exists()]
    if regime_paths:
        n_per = max(1, max_files // len(regime_paths))
        out: list[Path] = []
        for d in regime_paths:
            out.extend(sorted(d.glob("*.parquet"))[:n_per])
        return out
    return sorted(base.glob("**/*.parquet"))[:max_files]


def _load_concat(paths: list[Path], cols: list[str]) -> pd.DataFrame:
    """Load + concat a list of parquets, taking only the requested columns
    that exist. Tape boundaries get a sentinel `_tape_id` so we don't
    align trade events across two adjacent tapes."""
    if not paths:
        return pd.DataFrame()
    chunks = []
    for tid, p in enumerate(paths):
        try:
            d = pl.read_parquet(p).to_pandas()
        except Exception as e:
            print(f"  WARN: {p.name}: {e}")
            continue
        present = [c for c in cols if c in d.columns]
        d = d[present].copy()
        d["_tape_id"] = tid
        chunks.append(d)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _ensure_trade_sign(df: pd.DataFrame) -> pd.Series:
    """Return a +1/-1 series for trade direction. Fall back to inferring
    from trade_px vs midprice if trade_sign is absent."""
    if "trade_sign" in df.columns:
        return pd.to_numeric(df["trade_sign"], errors="coerce").fillna(0.0)
    if "trade_px" in df.columns and "bid_px" in df.columns and "ask_px" in df.columns:
        mid = (pd.to_numeric(df["bid_px"], errors="coerce") +
               pd.to_numeric(df["ask_px"], errors="coerce")) / 2.0
        tp = pd.to_numeric(df["trade_px"], errors="coerce")
        sign = pd.Series(0.0, index=df.index)
        sign[tp > mid] = 1.0
        sign[tp < mid] = -1.0
        return sign
    return pd.Series(0.0, index=df.index)


def _is_trade(df: pd.DataFrame) -> pd.Series:
    if "trade_sz" in df.columns:
        return pd.to_numeric(df["trade_sz"], errors="coerce").fillna(0.0) > 0
    if "event_type" in df.columns:
        return df["event_type"].astype(str).str.lower().str.contains("trade", na=False)
    return pd.Series(False, index=df.index)


def event_study(df: pd.DataFrame, window: int, max_trades: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, int]:
    """For each trade event, extract the queue size on the depleted side
    (ask_sz for buys, bid_sz for sells) at t-window..t+window around the
    trade.

    Returns (median_curve, q25_curve, q75_curve, n_trades_used).

    Per-trade normalization uses the **median of the pre-trade window**
    [t-window, t-1] rather than a single point at t=-window. The
    single-point normalizer was unstable: when q[t=-W] happened to be
    very small, the ratio blew up and dragged both the mean and the
    variance away from the true dip-and-recover signal.

    Across-trade aggregation uses the **median + IQR** rather than mean
    + std for the same robustness reason: a few outlier-trades shouldn't
    dominate the curve.
    """
    if df.empty or "bid_sz" not in df.columns or "ask_sz" not in df.columns:
        z = np.zeros(2 * window + 1)
        return z, z.copy(), z.copy(), 0
    bid_sz = pd.to_numeric(df["bid_sz"], errors="coerce").fillna(0.0).values
    ask_sz = pd.to_numeric(df["ask_sz"], errors="coerce").fillna(0.0).values
    sign = _ensure_trade_sign(df).values
    is_tr = _is_trade(df).values
    tape_id = df["_tape_id"].values if "_tape_id" in df.columns else np.zeros(len(df), dtype=int)
    n = len(df)

    # Find trade events with non-zero sign and a clean window inside the
    # same tape (no boundary crossings).
    cand = np.where(is_tr & (sign != 0))[0]
    valid = []
    for t in cand:
        lo, hi = t - window, t + window
        if lo < 0 or hi >= n:
            continue
        if tape_id[lo] != tape_id[hi]:
            continue
        valid.append(t)
    if not valid:
        z = np.zeros(2 * window + 1)
        return z, z.copy(), z.copy(), 0

    # Subsample for speed if too many
    if len(valid) > max_trades:
        valid = list(rng.choice(valid, size=max_trades, replace=False))

    # Stack: rows = trades, cols = relative offset (-window..+window).
    # "Depleted side" = ask if sign=+1 (buy hit ask), bid if sign=-1.
    depleted = np.empty((len(valid), 2 * window + 1))
    for i, t in enumerate(valid):
        win_b = bid_sz[t - window:t + window + 1]
        win_a = ask_sz[t - window:t + window + 1]
        depleted[i] = win_a if sign[t] > 0 else win_b

    # Per-trade normalization: divide each row by the MEDIAN of its
    # pre-trade window [t-window, t-1] (i.e. cols 0..window-1, exclusive
    # of t=0). Robust against single-point outliers at the leftmost
    # offset.
    pre = depleted[:, :window]
    pre_median = np.median(pre, axis=1, keepdims=True)
    pre_median = np.where(pre_median <= 0, np.nan, pre_median)
    norm = depleted / pre_median  # values centered around 1.0 pre-trade

    # Drop rows with any non-finite (zeros in pre-window create NaN)
    norm = norm[np.isfinite(norm).all(axis=1)]
    if len(norm) == 0:
        z = np.zeros(2 * window + 1)
        return z, z.copy(), z.copy(), 0

    # Across-trade aggregation: median + IQR (Q25, Q75)
    med = np.median(norm, axis=0)
    q25 = np.quantile(norm, 0.25, axis=0)
    q75 = np.quantile(norm, 0.75, axis=0)
    return med, q25, q75, len(norm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path,
                    default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--models", nargs="+", default=["v2", "v5", "v9"])
    ap.add_argument("--max-files-real", type=int, default=5)
    ap.add_argument("--max-files-synth", type=int, default=400)
    ap.add_argument("--window", type=int, default=30,
                    help="event offsets around trade (-W..+W)")
    ap.add_argument("--max-trades", type=int, default=5000,
                    help="cap per-source trade count for averaging speed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "report_figures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
    rng = np.random.default_rng(42)

    cols_needed = ["bid_sz", "ask_sz", "trade_sz", "trade_sign",
                   "trade_px", "bid_px", "ask_px", "event_type"]

    # Each tuple: (label, median_curve, q25_curve, q75_curve, n_trades)
    sources: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []

    print("loading real …")
    real_paths = _gather_paths_real(args.real_dir, args.ticker, args.max_files_real)
    real_df = _load_concat(real_paths, cols_needed)
    if not real_df.empty:
        med, q25, q75, n = event_study(real_df, args.window, args.max_trades, rng)
        print(f"  real: {n:,} trade events used")
        sources.append(("real", med, q25, q75, n))

    for model in args.models:
        print(f"loading {model} …")
        paths = _gather_paths_synth(args.synth_base, model, args.max_files_synth)
        if not paths:
            print(f"  SKIP: no parquets for {model}")
            continue
        df = _load_concat(paths, cols_needed)
        if df.empty:
            print(f"  SKIP: empty load for {model}")
            continue
        med, q25, q75, n = event_study(df, args.window, args.max_trades, rng)
        print(f"  {model}: {n:,} trade events used")
        sources.append((model, med, q25, q75, n))

    if len(sources) < 1:
        sys.exit("FATAL: no sources to plot")

    # ── Figure ──────────────────────────────────────────────────────────────
    offsets = np.arange(-args.window, args.window + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, med_curve, q25_curve, q75_curve, n in sources:
        c = SOURCE_COLOR.get(label, "#444")
        ax.plot(offsets, med_curve, color=c, linewidth=2.2,
                label=f"{label}  (n_trades={n:,})", alpha=0.95)
        ax.fill_between(offsets, q25_curve, q75_curve,
                        color=c, alpha=0.12, linewidth=0)
    ax.axvline(0, color="red", linestyle=":", linewidth=1, alpha=0.7,
               label="trade event (t = 0)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("event offset relative to trade event  (events, not seconds)")
    ax.set_ylabel("queue size on depleted side\n(normalized to pre-trade window median)")
    ax.set_title("Book Resilience: Queue Replenishment Around Trade Events\n"
                 "Depleted side = ask for buy trades, bid for sell trades. "
                 "Curves = median across trades; bands = IQR.",
                 pad=10)
    ax.legend(loc="lower right", fontsize=10)
    # Tight y-range around the typical [0.5, 1.5] band where the dip lives;
    # extreme outliers are already excluded from the median + IQR aggregation.
    ax.set_ylim(0.4, 1.6)
    fig.tight_layout()

    pdf = args.out / "fig_book_resilience.pdf"
    png = args.out / "fig_book_resilience.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")

    # Persist the per-source curves so the report can refer to numerics
    # even if the figure changes later.
    rows = []
    for label, med_curve, q25_curve, q75_curve, n in sources:
        for off, m, lo, hi in zip(offsets, med_curve, q25_curve, q75_curve):
            rows.append({"source": label, "n_trades": n,
                         "offset": int(off),
                         "median_norm_q": float(m),
                         "q25_norm_q": float(lo),
                         "q75_norm_q": float(hi)})
    pd.DataFrame(rows).to_csv(args.out / "book_resilience_curves.csv",
                              index=False)
    print(f"wrote {args.out / 'book_resilience_curves.csv'}")


if __name__ == "__main__":
    main()
