"""
89_event_tape_walkthrough.py — Event-level tape walkthrough figure.

The "what does the data actually look like" figure that no aggregate
metric in the paper exposes. 4 columns (real, v2, v5, v9) × 4 rows
(mid-price step function, queue depth bid+ask, spread, trade markers).
X-axis is event index in a 200-event slice taken from the middle of one
representative tape per source.

What the reader sees, by column:
  real  — mid_price is a staircase (89% zeros), trades sparse, spread
          pinned at 1 tick with brief widenings, queue smooth
  v2    — mid_price drifts continuously (no flat stretches), trades on
          most events (frac_zero ~0)
  v5    — like v2 but with heavier tails on the few moves
  v9    — mid_price has flat-then-jump structure (frac_zero=0.99),
          trades rare, but jump pattern may differ from real

Cluster-bound: needs the raw stitched parquets at
  data/processed/val/INTC_*.parquet                          (real)
  ~/orcd/scratch/diffmm/synthetic_v{2,5}_stitched/INTC/...   (v2, v5)
  ~/orcd/scratch/diffmm/synthetic/intc_v9/INTC/...           (v9)

Run on cluster login:
    python scripts/89_event_tape_walkthrough.py \
        --real-dir data/processed/val \
        --synth-base ~/orcd/scratch/diffmm \
        --models v2 v5 v9 \
        --events 200 \
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

# Same SYNTH_LAYOUT as scripts 83 / 86 / 87 — keep in sync.
SYNTH_LAYOUT = {
    "v2":          "synthetic_v2_stitched/INTC",
    "v5":          "synthetic_v5_stitched/INTC",
    "v9":          "synthetic/intc_v9/INTC",
}

# Distinct color per protagonist (matches PROTAGONIST_COLOR in script 80).
SOURCE_COLOR = {
    "real": "#000000",
    "v2":   "#1f77b4",
    "v5":   "#2ca02c",
    "v9":   "#9467bd",
}


def _safe_read(path: Path) -> pd.DataFrame:
    """Read a parquet file, return as pandas (empty on error)."""
    try:
        return pl.read_parquet(path).to_pandas()
    except Exception as e:
        print(f"  WARN: failed to read {path}: {e}")
        return pd.DataFrame()


def _concat_until(paths: list[Path], min_events: int, label: str) -> pd.DataFrame:
    """Read parquets one-by-one starting from the middle of the sorted
    list, concatenating until we accumulate `min_events` events or
    exhaust the list.

    Why: per-tape lengths are not uniform across sources. v2/v5 stitched
    parquets are typically 2560 events (10 diffusion windows × 256), but
    v9 stitched parquets at synthetic/intc_v9/INTC/... are often only
    256 events (single window, unstitched). For the walkthrough figure
    to render the same x-axis length per source we must concatenate
    adjacent tapes when one tape is too short.

    Tape boundaries appear as natural discontinuities in the rendered
    series (mid-price jumps, queue resets); we accept that as visible
    seams rather than try to smooth them away.
    """
    if not paths:
        return pd.DataFrame()
    # Walk outward from the middle so we use representative tapes
    mid = len(paths) // 2
    order = []
    for offset in range(len(paths)):
        if offset == 0:
            order.append(mid)
        else:
            for sign in (1, -1):
                idx = mid + sign * offset
                if 0 <= idx < len(paths):
                    order.append(idx)
    seen: set[int] = set()
    chunks: list[pd.DataFrame] = []
    n_total = 0
    used: list[str] = []
    for idx in order:
        if idx in seen:
            continue
        seen.add(idx)
        df = _safe_read(paths[idx])
        if df.empty:
            continue
        chunks.append(df)
        n_total += len(df)
        used.append(paths[idx].name)
        if n_total >= min_events:
            break
    if not chunks:
        return pd.DataFrame()
    print(f"{label}: concatenated {len(used)} tape(s), "
          f"{n_total:,} events: {used[0]}{(' + '+str(len(used)-1)+' more') if len(used)>1 else ''}")
    return pd.concat(chunks, ignore_index=True)


def _pick_real_tape(real_dir: Path, ticker: str, min_events: int) -> pd.DataFrame:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))
    if not paths:
        sys.exit(f"FATAL: no real parquets at {real_dir}/{ticker}_*.parquet")
    return _concat_until(paths, min_events, "real")


def _pick_synth_tape(synth_base: Path, model: str, min_events: int,
                      regime: str = "base") -> pd.DataFrame:
    rel = SYNTH_LAYOUT.get(model)
    if rel is None:
        return pd.DataFrame()
    base = synth_base / rel / regime
    if not base.exists():
        # Fallback: try without regime subdir
        base = synth_base / rel
        if not base.exists():
            return pd.DataFrame()
    paths = sorted(base.glob("**/*.parquet"))
    if not paths:
        return pd.DataFrame()
    return _concat_until(paths, min_events, model)


def _ensure_mid_price(df: pd.DataFrame) -> pd.Series | None:
    """Return a per-event mid-price series or None if not derivable."""
    if "mid" in df.columns:
        return pd.to_numeric(df["mid"], errors="coerce")
    if "bid_px" in df.columns and "ask_px" in df.columns:
        return (pd.to_numeric(df["bid_px"], errors="coerce") +
                pd.to_numeric(df["ask_px"], errors="coerce")) / 2.0
    if "mid_return" in df.columns:
        # Cumulative from mid_return (relative), starting at 1.0
        r = pd.to_numeric(df["mid_return"], errors="coerce").fillna(0.0)
        return (1.0 + r).cumprod()
    return None


def _ensure_spread(df: pd.DataFrame) -> pd.Series | None:
    if "spread" in df.columns:
        return pd.to_numeric(df["spread"], errors="coerce")
    if "bid_px" in df.columns and "ask_px" in df.columns:
        return (pd.to_numeric(df["ask_px"], errors="coerce") -
                pd.to_numeric(df["bid_px"], errors="coerce"))
    return None


def _ensure_trade_mask(df: pd.DataFrame) -> pd.Series | None:
    if "trade_sz" in df.columns:
        return pd.to_numeric(df["trade_sz"], errors="coerce").fillna(0.0) > 0
    if "event_type" in df.columns:
        et = df["event_type"].astype(str).str.lower()
        return et.str.contains("trade", na=False)
    return None


def _slice_middle(df: pd.DataFrame, n_events: int) -> pd.DataFrame:
    n = len(df)
    if n <= n_events:
        return df
    start = (n - n_events) // 2
    return df.iloc[start:start + n_events].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path,
                    default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--models", nargs="+", default=["v2", "v5", "v9"])
    ap.add_argument("--regime", default="base",
                    help="regime subdir for synth tapes (base/high_vol/...)")
    ap.add_argument("--events", type=int, default=200,
                    help="event slice length to plot (from tape middle)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "report_figures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

    sources: list[tuple[str, pd.DataFrame]] = []
    real_df = _pick_real_tape(args.real_dir, args.ticker, args.events)
    if real_df.empty:
        sys.exit("FATAL: real tape is empty")
    sources.append(("real", _slice_middle(real_df, args.events)))
    for m in args.models:
        sd = _pick_synth_tape(args.synth_base, m, args.events, args.regime)
        if sd.empty:
            print(f"  SKIP: no synth tape for {m}")
            continue
        sources.append((m, _slice_middle(sd, args.events)))

    if len(sources) < 2:
        sys.exit("FATAL: not enough sources to plot (need real + ≥1 synth)")

    n_cols = len(sources)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.6 * n_cols, 9.5),
                             sharex=True)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Per-row series collection (for shared y-range)
    row_data: list[list[np.ndarray]] = [[] for _ in range(4)]
    for label, df in sources:
        mid = _ensure_mid_price(df)
        depth = (pd.to_numeric(df.get("bid_sz", pd.Series(dtype=float)),
                               errors="coerce").fillna(0.0)
                 + pd.to_numeric(df.get("ask_sz", pd.Series(dtype=float)),
                                 errors="coerce").fillna(0.0))
        spread = _ensure_spread(df)
        trades = _ensure_trade_mask(df)
        row_data[0].append(mid.values if mid is not None else np.array([np.nan]))
        row_data[1].append(depth.values)
        row_data[2].append(spread.values if spread is not None
                           else np.array([np.nan]))
        row_data[3].append(np.where(trades, 1.0, 0.0) if trades is not None
                           else np.zeros(len(df)))

    # Plot row by row
    row_titles = ["Mid-price (step function)",
                  "Top-of-book queue depth  (bid_sz + ask_sz)",
                  "Spread",
                  "Trade markers"]
    for r, (title, series) in enumerate(zip(row_titles, row_data)):
        # Compute shared y-range for this row
        flat = np.concatenate([s for s in series if len(s) > 0 and np.isfinite(s).any()])
        finite = flat[np.isfinite(flat)] if len(flat) > 0 else np.array([])
        if r == 0 and len(finite) > 1:
            # Mid-price: normalize per-column to first value so the staircase
            # shape is visible across columns despite different absolute prices
            ylim = None
        elif r == 3:
            ylim = (-0.1, 1.1)
        elif len(finite) > 1:
            lo, hi = np.percentile(finite, [1, 99])
            pad = (hi - lo) * 0.1 + 1e-9
            ylim = (lo - pad, hi + pad)
        else:
            ylim = None

        for c, (label, df) in enumerate(sources):
            ax = axes[r, c]
            x = np.arange(len(df))
            y = series[c]
            color = SOURCE_COLOR.get(label, "#444")

            if r == 0:
                # Mid-price: normalize to first value, plot as step function.
                if len(y) > 0 and np.isfinite(y[0]):
                    y_norm = (y / y[0]) - 1.0
                else:
                    y_norm = y
                ax.step(x, y_norm * 1e4, where="post",
                        color=color, linewidth=1.0)
                ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
                if c == 0:
                    ax.set_ylabel("mid_t / mid_0 − 1\n(bps)")
            elif r == 3:
                # Trade markers: vertical lines at trade events
                trade_idx = np.where(y > 0)[0]
                ax.vlines(trade_idx, 0, 1, color=color, linewidth=0.7,
                          alpha=0.8)
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"trades\n(rate ≈ {np.mean(y):.2%})")
                # Re-annotate per-column trade rate so the rate-per-source
                # is legible right under the panel
                ax.text(0.98, 0.85, f"{np.mean(y):.1%}",
                        transform=ax.transAxes,
                        ha="right", va="top", fontsize=8,
                        color=color, fontweight="bold")
            else:
                ax.plot(x, y, color=color, linewidth=0.9, alpha=0.95)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if c == 0:
                    ax.set_ylabel(title.split("(")[0].strip())

            if r == 0:
                ax.set_title(label, fontsize=12, fontweight="bold",
                             color=color, pad=4)
            if r == len(row_titles) - 1:
                ax.set_xlabel("event index")
            ax.grid(True, alpha=0.25)

        # Annotate row title on the right side of the rightmost panel
        axes[r, -1].text(1.04, 0.5, title, transform=axes[r, -1].transAxes,
                         ha="left", va="center", fontsize=9, color="#555",
                         rotation=270)

    fig.suptitle(f"Event-Level Tape Walkthrough — {args.events} consecutive events from one representative {args.ticker} tape per source\n"
                 f"(mid-price normalized to start; trade rate = fraction of events with trade_sz > 0)",
                 fontsize=12, y=1.005)
    fig.tight_layout(rect=[0, 0, 0.97, 1.0])

    pdf = args.out / "fig_event_tape_walkthrough.pdf"
    png = args.out / "fig_event_tape_walkthrough.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
