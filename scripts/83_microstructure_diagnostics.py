"""
83_microstructure_diagnostics.py — Three quick microstructure analyses
that the main figure suite doesn't cover. Each one is independently useful;
all three share the same parquet-loading machinery.

  P2.1 — Spread-conditional dynamics
         When INTC's spread > 1 tick (rare ~5% of events), what do the models
         produce? Conditional means of bid_sz, ask_sz, |mid_return|.

  P2.2 — Event-type transition matrix
         2×2 matrix of trade→trade / trade→quote / quote→trade / quote→quote
         transition probabilities. Real has known structure; do models replicate?

  P2.3 — Intraday volume U-shape
         Aggregate trade_sz per minute-of-session bucket. Real has U-shape;
         most synth probably doesn't (uniform timestamps) but documenting which
         do is the point.

Cluster-bound (needs val + synth parquets). Run on the cluster login node:
    python scripts/83_microstructure_diagnostics.py \
        --real-dir data/processed/val \
        --synth-base $HOME/orcd/scratch/diffmm \
        --models v2 v5 v9 v9_b v7_b \
        --out results/microstructure_diagnostics

Outputs (all under --out):
    spread_conditional.csv       (model, n_wide, mean_bid_sz, mean_ask_sz, mean_abs_return)
    event_transitions.csv        (model, P_trade_to_trade, P_trade_to_quote, ...)
    intraday_volume.csv          (model, bucket_min_et, total_volume)
    fig_p2_1_spread_conditional.{pdf,png}
    fig_p2_2_event_transitions.{pdf,png}
    fig_p2_3_intraday_volume.{pdf,png}
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

# Same synth-tree mapping as the P0 orchestrator. Override on CLI if you've
# regenerated synthetic data into a different layout.
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


# ─── Loaders ───────────────────────────────────────────────────────────────
def load_synth_tapes(synth_base: Path, model: str, max_files: int = 50) -> pd.DataFrame:
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
            frames.append(pl.read_parquet(p).to_pandas())
        except Exception as e:
            print(f"    WARN: {p.name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_real(real_dir: Path, ticker: str, max_files: int) -> pd.DataFrame:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))[:max_files]
    if not paths:
        sys.exit(f"FATAL: no real parquets at {real_dir}")
    print(f"loading real: {len(paths)} {ticker} parquets")
    frames = [pl.read_parquet(p).to_pandas() for p in paths]
    return pd.concat(frames, ignore_index=True)


# ─── P2.1 — spread-conditional dynamics ────────────────────────────────────
def spread_conditional(df: pd.DataFrame, model: str,
                        tick_size: float = 0.01) -> dict:
    if df.empty or "spread" not in df.columns:
        return {"model": model, "n_total": 0, "n_wide": 0,
                "mean_bid_sz_wide": np.nan, "mean_ask_sz_wide": np.nan,
                "mean_abs_return_wide": np.nan,
                "frac_wide": np.nan}
    wide = df["spread"] > tick_size + 1e-12
    sub = df[wide]
    return {
        "model": model,
        "n_total": int(len(df)),
        "n_wide": int(wide.sum()),
        "frac_wide": float(wide.mean()),
        "mean_bid_sz_wide": float(sub["bid_sz"].mean())
                            if "bid_sz" in sub.columns and len(sub) else np.nan,
        "mean_ask_sz_wide": float(sub["ask_sz"].mean())
                            if "ask_sz" in sub.columns and len(sub) else np.nan,
        "mean_abs_return_wide": float(sub["mid_return"].abs().mean())
                                 if "mid_return" in sub.columns and len(sub) else np.nan,
    }


def fig_spread_conditional(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows).set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, ylabel in [
        (axes[0], "frac_wide",            "P(spread > 1 tick)"),
        (axes[1], "mean_bid_sz_wide",     "mean bid_sz | wide"),
        (axes[2], "mean_abs_return_wide", "mean |return| | wide"),
    ]:
        sub = df.dropna(subset=[col]).sort_values(col)
        ax.barh(sub.index, sub[col], color="#1f77b4", edgecolor="k", linewidth=0.3)
        ax.set_xlabel(ylabel)
        if "real" in df.index and pd.notna(df.loc["real", col]):
            ax.axvline(df.loc["real", col], color="black", linestyle="--",
                       linewidth=1, label=f"real = {df.loc['real', col]:.3g}")
            ax.legend(fontsize=8)
        ax.set_title(ylabel)
    fig.suptitle("Spread-conditional dynamics  (events with spread > 1 tick)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    pdf = out / "fig_p2_1_spread_conditional.pdf"
    png = out / "fig_p2_1_spread_conditional.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


# ─── P2.2 — event-type transitions ─────────────────────────────────────────
def event_transitions(df: pd.DataFrame, model: str) -> dict:
    if df.empty or "event_type" not in df.columns:
        return {"model": model}
    et = df["event_type"].astype(str).str.lower().str.strip()
    is_trade = et == "trade"
    if len(is_trade) < 2:
        return {"model": model}
    next_is_trade = is_trade.shift(-1)
    pairs = pd.DataFrame({"cur_trade": is_trade, "next_trade": next_is_trade}).dropna()
    if pairs.empty:
        return {"model": model}
    p_tt = float(((pairs["cur_trade"]) & (pairs["next_trade"])).mean())
    p_tq = float(((pairs["cur_trade"]) & (~pairs["next_trade"])).mean())
    p_qt = float(((~pairs["cur_trade"]) & (pairs["next_trade"])).mean())
    p_qq = float(((~pairs["cur_trade"]) & (~pairs["next_trade"])).mean())
    return {"model": model,
            "P_trade_to_trade": p_tt, "P_trade_to_quote": p_tq,
            "P_quote_to_trade": p_qt, "P_quote_to_quote": p_qq,
            "trade_fraction": float(is_trade.mean())}


def fig_event_transitions(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows).dropna(subset=["P_trade_to_trade"])
    if df.empty:
        return
    df = df.set_index("model").sort_values("P_trade_to_trade")
    fig, ax = plt.subplots(figsize=(11, 6))
    cols = ["P_trade_to_trade", "P_trade_to_quote",
            "P_quote_to_trade", "P_quote_to_quote"]
    bottom = np.zeros(len(df))
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for c, color in zip(cols, palette):
        ax.barh(df.index, df[c], left=bottom, label=c, color=color,
                edgecolor="white", linewidth=0.5)
        bottom = bottom + df[c].values
    ax.set_xlabel("transition probability")
    ax.set_title("Event-type transition matrix per model "
                 "(stacked = trade→{trade,quote} + quote→{trade,quote})",
                 pad=10)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    pdf = out / "fig_p2_2_event_transitions.pdf"
    png = out / "fig_p2_2_event_transitions.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


# ─── P2.3 — intraday volume U-shape ────────────────────────────────────────
def intraday_volume(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Returns a DataFrame with bucket_min_et + total_volume + model cols."""
    ts_col = next((c for c in ("ts_ns", "ts", "timestamp_ns") if c in df.columns),
                  None)
    if ts_col is None or "trade_sz" not in df.columns:
        return pd.DataFrame()
    ts = pd.to_datetime(df[ts_col].astype("int64"), unit="ns", utc=True) \
            .dt.tz_convert("America/New_York")
    bucket_min = ts.dt.hour * 60 + ts.dt.minute
    # 15-min buckets across 09:30-16:00 ET
    bins = np.arange(9 * 60 + 30, 16 * 60 + 1, 15)
    bucket = pd.cut(bucket_min, bins=bins, right=False)
    vol = df["trade_sz"].fillna(0).astype(float)
    g = pd.DataFrame({"bucket": bucket, "vol": vol}) \
            .groupby("bucket", observed=True)["vol"].sum()
    out = g.reset_index()
    out["bucket_min_et"] = out["bucket"].apply(lambda b: int(b.left)
                                                if hasattr(b, "left") else np.nan)
    out["model"] = model
    return out[["model", "bucket_min_et", "vol"]]


def fig_intraday_volume(frames: list[pd.DataFrame], out: Path) -> None:
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
              if frames else pd.DataFrame()
    if df.empty:
        print("[fig_p2_3] no timestamp data — skipped")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set_palette("tab10")
    for m, g in df.groupby("model"):
        g = g.sort_values("bucket_min_et")
        norm = g["vol"] / g["vol"].sum()  # share of day's volume
        ax.plot(g["bucket_min_et"], norm, marker="o", markersize=4,
                label=m, alpha=0.85,
                linewidth=2.5 if m == "real" else 1.2)
    ax.set_xlabel("minute of day (ET)")
    ax.set_ylabel("share of day volume")
    ax.set_title("Intraday volume distribution (real should be U-shaped)",
                 pad=10)
    # X-axis labels: hh:mm
    minutes = sorted(df["bucket_min_et"].unique())
    labels = [f"{m // 60:02d}:{m % 60:02d}" for m in minutes]
    ax.set_xticks(minutes[::4])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 4)],
                        rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper center", fontsize=9, ncol=4, framealpha=0.9)
    fig.tight_layout()
    pdf = out / "fig_p2_3_intraday_volume.pdf"
    png = out / "fig_p2_3_intraday_volume.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


# ─── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--models", nargs="+",
                    default=["v2", "v5", "v9", "v9_b", "v7_b"])
    ap.add_argument("--max-files-real", type=int, default=10)
    ap.add_argument("--max-files-synth", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "microstructure_diagnostics")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    real = load_real(args.real_dir, args.ticker, args.max_files_real)
    sources = {"real": real}
    for m in args.models:
        df = load_synth_tapes(args.synth_base, m, args.max_files_synth)
        if not df.empty:
            sources[m] = df
        else:
            print(f"  [{m}] no parquets — skipped")

    # Compute all three analyses
    spread_rows, transition_rows, intraday_frames = [], [], []
    for label, df in sources.items():
        spread_rows.append(spread_conditional(df, label))
        transition_rows.append(event_transitions(df, label))
        intraday_frames.append(intraday_volume(df, label))

    # Persist CSVs
    pd.DataFrame(spread_rows).to_csv(args.out / "spread_conditional.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(args.out / "event_transitions.csv", index=False)
    iv_df = pd.concat([f for f in intraday_frames if not f.empty], ignore_index=True) \
              if intraday_frames else pd.DataFrame()
    if not iv_df.empty:
        iv_df.to_csv(args.out / "intraday_volume.csv", index=False)

    print(f"wrote 3 CSVs to {args.out}")

    # Render figures
    fig_spread_conditional(spread_rows, args.out)
    fig_event_transitions(transition_rows, args.out)
    fig_intraday_volume(intraday_frames, args.out)


if __name__ == "__main__":
    main()
