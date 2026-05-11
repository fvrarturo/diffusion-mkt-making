"""
81_real_stylized_facts.py — Standalone "real-INTC" stylized fact figure panel
for §2.5 of the report. No comparison vs synth, just a pedagogical exhibit
showing what generators are aiming to reproduce.

Reads:  data/processed/val/INTC_*.parquet   (per-day canonical-schema parquets)
Writes: results/report_figures/fig_a_real_stylized_facts.{pdf,png}
        results/report_figures/real_stylized_facts_stats.json

Six panels:
  A. Return distribution histogram (highlight zero-mass + heavy tails)
  B. ACF of |returns| (slow decay: volatility clustering)
  C. ACF of returns  (~zero / slightly negative lag-1)
  D. ACF of trade-sign (positive persistence: order-flow autocorrelation)
  E. Spread distribution (95% one-tick concentration)
  F. Intraday volume pattern (U-shape, by minute-of-session bucket)

Intended to run on the cluster (where val parquets live) OR locally if the
data symlink resolves. Subsamples to keep memory bounded.

Run:
    python scripts/81_real_stylized_facts.py
    python scripts/81_real_stylized_facts.py --real-dir data/processed/val --ticker INTC --max-files 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_real(real_dir: Path, ticker: str, max_files: int) -> list[pd.DataFrame]:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))[:max_files]
    if not paths:
        sys.exit(f"FATAL: no parquets at {real_dir} matching {ticker}_*.parquet")
    print(f"loading {len(paths)} {ticker} parquets from {real_dir}")
    dfs = []
    for p in paths:
        try:
            d = pl.read_parquet(p).to_pandas()
            dfs.append(d)
        except Exception as e:
            print(f"  WARN: skipped {p.name}: {e}")
    return dfs


def acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = (x[:-k] * x[k:]).mean() / var
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--max-files", type=int, default=10,
                    help="cap on per-day parquets to load (each is large)")
    ap.add_argument("--max-events-acf", type=int, default=2_000_000,
                    help="subsample for ACF computation")
    ap.add_argument("--max-lag", type=int, default=100)
    args = ap.parse_args()

    dfs = load_real(args.real_dir, args.ticker, args.max_files)

    # Concat for marginals; per-day for ACFs (then average) to avoid edge artefacts
    all_df = pd.concat(dfs, ignore_index=True)
    n = len(all_df)
    print(f"total events loaded: {n:,}")

    # ─── Compute stats ──────────────────────────────────────────────────────
    rets = all_df["mid_return"].dropna().values
    sgns = all_df["trade_sign"].dropna().values if "trade_sign" in all_df.columns else np.array([])
    spreads = all_df["spread"].dropna().values if "spread" in all_df.columns else np.array([])

    # |returns| ACF and returns ACF — average across days (robust to
    # cross-day discontinuities)
    abs_ret_acfs, ret_acfs, sign_acfs = [], [], []
    for d in dfs:
        r = d["mid_return"].dropna().values[:args.max_events_acf]
        if len(r) > args.max_lag + 100:
            ret_acfs.append(acf(r, args.max_lag))
            abs_ret_acfs.append(acf(np.abs(r), args.max_lag))
        if "trade_sign" in d.columns:
            s = d["trade_sign"].dropna().values[:args.max_events_acf]
            if len(s) > args.max_lag + 100:
                sign_acfs.append(acf(s.astype(float), args.max_lag))
    ret_acf_mean = np.mean(ret_acfs, axis=0) if ret_acfs else np.zeros(args.max_lag + 1)
    abs_acf_mean = np.mean(abs_ret_acfs, axis=0) if abs_ret_acfs else np.zeros(args.max_lag + 1)
    sign_acf_mean = np.mean(sign_acfs, axis=0) if sign_acfs else np.zeros(args.max_lag + 1)

    # Intraday volume bucket — needs ts_ns or ts col
    intraday = None
    for ts_col in ("ts_ns", "ts", "timestamp_ns"):
        if ts_col in all_df.columns:
            ts = pd.to_datetime(all_df[ts_col].astype("int64"), unit="ns", utc=True)
            ts_et = ts.dt.tz_convert("America/New_York")
            mob = ts_et.dt.hour * 60 + ts_et.dt.minute    # minute of day, ET
            if "trade_sz" in all_df.columns:
                vol = all_df["trade_sz"].fillna(0).values
                bucket = pd.cut(mob, bins=np.arange(9 * 60 + 30, 16 * 60 + 1, 15))
                intraday = pd.DataFrame({"bucket": bucket, "vol": vol}) \
                            .groupby("bucket", observed=True)["vol"].sum()
            break

    # ─── Stats summary ──────────────────────────────────────────────────────
    summary = {
        "n_events_total":        int(n),
        "n_days":                len(dfs),
        "mid_return_kurtosis":   float(scipy_stats.kurtosis(rets[~np.isnan(rets)])),
        "mid_return_std":        float(np.std(rets)),
        "mid_return_skew":       float(scipy_stats.skew(rets)),
        "mid_return_frac_zero":  float(np.mean(rets == 0)),
        "mid_return_max":        float(np.max(np.abs(rets))),
        "ret_acf_lag1":          float(ret_acf_mean[1]),
        "abs_ret_acf_lag1":      float(abs_acf_mean[1]),
        "abs_ret_acf_lag50":     float(abs_acf_mean[min(50, len(abs_acf_mean)-1)]),
        "trade_sign_acf_lag1":   float(sign_acf_mean[1]) if len(sign_acf_mean) > 1 else None,
        "trade_sign_acf_lag10":  float(sign_acf_mean[10]) if len(sign_acf_mean) > 10 else None,
        "spread_1tick_frac":     (float(np.mean(np.isclose(spreads, 0.01)))
                                  if len(spreads) else None),
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    out_json = OUT / "real_stylized_facts_stats.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # ─── Six-panel figure ───────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # A) Return distribution (symlog)
    ax = axes[0, 0]
    ax.hist(rets, bins=300, color="#444", log=True)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel("mid_return")
    ax.set_ylabel("event count (log)")
    ax.set_title(f"A. Return distribution\n(kurt = {summary['mid_return_kurtosis']:.0f}, "
                 f"frac zero = {summary['mid_return_frac_zero']:.2%})")
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

    # B) |returns| ACF
    ax = axes[0, 1]
    ax.plot(np.arange(len(abs_acf_mean)), abs_acf_mean, color="#1f77b4", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF |return|")
    ax.set_title(f"B. ACF of |return|  (vol clustering)\nlag-50 = {summary['abs_ret_acf_lag50']:.3f}")

    # C) returns ACF
    ax = axes[0, 2]
    ax.plot(np.arange(len(ret_acf_mean)), ret_acf_mean, color="#2ca02c", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF return")
    ax.set_title(f"C. ACF of return\nlag-1 = {summary['ret_acf_lag1']:+.3f}")
    ax.set_ylim(-0.1, 0.1)

    # D) Trade-sign ACF
    ax = axes[1, 0]
    if len(sign_acf_mean) > 1:
        ax.plot(np.arange(len(sign_acf_mean)), sign_acf_mean, color="#d62728", linewidth=1.5)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF trade_sign")
        ax.set_title(f"D. ACF of trade sign  (order-flow persistence)\n"
                     f"lag-1 = {summary['trade_sign_acf_lag1']:+.3f}")
    else:
        ax.set_visible(False)

    # E) Spread distribution
    ax = axes[1, 1]
    if len(spreads):
        ax.hist(spreads * 100, bins=80, color="#9467bd", log=True)  # cents
        ax.set_xlabel("spread (cents)")
        ax.set_ylabel("event count (log)")
        ax.set_title(f"E. Spread distribution\n"
                     f"P(spread = 1 tick) = {summary['spread_1tick_frac']:.2%}")
    else:
        ax.set_visible(False)

    # F) Intraday volume U-shape
    ax = axes[1, 2]
    if intraday is not None and len(intraday) > 0:
        x = np.arange(len(intraday))
        ax.bar(x, intraday.values, color="#ff7f0e", edgecolor="k", linewidth=0.3)
        # Tick labels every 4 buckets (15-min buckets → every hour)
        labels = [str(c)[1:6] if i % 4 == 0 else "" for i, c in enumerate(intraday.index)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("aggregated volume")
        ax.set_title("F. Intraday volume  (U-shape)")
    else:
        ax.set_visible(False)
        print("  WARN: no timestamp column found — skipping intraday panel")

    fig.suptitle(f"Real INTC stylized facts ({summary['n_days']} val days, "
                 f"{summary['n_events_total']:,} events)",
                 fontsize=15, y=1.01)
    fig.tight_layout()

    pdf = OUT / "fig_a_real_stylized_facts.pdf"
    png = OUT / "fig_a_real_stylized_facts.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
