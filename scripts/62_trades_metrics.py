#!/usr/bin/env python3
"""
TRADES-inspired supplementary metrics for the diffusion LOB workstream.

Four metrics from Berti et al. 2025 ("Generating Realistic Market Simulations
with Diffusion Models"):

  1. Volume-Volatility Correlation distribution (TRADES Fig 3.2)
  2. Returns-Volatility Leverage Effect distribution (TRADES Fig 3.3)
  3. PCA Coverage / Convex Hull Overlap (TRADES Fig 2)
  4. Mid-Price Trace Visualization (TRADES Fig 3.6)

CPU only, runs on existing decoded parquets. Designed to slot alongside
`scripts/60_deep_diagnostic.py` and `scripts/61_ranking_mechanism.py`.

Usage:
    python scripts/62_trades_metrics.py \
        --real-dir data/processed/val \
        --synth-dirs \
            $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
            $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
        --labels v2 v9 \
        --out results/trades_metrics \
        --max-tapes 100
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ─────────────────────────────────────────────────────────────────
BUCKET_NS = 60 * 1_000_000_000   # 1-minute buckets in nanoseconds
MIN_BUCKETS_PER_TAPE = 5         # skip tapes with fewer buckets (Pearson unreliable)
MIN_EVENTS_PER_TAPE = 100        # skip very short tapes
PCA_FEATURES = [
    "mean_spread", "std_spread",
    "mean_bid_sz", "std_bid_sz",
    "mean_ask_sz", "std_ask_sz",
    "trade_fraction",
    "return_kurtosis", "return_std",
    "abs_return_acf_lag1",
    "ofi_proxy_std",
]


# ─── Data loading (per-tape, NOT concatenated — we need per-tape correlations) ──

def load_tapes_per_tape(directory: str | Path, max_per_regime: int = 100) -> list[tuple[str, pd.DataFrame]]:
    """Load parquets keeping each tape as a separate DataFrame.

    Returns a list of (tape_id, df) pairs. Recurses one level into regime
    subdirs if present; otherwise loads from the top-level dir.
    """
    p = Path(directory)
    sub_dirs = [d for d in p.iterdir() if d.is_dir()]
    parquet_paths: list[Path] = []
    if sub_dirs:
        for sd in sorted(sub_dirs):
            parquet_paths.extend(sorted(sd.glob("*.parquet"))[:max_per_regime])
    else:
        # Flat dir: try direct globs and one nested level
        parquet_paths = sorted(p.glob("*.parquet"))[:max_per_regime]
        if not parquet_paths:
            parquet_paths = sorted(p.glob("*/*.parquet"))[:max_per_regime]

    out: list[tuple[str, pd.DataFrame]] = []
    for pq in parquet_paths:
        try:
            df = pd.read_parquet(pq)
            tape_id = f"{pq.parent.name}/{pq.stem}" if pq.parent.name not in (".", "INTC") else pq.stem
            out.append((tape_id, df))
        except Exception as e:
            print(f"  WARN: skipping {pq}: {e}", file=sys.stderr)
    if not out:
        raise FileNotFoundError(f"No parquets loaded from {directory}")
    return out


# ─── Bucketing helper ───────────────────────────────────────────────────────

def bucket_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add a bucket index column (1-minute buckets from ts_ns).

    Returns the same DataFrame with `_bucket` column; or empty DataFrame
    if ts_ns is missing.
    """
    if "ts_ns" not in df.columns:
        return pd.DataFrame()
    ts = df["ts_ns"].astype(np.int64).values
    bucket = (ts - ts[0]) // BUCKET_NS
    out = df.copy()
    out["_bucket"] = bucket.astype(np.int64)
    return out


# ─── Metric 1: Volume-Volatility Correlation ────────────────────────────────

def vol_volatility_correlation(tape_id: str, df: pd.DataFrame) -> float | None:
    """Pearson correlation of (per-bucket volume, per-bucket volatility) for one tape."""
    if len(df) < MIN_EVENTS_PER_TAPE or "ts_ns" not in df.columns:
        return None
    bk = bucket_events(df)
    if bk.empty:
        return None
    # Volume per bucket = sum of trade_sz (NaN-safe)
    trade_sz = pd.to_numeric(bk.get("trade_sz"), errors="coerce").fillna(0.0)
    bk = bk.assign(_trade_sz=trade_sz)
    # Per-bucket aggregates. mid_return: std (or |sum| if too few nonzero).
    grp = bk.groupby("_bucket")
    volume = grp["_trade_sz"].sum()
    if "mid_return" not in bk.columns:
        return None
    mid_ret = pd.to_numeric(bk["mid_return"], errors="coerce").fillna(0.0)
    bk_for_vol = bk.assign(_mr=mid_ret)
    vol_std = bk_for_vol.groupby("_bucket")["_mr"].std(ddof=0).fillna(0.0)
    # Drop buckets with no variation (constant mid throughout the bucket)
    use = (vol_std > 0)
    if int(use.sum()) < MIN_BUCKETS_PER_TAPE:
        # Fallback: sum of abs returns per bucket
        vol_std = bk_for_vol.groupby("_bucket")["_mr"].apply(lambda s: float(np.abs(s.values).sum()))
        use = (vol_std > 0)
        if int(use.sum()) < MIN_BUCKETS_PER_TAPE:
            return None
    # Align indices
    common = volume.index.intersection(vol_std.index)
    if len(common) < MIN_BUCKETS_PER_TAPE:
        return None
    v = volume.loc[common].values
    s = vol_std.loc[common].values
    if np.std(v) == 0 or np.std(s) == 0:
        return None
    r, _ = scipy_stats.pearsonr(v, s)
    return float(r) if np.isfinite(r) else None


# ─── Metric 2: Leverage Effect (returns vs next-bucket volatility) ──────────

def leverage_effect(tape_id: str, df: pd.DataFrame) -> float | None:
    """Pearson correlation of (per-bucket return, next-bucket volatility)."""
    if len(df) < MIN_EVENTS_PER_TAPE or "ts_ns" not in df.columns:
        return None
    bk = bucket_events(df)
    if bk.empty or "mid_return" not in bk.columns:
        return None
    mid_ret = pd.to_numeric(bk["mid_return"], errors="coerce").fillna(0.0)
    bk = bk.assign(_mr=mid_ret)
    grp = bk.groupby("_bucket")
    ret = grp["_mr"].sum()
    vol_std = grp["_mr"].std(ddof=0).fillna(0.0)
    if len(ret) < MIN_BUCKETS_PER_TAPE + 1:
        return None
    # Align: return[i] vs vol[i+1]
    ret_in = ret.values[:-1]
    vol_next = vol_std.values[1:]
    if np.std(ret_in) == 0 or np.std(vol_next) == 0:
        return None
    r, _ = scipy_stats.pearsonr(ret_in, vol_next)
    return float(r) if np.isfinite(r) else None


# ─── Metric 3: PCA Coverage ─────────────────────────────────────────────────

def tape_feature_vector(df: pd.DataFrame) -> np.ndarray | None:
    """10-D summary vector for one tape (used for PCA coverage)."""
    if len(df) < MIN_EVENTS_PER_TAPE:
        return None

    def safe_num(col: str) -> np.ndarray:
        if col not in df.columns:
            return np.array([], dtype=np.float64)
        return pd.to_numeric(df[col], errors="coerce").dropna().values.astype(np.float64)

    spread = safe_num("spread")
    bid_sz = safe_num("bid_sz")
    ask_sz = safe_num("ask_sz")
    trade_sz = pd.to_numeric(df.get("trade_sz"), errors="coerce").fillna(0.0).values.astype(np.float64) if "trade_sz" in df.columns else np.zeros(len(df))
    mr = safe_num("mid_return")
    if any(len(x) == 0 for x in [spread, bid_sz, ask_sz, mr]):
        return None

    # ACF |return| at lag 1
    abs_mr = np.abs(mr)
    if len(abs_mr) < 2 or np.std(abs_mr) == 0:
        acf_lag1 = 0.0
    else:
        m = abs_mr.mean()
        var = abs_mr.var()
        acf_lag1 = float(np.mean((abs_mr[:-1] - m) * (abs_mr[1:] - m)) / var)

    # OFI proxy std: diff(bid_sz) - diff(ask_sz)
    if len(bid_sz) > 2 and len(ask_sz) > 2:
        # use df-aligned diffs to keep length consistent
        bs = pd.to_numeric(df.get("bid_sz"), errors="coerce").fillna(0.0).values
        as_ = pd.to_numeric(df.get("ask_sz"), errors="coerce").fillna(0.0).values
        ofi = np.diff(bs) - np.diff(as_)
        ofi_std = float(np.std(ofi))
    else:
        ofi_std = 0.0

    feats = [
        float(np.mean(spread)), float(np.std(spread)),
        float(np.mean(bid_sz)), float(np.std(bid_sz)),
        float(np.mean(ask_sz)), float(np.std(ask_sz)),
        float((trade_sz > 0).mean()),
        float(scipy_stats.kurtosis(mr)),
        float(np.std(mr)),
        acf_lag1,
        ofi_std,
    ]
    return np.array(feats, dtype=np.float64)


def compute_pca_coverage(real_vecs: np.ndarray, synth_vecs_by_model: dict[str, np.ndarray],
                         out_dir: str) -> dict[str, dict]:
    """PCA-fit on real, project synth, compute convex-hull coverage."""
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull, Delaunay

    # z-score using real's mean/std (per feature)
    mean = real_vecs.mean(axis=0)
    std = real_vecs.std(axis=0) + 1e-12
    real_z = (real_vecs - mean) / std

    pca = PCA(n_components=2).fit(real_z)
    real_pc = pca.transform(real_z)

    results: dict[str, dict] = {}
    # Real hull
    try:
        real_hull = ConvexHull(real_pc)
        real_area = float(real_hull.volume)  # in 2D, "volume" = area
    except Exception as e:
        print(f"  WARN: could not build real convex hull: {e}", file=sys.stderr)
        return {}

    # Try shapely; fall back to containment fraction
    try:
        from shapely.geometry import Polygon
        have_shapely = True
    except ImportError:
        have_shapely = False
        print("  WARN: shapely not available — falling back to containment fraction", file=sys.stderr)

    real_poly = None
    if have_shapely:
        try:
            real_poly = Polygon(real_pc[real_hull.vertices])
        except Exception:
            have_shapely = False

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(real_pc[:, 0], real_pc[:, 1], c="0.3", s=15, alpha=0.6, label=f"real (n={len(real_pc)})")
    ax.plot(real_pc[real_hull.vertices, 0].tolist() + [real_pc[real_hull.vertices[0], 0]],
            real_pc[real_hull.vertices, 1].tolist() + [real_pc[real_hull.vertices[0], 1]],
            "k--", lw=1.5, label="real hull")

    cmap = plt.get_cmap("tab10")
    for i, (label, vecs) in enumerate(synth_vecs_by_model.items()):
        if len(vecs) < 4:
            print(f"  WARN: {label} has < 4 tape vectors; skipping hull")
            continue
        z = (vecs - mean) / std
        pc = pca.transform(z)
        try:
            sh = ConvexHull(pc)
            sh_area = float(sh.volume)
        except Exception as e:
            print(f"  WARN: synth hull fail for {label}: {e}", file=sys.stderr)
            continue

        if have_shapely and real_poly is not None:
            try:
                synth_poly = Polygon(pc[sh.vertices])
                inter = real_poly.intersection(synth_poly).area
                coverage = inter / real_poly.area
                metric = "intersection_over_real_area"
            except Exception as e:
                # fallback per model
                tri = Delaunay(pc)
                inside = tri.find_simplex(real_pc) >= 0
                coverage = float(inside.mean())
                metric = "real_pts_inside_synth_hull"
        else:
            tri = Delaunay(pc)
            inside = tri.find_simplex(real_pc) >= 0
            coverage = float(inside.mean())
            metric = "real_pts_inside_synth_hull"

        color = cmap(i % 10)
        ax.scatter(pc[:, 0], pc[:, 1], c=[color], s=15, alpha=0.5, marker="^",
                   label=f"{label} (cov={coverage*100:.1f}%)")
        ax.plot(pc[sh.vertices, 0].tolist() + [pc[sh.vertices[0], 0]],
                pc[sh.vertices, 1].tolist() + [pc[sh.vertices[0], 1]],
                color=color, lw=1.0, ls="--")

        results[label] = {
            "coverage_pct": coverage * 100,
            "coverage_metric": metric,
            "n_tapes": len(pc),
            "pc1_mean": float(pc[:, 0].mean()), "pc1_std": float(pc[:, 0].std()),
            "pc2_mean": float(pc[:, 1].mean()), "pc2_std": float(pc[:, 1].std()),
            "real_hull_area": real_area,
            "synth_hull_area": sh_area,
        }

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA Coverage of Real Data Distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pca_coverage.pdf")
    fig.savefig(f"{out_dir}/pca_coverage.png", dpi=120)
    plt.close(fig)
    return results


# ─── KDE plot helper ────────────────────────────────────────────────────────

def kde_plot(data_by_model: dict[str, list[float]], real_label: str, title: str,
             xlabel: str, out_path: str, x_range: tuple[float, float] = (-1, 1)) -> None:
    """Overlaid KDE per model. real shaded; synth as lines."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cmap = plt.get_cmap("tab10")
    grid = np.linspace(x_range[0], x_range[1], 300)

    # Real first (shaded)
    if real_label in data_by_model and len(data_by_model[real_label]) >= 5:
        vals = np.array(data_by_model[real_label], dtype=np.float64)
        if np.std(vals) > 1e-12:
            kde = scipy_stats.gaussian_kde(vals)
            y = kde(grid)
            ax.fill_between(grid, y, color="0.3", alpha=0.3, label=f"{real_label} (n={len(vals)}, mean={vals.mean():+.3f})")
            ax.plot(grid, y, color="0.3", lw=2)

    color_idx = 0
    for label, vals in data_by_model.items():
        if label == real_label:
            continue
        if len(vals) < 5:
            print(f"  WARN: {label}: only {len(vals)} tape correlations, skipping KDE")
            continue
        v = np.array(vals, dtype=np.float64)
        if np.std(v) < 1e-12:
            print(f"  WARN: {label}: zero variance in correlations, skipping KDE")
            continue
        kde = scipy_stats.gaussian_kde(v)
        ax.plot(grid, kde(grid), color=cmap(color_idx % 10), lw=2,
                label=f"{label} (n={len(v)}, mean={v.mean():+.3f})")
        color_idx += 1

    ax.set_xlim(x_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.axvline(0, color="0.5", lw=0.5, ls=":")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=120)
    plt.close(fig)


# ─── Metric 4: Mid-Price Trace Visualization ────────────────────────────────

def midprice_trace_plot(label: str, synth_tapes: list[tuple[str, pd.DataFrame]],
                        real_tape: tuple[str, pd.DataFrame] | None,
                        out_path: str, n_synth: int = 5) -> None:
    """Overlay 5 synth tapes' normalized mid traces vs 1 real tape."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    cmap = plt.get_cmap("tab10")

    # Real (thick black)
    if real_tape is not None:
        rid, rdf = real_tape
        if "mid" in rdf.columns:
            mid = pd.to_numeric(rdf["mid"], errors="coerce").dropna().values
            if len(mid) > 0 and mid[0] > 0:
                rel = mid / mid[0]
                # Show first ~min(len, 50000) events to keep plot reasonable
                n = min(len(rel), 50_000)
                ax.plot(np.arange(n), rel[:n], color="black", lw=1.5, label=f"real ({rid})")

    # Synth (thin colored)
    plotted = 0
    for tid, df in synth_tapes[:n_synth]:
        if "mid" not in df.columns:
            continue
        mid = pd.to_numeric(df["mid"], errors="coerce").dropna().values
        if len(mid) == 0 or mid[0] <= 0:
            continue
        rel = mid / mid[0]
        ax.plot(np.arange(len(rel)), rel, color=cmap(plotted % 10),
                lw=0.7, alpha=0.7, label=f"synth seed {plotted+1}")
        plotted += 1

    ax.set_xlabel("event index")
    ax.set_ylabel("normalized mid-price")
    ax.set_title(f"Mid-Price Traces: {label} vs Real")
    ax.axhline(1.0, color="0.5", lw=0.4, ls=":")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=120)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--synth-dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="One label per synth dir (must match count)")
    ap.add_argument("--out", default="results/trades_metrics")
    ap.add_argument("--max-tapes", type=int, default=100)
    args = ap.parse_args()

    if len(args.synth_dirs) != len(args.labels):
        raise SystemExit("--synth-dirs and --labels must have same count")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    real_label = "real"

    # ── Load real tapes ──
    print(f"Loading real tapes from {args.real_dir} ...")
    real_tapes = load_tapes_per_tape(args.real_dir, max_per_regime=args.max_tapes)
    print(f"  loaded {len(real_tapes)} real tapes")

    # ── Load synth tapes per model ──
    synth_tapes_by_model: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for sdir, label in zip(args.synth_dirs, args.labels):
        try:
            print(f"Loading {label} from {sdir} ...")
            synth_tapes_by_model[label] = load_tapes_per_tape(sdir, max_per_regime=args.max_tapes)
            print(f"  {label}: {len(synth_tapes_by_model[label])} tapes")
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")

    if not synth_tapes_by_model:
        raise SystemExit("no synth data loaded; nothing to do")

    # ─── Metric 1: Volume-Volatility Correlation ───
    print("\n=== Metric 1: Volume-Volatility Correlation ===")
    vv_by_model: dict[str, list[float]] = {real_label: []}
    vv_rows: list[dict] = []
    for tid, df in real_tapes:
        r = vol_volatility_correlation(tid, df)
        if r is not None:
            vv_by_model[real_label].append(r)
            vv_rows.append({"model": real_label, "tape_id": tid, "pearson_r": r})
    print(f"  real: {len(vv_by_model[real_label])} usable tapes; "
          f"mean={np.mean(vv_by_model[real_label] or [np.nan]):+.3f}")
    for label, tapes in synth_tapes_by_model.items():
        vv_by_model[label] = []
        for tid, df in tapes:
            r = vol_volatility_correlation(tid, df)
            if r is not None:
                vv_by_model[label].append(r)
                vv_rows.append({"model": label, "tape_id": tid, "pearson_r": r})
        vals = vv_by_model[label]
        print(f"  {label}: {len(vals)} usable tapes; "
              f"mean={np.mean(vals or [np.nan]):+.3f}  std={np.std(vals or [0]):.3f}")
    pd.DataFrame(vv_rows).to_csv(f"{out_dir}/vol_volatility_correlation.csv", index=False)
    kde_plot(vv_by_model, real_label,
             title="Volume-Volatility Correlation Distribution",
             xlabel="per-tape Pearson(r)",
             out_path=f"{out_dir}/vol_volatility_correlation.pdf")

    # ─── Metric 2: Leverage Effect ───
    print("\n=== Metric 2: Returns-Volatility (Leverage) Correlation ===")
    le_by_model: dict[str, list[float]] = {real_label: []}
    le_rows: list[dict] = []
    for tid, df in real_tapes:
        r = leverage_effect(tid, df)
        if r is not None:
            le_by_model[real_label].append(r)
            le_rows.append({"model": real_label, "tape_id": tid, "pearson_r": r})
    print(f"  real: {len(le_by_model[real_label])} usable tapes; "
          f"mean={np.mean(le_by_model[real_label] or [np.nan]):+.3f}")
    for label, tapes in synth_tapes_by_model.items():
        le_by_model[label] = []
        for tid, df in tapes:
            r = leverage_effect(tid, df)
            if r is not None:
                le_by_model[label].append(r)
                le_rows.append({"model": label, "tape_id": tid, "pearson_r": r})
        vals = le_by_model[label]
        print(f"  {label}: {len(vals)} usable tapes; "
              f"mean={np.mean(vals or [np.nan]):+.3f}  std={np.std(vals or [0]):.3f}")
    pd.DataFrame(le_rows).to_csv(f"{out_dir}/leverage_effect.csv", index=False)
    kde_plot(le_by_model, real_label,
             title="Returns-Volatility (Leverage Effect) Correlation Distribution",
             xlabel="per-tape Pearson(r)  [negative = leverage effect]",
             out_path=f"{out_dir}/leverage_effect.pdf")

    # ─── Metric 3: PCA Coverage ───
    print("\n=== Metric 3: PCA Coverage / Convex Hull ===")
    real_vecs = []
    for tid, df in real_tapes:
        v = tape_feature_vector(df)
        if v is not None and np.all(np.isfinite(v)):
            real_vecs.append(v)
    real_vecs_arr = np.array(real_vecs)
    print(f"  real: {len(real_vecs)} feature vectors")

    synth_vecs: dict[str, np.ndarray] = {}
    for label, tapes in synth_tapes_by_model.items():
        vecs = []
        for tid, df in tapes:
            v = tape_feature_vector(df)
            if v is not None and np.all(np.isfinite(v)):
                vecs.append(v)
        synth_vecs[label] = np.array(vecs)
        print(f"  {label}: {len(vecs)} feature vectors")

    pca_results = {}
    if len(real_vecs_arr) >= 4:
        pca_results = compute_pca_coverage(real_vecs_arr, synth_vecs, out_dir)
        pca_rows = [{"model": k, **v} for k, v in pca_results.items()]
        if pca_rows:
            pd.DataFrame(pca_rows).to_csv(f"{out_dir}/pca_coverage.csv", index=False)
        for label, info in pca_results.items():
            print(f"  {label}: coverage = {info['coverage_pct']:.1f}% ({info['coverage_metric']}),  "
                  f"n_tapes={info['n_tapes']}")
    else:
        print(f"  WARN: only {len(real_vecs_arr)} real feature vectors — need ≥4 for PCA hull")

    # ─── Metric 4: Mid-Price Traces (one fig per model) ───
    print("\n=== Metric 4: Mid-Price Traces ===")
    real_tape_for_trace = real_tapes[0] if real_tapes else None
    for label, tapes in synth_tapes_by_model.items():
        # Filter to "base" regime if regime-keyed; else first 5
        base_tapes = [(tid, df) for tid, df in tapes if "base" in tid.lower()][:5]
        if len(base_tapes) < 5:
            base_tapes = tapes[:5]
        out_path = f"{out_dir}/midprice_traces_{label}.pdf"
        midprice_trace_plot(label, base_tapes, real_tape_for_trace, out_path)
        print(f"  {label}: wrote {out_path}")

    # ─── Text report ───
    lines = []
    lines.append("=" * 80)
    lines.append("TRADES SUPPLEMENTARY METRICS — SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    def section(title: str):
        lines.append("─" * 80)
        lines.append(title)
        lines.append("─" * 80)

    section("Metric 1: Volume-Volatility Correlation (TRADES Fig 3.2)")
    lines.append("Stylized fact: trading volume and price volatility are positively correlated.")
    lines.append(f"  {'model':<14s}  {'n_tapes':>8s}  {'mean':>10s}  {'std':>10s}")
    for label in [real_label] + list(synth_tapes_by_model.keys()):
        v = vv_by_model.get(label, [])
        if v:
            lines.append(f"  {label:<14s}  {len(v):>8d}  {np.mean(v):>+10.4f}  {np.std(v):>10.4f}")
        else:
            lines.append(f"  {label:<14s}  {0:>8d}  {'-':>10s}  {'-':>10s}")
    lines.append("")

    section("Metric 2: Returns-Volatility Correlation / Leverage Effect (TRADES Fig 3.3)")
    lines.append("Stylized fact: negative correlation (leverage effect, Black 1976).")
    lines.append(f"  {'model':<14s}  {'n_tapes':>8s}  {'mean':>10s}  {'std':>10s}")
    for label in [real_label] + list(synth_tapes_by_model.keys()):
        v = le_by_model.get(label, [])
        if v:
            lines.append(f"  {label:<14s}  {len(v):>8d}  {np.mean(v):>+10.4f}  {np.std(v):>10.4f}")
        else:
            lines.append(f"  {label:<14s}  {0:>8d}  {'-':>10s}  {'-':>10s}")
    lines.append("")

    section("Metric 3: PCA Coverage of Real Data Distribution (TRADES Fig 2)")
    if pca_results:
        lines.append(f"  {'model':<14s}  {'coverage':>10s}  {'metric':<35s}  {'n_tapes':>8s}")
        for label, info in pca_results.items():
            lines.append(f"  {label:<14s}  {info['coverage_pct']:>9.1f}%  "
                         f"{info['coverage_metric']:<35s}  {info['n_tapes']:>8d}")
    else:
        lines.append("  (PCA coverage skipped — insufficient real feature vectors)")
    lines.append("")

    section("Metric 4: Mid-Price Trace Figures")
    for label in synth_tapes_by_model.keys():
        lines.append(f"  {label}: {out_dir}/midprice_traces_{label}.pdf")

    report_path = f"{out_dir}/trades_metrics_summary.txt"
    Path(report_path).write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
