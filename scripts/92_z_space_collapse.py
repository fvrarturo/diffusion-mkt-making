"""
92_z_space_collapse.py — z-space histograms for v8 (no dequant) vs v9
(dequant) on `mid_return`, with N(0,1) reference overlay.

The visual proof of the §5.7 zero-inflation pathology mechanism.

The deep diagnostic CSV reports v8 mid_return collapses to z-mean=-2.55,
z-std=0.30 (i.e. ~99% of v8's z-values cram into a 0.06-wide band
around z=-2.55), while v9's dequantized z-distribution recovers a
spread closer to N(0,1). This script makes that visual:

  Panel A — v8: a narrow spike at z≈-2.55 next to the smooth N(0,1) bell.
  Panel B — v9: a much-wider distribution overlapping the bell, with
            the residual zero-mass artifact still visible at z≈-2.55
            but no longer dominant.

Reuses the `forward_copula` logic from `scripts/60_deep_diagnostic.py`
(empirical-CDF rank → uniform → Φ⁻¹) so the z-values match what the
deep diagnostic reported.

Cluster-bound: needs the synth parquets at
  ~/orcd/scratch/diffmm/synthetic/intc_v{8,9}/INTC/...
plus the copula norm-stats files at
  data/processed/norm_stats_INTC.copula.{json,npz}

Run on cluster login:
    python scripts/92_z_space_collapse.py \
        --synth-base ~/orcd/scratch/diffmm \
        --copula-json data/processed/norm_stats_INTC.copula.json \
        --models v8 v9 \
        --feature mid_return \
        --max-files-synth 1000 \
        --out results/report_figures
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats as scipy_stats

# Force line-buffered stdout so progress is visible when piped through tee.
print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parent.parent

SYNTH_LAYOUT = {
    "v8":   "synthetic/intc_v8/INTC",
    "v8_b": "synthetic/intc_v8_b/INTC",
    "v9":   "synthetic/intc_v9/INTC",
    "v9_b": "synthetic/intc_v9_b/INTC",
}

SOURCE_COLOR = {
    "v8":   "#bcbd22",  # olive — the failure
    "v8_b": "#8c8c2d",
    "v9":   "#9467bd",  # protagonist purple — the fix
    "v9_b": "#7a4ea0",
}


# ─── Forward copula (mirrors scripts/60_deep_diagnostic.py) ─────────────
def load_copula_transform(copula_json: str) -> tuple[dict, dict]:
    with open(copula_json) as f:
        meta = json.load(f)
    npz_path = copula_json.replace(".json", ".npz")
    arrays = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    return meta, arrays


def forward_copula(values: np.ndarray, sorted_vals: np.ndarray,
                   clip_q: float = 0.001) -> np.ndarray:
    ranks = np.searchsorted(sorted_vals, values)
    u = (ranks + 0.5) / (len(sorted_vals) + 1)
    u = np.clip(u, clip_q, 1.0 - clip_q)
    return scipy_stats.norm.ppf(u)


# ─── Synth tape loader (balanced across regimes) ────────────────────────
def gather_paths_synth(synth_base: Path, model: str, max_files: int) -> list[Path]:
    rel = SYNTH_LAYOUT.get(model)
    if rel is None:
        print(f"  [{model}] ERROR: model not in SYNTH_LAYOUT")
        return []
    base = synth_base / rel
    if not base.exists():
        print(f"  [{model}] ERROR: base path does not exist: {base}")
        # Suggest alternate layouts the user might want to try
        parent = base.parent
        if parent.exists():
            siblings = sorted([p.name for p in parent.iterdir() if p.is_dir()])[:10]
            print(f"  [{model}]   parent ({parent}) contains: {siblings}")
        return []
    regimes = ["base", "high_vol", "toxic", "thin"]
    regime_paths = [d for d in (base / r for r in regimes) if d.exists()]
    if regime_paths:
        n_per = max(1, max_files // len(regime_paths))
        out: list[Path] = []
        regime_hits = []
        for d in regime_paths:
            sub = sorted(d.glob("*.parquet"))[:n_per]
            out.extend(sub)
            regime_hits.append(f"{d.name}={len(sub)}")
        print(f"  [{model}] per-regime: {' '.join(regime_hits)} "
              f"({len(out)} files)")
        return out
    flat = sorted(base.glob("**/*.parquet"))[:max_files]
    print(f"  [{model}] no regime subdirs; recursive glob got {len(flat)} files")
    return flat


def load_feature_values(paths: list[Path], feature: str,
                         max_n: int = 5_000_000) -> np.ndarray:
    if not paths:
        return np.array([])
    chunks = []
    n = 0
    for p in paths:
        try:
            d = pl.read_parquet(p, columns=[feature]).to_pandas()
        except Exception as e:
            print(f"  WARN: {p.name}: {e}")
            continue
        v = d[feature].dropna().values
        chunks.append(v)
        n += len(v)
        if n >= max_n:
            break
    if not chunks:
        return np.array([])
    arr = np.concatenate(chunks)
    if len(arr) > max_n:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=max_n, replace=False)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-base", type=Path,
                    default=Path.home() / "orcd" / "scratch" / "diffmm")
    ap.add_argument("--copula-json", type=str,
                    default="data/processed/norm_stats_INTC.copula.json")
    ap.add_argument("--models", nargs="+", default=["v8", "v9"])
    ap.add_argument("--feature", default="mid_return",
                    help="feature to z-transform (default: mid_return)")
    ap.add_argument("--max-files-synth", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "report_figures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

    # Load copula transform
    print(f"loading copula transform from {args.copula_json}")
    meta, arrays = load_copula_transform(args.copula_json)
    feature_columns = meta.get("feature_columns", [])
    if args.feature not in feature_columns:
        sys.exit(f"FATAL: feature {args.feature!r} not in copula feature_columns "
                 f"({feature_columns})")
    feat_idx = feature_columns.index(args.feature)
    key = f"ecdf_{feat_idx}"
    if key not in arrays:
        sys.exit(f"FATAL: no ecdf array for feature index {feat_idx} "
                 f"({args.feature}); copula was probably not fit on this column")
    sorted_vals = arrays[key]
    clip_q = meta.get("clip_quantile", 0.001)
    print(f"  ECDF over {len(sorted_vals):,} training values "
          f"for feature {args.feature!r}")

    # Compute z-values per model
    print(f"models to process: {args.models}")
    print(f"synth_base: {args.synth_base}  (exists: {args.synth_base.exists()})")
    z_per_model: dict[str, np.ndarray] = {}
    for m in args.models:
        print(f"loading {m} ...")
        paths = gather_paths_synth(args.synth_base, m, args.max_files_synth)
        if not paths:
            print(f"  SKIP: no parquets resolved for {m}")
            continue
        vals = load_feature_values(paths, args.feature)
        if len(vals) == 0:
            print(f"  SKIP: no {args.feature} values loaded for {m}")
            continue
        z = forward_copula(vals, sorted_vals, clip_q)
        z_per_model[m] = z
        print(f"  {m}: {len(z):,} z-values  "
              f"mean={z.mean():.3f}  std={z.std():.3f}  "
              f"kurt={scipy_stats.kurtosis(z):.1f}")

    if not z_per_model:
        sys.exit("FATAL: no models produced z-values")

    # ── Figure ──────────────────────────────────────────────────────────
    n_panels = len(z_per_model)
    # 5.5 in/panel keeps annotations legible across 1-4 panels; min 9 for
    # single-panel cases.
    panel_w = 5.5
    fig_w = max(9.0, panel_w * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.6),
                             sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    z_grid = np.linspace(-4, 4, 600)
    n01_density = scipy_stats.norm.pdf(z_grid)

    bins = np.linspace(-4, 4, 200)

    for ax, (m, z) in zip(axes, z_per_model.items()):
        c = SOURCE_COLOR.get(m, "#444")
        ax.hist(z, bins=bins, density=True, color=c,
                alpha=0.75, edgecolor="none",
                label=f"{m} synth z\n(mean={z.mean():.2f}, "
                       f"std={z.std():.2f}, kurt={scipy_stats.kurtosis(z):.0f})")
        ax.plot(z_grid, n01_density, color="black", linewidth=1.6,
                linestyle="--", label="$\\mathcal{N}(0,1)$ target")
        ax.axvline(0, color="gray", linewidth=0.6, alpha=0.6)
        ax.axvline(z.mean(), color=c, linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"{m} z mean = {z.mean():.2f}")
        ax.set_yscale("log")
        ax.set_xlim(-4, 4)
        ax.set_ylim(1e-3, 50)
        ax.set_xlabel(f"$z = \\Phi^{{-1}}(\\hat{{F}}({args.feature}))$")
        ax.set_ylabel("density (log)")
        ax.set_title(f"{m}", fontsize=13, fontweight="bold", color=c)
        ax.legend(fontsize=8, loc="upper right")

    title_prefix = ("z-Space Collapse and Recovery: copula-transformed "
                    f"$\\mathtt{{{args.feature}}}$ for {' vs '.join(args.models)}")
    fig.suptitle(f"{title_prefix}",
                 fontsize=11, y=0.98)
    fig.tight_layout()


    pdf = args.out / "fig_z_space_collapse.pdf"
    png = args.out / "fig_z_space_collapse.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")

    # Persist per-model z-stats so the report can refer to numerics later
    rows = []
    for m, z in z_per_model.items():
        rows.append({
            "model": m, "feature": args.feature, "n": int(len(z)),
            "z_mean": float(z.mean()), "z_std": float(z.std()),
            "z_skew": float(scipy_stats.skew(z)),
            "z_kurtosis": float(scipy_stats.kurtosis(z)),
            "z_p01": float(np.quantile(z, 0.01)),
            "z_p99": float(np.quantile(z, 0.99)),
            "z_min": float(z.min()), "z_max": float(z.max()),
        })
    out_csv = args.out / "z_space_collapse_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
