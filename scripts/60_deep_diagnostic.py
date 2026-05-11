#!/usr/bin/env python3
"""
Deep cross-model diagnostic suite for the diffusion LOB generator workstream.

Goes beyond the validation suite (Work5) by examining:
  A. Per-feature marginal calibration (not just return kurtosis)
  B. Cross-feature joint structure (correlation matrices, not just OFI slope)
  C. z-space calibration for copula models (what the inverse CDF actually receives)
  D. Window-level statistics (degenerate window detection, per-window variance)
  E. Temporal structure per feature (not just |return| ACF)
  F. Regime-conditional analysis

Usage (on cluster):
  python scripts/60_deep_diagnostic.py \
      --real-dir data/processed/val \
      --synth-dirs \
          $HOME/orcd/scratch/diffmm/synthetic/intc_v8/INTC \
          $HOME/orcd/scratch/diffmm/synthetic/intc_v8_b/INTC \
      --labels v8 v8_b \
      --copula-json data/processed/norm_stats_INTC.copula.json \
      --copula-models v8 v8_b \
      --out results/deep_diagnostic

Produces:
  - deep_diagnostic_summary.txt    (human-readable full report)
  - per_feature_stats.csv          (every feature × every model × every stat)
  - correlation_comparison.csv     (Pearson + Spearman matrices, real vs each model)
  - window_level_stats.csv         (per-window mean/std/kurt for each model)
  - z_space_stats.csv              (for copula models: z-space distribution per feature)
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
# Configuration — adapt these to your column names
# ═══════════════════════════════════════════════════════════════════

# Columns present in the canonical decoded parquets (see src/diffmm/generator/decode.py).
# All entries here are columns the schema validator emits; missing ones are skipped
# at runtime in `available = [c for c in CONTINUOUS_FEATURES if c in real_df.columns]`.
CONTINUOUS_FEATURES = [
    "mid_return",   # headline metric — kurtosis, OFI regression
    "spread",       # ask_px - bid_px
    "bid_sz", "ask_sz",
    "bid_px", "ask_px", "mid",
    "bid_dist", "ask_dist",
    "trade_sz",     # null on quote_update events
    "trade_dist",   # null on quote_update events
]

# Which column is "the return" for OFI regression, kurtosis headline, etc.
RETURN_COL = "mid_return"

# How to compute OFI from the available columns.
# Decoded parquets do NOT have an OFI column — compute proxy from bid/ask size diffs
# (per-tape, to avoid cross-tape diffs polluting window boundaries).
OFI_COL = None

# Trade-sign column — yes, our decoded parquets have it (Int8, ∈{-1, 0, +1, None})
TRADE_SIGN_COL = "trade_sign"

# Maximum tapes to load per model (for speed; set None for all)
MAX_TAPES_PER_REGIME = 50

# Maximum events per model for correlation analysis (subsample for speed)
MAX_EVENTS_FOR_CORR = 500_000


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_tapes(synth_dir: str, max_per_regime: Optional[int] = MAX_TAPES_PER_REGIME) -> pd.DataFrame:
    """Load stitched parquets from a synth directory. Expects regime subdirs."""
    frames = []
    synth_path = Path(synth_dir)

    # Try regime subdirs first; fall back to flat directory
    regime_dirs = [d for d in synth_path.iterdir() if d.is_dir()]
    if not regime_dirs:
        regime_dirs = [synth_path]

    for regime_dir in sorted(regime_dirs):
        parquets = sorted(regime_dir.glob("*.parquet"))
        if max_per_regime:
            parquets = parquets[:max_per_regime]
        for pq in parquets:
            try:
                df = pd.read_parquet(pq)
                df["_regime"] = regime_dir.name
                df["_tape_id"] = pq.stem
                frames.append(df)
            except Exception as e:
                print(f"  WARN: skipping {pq}: {e}", file=sys.stderr)
    if not frames:
        raise FileNotFoundError(f"No parquets found in {synth_dir}")
    return pd.concat(frames, ignore_index=True)


def load_real(real_dir: str) -> pd.DataFrame:
    """Load real validation parquets. Searches `<real_dir>/*.parquet` first,
    then `<real_dir>/*/*.parquet` (e.g. ticker subdir like `val/INTC/*.parquet`)."""
    p = Path(real_dir)
    parquets = sorted(p.glob("*.parquet"))
    if not parquets:
        parquets = sorted(p.glob("*/*.parquet"))
    frames = []
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
            df["_tape_id"] = pq.stem
            frames.append(df)
        except Exception:
            pass
    if not frames:
        raise FileNotFoundError(f"No parquets in {real_dir} (also tried {real_dir}/*/)")
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════
# Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OFI proxy and trade sign if not present.

    OFI is computed PER-TAPE so window boundaries don't pollute the diff.
    For v8 (256-event raw windows) this matters — across-window diffs would
    inject artificial OFI spikes at every tape boundary.
    """
    df = df.copy()

    # OFI proxy: Δbid_sz − Δask_sz (simplified Cont-Kukanov-Stoikov)
    if (OFI_COL is None or OFI_COL not in df.columns):
        if "bid_sz" in df.columns and "ask_sz" in df.columns:
            if "_tape_id" in df.columns:
                # Per-tape diff — first row of each tape gets fillna(0)
                df["ofi_proxy"] = (
                    df.groupby("_tape_id")["bid_sz"].diff().fillna(0)
                    - df.groupby("_tape_id")["ask_sz"].diff().fillna(0)
                )
            else:
                df["ofi_proxy"] = df["bid_sz"].diff().fillna(0) - df["ask_sz"].diff().fillna(0)
        else:
            df["ofi_proxy"] = 0.0
    else:
        df["ofi_proxy"] = df[OFI_COL]

    # Trade sign: prefer the canonical column; fallback to sign(mid_return).
    # Note our trade_sign is null on quote_update events — fillna(0) so
    # downstream stats include the no-trade rows as "no signed trade".
    if TRADE_SIGN_COL is not None and TRADE_SIGN_COL in df.columns:
        df["trade_sign"] = pd.to_numeric(df[TRADE_SIGN_COL], errors="coerce").fillna(0)
    elif RETURN_COL in df.columns:
        df["trade_sign"] = np.sign(df[RETURN_COL])
    else:
        df["trade_sign"] = 0.0

    return df


# ═══════════════════════════════════════════════════════════════════
# PART A: Per-feature marginal statistics
# ═══════════════════════════════════════════════════════════════════

def per_feature_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute comprehensive marginal stats for each continuous feature."""
    rows = []
    for col in CONTINUOUS_FEATURES:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) < 10:
            continue

        row = {
            "model": label,
            "feature": col,
            "n": len(vals),
            "mean": np.mean(vals),
            "std": np.std(vals),
            "skew": scipy_stats.skew(vals),
            "kurtosis": scipy_stats.kurtosis(vals),  # excess kurtosis
            "min": np.min(vals),
            "p01": np.percentile(vals, 1),
            "p05": np.percentile(vals, 5),
            "p25": np.percentile(vals, 25),
            "p50": np.percentile(vals, 50),
            "p75": np.percentile(vals, 75),
            "p95": np.percentile(vals, 95),
            "p99": np.percentile(vals, 99),
            "max": np.max(vals),
            "frac_zero": np.mean(vals == 0),
            "frac_positive": np.mean(vals > 0),
            "frac_negative": np.mean(vals < 0),
            # Tail mass: what fraction of values exceed ±2σ, ±3σ, ±4σ
            "frac_gt_2sigma": np.mean(np.abs(vals - np.mean(vals)) > 2 * np.std(vals)) if np.std(vals) > 0 else 0,
            "frac_gt_3sigma": np.mean(np.abs(vals - np.mean(vals)) > 3 * np.std(vals)) if np.std(vals) > 0 else 0,
            "frac_gt_4sigma": np.mean(np.abs(vals - np.mean(vals)) > 4 * np.std(vals)) if np.std(vals) > 0 else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PART B: Cross-feature joint structure
# ═══════════════════════════════════════════════════════════════════

def correlation_matrices(df: pd.DataFrame, label: str) -> dict:
    """Compute Pearson and Spearman correlation matrices for available features."""
    cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) > MAX_EVENTS_FOR_CORR:
        sub = sub.sample(MAX_EVENTS_FOR_CORR, random_state=42)

    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    return {"label": label, "pearson": pearson, "spearman": spearman, "cols": cols}


def ofi_return_regression(df: pd.DataFrame, label: str, deltas: list[int] = [1, 5, 10, 50, 100]) -> pd.DataFrame:
    """OFI → future return regression at multiple horizons."""
    rows = []
    if RETURN_COL not in df.columns or "ofi_proxy" not in df.columns:
        return pd.DataFrame()

    ofi = df["ofi_proxy"].values
    ret = df[RETURN_COL].values

    for delta in deltas:
        if delta >= len(ret):
            continue
        # Forward return: sum of next delta returns
        fwd_ret = pd.Series(ret).rolling(delta).sum().shift(-delta).values
        mask = np.isfinite(ofi) & np.isfinite(fwd_ret)
        if mask.sum() < 100:
            continue

        x, y = ofi[mask], fwd_ret[mask]
        if np.std(x) == 0 or np.std(y) == 0:
            rows.append({"model": label, "delta": delta, "slope": 0, "r2": 0,
                         "slope_sign": "zero", "pearson_r": 0, "spearman_r": 0})
            continue

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        sp_r, _ = scipy_stats.spearmanr(x, y)
        rows.append({
            "model": label, "delta": delta,
            "slope": slope, "r2": r_value ** 2, "p_value": p_value,
            "slope_sign": "+" if slope > 0 else "-",
            "pearson_r": r_value, "spearman_r": sp_r,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PART C: z-space calibration (copula models only)
# ═══════════════════════════════════════════════════════════════════

def load_copula_transform(copula_json: str):
    """Load CopulaTransform from saved JSON. Returns a lightweight object
    that can forward-transform original-scale data to z-space."""
    with open(copula_json) as f:
        meta = json.load(f)

    # Load the sorted values for each feature
    npz_path = copula_json.replace(".json", ".npz")
    if os.path.exists(npz_path):
        arrays = dict(np.load(npz_path))
    else:
        arrays = {}

    return meta, arrays


def forward_copula(values: np.ndarray, sorted_vals: np.ndarray, clip_q: float = 0.001) -> np.ndarray:
    """Apply empirical CDF + probit to get z-values."""
    ranks = np.searchsorted(sorted_vals, values)
    u = (ranks + 0.5) / (len(sorted_vals) + 1)
    u = np.clip(u, clip_q, 1.0 - clip_q)
    return scipy_stats.norm.ppf(u)


def z_space_analysis(df: pd.DataFrame, copula_meta: dict, copula_arrays: dict,
                     label: str,
                     force_continuous: tuple[str, ...] = ("mid_return",)) -> pd.DataFrame:
    """For copula models: forward-transform synth output to z-space and analyze.

    Schema (matches src/diffmm/data/copula_transform.py CopulaTransform.save):
      meta["feature_columns"]    — list of column names, ordered by index
      meta["discrete_features"]  — list of indices (these use z-score, not CDF)
      meta["ecdf_keys"]          — list of indices that have CDF arrays
      arrays[f"ecdf_{idx}"]      — sorted training values for column idx

    `force_continuous` names features whose z-space stats we want even if they
    are flagged as discrete in the copula metadata. mid_return is the canonical
    case: it is zero-inflated (point mass at 0 → flagged discrete) but still
    carries a meaningful continuous component whose z-collapse is the heart of
    the v8/v9 copula story.
    """
    rows = []
    clip_q = copula_meta.get("clip_quantile", 0.001)
    feature_columns = copula_meta.get("feature_columns", [])
    discrete = set(copula_meta.get("discrete_features", []))

    for idx, fname in enumerate(feature_columns):
        if idx in discrete and fname not in force_continuous:
            continue
        if fname not in df.columns:
            continue
        key = f"ecdf_{idx}"
        if key not in copula_arrays:
            continue

        vals = df[fname].dropna().values
        if len(vals) < 10:
            continue
        sorted_vals = copula_arrays[key]
        z_vals = forward_copula(vals, sorted_vals, clip_q)
        col = fname  # keep downstream code referring to `col` working

        # Is the z-space distribution actually N(0,1)?
        row = {
            "model": label,
            "feature": col,
            "z_mean": np.mean(z_vals),
            "z_std": np.std(z_vals),
            "z_skew": scipy_stats.skew(z_vals),
            "z_kurtosis": scipy_stats.kurtosis(z_vals),  # excess; should be 0 for Gaussian
            "z_min": np.min(z_vals),
            "z_max": np.max(z_vals),
            "z_p01": np.percentile(z_vals, 1),
            "z_p99": np.percentile(z_vals, 99),
            # Kolmogorov-Smirnov test against N(0,1)
            "z_ks_stat": scipy_stats.kstest(z_vals[:10000], "norm")[0],
            "z_ks_pval": scipy_stats.kstest(z_vals[:10000], "norm")[1],
            # Tail mass in z-space
            "z_frac_gt_2": np.mean(np.abs(z_vals) > 2),  # should be ~4.56% for N(0,1)
            "z_frac_gt_3": np.mean(np.abs(z_vals) > 3),  # should be ~0.27%
            "z_frac_gt_4": np.mean(np.abs(z_vals) > 4),  # should be ~0.006%
            # Reference: N(0,1) expectations
            "ref_frac_gt_2": 0.0456,
            "ref_frac_gt_3": 0.0027,
            "ref_frac_gt_4": 0.0001,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PART D: Window-level statistics
# ═══════════════════════════════════════════════════════════════════

def window_level_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-window (per-tape) statistics to detect degenerate windows."""
    rows = []
    for tape_id, group in df.groupby("_tape_id"):
        if RETURN_COL not in group.columns:
            continue
        vals = group[RETURN_COL].dropna().values
        if len(vals) < 10:
            continue

        row = {
            "model": label,
            "tape_id": tape_id,
            "n_events": len(group),
            "ret_mean": np.mean(vals),
            "ret_std": np.std(vals),
            "ret_kurtosis": scipy_stats.kurtosis(vals) if np.std(vals) > 1e-20 else np.nan,
            "ret_min": np.min(vals),
            "ret_max": np.max(vals),
            "is_degenerate": np.std(vals) < 1e-15,
            "frac_zero_return": np.mean(vals == 0),
        }

        # Per-window OFI-return correlation
        if "ofi_proxy" in group.columns:
            ofi_vals = group["ofi_proxy"].dropna().values[:len(vals)]
            if len(ofi_vals) == len(vals) and np.std(ofi_vals) > 0 and np.std(vals) > 0:
                row["ofi_ret_corr"] = np.corrcoef(ofi_vals, vals)[0, 1]
            else:
                row["ofi_ret_corr"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PART E: Temporal structure per feature
# ═══════════════════════════════════════════════════════════════════

def temporal_stats(df: pd.DataFrame, label: str, lags: list[int] = [1, 5, 10, 50, 100]) -> pd.DataFrame:
    """ACF of each feature at multiple lags, plus trade-sign persistence."""
    rows = []
    for col in CONTINUOUS_FEATURES + ["ofi_proxy", "trade_sign"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) < 200:
            continue

        for lag in lags:
            if lag >= len(vals) - 1:
                continue
            # ACF at this lag
            n = len(vals)
            mean = np.mean(vals)
            var = np.var(vals)
            if var < 1e-30:
                acf = 0.0
            else:
                acf = np.mean((vals[:n - lag] - mean) * (vals[lag:] - mean)) / var

            rows.append({
                "model": label,
                "feature": col,
                "lag": lag,
                "acf": acf,
            })

        # Also: ACF of absolute values (volatility clustering proxy)
        if col == RETURN_COL:
            abs_vals = np.abs(vals)
            abs_mean = np.mean(abs_vals)
            abs_var = np.var(abs_vals)
            for lag in lags:
                if lag >= len(abs_vals) - 1 or abs_var < 1e-30:
                    continue
                acf = np.mean((abs_vals[:len(abs_vals) - lag] - abs_mean) *
                              (abs_vals[lag:] - abs_mean)) / abs_var
                rows.append({
                    "model": label,
                    "feature": f"|{col}|",
                    "lag": lag,
                    "acf": acf,
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PART F: Regime-conditional analysis
# ═══════════════════════════════════════════════════════════════════

def regime_conditional_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-regime marginal stats for key features."""
    if "_regime" not in df.columns:
        return pd.DataFrame()
    rows = []
    for regime, group in df.groupby("_regime"):
        for col in [RETURN_COL, "spread", "ofi_proxy"]:
            if col not in group.columns:
                continue
            vals = group[col].dropna().values
            if len(vals) < 10:
                continue
            rows.append({
                "model": label, "regime": regime, "feature": col,
                "mean": np.mean(vals), "std": np.std(vals),
                "skew": scipy_stats.skew(vals),
                "kurtosis": scipy_stats.kurtosis(vals),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def write_report(out_dir: str, real_label: str, model_labels: list[str],
                 feat_stats: pd.DataFrame, corr_results: list[dict],
                 ofi_regs: pd.DataFrame, window_stats: pd.DataFrame,
                 temporal: pd.DataFrame, regime_stats: pd.DataFrame,
                 z_stats: Optional[pd.DataFrame] = None):
    """Write all output files."""
    os.makedirs(out_dir, exist_ok=True)

    # Save CSVs
    feat_stats.to_csv(f"{out_dir}/per_feature_stats.csv", index=False)
    ofi_regs.to_csv(f"{out_dir}/ofi_regression.csv", index=False)
    window_stats.to_csv(f"{out_dir}/window_level_stats.csv", index=False)
    temporal.to_csv(f"{out_dir}/temporal_acf.csv", index=False)
    if len(regime_stats) > 0:
        regime_stats.to_csv(f"{out_dir}/regime_conditional.csv", index=False)
    if z_stats is not None and len(z_stats) > 0:
        z_stats.to_csv(f"{out_dir}/z_space_stats.csv", index=False)

    # Save correlation matrices
    corr_rows = []
    for cr in corr_results:
        for method in ["pearson", "spearman"]:
            mat = cr[method]
            for i, c1 in enumerate(cr["cols"]):
                for j, c2 in enumerate(cr["cols"]):
                    corr_rows.append({
                        "model": cr["label"], "method": method,
                        "feature_1": c1, "feature_2": c2,
                        "correlation": mat.iloc[i, j],
                    })
    pd.DataFrame(corr_rows).to_csv(f"{out_dir}/correlation_comparison.csv", index=False)

    # ── Human-readable summary report ──
    lines = []
    lines.append("=" * 80)
    lines.append("DEEP DIAGNOSTIC REPORT — Diffusion LOB Generator Workstream")
    lines.append("=" * 80)
    lines.append("")

    # ── A. Per-feature marginal comparison ──
    lines.append("─" * 80)
    lines.append("A. PER-FEATURE MARGINAL COMPARISON")
    lines.append("─" * 80)
    for feature in feat_stats["feature"].unique():
        lines.append(f"\n  Feature: {feature}")
        sub = feat_stats[feat_stats["feature"] == feature].set_index("model")
        if real_label in sub.index:
            real_row = sub.loc[real_label]
            lines.append(f"    Real:  mean={real_row['mean']:.6g}  std={real_row['std']:.6g}  "
                         f"skew={real_row['skew']:.3f}  kurt={real_row['kurtosis']:.1f}  "
                         f"range=[{real_row['min']:.6g}, {real_row['max']:.6g}]")
            lines.append(f"           tail mass: >2σ={real_row['frac_gt_2sigma']:.4f}  "
                         f">3σ={real_row['frac_gt_3sigma']:.4f}  "
                         f">4σ={real_row['frac_gt_4sigma']:.4f}  "
                         f"frac_zero={real_row['frac_zero']:.4f}")
        for model in model_labels:
            if model not in sub.index:
                continue
            r = sub.loc[model]
            # Compute ratios vs real
            std_ratio = r["std"] / real_row["std"] if real_row["std"] > 0 else np.nan
            lines.append(f"    {model:12s}: mean={r['mean']:.6g}  std={r['std']:.6g} ({std_ratio:.2f}×real)  "
                         f"skew={r['skew']:.3f}  kurt={r['kurtosis']:.1f}  "
                         f"range=[{r['min']:.6g}, {r['max']:.6g}]")
            lines.append(f"           {'':12s}  tail mass: >2σ={r['frac_gt_2sigma']:.4f}  "
                         f">3σ={r['frac_gt_3sigma']:.4f}  "
                         f">4σ={r['frac_gt_4sigma']:.4f}  "
                         f"frac_zero={r['frac_zero']:.4f}")

    # ── B. Cross-feature correlation comparison ──
    lines.append("\n" + "─" * 80)
    lines.append("B. CROSS-FEATURE JOINT STRUCTURE")
    lines.append("─" * 80)

    # Find real correlation matrix
    real_corr = None
    for cr in corr_results:
        if cr["label"] == real_label:
            real_corr = cr
            break

    if real_corr is not None:
        for cr in corr_results:
            if cr["label"] == real_label:
                continue
            for method in ["pearson", "spearman"]:
                diff = cr[method] - real_corr[method]
                frob = np.sqrt((diff ** 2).sum().sum())
                max_diff = diff.abs().values[np.triu_indices_from(diff.values, k=1)].max()
                lines.append(f"\n  {cr['label']} vs Real — {method.capitalize()}")
                lines.append(f"    Frobenius norm of difference: {frob:.4f}")
                lines.append(f"    Max absolute pairwise difference: {max_diff:.4f}")

                # Show the biggest discrepancies
                n = len(cr["cols"])
                diffs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        diffs.append({
                            "pair": f"{cr['cols'][i]} × {cr['cols'][j]}",
                            "real": real_corr[method].iloc[i, j],
                            "synth": cr[method].iloc[i, j],
                            "diff": diff.iloc[i, j],
                        })
                diffs.sort(key=lambda x: abs(x["diff"]), reverse=True)
                lines.append(f"    Top 5 mismatched pairs:")
                for d in diffs[:5]:
                    lines.append(f"      {d['pair']:30s}  real={d['real']:+.3f}  "
                                 f"synth={d['synth']:+.3f}  diff={d['diff']:+.3f}")

    # ── B2. OFI → return regression ──
    lines.append("\n  OFI → return regression:")
    if len(ofi_regs) > 0:
        for delta in sorted(ofi_regs["delta"].unique()):
            lines.append(f"\n    Δ = {delta}:")
            sub = ofi_regs[ofi_regs["delta"] == delta].set_index("model")
            for model in [real_label] + model_labels:
                if model not in sub.index:
                    continue
                r = sub.loc[model]
                lines.append(f"      {model:12s}: slope={r['slope']:.6g}  R²={r['r2']:.4f}  "
                             f"sign={r['slope_sign']}  spearman_r={r['spearman_r']:.3f}")

    # ── C. z-space calibration ──
    if z_stats is not None and len(z_stats) > 0:
        lines.append("\n" + "─" * 80)
        lines.append("C. Z-SPACE CALIBRATION (copula models)")
        lines.append("    Reference: N(0,1) has std=1.0, kurt=0.0, >2σ=4.56%, >3σ=0.27%")
        lines.append("─" * 80)
        for _, row in z_stats.iterrows():
            lines.append(f"\n  {row['model']} / {row['feature']}:")
            lines.append(f"    z-mean={row['z_mean']:.4f}  z-std={row['z_std']:.4f}  "
                         f"z-skew={row['z_skew']:.3f}  z-kurt={row['z_kurtosis']:.3f}")
            lines.append(f"    z-range=[{row['z_min']:.2f}, {row['z_max']:.2f}]  "
                         f"z-p01={row['z_p01']:.2f}  z-p99={row['z_p99']:.2f}")
            gt2_ratio = row["z_frac_gt_2"] / row["ref_frac_gt_2"] if row["ref_frac_gt_2"] > 0 else np.nan
            gt3_ratio = row["z_frac_gt_3"] / row["ref_frac_gt_3"] if row["ref_frac_gt_3"] > 0 else np.nan
            lines.append(f"    tail: >2σ={row['z_frac_gt_2']:.4f} ({gt2_ratio:.2f}× expected)  "
                         f">3σ={row['z_frac_gt_3']:.4f} ({gt3_ratio:.2f}× expected)  "
                         f">4σ={row['z_frac_gt_4']:.4f}")
            lines.append(f"    KS test: stat={row['z_ks_stat']:.4f}  p={row['z_ks_pval']:.4g}")

    # ── D. Window-level statistics ──
    lines.append("\n" + "─" * 80)
    lines.append("D. WINDOW-LEVEL STATISTICS")
    lines.append("─" * 80)
    for model in [real_label] + model_labels:
        sub = window_stats[window_stats["model"] == model]
        if len(sub) == 0:
            continue
        n_degen = sub["is_degenerate"].sum()
        lines.append(f"\n  {model}: {len(sub)} windows, {n_degen} degenerate ({100*n_degen/len(sub):.1f}%)")
        lines.append(f"    per-window ret_std: mean={sub['ret_std'].mean():.6g}  "
                     f"median={sub['ret_std'].median():.6g}  "
                     f"min={sub['ret_std'].min():.6g}  max={sub['ret_std'].max():.6g}")
        lines.append(f"    per-window ret_kurtosis: mean={sub['ret_kurtosis'].mean():.2f}  "
                     f"median={sub['ret_kurtosis'].median():.2f}")
        lines.append(f"    per-window frac_zero_return: mean={sub['frac_zero_return'].mean():.4f}")
        if "ofi_ret_corr" in sub.columns:
            valid_corr = sub["ofi_ret_corr"].dropna()
            if len(valid_corr) > 0:
                lines.append(f"    per-window OFI-return corr: mean={valid_corr.mean():.4f}  "
                             f"median={valid_corr.median():.4f}  "
                             f"frac_positive={np.mean(valid_corr > 0):.3f}")

    # ── E. Temporal structure ──
    lines.append("\n" + "─" * 80)
    lines.append("E. TEMPORAL STRUCTURE (ACF at key lags)")
    lines.append("─" * 80)
    for feature in temporal["feature"].unique():
        lines.append(f"\n  {feature}:")
        sub = temporal[temporal["feature"] == feature]
        pivot = sub.pivot_table(index="lag", columns="model", values="acf")
        header = f"    {'lag':>5s}"
        for col in [real_label] + model_labels:
            if col in pivot.columns:
                header += f"  {col:>10s}"
        lines.append(header)
        for lag in sorted(pivot.index):
            row_str = f"    {lag:5d}"
            for col in [real_label] + model_labels:
                if col in pivot.columns:
                    val = pivot.loc[lag, col]
                    row_str += f"  {val:10.4f}"
            lines.append(row_str)

    # ── F. Regime-conditional ──
    if len(regime_stats) > 0:
        lines.append("\n" + "─" * 80)
        lines.append("F. REGIME-CONDITIONAL ANALYSIS")
        lines.append("─" * 80)
        for feature in regime_stats["feature"].unique():
            lines.append(f"\n  {feature}:")
            sub = regime_stats[regime_stats["feature"] == feature]
            for model in model_labels:
                msub = sub[sub["model"] == model]
                if len(msub) == 0:
                    continue
                lines.append(f"    {model}:")
                for _, row in msub.iterrows():
                    lines.append(f"      {row['regime']:12s}  mean={row['mean']:.6g}  "
                                 f"std={row['std']:.6g}  kurt={row['kurtosis']:.1f}")

    # ── Summary verdict ──
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY — KEY QUESTIONS ANSWERED")
    lines.append("=" * 80)
    lines.append("""
  1. PER-FEATURE: Which features are well-calibrated vs broken across models?
     → See Section A. Compare std ratios, kurtosis, tail mass.

  2. JOINT STRUCTURE: Which feature-pair correlations break in post-v2 models?
     → See Section B. The Frobenius norm + top-5 pairs show exactly what breaks.

  3. Z-SPACE (copula): Is the model producing well-calibrated Gaussians?
     → See Section C. If z-std << 1.0 or z-kurtosis << 0, the model is
        sub-Gaussian and the inverse CDF can't restore tails.

  4. DEGENERATE WINDOWS: Are there dead windows dragging down aggregate stats?
     → See Section D. High frac_degenerate or frac_zero_return is a red flag.

  5. PER-WINDOW OFI CORRELATION: Is the OFI→return joint broken globally or
     only in certain windows/regimes?
     → See Section D (per-window OFI-return corr) and Section F (per-regime).

  6. TEMPORAL STRUCTURE PER FEATURE: Is the ACF failure return-specific or
     do other features also lose persistence?
     → See Section E. If all features lose ACF, it's a model-wide issue.
        If only returns lose ACF, it's feature-specific.
""")

    report_path = f"{out_dir}/deep_diagnostic_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}")
    print(f"CSVs written to {out_dir}/")

    # Also print to stdout
    for line in lines:
        print(line)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep cross-model diagnostic suite")
    parser.add_argument("--real-dir", required=True, help="Directory with real validation parquets")
    parser.add_argument("--synth-dirs", nargs="+", required=True, help="Directories with synth parquets (one per model)")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each synth dir (must match synth-dirs count)")
    parser.add_argument("--copula-json", default=None, help="Path to CopulaTransform .copula.json (for z-space analysis)")
    parser.add_argument("--copula-models", nargs="*", default=[], help="Which model labels used copula (for z-space analysis)")
    parser.add_argument("--out", default="results/deep_diagnostic", help="Output directory")
    parser.add_argument("--max-tapes", type=int, default=MAX_TAPES_PER_REGIME, help="Max tapes to load per regime")
    args = parser.parse_args()

    assert len(args.synth_dirs) == len(args.labels), "Must have same number of synth dirs and labels"

    global CONTINUOUS_FEATURES
    real_label = "real"

    # Load real data
    print(f"Loading real data from {args.real_dir} ...")
    real_df = load_real(args.real_dir)
    real_df = add_derived_features(real_df)
    print(f"  {len(real_df)} events loaded from real data")

    # Detect available columns and adjust CONTINUOUS_FEATURES
    available = [c for c in CONTINUOUS_FEATURES if c in real_df.columns]
    if not available:
        print(f"WARNING: None of {CONTINUOUS_FEATURES} found in real data.")
        print(f"  Available columns: {list(real_df.columns)}")
        print(f"  Edit CONTINUOUS_FEATURES at the top of this script to match your data.")
        sys.exit(1)
    missing = [c for c in CONTINUOUS_FEATURES if c not in real_df.columns]
    if missing:
        print(f"  Note: missing columns (skipped): {missing}")
    CONTINUOUS_FEATURES = available

    # Load synth data
    synth_dfs = {}
    for sdir, label in zip(args.synth_dirs, args.labels):
        print(f"Loading {label} from {sdir} ...")
        try:
            df = load_tapes(sdir, args.max_tapes)
            df = add_derived_features(df)
            synth_dfs[label] = df
            print(f"  {len(df)} events loaded")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if not synth_dfs:
        print("No synth data loaded. Exiting.")
        sys.exit(1)

    model_labels = list(synth_dfs.keys())

    # Load copula transform if provided
    copula_meta, copula_arrays = None, None
    if args.copula_json and os.path.exists(args.copula_json):
        print(f"Loading copula transform from {args.copula_json} ...")
        copula_meta, copula_arrays = load_copula_transform(args.copula_json)

    # ── Run all analyses ──
    all_feat_stats = []
    all_corr_results = []
    all_ofi_regs = []
    all_window_stats = []
    all_temporal = []
    all_regime_stats = []
    all_z_stats = []

    # Real data
    print("Analyzing real data ...")
    all_feat_stats.append(per_feature_stats(real_df, real_label))
    all_corr_results.append(correlation_matrices(real_df, real_label))
    all_ofi_regs.append(ofi_return_regression(real_df, real_label))
    all_window_stats.append(window_level_stats(real_df, real_label))
    all_temporal.append(temporal_stats(real_df, real_label))

    # Each model
    for label, df in synth_dfs.items():
        print(f"Analyzing {label} ...")
        all_feat_stats.append(per_feature_stats(df, label))
        all_corr_results.append(correlation_matrices(df, label))
        all_ofi_regs.append(ofi_return_regression(df, label))
        all_window_stats.append(window_level_stats(df, label))
        all_temporal.append(temporal_stats(df, label))
        all_regime_stats.append(regime_conditional_stats(df, label))

        # z-space analysis for copula models
        if label in args.copula_models and copula_meta is not None:
            print(f"  z-space analysis for {label} ...")
            all_z_stats.append(z_space_analysis(df, copula_meta, copula_arrays, label))

    # Combine
    feat_stats = pd.concat(all_feat_stats, ignore_index=True)
    ofi_regs = pd.concat(all_ofi_regs, ignore_index=True) if all_ofi_regs else pd.DataFrame()
    window_stats = pd.concat(all_window_stats, ignore_index=True) if all_window_stats else pd.DataFrame()
    temporal_df = pd.concat(all_temporal, ignore_index=True) if all_temporal else pd.DataFrame()
    regime_df = pd.concat(all_regime_stats, ignore_index=True) if all_regime_stats else pd.DataFrame()
    z_df = pd.concat(all_z_stats, ignore_index=True) if all_z_stats else None

    # Write report
    write_report(
        args.out, real_label, model_labels,
        feat_stats, all_corr_results, ofi_regs,
        window_stats, temporal_df, regime_df, z_df
    )


if __name__ == "__main__":
    main()