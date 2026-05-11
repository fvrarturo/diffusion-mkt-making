"""
88_per_regime_g1.py — Per-regime G1 visualization from P0.D outputs.

Reads:
    results/_archive/<model>/validation_<regime>/G1_pass_fail_summary.csv
        for model in {v2, v5, v9} × regime in {base, high_vol, toxic, thin}

Writes:
    results/report_figures/fig_p0d_g1_per_regime.{pdf,png}     (heatmap)
    results/report_figures/fig_p0d_g1_per_regime_bars.{pdf,png}(grouped bars)
    results/report_figures/per_regime_g1_summary.csv           (long-format)

Two views:
  HEATMAP — 16 G1 checks × (3 models × 4 regimes) cells, green/red pass/fail.
            Best for spotting which checks degrade in which regimes.
  BARS    — pass-count by (model, regime), grouped by model. Best for the
            "regime conditioning works because high-vol/toxic regimes show
            distinct pass rates" headline.

Local-only (reads small CSVs). Run after pulling P0.D results back to Mac:
    python scripts/88_per_regime_g1.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "results" / "_archive"
OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["v2", "v5", "v9"]
REGIMES = ["base", "high_vol", "toxic", "thin"]

# Per-check real reference values (used when CSV Value field is a raw float
# rather than a "real=X, synth=Y" formatted string). Same dict as the one
# in scripts/80_report_figures.py — kept in sync.
REAL_VALUES = {
    "Return W-1":                      0.0,
    "Spread W-1":                      0.0,
    "Return excess kurtosis":          755.0,
    "Spread 1-tick fraction":          0.95,
    "Trade fraction (synth vs real)":  0.0477,
    "ACF(|r|) lag-50 deviation":       0.0,
    "ACF(|r|) power-law":              0.21,
    "ACF(|r|) power-law β":            0.21,
    "Trade-sign lag-1 ACF sign":       0.675,
    "Return lag-1 ACF":                0.0,
    "RV signature shape":              0.0,
    "E[r|I] sign at =10":              3.68e-05,
    "E[r|I] sign at Δ=10":             3.68e-05,
    "E[r|I] sign at =50":              1.107e-04,
    "E[r|I] sign at Δ=50":             1.107e-04,
    "Joint spread-imbalance JSD":      0.0,
    "Conditional moments (|z|<2)":     0.0,
    "Crossed books rate (decode)":     0.0,
    "Size tail exponent":              1.567,
}


# ─── Value parsers ────────────────────────────────────────────────────────
_NUM_RE = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
_SYNTH_PATTERNS = [
    re.compile(r'synth\s*=\s*' + _NUM_RE),
    re.compile(r'β_synth\s*=\s*' + _NUM_RE),
    re.compile(r'ratio_synth\s*=\s*' + _NUM_RE),
]
_REAL_PATTERNS = [
    re.compile(r'real\s*=\s*' + _NUM_RE),
    re.compile(r'β_real\s*=\s*' + _NUM_RE),
    re.compile(r'ratio_real\s*=\s*' + _NUM_RE),
]


def _try_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_synth_value(v) -> float | None:
    """Extract synth value from a CSV Value field. Handles raw floats and
    'real=X, synth=Y' / 'β_real=X, β_synth=Y' / 'ratio_real=X, ratio_synth=Y'
    formatted strings."""
    f = _try_float(v)
    if f is not None:
        return f
    s = str(v)
    for pat in _SYNTH_PATTERNS:
        m = pat.search(s)
        if m:
            return float(m.group(1))
    return None


def parse_real_value(v) -> float | None:
    """Extract real reference value from the same Value field, if present."""
    s = str(v)
    for pat in _REAL_PATTERNS:
        m = pat.search(s)
        if m:
            return float(m.group(1))
    return None

REGIME_DISPLAY = {
    "base":     "base",
    "high_vol": "high\nvol",
    "toxic":    "toxic",
    "thin":     "thin",
}
MODEL_GROUP_COLOR = {"v2": "#1f77b4", "v5": "#2ca02c", "v9": "#9467bd"}


def load_per_regime() -> pd.DataFrame:
    """Returns long-format frame with columns: model, regime, check, pass, value.

    Column matching is case-insensitive — script 50's G1_pass_fail_summary.csv
    uses capitalized headers (Check, Pass, Value) while older validators may
    use lowercase. We also explicitly skip the bare '#' index column when
    falling back to the first-column heuristic.
    """
    def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower_to_orig = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_to_orig:
                return lower_to_orig[cand.lower()]
        return None

    rows: list[dict] = []
    for m in MODELS:
        for r in REGIMES:
            csv = ARCHIVE / m / f"validation_{r}" / "G1_pass_fail_summary.csv"
            if not csv.exists():
                print(f"  MISSING {csv.relative_to(ROOT)}")
                continue
            df = pd.read_csv(csv)
            check_col = _find(df, ["check", "criterion", "name"]) \
                        or next((c for c in df.columns if c != "#"), df.columns[0])
            pass_col = _find(df, ["pass", "passed", "result"])
            value_col = _find(df, ["value", "synth_value", "metric"])
            for _, row in df.iterrows():
                p = row.get(pass_col, np.nan) if pass_col else np.nan
                p_num = (1.0 if (p is True or str(p).strip().lower() == "true")
                         else 0.0 if (p is False or str(p).strip().lower() == "false")
                         else np.nan)
                check = str(row[check_col])
                raw = row.get(value_col, np.nan) if value_col else np.nan
                synth_v = parse_synth_value(raw)
                # Real reference: prefer the value embedded in the Value
                # field ("real=X"); fall back to the per-check dict.
                real_v = parse_real_value(raw)
                if real_v is None:
                    real_v = REAL_VALUES.get(check)
                rows.append({
                    "model": m, "regime": r,
                    "check": check,
                    "pass": p_num,
                    "value_raw": raw,
                    "synth": synth_v,
                    "real": real_v,
                })
    return pd.DataFrame(rows)


def _fmt(x: float | None) -> str:
    """Compact numeric formatter for heatmap cells."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    a = abs(x)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e4:
        return f"{x:.2g}"
    if a < 1:
        return f"{x:.3g}"
    return f"{x:.2f}"


# ─── Plot A: numerical heatmap with per-row cross-sectional gradient ──────
def fig_heatmap(df: pd.DataFrame) -> None:
    """Each cell shows the synth value; color encodes per-row error vs the
    real reference (greenest = closest to real, reddest = farthest).
    Coloring is cross-sectional WITHIN each row, so a check whose magnitudes
    span 6 orders is still legible.
    """
    if df.empty:
        print("[heatmap] no data — skipped"); return
    # Drop checks that don't produce a numeric synth value
    # ("Conditional moments (|z|<2)" outputs the string "majority").
    DROP_CHECKS = {"TOTAL_PASS", "Conditional moments (|z|<2)"}
    df = df[~df["check"].isin(DROP_CHECKS)].copy()
    df["mr"] = df["model"] + "_" + df["regime"]

    val_pivot = df.pivot(index="check", columns="mr", values="synth").astype(float)
    pass_pivot = df.pivot(index="check", columns="mr", values="pass").astype(float)
    real_series = df.dropna(subset=["real"]).groupby("check")["real"].first()

    col_order = [f"{m}_{r}" for m in MODELS for r in REGIMES
                 if f"{m}_{r}" in val_pivot.columns]
    val_pivot = val_pivot[col_order]
    pass_pivot = pass_pivot[col_order]

    # Per-row error from real reference, then min-max normalize per row to [0, 1].
    # Rows without a real reference get a uniform mid-grey color (0.5).
    err = pd.DataFrame(index=val_pivot.index, columns=val_pivot.columns, dtype=float)
    for chk in val_pivot.index:
        if chk not in real_series.index:
            err.loc[chk, :] = np.nan
            continue
        ref = real_series.loc[chk]
        err.loc[chk, :] = (val_pivot.loc[chk] - ref).abs()
    err_norm = err.copy()
    for chk in err_norm.index:
        row = err_norm.loc[chk].values.astype(float)
        finite = row[np.isfinite(row)]
        if len(finite) == 0:
            err_norm.loc[chk] = np.nan
            continue
        lo, hi = finite.min(), finite.max()
        if hi - lo < 1e-15:
            err_norm.loc[chk] = 0.0
        else:
            err_norm.loc[chk] = (row - lo) / (hi - lo)

    # Sort rows by mean per-row error rank (best-overall checks at top)
    err_norm["_mean"] = err_norm.mean(axis=1, skipna=True)
    err_norm = err_norm.sort_values("_mean").drop(columns="_mean")
    val_pivot = val_pivot.reindex(err_norm.index)
    pass_pivot = pass_pivot.reindex(err_norm.index)

    # Annotation grid: raw synth value (no pass-marker — keep cells uncluttered)
    annot = pd.DataFrame(index=val_pivot.index, columns=val_pivot.columns, dtype=object)
    for chk in val_pivot.index:
        for col in val_pivot.columns:
            annot.loc[chk, col] = _fmt(val_pivot.loc[chk, col])

    # Add a "real" column on the right showing the per-check reference
    real_col_name = "REAL"
    err_norm[real_col_name] = 0.0   # always best (= itself)
    annot[real_col_name] = real_series.reindex(err_norm.index).map(_fmt)
    real_v_for_pivot = real_series.reindex(err_norm.index).fillna(np.nan)
    val_pivot[real_col_name] = real_v_for_pivot

    cmap = LinearSegmentedColormap.from_list(
        "g_to_r", ["#1a7d3a", "#7fc97f", "#f5f5dc", "#fdae61", "#d73027"])

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(err_norm, cmap=cmap, vmin=0, vmax=1, cbar_kws={
                    "label": "per-row error (0 = best in row, 1 = worst in row)",
                    "shrink": 0.6,
                },
                linewidths=0.6, linecolor="white",
                annot=annot, fmt="", annot_kws={"fontsize": 8},
                ax=ax)

    # Header: model labels above each model's 4-regime block.
    # Put labels just BELOW the title (via axes coords) so they never collide
    # with the title regardless of pad.
    col_labels = [REGIME_DISPLAY[r] for m in MODELS for r in REGIMES
                  if f"{m}_{r}" in col_order] + ["real"]
    ax.set_xticklabels(col_labels, rotation=0, fontsize=9)
    n_per_model = len(REGIMES)
    for i, m in enumerate(MODELS):
        x_mid = i * n_per_model + n_per_model / 2
        ax.text(x_mid, -0.4, m, ha="center", va="bottom",
                fontsize=14, fontweight="bold",
                color=MODEL_GROUP_COLOR[m])
        if i > 0:
            ax.axvline(i * n_per_model, color="black", linewidth=1.5)
    # Divider between synth columns and the REAL column
    ax.axvline(len(MODELS) * n_per_model, color="black", linewidth=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Per-Regime G1: synth values with cross-sectional error gradient",
                 pad=38, fontsize=12)
    pdf = OUT / "fig_p0d_g1_per_regime.pdf"
    png = OUT / "fig_p0d_g1_per_regime.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


# ─── Plot B: pass-count bars (model × regime) ─────────────────────────────
def fig_bars(df: pd.DataFrame) -> None:
    if df.empty:
        print("[bars] no data — skipped"); return
    df = df[df["check"] != "TOTAL_PASS"].copy()
    counts = df.groupby(["model", "regime"])["pass"].sum().reset_index()
    counts["regime"] = pd.Categorical(counts["regime"],
                                        categories=REGIMES, ordered=True)
    n_checks = df["check"].nunique()

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(REGIMES))
    w = 0.27
    for i, m in enumerate(MODELS):
        sub = counts[counts["model"] == m].set_index("regime").reindex(REGIMES)
        ax.bar(x + (i - 1) * w, sub["pass"].fillna(0), w,
               label=m, color=MODEL_GROUP_COLOR[m],
               edgecolor="k", linewidth=0.4)
        # Annotate each bar
        for j, v in enumerate(sub["pass"].fillna(0)):
            if pd.notna(v):
                ax.annotate(f"{int(v)}/{n_checks}",
                            (x[j] + (i - 1) * w, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_DISPLAY[r].replace("\n", " ") for r in REGIMES])
    ax.set_xlabel("regime (per-regime sampling, B/C/D validation)")
    ax.set_ylabel(f"G1 pass count (out of {n_checks})")
    ax.set_title("Regime Conditioning Effectiveness — pass count varies by regime\n"
                 "(if conditioning works, hard regimes (toxic/thin) should show distinct results)",
                 pad=10)
    ax.legend(title="generator", loc="upper right", fontsize=9)
    ax.set_ylim(0, n_checks + 1)
    pdf = OUT / "fig_p0d_g1_per_regime_bars.pdf"
    png = OUT / "fig_p0d_g1_per_regime_bars.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


def main() -> None:
    print(f"loading per-regime G1 outputs from {ARCHIVE}")
    df = load_per_regime()
    if df.empty:
        print("no data found — make sure P0.D outputs are pulled to "
              "results/_archive/<model>/validation_<regime>/")
        return
    print(f"loaded {len(df)} (model, regime, check) rows  "
          f"({df['model'].nunique()} models × {df['regime'].nunique()} regimes × "
          f"{df['check'].nunique()} checks)")

    out_csv = OUT / "per_regime_g1_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    fig_heatmap(df)
    fig_bars(df)


if __name__ == "__main__":
    main()
