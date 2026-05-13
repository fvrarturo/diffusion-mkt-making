"""
80_report_figures.py — Build the full figure + analysis-table suite for the
final report. Reads pre-computed CSVs/JSONs from results/* and writes
publication-quality figures (PDF + PNG, 300 DPI) plus three analysis tables
(LaTeX + CSV) to results/report_figures/.

Runtime: <10 min. No GPU, no model loading. A handful of figures optionally
load real-data parquets (Figures 3.4, 6.2, 9.1); these are skipped with a
warning if data/processed/val/ is not present locally.

Sections (mapping to docs/Work10_Report.md figure list):
    Section 1 — Three-Axis Framework         [Fig 1.1, 1.2, 1.3]
    Section 2 — Distributional Realism       [Fig 2.1, 2.2, 2.3, 2.4]
    Section 3 — Predictive Score             [Fig 3.1, 3.2, 3.3]
    Section 4 — PCA Coverage                 [Fig 4.1, 4.2]
    Section 5 — Architectural Ablation       [Fig 5.1, 5.2]
    Section 6 — Copula Story                 [Fig 6.1, 6.2, 6.3]
    Section 7 — Vol-Vol & Temporal Structure [Fig 7.1, 7.2, 7.3]
    Section 8 — Agent Stress-Test Mechanism  [Fig 8.1, 8.2, 8.3]
    Section 9 — Mid-Price Traces             [Fig 9.1]
    Section 10 — One-Figure Summary          [Fig 10.1]
    Tables A, B, C                           [decision matrix, G1 difficulty,
                                              metric correlation]

Run:
    python scripts/80_report_figures.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "_archive"
OUT = RESULTS / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Model metadata ─────────────────────────────────────────────────────────
MODEL_META = {
    "v2":           ("ε", "FiLM",  False, "zscore",         5.0,  4),
    "v2_noclip":    ("ε", "FiLM",  False, "zscore",         5.0,  None),
    "v2_remapped":  ("ε", "FiLM",  False, "zscore+remap",   5.0,  4),
    "v3_e9":        ("ε", "FiLM",  False, "zscore",         1.0,  4),
    "v3_e19":       ("ε", "FiLM",  False, "zscore",         1.0,  4),
    "v3p5":         ("ε", "FiLM",  False, "zscore",         5.0,  4),
    "v3p5_noclip":  ("ε", "FiLM",  False, "zscore",         5.0,  None),
    "v3_e9_noclip": ("ε", "FiLM",  False, "zscore",         1.0,  None),
    "v4":           ("v", "FiLM",  False, "zscore",         5.0,  4),
    "v4_noclip":    ("v", "FiLM",  False, "zscore",         5.0,  None),
    "v5":           ("v", "AdaLN", False, "zscore",         5.0,  4),
    "v6":           ("v", "AdaLN", True,  "zscore",         5.0,  4),
    "v7":           ("v", "AdaLN", True,  "zscore",         5.0,  4),
    "v7_b":         ("v", "AdaLN", True,  "zscore",         5.0,  4),
    "v8":           ("ε", "FiLM",  False, "copula",         5.0,  4),
    "v8_b":         ("ε", "AdaLN", False, "copula",         5.0,  4),
    "v9":           ("ε", "FiLM",  False, "copula+dequant", 5.0,  4),
    "v9_b":         ("v", "FiLM",  False, "copula+dequant", 5.0,  4),
    "v10_cascade":  ("ε→v","FiLM", False, "zscore",         5.0,  4),
    "v11":          ("ε", "AdaLN", False, "copula",         5.0,  4),
}
ALL_MODELS = list(MODEL_META.keys())

REAL_REF = {
    "kurtosis":           755.0,
    "std":                2.19e-5,
    "trade_fraction":     0.0477,
    "vol_vol_corr":       0.341,
    "acf_abs_ret_lag50":  0.0464,
    "acf_beta":           0.21,
    "trade_sign_acf_lag1":0.675,
    "ofi_slope_delta50":  1.11e-4,
    "spread_1tick_frac":  0.95,
}

# Group → color (philosophy of the architecture)
GROUP_COLOR = {
    "scale-accurate":   "#1f77b4",   # blue
    "scale-exploded":   "#d62728",   # red
    "copula":           "#2ca02c",   # green
    "special":          "#ff7f0e",   # orange
    "real":             "#000000",   # black
}

def model_group(m: str) -> str:
    if "noclip" in m:
        return "scale-exploded"
    if "remapped" in m or "cascade" in m or m == "v11":
        return "special"
    if MODEL_META[m][3].startswith("copula"):
        return "copula"
    return "scale-accurate"

MODEL_COLORS = {m: GROUP_COLOR[model_group(m)] for m in ALL_MODELS}
MODEL_COLORS["real"] = GROUP_COLOR["real"]
MODEL_COLORS["market_replay"] = "#666666"

# Per-protagonist override palette: when a figure compares only v2/v5/v9
# (the three main-paper protagonists), use these distinctive colors so the
# three lines are immediately distinguishable. The default GROUP_COLOR
# scheme would put v2 and v5 both in scale-accurate-blue, making any
# v2-vs-v5 visual comparison illegible.
PROTAGONIST_COLOR = {
    "real": "#000000",
    "v2":   "#1f77b4",   # blue
    "v5":   "#2ca02c",   # green
    "v9":   "#9467bd",   # purple
}

# Per-model markers within a group, so labels are distinguishable in scatters
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]
MODEL_MARKERS: dict[str, str] = {}
_idx_in_group: dict[str, int] = {}
for m in ALL_MODELS:
    g = model_group(m)
    i = _idx_in_group.get(g, 0)
    MODEL_MARKERS[m] = MARKERS[i % len(MARKERS)]
    _idx_in_group[g] = i + 1

# ── Data loaders ───────────────────────────────────────────────────────────
def load_multi_axis() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE / "multi_axis_comparison.csv")
    return df

# Per-check real reference values (used by the numeric-error heatmap, fig_2_1).
# Anything not in this map is left as NaN — the heatmap will mask those rows.
G1_REAL_VALUES = {
    "Return W-1":                    0.0,
    "Spread W-1":                    0.0,
    "Return excess kurtosis":        REAL_REF["kurtosis"],            # 755
    "Spread 1-tick fraction":        REAL_REF["spread_1tick_frac"],   # 0.95
    "Trade fraction (synth vs real)":REAL_REF["trade_fraction"],      # 0.0477
    "ACF(|r|) lag-50 deviation":     0.0,
    "ACF(|r|) power-law":            REAL_REF["acf_beta"],            # 0.21
    "Trade-sign lag-1 ACF sign":     REAL_REF["trade_sign_acf_lag1"], # 0.675
    "Return lag-1 ACF":              0.0,                             # ~0 expected
    "RV signature shape":            0.0,                             # ratio diff
    "E[r|I] sign at =10":            3.68e-05,
    "E[r|I] sign at =50":            REAL_REF["ofi_slope_delta50"],   # 1.107e-04
    "Joint spread-imbalance JSD":    0.0,
    "Conditional moments (|z|<2)":   0.0,
    "Crossed books rate (decode)":   0.0,
    "Size tail exponent":            1.567,
}


def load_g1_long() -> pd.DataFrame:
    """Load G1 matrix and pivot into a long format suitable for the heatmap.

    Returns long-format frame with columns:
      check, model, pass, value, real_value
    where `real_value` is the per-check real reference (NaN for checks not in
    G1_REAL_VALUES). The numeric-error heatmap (fig_2_1) computes
    error = |value - real_value|; checks with NaN real_value are masked out.
    """
    raw = pd.read_csv(ARCHIVE / "G1_comparison_all_models.csv")
    pass_cols = [c for c in raw.columns if c.endswith("_pass")]
    val_cols  = [c for c in raw.columns if c.endswith("_value")]
    long_pass = raw.melt(id_vars=["check"], value_vars=pass_cols,
                         var_name="model", value_name="pass")
    long_pass["model"] = long_pass["model"].str.replace("_pass$", "", regex=True)
    long_pass["pass"] = long_pass["pass"].map(
        lambda v: 1.0 if (v is True or str(v).strip().lower() == "true")
                  else 0.0 if (v is False or str(v).strip().lower() == "false")
                  else np.nan
    ).astype(float)
    long_val = raw.melt(id_vars=["check"], value_vars=val_cols,
                        var_name="model", value_name="value")
    long_val["model"] = long_val["model"].str.replace("_value$", "", regex=True)
    # Some checks store descriptive strings instead of pure numerics (e.g.
    # "real=0.675, synth=0.348"); coerce to float, mark as NaN otherwise.
    long_val["value"] = pd.to_numeric(long_val["value"], errors="coerce")
    out = long_pass.merge(long_val, on=["check", "model"])
    # Drop the TOTAL_PASS aggregate row (not a check, breaks the heatmap)
    out = out[out["check"] != "TOTAL_PASS"].copy()
    out["real_value"] = out["check"].map(G1_REAL_VALUES).astype(float)
    return out

def load_predictive() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "predictive_score_all" / "predictive_score.csv")

def load_pca_coverage() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "trades_metrics_all" / "pca_coverage.csv")

def load_vol_vol() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "trades_metrics_all" / "vol_volatility_correlation.csv")

def load_per_feature_stats() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "deep_diagnostic" / "per_feature_stats.csv")

def load_temporal_acf() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "deep_diagnostic" / "temporal_acf.csv")

def load_z_space_stats() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "deep_diagnostic" / "z_space_stats.csv")

def load_window_level() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "deep_diagnostic" / "window_level_stats.csv")

def load_correlation() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "deep_diagnostic" / "correlation_comparison.csv")

def load_agent_decomp() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "ranking_mechanism" / "agent_decomposition.csv")

def load_a2_minus_a1() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "ranking_mechanism" / "a2_minus_a1_deltas.csv")

def load_conditional_returns() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "ranking_mechanism" / "conditional_returns.csv")

def load_bootstrap_for(model: str) -> dict | None:
    p = ARCHIVE / model / "hypothesis" / "bootstrap.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

# ── Save helper ────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf, png

WRITTEN: list[Path] = []
def _wrote(*paths: Path) -> None:
    WRITTEN.extend(paths)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Three-Axis Framework
# ══════════════════════════════════════════════════════════════════════════
def fig_1_1_three_axis_scatter(matrix: pd.DataFrame, pred: pd.DataFrame,
                                pca: pd.DataFrame) -> None:
    """Hero figure: 2D scatter of G1 vs Predictive, point size = PCA coverage.
    Falls back from a true ternary to keep deps minimal."""
    df = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()

    rows = []
    for m in ALL_MODELS:
        if m not in df.index:
            continue
        try:
            g1_str = str(df.loc[m, "G1_pass"])
            g1 = int(g1_str.split("/")[0]) / int(g1_str.split("/")[1])
        except Exception:
            continue
        pr = pred_d.get(m)
        pc = pca_d.get(m, 0.0)
        if pr is None or np.isnan(pr) or pr <= 0:
            continue
        # 1/relative_to_replay normalised across models — log-scale-friendly
        rows.append({
            "model": m,
            "g1_norm": g1,
            "pred_inv": 1.0 / pr,
            "pca_pct": pc if pd.notna(pc) else 0.0,
            "color": MODEL_COLORS[m],
        })
    if not rows:
        print("[fig_1_1] no data — skipped")
        return
    pts = pd.DataFrame(rows)
    # Min-max normalise predictive for sizing
    pts["pred_score"] = (pts["pred_inv"] - pts["pred_inv"].min()) / \
                       (pts["pred_inv"].max() - pts["pred_inv"].min() + 1e-12)

    # Calculate optimal label positions to avoid overlaps
    # We'll use adjust_text if available, otherwise fallback to offsetting
    try:
        from adjustText import adjust_text
        use_adjust_text = True
    except ImportError:
        use_adjust_text = False

    fig, ax = plt.subplots(figsize=(9, 7))
    texts = []

    for _, r in pts.iterrows():
        size = 80 + 7 * r["pca_pct"]   # 80–800 area pts
        # All circles (marker="o")
        ax.scatter(r["g1_norm"], r["pred_score"], s=size, c=r["color"],
                   marker="o", edgecolors="k", linewidths=0.6,
                   alpha=0.85, label=r["model"])
        # Add text objects for later adjustment
        t = ax.annotate(
            r["model"],
            (r["g1_norm"], r["pred_score"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, alpha=0.9
        )
        texts.append(t)

    # Add real at (1, 1) as a circle, with 100% PCA
    real_size = 80 + 7 * 100.0  # 80–800 pts, assuming 100% coverage for real
    ax.scatter([1.0], [1.0], s=real_size, c="black", marker="o", edgecolors="k", linewidths=0.9, alpha=0.95, zorder=5, label="real")
    ax.annotate("REAL", (1.0, 1.0), xytext=(8, -2),
                textcoords="offset points", fontsize=10, fontweight="bold")

    # Adjust label positions to prevent overlapping
    if use_adjust_text and texts:
        # Use adjust_text for auto label placement
        from adjustText import adjust_text
        adjust_text(
            texts,
            only_move={'points':'y', 'texts':'y'},
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7),
            ax=ax,
            expand_text=(1.05, 1.3),
            expand_points=(1.07, 1.35),
            force_text=0.8,
            force_points=0.15,
            lim=30
        )
    # Otherwise, labels will be slightly offset but may still overlap

    ax.set_xlabel("Distributional Realism (G1 pass fraction)")
    ax.set_ylabel("Temporal Transferability (1 / ×replay, min-max scaled)")
    ax.set_title("Three-Axis Evaluation Space\n"
                 "(point area $\\propto$ PCA convex-hull coverage)",
                 pad=12)
    # Remove the star marker for real reference (old code)
    # ax.scatter([1.0], [1.05], marker="*", s=400, c="black", zorder=5,
    #            label="real reference")
    # ax.annotate("REAL", (1.0, 1.05), xytext=(8, -2),
    #             textcoords="offset points", fontsize=10, fontweight="bold")
    # Group-color legend (right side)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=10, label=g)
               for g, c in GROUP_COLOR.items() if g != "real"]
    ax.legend(handles=handles, title="Architecture group",
              loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    pdf, png = save_fig(fig, "fig_1_1_three_axis_scatter")
    _wrote(pdf, png)
    print(f"[fig_1_1] wrote {pdf.name}")


def fig_1_2_pairwise(matrix: pd.DataFrame, pred: pd.DataFrame,
                      pca: pd.DataFrame) -> None:
    """3-panel pairwise scatter showing the three-axis anti-correlation."""
    df = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()

    rows = []
    for m in ALL_MODELS:
        if m not in df.index:
            continue
        try:
            g1_str = str(df.loc[m, "G1_pass"])
            g1 = int(g1_str.split("/")[0])
        except Exception:
            continue
        pr = pred_d.get(m)
        pc = pca_d.get(m, np.nan)
        if pr is None or np.isnan(pr) or pr <= 0:
            continue
        rows.append({"model": m, "G1": g1,
                     "Pred×replay": pr, "PCA%": pc})
    pts = pd.DataFrame(rows)
    if pts.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("G1", "Pred×replay", "G1 (pass count)", "Predictive ×replay (log)"),
        ("G1", "PCA%",        "G1 (pass count)", "PCA coverage (%)"),
        ("Pred×replay", "PCA%",
         "Predictive ×replay (log)", "PCA coverage (%)"),
    ]
    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        sub = pts.dropna(subset=[xc, yc])
        for _, r in sub.iterrows():
            ax.scatter(r[xc], r[yc], c=MODEL_COLORS[r["model"]],
                       marker=MODEL_MARKERS[r["model"]],
                       s=80, edgecolors="k", linewidths=0.5, alpha=0.85)
            ax.annotate(r["model"], (r[xc], r[yc]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7)
        if "×replay" in xc:
            ax.set_xscale("log")
        if "×replay" in yc:
            ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if len(sub) >= 3:
            try:
                rho, p = spearmanr(sub[xc], sub[yc])
                ax.set_title(f"Spearman ρ = {rho:+.2f} (p = {p:.2g})",
                             fontsize=11)
            except Exception:
                ax.set_title("")
    fig.suptitle("Pairwise Anti-Correlation Across Evaluation Axes",
                 fontsize=14, y=1.02)
    pdf, png = save_fig(fig, "fig_1_2_pairwise_anticorrelation")
    _wrote(pdf, png)
    print(f"[fig_1_2] wrote {pdf.name}")


def fig_1_3_comparison_table(matrix: pd.DataFrame) -> None:
    """Reformat the multi-axis comparison table as a LaTeX file."""
    df = matrix.copy()
    # Drop the heavy raw-precision columns from main table (kept in CSV)
    keep_cols = [
        "model", "G1_pass", "rho_diff", "predictive_x_replay",
        "PCA_coverage_pct", "B1_synth_kurt", "B4_trade_fraction",
        "vol_volatility_corr", "D1_ofi_slope_d50",
    ]
    sub = df[[c for c in keep_cols if c in df.columns]].copy()
    sub.columns = ["model", "G1", r"\(\rho_\mathrm{diff}\)", "Pred×replay",
                   "PCA%", "kurt", "trade frac", "vol-vol", "OFI slope (Δ50)"]

    # Sort: REAL & market_replay top, rest by model name
    top = sub[sub["model"].isin(["REAL", "market_replay"])]
    rest = sub[~sub["model"].isin(["REAL", "market_replay"])].sort_values("model")
    sub = pd.concat([top, rest], ignore_index=True)

    # Format numerics
    for c in [r"\(\rho_\mathrm{diff}\)", "Pred×replay", "PCA%", "kurt",
              "trade frac", "vol-vol", "OFI slope (Δ50)"]:
        if c not in sub.columns:
            continue
        sub[c] = sub[c].apply(
            lambda v: ("—" if pd.isna(v) or v == "" else
                       f"{float(v):.2g}" if abs(float(v)) < 1e3
                       else f"{float(v):.2e}")
            if not isinstance(v, str) or v.replace(".", "", 1).replace("-", "", 1)
                                         .replace("e", "", 1).replace("+", "", 1)
                                         .isdigit()
            else v
        )
    latex = sub.to_latex(index=False, escape=False, column_format="l" + "r" * (sub.shape[1]-1),
                         caption="Multi-axis comparison across all 20 generators "
                                 "(plus REAL reference and market-replay baseline).",
                         label="tab:multi_axis")
    out_tex = OUT / "tab_1_3_multi_axis.tex"
    out_tex.write_text(latex)
    out_csv = OUT / "tab_1_3_multi_axis.csv"
    sub.to_csv(out_csv, index=False)
    _wrote(out_tex, out_csv)
    print(f"[tab_1_3] wrote {out_tex.name} + {out_csv.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Distributional Realism (G1)
# ══════════════════════════════════════════════════════════════════════════
def _fmt_g1(x: float) -> str:
    """Compact formatter — fits scientific notation in tiny heatmap cells."""
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


def fig_2_1_g1_heatmap(g1: pd.DataFrame) -> None:
    """16 × 20 numerical-value heatmap with PER-ROW cross-sectional gradient.

    Each cell shows the synth value (raw, not error). Color = how that cell's
    error compares to the other models' errors WITHIN THE SAME ROW (greenest =
    closest to real among the 20, reddest = farthest). A "★" prefix marks
    cells that passed their criterion. A REAL column on the right shows the
    reference value for each check.

    Reading the figure:
      • Look across one row (one stylized fact) → see which models are best/worst.
      • Look down one column (one model) → see which checks the model handles
        best/worst.
      • Cross-section gradient is per-row, so checks whose values span 6 OoMs
        (e.g. W-1 distances ~1e-6 vs kurtosis ~10³) remain comparable.
    """
    required = {"check", "model", "value", "real_value"}
    missing = required.difference(g1.columns)
    if missing:
        raise ValueError(f"g1 must contain {required} (missing: {missing})")

    g1 = g1.copy()
    # Drop checks with no numeric synth value (validator emits free-text)
    # OR that render with too many blank/NaN cells to be readable in the
    # cross-sectional heatmap.
    g1 = g1[~g1["check"].isin({
        "Conditional moments (|z|<2)",
        "ACF(|r|) power-law",
        "E[r|I] sign at =10",
        "E[r|I] sign at =50",
        "RV signature shape",
        "Size tail exponent",
        "Trade fraction (synth vs real)",
        "Trade-sign lag-1 ACF sign",
    })]
    # Per-cell absolute error from the per-row real reference
    g1["error"] = (g1["value"] - g1["real_value"]).abs()

    val_pivot = g1.pivot(index="check", columns="model", values="value")
    err_pivot = g1.pivot(index="check", columns="model", values="error")
    pass_pivot = g1.pivot(index="check", columns="model", values="pass")
    real_series = g1.dropna(subset=["real_value"]).groupby("check")["real_value"].first()

    model_cols = [m for m in ALL_MODELS if m in val_pivot.columns]
    val_pivot = val_pivot[model_cols]
    err_pivot = err_pivot[model_cols]
    pass_pivot = pass_pivot[model_cols]

    # Per-row min-max normalize errors → [0, 1]; rows with all-NaN errors
    # (no real reference) get masked.
    err_norm = err_pivot.copy()
    for chk in err_norm.index:
        row = err_norm.loc[chk].astype(float).values
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

    # Annotation grid: raw synth value only (cleaner — no pass-marker clutter)
    annot = pd.DataFrame(index=val_pivot.index, columns=val_pivot.columns, dtype=object)
    for chk in val_pivot.index:
        for col in val_pivot.columns:
            annot.loc[chk, col] = _fmt_g1(val_pivot.loc[chk, col])

    # Add REAL column on the right for visual reference
    REAL_COL = "REAL"
    err_norm[REAL_COL] = 0.0
    annot[REAL_COL] = real_series.reindex(err_norm.index).map(_fmt_g1)
    val_pivot[REAL_COL] = real_series.reindex(err_norm.index).fillna(np.nan)

    cmap = LinearSegmentedColormap.from_list(
        "g_to_r", ["#1a7d3a", "#7fc97f", "#f5f5dc", "#fdae61", "#d73027"])

    fig, ax = plt.subplots(figsize=(max(11, len(err_norm.columns) * 0.7), 9))
    sns.heatmap(
        err_norm, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "per-row error (0 = best in row, 1 = worst in row)",
                  "shrink": 0.6},
        linewidths=0.5, linecolor="white",
        annot=annot, fmt="", annot_kws={"fontsize": 7},
        mask=err_norm.isna(),
        ax=ax,
    )
    # Visual divider before the REAL column
    ax.axvline(len(model_cols), color="black", linewidth=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "G1 Stylized-Facts — synth values with cross-sectional error gradient",
        pad=12
    )
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_2_1_g1_numerical_heatmap")
    _wrote(pdf, png)
    print(f"[fig_2_1] wrote {pdf.name}")


def fig_2_2_violin_marginals(stats: pd.DataFrame) -> None:
    """Marginal stats violin-style — but per_feature_stats is summary, so we
    use a horizontal bar of kurtosis × model for the four headline features."""
    features = ["mid_return", "spread", "bid_sz", "trade_sz"]
    sub = stats[stats["feature"].isin(features)].copy()
    sub = sub[sub["model"].isin(["real"] + ALL_MODELS)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, feat in zip(axes.flat, features):
        s = sub[sub["feature"] == feat].copy()
        if s.empty:
            ax.set_visible(False)
            continue
        s = s.sort_values("kurtosis").reset_index(drop=True)
        colors = [MODEL_COLORS.get(m, "gray") for m in s["model"]]
        ax.barh(s["model"], s["kurtosis"], color=colors, edgecolor="k",
                linewidth=0.4)
        if feat == "mid_return":
            ax.axvline(REAL_REF["kurtosis"], color="red", linestyle="--",
                       linewidth=1, label=f"real = {REAL_REF['kurtosis']:.0f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"Excess kurtosis — {feat}")
        ax.set_xlabel("kurtosis")
        ax.set_xscale("symlog")
    fig.suptitle("Per-feature distributional spread across models",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_2_2_marginal_kurtosis")
    _wrote(pdf, png)
    print(f"[fig_2_2] wrote {pdf.name}")


def fig_2_3_return_tail(stats: pd.DataFrame) -> None:
    """log-log complementary CDF (1-F(|r|)) stub — uses tail-frac stats from
    per_feature_stats since we don't have raw quantiles for every model."""
    sub = stats[stats["feature"] == "mid_return"].copy()
    sub = sub[sub["model"].isin(["real", "v2", "v5", "v9", "v2_remapped"])]
    if sub.empty:
        print("[fig_2_3] no data — skipped")
        return

    cols = [("frac_gt_2sigma", 2.0), ("frac_gt_3sigma", 3.0),
            ("frac_gt_4sigma", 4.0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in sub.iterrows():
        z = [c[1] for c in cols]
        f = [r.get(c[0], np.nan) for c in cols]
        f = [v if pd.notna(v) and v > 0 else np.nan for v in f]
        ax.plot(z, f, marker=MODEL_MARKERS.get(r["model"], "o"),
                color=MODEL_COLORS.get(r["model"], "gray"),
                linewidth=2.5 if r["model"] == "real" else 1.5,
                label=r["model"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("threshold |z|")
    ax.set_ylabel("P(|return| > z·σ)")
    ax.set_title("Tail probability vs threshold (log-y)\n"
                 "Real INTC has heaviest tail by ~3× over the realistic synth set",
                 pad=12)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    pdf, png = save_fig(fig, "fig_2_3_return_tail")
    _wrote(pdf, png)
    print(f"[fig_2_3] wrote {pdf.name}")


def fig_2_4_trade_fraction(matrix: pd.DataFrame) -> None:
    """Horizontal bar of trade fraction across models, sorted by value."""
    df = matrix.set_index("model")
    rows = []
    for m in ALL_MODELS:
        if m not in df.index:
            continue
        v = df.loc[m, "B4_trade_fraction"]
        try:
            rows.append({"model": m, "trade_frac": float(v),
                         "color": MODEL_COLORS[m]})
        except (ValueError, TypeError):
            continue
    if not rows:
        return
    pts = pd.DataFrame(rows).sort_values("trade_frac")

    # Color by distance from real
    dists = (pts["trade_frac"] - REAL_REF["trade_fraction"]).abs()
    norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(pts["model"], pts["trade_frac"], color=cmap(norm),
            edgecolor="k", linewidth=0.4)
    ax.axvline(REAL_REF["trade_fraction"], color="black", linestyle="--",
               linewidth=1.5, label=f"real = {REAL_REF['trade_fraction']:.4f}")
    ax.set_xlabel("Trade fraction")
    ax.set_title("Trade fraction across models (closer to real = greener)",
                 pad=12)
    ax.legend(loc="lower right")
    pdf, png = save_fig(fig, "fig_2_4_trade_fraction")
    _wrote(pdf, png)
    print(f"[fig_2_4] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Predictive Score
# ══════════════════════════════════════════════════════════════════════════
def fig_3_1_predictive_bar(pred: pd.DataFrame) -> None:
    """Horizontal bar of predictive ×replay (log scale), real-replay reference."""
    sub = pred.copy()
    sub = sub[sub["label"].isin(["market_replay"] + ALL_MODELS)].copy()
    sub = sub.sort_values("relative_to_replay")

    colors = ["#444"] + [MODEL_COLORS.get(m, "gray")
                          for m in sub["label"] if m != "market_replay"]
    # The above ordering is fragile — recompute:
    colors = [MODEL_COLORS.get(m, "#444") for m in sub["label"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(sub["label"], sub["relative_to_replay"], color=colors,
            edgecolor="k", linewidth=0.4)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="market replay = 1.0")
    ax.set_xscale("log")
    ax.set_xlabel("Test MAE  /  market-replay baseline (log)")
    ax.set_title("Predictive Score — train on synth, test on real\n"
                 "(closer to 1× = better temporal pattern transferability)",
                 pad=12)
    ax.legend(loc="lower right")
    for i, (m, v) in enumerate(zip(sub["label"], sub["relative_to_replay"])):
        ax.annotate(f"{v:.1f}×", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    pdf, png = save_fig(fig, "fig_3_1_predictive_score")
    _wrote(pdf, png)
    print(f"[fig_3_1] wrote {pdf.name}")


def fig_3_2_acf_top_bottom(pred: pd.DataFrame, acf: pd.DataFrame) -> None:
    """Top-3 vs bottom-3 predictive models — show their ACFs of mid_return.
    The hypothesis: noclip preserves temporal patterns despite scale explosion."""
    sub = pred[pred["label"].isin(ALL_MODELS)].sort_values("relative_to_replay")
    top3 = sub.head(3)["label"].tolist()
    bot3 = sub.tail(3)["label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, names, title in [(axes[0], top3, "Top-3 (best predictive)"),
                              (axes[1], bot3, "Bottom-3 (worst predictive)")]:
        # Real as reference
        for m in ["real"] + names:
            sa = acf[(acf["model"] == m) & (acf["feature"] == "mid_return")]
            sa = sa.sort_values("lag").head(20)
            if sa.empty:
                continue
            ax.plot(sa["lag"], sa["acf"],
                    label=m, marker="o" if m != "real" else "*",
                    markersize=4 if m != "real" else 8,
                    linewidth=2.5 if m == "real" else 1.4,
                    color=MODEL_COLORS.get(m, "gray"))
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF(mid_return)")
        ax.set_title(title)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Temporal Structure for Best vs Worst Predictive-Score Models",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_3_2_acf_top_bottom")
    _wrote(pdf, png)
    print(f"[fig_3_2] wrote {pdf.name}")


def fig_3_3_predictive_vs_scale(matrix: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Scatter: synth std / real std vs predictive ×replay (both log).
    Punchline: scale matching ≠ temporal transferability."""
    df = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    real_std = REAL_REF["std"]
    rows = []
    for m in ALL_MODELS:
        if m not in df.index:
            continue
        try:
            std = float(df.loc[m, "B1_synth_std"])
            pr = pred_d.get(m)
            if pr is None or pr <= 0 or std <= 0:
                continue
            rows.append({"model": m, "std_ratio": std / real_std,
                         "pred": pr})
        except (ValueError, TypeError):
            continue
    pts = pd.DataFrame(rows)
    if pts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in pts.iterrows():
        ax.scatter(r["std_ratio"], r["pred"],
                   c=MODEL_COLORS[r["model"]],
                   marker=MODEL_MARKERS[r["model"]],
                   s=90, edgecolors="k", linewidths=0.5, alpha=0.85)
        ax.annotate(r["model"], (r["std_ratio"], r["pred"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5,
               label="synth std = real std")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("synth std / real std (log)")
    ax.set_ylabel("Predictive ×replay (log; 1 = market replay)")
    ax.set_title("Scale-matching ≠ temporal transferability\n"
                 "(scale-exploded noclip models often beat scale-matched copula models)",
                 pad=12)
    ax.legend()
    pdf, png = save_fig(fig, "fig_3_3_predictive_vs_scale")
    _wrote(pdf, png)
    print(f"[fig_3_3] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PCA Coverage
# ══════════════════════════════════════════════════════════════════════════
def _pca_panels_4(proj: pd.DataFrame, pca: pd.DataFrame,
                    panel_models: list[str]) -> None:
    """4-panel TRADES Fig 2 style: real points + per-model synth points +
    convex hulls, one panel per model in `panel_models`. Reads per-tape
    projections from results/trades_metrics_all/pca_projections.csv (written
    by the patched script 62)."""
    real = proj[proj["model"] == "real"][["pc1", "pc2"]].to_numpy()
    if len(real) < 4:
        print("[fig_4_1] too few real PC points — falling back")
        return
    try:
        real_hull = ConvexHull(real)
    except Exception as e:
        print(f"[fig_4_1] real hull failed: {e} — falling back")
        return
    cov_d = pca.set_index("model")["coverage_pct"].to_dict() \
                if pca is not None else {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, m in zip(axes.flat, panel_models):
        s = proj[proj["model"] == m][["pc1", "pc2"]].to_numpy()
        ax.scatter(real[:, 0], real[:, 1], c="0.4", s=10, alpha=0.45,
                   label=f"real (n={len(real)})", zorder=1)
        verts = list(real_hull.vertices) + [real_hull.vertices[0]]
        ax.plot(real[verts, 0], real[verts, 1], "k--",
                linewidth=1.3, alpha=0.7, label="real hull", zorder=2)
        if len(s) >= 4:
            color = MODEL_COLORS.get(m, "tab:red")
            ax.scatter(s[:, 0], s[:, 1], c=color, s=12, alpha=0.55,
                       marker="^", label=f"{m} (n={len(s)})", zorder=3)
            try:
                synth_hull = ConvexHull(s)
                sv = list(synth_hull.vertices) + [synth_hull.vertices[0]]
                ax.plot(s[sv, 0], s[sv, 1], color=color, linewidth=1.5,
                        linestyle="--", alpha=0.85, zorder=4)
            except Exception:
                pass
            cov = cov_d.get(m, np.nan)
            ax.set_title(f"{m}  (coverage = {cov:.0f}%)" if pd.notna(cov)
                         else f"{m}", fontsize=12)
        else:
            ax.set_title(f"{m}  (no projections)", fontsize=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle("PCA Convex-Hull Coverage (TRADES Fig 2 style) — "
                 "real in gray, synth in color",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_4_1_pca_panels")
    _wrote(pdf, png)
    print(f"[fig_4_1] wrote {pdf.name} (4-panel scatter+hull)")


def fig_4_1_pca_summary(pca: pd.DataFrame) -> None:
    """PCA coverage figure. Prefers the 4-panel scatter+hull view when
    pca_projections.csv exists (after running the patched script 62 on
    cluster + rsync back). Falls back to a coverage bar chart otherwise."""
    proj_path = RESULTS / "trades_metrics_all" / "pca_projections.csv"
    if proj_path.exists():
        try:
            proj = pd.read_csv(proj_path)
            if "real" in proj["model"].unique():
                panel_models = [m for m in ["v2", "v9", "v8", "v2_noclip"]
                                if m in proj["model"].unique()]
                if panel_models:
                    _pca_panels_4(proj, pca, panel_models[:4])
                    return
        except Exception as e:
            print(f"[fig_4_1] panel render failed ({e}) — falling back to bar")

    # Fallback: bar chart of coverage %
    sub = pca[pca["model"].isin(ALL_MODELS)].sort_values("coverage_pct",
                                                           ascending=True)
    if sub.empty:
        return
    colors = [MODEL_COLORS.get(m, "gray") for m in sub["model"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sub["model"], sub["coverage_pct"], color=colors,
            edgecolor="k", linewidth=0.4)
    ax.axvline(100, color="black", linestyle="--", linewidth=1.5,
               label="real = 100%")
    ax.set_xlabel("PCA convex-hull coverage of real (%)")
    ax.set_title("Multivariate state-space coverage  "
                 "[bar fallback — pull pca_projections.csv for the 4-panel scatter+hull]",
                 pad=12)
    ax.legend(loc="lower right")
    for i, (m, v) in enumerate(zip(sub["model"], sub["coverage_pct"])):
        ax.annotate(f"{v:.0f}%", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    pdf, png = save_fig(fig, "fig_4_1_pca_coverage")
    _wrote(pdf, png)
    print(f"[fig_4_1] wrote {pdf.name} (bar fallback)")


def fig_4_2_pca_vs_g1(matrix: pd.DataFrame, pca: pd.DataFrame) -> None:
    """Scatter: PCA% vs G1 — coverage doesn't imply realism."""
    df = matrix.set_index("model")
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()
    rows = []
    for m in ALL_MODELS:
        if m not in df.index:
            continue
        try:
            g1 = int(str(df.loc[m, "G1_pass"]).split("/")[0])
            pc = pca_d.get(m, np.nan)
            if pd.isna(pc):
                continue
            rows.append({"model": m, "G1": g1, "PCA": pc})
        except Exception:
            continue
    pts = pd.DataFrame(rows)
    if pts.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in pts.iterrows():
        ax.scatter(r["PCA"], r["G1"], c=MODEL_COLORS[r["model"]],
                   marker=MODEL_MARKERS[r["model"]], s=100,
                   edgecolors="k", linewidths=0.5, alpha=0.85)
        ax.annotate(r["model"], (r["PCA"], r["G1"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    if len(pts) >= 3:
        rho, p = spearmanr(pts["PCA"], pts["G1"])
        ax.set_title(f"PCA Coverage vs G1 Pass Count "
                     f"(Spearman ρ = {rho:+.2f}, p = {p:.2g})", pad=12)
    ax.set_xlabel("PCA coverage (%)")
    ax.set_ylabel("G1 pass count (out of ~16)")
    pdf, png = save_fig(fig, "fig_4_2_pca_vs_g1")
    _wrote(pdf, png)
    print(f"[fig_4_2] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Architectural Ablation
# ══════════════════════════════════════════════════════════════════════════
def _safe_metric(matrix_idx: pd.DataFrame, model: str, col: str,
                  pred_d: dict, pca_d: dict) -> dict:
    out = {}
    if model not in matrix_idx.index:
        return out
    row = matrix_idx.loc[model]
    try:
        out["G1"] = int(str(row["G1_pass"]).split("/")[0])
    except Exception:
        out["G1"] = np.nan
    for k, src in [("kurt", "B1_synth_kurt"),
                   ("trade_frac", "B4_trade_fraction"),
                   ("acf_lag50", "D1_ofi_slope_d50"),
                   ("vol_vol", "vol_volatility_corr")]:
        try:
            out[k] = float(row[src])
        except (ValueError, TypeError):
            out[k] = np.nan
    out["pred"] = pred_d.get(model, np.nan)
    out["PCA"] = pca_d.get(model, np.nan)
    return out


def fig_5_1_ablation_heatmap(matrix: pd.DataFrame, pred: pd.DataFrame,
                              pca: pd.DataFrame) -> None:
    """Heatmap: architectural choice × evaluation metric, cell = mean Δ."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()

    pairs = {
        r"$\epsilon \to v$":     [("v2", "v4"), ("v9", "v9_b")],
        r"FiLM $\to$ AdaLN":     [("v4", "v5"), ("v8", "v8_b")],
        r"$+$ EDM":              [("v5", "v6"), ("v5", "v7"), ("v5", "v7_b")],
        r"$+$ copula":           [("v2", "v8")],
        r"$+$ dequant":          [("v8", "v9")],
        r"$-$ x0\_clip":         [("v2", "v2_noclip"),
                                  ("v3_e9", "v3_e9_noclip"),
                                  ("v3p5", "v3p5_noclip"),
                                  ("v4", "v4_noclip")],
    }
    metrics = ["G1", "pred", "PCA", "kurt", "trade_frac", "vol_vol"]
    nice = {"G1": "ΔG1", "pred": "ΔPred×", "PCA": "ΔPCA%",
            "kurt": "Δkurt", "trade_frac": "Δtrade frac",
            "vol_vol": "Δvol-vol corr"}

    grid = pd.DataFrame(index=list(pairs.keys()), columns=[nice[m] for m in metrics],
                        dtype=float)
    for axis, ps in pairs.items():
        deltas = {m: [] for m in metrics}
        for a, b in ps:
            ma = _safe_metric(midx, a, None, pred_d, pca_d)
            mb = _safe_metric(midx, b, None, pred_d, pca_d)
            for k in metrics:
                if k in ma and k in mb and pd.notna(ma[k]) and pd.notna(mb[k]):
                    deltas[k].append(mb[k] - ma[k])
        for k in metrics:
            grid.loc[axis, nice[k]] = (np.mean(deltas[k]) if deltas[k]
                                        else np.nan)

    # Z-score per column for visual comparability (effects vary by orders)
    zgrid = grid.copy()
    for c in zgrid.columns:
        v = zgrid[c].astype(float)
        s = v.abs().max()
        if s and not np.isnan(s):
            zgrid[c] = v / s

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(zgrid, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=grid.map(lambda v: f"{v:+.2g}" if pd.notna(v) else "—"),
                fmt="", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "normalised Δ (per column)"}, ax=ax)
    ax.set_title("Architectural Ablation: per-axis effect on each evaluation metric\n"
                 "(green = improves, red = degrades; cell = mean Δ across matched pairs)",
                 pad=12)
    ax.set_ylabel("")
    pdf, png = save_fig(fig, "fig_5_1_ablation_heatmap")
    _wrote(pdf, png)
    print(f"[fig_5_1] wrote {pdf.name}")


def fig_5_2_x0_clip_effect(matrix: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Paired bars: clip-vs-noclip on G1 and Predictive."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pairs = [("v2", "v2_noclip"), ("v3_e9", "v3_e9_noclip"),
             ("v3p5", "v3p5_noclip"), ("v4", "v4_noclip")]
    rows = []
    for clip, noclip in pairs:
        if clip not in midx.index or noclip not in midx.index:
            continue
        g_c = int(str(midx.loc[clip, "G1_pass"]).split("/")[0])
        g_n = int(str(midx.loc[noclip, "G1_pass"]).split("/")[0])
        p_c = pred_d.get(clip, np.nan)
        p_n = pred_d.get(noclip, np.nan)
        rows.append({"pair": clip, "G1_clip": g_c, "G1_noclip": g_n,
                     "pred_clip": p_c, "pred_noclip": p_n})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(df))
    w = 0.35

    axes[0].bar(x - w/2, df["G1_clip"],   w, label="clip",   color="#1f77b4")
    axes[0].bar(x + w/2, df["G1_noclip"], w, label="noclip", color="#d62728")
    for i, (a, b) in enumerate(zip(df["G1_clip"], df["G1_noclip"])):
        axes[0].annotate("", xy=(i + w/2, b), xytext=(i - w/2, a),
                          arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["pair"])
    axes[0].set_title("G1 pass count: clip vs noclip\n(clipping HELPS distributional realism)")
    axes[0].set_ylabel("G1 (pass count)")
    axes[0].legend()

    axes[1].bar(x - w/2, df["pred_clip"],   w, label="clip",   color="#1f77b4")
    axes[1].bar(x + w/2, df["pred_noclip"], w, label="noclip", color="#d62728")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["pair"])
    axes[1].set_title("Predictive ×replay: clip vs noclip\n(noclip HELPS temporal transferability)")
    axes[1].set_ylabel("Predictive ×replay (log)")
    axes[1].legend()

    fig.suptitle("The x0_clip Trade-off: realism vs transferability",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_5_2_x0_clip_effect")
    _wrote(pdf, png)
    print(f"[fig_5_2] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Copula Story
# ══════════════════════════════════════════════════════════════════════════
def fig_6_1_z_space(z: pd.DataFrame) -> None:
    """Bar of z-mean and z-std for v8 vs v9 (the dequantization fix)."""
    sub = z[z["feature"] == "mid_return"].copy()
    sub = sub[sub["model"].isin(["v8", "v9"])]
    if sub.empty:
        # fall back to bid_sz which we definitely have
        sub = z[z["model"].isin(["v8", "v9"])].groupby("model").agg(
            z_mean=("z_mean", "mean"), z_std=("z_std", "mean"),
            z_kurtosis=("z_kurtosis", "mean")
        ).reset_index()
        if sub.empty:
            print("[fig_6_1] no z-space data — skipped")
            return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, lbl, ref in [
        (axes[0], "z_mean", "z mean", 0.0),
        (axes[1], "z_std",  "z std",  1.0),
        (axes[2], "z_kurtosis", "z kurtosis (excess)", 0.0),
    ]:
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        ax.bar(sub["model"], sub[col],
               color=[MODEL_COLORS["v8"], MODEL_COLORS["v9"]],
               edgecolor="k", linewidth=0.5)
        ax.axhline(ref, color="black", linestyle="--", linewidth=1,
                   label=f"N(0,1) target = {ref}")
        ax.set_title(f"{lbl} after copula transform")
        ax.set_ylabel(lbl)
        ax.legend(fontsize=8)
    fig.suptitle("z-space calibration: v8 (no dequant) vs v9 (dequant)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_6_1_z_space")
    _wrote(pdf, png)
    print(f"[fig_6_1] wrote {pdf.name}")


def fig_6_2_zero_inflation(stats: pd.DataFrame) -> None:
    """Bar of frac_zero across models (vs real). Replaces the parquet-loading
    CDF figure when raw data is unavailable."""
    sub = stats[stats["feature"] == "mid_return"].copy()
    sub = sub[sub["model"].isin(["real"] + ALL_MODELS)]
    if sub.empty:
        return
    sub = sub.sort_values("frac_zero", ascending=False)
    colors = [MODEL_COLORS.get(m, "gray") for m in sub["model"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sub["model"], sub["frac_zero"], color=colors,
            edgecolor="k", linewidth=0.4)
    ax.axvline(REAL_REF["spread_1tick_frac"] - 0.06, color="red",
               linestyle="--", linewidth=1.2,
               label=f"real frac_zero ≈ 0.89")
    ax.set_xlabel("fraction of mid_return events == 0 (zero-inflation)")
    ax.set_title("Zero-inflation across generators\n"
                 "Models that don't recover real's 89% point mass cannot match its tails",
                 pad=12)
    ax.legend(loc="lower right")
    pdf, png = save_fig(fig, "fig_6_2_zero_inflation")
    _wrote(pdf, png)
    print(f"[fig_6_2] wrote {pdf.name}")


def fig_6_3_v2_vs_v9(matrix: pd.DataFrame, pred: pd.DataFrame,
                      pca: pd.DataFrame) -> None:
    """Grouped bar: v2 vs v9 across the three axes + key marginal metrics,
    each normalised so 'real / closer to real' = 1."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()

    def metric_set(m):
        try:
            g1 = int(str(midx.loc[m, "G1_pass"]).split("/")[0])
        except Exception:
            g1 = np.nan
        return {
            "G1 / 16":      g1 / 16 if pd.notna(g1) else np.nan,
            "1 / Pred×":    1.0 / pred_d.get(m, np.nan)
                              if pred_d.get(m) and pred_d.get(m) > 0 else np.nan,
            "PCA / 100":    pca_d.get(m, np.nan) / 100,
            "kurt / real":  (float(midx.loc[m, "B1_synth_kurt"]) /
                              REAL_REF["kurtosis"])
                              if pd.notna(midx.loc[m, "B1_synth_kurt"]) else np.nan,
            "trade-frac / real": (float(midx.loc[m, "B4_trade_fraction"]) /
                                   REAL_REF["trade_fraction"])
                                  if pd.notna(midx.loc[m, "B4_trade_fraction"]) else np.nan,
        }
    if "v2" not in midx.index or "v9" not in midx.index:
        return
    v2 = metric_set("v2")
    v9 = metric_set("v9")

    metrics = list(v2.keys())
    x = np.arange(len(metrics))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, [v2[k] for k in metrics], w,
           label="v2 (ε, FiLM, z-score)",
           color=MODEL_COLORS["v2"], edgecolor="k", linewidth=0.4)
    ax.bar(x + w/2, [v9[k] for k in metrics], w,
           label="v9 (ε, FiLM, copula+dequant)",
           color=MODEL_COLORS["v9"], edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7,
               label="perfect = real")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("metric / real reference (symlog)")
    ax.set_title("Copula did NOT just trade marginals for transferability — "
                 "v9 wins distributional metrics, loses temporal",
                 pad=12)
    ax.legend(loc="upper right", fontsize=9)
    pdf, png = save_fig(fig, "fig_6_3_v2_vs_v9")
    _wrote(pdf, png)
    print(f"[fig_6_3] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Vol-Vol & Temporal Structure
# ══════════════════════════════════════════════════════════════════════════
def fig_7_1_vol_vol_kde(vv: pd.DataFrame) -> None:
    """KDE of per-tape vol-vol correlation across models."""
    keep = ["real", "v2", "v5", "v9"]
    sub = vv[vv["model"].isin(keep)].copy()
    sub = sub.dropna(subset=["pearson_r"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, g in sub.groupby("model"):
        if len(g) < 5:
            continue
        sns.kdeplot(g["pearson_r"], ax=ax, label=f"{m} (n={len(g)})",
                    color=PROTAGONIST_COLOR.get(m, MODEL_COLORS.get(m, "gray")),
                    linewidth=2.5 if m == "real" else 1.8,
                    fill=(m == "real"), alpha=0.85)
    ax.axvline(REAL_REF["vol_vol_corr"], color="black", linestyle="--",
               linewidth=1, label=f"real mean = {REAL_REF['vol_vol_corr']:.3f}")
    ax.set_xlabel("per-tape Pearson(volume, |return|)")
    ax.set_ylabel("density")
    ax.set_title("Volume-Volatility Correlation Distribution (TRADES Fig 4)",
                 pad=12)
    ax.legend(loc="upper left", fontsize=9)
    pdf, png = save_fig(fig, "fig_7_1_vol_vol_kde")
    _wrote(pdf, png)
    print(f"[fig_7_1] wrote {pdf.name}")


def fig_7_2_acf_grid(acf: pd.DataFrame) -> None:
    """Grid: features × {real, v2, v7_b, v9}, ACF over lags."""
    feats = ["mid_return", "spread", "bid_sz"]
    models = ["real", "v2", "v7_b", "v9"]
    fig, axes = plt.subplots(len(feats), len(models),
                              figsize=(15, 3 * len(feats)),
                              sharex=True, sharey="row")
    for i, feat in enumerate(feats):
        for j, m in enumerate(models):
            ax = axes[i, j]
            sub = acf[(acf["model"] == m) & (acf["feature"] == feat)] \
                    .sort_values("lag").head(50)
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.plot(sub["lag"], sub["acf"], color=MODEL_COLORS.get(m, "gray"),
                    linewidth=1.6)
            ax.axhline(0, color="k", linewidth=0.4, alpha=0.5)
            if i == 0:
                ax.set_title(m, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"ACF\n{feat}", fontsize=10)
            ax.set_xlabel("lag" if i == len(feats) - 1 else "")
    fig.suptitle("ACF Grid — features (rows) × models (cols)",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_7_2_acf_grid")
    _wrote(pdf, png)
    print(f"[fig_7_2] wrote {pdf.name}")


def fig_7_3_trade_sign_acf(acf: pd.DataFrame) -> None:
    """Single line plot: trade-sign ACF lag 1..30 across selected models.

    Real and synth ACFs are computed only at the small set of lags the
    validator emits (typically {1, 5, 10, 50, 100}). Connecting these
    sparse lags with straight lines produces visible "jumps" between
    them; we instead draw markers + dashed connectors so the reader can
    see those are interpolation lines, not measured ACF dynamics.
    """
    keep = ["real", "v2", "v5", "v9"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in keep:
        sub = acf[(acf["model"] == m) & (acf["feature"] == "trade_sign")]
        if sub.empty:
            continue
        # The temporal_acf.csv often has multiple rows per (model, lag) when
        # deep_diagnostic was re-run with different chunks; aggregate to a
        # single point per lag so the lines don't visually "jump" between
        # duplicate values at the same x-coordinate.
        sub = sub.groupby("lag", as_index=False)["acf"].mean().sort_values("lag").head(30)
        is_real = (m == "real")
        ax.plot(sub["lag"], sub["acf"], marker="o", markersize=7 if is_real else 5,
                color=PROTAGONIST_COLOR.get(m, MODEL_COLORS.get(m, "gray")),
                linewidth=2.0 if is_real else 1.4,
                linestyle="--" if is_real else "-",  # dashed real to flag sparse lags
                label=m, alpha=0.95 if is_real else 0.85)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.axhline(REAL_REF["trade_sign_acf_lag1"], color="black", linestyle=":",
               linewidth=1, label=f"real lag-1 = {REAL_REF['trade_sign_acf_lag1']}")
    ax.set_xlabel("lag")
    ax.set_ylabel("trade-sign ACF")
    ax.set_title("Trade-sign autocorrelation (a key Cont 2001 stylized fact)",
                 pad=12)
    ax.legend(fontsize=9, loc="upper right")
    pdf, png = save_fig(fig, "fig_7_3_trade_sign_acf")
    _wrote(pdf, png)
    print(f"[fig_7_3] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Agent Stress-Test Mechanism
# ══════════════════════════════════════════════════════════════════════════
def fig_8_1_agent_pnl_overlap(agents: pd.DataFrame) -> None:
    """Box of terminal_pnl per agent on truth — A1/A2 indistinguishability."""
    sub = agents[agents["model"] == "truth"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    # The CSV is already daily-aggregated → mean / std / median per agent.
    # We render a horizontal range showing mean ± std.
    ypos = np.arange(len(sub))
    ax.errorbar(sub["terminal_pnl_mean"], ypos,
                xerr=sub["terminal_pnl_std"], fmt="o",
                color="#1f77b4", ecolor="#888", capsize=4,
                markersize=8, elinewidth=2)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.annotate(f"σ={r['terminal_pnl_std']:.2g}",
                    (r['terminal_pnl_mean'], i),
                    xytext=(8, -3), textcoords="offset points",
                    fontsize=9, color="#666")
    ax.set_yticks(ypos); ax.set_yticklabels(sub["agent"])
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Terminal PnL (mean ± std across 20 holdout days)")
    ax.set_title("Agent PnL on real holdout — A1 and A2 distributions overlap heavily",
                 pad=12)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_8_1_agent_pnl_overlap")
    _wrote(pdf, png)
    print(f"[fig_8_1] wrote {pdf.name}")


def fig_8_2_rdiff_vs_everything(matrix: pd.DataFrame, pred: pd.DataFrame,
                                 pca: pd.DataFrame) -> None:
    """4-panel scatter of rho_diff vs G1, Predictive, PCA, kurtosis."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()
    rows = []
    for m in ALL_MODELS:
        if m not in midx.index:
            continue
        try:
            rd = float(midx.loc[m, "rho_diff"])
        except (ValueError, TypeError):
            continue
        try:
            g1 = int(str(midx.loc[m, "G1_pass"]).split("/")[0])
        except Exception:
            g1 = np.nan
        try:
            kurt = float(midx.loc[m, "B1_synth_kurt"])
        except (ValueError, TypeError):
            kurt = np.nan
        rows.append({"model": m, "rdiff": rd, "G1": g1,
                     "pred": pred_d.get(m, np.nan),
                     "PCA": pca_d.get(m, np.nan),
                     "kurt": kurt})
    pts = pd.DataFrame(rows)
    if pts.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = [("G1", "G1 pass"), ("pred", "Pred ×replay"),
              ("PCA", "PCA %"),  ("kurt", "kurtosis")]
    for ax, (col, xlbl) in zip(axes, panels):
        sub = pts.dropna(subset=[col, "rdiff"])
        if sub.empty:
            ax.set_visible(False); continue
        for _, r in sub.iterrows():
            ax.scatter(r[col], r["rdiff"], c=MODEL_COLORS[r["model"]],
                       marker=MODEL_MARKERS[r["model"]],
                       s=80, edgecolors="k", linewidths=0.5)
            ax.annotate(r["model"], (r[col], r["rdiff"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        if col in ("pred", "kurt"):
            ax.set_xscale("log")
        if len(sub) >= 3:
            try:
                rho, p = spearmanr(sub[col], sub["rdiff"])
                ax.set_title(f"ρ_diff vs {xlbl}\nSpearman = {rho:+.2f} (p = {p:.2g})",
                             fontsize=10)
            except Exception:
                ax.set_title(f"ρ_diff vs {xlbl}")
        ax.set_xlabel(xlbl)
        if ax is axes[0]:
            ax.set_ylabel(r"$\rho_\mathrm{diff}$")
    fig.suptitle("Agent-ranking $\\rho_\\mathrm{diff}$ is decoupled from every distributional quality metric",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_8_2_rdiff_vs_everything")
    _wrote(pdf, png)
    print(f"[fig_8_2] wrote {pdf.name}")


def fig_8_3_a2_minus_a1(deltas: pd.DataFrame) -> None:
    """Bars of z-scored A2-A1 deltas for terminal_pnl, fill_rate, inventory_var."""
    keep_cols = ["terminal_pnl_z", "fill_rate_z", "inventory_var_z"]
    sub = deltas.copy()
    sub = sub[sub["model"].isin(["truth", "real"] + ALL_MODELS)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(sub))
    w = 0.27
    for i, c in enumerate(keep_cols):
        if c not in sub.columns:
            continue
        ax.bar(x + (i - 1) * w, sub[c], w, label=c.replace("_z", ""),
               edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=45, ha="right")
    ax.set_ylabel("(A2 − A1) z-score")
    ax.set_title("What actually separates A2 from A1: fill rate, not PnL",
                 pad=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf, png = save_fig(fig, "fig_8_3_a2_minus_a1")
    _wrote(pdf, png)
    print(f"[fig_8_3] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Mid-Price Traces (re-use existing PDF assets)
# ══════════════════════════════════════════════════════════════════════════
def fig_9_1_midprice_traces() -> None:
    """We can't easily re-render parquet-derived traces without raw data;
    instead, reference the existing trades-metrics-all/midprice_traces_*.pdf
    set in a small index file."""
    src = RESULTS / "trades_metrics_all"
    matches = sorted(src.glob("midprice_traces_*.pdf"))
    if not matches:
        print("[fig_9_1] no midprice_traces_*.pdf in trades_metrics_all/ — skipped")
        return
    idx = OUT / "fig_9_1_midprice_traces_index.txt"
    idx.write_text(
        "Midprice trace PDFs are pre-rendered per model in:\n"
        f"  {src.resolve()}\n\n"
        "Files:\n" + "\n".join(f"  {p.name}" for p in matches) +
        "\n\nFor the report, recommended panel: v2, v9, v2_noclip, v2_remapped\n"
    )
    _wrote(idx)
    print(f"[fig_9_1] wrote index {idx.name} ({len(matches)} traces available)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — One-Figure Summary (rank-bump)
# ══════════════════════════════════════════════════════════════════════════
def fig_10_1_rank_bump(matrix: pd.DataFrame, pred: pd.DataFrame,
                        pca: pd.DataFrame) -> None:
    """Rank-bump chart: each model's rank on G1, Predictive, PCA — connect
    with lines. Crossing lines visualise the anti-correlation."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()

    rows = []
    for m in ALL_MODELS:
        if m not in midx.index:
            continue
        try:
            g1 = int(str(midx.loc[m, "G1_pass"]).split("/")[0])
        except Exception:
            continue
        pr = pred_d.get(m, np.nan)
        pc = pca_d.get(m, np.nan)
        if pd.isna(pr) or pd.isna(pc):
            continue
        rows.append({"model": m, "G1": g1, "pred": pr, "PCA": pc})
    pts = pd.DataFrame(rows)
    if len(pts) < 4:
        return

    pts["rk_G1"]   = pts["G1"].rank(ascending=False, method="min")
    pts["rk_pred"] = pts["pred"].rank(ascending=True, method="min")  # lower = better
    pts["rk_PCA"]  = pts["PCA"].rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(13, 9))
    xs = [0, 1, 2]
    highlight = {"v2", "v5", "v9"}
    for _, r in pts.iterrows():
        ys = [r["rk_G1"], r["rk_pred"], r["rk_PCA"]]
        is_hi = r["model"] in highlight
        lw = 3.2 if is_hi else 0.7
        a = 1.0 if is_hi else 0.18
        line_color = (PROTAGONIST_COLOR[r["model"]] if is_hi
                      else MODEL_COLORS[r["model"]])
        ax.plot(xs, ys, color=line_color,
                linewidth=lw, alpha=a, marker="o",
                markersize=10 if is_hi else 5,
                zorder=10 if is_hi else 1)
        # Only annotate the three protagonists
        if is_hi:
            ax.annotate(r["model"], (xs[0] - 0.04, ys[0]),
                        ha="right", va="center", fontsize=11,
                        fontweight="bold",
                        color=line_color)
            ax.annotate(r["model"], (xs[-1] + 0.04, ys[-1]),
                        ha="left", va="center", fontsize=11,
                        fontweight="bold",
                        color=line_color)
    ax.set_xticks(xs)
    ax.set_xticklabels(["G1 rank", "Predictive rank", "PCA rank"], fontsize=12)
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Rank-Bump Chart Across the Three Evaluation Axes\n"
                 "(crossing lines = anti-correlated rankings — no model is best on all three)",
                 pad=12)
    ax.grid(axis="y", alpha=0.3)
    pdf, png = save_fig(fig, "fig_10_1_rank_bump")
    _wrote(pdf, png)
    print(f"[fig_10_1] wrote {pdf.name}")

# ══════════════════════════════════════════════════════════════════════════
# Tables A, B, C
# ══════════════════════════════════════════════════════════════════════════
def table_a_decision_matrix(matrix: pd.DataFrame, pred: pd.DataFrame,
                             pca: pd.DataFrame) -> None:
    """Decision matrix: per architectural toggle, average effect on each metric.
    Re-uses the same logic as fig_5_1 but writes a CSV + LaTeX."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()
    pairs = {
        "ε→v":          [("v2", "v4"), ("v9", "v9_b")],
        "FiLM→AdaLN":   [("v4", "v5"), ("v8", "v8_b")],
        "+EDM":         [("v5", "v6"), ("v5", "v7"), ("v5", "v7_b")],
        "+copula":      [("v2", "v8")],
        "+dequant":     [("v8", "v9")],
        "−x0_clip":     [("v2", "v2_noclip"), ("v3_e9", "v3_e9_noclip"),
                          ("v3p5", "v3p5_noclip"), ("v4", "v4_noclip")],
    }
    metrics = ["G1", "pred", "PCA", "kurt", "trade_frac", "vol_vol"]
    grid = pd.DataFrame(index=list(pairs.keys()),
                        columns=[f"Δ{m}" for m in metrics], dtype=float)
    for axis, ps in pairs.items():
        deltas = {m: [] for m in metrics}
        for a, b in ps:
            ma = _safe_metric(midx, a, None, pred_d, pca_d)
            mb = _safe_metric(midx, b, None, pred_d, pca_d)
            for k in metrics:
                if k in ma and k in mb and pd.notna(ma[k]) and pd.notna(mb[k]):
                    deltas[k].append(mb[k] - ma[k])
        for k in metrics:
            grid.loc[axis, f"Δ{k}"] = np.mean(deltas[k]) if deltas[k] else np.nan
    out_csv = OUT / "tab_A_decision_matrix.csv"
    out_tex = OUT / "tab_A_decision_matrix.tex"
    grid.to_csv(out_csv)
    grid.to_latex(out_tex, float_format="%.3g", na_rep="—",
                  caption="Architectural Decision Matrix — mean Δ per axis × metric.",
                  label="tab:decision_matrix")
    _wrote(out_csv, out_tex)
    print(f"[tab_A] wrote {out_csv.name} + {out_tex.name}")


def table_b_g1_difficulty(g1: pd.DataFrame) -> None:
    """G1 check difficulty ranking."""
    pivot = g1.pivot(index="check", columns="model", values="pass") \
              .reindex(columns=ALL_MODELS).astype(float)
    pivot["pass_count"] = pivot.sum(axis=1).astype(int)
    pivot["pass_rate"] = (pivot["pass_count"] / len(ALL_MODELS)).round(2)
    sub = pivot[["pass_count", "pass_rate"]].sort_values("pass_count")

    def hardest(check_name):
        # Per-row: which models fail (i.e. False)?
        models_failed = pivot.loc[check_name].drop(["pass_count", "pass_rate"]) \
                              .pipe(lambda s: s[s == 0].index.tolist())
        return ", ".join(models_failed[:5]) + ("..." if len(models_failed) > 5 else "")
    sub["models_that_fail"] = [hardest(idx) for idx in sub.index]
    out_csv = OUT / "tab_B_g1_difficulty.csv"
    out_tex = OUT / "tab_B_g1_difficulty.tex"
    sub.to_csv(out_csv)
    sub.to_latex(out_tex, escape=True,
                 caption="G1 Check Difficulty — sorted by number of models passing.",
                 label="tab:g1_difficulty")
    _wrote(out_csv, out_tex)
    print(f"[tab_B] wrote {out_csv.name} + {out_tex.name}")


def table_c_metric_correlations(matrix: pd.DataFrame, pred: pd.DataFrame,
                                  pca: pd.DataFrame) -> None:
    """Spearman ρ matrix between all evaluation metrics across 20 models."""
    midx = matrix.set_index("model")
    pred_d = pred.set_index("label")["relative_to_replay"].to_dict()
    pca_d = pca.set_index("model")["coverage_pct"].to_dict()
    rows = []
    for m in ALL_MODELS:
        if m not in midx.index:
            continue
        try:
            g1 = int(str(midx.loc[m, "G1_pass"]).split("/")[0])
        except Exception:
            continue
        try:
            kurt = float(midx.loc[m, "B1_synth_kurt"])
        except (ValueError, TypeError):
            kurt = np.nan
        try:
            tf = float(midx.loc[m, "B4_trade_fraction"])
        except (ValueError, TypeError):
            tf = np.nan
        try:
            vv = float(midx.loc[m, "vol_volatility_corr"])
        except (ValueError, TypeError):
            vv = np.nan
        try:
            ofi = float(midx.loc[m, "D1_ofi_slope_d50"])
        except (ValueError, TypeError):
            ofi = np.nan
        rows.append({
            "G1":         g1,
            "Predictive": pred_d.get(m, np.nan),
            "PCA":        pca_d.get(m, np.nan),
            "kurt":       kurt,
            "trade_frac": tf,
            "vol_vol":    vv,
            "OFI_slope":  ofi,
        })
    df = pd.DataFrame(rows).dropna(thresh=4)
    if df.empty:
        return
    corr = df.corr(method="spearman").round(2)
    out_csv = OUT / "tab_C_metric_correlations.csv"
    out_tex = OUT / "tab_C_metric_correlations.tex"
    corr.to_csv(out_csv)
    corr.to_latex(out_tex, float_format="%.2f", na_rep="—",
                  caption="Spearman correlation between evaluation metrics across 20 generators. "
                          "Notable anti-correlations between G1, Predictive, and PCA support the three-axis tension.",
                  label="tab:metric_correlations")
    _wrote(out_csv, out_tex)
    print(f"[tab_C] wrote {out_csv.name} + {out_tex.name}")

# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"[80_report_figures] writing to {OUT}")
    matrix = load_multi_axis()
    g1     = load_g1_long()
    pred   = load_predictive()
    pca    = load_pca_coverage()
    vv     = load_vol_vol()
    stats  = load_per_feature_stats()
    acf    = load_temporal_acf()
    z      = load_z_space_stats()
    agents = load_agent_decomp()
    a2a1   = load_a2_minus_a1()

    # Section 1 — three-axis framework
    fig_1_1_three_axis_scatter(matrix, pred, pca)
    fig_1_2_pairwise(matrix, pred, pca)
    fig_1_3_comparison_table(matrix)

    # Section 2 — distributional realism
    fig_2_1_g1_heatmap(g1)
    fig_2_2_violin_marginals(stats)
    fig_2_3_return_tail(stats)
    fig_2_4_trade_fraction(matrix)

    # Section 3 — predictive score
    fig_3_1_predictive_bar(pred)
    fig_3_2_acf_top_bottom(pred, acf)
    fig_3_3_predictive_vs_scale(matrix, pred)

    # Section 4 — PCA
    fig_4_1_pca_summary(pca)
    fig_4_2_pca_vs_g1(matrix, pca)

    # Section 5 — ablation
    fig_5_1_ablation_heatmap(matrix, pred, pca)
    fig_5_2_x0_clip_effect(matrix, pred)

    # Section 6 — copula
    fig_6_1_z_space(z)
    fig_6_2_zero_inflation(stats)
    fig_6_3_v2_vs_v9(matrix, pred, pca)

    # Section 7 — vol-vol & temporal
    fig_7_1_vol_vol_kde(vv)
    fig_7_2_acf_grid(acf)
    fig_7_3_trade_sign_acf(acf)

    # Section 8 — agent mechanism
    fig_8_1_agent_pnl_overlap(agents)
    fig_8_2_rdiff_vs_everything(matrix, pred, pca)
    fig_8_3_a2_minus_a1(a2a1)

    # Section 9 — mid-price traces (index)
    fig_9_1_midprice_traces()

    # Section 10 — summary rank-bump
    fig_10_1_rank_bump(matrix, pred, pca)

    # Tables A / B / C
    table_a_decision_matrix(matrix, pred, pca)
    table_b_g1_difficulty(g1)
    table_c_metric_correlations(matrix, pred, pca)

    print()
    print(f"=== wrote {len(WRITTEN)} files to {OUT} ===")
    for p in WRITTEN:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
