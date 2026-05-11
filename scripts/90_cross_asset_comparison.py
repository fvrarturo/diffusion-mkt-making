"""
90_cross_asset_comparison.py — Cross-asset (INTC, TSLA, SPY) × architecture
(v2, v5) comparison for §6 of the report.

Tests the central question: is the three-axis tension structural to MSE-based
diffusion training on heavy-tailed multivariate financial data, or is it
INTC-specific (a zero-inflation artifact)?

Reads:
  results/_archive/multi_axis_comparison.csv     (INTC v2/v5 + REAL row)
  results/_archive/tsla_v{2,5}/validation/G1_pass_fail_summary.csv
  results/trades_metrics_tsla/pca_coverage.csv
  results/predictive_score_tsla/predictive_score.csv
  (when available) results/_archive/spy_v{2,5}/...
                   results/trades_metrics_spy/...
                   results/predictive_score_spy/...

Writes:
  results/report_figures/tab_6_1_cross_asset.{csv,tex}    (ticker × model × axis)
  results/report_figures/fig_6_1_cross_asset_ternary.{pdf,png}
  results/report_figures/fig_6_2_cross_asset_axis_bars.{pdf,png}

Run:
  python scripts/90_cross_asset_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "_archive"
OUT = RESULTS / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

TICKERS = ["INTC", "TSLA", "SPY"]
MODELS = ["v2", "v5"]

TICKER_COLOR  = {"INTC": "#1f77b4", "TSLA": "#d62728", "SPY":  "#2ca02c"}
MODEL_MARKER  = {"v2": "o", "v5": "^"}
MODEL_LINE    = {"v2": "-", "v5": "--"}


# ─── Loaders ───────────────────────────────────────────────────────────────
def _load_g1_pass(path: Path) -> tuple[int | None, int | None]:
    """Returns (passes, total) from a G1_pass_fail_summary.csv."""
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df = df[df.get("Check", df.get("check", pd.Series(dtype=str))) != "TOTAL_PASS"]
    pass_col = next((c for c in ["Pass", "pass", "passed", "result"]
                     if c in df.columns), None)
    if pass_col is None:
        return None, len(df)
    pc = df[pass_col].astype(str).str.lower().eq("true").sum()
    return int(pc), int(len(df))


def load_ticker(ticker: str) -> pd.DataFrame:
    """Returns one row per model with G1, predictive, PCA, kurtosis, trade_frac.
    Returns empty rows for missing data so the table layout stays consistent."""
    rows = []

    if ticker == "INTC":
        # INTC numbers come from the central multi_axis_comparison.csv
        try:
            mac = pd.read_csv(ARCHIVE / "multi_axis_comparison.csv")
            mac = mac.set_index("model")
            for m in MODELS:
                if m not in mac.index:
                    continue
                r = mac.loc[m]
                try:
                    g1 = int(str(r["G1_pass"]).split("/")[0])
                    g1_total = int(str(r["G1_pass"]).split("/")[1])
                except Exception:
                    g1, g1_total = None, None
                rows.append({
                    "ticker":     ticker,
                    "model":      m,
                    "G1_pass":    g1,
                    "G1_total":   g1_total,
                    "Pred_x":     float(r["predictive_x_replay"]) if pd.notna(r["predictive_x_replay"]) else None,
                    "PCA_pct":    float(r["PCA_coverage_pct"]) if pd.notna(r["PCA_coverage_pct"]) else None,
                    "kurtosis":   float(r["B1_synth_kurt"]) if pd.notna(r["B1_synth_kurt"]) else None,
                    "trade_frac": float(r["B4_trade_fraction"]) if pd.notna(r["B4_trade_fraction"]) else None,
                })
        except Exception as e:
            print(f"  [{ticker}] failed to load multi_axis_comparison: {e}")
        return pd.DataFrame(rows)

    # TSLA/SPY: pull from per-ticker dirs
    pred_path = RESULTS / f"predictive_score_{ticker.lower()}" / "predictive_score.csv"
    pca_path  = RESULTS / f"trades_metrics_{ticker.lower()}" / "pca_coverage.csv"

    pred_d = {}
    if pred_path.exists():
        try:
            pdf = pd.read_csv(pred_path)
            pred_d = pdf.set_index("label")["relative_to_replay"].to_dict()
        except Exception as e:
            print(f"  [{ticker}] predictive read failed: {e}")
    pca_d = {}
    if pca_path.exists():
        try:
            pdf = pd.read_csv(pca_path)
            pca_d = pdf.set_index("model")["coverage_pct"].to_dict()
        except Exception as e:
            print(f"  [{ticker}] pca read failed: {e}")

    for m in MODELS:
        # G1 from per-model validation dir
        g1_path = ARCHIVE / f"{ticker.lower()}_{m}" / "validation" / "G1_pass_fail_summary.csv"
        g1_pass, g1_total = _load_g1_pass(g1_path)

        # Marginal stats (kurtosis, trade_frac) from B1/B4 CSVs
        kurt = None
        trade_frac = None
        b1_path = ARCHIVE / f"{ticker.lower()}_{m}" / "validation" / "B1_return_stats.csv"
        b4_path = ARCHIVE / f"{ticker.lower()}_{m}" / "validation" / "B4_event_stats.csv"
        if b1_path.exists():
            try:
                b1 = pd.read_csv(b1_path)
                if "metric" in b1.columns and "value" in b1.columns:
                    s = b1[b1["metric"] == "synth_kurt"]["value"]
                    if not s.empty:
                        kurt = float(s.iloc[0])
            except Exception:
                pass
        if b4_path.exists():
            try:
                b4 = pd.read_csv(b4_path)
                if "metric" in b4.columns and "value" in b4.columns:
                    s = b4[b4["metric"] == "synth_trade_frac"]["value"]
                    if not s.empty:
                        trade_frac = float(s.iloc[0])
            except Exception:
                pass

        rows.append({
            "ticker":     ticker,
            "model":      m,
            "G1_pass":    g1_pass,
            "G1_total":   g1_total,
            "Pred_x":     pred_d.get(m),
            "PCA_pct":    pca_d.get(m),
            "kurtosis":   kurt,
            "trade_frac": trade_frac,
        })
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    frames = [load_ticker(t) for t in TICKERS]
    df = pd.concat(frames, ignore_index=True)
    return df


# ─── Table 6.1 ─────────────────────────────────────────────────────────────
def write_comparison_table(df: pd.DataFrame) -> None:
    cols = ["ticker", "model", "G1_pass", "G1_total", "Pred_x",
            "PCA_pct", "kurtosis", "trade_frac"]
    sub = df[cols].copy()

    def _fmt_g1(r):
        if pd.isna(r["G1_pass"]) or pd.isna(r["G1_total"]):
            return "—"
        return f"{int(r['G1_pass'])}/{int(r['G1_total'])}"

    sub["G1"] = sub.apply(_fmt_g1, axis=1)
    sub["Pred ×replay"] = sub["Pred_x"].apply(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    sub["PCA %"]        = sub["PCA_pct"].apply(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
    sub["kurtosis"]     = sub["kurtosis"].apply(lambda v: "—" if pd.isna(v) else f"{v:.0f}")
    sub["trade frac"]   = sub["trade_frac"].apply(lambda v: "—" if pd.isna(v) else f"{v:.3f}")

    out = sub[["ticker", "model", "G1", "Pred ×replay",
               "PCA %", "kurtosis", "trade frac"]]
    out_csv = OUT / "tab_6_1_cross_asset.csv"
    out_tex = OUT / "tab_6_1_cross_asset.tex"
    out.to_csv(out_csv, index=False)
    latex = out.to_latex(index=False, escape=True,
                         column_format="l" + "l" + "r" * 5,
                         caption="Cross-asset comparison: ε+FiLM (v2) vs v+AdaLN-Zero (v5) "
                                 "across INTC (large-tick), TSLA (small-tick), SPY (ETF). "
                                 "G1 = stylized-fact pass count; Pred ×replay = "
                                 "test MAE relative to market-replay baseline; "
                                 "PCA \\% = convex-hull coverage of real.",
                         label="tab:cross_asset")
    out_tex.write_text(latex)
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")
    print()
    print(out.to_string(index=False))


# ─── Fig 6.1: Three-axis ternary ────────────────────────────────────────────
def fig_three_axis_scatter(df: pd.DataFrame) -> None:
    """2D scatter: G1 (x) vs 1/Pred×replay (y), point size = PCA coverage,
    color = ticker, marker = architecture. Same encoding as fig_1_1 but with
    cross-asset points."""
    pts = df.dropna(subset=["G1_pass", "G1_total", "Pred_x", "PCA_pct"]).copy()
    if pts.empty:
        print("[fig_6_1] no complete rows — skipped")
        return
    pts["G1_norm"]   = pts["G1_pass"] / pts["G1_total"]
    pts["pred_inv"]  = 1.0 / pts["Pred_x"].clip(lower=1e-6)
    if pts["pred_inv"].max() > pts["pred_inv"].min():
        pts["pred_score"] = (pts["pred_inv"] - pts["pred_inv"].min()) / \
                             (pts["pred_inv"].max() - pts["pred_inv"].min())
    else:
        pts["pred_score"] = 0.5

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in pts.iterrows():
        size = 80 + 8 * r["PCA_pct"]
        ax.scatter(r["G1_norm"], r["pred_score"],
                   s=size, c=TICKER_COLOR[r["ticker"]],
                   marker=MODEL_MARKER[r["model"]],
                   edgecolors="k", linewidths=0.7, alpha=0.85,
                   label=f"{r['ticker']} {r['model']}")
        ax.annotate(f"{r['ticker']}\n{r['model']}",
                    (r["G1_norm"], r["pred_score"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)

    # Real reference (perfect)
    ax.scatter([1.0], [1.05], marker="*", s=400, c="black", zorder=5)
    ax.annotate("REAL", (1.0, 1.05), xytext=(8, -2),
                textcoords="offset points", fontsize=10, fontweight="bold")

    # Group legend (color by ticker, marker by model)
    h = []
    for tk in TICKERS:
        if tk in pts["ticker"].values:
            h.append(plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=TICKER_COLOR[tk], markersize=10,
                                markeredgecolor="k", label=tk))
    for mk in MODELS:
        h.append(plt.Line2D([0], [0], marker=MODEL_MARKER[mk], color="k",
                            markerfacecolor="gray", markersize=10,
                            linestyle="", label=mk))
    ax.legend(handles=h, loc="lower left", fontsize=9, framealpha=0.9,
              title="ticker / model")

    ax.set_xlabel("Distributional Realism (G1 pass fraction)")
    ax.set_ylabel("Temporal Transferability (1 / ×replay, min-max scaled)")
    ax.set_title("Cross-Asset Three-Axis Comparison\n"
                 "(point area $\\propto$ PCA convex-hull coverage)",
                 pad=12)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    pdf = OUT / "fig_6_1_cross_asset_ternary.pdf"
    png = OUT / "fig_6_1_cross_asset_ternary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


# ─── Fig 6.2: Per-axis grouped bars ─────────────────────────────────────────
def fig_axis_bars(df: pd.DataFrame) -> None:
    """3-panel grouped bar chart — one panel per axis. Each panel: ticker on
    x-axis, two bars per ticker (v2 + v5). Lets you read off per-axis
    cross-asset effects directly."""
    panels = [
        ("G1_norm",   "G1 pass fraction (out of 15-16)", "G1 — distributional realism"),
        ("Pred_x",    "Predictive ×replay  (lower = better; 1.0 = market replay)",
                      "Predictive Score — temporal transferability"),
        ("PCA_pct",   "PCA convex-hull coverage of real (%)",
                      "PCA — multivariate state-space coverage"),
    ]
    df = df.copy()
    df["G1_norm"] = df["G1_pass"] / df["G1_total"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(TICKERS))
    w = 0.35

    for ax, (col, ylabel, title) in zip(axes, panels):
        for i, m in enumerate(MODELS):
            sub = df[df["model"] == m].set_index("ticker").reindex(TICKERS)
            vals = sub[col].astype(float).values
            ax.bar(x + (i - 0.5) * w, vals, w,
                   color=[TICKER_COLOR[t] for t in TICKERS],
                   edgecolor="k", linewidth=0.5,
                   alpha=0.7 if m == "v5" else 0.95,
                   hatch="" if m == "v2" else "///",
                   label=m)
            for j, v in enumerate(vals):
                if pd.notna(v):
                    ax.annotate(f"{v:.2f}" if col == "G1_norm" else
                                f"{v:.1f}" if col == "PCA_pct" else
                                f"{v:.1f}×",
                                (x[j] + (i - 0.5) * w, v),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(TICKERS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if col == "Pred_x":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                       label="market replay = 1.0")
        if col == "PCA_pct":
            ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.6,
                       label="real = 100%")

    # Single legend (shared) — model styles only (color is encoded by ticker)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="k", label="v2 (ε+FiLM)"),
        plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="k", hatch="///", alpha=0.7, label="v5 (v+AdaLN)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=10)
    fig.suptitle("Cross-asset three-axis comparison — does the INTC tension generalize?",
                 fontsize=14, y=1.07)
    fig.tight_layout()
    pdf = OUT / "fig_6_2_cross_asset_axis_bars.pdf"
    png = OUT / "fig_6_2_cross_asset_axis_bars.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


def main() -> None:
    print("Loading cross-asset data...")
    df = load_all()
    print(f"loaded {len(df)} rows: {df['ticker'].value_counts().to_dict()}")
    print()
    write_comparison_table(df)
    print()
    fig_three_axis_scatter(df)
    fig_axis_bars(df)


if __name__ == "__main__":
    main()
