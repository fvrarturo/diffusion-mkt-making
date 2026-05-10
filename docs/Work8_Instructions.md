# Report Analysis Suite — "The Three-Axis Tension in Diffusion-Based LOB Generation"

## Context for the coding agent

You are building the complete analysis and figure suite for a course report (MIT Sloan 15.458). The project trained 20 diffusion model checkpoints to generate synthetic Limit Order Book (LOB) data for INTC (Intel), varying across 5 architectural axes: prediction target (ε vs v), conditioning module (FiLM vs AdaLN-Zero), preconditioning (none vs EDM), data transform (z-score vs copula vs copula+dequantization), and loss weighting (Min-SNR γ).

The central finding: three evaluation dimensions — distributional realism (G1 stylized facts), temporal pattern transferability (predictive score), and multivariate coverage (PCA) — are empirically anti-correlated across architectures. No model optimizes all three. This tradeoff is structural to MSE-based diffusion training on heavy-tailed multivariate financial data.

## Data available

All results are pre-computed. You are reading CSVs/JSONs and producing figures + analyses.

### File locations

```
# Per-model validation results (G1 pass/fail + all metric CSVs)
results/_archive/{model}/validation/
  G1_pass_fail_summary.csv
  B1_return_stats.csv
  B2_spread_stats.csv
  C1_acf_stats.csv
  D1_regression_stats.csv
  F1b_cosine_similarity.csv
  ...

# Collated G1 across all models
results/_archive/G1_comparison_all_models.csv

# TRADES supplementary metrics
results/trades_metrics_all/
  vol_volatility_correlation.csv      # columns: model, tape_id, pearson_r
  leverage_effect.csv
  pca_coverage.csv                    # columns: model, coverage_pct, ...
  pca_projections.csv                 # columns: model, tape_id, pc1, pc2

# Predictive score
results/predictive_score_all/
  predictive_score.csv                # columns: model, test_mae, relative_to_replay

# Deep diagnostic
results/deep_diagnostic/
  per_feature_stats.csv
  correlation_comparison.csv
  temporal_acf.csv
  window_level_stats.csv
  z_space_stats.csv

# Ranking mechanism
results/ranking_mechanism/
  agent_decomposition.csv
  a2_minus_a1_deltas.csv
  conditional_returns.csv

# Hypothesis test results
results/_archive/{model}/hypothesis/bootstrap.json

# Synthetic parquets (for any additional per-tape analysis)
$HOME/orcd/scratch/diffmm/synthetic_*/INTC/{base,high_vol,toxic,thin}/*.parquet

# Real validation data
data/processed/val/INTC_*.parquet
```

### Model metadata (for labeling/grouping)

```python
MODEL_META = {
    # model_name: (pred_type, conditioning, edm, data_transform, min_snr_gamma, x0_clip)
    "v2":           ("ε", "FiLM",     False, "zscore",          5.0,  4),
    "v2_noclip":    ("ε", "FiLM",     False, "zscore",          5.0,  None),
    "v2_remapped":  ("ε", "FiLM",     False, "zscore+remap",    5.0,  4),
    "v3_e9":        ("ε", "FiLM",     False, "zscore",          1.0,  4),
    "v3_e19":       ("ε", "FiLM",     False, "zscore",          1.0,  4),
    "v3p5":         ("ε", "FiLM",     False, "zscore",          5.0,  4),
    "v3p5_noclip":  ("ε", "FiLM",     False, "zscore",          5.0,  None),
    "v3_e9_noclip": ("ε", "FiLM",     False, "zscore",          1.0,  None),
    "v4":           ("v", "FiLM",     False, "zscore",          5.0,  4),
    "v4_noclip":    ("v", "FiLM",     False, "zscore",          5.0,  None),
    "v5":           ("v", "AdaLN",    False, "zscore",          5.0,  4),
    "v6":           ("v", "AdaLN",    True,  "zscore",          5.0,  4),
    "v7":           ("v", "AdaLN",    True,  "zscore",          5.0,  4),
    "v7_b":         ("v", "AdaLN",    True,  "zscore",          5.0,  4),
    "v8":           ("ε", "FiLM",     False, "copula",          5.0,  4),
    "v8_b":         ("ε", "AdaLN",    False, "copula",          5.0,  4),
    "v9":           ("ε", "FiLM",     False, "copula+dequant",  5.0,  4),
    "v9_b":         ("v", "FiLM",     False, "copula+dequant",  5.0,  4),
    "v10_cascade":  ("ε→v", "FiLM",  False, "zscore",          5.0,  4),
    "v11":          ("ε", "AdaLN",    False, "copula",          5.0,  4),
}

# Real reference values
REAL_REF = {
    "kurtosis": 755,
    "std": 2.19e-5,
    "trade_fraction": 0.0477,
    "vol_vol_corr": 0.341,
    "acf_abs_ret_lag50": 0.0464,
    "acf_beta": 0.21,
    "trade_sign_acf_lag1": 0.675,
    "ofi_slope_delta50": 1.11e-4,
    "spread_1tick_frac": 0.95,
}
```

## Output

Create a single script `scripts/80_report_figures.py` that produces ALL figures and analysis tables below, writing to `results/report_figures/`. Use matplotlib with a consistent style throughout (suggest `seaborn-v0_8-whitegrid`, fontsize 11, figsize defaults noted per figure). Use a consistent color scheme: assign each model a fixed color from a qualitative palette (tab20), and use the same color everywhere.

All figures should be publication-quality: PDF + PNG at 300 DPI, clean axis labels, legends outside the plot area where space is tight, no chartjunk.

---

## SECTION 1: The Three-Axis Framework

### Figure 1.1 — The Three-Axis Radar/Triangle Plot (Hero Figure)

**Purpose**: Single figure showing the three-way tension. Each model is a point in a triangle where the three vertices are (G1 score, 1/predictive_score, PCA coverage). Models that are good on all three are near the center; models that sacrifice one axis are pulled toward the opposite edge.

**Implementation**:
- Ternary plot (use `python-ternary` package if available, otherwise a custom matplotlib triangle)
- Three axes: 
  - Axis A: G1 normalized = G1_pass_count / 15 (range 0-1)
  - Axis B: Predictive = 1 / (relative_to_replay) normalized to [0, 1] via min-max across models
  - Axis C: PCA_coverage / 100 (range 0-1)
- Each model is a labeled scatter point, colored by its "philosophy" group:
  - Blue: scale-accurate (v2, v3_e9, v3p5, v4, v7, v7_b, etc. — clip=4, no copula)
  - Red: scale-exploded (v2_noclip, v3_e9_noclip, v3p5_noclip — no clip)
  - Green: copula-transformed (v8, v9, v9_b — copula)
  - Orange: special (v2_remapped, v10_cascade, v11)
- Annotate: real reference at (1, 1, 1) as a star
- Figsize: 8×8
- Title: "Three-Axis Evaluation Space for LOB Diffusion Generators"

**Fallback if ternary is hard**: 2D scatter of G1 vs Predictive with point SIZE = PCA coverage. Still shows the anti-correlation clearly.

### Figure 1.2 — Pairwise Anti-Correlation Scatter Matrix

**Purpose**: Show the pairwise relationships between all three axes explicitly.

**Implementation**:
- 3×3 scatter matrix (or 3 panels: G1 vs Predictive, G1 vs PCA, Predictive vs PCA)
- Each panel: scatter of models, labeled, colored by group
- Add Spearman ρ annotation in each panel
- Add a regression line (or LOWESS) to show the trend
- Figsize: 14×5 (three panels side by side)
- Exclude degenerate models (v8_b, v11) that have missing values

### Figure 1.3 — The Complete Comparison Table (Formatted)

**Purpose**: The canonical multi-axis comparison table, formatted for the report.

**Implementation**:
- Read the markdown table provided above (or reconstruct from CSVs)
- Output as a LaTeX-formatted table AND as a matplotlib table figure
- Highlight cells: green for "best in column", red for "worst in column", yellow for "within 20% of real"
- Group rows by architecture philosophy (scale-accurate, scale-exploded, copula)
- Include column headers with real reference values

---

## SECTION 2: Distributional Realism (G1 Axis) Deep Dive

### Figure 2.1 — G1 Pass/Fail Heatmap

**Purpose**: Visual representation of the 16×20 G1 matrix.

**Implementation**:
- Heatmap with models as columns, G1 checks as rows
- Color: green=pass, red=fail, gray=n/a
- Annotate each cell with the actual value (small font)
- Sort columns by total pass count (best to worst)
- Sort rows by "difficulty" (how many models pass each check — hardest checks at top)
- Figsize: 16×10
- Add a summary row at the bottom showing pass counts

### Figure 2.2 — Per-Feature Marginal Comparison (Violin Plots)

**Purpose**: Show the distribution of key features across models vs real.

**Implementation**:
- 4 panels: mid_return, spread, bid_sz, trade_sz
- Each panel: violin plot with real on the left (black), then each model as a colored violin
- Use log scale for trade_sz (heavy-tailed)
- Only show a subset of models (v2, v5, v7_b, v9, v2_remapped, real) to avoid clutter
- Annotate each violin with kurtosis value
- Figsize: 16×10 (2×2 grid)

### Figure 2.3 — Return Distribution Comparison (Log-Log)

**Purpose**: Directly compare return tail behavior (TRADES Fig 3.4 equivalent).

**Implementation**:
- Plot the complementary CDF (1 - F(|r|)) on log-log axes for mid_return
- Lines for: real, v2, v5, v9, v2_remapped
- Real should be thick black, synthetics colored
- The slope on log-log is the tail exponent — heavier tails = flatter slope
- Annotate kurtosis for each line
- Figsize: 8×6

### Figure 2.4 — Trade Fraction Bar Chart

**Purpose**: Show how dramatically trade fraction varies across models.

**Implementation**:
- Horizontal bar chart: model name on y-axis, trade fraction on x-axis
- Add a vertical dashed line at real = 0.0477
- Color bars by distance from real (green = close, red = far)
- Sort by trade fraction
- Figsize: 10×8

---

## SECTION 3: Temporal Pattern Transferability (Predictive Score Axis) Deep Dive

### Figure 3.1 — Predictive Score Bar Chart (Enhanced)

**Purpose**: The TRADES Table 1 equivalent, visual.

**Implementation**:
- Horizontal bar chart: model name on y-axis, test MAE on x-axis (log scale)
- Market replay as a vertical dashed reference line
- Color by group (scale-accurate, scale-exploded, copula)
- Annotate each bar with the ×replay ratio
- Sort by MAE (best at top)
- Figsize: 10×8

### Figure 3.2 — What Makes Predictive Score Good? (Feature Importance)

**Purpose**: Understand WHY noclip models win on predictive score.

**Implementation**:
- For the top-3 and bottom-3 models on predictive score, plot:
  - Panel A: ACF of mid_return at lags 1-100 (temporal pattern fidelity)
  - Panel B: Cross-correlation between bid_sz and mid_return at lags -10..+10
  - Panel C: Feature-feature Pearson correlation matrix (heatmap), compared to real
- The hypothesis: noclip models preserve temporal cross-feature patterns despite scale explosion; copula models disrupt them despite correct scale
- Figsize: 16×12 (3×2: top row = best predictive, bottom row = worst predictive)

### Figure 3.3 — Predictive Score vs Return Scale

**Purpose**: Show that predictive score is NOT about matching return magnitude.

**Implementation**:
- Scatter: x = synth return std / real return std (log scale), y = predictive score (log scale)
- Label each point with model name
- Show that the relationship is NOT monotone — v2_noclip (std 44×) beats v9 (std 0.6×) by 26×
- Add annotation: "Scale matching ≠ temporal transferability"
- Figsize: 8×6

---

## SECTION 4: Multivariate Coverage (PCA Axis) Deep Dive

### Figure 4.1 — PCA Scatter with Convex Hulls (Enhanced)

**Purpose**: Expanded version of the TRADES Fig 2.

**Implementation**:
- Use the PCA projections from pca_projections.csv (or recompute)
- 4-panel figure showing the PCA space for 4 representative models:
  - Panel A: v2 (scale-accurate, 58.6% coverage)
  - Panel B: v9 (copula, 71.4%)
  - Panel C: v8 (copula no dequant, 85.2%)
  - Panel D: v2_noclip (scale-exploded, 0%)
- Each panel: real points as gray dots, synth as colored dots, convex hulls drawn
- Annotate coverage % in each panel
- Figsize: 12×12 (2×2)

### Figure 4.2 — PCA Coverage vs G1 Scatter

**Purpose**: Show PCA and G1 are partially decoupled.

**Implementation**:
- Scatter: x = PCA coverage, y = G1 pass count
- Label each point with model name
- v8 has 85% PCA but only 8/15 G1 — PCA coverage doesn't guarantee stylized fact compliance
- Figsize: 8×6

---

## SECTION 5: The Architectural Ablation

### Figure 5.1 — Ablation Impact Diagram

**Purpose**: Show the causal effect of each architectural choice on each evaluation axis.

**Implementation**:
- Matrix/heatmap where:
  - Rows: architectural changes (ε→v, FiLM→AdaLN, +EDM, +copula, +dequant, -clip)
  - Columns: evaluation metrics (G1, Predictive, PCA, kurtosis, trade_frac, ACF_lag50, vol_vol_corr)
  - Cell value: average change in metric when this architectural feature is toggled (controlling for other features as much as possible)
  - Color: green = improves, red = degrades, white = no effect
- Specific comparisons to compute:
  - ε→v: compare (v2 vs v4), (v9 vs v9_b)
  - FiLM→AdaLN: compare (v4 vs v5), (v8 vs v8_b)
  - +EDM: compare (v5 vs v6/v7/v7_b)
  - +copula: compare (v2 vs v8)
  - +dequant: compare (v8 vs v9)
  - -clip: compare (v2 vs v2_noclip), (v3_e9 vs v3_e9_noclip)
- Figsize: 12×6

### Figure 5.2 — The x0_clip Effect (Paired Comparison)

**Purpose**: The single most surprising finding — removing clipping destroys G1 but dramatically improves predictive score.

**Implementation**:
- Paired bar chart: for each clip/noclip pair (v2/v2_noclip, v3_e9/v3_e9_noclip, v3p5/v3p5_noclip), show:
  - Left bar: G1 pass count (clip vs noclip)
  - Right bar: Predictive ×replay (clip vs noclip)
- Connected by lines to show the paired change
- Annotate the direction: "clip improves G1, noclip improves predictive"
- Figsize: 10×6

---

## SECTION 6: The Copula Story

### Figure 6.1 — z-Space Calibration Before/After Dequantization

**Purpose**: Show the copula failure (v8 z-collapse) and the dequantization fix (v9).

**Implementation**:
- 2 panels for mid_return feature:
  - Panel A (v8): histogram of z-values. Show the collapsed spike at z≈-2.55
  - Panel B (v9): histogram of z-values after dequantization. Should show wider spread
- Overlay the N(0,1) reference PDF in both panels
- Read z-space data from z_space_stats.csv or recompute from the copula transform + synth parquets
- Figsize: 12×5

### Figure 6.2 — The Zero-Inflation Problem

**Purpose**: Show how the 89% point mass at mid_return=0 breaks the copula CDF.

**Implementation**:
- 3 panels:
  - Panel A: Empirical CDF of real mid_return (show the massive jump at 0)
  - Panel B: The probit-transformed z-values for v8 (show the collapsed constant)
  - Panel C: The probit-transformed z-values for v9 with dequantization (show the spread)
- Figsize: 14×4

### Figure 6.3 — Copula: What It Fixed vs What It Broke

**Purpose**: Side-by-side comparison of v2 vs v9 on every metric category.

**Implementation**:
- Grouped bar chart with metric categories on x-axis, two bars per category (v2, v9)
- Categories: G1 count, kurtosis ratio, trade_frac ratio, ACF lag-50 ratio, PCA%, Predictive×
- Normalize each metric so that "perfect=1" (ratio to real for distributional metrics, 1/×replay for predictive)
- Show that v9 wins on trade_frac and ACF but loses on predictive
- Figsize: 12×6

---

## SECTION 7: Volume-Volatility and Temporal Structure

### Figure 7.1 — Volume-Volatility Correlation Distribution (Enhanced)

**Purpose**: Publication version of the TRADES metric.

**Implementation**:
- KDE plot of per-tape volume-volatility correlations
- Show: real (thick black filled), v2 (blue), v3p5 (orange), v7_b (green)
- Only show models that had >50 usable tapes
- Add vertical dashed line at real mean (+0.341)
- Note: copula models (v8, v9) have 0 usable tapes — add text annotation explaining why
- Figsize: 8×5

### Figure 7.2 — ACF Comparison Grid

**Purpose**: Show temporal structure across all features for key models.

**Implementation**:
- Grid of ACF plots: rows = features (mid_return, |mid_return|, spread, bid_sz, trade_sign), cols = models (real, v2, v7_b, v9)
- Each cell: ACF at lags 1, 5, 10, 50, 100 as a line plot
- Color cells by how close to real (green border = good match, red = bad)
- Read from temporal_acf.csv
- Figsize: 16×12

### Figure 7.3 — Trade-Sign Persistence Across Models

**Purpose**: Show trade-sign ACF (a key stylized fact) across all models.

**Implementation**:
- Line plot: x = lag (1-100), y = trade-sign ACF
- One line per model (subset: real, v2, v4, v7_b, v9)
- Real as thick black
- Annotate lag-1 ACF value for each model
- Figsize: 8×5

---

## SECTION 8: The Agent-Ranking Mechanism

### Figure 8.1 — Why ρ_diff Fails: Agent Similarity

**Purpose**: Show that A1 and A2 are economically indistinguishable.

**Implementation**:
- Box/violin plot of per-day PnL for each agent (A0, A1, A2, A3) on truth data
- Show that A1 and A2 distributions overlap almost completely
- Add effect size annotation (Cohen's d between A1 and A2)
- Read from agent_decomposition.csv
- Figsize: 10×5

### Figure 8.2 — ρ_diff vs Everything Else

**Purpose**: Show ρ_diff is uncorrelated with every distributional quality metric.

**Implementation**:
- 4-panel scatter: ρ_diff vs (G1, Predictive, PCA, kurtosis)
- Only include models that have ρ_diff values
- Show that there's no relationship
- Include Spearman ρ annotation in each panel
- Figsize: 14×4

### Figure 8.3 — Fill-Rate Mechanism Decomposition

**Purpose**: Show what ACTUALLY differs between A1 and A2 (fill rate, not PnL).

**Implementation**:
- Grouped bar chart: models on x-axis, A2-A1 z-score delta on y-axis
- Separate bars for: terminal_pnl, fill_rate, inventory_var
- Show that fill_rate is the consistent positive delta, not PnL
- Read from a2_minus_a1_deltas.csv
- Figsize: 12×5

---

## SECTION 9: Mid-Price Traces and Visual Realism

### Figure 9.1 — Selected Mid-Price Trace Comparison (4-Panel)

**Purpose**: Visual realism comparison for the report.

**Implementation**:
- 4 panels: v2, v9, v2_noclip, v2_remapped
- Each panel: 5 synth traces (thin colored) + 1 real trace (thick black)
- Normalize to start at 1.0
- Same y-axis scale across all panels for comparison
- Figsize: 14×8 (2×2)
- Recreate from parquets (don't reuse the PDFs from metric 4 — make consistent style)

---

## SECTION 10: Summary Figures for the Abstract/Introduction

### Figure 10.1 — The One-Figure Summary

**Purpose**: If the reader sees only one figure, it should be this. Shows the three-axis tension with the key insight.

**Implementation**:
- 3 horizontal panels stacked:
  - Top: Models ranked by G1 (bar chart, left to right = best to worst)
  - Middle: Same models ranked by Predictive Score
  - Bottom: Same models ranked by PCA Coverage
- Connect each model across the three panels with lines (like a bump chart / rank plot)
- The crossing lines visually show the anti-correlation
- Highlight the key models (v2, v9, v3_e9_noclip) with thick lines
- Figsize: 12×10

---

## Additional Analysis Tables (CSV output, not figures)

### Table A — Architecture Decision Matrix

For each architectural choice, compute the average effect on each metric:

```
| Architectural choice | Avg ΔG1 | Avg ΔPredictive | Avg ΔPCA | Avg ΔKurtosis | Avg ΔTradeFrac |
```

Computed by averaging across all matched pairs where only that one choice differs.

### Table B — Per-Check Difficulty Ranking

For each G1 check, count how many models pass. Rank by difficulty. This tells the reader which stylized facts are easiest/hardest to satisfy:

```
| G1 Check | Pass Rate (n/20) | Hardest models | Easiest models |
```

### Table C — Metric Correlation Matrix

Compute Spearman ρ between ALL evaluation metrics across all models:

```
| | G1 | Predict | PCA | kurt | trade_frac | ACF_lag50 | vol_vol | OFI_slope |
```

This is the quantitative backing for the "anti-correlation" claim.

---

## Technical Requirements

- Python 3.11, matplotlib, seaborn, numpy, scipy, pandas, polars
- Optional: python-ternary (pip install python-ternary) for Figure 1.1
- All figures: PDF + PNG at 300 DPI
- Consistent color scheme across all figures (define a MODEL_COLORS dict at the top)
- Consistent font sizes: title=14, axis_labels=12, tick_labels=10, annotations=9
- All outputs to `results/report_figures/`
- Print a summary of all files written at the end
- Handle missing data gracefully (some models have NaN for some metrics — skip or annotate)

## Runtime

Should be < 10 minutes total. All data is pre-computed; this script only reads CSVs and makes plots. No GPU, no model loading, no parquet loading except for Figures 6.2, 9.1 (which load a few tapes each).