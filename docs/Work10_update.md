# Work 10 — Report Build Status & Cluster Action Plan

**Updated:** 2026-05-10 (during overnight TSLA/SPY runs)
**Inputs:** [Work10_Report.md](Work10_Report.md) (the brief)
**Outputs so far:** [final_report.tex](final_report.tex), [scripts/80_report_figures.py](../scripts/80_report_figures.py), [results/report_figures/](../results/report_figures)

This doc is the running ledger of (a) what's already produced locally, (b) what's still missing and where it lives, (c) the prioritized cluster work needed to finish the report. Update in place as items move between buckets.

---

## 1. Done locally

### 1.1 Document skeleton
- [final_report.tex](final_report.tex) — full sectioned skeleton, no body prose. Title reframed to lead with the three-axis tension. All §/§§/§§§ headers in place; figure & table placeholders inline as comments. Compiles as-is.

### 1.2 Figure & table suite — `scripts/80_report_figures.py`
One-shot driver. **Run:** `python scripts/80_report_figures.py` (≈30 s, no GPU). Reads pre-computed CSVs/JSONs from `results/*` and writes to `results/report_figures/`. Already executed against current local data: **55 files written** (25 figures × {PDF, PNG} + 4 tables × {CSV, TeX} + 1 trace index).

**Per-section production status** (mapping to Work10 spec):

| § | Figure | Backing data (local file) | Output | Notes |
|---|---|---|---|---|
| 1.1 | Three-axis scatter (hero) | multi_axis_comparison.csv + predictive_score.csv + pca_coverage.csv | `fig_1_1_three_axis_scatter.{pdf,png}` | 2D scatter + size-encoded PCA (no `python-ternary` dep) |
| 1.2 | Pairwise anti-correlation | same | `fig_1_2_pairwise_anticorrelation` | Spearman ρ annotated |
| 1.3 | Multi-axis comparison table | multi_axis_comparison.csv | `tab_1_3_multi_axis.{tex,csv}` | REAL row at top |
| 2.1 | G1 pass/fail heatmap | G1_comparison_all_models.csv | `fig_2_1_g1_heatmap` | sorted by difficulty + by strength |
| 2.2 | Marginal kurtosis | deep_diagnostic/per_feature_stats.csv | `fig_2_2_marginal_kurtosis` | 4-panel for mid_return / spread / bid_sz / trade_sz |
| 2.3 | Return tail (z-σ thresholds) | per_feature_stats.csv (frac_gt_{2,3,4}sigma) | `fig_2_3_return_tail` | log-y; uses summary stats |
| 2.4 | Trade-fraction bar | multi_axis_comparison.csv (B4_trade_fraction) | `fig_2_4_trade_fraction` | colored by distance from real |
| 3.1 | Predictive bar (log) | predictive_score.csv | `fig_3_1_predictive_score` | sorted, replay reference line |
| 3.2 | ACF top-3 vs bottom-3 | predictive_score.csv + temporal_acf.csv | `fig_3_2_acf_top_bottom` | mid_return ACF for top vs worst |
| 3.3 | Predictive vs scale | multi_axis_comparison.csv (B1_synth_std) + pred | `fig_3_3_predictive_vs_scale` | log-log scatter |
| 4.1 | PCA coverage | trades_metrics_all/pca_coverage.csv | `fig_4_1_pca_coverage` | bar chart fallback (no per-model projections in local archive — see §3 P1.D) |
| 4.2 | PCA vs G1 | matrix + pca_coverage | `fig_4_2_pca_vs_g1` | shows decoupling |
| 5.1 | Ablation heatmap | matrix + pred + pca | `fig_5_1_ablation_heatmap` | per-axis effect on each metric |
| 5.2 | x0_clip paired effect | matrix + pred | `fig_5_2_x0_clip_effect` | clip vs noclip on G1 + Pred |
| 6.1 | z-space calibration | deep_diagnostic/z_space_stats.csv | `fig_6_1_z_space` | mid_return z-stats missing → falls back to bid_sz (see §3 P1.E) |
| 6.2 | Zero-inflation across models | per_feature_stats.csv (frac_zero) | `fig_6_2_zero_inflation` | bar replacement for the parquet-CDF figure |
| 6.3 | v2 vs v9 grouped bars | matrix + pred + pca | `fig_6_3_v2_vs_v9` | normalised (metric / real) |
| 7.1 | Vol-vol KDE | trades_metrics_all/vol_volatility_correlation.csv | `fig_7_1_vol_vol_kde` | per-tape KDE, real-mean reference |
| 7.2 | ACF grid (features × models) | temporal_acf.csv | `fig_7_2_acf_grid` | 3 features × 4 models |
| 7.3 | Trade-sign ACF | temporal_acf.csv | `fig_7_3_trade_sign_acf` | line plot, real reference |
| 8.1 | Agent PnL on truth | ranking_mechanism/agent_decomposition.csv | `fig_8_1_agent_pnl_overlap` | mean ± std, A1/A2 overlap obvious |
| 8.2 | ρ_diff vs everything | matrix + pred + pca | `fig_8_2_rdiff_vs_everything` | 4-panel scatter + Spearman annotations |
| 8.3 | A2 − A1 z-score deltas | ranking_mechanism/a2_minus_a1_deltas.csv | `fig_8_3_a2_minus_a1` | fill_rate is the consistent positive delta |
| 9.1 | Mid-price traces | (index file → 20 pre-rendered PDFs in `trades_metrics_all/`) | `fig_9_1_midprice_traces_index.txt` | placeholder — see §3 P2.A for re-render |
| 10.1 | Rank-bump chart | matrix + pred + pca | `fig_10_1_rank_bump` | crossing lines visualise anti-correlation |
| Tab A | Decision matrix | as fig 5.1 | `tab_A_decision_matrix.{csv,tex}` | mean Δ per axis × metric |
| Tab B | G1 difficulty ranking | G1_comparison_all_models.csv | `tab_B_g1_difficulty.{csv,tex}` | with "models that fail" column |
| Tab C | Metric correlation matrix | matrix + pred + pca | `tab_C_metric_correlations.{csv,tex}` | Spearman across 20 generators |

### 1.3 What's effectively final right now
With current local data, §§1, 2, 3, 5, 7, 8, 10 are figure-complete. §4 (PCA), §6 (copula), §9 (traces) are partially complete but recoverable with one of the cluster jobs in §3.

---

## 2. Local data inventory (what was already pulled from cluster)

```
results/_archive/multi_axis_comparison.{csv,md}        ← 22 rows × 18 cols, all models
results/_archive/G1_comparison_all_models.csv          ← 16 checks × 20 models (pass + value)
results/_archive/<model>/validation/                   ← B*, C*, D*, F1b for all 20 models
results/_archive/<model>/hypothesis/bootstrap.json     ← v8, v9, v9_b, v10_cascade, v11, v8_b, v2_remapped
results/predictive_score_all/predictive_score.csv      ← all 20 models
results/trades_metrics_all/{pca_coverage, vol_vol, leverage_effect, midprice_traces_*}
results/deep_diagnostic/{per_feature_stats, correlation_comparison, temporal_acf,
                         window_level_stats, z_space_stats, regime_conditional,
                         ofi_regression}.csv          ← partial coverage (see gaps in §3)
results/ranking_mechanism/{agent_decomposition, a2_minus_a1_deltas, conditional_returns}.csv
                                                       ← truth, real, v2/v8/v9/v9_b only
```

**Not pulled yet** (would need a small rsync from cluster):
- Per-model PCA projections (only summary `pca_coverage.csv` is local)
- Per-checkpoint training/val loss CSVs (Lightning logs)
- Per-checkpoint regime embedding weights

---

## 3. Cluster action plan — prioritized

Each item lists: *what*, *why* (which report section it feeds), *how* (exact command sketch), *runtime*, *status*.

### P0 — Blocking: must complete before report is final

#### P0.A — TSLA + SPY trainings (already in flight)
- **Why:** §7 cross-asset validation — central to the "is the tension structural?" question.
- **Status:** Running as of 2026-05-10. TSLA v2 + v5 on favara (mit_normal_gpu, ~25k steps in @ 17:00); SPY v2 + v5 on esampi (mit_normal_gpu, started ~17:00). Orchestrated by [scripts/utils/run_cross_ticker_v2_v5.sh](../scripts/utils/run_cross_ticker_v2_v5.sh) and the esampi-side derivative.
- **Outputs to pull on completion:**
  ```
  results/_archive/{tsla,spy}_{v2,v5}/validation/         (G1, B*, C*, D* CSVs)
  results/trades_metrics_{tsla,spy}/{pca_coverage, vol_vol, ...}.csv
  results/predictive_score_{tsla,spy}/predictive_score.csv
  ```
- **Report sections that consume these:** §7.2 (TSLA), §7.3 (SPY), §7.4 (cross-asset findings + ternary), §1.1 hero figure (could add INTC×{TSLA,SPY} points).

#### P0.B — Deep diagnostic on v5 and v9 (currently missing)
- **Why:** v5 and v9 are two of the three "protagonists" for §4.1 / §5 / §6. Without their full per-feature stats, §5.2 (Table 5.2) and §6.4 are thin.
- **Where the gap is:** `results/deep_diagnostic/per_feature_stats.csv` currently has v2, v7_b, v8, v8_b, real. **Missing v5, v9, v9_b.**
- **Command (run on cluster, login node CPU is fine):**
  ```bash
  python scripts/60_deep_diagnostic.py \
      --real-dir data/processed/val \
      --synth-dirs $HOME/orcd/scratch/diffmm/synthetic/intc_v5/INTC \
                   $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
                   $HOME/orcd/scratch/diffmm/synthetic/intc_v9_b/INTC \
      --labels v5 v9 v9_b \
      --out results/deep_diagnostic_v5_v9 \
      --max-tapes 100
  ```
- **Runtime:** ~10 min.
- **After:** rsync `results/deep_diagnostic_v5_v9/` back, then merge into the main `results/deep_diagnostic/per_feature_stats.csv` (plain `pd.concat` + dedup by (model, feature)).

#### P0.C — Conditional-on-trade return stats for all 20 models
- **Why:** §3.3 ("the zero-inflation pathology") and Table 5.1 want `frac_zero`, `cond_kurt`, `cond_std` for every model so the comparison table is complete.
- **Where the gap is:** `results/ranking_mechanism/conditional_returns.csv` has only real, v2, v8, v9, v9_b.
- **Command:** extend [scripts/61_ranking_mechanism.py](../scripts/61_ranking_mechanism.py) (or write a thin loop) to iterate over all 20 synth dirs:
  ```bash
  python scripts/61_ranking_mechanism.py \
      --real-dir data/processed/val \
      --synth-dirs $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
                   $HOME/orcd/scratch/diffmm/synthetic_v2_noclip_stitched/INTC \
                   ...   # all 20
      --labels v2 v2_noclip ... \
      --out results/ranking_mechanism_full \
      --max-files 50
  ```
- **Runtime:** ~10 min.
- **After:** drop in for the existing `conditional_returns.csv` — `fig_6_2_zero_inflation` and Table 5.1 will pick it up automatically on next run.

#### P0.D — Per-regime G1 for v2 / v5 / v9
- **Why:** §4.2.2 ("Conditioning: FiLM vs AdaLN") needs a number for "does regime conditioning actually work?" Per-regime G1 pass-counts answer that directly.
- **Approach:** run `scripts/50_validate_generator.py` once per (regime, model), pointing `--synth-dir` at the regime sub-folder of the synth tree:
  ```bash
  for regime in base high_vol toxic thin; do
    for model in v2 v5 v9; do
      OUT=results/_archive/${model}/validation_${regime}
      SYNTH=$HOME/orcd/scratch/diffmm/synthetic_${model}_stitched/INTC
      mkdir -p "$OUT"
      python scripts/50_validate_generator.py \
          --parts B C D \
          --real-dir data/processed/val \
          --synth-dir "$SYNTH/$regime" \
          --ticker INTC --out-dir "$OUT" \
          --n-synth-per-regime 60
    done
  done
  ```
  *(Verify the synth tree structure first — the script may need `--regime-filter` instead.)*
- **Runtime:** ~20 min.
- **Report section:** §4.2.2, §5.6 ("ablation: what drives what" gets a regime row).

### P1 — Highly valuable, run when GPU/login slots open

#### P1.A — Pull PCA projections to enable a true Fig 4.1
- **Why:** the Work10 spec asks for a 4-panel PCA scatter with convex hulls (real points + synth points + hull lines). The figure script currently substitutes a coverage bar.
- **Source:** `results/trades_metrics_all/pca_projections.csv` — produced by `scripts/62_trades_metrics.py` *if you pass `--save-projections`*. Check whether the existing run did so.
- **If the file isn't there:** re-run script 62 once with the projections flag enabled; ~5 min. Then add a `fig_4_1_pca_panels()` function that consumes it and replace the current bar.

#### P1.B — Z-space stats for `mid_return` on v8 / v9
- **Why:** §6.1 / §6.2 want to show the v8 z-collapse on `mid_return` specifically (not bid_sz). Currently the figure falls back to bid_sz.
- **Where:** the deep-diagnostic z-space block likely skipped `mid_return` because the copula transform output for v8 is degenerate. Need to force it to compute on `mid_return` (with z-mean / z-std / z-kurtosis) and append rows.
- **Command:** small extension to [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) or a one-off script that loads the v8/v9 copula z-arrays from the synth parquets and writes the same schema as `z_space_stats.csv`.
- **Runtime:** 10 min code + 5 min compute.

#### P1.C — Training & validation loss curves for v2 / v5 / v9
- **Why:** §3.3 of methods + Appendix B benefit from showing convergence behaviour.
- **Source:** Lightning's CSVLogger in `~/orcd/scratch/diffmm/checkpoints/<run>/lightning_logs/version_*/metrics.csv`.
- **Command:** quick rsync of just those CSVs (kilobytes), then a small standalone plotting script.

#### P1.D — Real-data standalone stylized-fact panel (§3.5)
- **Why:** the report wants a "real INTC only" panel showing what generators are trying to reproduce — independent of the comparison figures.
- **Compute locally if `data/processed/val/INTC_*.parquet` is rsynced** (≈300 MB). Otherwise wrap as a cluster script.
- **Outputs:** return histogram, |return| ACF, return ACF, trade-sign ACF, spread distribution, intraday volume.
- **Runtime:** 3 min.

### P2 — Nice-to-have, only if time permits

| Item | Why | Runtime | Where |
|---|---|---|---|
| Re-render Fig 9.1 with consistent style across 4 selected models | currently using pre-existing PDFs from script 62 | 5 min | local if parquets pulled, else cluster |
| Guidance-weight sweep on v2 (w ∈ {0,0.5,1,2,4,8}) | §4.2.2 conditioning sensitivity, §8.4 future work | 2 h | cluster GPU |
| Per-timestep denoising quality for v2/v5/v9 | §3.2.1 (ε vs v intuition), §4.6 | 30 min | cluster GPU |
| Regime-embedding t-SNE | §3.2.2 conditioning | 5 min × N | cluster (needs ckpt load) |
| LSTM-size robustness sweep on predictive score | §8.3 limitations | 1 h | cluster |
| Spread-conditional dynamics (mean bid_sz / ask_sz / |return| | spread > 1 tick) | §4.2 distributional realism | 5 min | local with parquets |
| Event-type transition matrix | §4.3 temporal structure | 3 min | local with parquets |
| Intraday volume U-shape | §2.5 stylized facts | 3 min | local with parquets |

### P3 — Out of scope for this report
- Autoregressive baseline (TRADES-style sequence model) — flagged as future work in §8.4.
- Mixture-of-experts (ε-pred bulk + v-pred head for tails) — same.
- New holdout periods longer than 20 days.

---

## 4. Sync plan

### 4.1 Code → cluster
After editing/adding any of:
- `scripts/60_deep_diagnostic.py`, `scripts/61_ranking_mechanism.py`, `scripts/50_validate_generator.py` extensions
- New helper scripts (`scripts/utils/run_p0_full.sh`)

push with the same rsync pattern as before:
```bash
cd ~/Desktop/diff-market-making && rsync -avzR \
    scripts/<modified files> \
    favara@orcd-login.mit.edu:diff_mm/
```

### 4.2 Results → Mac (after each cluster batch finishes)
```bash
rsync -avz \
    favara@orcd-login.mit.edu:diff_mm/results/_archive/ \
    ~/Desktop/diff-market-making/results/_archive/

rsync -avz \
    favara@orcd-login.mit.edu:diff_mm/results/{deep_diagnostic_v5_v9,ranking_mechanism_full,trades_metrics_{tsla,spy},predictive_score_{tsla,spy}}/ \
    ~/Desktop/diff-market-making/results/

# For SPY-only artifacts produced under esampi:
rsync -avz \
    esampi@orcd-login.mit.edu:diff_mm/results/_archive/spy_{v2,v5}/ \
    ~/Desktop/diff-market-making/results/_archive/
```

### 4.3 Re-build figure suite locally
```bash
python scripts/80_report_figures.py
```
Idempotent. Picks up whatever new CSVs are in `results/`.

---

## 5. Tomorrow-morning checklist

1. **Check overnight runs** — `squeue -u favara` on cluster, `squeue -u esampi` on cluster. Look for `=== ALL DONE ===` in the orchestrator logs.
2. **Pull cross-asset results** (P0.A) via the rsync block above.
3. **Submit P0.B + P0.C + P0.D** as a single batch (they're CPU-light and small).
4. **Rsync the new CSVs back**, drop the v5/v9 deep-diagnostic rows into `per_feature_stats.csv`, drop the all-models conditional_returns into `ranking_mechanism/conditional_returns.csv`.
5. **Re-run `scripts/80_report_figures.py`** — should now have:
   - Hero figure (1.1) including TSLA/SPY data points
   - PCA panels (4.1) replacing the bar fallback (if P1.A done)
   - z-space (6.1) showing mid_return collapse explicitly (if P1.B done)
   - Tighter Table 5.1 with all 20 conditional-return columns
6. **Start filling prose** in [final_report.tex](final_report.tex) — §§ in this order: 4 (results, the meat) → 5 (copula case study) → 6 (agents) → 7 (cross-asset) → 2 (data, deep but slower-paced) → 3 (methods) → 1 (intro) → 0 (executive summary, written last).
