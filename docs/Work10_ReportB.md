# Cluster-Figures Deep Inspection (results_cluster/)

**Goal:** identify which of the cluster artifacts now sitting in
`results_cluster/` are genuine TRADES-style visual content
(real-vs-synth comparisons with structural depth) versus content that
would communicate at least as well as a table. Drives the next round
of figure wiring into `docs/final_report.tex`.

**Inventory headcount:** 963 PDFs, 645 CSVs across 57 top-level dirs.

---

## 1. Top-level directory map

| Dir | Producer | PDFs | What it is |
|---|---|---|---|
| `_archive/<model>/validation*/` | script 50 (validate_generator) | 745 | per-model B/C/D suite — the **canonical real-vs-synth visual battery** |
| `_archive/<model>/hypothesis/` | script 51 (hypothesis_test) | 8 | "figure7" — bootstrap test ρ_diff vs ρ_hist |
| `_archive/<model>/synthetic_backtests/` | script 70 (backtests) | 0 (CSVs only) | per-agent PnL JSON + metrics CSV |
| `_archive/<model>/holdout/` | script 70 | 0 (CSVs only) | real-holdout agent PnL reference |
| `trades_metrics_all/` | script 62 (trades_metrics) | 23 | mid-price traces (per model), PCA coverage, leverage, vol-vol — **TRADES Fig 6/8 style** |
| `trades_metrics_tsla/` | script 62 | 5 | same suite, TSLA |
| `trades_style/` | scripts 86 + 87 | 2 | intraday LOB volume (**TRADES Fig 4 style**) + return-distribution full |
| `microstructure_diagnostics/` | script 83 | 3 | spread-conditional dynamics, event transitions, intraday volume |
| `validation/INTC_<model>/` | script 50 (older runs) | 190 | duplicate B/C/D/E/F per model — kept the `_archive/` copies are newer |
| `predictive_score_all/` | script 63 | 1 | LSTM MAE bar across 20 models |
| `predictive_lstm_h{32,64,128}/` | script 63 with `--hidden-dim` | 3 | LSTM-size robustness sweep |
| `predictive_score_tsla/` | script 63 | 1 | TSLA predictive score |
| `report_figures/` | script 80 (orchestrator) | 2 | currently outdated; script 80 should be re-run locally |
| `ranking_mechanism*/` | script 61 | 0 (CSVs only) | agent decomposition + a2_minus_a1_deltas + conditional_returns |
| `metrics/`, `default/`, `intc_v*/`, `lightning_logs/` | training (script 11) | 0 | training-time logs only |
| `agent_params/` | manual / script 70 | 0 | A0–A3 calibrated parameters |

The two stand-out finds in the inventory:

1. **`trades_metrics_all/pca_projections.csv` exists for INTC** — earlier
   I'd told the user this was blocked on a cluster pull. It is not. The
   PCA 4-panel scatter+hull upgrade in `scripts/80_report_figures.py`
   will trigger as soon as the file lands at
   `results/trades_metrics_all/pca_projections.csv`.
2. **TSLA validation B/C/D suite is complete in `_archive/tsla_v{2,5}/`.**
   12 PDFs per ticker × 2 architectures, ready to surface in §7.

---

## 2. The validation B/C/D suite — TRADES-style by construction

This is the meat. Every trained model has, in
`_archive/<model>/validation/`, the same 12-figure battery:

| Code | Title | What it shows | TRADES-paper analogue |
|---|---|---|---|
| **B1** | Return distribution | 4-panel: density (log-y) with kurt annotated, QQ plot, kurt bar, Wasserstein-1 vs inter-day band, KS stat in header | TRADES Fig 3 |
| B2 | Spread distribution | Real vs synth spread (cents) | Fig 5(b) (LOBSTER paper) |
| B3 | Size distribution | trade_sz + book_sz densities | TRADES App. C |
| B4 | Event composition | event-type pie/bar | --- |
| **C1** | ACF of |return| | Real ±2σ band + synth overlay + log-log inset with power-law β | TRADES Fig 5(a) — **the** volatility-clustering panel |
| **C2** | ACF of trade_sign | Order-flow persistence, log-log decay annotated | TRADES Fig 5(c) |
| C3 | ACF of return | "Should be near zero" — efficient markets check | TRADES Fig 5(a) sibling |
| **C4** | RV signature | Realized variance vs sampling frequency | TRADES Fig 6 (microstructure noise signature) |
| C5 | ACF of OFI | OFI persistence | --- |
| **D1** | Conditional return on imbalance | 3-panel at Δ=1, 10, 50; the OFI→return signal with regression β and R² | TRADES Fig 7 — **the** information-content panel |
| D2 | Joint spread × imbalance | 2-D heatmap, real vs synth | --- |
| D3 | Conditional moments | Per-quantile mean/std of returns | --- |

**Verified content (sampled from `_archive/v2/validation/`):**
- `B1_return_distribution.pdf` is exactly the TRADES Fig 3 layout —
  real kurt = **755.13** (matches body), synth kurt = **121.88**
  (matches body's "v2 kurtosis 122"), KS = 0.5775, W1 pass: True with
  inter-day band [0, 4.53e-06]. Rich, four-panel, immediately legible.
- `C1_acf_abs_return.pdf` is also TRADES-paper-perfect: real ACF in
  blue with a ±2σ inter-day shaded band, synth ACF in orange overlaid,
  with a log-log decay inset showing the power-law β (real **0.21**,
  v2 synth **0.66**) — i.e.\ synth has too-strong clustering relative
  to real, the exact failure mode the body's §5.2 alludes to.
- `D1_conditional_return_imbalance.pdf` is again TRADES Fig 7 layout:
  three horizon panels (Δ=1, 10, 50) with bin-centered conditional
  return curves; real has the canonical S-curve (β grows from 4.25e-06
  at Δ=1 to 1.11e-04 at Δ=50, R² grows from 0.012 to 0.172), v2 synth
  is essentially flat (β ≈ 10× smaller, R² 10× smaller) — the OFI
  signal is real but v2 doesn't reproduce it. This is the **mechanical
  explanation** behind the body's claim that the A2 (OFI-aware) agent
  doesn't outperform A1 on most generators.

**Per-regime breakdown** is available for v2, v5, v9 only:
`validation_{base,high_vol,toxic,thin}/` × 12 figures each = 48 PDFs
per protagonist. This means we can show, for the three protagonists,
that the C1 / D1 mechanisms behave differently across regimes. A
genuinely novel exhibit that the body does not currently surface.

**Recommendation for the report:**
- **High-value to wire (the heart of the visual battery):**
  - `B1_return_distribution.pdf` for **v2/v5/v9** at §5.2 (the kurtosis
    story has been told in prose but never shown side-by-side per model).
    Three small subfigures or a 1×3 grid.
  - `C1_acf_abs_return.pdf` for v2/v5/v9 at §5.2 (volatility clustering
    is in the Cont checklist but never visually shown in our report).
  - `D1_conditional_return_imbalance.pdf` for v2 + v9 at §6 (this is
    the OFI signal that A2 trades on; showing real has the structure
    and synth doesn't is the missing visual for the
    A2≈A1 finding in §6.4).
- **Medium-value, optional:**
  - `C2_acf_trade_sign.pdf` for v2/v5/v9 — order-flow persistence is
    stylized fact (iv) in §2.4; would be a natural §5.2 add.
  - `C4_rv_signature.pdf` for one protagonist — the RV-signature
    microstructure-noise picture is a TRADES standard.
- **Low-value (could be table or skipped):**
  - B2/B3/B4 (size and event-composition distributions) — bar charts.
  - C3/C5 ACFs (less central to our story).
  - D2 joint heatmap — pretty, but the body doesn't reference the
    joint spread×imbalance claim anywhere.
  - D3 conditional moments — emits a categorical label per the
    validator (we already drop it from G1 heatmaps).
- **Per-regime exhibit (genuinely novel, not in plan):** a 4×3 grid of
  `C1_acf_abs_return` across (base, high_vol, toxic, thin) × (v2, v5,
  v9) would show whether the regime conditioning is doing anything for
  the vol-clustering structure. Goes in App. or as an extra §5
  subfigure.

---

## 3. The TRADES-paper-style standalone figures

| Path | Status | Verdict |
|---|---|---|
| `trades_style/fig_intraday_lob_volume.pdf` | already wired in §8.3 (Limitations) | ✓ keep |
| `trades_style/fig_return_distribution_full.pdf` | broken (QQ flat, kurt numbers don't match body) | ✗ skip |
| `trades_metrics_all/midprice_traces_v{2,5,9}.pdf` + 18 others | clean, TRADES Fig 8 layout | **wire v2/v5/v9 + one noclip at §5.1 or §5.7** |
| `trades_metrics_all/leverage_effect.pdf` | exists, not yet inspected | inspect before wiring |
| `trades_metrics_all/vol_volatility_correlation.pdf` | exists, not yet inspected | inspect before wiring |
| `trades_metrics_all/pca_coverage.pdf` | bar-chart fallback (no pca_projections in script 80 input dir locally) | will auto-upgrade once we copy `pca_projections.csv` over from cluster |
| `trades_metrics_tsla/midprice_traces_v{2,5}.pdf` | TSLA equivalent | wire in §7 |
| `microstructure_diagnostics/fig_p2_1_spread_conditional.pdf` | three-bar comparison of real-vs-v2/v5/v9 on spread-conditional moments — substantively interesting but bar-chart format | **table** |
| `microstructure_diagnostics/fig_p2_2_event_transitions.pdf` | stacked-bar transition matrix per model; **shows v9 reproduces real's event-type transitions while v2 and v5 don't** — substantive, but a 4×4 transition matrix per model is a table | **table** (or one figure with the v2/v5/v9/real comparison overlaid) |
| `microstructure_diagnostics/fig_p2_3_intraday_volume.pdf` | 15-min bucket volume real vs synth | overlaps `fig_intraday_lob_volume`; skip |

**Sampled content for verification:**

- `midprice_traces_v2.pdf` is a two-panel TRADES Fig 8 layout: panel A
  overlays real's 10K-event slice from val/INTC_2025-04-09 (mid swings
  from 1.000 to 1.005, +50bp mid-day) against two synth lines; panel B
  is a synth-only zoom autoscaled to v2 (synth #1 ends at +0.0008 ≈
  8bp — v2 cannot reach real's session-scale move at all). This is the
  single cleanest visual demonstration of the body's repeated claim
  that v2's std is too narrow.
- `microstructure_diagnostics/fig_p2_2_event_transitions.pdf` shows
  real's event-type transition matrix as ~95% quote→quote, ~3%
  trade→quote, ~2% quote→trade, ~1% trade→trade. v9's stack is
  visually indistinguishable from real (the copula generator preserves
  this!). v2 and v5 emit massively wrong transitions — ~40% of events
  are trade→trade and trade→quote. **This is a substantive finding
  the body does not currently mention** and is a strong supporting
  exhibit for §5.7 (v9 is the only protagonist that gets the
  trade-fraction AND the event-transition structure right).

---

## 4. The E/F training diagnostics

In `_archive/<model>/validation/`, models also have:

- **E1** training_curves — sampled checkpoint v9 is empty/broken (just
  the v1/v2 best-val-loss bars, panel (a) blank). Would need a re-run
  of script 50 with Lightning metrics.csv pulled. **Skip unless
  re-generated.**
- E2 per_timestep_quality, E3 x0_clip_activation, E4 noise_schedule —
  19 versions each across models, not inspected
- F1 regime_embedding_pca, F1b embedding_cosine, F2 FiLM_modulation,
  F3 attention_maps, F4 guidance_sweep, F5 per_feature_loss — 19
  versions each

The F-family supports the methodology section (FiLM modulation
collapse finding referenced in §3.2.2) but the body cites the finding
verbally without showing it. **F2_film_modulation could be a 1-figure
add in §3.2.2 or in an appendix** if a representative checkpoint shows
the saturation clearly.

---

## 5. What's "could-be-a-table" (the user's stated dislike)

The body currently includes two figures the user has flagged as
table-replaceable:

- `fig_3_3_predictive_vs_scale.pdf` — scatter of predictive score vs
  scale ratio. Substantively important (anti-correlation finding) but
  the same information could be 3 rows of a table: copula family at
  $\sigma_\text{synth}/\sigma_\text{real}\approx 0.6$ with predictive
  $> 100\times$, scale-accurate at $\sim 1.0$ with predictive
  $10$–$36\times$, noclip at $\sim 40$ with predictive $4$–$14\times$.
- `fig_1_2_pairwise_anticorrelation.pdf` — three pairwise scatter
  panels. The body already extracts the load-bearing numbers
  (ρ = -0.17, +0.26, +0.57) into `tab:correlation_subset`. The figure
  shows the same numbers visually.

By the same standard, **also table-replaceable:**

- `predictive_score_all/predictive_score.pdf` — 20-model bar chart.
  Body's prose already orders the headline cells.
- Most `B*` distribution figures (B2, B3, B4) — single-panel bars.
- `_archive/<model>/hypothesis/figure7.pdf` — 2-bar bootstrap, error
  bars. The whole figure is two numbers (ρ_hist, ρ_diff) and their
  confidence intervals.
- `microstructure_diagnostics/fig_p2_1_spread_conditional.pdf` — 3
  panels × 4 bars each = 12 numbers, naturally a table.

By contrast, **figures whose visual content is genuinely structural**
and not table-reducible:

- All `B1_return_distribution.pdf` — the density overlay is the point.
- All `C1_acf_abs_return.pdf` — the decay curve shape is the point.
- All `D1_conditional_return_imbalance.pdf` — the S-curve shape is the
  point.
- `midprice_traces_*.pdf` — the path shape is the point.
- `fig_intraday_lob_volume.pdf` — the U-shape is the point (already
  wired).
- `fig_a_real_stylized_facts.pdf` — six-panel pedagogical reference
  (already wired).
- `microstructure_diagnostics/fig_p2_2_event_transitions.pdf` —
  arguably table-reducible (4×4 matrix), but visually shows
  v9-matches-real in a way the table doesn't.

---

## 6. Concrete recommendations

### 6.1 Immediate easy wins (no fixes needed)

| Order | Figure | Section | Why |
|---|---|---|---|
| 1 | `pca_projections.csv` → copy to `results/trades_metrics_all/`, re-run script 80 | unlocks proper §5.4 PCA 4-panel | the file exists at `results_cluster/trades_metrics_all/pca_projections.csv`; this is the blocker that was previously cluster-pending |
| 2 | `B1_return_distribution.pdf` (v2, v5, v9 stacked) | §5.2 | the kurtosis-overshoot story currently has no figure; B1 is the canonical one |
| 3 | `C1_acf_abs_return.pdf` (v2, v5, v9 stacked) | §5.2 | volatility-clustering story has no figure either |
| 4 | `D1_conditional_return_imbalance.pdf` (v2, v9) | §6 | mechanical explanation for A2≈A1; the body has no visual for the OFI signal failure |
| 5 | `midprice_traces_v2.pdf` + `_v5.pdf` + `_v9.pdf` (or a 1×3 panel) | §5.1 or §5.7 | shows the scale-suppression directly |
| 6 | `fig_p2_2_event_transitions.pdf` | §5.7 supporting exhibit | makes the "v9 is the only protagonist that gets event composition right" claim visual |

### 6.2 Medium-effort wins (require script edits or new figures)

| Order | Figure | Section | Required work |
|---|---|---|---|
| 7 | Per-regime C1 grid (v2/v5/v9 × base/high_vol/toxic/thin) | App. or §5.2 | new script that composes 12 existing per-regime PDFs into a 3×4 montage; uses existing artifacts |
| 8 | TSLA `B1_return_distribution.pdf` and `C1_acf_abs_return.pdf` (v2, v5) | §7 | already exist in `_archive/tsla_v{2,5}/validation/`; same wiring as INTC |
| 9 | TSLA `midprice_traces_v{2,5}.pdf` | §7 | already exist in `trades_metrics_tsla/` |
| 10 | Fix `E1_training_curves.pdf` | App. B | requires pulling Lightning metrics.csv (P1.C — already deferred); can defer |
| 11 | `F2_film_modulation.pdf` for one model | §3.2.2 | not yet inspected; substantive supporting evidence for the FiLM-collapse story |

### 6.3 Things to drop / table-ify

| Figure | Action | Reason |
|---|---|---|
| `fig_3_3_predictive_vs_scale.pdf` (already in §5.3) | replace with 3-row table | user-flagged as could-be-table |
| `fig_1_2_pairwise_anticorrelation.pdf` (already in §5.5) | drop figure, keep `tab:correlation_subset` | user-flagged |
| `fig_return_distribution_full.pdf` (in `trades_style/`) | skip | broken Q-Q, kurt mismatch |
| `predictive_score_all/predictive_score.pdf` | not in report; don't add | bar chart easily as table |
| `hypothesis/figure7.pdf` (any model) | not in report; don't add | 2-bar bootstrap |
| `microstructure_diagnostics/fig_p2_1_spread_conditional.pdf` | table | 3-panel bars |
| Most `B2/B3/B4` per model | don't surface in body | bar-chart distributions |

### 6.4 The §5.3 duplicate I flagged earlier (still open)

`fig_2_3_return_tail.pdf` is included twice — once correctly at §5.2
and once as a placeholder for the predictive-score bar in §5.3 (which
the caption describes). The right swap is to `fig_3_1_predictive_score.pdf`
in `results/report_figures/` (which is a 20-model log-bar chart).
**This figure is also table-replaceable** by the user's standard
(it's a sorted bar chart), so the cleanest fix is to drop the §5.3
figure entirely and convert it to a sentence + an inline ordered list,
or keep it because the visual sort is what the prose narrates.

---

## 7. Wiring plan (proposed order)

1. **Path-fix to consume `results_cluster/`.** Add it to
   `\graphicspath` so wiring is `cd report ; pdflatex` clean:
   ```latex
   \graphicspath{{../results/report_figures/}{../results/}{../results_cluster/}}
   ```
   This makes every PDF in the cluster pull addressable by basename.
2. **Copy `pca_projections.csv` into the local
   `results/trades_metrics_all/`** so the local figure suite
   auto-upgrades on next run of script 80. Then re-run script 80.
3. **§5.2** — add the B1 (return distribution) and C1 (|return| ACF)
   protagonist panels. These are the highest-information visual
   additions and currently the body has zero per-feature density
   panels.
4. **§5.7** — add midprice traces and the event-transitions exhibit.
5. **§6** — add D1 (conditional return on imbalance) for v2 and v9.
6. **§7** — surface TSLA B1, C1, midprice traces.
7. **Drop §5.3 duplicate** (`fig_2_3_return_tail.pdf` second instance)
   and optionally drop §5.5's pairwise scatter per user's preference
   for table-only on those metrics.
8. **App.** — per-regime C1 grid (composition script; uses existing
   PDFs).

Each step is one Edit + one section caption write. Pace: ~6 edits to
add the protagonist B1/C1/D1 trio across §5.2 and §6, ~3 edits for the
cross-asset additions in §7, and one cleanup edit for §5.3.

---

## 8. Open questions for the user

1. The per-regime C1 / D1 grid (3×4 = 12 panels) is genuinely new
   material we never planned for and would require a new figure
   composer. Worth doing for the report, or appendix-only?
2. For the §5.2 protagonist add, do you want B1+C1 as
   1×3 row-per-feature subfigures, or stacked 3×2 grids (one big
   figure per feature)?
3. The TSLA validation suite is complete (v2/v5 × 12 figures). Should
   §7 grow to include B1/C1/midprice or stay table-only?
4. The §5.3 duplicate: do you want `fig_3_1_predictive_score` (the bar
   chart) wired in or drop the figure outright?
