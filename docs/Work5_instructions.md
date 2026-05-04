# Work5 Instructions — Exhaustive Diffusion Model Analysis

**Project:** Stress-Testing HF Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Date:** 2026-04-27
**Scope:** Everything needed to characterize the v2 generator's statistical quality, training dynamics, and internal representations. This document produces the report's Section 6.5 (Generator Validation) and Section 7.1 (Generator Validation Results).

---

## Context: What Exists

- **v2 checkpoint**: `intc_v2/ckpt-epoch004-valloss0.0106.ckpt`, ~10M params, 8 epochs (target 10, hit wall-time cap), Min-SNR γ=5.0, price-invariant 8-feature representation.
- **Real validation tapes**: 10 days in `data/processed/val/INTC_*.parquet`, 23-column canonical schema.
- **Real training tapes**: 60 days in `data/processed/train/INTC_*.parquet`.
- **v2 synthetic tapes (unconditional baseline)**: 6,400 stitched Parquets (2,560 events each) in `synthetic_v2_stitched/INTC/{base,high_vol,toxic,thin}/`.
- **v1 checkpoint** (for comparison): `intc_v1_short/best.ckpt`, 1.6M params, 5 epochs, uniform MSE, absolute-price features.
- **NormStats**: `data/processed/norm_stats_INTC.json` with per-feature mean/std and `anchor_mid`.
- **CSVLogger outputs** from both training runs (if saved; check `checkpoints/intc_v1_short/` and `checkpoints/intc_v2/` for `metrics.csv` or Lightning logs).

## What This Document Produces

A new module `src/diffmm/eval/stylized.py`, a new visualization module `src/diffmm/viz/validation.py`, a driver script `scripts/50_validate_generator.py`, and a results directory `results/validation/INTC/` containing ~20 figures and 3 CSV summary tables. These are the evidence base for whether the generator is credible enough to trust the downstream ranking exercise.

All analysis is compute-light (statistics on already-generated data plus a handful of forward passes through the checkpoint). No GPU training. Estimated total wall-clock: 2–4 hours of coding, <30 minutes of compute.

---

## Part A — Data Loading Conventions

Every analysis in this document operates on two pools of data:

```python
# Real pool: all 10 validation-day tapes concatenated
real_tapes = [read_tape(p) for p in sorted(val_dir.glob("INTC_*.parquet"))]
real_all = pl.concat(real_tapes)

# Synthetic pool: sample 1,000 stitched tapes across all 4 regimes (250 per regime)
# For regime-specific analyses, filter by regime_label
synth_tapes = [read_tape(p) for p in sample(sorted(synth_dir.rglob("*.parquet")), 1000)]
synth_all = pl.concat(synth_tapes)
```

For marginal and temporal tests, use all events from both pools. For conditional tests, partition by regime label. For per-day analyses, iterate over individual tapes.

All comparisons are **real validation vs. synthetic**. Training data is used only where explicitly stated (norm stats, quantile references). Holdout is never touched.

---

## Part B — Marginal Distribution Tests

These answer: "do individual feature values from the generator look like real data, one feature at a time?"

### B.1 — Mid-Return Distribution

The most fundamental test. Mid-returns are the core price-dynamics signal the model must capture.

**Compute:**

1. Extract `mid_return` from real\_all (drop nulls / first-row zeros) and synth\_all (same). Call these `r_real` and `r_synth`.
2. **Summary statistics table**: for both pools, compute mean, std, skewness (`scipy.stats.skew`), excess kurtosis (`scipy.stats.kurtosis` with `fisher=True`), min, max, 1st/5th/95th/99th percentiles.
3. **Wasserstein-1 distance**: `scipy.stats.wasserstein_distance(r_real, r_synth)`. This is the LOB-Bench primary metric.
4. **Inter-day Wasserstein reference**: compute W-1 between each pair of the 10 real validation days (45 pairs). Report mean ± std. The generator passes if its W-1 is below 2× this inter-day mean.
5. **Kolmogorov-Smirnov test**: `scipy.stats.ks_2samp(r_real, r_synth)`. Report the KS statistic and p-value. Note: with millions of events, KS will almost certainly reject — this is expected. The magnitude of the KS statistic matters more than the p-value.
6. **Anderson-Darling test** (if both samples are reasonably sized): `scipy.stats.anderson_ksamp([r_real, r_synth])`. More sensitive to tail differences than KS.

**Visualize (4-panel figure):**

- **(a) Histogram overlay** — both distributions on the same axes, log scale on y-axis, 200 bins spanning [−0.005, +0.005] (clip extremes for visibility). Real in blue, synthetic in orange. If the synthetic distribution is visibly narrower or has thinner tails, the model is under-representing extreme returns.
- **(b) QQ plot** — real quantiles on x, synthetic quantiles on y, computed at 1000 evenly-spaced probability levels from 0.001 to 0.999. Perfect match = 45° line (draw it). Tail deviations appear as the curve bending away from the line at the extremes. Add a ±2σ bootstrap confidence band around the 45° line by resampling the real pool 200 times.
- **(c) Excess kurtosis comparison** — two bars (real, synth). Financial returns have excess kurtosis >> 0 (leptokurtic). If synthetic kurtosis < 0, the model is producing light-tailed returns — a fundamental failure. Annotate both values on the bars.
- **(d) W-1 bar with inter-day reference band** — one bar for real-vs-synth W-1, shaded band showing [mean − 2σ, mean + 2σ] of the inter-day W-1 distribution. Pass/fail is visually obvious: bar inside the band = pass.

**Output:** `results/validation/INTC/B1_return_distribution.pdf`, `results/validation/INTC/B1_return_stats.csv`.

**Pass criteria:**
- W-1 < 2× inter-day mean W-1.
- Synthetic excess kurtosis > 0 (leptokurtic).
- Synthetic skewness within ±0.5 of real.
- QQ plot: no systematic deviation beyond 2σ band for central 90% of the distribution (tails may deviate).

---

### B.2 — Spread Distribution

For large-tick INTC, spread is almost always exactly $0.01 (one tick). The model must capture this discrete concentration.

**Compute:**

1. Extract `spread` from both pools. For real data, also compute from `ask_px - bid_px` as a sanity cross-check against the stored `spread` column.
2. **1-tick fraction**: fraction of events with spread == $0.01 (within float tolerance, say |spread − 0.01| < 0.001). For real INTC this should be >90%. Report for both.
3. **Multi-tick fraction**: fraction at $0.02, $0.03, etc. Build a discrete PMF over spread values rounded to the nearest cent.
4. **W-1**: `wasserstein_distance(spread_real, spread_synth)`.
5. **Chi-squared test on the discrete PMF**: bin spreads into {$0.01, $0.02, $0.03, $0.04, ≥$0.05}. Compute expected frequencies from the real PMF, observed from synthetic. `scipy.stats.chisquare(observed, expected)`. This tests whether the model matches the *discrete* spread structure, not just the continuous distribution.

**Visualize (3-panel figure):**

- **(a) Histogram** — bar chart of spread values (in cents) for real (blue) and synthetic (orange), side by side. X-axis: 1¢, 2¢, 3¢, 4¢, 5¢+. Y-axis: fraction of events. The 1¢ bar should dominate both.
- **(b) CDF overlay** — empirical CDFs of spread for both pools. For INTC, the real CDF should jump to >0.9 at $0.01.
- **(c) W-1 bar with inter-day band** — same format as B.1(d).

**Output:** `results/validation/INTC/B2_spread_distribution.pdf`, `results/validation/INTC/B2_spread_stats.csv`.

**Pass criteria:**
- Synthetic 1-tick fraction > 80% (real should be >90%; allow 10pp degradation).
- W-1 < 2× inter-day W-1.
- Chi-squared test: inspect magnitude of statistic (p-value will reject with large N; look at per-bin residuals instead).

---

### B.3 — Size Distribution (bid\_sz, ask\_sz)

Real LOB sizes are heavy-tailed: many small quotes, few very large ones. The model must capture this.

**Compute:**

1. Extract `bid_sz` and `ask_sz` from both pools.
2. Summary statistics: mean, median, std, skewness, kurtosis, 1st/99th percentiles for each.
3. W-1 on log-transformed sizes: `wasserstein_distance(log(bid_sz_real+1), log(bid_sz_synth+1))`.
4. **Power-law tail test**: for sizes > median, fit `P(sz > x) ~ x^{-α}` using `scipy.stats.powerlaw.fit` or a simple log-log linear regression on the empirical CCDF tail. Compare α\_real vs α\_synth. If α\_synth is much larger (steeper tail drop), the model is compressing the size distribution.

**Visualize (2-panel figure):**

- **(a) Log-log CCDF** — complementary CDF of bid\_sz for real and synthetic, both on log-log axes. If both are approximately linear with similar slopes, the tail behavior matches.
- **(b) QQ plot of log(bid\_sz)** — real quantiles on x, synthetic on y.

**Output:** `results/validation/INTC/B3_size_distribution.pdf`.

**Pass criteria:**
- α\_synth within ±30% of α\_real.
- W-1 on log-sizes < 2× inter-day W-1.

---

### B.4 — Event-Type Composition

This is where you already know there's a problem. Document it rigorously.

**Compute:**

1. **Trade fraction**: fraction of events with `event_type == 'trade'` for real vs. synthetic. Real INTC: ~10%. Your v2 synthetic: ~58% (from Work4 §1.5). Report both.
2. **Per-regime trade fraction**: compute separately for each synthetic regime label. Does trade fraction differ across regimes? (Work3 §2.4 said trade frequency was the main thing regime conditioning modulated.)
3. **Trade size distribution** conditional on event being a trade: histogram of `trade_sz` for trade events only, real vs. synthetic.
4. **Trade sign distribution**: fraction of buy (+1) vs sell (−1) for trade events, real vs. synthetic. Should be approximately 50/50 in both (no directional bias at the population level).
5. **Inter-trade interval**: for trade events only, compute the distribution of event-index gaps between consecutive trades. In real data, trades cluster (a burst of trades, then a quiet period of quote updates). In synthetic, if trades are scattered uniformly among quotes, the inter-trade gap distribution will be geometric rather than bursty.

**Visualize (3-panel figure):**

- **(a) Trade fraction comparison** — two bars: real, synth. Annotate exact percentages. This will be a dramatic bar chart showing the ~6× discrepancy.
- **(b) Trade fraction by regime** — grouped bar chart: x=regime, y=trade fraction. Shows how conditioning modulates trade intensity.
- **(c) Inter-trade gap histogram** — real and synth overlaid. If the shapes differ qualitatively (e.g., exponential vs. power-law), the model isn't capturing trade clustering.

**Output:** `results/validation/INTC/B4_event_composition.pdf`, `results/validation/INTC/B4_event_stats.csv`.

**Pass criteria:**
- This will fail. The trade fraction is inverted. The purpose of documenting it is to (a) show you know, (b) quantify the magnitude, and (c) discuss implications for the fill simulator. The report should note that because the fill simulator only triggers on trade events, the elevated synthetic trade rate means synthetic backtests produce more fills per unit time than real backtests. This inflates PnL variance on synthetic tapes but doesn't bias the *direction* of the PnL difference between agents (since all agents face the same synthetic tape).

---

### B.5 — Per-Feature Marginal Summary Table

A single table consolidating marginal diagnostics across all 8 generator features plus derived `spread` and `mid`.

| Feature | Real mean | Synth mean | Real std | Synth std | W-1 | Inter-day W-1 ref | Pass? |
|---|---|---|---|---|---|---|---|
| mid\_return | | | | | | | |
| bid\_dist | | | | | | | |
| ask\_dist | | | | | | | |
| trade\_dist | | | | | | | |
| bid\_sz | | | | | | | |
| ask\_sz | | | | | | | |
| trade\_sz | | | | | | | |
| trade\_sign | | | | | | | |
| spread (derived) | | | | | | | |

**Compute:** Loop over features. For each, compute both pool's summary stats + W-1 + inter-day W-1 reference. Mark pass/fail per the 2× criterion.

**Output:** `results/validation/INTC/B5_marginal_summary.csv`.

---

## Part C — Temporal Dependence Structure

These answer: "does the generator produce *sequences* with realistic dynamics, not just realistic snapshots?"

Marginals being correct is necessary but not sufficient. A model that produces iid draws from the correct marginal distribution would pass all Part B tests but fail every Part C test. Financial microstructure has rich temporal dependence: volatility clusters, order flow persists, returns mean-revert at short lags. The model must capture these.

### C.1 — Volatility Clustering (Absolute-Return ACF)

The most canonical stylized fact in financial time series. Absolute returns exhibit long-memory autocorrelation with slow power-law decay.

**Compute:**

1. Compute `|mid_return|` for real and synthetic.
2. ACF out to lag 200 using `statsmodels.tsa.stattools.acf(x, nlags=200, fft=True)`. Compute separately for each real validation day (10 curves), each synthetic tape (subsample 100 tapes), and on the concatenated pools.
3. **Per-day bootstrap confidence band**: from the 10 real-day ACF curves, compute pointwise mean ± 2σ at each lag. This is the "natural variation" band.
4. **Lag-50 deviation**: compute `|ACF_synth(50) - ACF_real(50)| / ACF_real(50)`. The proposal's kill criterion: >50% deviation → fail.
5. **Power-law decay fit**: on the real ACF, fit `ACF(lag) ~ lag^{-β}` for lags 10–200 (log-log linear regression). Compute β\_real and β\_synth. Real financial data typically has β ≈ 0.2–0.4 (slow decay). If β\_synth >> β\_real, synthetic volatility clustering decays too fast (model doesn't capture long memory).

**Visualize:**

- **Main plot**: x=lag, y=ACF. Blue line = real (concatenated), blue shaded band = per-day ±2σ, orange line = synthetic (concatenated). Log scale on x-axis to emphasize the power-law decay behavior.
- **Inset**: log(lag) vs log(ACF) for lags 10–200, with fitted lines for both. Slopes are −β.

**Output:** `results/validation/INTC/C1_acf_abs_return.pdf`.

**Pass criteria:**
- Lag-50 deviation < 50%.
- β\_synth within ±50% of β\_real (same order of magnitude of memory).
- Synthetic ACF curve visually within the per-day confidence band for lags 1–50.

---

### C.2 — Order Flow Persistence (Trade-Sign ACF)

Real markets exhibit positive short-lag autocorrelation in trade signs: buy orders cluster, sell orders cluster. This is the Bouchaud–Farmer–Lillo "long memory of order flow" effect, driven by institutional order-splitting.

**Compute:**

1. Filter to trade events only. Extract `trade_sign` sequences.
2. ACF of `trade_sign` out to lag 100 for real and synthetic.
3. Same per-day confidence band as C.1.
4. **Lag-1 autocorrelation**: the single most diagnostic number. Real INTC should have positive lag-1 ACF (somewhere between 0.1 and 0.5, depending on the stock and time period). If synthetic lag-1 ACF ≈ 0, the model is producing iid trade signs — it hasn't learned order-flow persistence.
5. **Runs test** (`statsmodels.sandbox.stats.runs.runstest_1samp` or manual): count the number of sign runs (consecutive same-sign trades) in real vs. synthetic sequences. Fewer runs = more clustering. Report the z-statistic.

**Visualize:**

- x=lag, y=ACF(trade\_sign). Blue = real, shaded = confidence band, orange = synthetic. Annotate lag-1 values.

**Output:** `results/validation/INTC/C2_acf_trade_sign.pdf`.

**Pass criteria:**
- Synthetic lag-1 ACF has the same sign as real (positive).
- Magnitude within 50% of real lag-1 ACF.

---

### C.3 — Return Autocorrelation (Bid-Ask Bounce)

Raw microstructure returns are slightly negatively autocorrelated at lag 1 due to the bid-ask bounce: a buy trade pushes the last trade price to the ask, a sell pushes it to the bid, creating mechanical negative autocorrelation. Returns should be approximately uncorrelated at lags > 1.

**Compute:**

1. ACF of `mid_return` out to lag 50 for real and synthetic.
2. Ljung-Box test (`statsmodels.stats.diagnostic.acorr_ljungbox`) at lags [1, 5, 10, 20] for both. Report Q-statistics and p-values.

**Visualize:**

- ACF plot with 95% significance bands (horizontal dashed lines at ±1.96/√N). Real and synthetic overlaid.

**Output:** `results/validation/INTC/C3_acf_return.pdf`.

**Pass criteria:**
- Lag-1 ACF has the correct sign (negative or near-zero for both).
- No spurious positive autocorrelation at lags 1–10 in the synthetic (would indicate the model is producing trending sequences).

---

### C.4 — Realized Volatility Signature Plot

The Andersen–Bollerslev–Diebold volatility signature: compute realized variance at multiple sampling frequencies and plot RV vs. frequency. At high frequencies, microstructure noise inflates RV. At lower frequencies, RV converges to the "true" integrated variance. The shape of this curve encodes the noise structure of the price process.

**Compute:**

1. For each sampling frequency Δ ∈ {1, 2, 5, 10, 25, 50, 100, 256} events:
   - Subsample `mid` at every Δ-th event.
   - Compute log-returns on the subsampled series.
   - Compute realized variance = sum of squared returns.
   - Normalize to per-event units for comparability.
2. Do this for each real validation day (10 curves) and 100 synthetic tapes.
3. Compute pointwise mean ± 2σ bands for both families.

**Visualize:**

- x = sampling frequency (events, log scale), y = realized variance (log scale). Real mean curve + band, synthetic mean curve + band.
- Key diagnostic: at high frequency (Δ=1), RV should be elevated due to bid-ask bounce. If synthetic RV at Δ=1 is flat (no elevation), the model isn't producing realistic microstructure noise.

**Output:** `results/validation/INTC/C4_rv_signature.pdf`.

**Pass criteria:**
- Qualitative shape match: both curves show declining RV as Δ increases.
- Ratio of RV(Δ=1) / RV(Δ=100) is within 2× between real and synthetic.

---

### C.5 — OFI Autocorrelation

Order flow imbalance (the net signed change in top-of-book depth) exhibits positive autocorrelation at short lags in real markets, reflecting persistent pressure.

**Compute:**

1. Compute an event-level OFI proxy: `ofi_proxy = Δbid_sz · 1{bid_px ≥ bid_px_prev} − Δask_sz · 1{ask_px ≤ ask_px_prev}`. This is a simplified version of Cont–Kukanov–Stoikov (2014) adapted for the top-of-book data you have.
2. ACF of `ofi_proxy` out to lag 100, real vs. synthetic.

**Visualize:**

- ACF overlay with confidence band. Same format as C.1.

**Output:** `results/validation/INTC/C5_acf_ofi.pdf`.

**Pass criteria:**
- Synthetic OFI ACF has the same sign at lag 1 as real.
- Decay rate within same order of magnitude.

---

## Part D — Cross-Feature and Conditional Structure

These answer: "does the generator preserve relationships *between* features — the microstructure mechanisms that agents actually exploit?"

### D.1 — Conditional Return on Imbalance (THE Critical Test)

This is the single most important cross-feature validation for the project. The OFI → return relationship (Cont–Kukanov–Stoikov 2014) is the mechanism Agent A2 exploits. If the synthetic data doesn't preserve this relationship, the synthetic environment is structurally unable to differentiate A2 from A1.

**Compute:**

1. Compute top-of-book imbalance: `I_t = (bid_sz - ask_sz) / (bid_sz + ask_sz)`, ranging in [−1, +1].
2. For both pools, bin I\_t into 20 equal-width bins on [−1, +1].
3. For each bin, compute the mean `mid_return` over the next Δ events. Use Δ ∈ {1, 10, 50}.
4. **Sign test**: the relationship should be monotonically positive (positive imbalance → positive expected return). This is a fundamental microstructure regularity. If the synthetic relationship is flat or has the wrong sign, the generator is producing data that violates the core price-formation mechanism. The proposal's kill criterion: wrong sign → fail.
5. **Linear regression**: fit `E[r_{t+Δ} | I_t] = α + β · I_t` on both pools. Report β\_real, β\_synth, R²\_real, R²\_synth.
6. **Bootstrap confidence bands**: resample real validation days 200 times, recompute the binned conditional means each time, get pointwise ±2σ bands.

**Visualize:**

- x = imbalance bin center, y = E[r\_{t+Δ} | I\_t]. Blue line = real, blue band = bootstrap CI, orange line = synthetic. One panel per Δ value (3 panels).
- Annotate β and R² for both on each panel.

**Output:** `results/validation/INTC/D1_conditional_return_imbalance.pdf`, `results/validation/INTC/D1_regression_stats.csv`.

**Pass criteria (hard):**
- β\_synth has the correct sign (positive) at Δ=10 and Δ=50.
- β\_synth magnitude is within one order of magnitude of β\_real.

---

### D.2 — Joint Spread-Imbalance Distribution

**Compute:**

1. Build 2D histograms: x = spread (binned: 1¢, 2¢, 3¢, 4¢, 5¢+), y = imbalance (binned into 5 equal-width bins on [−1, +1]). Normalize to joint PMFs.
2. **Jensen-Shannon divergence**: `scipy.spatial.distance.jensenshannon(pmf_real.ravel(), pmf_synth.ravel())`. A symmetric, bounded [0, 1] divergence measure. Lower = better. JSD < 0.1 is excellent; JSD > 0.3 indicates substantial distributional mismatch.
3. **Marginal consistency check**: the marginal distributions of spread and imbalance from the 2D histogram should match the univariate histograms from B.2 and B.3 (they will by construction — this is just a sanity check on the binning).

**Visualize:**

- Side-by-side 2D heatmaps: left = real, right = synthetic. Same color scale. Annotate JSD.

**Output:** `results/validation/INTC/D2_joint_spread_imbalance.pdf`.

**Pass criteria:**
- JSD < 0.3.
- Dominant cell (spread=1¢, imbalance≈0) has similar probability mass in both.

---

### D.3 — Conditional Moments by Regime Label

This tests whether conditioning actually works — the core of the "stress-directed generation" claim.

**Compute:**

For each regime label ∈ {base, high\_vol, toxic, thin}, compute five statistics on both real validation days (partitioned by their *observed* regime labels) and synthetic tapes (generated under that regime condition):

| Statistic | Definition |
|---|---|
| mean\_abs\_return | mean(\|mid\_return\|) — volatility proxy |
| mean\_spread | mean(spread) |
| std\_return | std(mid\_return) — intra-window volatility |
| trade\_fraction | #trades / #events |
| mean\_bid\_sz | mean(bid\_sz) — depth proxy |

For each (regime, statistic) pair:
1. Compute real conditional mean and std (from the 10 val days, weighted by number of events per regime in each day).
2. Compute synthetic conditional mean (from the 250 synthetic tapes under that regime).
3. Z-score: `z = (synth_mean - real_mean) / real_std`. If |z| > 2, the model is generating moments outside the plausible range for that regime.

**Test monotonicity:** Does mean\_abs\_return increase from base → high\_vol? Does trade\_fraction increase from base → toxic/thin? If conditioning is working, the synthetic moments should shift in the direction implied by the regime label. If all regimes produce the same moments, conditioning is cosmetic.

**Visualize:**

- **Heatmap**: rows = regimes (base, high\_vol, toxic, thin), columns = statistics. Cells colored by z-score (diverging colormap: green = |z| < 1, yellow = 1 < |z| < 2, red = |z| > 2). Annotate each cell with (real\_mean, synth\_mean).
- **Grouped bar chart**: for each statistic, 4 bars (one per regime), colored by real/synth. Shows the monotonicity pattern.

**Output:** `results/validation/INTC/D3_conditional_moments.pdf`, `results/validation/INTC/D3_conditional_moments.csv`.

**Pass criteria:**
- Majority of cells have |z| < 2.
- At least one statistic shows monotonic shift across regimes in the synthetic data.

---

## Part E — Diffusion Model Training Diagnostics

These explain *how* the model was trained and whether the training was adequate.

### E.1 — Training Loss Curves (v1 vs v2)

**Compute:**

1. Load CSVLogger outputs from both training runs. These should contain per-step train\_loss and per-epoch val\_loss.
2. If CSVLogger outputs don't exist (Lightning may have put them in a `lightning_logs/` directory inside the checkpoint folder), reconstruct from SLURM `.out` files where you printed loss every 20 steps.

**Visualize (2-panel figure):**

- **(a) Train loss vs. step** — log-y scale. v1 in dashed blue, v2 in solid orange. Annotate epoch boundaries with vertical lines. Annotate the Min-SNR activation point (v2 only — it's active from step 0, but annotate where the loss curves diverge).
- **(b) Val loss vs. epoch** — linear scale. Both curves, with final values annotated: v1=0.0160, v2=0.0106.

**Output:** `results/validation/INTC/E1_training_curves.pdf`.

---

### E.2 — Per-Timestep Denoising Quality (v1 vs v2)

This is the quantitative evidence that Min-SNR training improved the model at the timesteps that matter.

**Compute:**

1. For both checkpoints, run the following evaluation loop on 200 random validation windows:
   ```python
   for t in [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 999]:
       eps = torch.randn_like(x0)
       x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
       eps_pred = model(x_t, t, c)
       mse_t = ((eps_pred - eps) ** 2).mean(dim=[1,2])  # per-sample MSE
       record: t, mse_t.mean(), mse_t.std(), eps_pred.std()
   ```
2. Compute the Min-SNR weight at each t for reference: `w(t) = min(SNR(t), 5) / SNR(t)` where `SNR(t) = alpha_bar[t] / (1 - alpha_bar[t])`.

**Visualize (2-panel figure):**

- **(a) MSE vs timestep** — x=t, y=mean MSE. v1 dashed blue, v2 solid orange. v2 should be flatter (more uniform quality across timesteps) and lower in the middle range.
- **(b) Min-SNR weight function** — secondary y-axis showing `w(t)`. This shows the reader how the training loss was rebalanced: weight drops toward zero at low t (easy clean samples), stays at 1.0 at high t.

Additional annotation: mark the "critical range" t ∈ [200, 600] where the denoiser has the most informative work to do (signal and noise are comparable in magnitude). v2's MSE in this range should be visibly lower than v1's.

**Output:** `results/validation/INTC/E2_per_timestep_quality.pdf`, `results/validation/INTC/E2_per_timestep_stats.csv`.

---

### E.3 — x0\_clip Activation Rate

**Compute:**

1. During DDIM sampling (using the v2 checkpoint), at each of the 200 reverse steps, count the fraction of elements in the (B, L, F) tensor where `|x0_pred| > x0_clip` before clamping.
2. Report: per-timestep activation rate, overall activation rate.
3. For comparison, run the same analysis on the v1 checkpoint.

**Implementation:** Add an accumulator inside `ddim_sample`:
```python
clip_mask = (x0_pred.abs() > x0_clip)
clip_count_per_step[step] = clip_mask.float().mean().item()
```

Run on 50 random seeds.

**Visualize:**

- x=DDIM step (0 to 199), y=activation rate. v1 in dashed (expect high activation — the model was rescued by clipping), v2 in solid (expect <1% at most steps). Log-y scale if v1 rates are large.

**Output:** `results/validation/INTC/E3_x0_clip_activation.pdf`.

**Interpretation:** If v2 activation rate is <1% everywhere, the model is generating well-behaved predictions and the clamp is purely defensive. If activation rate is elevated at specific timesteps, those are the model's weak spots — document which and correlate with the MSE curve from E.2.

---

### E.4 — Noise Schedule Visualization

Provide the reader with a reference for the diffusion process itself.

**Compute:** From the linear schedule, compute and plot:
- α̅\_t (cumulative signal retention) vs t
- √α̅\_t (signal scale) vs t
- √(1−α̅\_t) (noise scale) vs t
- SNR(t) = α̅\_t / (1−α̅\_t) vs t (log scale on y)

**Visualize:** 4 curves on one figure with dual y-axes. Annotate: "signal dominates" (left), "noise dominates" (right), "critical range" (middle).

**Output:** `results/validation/INTC/E4_noise_schedule.pdf`.

---

## Part F — Model Internals and Interpretability

These go beyond "does the model work" to explain "how does the model work."

### F.1 — Regime Embedding Space

**Compute:**

1. Extract the learned embedding vectors from `model.regime_embed` for each conditioning axis.
2. For each of the 81 possible (c\_vol, c\_vpin, c\_imb, c\_tod) tuples (3^4, though many are rare or empty in the training data), concatenate the per-axis embeddings as the model does at inference time. This gives one vector per conditioning tuple.
3. Also extract the null embedding (the ∅ token used during CFG dropout).
4. Project to 2D with PCA (use the first two principal components). Record variance explained.
5. Also try t-SNE with perplexity=10 for a nonlinear view.

**Visualize:**

- Scatter plot: each point is a conditioning tuple, colored by its regime label (base=gray, high\_vol=red, toxic=purple, thin=blue). The null embedding as a black star.
- If the four regime clusters separate in embedding space, the model's FiLM layers can distinguish regimes. If they overlap completely, the embeddings are uninformative.
- Annotate the fraction of variance explained by PC1 and PC2. If >80%, the embedding space is effectively low-dimensional.

**Output:** `results/validation/INTC/F1_regime_embedding_pca.pdf`.

---

### F.2 — FiLM Modulation Magnitude

**Compute:**

1. For each FiLM layer (there are 2: start and end), run a forward pass with each regime condition on a fixed input and extract the learned γ and β vectors.
2. Compute `||γ(c) - 1||₂` (deviation from identity scaling) and `||β(c)||₂` (shift magnitude) for each regime c.
3. Also compute for the null condition.

**Implementation:** Register forward hooks on the FiLM layers:
```python
film_outputs = {}
def hook_fn(name):
    def fn(module, input, output):
        film_outputs[name] = output
    return fn
model.film_start.register_forward_hook(hook_fn("film_start"))
model.film_end.register_forward_hook(hook_fn("film_end"))
```

Run one forward pass per regime label (4 + null = 5 passes).

**Visualize:**

- Grouped bar chart: x = regime (base, high\_vol, toxic, thin, null), bars = (γ deviation, β magnitude) for each FiLM layer (4 bars per regime: γ\_start, β\_start, γ\_end, β\_end).
- If bars are near zero for all regimes, FiLM isn't modulating — conditioning is cosmetic. If bars differ across regimes, FiLM is doing something.

**Output:** `results/validation/INTC/F2_film_modulation.pdf`.

---

### F.3 — Attention Weight Analysis

**Compute:**

1. Pick 3 representative validation windows: one from a calm period (low vol, low VPIN), one from a volatile period (high vol), and one from a toxic period (high VPIN).
2. For each window, run a forward pass at 3 diffusion timesteps: t=900 (early denoising — input is mostly noise), t=500 (middle — mixed signal/noise), t=100 (late — mostly clean).
3. Extract attention weights from the first self-attention layer's first head. These are (L, L) = (256, 256) matrices.

**Implementation:**
```python
attn_weights = {}
def attn_hook(name):
    def fn(module, input, output):
        # For nn.MultiheadAttention, output[1] is the attention weights
        attn_weights[name] = output[1]
    return fn
model.transformer_blocks[0].self_attn.register_forward_hook(attn_hook("block0"))
```

**Visualize (3×3 grid):**

- Rows = timesteps (t=900, 500, 100), columns = windows (calm, volatile, toxic).
- Each cell: 256×256 heatmap of attention weights (viridis colormap, with row-normalization so each row sums to 1).
- Look for:
  - **Diagonal dominance**: each event attends to nearby events — expected for local microstructure dynamics.
  - **Column spikes**: certain events are universally attended to — these are "anchor events" (might correspond to trades or spread changes).
  - **Timestep progression**: early denoising (t=900) should show diffuse attention (everything looks like noise, attend broadly); late denoising (t=100) should show focused attention (attend to local context for fine refinement).
  - **Regime differences**: if the attention pattern differs between calm and toxic windows at the same timestep, the model is using regime-specific processing strategies.

**Output:** `results/validation/INTC/F3_attention_maps.pdf`.

---

### F.4 — Guidance Weight Sensitivity Sweep

**Compute:**

1. Generate 50 synthetic windows at each guidance weight w ∈ {0.0, 0.5, 1.0, 2.0, 3.0, 5.0} under the "high\_vol" regime condition. Use the same 50 seeds for each w.
2. For each w, compute: mean(|mid\_return|), mean(spread), std(mid\_return), trade\_fraction, mean(bid\_sz).
3. Also compute the same statistics on the real validation data partitioned to the "high\_vol" regime — this is the target the conditioning is trying to hit.

**Visualize (5-panel figure):**

- One panel per statistic. x = guidance weight w, y = statistic value. Blue horizontal line = real conditional moment (the target). Orange curve = synthetic moment at each w. Error bars from the 50 seeds.
- At w=0 (unconditional), the synthetic moment should equal the training-set population mean (no conditioning). As w increases, it should move toward the real conditional moment. At very high w, it may overshoot.
- Mark the current default w=1 with a vertical dashed line.

**Output:** `results/validation/INTC/F4_guidance_sweep.pdf`.

**Interpretation:** This figure tells the reader (and yourself) whether the chosen w=1 is optimal, and whether increasing w would improve regime fidelity or cause over-conditioning artifacts. If the curves are flat (moment doesn't change with w), conditioning is not working at the sampling level. If they reach the target at some w* and then overshoot, report w* and discuss whether switching from w=1 to w* would improve the central test.

---

### F.5 — Feature-Channel Contribution to Loss

**Compute:**

1. For 200 random validation windows, compute per-feature MSE at the middle timestep t=300 (where the model has the most informative work):
   ```python
   per_feature_mse = ((eps_pred - eps) ** 2).mean(dim=[0, 1])  # shape (F,)
   ```
2. This gives 8 values (one per feature channel). Which features does the model predict well? Which poorly?

**Visualize:**

- Bar chart: x = feature name (bid\_dist, ask\_dist, ..., trade\_sign), y = per-channel MSE at t=300. Annotate the ideal value (1.0 for normalized ε).

**Output:** `results/validation/INTC/F5_per_feature_loss.pdf`.

**Interpretation:** If `trade_sz` or `trade_sign` channels have much higher MSE than price-distance channels, it explains the inverted trade/quote ratio — the model is less confident about trade events and defaults to generating them liberally. If `mid_return` has low MSE, the price dynamics are well-captured even if event-type composition is wrong.

---

## Part G — Consolidated Validation Report

### G.1 — Pass/Fail Summary Table

| # | Check | Metric | Value | Criterion | Pass? |
|---|---|---|---|---|---|
| 1 | Return W-1 | B.1 | | < 2× inter-day | |
| 2 | Return kurtosis | B.1 | | > 0 | |
| 3 | Spread 1-tick fraction | B.2 | | > 80% | |
| 4 | Spread W-1 | B.2 | | < 2× inter-day | |
| 5 | Size tail exponent | B.3 | | within ±30% of real | |
| 6 | Trade fraction | B.4 | | ~10% (real); report actual | |
| 7 | ACF(\|r\|) lag-50 deviation | C.1 | | < 50% | |
| 8 | ACF(\|r\|) power-law β | C.1 | | within ±50% of real | |
| 9 | Trade-sign lag-1 ACF sign | C.2 | | positive | |
| 10 | Return lag-1 ACF sign | C.3 | | negative or zero | |
| 11 | RV signature shape | C.4 | | qualitative match | |
| 12 | E[r\|I] sign at Δ=10 | D.1 | | positive β | |
| 13 | E[r\|I] sign at Δ=50 | D.1 | | positive β | |
| 14 | Joint spread-imbalance JSD | D.2 | | < 0.3 | |
| 15 | Conditional moments z-scores | D.3 | | majority \|z\| < 2 | |
| 16 | Crossed books rate | (decode) | | < 10⁻⁵ | |

This table goes directly into the report as Table 5.

### G.2 — Narrative Summary for the Report

Based on the pass/fail pattern, write one of three narratives:

**Most checks pass:** "The v2 generator preserves the primary marginal distributions, temporal dependence structure, and cross-feature relationships of real INTC microstructure. Known failures include [list]. The generator is credible as a conditional sampling engine for the downstream ranking exercise, with the caveats that [failures] limit the realism of [specific aspect]."

**Mixed results:** "The v2 generator captures [passing aspects] but fails to reproduce [failing aspects]. The most consequential failure for the project is [whichever failure most directly undermines the agent comparison — likely the trade/quote ratio or the conditional-return sign]. The downstream ranking results should be interpreted with this limitation in mind."

**Most checks fail:** "The v2 generator does not adequately reproduce the statistical properties of real INTC microstructure. [List failures.] The central hypothesis test result should be understood as a test of the *methodology pipeline*, not as evidence about diffusion-based stress testing in general, since the generator itself does not meet the validation standard."

---

## Implementation Structure

### New files

```
src/diffmm/eval/stylized.py          # All Part B + C + D computations
src/diffmm/eval/model_diagnostics.py  # All Part E + F computations
src/diffmm/viz/validation.py          # All plotting functions
scripts/50_validate_generator.py       # Driver script (Hydra-configured)
```

### `stylized.py` function signatures

```python
def return_diagnostics(real: pl.DataFrame, synth: pl.DataFrame) -> ReturnStats
def spread_diagnostics(real: pl.DataFrame, synth: pl.DataFrame) -> SpreadStats
def size_diagnostics(real: pl.DataFrame, synth: pl.DataFrame) -> SizeStats
def event_composition(real: pl.DataFrame, synth: pl.DataFrame) -> EventStats
def marginal_summary(real: pl.DataFrame, synth: pl.DataFrame) -> pd.DataFrame
def acf_abs_return(real: pl.DataFrame, synth: pl.DataFrame, max_lag=200) -> ACFResult
def acf_trade_sign(real: pl.DataFrame, synth: pl.DataFrame, max_lag=100) -> ACFResult
def acf_return(real: pl.DataFrame, synth: pl.DataFrame, max_lag=50) -> ACFResult
def rv_signature(real: pl.DataFrame, synth: pl.DataFrame, deltas=[1,2,5,10,25,50,100,256]) -> RVResult
def conditional_return_on_imbalance(real, synth, n_bins=20, deltas=[1,10,50]) -> CondReturnResult
def joint_spread_imbalance(real, synth) -> JointResult
def conditional_moments_by_regime(real, synth) -> CondMomentsResult
```

### `model_diagnostics.py` function signatures

```python
def per_timestep_mse(model, val_dataset, timesteps, n_samples=200) -> pd.DataFrame
def x0_clip_activation(model, schedule, n_seeds=50, n_steps=200, clip=4.0) -> pd.DataFrame
def regime_embeddings_pca(model) -> tuple[np.ndarray, np.ndarray, float]
def film_modulation_magnitude(model, regime_conditions) -> pd.DataFrame
def attention_maps(model, windows, timesteps) -> dict[str, np.ndarray]
def guidance_sweep(model, schedule, norm_stats, regime, ws, n_seeds=50) -> pd.DataFrame
def per_feature_mse(model, val_dataset, t=300, n_samples=200) -> pd.DataFrame
```

### `scripts/50_validate_generator.py`

```python
# Loads checkpoint, norm_stats, real val tapes, synthetic tapes
# Runs all analyses
# Writes all figures and CSVs to results/validation/INTC/
# Writes consolidated pass/fail table
# Takes ~20 min total (mostly the model forward passes for Part E+F)
```

### Execution

```bash
# CPU-only for Parts B, C, D (statistics on Parquets)
python scripts/50_validate_generator.py \
    --parts B C D \
    --real-dir data/processed/val/ \
    --synth-dir /orcd/scratch/.../synthetic_v2_stitched/INTC/ \
    --ticker INTC \
    --out-dir results/validation/INTC/

# GPU for Parts E, F (model forward passes)
sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=01:00:00 --mem=32G \
    --wrap "source slurms/_common.sh && python scripts/50_validate_generator.py \
        --parts E F \
        --checkpoint /orcd/scratch/.../intc_v2/ckpt-epoch004-valloss0.0106.ckpt \
        --v1-checkpoint /orcd/scratch/.../intc_v1_short/best.ckpt \
        --norm-stats data/processed/norm_stats_INTC.json \
        --real-dir data/processed/val/ \
        --synth-dir /orcd/scratch/.../synthetic_v2_stitched/INTC/ \
        --ticker INTC \
        --out-dir results/validation/INTC/"
```

---

## Output Inventory

After running all parts, `results/validation/INTC/` should contain:

```
B1_return_distribution.pdf       B1_return_stats.csv
B2_spread_distribution.pdf       B2_spread_stats.csv
B3_size_distribution.pdf
B4_event_composition.pdf         B4_event_stats.csv
B5_marginal_summary.csv
C1_acf_abs_return.pdf
C2_acf_trade_sign.pdf
C3_acf_return.pdf
C4_rv_signature.pdf
C5_acf_ofi.pdf
D1_conditional_return_imbalance.pdf   D1_regression_stats.csv
D2_joint_spread_imbalance.pdf
D3_conditional_moments.pdf            D3_conditional_moments.csv
E1_training_curves.pdf
E2_per_timestep_quality.pdf           E2_per_timestep_stats.csv
E3_x0_clip_activation.pdf
E4_noise_schedule.pdf
F1_regime_embedding_pca.pdf
F2_film_modulation.pdf
F3_attention_maps.pdf
F4_guidance_sweep.pdf
F5_per_feature_loss.pdf
G1_pass_fail_summary.csv
```

23 figures, 8 CSVs. These are the analytical backbone of the report's generator validation section.