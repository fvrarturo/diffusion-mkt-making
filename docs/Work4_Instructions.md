# Remaining Work Plan — Exhaustive Instructions

**Project:** Stress-Testing HF Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Date:** 2026-04-22
**Status:** Central hypothesis test complete (Work3). This document specifies every remaining deliverable.

---

## Overview: What Exists and What's Missing

The pipeline is end-to-end functional. The central hypothesis test produced a directional result (ρ\_diff = +0.5, ρ\_hist = −1.0) but with wide confidence intervals (p = 0.457). What's missing is the analytical depth that turns a proof-of-concept into a credible report. Specifically:

1. The synthetic windows are too short for meaningful agent backtesting (Bug #38).
2. The generator was never formally validated (Phase 2.4 skipped).
3. There is no interpretability analysis of the diffusion model's internals.
4. There is no decomposition of agent PnL into economically meaningful components.
5. The ranking exercise has only 3 agents (6 possible orderings, coarse ρ grid).
6. TSLA is untouched despite data being ready.
7. The report is unwritten.

This plan is organized into seven workstreams, each with atomic tasks, success criteria, code locations, and estimated effort. Workstreams are ordered by priority (do them in this order; stop wherever the deadline forces you).

---

## Workstream 1 — Synthetic Window Concatenation

**Why this is priority 1.** The current 256-event synthetic windows produce ≤1 fill per window for most agents. 75% of per-window PnL values are exactly $0. The worst-quartile ranking metric (the proposal's preferred tail-aware measure) collapses to zero, forcing a fallback to mean PnL. This means the central test is running on a metric the proposal didn't intend. Longer synthetic "days" fix this without retraining.

### Task 1.1 — Decode-side window concatenation

**What.** Modify `decode.py` (or add a new `stitch.py` module) to chain N sampled 256-event windows into a single longer tape. Default N=10 → 2,560-event pseudo-days (~4 minutes of INTC wall-clock at the synthetic event rate).

**Implementation details.**

1. Sample N=10 windows using the existing DDIM sampler with the same (regime, seed) but incrementing a sub-seed or sequential chunk index.
2. For stitching, the critical operation is `mid_return` continuity: the last mid of window k becomes the anchor for window k+1. Concretely: `anchor_mid_{k+1} = anchor_mid_k * exp(sum(mid_return_k))`.
3. Concatenate all event rows, re-index `event_idx` monotonically, and re-compute `ts_ns` as continuous uniform spacing from window 0 through window N−1 (no gaps).
4. The stitched tape must still validate against the canonical schema. All 23 columns present.
5. Write one Parquet per (regime, seed) as before, just 10× longer.

**Boundary conditions to handle.**

- `mid_return[0]` of each subsequent window: set to 0.0 (fresh window start) or compute as `log(first_mid_{k+1} / last_mid_k)`. The former is simpler and introduces a discontinuity of at most a few bps; the latter is smoother but requires post-hoc adjustment. Recommend: set to 0.0 and document.
- Regime conditioning: all N windows use the same regime label. The agent sees a 4-minute tape under one consistent stress condition.
- Trade/quote statistics: after stitching, verify that the per-tape fill count rises to ~5-15 per agent (sufficient for worst-quartile to be non-degenerate).

**Success criteria.**

- Stitched tapes have 2,560 events each, validate against schema.
- At least 2 of 3 agents have non-zero 25th-percentile PnL across stitched tapes.
- `rank_by_worst_quartile` produces a non-degenerate ranking (not all zeros).

**Where it lives.** `src/diffmm/generator/stitch.py` (new) or extension to `decode.py`. Script: `scripts/14_stitch_synthetic.py` (new). Can also be integrated into `run_sampling_manifest.py`.

**Effort.** ~3 hours code + test. ~15 min re-sampling (same checkpoint, just decode differently). ~5 min re-running synthetic replay.

### Task 1.2 — Re-run the full evaluation with stitched tapes

**What.** After stitching, re-execute:

1. `scripts/22_replay_synthetic.py` on the stitched tapes.
2. `scripts/40_open_holdout.py` (already done — π\_truth doesn't change).
3. `scripts/41_hypothesis_test.py` with updated π\_diff.

**Success criteria.**

- π\_diff now uses worst-quartile ranking (no fallback warning in the log).
- New bootstrap p-value reported.
- Updated `figure7.pdf`.

**Effort.** ~30 min (mostly re-running scripts).

---

## Workstream 2 — Generator Validation and Interpretability

This is the analytical core that was skipped (Phase 2.4). It produces the evidence for "the generator is credible enough to trust the downstream ranking." Industry-standard generative model evaluation for financial time series uses three tiers: marginal distributional tests, temporal dependence structure, and cross-feature / conditional structure. We add a fourth tier for diffusion-specific model internals.

### Tier 1 — Marginal Distribution Tests

These answer: "does the generator produce events whose individual feature values look like real data?"

**Task 2.1 — Return distribution comparison**

Compute log-returns of the mid-price for both real validation tapes and v2 synthetic tapes. Compare:

1. **Histogram overlay** (log scale on y-axis). Real vs. synthetic mid-return distributions on the same axes. Expect heavy tails in both; the synthetic distribution should not be Gaussian even if the real one isn't.
2. **QQ plot** — real quantiles on x-axis, synthetic quantiles on y-axis. Perfect match = 45° line. Tail deviations are the diagnostic: if the synthetic QQ bends away from the line in the tails, the generator is under-representing extreme returns.
3. **Wasserstein-1 distance** (Earth Mover's Distance) between the real and synthetic return distributions. This is the LOB-Bench primary metric. Compute using `scipy.stats.wasserstein_distance`. Report the value and compare it to the inter-day Wasserstein variation within the real data (compute W-1 between each pair of real validation days and report the mean ± std). The generator passes if its W-1 is below 2× the inter-day real variation.
4. **Excess kurtosis** — real vs. synthetic. Financial returns have kurtosis >>3 (leptokurtic). Report both values. If synthetic kurtosis < 3, the generator is failing to produce fat tails.
5. **Skewness** — real vs. synthetic. Should be near zero for both (microstructure returns are approximately symmetric at short horizons).

**Implementation.** New function `compute_return_diagnostics(real_tapes, synth_tapes)` in `src/diffmm/eval/stylized.py`. Plots go in `src/diffmm/viz/validation.py`.

**Output.** `results/validation/INTC/return_distribution.{pdf,png}` — 4-panel figure: (a) histogram overlay, (b) QQ plot, (c) W-1 bar with inter-day reference band, (d) kurtosis/skewness comparison bar chart.

**Task 2.2 — Spread distribution comparison**

1. **Histogram** of `spread` (in dollars) for real vs. synthetic. For large-tick INTC, the real distribution should be heavily concentrated at $0.01 (one tick). If the synthetic distribution has significant mass at other spread values, the generator is failing to capture the large-tick constraint.
2. **Wasserstein-1** between real and synthetic spread distributions.
3. **Spread = 1 tick fraction** — what fraction of events have spread exactly equal to tick size ($0.01)? For real INTC this should be >90%. Report for both real and synthetic.

**Output.** `results/validation/INTC/spread_distribution.{pdf,png}` — 2-panel: (a) histogram, (b) W-1 with reference band.

**Task 2.3 — Size distribution comparison**

1. **Histogram** of `bid_sz` and `ask_sz` (log scale). Real LOBs have heavy-tailed size distributions (many small quotes, few very large ones).
2. **QQ plot** of log(bid\_sz) real vs. synthetic.
3. **Wasserstein-1** on log-transformed sizes.

**Output.** `results/validation/INTC/size_distribution.{pdf,png}`.

**Task 2.4 — Trade intensity comparison**

1. **Trade fraction** — fraction of events that are trades (vs. quote updates). Real INTC has ~1 trade per 10 quotes. Work3 §3.5 noted synthetic ratio was inverted (~1.2 trades per quote). This is a critical diagnostic — report it prominently and discuss implications.
2. **Inter-event time distribution** — histogram of Δt between consecutive events (in nanoseconds or milliseconds). Note: synthetic tapes use uniform 100ms spacing, so this comparison is informative about the decode model, not the generator. Document this limitation.
3. **Trade size distribution** — histogram of `trade_sz` conditional on `event_type == 'trade'`.

**Output.** `results/validation/INTC/trade_intensity.{pdf,png}`.

### Tier 2 — Temporal Dependence Structure

These answer: "does the generator produce sequences with realistic dynamics, not just realistic marginals?"

**Task 2.5 — Volatility clustering (absolute-return ACF)**

1. Compute `|mid_return|` for real and synthetic.
2. Plot the autocorrelation function (ACF) out to lag 200 events for both, with 95% bootstrap confidence bands on the real ACF.
3. The well-known stylized fact: absolute returns exhibit long-memory / slow-decaying ACF (power-law ~lag^{-0.3}). If the synthetic ACF drops to zero quickly, the generator is failing to capture volatility clustering.
4. **Quantitative test**: compute the ACF at lag 50 for both. The proposal's kill criterion is: synthetic ACF at lag 50 deviates by >50% from real. Report the deviation.

**Implementation.** Use `statsmodels.tsa.stattools.acf` with `nlags=200`. Bootstrap by resampling validation days.

**Output.** `results/validation/INTC/acf_abs_return.{pdf,png}` — real and synthetic ACF curves with confidence band.

**Task 2.6 — Trade-sign autocorrelation (order flow persistence)**

1. Compute ACF of `trade_sign` (restricted to trade events only) out to lag 100.
2. Real markets exhibit positive short-lag autocorrelation in trade signs (order-splitting creates runs of same-side trades). This is the Bouchaud–Farmer–Lillo "long memory of order flow" effect.
3. If synthetic trade signs are iid (ACF ≈ 0 at all lags), the generator is not capturing order-flow persistence.

**Output.** `results/validation/INTC/acf_trade_sign.{pdf,png}`.

**Task 2.7 — Mid-price return autocorrelation**

1. ACF of raw `mid_return` out to lag 50.
2. Real microstructure returns are slightly negatively autocorrelated at lag 1 (bid-ask bounce) and approximately zero thereafter.
3. If synthetic returns are positively autocorrelated, the generator is producing trending sequences that don't exist in real data — a major failure.

**Output.** `results/validation/INTC/acf_return.{pdf,png}`.

**Task 2.8 — Realized volatility signature plot**

1. Compute realized variance at multiple sampling frequencies: every 1, 5, 10, 25, 50, 100, 256 events.
2. Plot RV vs. sampling frequency for real and synthetic.
3. The real signature should show microstructure noise at high frequencies (RV increases as frequency increases) and converge at lower frequencies. The synthetic pattern should match.
4. This is the Andersen–Bollerslev–Diebold signature plot, a standard industry diagnostic for synthetic data quality.

**Output.** `results/validation/INTC/rv_signature.{pdf,png}`.

### Tier 3 — Cross-Feature and Conditional Structure

These answer: "does the generator preserve the relationships between features that matter for market-making?"

**Task 2.9 — Conditional return on imbalance (the OFI → return signal)**

This is the single most important cross-feature test for the project. It tests whether the synthetic data preserves the mechanism that Agent A2 exploits.

1. Bin imbalance `I_t = (bid_sz - ask_sz)/(bid_sz + ask_sz)` into 10 equal-width bins on [−1, +1].
2. For each bin, compute the mean `mid_return` over the next Δ events (Δ = 10, 50).
3. Plot E[r\_{t+Δ} | I\_t] for real and synthetic on the same axes.
4. **Sign test**: the relationship should be monotonically positive (higher bid-side imbalance → positive future returns). This is Cont–Kukanov–Stoikov's core finding. If the synthetic relationship has the wrong sign, the generator is producing data that violates a fundamental microstructure regularity. The proposal's kill criterion: conditional mean return given imbalance has wrong sign.
5. **Slope test**: fit a linear regression `E[r | I] = α + β·I` on both real and synthetic. Compare β values. Report R² for both.

**Output.** `results/validation/INTC/conditional_return_imbalance.{pdf,png}` — overlay plot with real and synthetic curves + confidence bands.

**Task 2.10 — Joint spread-imbalance distribution**

1. Compute the 2D histogram of (spread, imbalance) for real and synthetic.
2. For INTC (large-tick), most mass should be at spread = $0.01 regardless of imbalance.
3. Compute the Jensen-Shannon divergence between the 2D real and synthetic histograms as a single summary number.

**Output.** `results/validation/INTC/joint_spread_imbalance.{pdf,png}` — side-by-side 2D heatmaps.

**Task 2.11 — Conditional moments by regime label**

This tests whether conditioning actually works — the core of the "stress-directed generation" claim.

1. For each regime label ∈ {base, high\_vol, toxic, thin}:
   - Compute on **real validation** days (partitioned by their observed regime labels): mean(|mid\_return|), mean(spread), std(mid\_return), mean(trade\_sz), trade\_fraction.
   - Compute the same on **synthetic** tapes generated under that regime condition.
2. Present as a table: rows = regimes, columns = statistics, cells = (real, synthetic) pairs.
3. For each (regime, statistic) pair, compute the z-score: `z = (synth_mean - real_mean) / real_std`. If |z| > 2, conditioning is producing moments outside the plausible range.
4. Test **monotonicity**: does mean(|mid\_return|) increase from base → high\_vol → toxic? Does trade\_fraction differ? If regime labels don't separate in the synthetic data, the conditioning is weak (which you've already observed — document it with numbers).

**Output.** `results/validation/INTC/conditional_moments_table.csv` + `results/validation/INTC/conditional_moments_heatmap.{pdf,png}`.

### Tier 4 — Diffusion Model Internals

These are not standard LOB-Bench checks. They are model-interpretability analyses that explain *why* the generator produces what it produces and where it's weakest.

**Task 2.12 — Per-timestep denoising quality (v1 vs. v2 comparison)**

You already have the per-timestep ε-prediction std from the diagnostic sweep (Work2 §3.4, Work3 §2.1). Formalize this into a proper figure:

1. For both v1 and v2 checkpoints, compute ε-prediction statistics at timesteps t ∈ {20, 50, 100, 200, 300, 500, 700, 900, 999}.
2. For each t, report: mean(ε\_pred), std(ε\_pred), MSE(ε\_pred, ε\_true) averaged over 100 random validation windows.
3. Plot: x = timestep t, y = MSE(t). Two curves: v1 (uniform MSE training) and v2 (Min-SNR). The v2 curve should be flatter (more uniform quality across timesteps), with lower MSE in the critical middle range (t ≈ 200–500).
4. Overlay the Min-SNR weight function `min(SNR(t), 5) / SNR(t)` on a secondary y-axis to show how training loss was rebalanced.

**Implementation.** Extend `scripts/utils/diagnose_sampling.py` to accept a list of checkpoints and produce a comparison figure.

**Output.** `results/validation/INTC/per_timestep_denoising.{pdf,png}`.

**Task 2.13 — Training loss curves (v1 vs. v2)**

1. Load the CSVLogger outputs from both training runs.
2. Plot train loss vs. step for both v1 and v2 on the same axes (log scale on y).
3. Annotate: epoch boundaries, val-loss checkpoints, the learning-rate schedule.
4. On a separate panel or inset: plot val loss vs. epoch for both.

This shows the reader that (a) training converged, (b) Min-SNR changed the loss landscape, (c) v2 achieved lower val loss than v1.

**Output.** `results/validation/INTC/training_curves.{pdf,png}`.

**Task 2.14 — x0\_clip activation rate**

1. During sampling, count what fraction of events trigger the `x0_clip=4.0` clamp.
2. Report separately per regime and per diffusion timestep.
3. A well-trained model should have <1% clamp activation. Higher rates indicate the model is being rescued by the clamp rather than producing good predictions.
4. Compare v1 and v2 clamp rates. v1 should be much higher (justifying the v2 retraining).

**Implementation.** Add a `clip_count` accumulator inside `ddim_sample`. Can be done with a one-line `(x0_pred.abs() > x0_clip).sum()` per step.

**Output.** `results/validation/INTC/x0_clip_activation.csv` + summary in the report.

**Task 2.15 — Attention weight visualization**

1. For 5 representative validation windows (one per regime + one average), extract the self-attention weights from the transformer denoiser at 3 diffusion timesteps (t=900 early, t=500 middle, t=100 late).
2. Plot attention weight matrices (256×256 heatmaps) for the first attention head.
3. Look for: (a) diagonal dominance (each event attends to nearby events — expected), (b) column spikes (certain events are universally attended to — might indicate trade events or spread changes that anchor the denoising), (c) differences across timesteps (early denoising should be diffuse, late denoising should be focused).
4. If attention patterns differ by regime, that's evidence the FiLM conditioning is influencing the transformer's internal processing, not just scaling outputs.

**Implementation.** Register forward hooks on the attention layers in `TradesStyleDenoiser`. Store attention weights during a forward pass. Plot with `matplotlib.imshow`.

**Output.** `results/validation/INTC/attention_maps/` — one figure per (window, timestep) combination. Selected examples in the report.

**Task 2.16 — Regime embedding space analysis**

1. Extract the learned regime embeddings from `RegimeEmbedding` for all (c\_vol, c\_vpin, c\_imb, c\_tod) combinations (3×3×3×3 = 81 cells, though many are empty).
2. Concatenate the per-axis embeddings as the model does at inference.
3. Project to 2D with PCA or t-SNE.
4. Color by regime label (base/high\_vol/toxic/thin).
5. Check: do the four regime clusters separate in embedding space? If they overlap completely, the model's FiLM layers can't distinguish regimes. If they separate, the conditioning has learned meaningful structure.
6. Also compare to the null embedding (the ∅ token used for CFG dropout). Its position relative to the regime clusters indicates what "unconditional" means to the model.

**Implementation.** Direct weight extraction from `model.regime_embed`. No inference needed.

**Output.** `results/validation/INTC/regime_embedding_space.{pdf,png}`.

**Task 2.17 — FiLM modulation magnitude analysis**

1. For each FiLM layer (there are 2 in the current architecture — start and end), extract the learned γ and β vectors for each regime condition.
2. Compute `||γ(c) - 1||` and `||β(c)||` for each regime c. These measure how much the FiLM layer deviates from identity for each condition.
3. Plot as a grouped bar chart: x = regime, bars = (γ deviation, β deviation) for each FiLM layer.
4. If all bars are near zero, the FiLM layers aren't doing anything — conditioning is cosmetic. If bars differ across regimes, FiLM is actively modulating the hidden representations.

**Implementation.** Forward pass with each regime condition through the model, hook the FiLM layers, extract γ and β.

**Output.** `results/validation/INTC/film_modulation.{pdf,png}`.

**Task 2.18 — Guidance weight sensitivity sweep**

1. Generate 50 synthetic windows at each guidance weight w ∈ {0, 0.5, 1, 2, 3, 5} under the "high\_vol" regime condition.
2. For each w, compute: mean(|mid\_return|), mean(spread), std(mid\_return), trade\_fraction.
3. Plot each statistic vs. w.
4. Expect: at w=0 (unconditional), statistics match the training-set average. As w increases, statistics should move toward the "high\_vol" conditional moments from the real data. At very high w, they may overshoot (a known CFG failure mode — "over-conditioned" samples that are regime-caricatures rather than realistic).
5. Identify the sweet spot where conditional moments best match real conditional moments. This justifies the choice of w=1 (or suggests a better value).

**Implementation.** Modify `diagnose_sampling.py` to sweep w and compute per-regime statistics.

**Output.** `results/validation/INTC/guidance_sweep.{pdf,png}`.

### Summary table — generator validation

| Check | Metric | Pass criterion | Proposal reference |
|---|---|---|---|
| Return distribution | Wasserstein-1 | < 2× inter-day real W-1 | §4.5 / §7.1 |
| Return tails | Excess kurtosis | Synthetic kurtosis > 3 | §7.1 |
| Spread distribution | W-1 + 1-tick fraction | Synth 1-tick frac > 80% (INTC) | §7.1 |
| Volatility clustering | ACF(\|r\|) at lag 50 | < 50% deviation from real | §7.1 kill criterion |
| OFI → return signal | E[r \| I] sign | Correct (positive β) | §7.1 kill criterion |
| Conditional moments | z-score per (regime, stat) | \|z\| < 2 for majority | §7.2 |
| Crossed books | Rate | < 10⁻⁵ | §7.1 kill criterion |
| Per-timestep quality | MSE(t) v2 vs v1 | v2 flatter, lower at middle-t | (internal) |
| Clip activation | Fraction | < 1% at v2 | (internal) |

**Total effort for Workstream 2.** ~2 days. The analysis code is mostly compute-light (statistics on already-generated data). The attention and FiLM analyses require forward passes through the model but no GPU training.

---

## Workstream 3 — Agent Interpretability and PnL Decomposition

The current result reports one number per agent (Sharpe). This section adds the analytical depth to explain *why* agents rank the way they do.

### Task 3.1 — PnL decomposition: spread capture vs. adverse selection vs. inventory risk

For each agent, decompose daily PnL into three components:

1. **Gross spread capture** = sum over all fills of (half-spread earned). For a buy fill at price p when the agent's ask was a, gross capture on that fill is `a - mid_at_fill`. For a sell fill at price p when the agent's bid was b, it's `mid_at_fill - b`. Sum across all fills in the day.
2. **Adverse selection cost** = sum over all fills of (mark-to-market loss from mid-move after fill). For each fill at time t, compute `mid_{t+Δ} - mid_t` (for a buy fill; negate for a sell fill), where Δ is a short horizon (e.g., 50 events or 5 seconds). This measures how much the mid moved against the agent after each fill.
3. **Inventory cost** = terminal liquidation PnL − (sum of fill PnLs at execution prices). This captures the drift-vs-inventory exposure: did holding inventory help or hurt?

Verify: `PnL_total = spread_capture - adverse_selection + inventory_cost` (approximately; rounding and liquidation details may cause small discrepancies).

**Implementation.** Extend `BacktestResult` to store per-fill details: `fill_time`, `fill_side`, `fill_price`, `mid_at_fill`, `mid_after_fill`. New function `decompose_pnl(result)` in `src/diffmm/eval/metrics.py`.

**Output.** 
- `results/analysis/INTC/pnl_decomposition_real.csv` — per (agent, day) decomposition on val days.
- `results/analysis/INTC/pnl_decomposition_synth.csv` — per (agent, regime, seed) decomposition on synthetic tapes.
- Figure: grouped bar chart of (spread\_capture, adverse\_selection, inventory\_cost) × 3 agents, for real val and each synthetic regime side by side. This is the microstructure story of why agents differ.

**Interpretation guide.** A1 (bare AS) should have the highest adverse selection cost (no protective mechanisms). A3 (VPIN-gated) should have lower adverse selection in toxic regimes (it widens the spread, reducing fills during informed-flow episodes) but also lower spread capture (fewer fills total). A2 (OFI-skew) should have intermediate adverse selection (it leans away from toxic flow direction).

### Task 3.2 — Fill rate analysis by regime

1. Compute fills-per-event for each agent on each regime (real val + synthetic).
2. Plot as a heatmap: rows = agents, columns = regimes, cells = fill rate.
3. Key question: does A3's VPIN-gating actually reduce fill rate in toxic regimes vs. base? If not, the gating isn't working as intended.

**Output.** `results/analysis/INTC/fill_rate_heatmap.{pdf,png}`.

### Task 3.3 — Inventory distribution analysis

1. For each agent on each regime, compute the distribution of inventory `q_t` over the backtest.
2. Report: mean(|q|), max(|q|), std(q), fraction of time at inventory limit.
3. Plot inventory trajectories for each agent on a selected real val day (the most volatile one) and a synthetic toxic day side by side.
4. Key question: does A1's inventory swing wildly while A3's stays tighter? Does A2's OFI-skew cause it to accumulate directional inventory?

**Output.** `results/analysis/INTC/inventory_analysis.{pdf,png}` — 3-panel (one per agent) showing inventory time-series on a selected stress day.

### Task 3.4 — Quote placement analysis

1. For each agent, compute the distribution of posted quotes relative to the NBBO:
   - `bid_distance = mid - bid_posted` (in ticks)
   - `ask_distance = ask_posted - mid` (in ticks)
2. Histogram these distances for each agent.
3. Key question: does the AS optimal spread formula produce quotes that are competitive (within 1-2 ticks of mid) or wide (5+ ticks)? How does this change across regimes?
4. Report the "time at best" fraction: what fraction of events does each agent have a quote at the NBBO? (For INTC with 1-tick spread, posting at 1-tick distance from mid means posting at the NBBO.)

**Output.** `results/analysis/INTC/quote_placement.{pdf,png}`.

### Task 3.5 — Agent sensitivity analysis (calibration parameter sweeps)

1. Sweep γ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} for A1 on the real validation set. Report Sharpe at each γ.
2. Sweep β\_OFI ∈ {0.1, 0.5, 1.0, 2.0, 5.0} for A2. Report Sharpe and adverse-selection cost.
3. Sweep α ∈ {0.5, 1.0, 2.0, 3.0, 5.0} for A3. Report Sharpe and worst-day PnL.
4. Present as line plots: x = parameter, y = Sharpe (or other metric).
5. This answers: are the agent differences robust to calibration, or are they artifacts of the chosen defaults?

**Implementation.** Loop over parameter grids in `scripts/20_calibrate_agents.py` with overrides. Run `scripts/21_replay_real.py` for each. CPU-only, fast.

**Output.** `results/analysis/INTC/agent_sensitivity.{pdf,png}` — 3 panels (one per agent).

### Task 3.6 — Real-vs-synthetic comparison of agent behavior

1. For each agent, compute the same metrics (mean PnL, Sharpe, fill rate, adverse selection, inventory variance) on both real val tapes and synthetic tapes from the "base" regime.
2. Present as a paired comparison table. If base-regime synthetic produces similar agent behavior to real val, the synthetic environment is a reasonable proxy. If not, document the discrepancy and discuss why.

**Output.** `results/analysis/INTC/real_vs_synth_agent_comparison.csv`.

**Total effort for Workstream 3.** ~1.5 days. Mostly analysis code operating on already-computed backtest results. The per-fill detail extension to `BacktestResult` is the largest single change.

---

## Workstream 4 — Fourth Agent (Statistical Power)

### Task 4.1 — Implement A0: Constant-Spread Baseline

A naive agent that always posts a fixed spread (1 tick on each side of mid) with no inventory adjustment, no OFI, no VPIN. This is the zero-intelligence market-making baseline.

```python
class ConstantSpreadAgent(MarketMakingAgent):
    def on_event(self, state: LOBState) -> Quote:
        return Quote(
            bid_px=round_to_tick(state.mid - self.tick_size),
            ask_px=round_to_tick(state.mid + self.tick_size),
            bid_sz=self.base_size,
            ask_sz=self.base_size,
        )
```

No calibration needed. `tick_size` and `base_size` from config.

### Task 4.2 — Re-run all replays with 4 agents

1. Re-run `scripts/21_replay_real.py` with A0 added.
2. Re-run `scripts/22_replay_synthetic.py` with A0 added.
3. Re-run `scripts/40_open_holdout.py` with A0 added. (Note: this means opening the holdout a second time. Document this clearly. Alternatively, if the holdout metrics CSV already has per-event data, you can replay A0 on the stored tapes without re-opening. Decide based on what the firewall allows.)
4. Re-run `scripts/41_hypothesis_test.py`.

**Impact on statistical power.** With 4 agents: 24 possible rankings, Spearman ρ ∈ {-1, -0.8, -0.6, ..., +0.8, +1} (11 values vs. 5 with 3 agents). Bootstrap CIs tighten substantially. The p-value becomes more informative.

### Task 4.3 — Optional: Implement A4: Microprice-Based Agent

If time permits, implement a fifth agent using Stoikov (2018) microprice:

```
micro = ask * I⁺ + bid * (1 - I⁺)
where I⁺ = bid_sz / (bid_sz + ask_sz)
```

Use microprice instead of mid as the fair-value anchor for the AS reservation price. This tests whether a better fair-value estimator improves robustness under stress.

**Effort.** A0: ~1 hour (trivial). A4: ~2 hours. Re-running replays: ~30 min.

---

## Workstream 5 — TSLA Pipeline

### Task 5.1 — Re-finalize TSLA canonical tapes

TSLA intermediate Parquets already exist from Phase 1. Re-run Stage A2 with the v2 schema (adds relative features):

```bash
python scripts/03_finalize_tapes.py data=tsla_top
```

~1 min. Verify: `data/processed/{train,val,holdout}/TSLA_*.parquet` exist with 23 columns.

### Task 5.2 — Train TSLA v2 generator

Same architecture and hyperparameters as INTC v2. Submit to cluster:

```bash
sbatch slurms/ph2_finetune.slurm data=tsla_top \
    generator.training.max_epochs=10 \
    experiment_name=tsla_v2
```

Expected: ~3-4 hours on L40S (TSLA has more events per day than INTC).

### Task 5.3 — Sample TSLA synthetic data

Same manifest-driven sampling as INTC:

```bash
python scripts/utils/run_sampling_manifest.py \
    --manifest data/manifests/ph2_sample_TSLA.json \
    --checkpoint checkpoints/tsla_v2/best.ckpt \
    --norm-stats data/processed/norm_stats_TSLA.json \
    --out-root /orcd/scratch/.../synthetic_v2/TSLA/
```

~15 min. Then stitch (if Workstream 1 is done).

### Task 5.4 — Calibrate agents on TSLA

```bash
python scripts/20_calibrate_agents.py data=tsla_top
```

TSLA has different microstructure from INTC: smaller effective tick, higher volatility, higher trade rate. Expect different κ and σ values.

### Task 5.5 — Run full TSLA evaluation

```bash
python scripts/21_replay_real.py data=tsla_top
python scripts/22_replay_synthetic.py data=tsla_top
python scripts/40_open_holdout.py data=tsla_top
python scripts/41_hypothesis_test.py data=tsla_top
```

### Task 5.6 — TSLA validation diagnostics

Run the same Workstream 2 diagnostics on TSLA. Key differences to expect:
- Spread distribution: TSLA should NOT be concentrated at 1 tick (it's a small-tick stock). The spread distribution should be wider and more continuous.
- Return distribution: higher kurtosis (TSLA is more volatile).
- OFI → return signal: may be weaker or stronger depending on TSLA's microstructure.

### Task 5.7 — Cross-ticker comparison

1. Present the hypothesis test results side by side: INTC vs. TSLA.
2. Do the rankings agree? (Same agents rank similarly on both tickers → robust methodology.)
3. Do the generators have similar strengths/weaknesses? (Both fail on the same stylized facts → systematic limitation. Different failures → ticker-specific.)

**Total effort for Workstream 5.** ~1 day (mostly GPU queue time for training).

---

## Workstream 6 — Report

### Structure (per the project rubric)

1. **Title** — "Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual Top-of-Book Tapes" (or the shorter version from the submitted proposal).

2. **Executive summary** (~250 words) — plain English. Central question, method, key finding (direction positive, statistically inconclusive), implication (synthetic stress testing shows promise for strategy evaluation but needs longer horizons and more agents for statistical power).

3. **Project definition** — the replay problem (Section 2.1 of the proposal), why diffusion models (Section 2.2), the central hypothesis statement.

4. **Scope definition** — in-scope / out-of-scope (updated from proposal to reflect actual decisions: trained from scratch instead of fine-tuning TRADES, NBBO-only instead of depth, top-of-book fill model).

5. **Data** — TAQ via WRDS, schema (Table 1 from proposal), cleaning rules (Work1 §2), NBBO construction (Work1 §4.3), Lee-Ready signing, regime feature engineering (vol, VPIN, imbalance, TOD), train/val/holdout splits.

6. **Methods**
   - 6.1 Generator architecture (TRADES-style transformer denoiser, FiLM conditioning, 10M params)
   - 6.2 Price-invariant feature representation (the v1 → v2 story, why absolute prices fail)
   - 6.3 Min-SNR loss weighting (the per-timestep quality problem and the fix)
   - 6.4 DDIM sampling with CFG and x0\_clip
   - 6.5 Generator validation (Workstream 2 results)
   - 6.6 Market-making agents (A0/A1/A2/A3 definitions, calibration)
   - 6.7 Fill simulator (conservative queue model, provenance-blind)
   - 6.8 Experimental design (regime factorial, evaluation matrix, metrics)
   - 6.9 Central hypothesis test (holdout firewall, bootstrap)

7. **Results**
   - 7.1 Generator validation results (Workstream 2 figures and pass/fail table)
   - 7.2 Agent performance on real data (val Sharpe, PnL decomposition)
   - 7.3 Agent performance on synthetic data (regime × agent matrix, PnL decomposition)
   - 7.4 Central hypothesis result (Figure 7, bootstrap CI, p-value)
   - 7.5 TSLA results (if completed)

8. **Discussion**
   - 8.1 What worked (price-invariant features, Min-SNR, holdout firewall)
   - 8.2 What didn't (regime conditioning is weak, short synthetic windows, negative agent Sharpe)
   - 8.3 Limitations (3 agents, 20 holdout days, top-of-book only, no market impact)
   - 8.4 Extensions (longer windows, more agents, depth data, interactive simulation)

9. **References** — the 20 papers from the reading plan + any additional references.

10. **AI disclosure** — Claude used for architecture planning, code review, and drafting. All code written by the team. All numbers from committed code on committed data.

11. **Appendix** — OFI sign conventions, software stack, key code snippets, additional figures, full bug ledger (summary).

### Figures list (minimum)

| Figure | Content | Source |
|---|---|---|
| 1 | Pipeline diagram | TikZ or hand-drawn |
| 2 | Diffusion forward/reverse schematic | TikZ |
| 3 | Denoiser architecture | TikZ |
| 4a | Return distribution (real vs. synth) | Workstream 2 Task 2.1 |
| 4b | Volatility clustering ACF | Workstream 2 Task 2.5 |
| 4c | Spread distribution | Workstream 2 Task 2.2 |
| 4d | Conditional return on imbalance | Workstream 2 Task 2.9 |
| 5 | Per-timestep denoising quality (v1 vs v2) | Workstream 2 Task 2.12 |
| 6 | Training curves (v1 vs v2) | Workstream 2 Task 2.13 |
| 7 | Agent performance by regime (grouped bar) | Workstream 3 |
| 8 | PnL decomposition by agent and regime | Workstream 3 Task 3.1 |
| 9 | Tail-risk boxplots | Workstream 3 |
| 10 | Central result: ρ\_hist vs ρ\_diff | Phase 5 output |
| 11 | Regime embedding space (PCA) | Workstream 2 Task 2.16 |
| 12 | Guidance weight sensitivity | Workstream 2 Task 2.18 |

### Tables list (minimum)

| Table | Content |
|---|---|
| 1 | TAQ fields used |
| 2 | Filter rules |
| 3 | Train/val/holdout splits |
| 4 | Generator hyperparameters (v1 vs v2) |
| 5 | Validation pass/fail summary |
| 6 | Agent definitions and parameter values |
| 7 | Calibrated parameters (κ, σ, γ) |
| 8 | Evaluation matrix: Sharpe × (agent, regime) |
| 9 | PnL decomposition |
| 10 | Central result: rankings and ρ values |
| 11 | Conditional moments (real vs synthetic by regime) |

**Effort.** ~3 days for writing. Start with the methods section (most of the content is in Work0–Work3). Results section writes itself from the figures. Discussion is the most intellectually demanding section.

---

## Workstream 7 — Repository Cleanup and Reproducibility

### Task 7.1 — Pin all dependency versions

`pip freeze > requirements-lock.txt` on the cluster after the final run. Verify that `environment.yml` matches.

### Task 7.2 — Write REPRODUCE.md

Step-by-step from raw data download to Figure 10 (the central result). Include:
- Exact WRDS query / LOBSTER download instructions.
- All `sbatch` commands in order.
- All random seeds.
- Expected output hashes for key intermediate files (SPLITS.json, quantiles.json, norm\_stats.json).

### Task 7.3 — Data manifests

Verify `data/raw/MANIFEST.md5` is complete. Add manifests for processed data and synthetic data.

### Task 7.4 — Cross-team reproducibility check

Have a teammate who didn't write the pipeline attempt to reproduce the central figure from a clean clone.

**Effort.** ~1 day.

---

## Execution Timeline

Assuming ~5 working days remain:

| Day | Morning | Afternoon |
|---|---|---|
| **1** | Workstream 1 (window stitching + re-run evaluation) | Workstream 4 (A0 agent + re-run all with 4 agents) |
| **2** | Workstream 2, Tier 1+2 (marginals + temporal structure) | Workstream 2, Tier 3 (conditional structure) |
| **3** | Workstream 2, Tier 4 (model internals) | Workstream 3 (PnL decomposition + agent analysis) |
| **4** | Workstream 5 (TSLA: finalize, train, sample, replay — GPU queue determines pace) | Workstream 6 (start report: methods + results sections) |
| **5** | Workstream 6 (finish report: discussion + executive summary) | Workstream 7 (repo cleanup + reproducibility) |

If time is shorter than 5 days, cut in this order (last = cut first):

1. Cut Workstream 5 (TSLA) — single-ticker result is still valid.
2. Cut Workstream 2 Tier 4 tasks 2.15–2.17 (attention/FiLM/embedding analysis) — nice-to-have, not essential.
3. Cut Workstream 4 Task 4.3 (microprice agent) — A0 alone gives the statistical power boost.
4. Cut Workstream 3 Tasks 3.4–3.5 (quote placement + sensitivity sweep) — PnL decomposition is the priority.
5. Never cut Workstream 1 (window stitching) — it fixes the worst-quartile degeneracy.
6. Never cut Workstream 2 Tiers 1–3 (basic validation) — the report needs these.
7. Never cut Workstream 6 (the report) — it IS the deliverable.