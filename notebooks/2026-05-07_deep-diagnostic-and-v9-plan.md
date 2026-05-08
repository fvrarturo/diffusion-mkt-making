# Lab notes — Deep diagnostic + v9 (dequantization) plan

**Date:** 2026-05-07
**Branch:** `work_4may`
**Goal:** Diagnose v8's marginal failure (kurt 2.6 vs predicted 400-1000), implement the fix, plan v9.

## TL;DR

The v8 failure has a clean root cause: **the copula CDF is discontinuous at point masses** (mid_return is 89% exact zeros for INTC; trade_sz is 95% zeros). The probit Φ⁻¹ collapses each point mass to a single z-value, the diffusion model trains on near-constant z targets for these features, and the inverse CDF can't restore heavy tails.

**Fix: dequantization.** At fit time, spread point masses uniformly over a thin band [v−ε, v+ε] before building the empirical CDF. At training-time normalize, add fresh U(−ε,+ε) noise per call so the model sees smoothly-distributed z. At sample-time denormalize, snap any value within ε of v_pm back to v_pm. Standard technique from the normalizing-flow literature (Theis et al. 2016).

Implementation: ~70 LOC in `src/diffmm/data/copula_transform.py`, all 11 copula tests pass (including 4 new dequantization round-trip tests). Config knob: `copula.dequantize: true` (default).

## What the deep diagnostic showed

Ran `scripts/60_deep_diagnostic.py` across v2 / v7_b / v8 / v8_b on real INTC val data. Key findings:

### 1. v8 z-space is collapsed for point-mass features

`v8 / mid_return`:
- z-mean = -2.55, z-std = 0.30
- z-p01 = -2.61, z-p99 = -2.55 (range of width 0.06!)
- KS stat vs N(0,1) = 0.99 (clearly not Gaussian)

The model emits a near-constant z, far from N(0,1). After inverse CDF, this produces small-magnitude returns clustered around 0 — kurtosis 2.6 instead of the predicted 400-1000.

Same pattern for `trade_sz` (z-mean=2.22, z-std=0.46) and `trade_dist` (z-mean=-2.21).

### 2. v8 has STRONG return autocorrelation (lag-1 ACF = +0.33 vs real -0.05)

| Lag | Real | v2 | v7_b | **v8** |
|---|---|---|---|---|
| 1 | -0.05 | +0.04 | +0.01 | **+0.33** |
| 10 | 0.00 | -0.00 | -0.00 | **+0.29** |
| 100 | -0.00 | +0.00 | +0.00 | **+0.19** |

Returns drift slowly in v8 — the visible symptom of the model emitting a slowly-varying near-constant z. Real returns are essentially uncorrelated (martingale property of fair-priced returns).

v8's price ACF goes negative at lag 100 (−0.21) — prices mean-revert on a 100-event timescale, which is unphysical.

### 3. Sizes preserve ACF; returns/prices/trade_sign break

v8's `bid_sz`, `ask_sz` ACFs match real reasonably. The temporal failure is feature-specific to returns, prices (downstream of returns), and trade_sign.

### 4. Regime-conditional collapse (severity varies)

v8 mid_return kurt by regime: base=62.8, high_vol=0.5, thin=−0.5, toxic=8.9. The model is more broken in `high_vol` and `thin` regimes. Aggregate kurt 2.6 averages these out.

### 5. v8_b is fully degenerate (confirmed)

100% degenerate windows, 0 trades, all features zero variance. Sampling-time collapse.

## Root-cause analysis (the v9 diagnosis)

The v8 failure is NOT what we initially hypothesized:

- ❌ Min-SNR sub-Gaussian compression
- ❌ Sampler discretization
- ❌ AdaLN-Zero failure (that's v8_b's specific issue)

The actual cause is **CDF discontinuity at point masses**, a property of the data, not the model.

INTC `mid_return` distribution:
- 88.85% of events have mid_return = 0 EXACTLY (large-tick stock; mid doesn't change between most consecutive events)
- 11.15% are continuous values in [-0.0066, +0.0043]

When `CopulaTransform.fit` builds the empirical CDF on this distribution:
- The CDF has a flat plateau around x=0 (89% of mass at one point)
- `searchsorted(side='left')` maps every x=0 input to the same rank
- `Φ⁻¹` then collapses all 89% to a single z value (e.g. z ≈ −1.6)

The diffusion model trains on data where 89% of the mid_return target z is exactly z = −1.6 (a constant). The model correctly learns this — it predicts ε that, after the schedule, produces z ≈ −1.6 for mid_return on most events.

When the inverse CDF then sees synth z's clustered around the same constant, it returns the corresponding x ≈ 0 (which is correct), plus tiny linear-interpolation noise around that. So synth mid_return is close to 0 with small spread → low kurtosis → no heavy-tail recovery.

The high return autocorrelation is the cherry on top: when the model emits z values centered at a constant with small drift over time, the decoded returns drift slowly — not a martingale.

This is a clean, falsifiable diagnosis. The copula approach doesn't fail "in general" — it fails specifically on zero-inflated mixed distributions, which describes most financial microstructure features.

## The dequantization fix

Standard technique from normalizing-flow literature (Theis et al. 2016, "A note on the evaluation of generative models"). Three changes:

**At fit time:** detect point masses (frequencies ≥ `point_mass_threshold`, default 5%). For each point mass v_pm, compute ε = (smallest non-zero distance from v_pm to any other value, excluding other point masses) / 2. Replace point-mass values in the training data with v_pm + U(−ε, +ε) before sorting/storing the empirical CDF.

**At normalize (training):** for any input value equal to a stored v_pm, add fresh U(−ε,+ε) noise per call. The model now sees a smoothly-distributed z target across the full point-mass region.

**At denormalize (sampling):** apply inverse CDF as before; then snap any sampled value within ε of v_pm back to v_pm. This restores the zero-inflated structure exactly (89% of synth mid_return will be 0).

For INTC mid_return: smallest 1-tick move is 0.01/40 ≈ 2.5e-4, so ε ≈ 1.25e-4. The 89% of zeros get spread uniformly over [-1.25e-4, +1.25e-4] in x-space; after the CDF that's a thin Gaussian band in z-space. After inverse + snap, sampled mid_return = 0 wherever the model predicts z in the corresponding range, exactly as the data structure requires.

## Predicted outcomes for v9 / v9_b

**v9** (copula-dequantized + FiLM + ε-pred, the "fix v8's preprocessing" run):
- Kurtosis: 100-800 (matches real range; the inverse CDF can now actually reach the tails)
- mid_return ACF lag-1: ~0 (martingale restored — model has well-distributed z to work with)
- ρ_diff: ≥ +0.40, hopefully near +0.80 like v8 (since FiLM+copula already produced the correct A2>A1 ranking)
- frac_zero on synth mid_return: ~0.80 (close to real 0.89)

**v9_b** (copula-dequantized + FiLM + v-pred, the "does copula rescue v-pred?" run):
- Same dequantization, same FiLM
- v-prediction tests whether the broken-correlations failure of v6/v7 was caused by the parameterization OR by the marginal mismatch
- If v-pred + dequantized copula gives ρ_diff ≥ +0.40, parameterization was incidental; the copula handles what was previously the v-pred failure mode
- If ρ_diff << +0.40, v-pred has its own joint-structure-breaking mechanism independent of marginals

These are the cleanest A/B tests in the workstream so far: same data preprocessing, same conditioning, only the prediction target differs.

## Decision tree for v9 outcomes

| Outcome | What it means | Next step |
|---|---|---|
| v9 kurt 100-800, ρ_diff ~+0.80 | Dequantization fix worked. Phase E delivers the predicted result. | Ship v9. Write up. |
| v9 kurt 100-800, ρ_diff < +0.40 | Marginals fixed but ranking lost. Copula-dequant changes the joint structure too much. | Investigate Section B (cross-feature corr) of next deep-diagnostic. |
| v9 kurt < 50 | Dequantization didn't fix it; some other mechanism is suppressing model output magnitude. | Re-examine the diagnostic; consider Min-SNR or sampler discretization (would explain the remaining gap). |
| v9_b ≥ v9 on ρ_diff | v-pred is rescued by copula-dequant. | Interesting result — write up. v-pred wasn't the problem alone; it was v-pred PLUS marginal mismatch. |
| v9_b << v9 on ρ_diff | v-pred breaks something even with clean marginals. | Confirms the v-pred-breaks-joints story from Work6_FINAL §9 was real, not a side-effect of marginal mismatch. |

## File index for this session

- `src/diffmm/data/copula_transform.py` — added `point_masses` field, `_compute_dequant_eps` helper, dequantization in `fit/normalize/denormalize`, JSON schema for save/load.
- `tests/test_validation.py` — 4 new tests: detection, round-trip, well-spread-z, save/load preservation. All 11 copula tests pass; 49 validation tests total pass.
- `configs/generator/finetune.yaml` — added `copula.dequantize: true`, `copula.point_mass_threshold: 0.05`.
- `scripts/11_finetune_generator.py` — passes `dequantize` + `point_mass_threshold` to `CopulaTransform.fit`; logs the detected point masses + ε per feature.
- `scripts/60_deep_diagnostic.py` — the deep-diagnostic script that produced these findings (saved at last session).

The denormalize path in `13_sample_synthetic.py` and `decode.py` is unchanged — they call `norm.denormalize(z)` polymorphically; the requantization happens inside `CopulaTransform.denormalize` itself.

## Why this is a real contribution worth writing up

The core insight is that **copula-based generative models are pathological on zero-inflated distributions**, and the dequantization remedy (standard for normalizing flows) carries over directly. This isn't a fact about diffusion models specifically — it's a fact about applying continuous-CDF probit to discontinuous distributions. The same failure mode would appear if you applied this approach to *any* zero-inflated dataset (insurance claims, web event counts, financial trade events), and the same fix would resolve it.

The v8 diagnostic + v9 fix together tell a story the field doesn't currently flag: **before applying copula transforms to financial microstructure data, audit your features for point masses ≥ 5% and dequantize them**. That belongs in the report's discussion as a methodological note for anyone reproducing this.

## v9 cluster submission log (2026-05-07)

### First attempt — jobs 13515325 (v9) / 13515326 (v9_b)

Both jobs started cleanly with FiLM + (eps / v) and copula.dequantize=true. But the first copula-fit log line revealed a bug in `_compute_dequant_eps`:

```
point_masses={4: [(0.0, 0.0)], 5: [(0.0, 0.5)], 7: [(0.0, 4.618e-05)]}
```

**Feature 4 (`trade_dist`) had ε = 0.0** — effectively no dequantization. trade_sz (ε=0.5) and mid_return (ε=4.6e-5) were fine.

### Root cause of the second bug

The original `_compute_dequant_eps` used `min(|diffs_nonzero|) / 2`. For features with a truly continuous distribution near the point mass (like `trade_dist`, where many trades happen at tiny distances from mid), the minimum non-zero distance is dominated by a few subnormal-magnitude trade events, driving ε to ~0. Floating-point artifacts and rare near-mid trades produced a min so small that `round(eps, 8)` displayed as exactly 0.0 in the log.

The naive expectation ("ε = half the gap to the nearest non-PM value") is correct in theory but pathologically sensitive in practice when "the nearest non-PM value" can be a continuum approaching the point mass.

### Fix — robust quantile-based ε

Replaced `min` with the **5th percentile** of `|diffs_nonzero|` divided by 2, plus a sanity floor of `feat_std × 1e-4`. Robust to subnormal-magnitude artifacts; captures the natural separation scale rather than the worst-case floating-point noise.

```python
eps = float(np.quantile(diffs_nonzero, 0.05) / 2.0)
feat_scale = float(np.std(other)) + 1e-12
return max(eps, feat_scale * 1e-4)
```

All 11 copula tests still pass (the existing tests don't exercise the floating-point edge case directly, but the round-trip and well-spread-z tests both hold).

### Second attempt — jobs 13515821 (v9) / 13515822 (v9_b)

After cancelling and re-submitting:

```
point_masses={4: [(0.0, 1.276e-05)], 5: [(0.0, 0.5)], 7: [(0.0, 4.93e-05)]}
```

All three features have non-zero ε. Race condition cleanly resolved this time: v9 fitted at 16:26:25, v9_b loaded the cached copula at 16:26:51 (no double-fit, no overwrite collision).

### Note for the report's methodology section

The implementation gotcha is worth flagging explicitly: when computing the dequantization band-width ε for a copula transform, use a **robust quantile** (5th percentile or similar), not the literal minimum. The literal min collapses to ~0 whenever the "non-point-mass" portion of the distribution has any density near the point mass — which, for financial microstructure features, is the rule rather than the exception.

### Race condition (worth fixing post-v9)

Both jobs in the first attempt independently fit and saved to `data/processed/norm_stats_INTC.copula.json`. The second writer's output replaced the first's; results were identical (deterministic fit given the same training data + RNG seed) so no real damage. Going forward, the copula path should be namespaced by `experiment_name` so parallel runs don't race. Low priority — flagged for a follow-up commit.

## Status as of 2026-05-07 16:30

- v9 (jobs 13515821) and v9_b (13515822) running on GPU partition. Both confirmed correct config + correct dequantization per the second-attempt fit log.
- Expected checkpoint by ~18:00. Then post-training pipeline: sample on CPU partition (`slurms/ph2_sample_cpu.slurm`), ph6 validate, ph5 hypothesis test. Then re-run `60_deep_diagnostic.py` over v2 + v8 + v9 + v9_b for the side-by-side comparison.
- The killer post-sampling check: `frac_zero` on synth `mid_return` should be ~0.80 (close to real's 0.89). v8 had it at 0.19 — if v9 hits 0.80, the dequantization round-trip works as designed and the marginal recovery story is solid.

## v9 outcome (2026-05-08)

The orchestrator finished overnight. Both runs sampled (CPU, ~1h each), validated, and ran ph5 hypothesis. Headline:

| Run | ρ_diff | π_diff (synth ranking) | vs v8 |
|---|---|---|---|
| v8 (no dequant) | **+0.80** | `[A0, A2, A3, A1]` | (baseline) |
| **v9** (dequant, ε-pred) | **−0.80** | `[A3, A1, A0, A2]` | sign FLIPPED |
| **v9_b** (dequant, v-pred) | **−1.00** | `[A3, A1, A2, A0]` | exact reverse of truth |

ρ = −1 means v9_b's ranking is the perfect anti-image of held-out truth. Both p-values = 1.0 (one-sided). This is a clean, dramatic negative result.

### What the marginals look like

| Model | frac_zero | cond_std | cond_kurt |
|---|---|---|---|
| real | 0.89 | 6.76e-5 | **55.8** |
| v8 (no dequant) | 0.17 | 1.65e-5 | 155.7 |
| **v9** | **0.99** | 1.23e-4 | **−1.7** |
| **v9_b** | 0.99 | 1.42e-4 | **−1.6** |

Dequantization "fixed" the marginal problem: frac_zero is now ≈ real (over-corrected: 0.99 vs 0.89). But the conditional return distribution is now **uniform-noise-shaped** (kurt ≈ −1.7, the kurtosis of a uniform distribution). Heavy tails NOT restored.

### Why the conditional kurtosis collapsed: ε too large

The dequant band for `mid_return` was ε = 4.93e-5. Real conditional std (the std over non-zero returns) is 6.76e-5. So **ε / cond_std ≈ 0.73** — the dequant noise is essentially the same scale as the actual signal. The model learned to emit "uniform noise within the band" rather than reproduce the heavy-tailed continuous part of the distribution.

**Rule to add to the methodology section:** dequantization ε must be at least an order of magnitude *below* the conditional standard deviation of the non-point-mass portion of the distribution. For mid_return: ε should have been ~1e-6 to 1e-7 (below the smallest tick-return scale), not ~5e-5. The 5th-percentile-based ε we used was robust against floating-point artifacts but produced a band too wide for the natural feature scale.

### Why ρ_diff flipped — the corrected causal story

My initial reading was that **dequantization changed the joint structure and broke ρ_diff**. That framing is wrong on the causal direction. Look at v8's OFI→return regression from the deep diagnostic (2026-05-07):

```
v8 OFI→return slopes by horizon Δ:
  Δ=1:   slope=-4.59e-10   sign=-
  Δ=10:  slope=-1.60e-09   sign=-
  Δ=50:  slope=-1.05e-08   sign=-
  Δ=100: slope=-2.68e-08   sign=-
```

**v8's OFI joint was already wrong-signed at every horizon, BEFORE any dequantization.** Same direction as v6/v7/v7_b. The copula transform broke the OFI→return joint the moment it was applied — dequantization is irrelevant to that.

What was different about v8 vs v9 is the **PnL consequences** of the broken joint:

- **v8**: returns were collapsed (z-mean = −2.55, z-std = 0.30 → near-constant model output). Wrong-sign OFI signal had ~zero PnL effect because returns were tiny everywhere. All agents earned ≈ −$6 per tape regardless of strategy. The ρ_diff = +0.80 was measuring **fill-count noise on top of near-zero returns** — happened to align with truth's tail-risk dimension by accident.

- **v9**: dequantization restored return variance to ~real scale. **Now the wrong-sign OFI has real PnL consequences.** A2_AS_OFI tilts quotes per the OFI signal, gets more fills (3× A1's fill rate), and every fill is in the wrong direction → A2 loses the most → A2 ranks LAST.

The numbers from `61_ranking_mechanism.py`:

| agent | v9 pnl | v9 fills | v9 fill_rate |
|---|---|---|---|
| A0_const | +0.004 | 0.0002 | 53 |
| A1_AS | +0.005 | 0.0003 | 104 |
| **A2_AS_OFI** | **−0.005** | **0.0007** | **284** ← most active, only loser |
| A3_AS_VPIN | +0.006 | 0.0002 | 68 |

A2 is the only agent that uses the OFI signal. A2 is the only loser. Mechanism is unambiguous.

### What this means for "should we try v9.1 with smaller ε?"

A v9.1 with ε ~ 1e-7 (an order of magnitude below cond_std) would fix the marginal pathology — `frac_zero` should land at ~0.89, conditional kurt should recover toward real's 55. **But it would NOT recover ρ_diff.** The OFI→return joint is broken by the copula transform itself, not by the dequantization band-width. Smaller ε won't reverse the wrong sign on the OFI slope.

This is the practical implication: **the copula approach is incompatible with preserving weak cross-feature correlations.** Real-data Spearman ρ(OFI, mid_return) = 0.071 — that's noise-level. A rank correlation that weak cannot survive the round-trip through:
1. forward Gaussianization (transforms to z-space)
2. diffusion training on Gaussianized data (model learns smoothed joints)
3. sampling (DDIM with ε-prediction)
4. inverse CDF (back to original scale)

Each stage adds noise. A 0.071 rank correlation is below the noise floor of this pipeline. The copula's *mathematical* guarantee that monotone transforms preserve Spearman ρ is theoretically true and practically irrelevant when ρ ≈ 0 to begin with.

### Reframed story for the report

The report should NOT say "we tried dequantization and it broke ρ_diff." That's the wrong causal story.

The correct framing:

> **The copula transform fails to preserve the economically important cross-feature relationships in financial microstructure data because those relationships are noise-level in the original distribution.** Real-data Spearman ρ(OFI, return) = 0.071 — the mathematical property the copula preserves (monotone-invariance of rank correlations) is meaningful only when those correlations are well above the diffusion-training noise floor. For LOB data where the agent-relevant joints are 0.07-class, the copula approach is structurally inadequate.
>
> v8 produced an apparently-good ρ_diff = +0.80 only because return collapse made all agents perform identically — the ranking was fill-count noise, not signal. v9, by restoring return variance via dequantization, exposed the underlying wrong-sign OFI joint. ρ_diff = −0.80 is the honest measurement of the same broken joint that v8 had been hiding.

This is a sharper finding than "marginals vs joints" — it identifies a specific class of cross-feature relationships (weak Spearman, < 0.1) where copula-diffusion approaches will fail by construction.

### Updated v8 → v9 → v9.1 logic

- v8: artifact (return collapse hides broken OFI; ρ_diff = +0.80 is noise-on-noise)
- v9: real measurement of the broken OFI; ρ_diff = −0.80 is the honest result with v8's marginal-collapse pathology removed
- v9.1 (smaller ε): would improve marginals but not joints. Not worth running unless we want to confirm the prediction.

The right baseline comparison for v9 is **v2** (ρ_diff = +0.40, no copula at all), not v8.

### What's worth doing next

Two paths forward, depending on what the report needs:

1. **Stop here, write up.** The marginals-vs-joints story is now clean: copula transform breaks weak cross-feature correlations regardless of marginal quality. v9 is the conclusive demonstration. The report frames v8 as the artifact, v9 as the honest measurement, v2 as the only generator that preserves the relevant joints (because v2 doesn't transform the data at all — it just z-scores).

2. **Try v9.1 to confirm the ε-vs-cond_std rule** (ε = 1e-7 instead of 4.93e-5). Predicted result: frac_zero closer to 0.89, conditional kurt recovers toward 55, ρ_diff stays negative (around −0.6 to −0.8). Only worth running if we want to formally separate the ε-too-large finding from the copula-breaks-OFI finding.

For a class report at the course-project scale, option 1 is cleaner. The v9 result already tells the story.

## File index for this session

(continued from earlier)

- `scripts/utils/run_v9_post_training.sh` — orchestrator that ran the full v9/v9_b post-training pipeline overnight (sample, ph4, ph6, ph5, archive, side-by-side diagnostic).
- `scripts/61_ranking_mechanism.py` — used to decompose A1/A2 differences across all models; the per-agent table for v9 directly identifies A2 as the only loser.
- `results/_archive/v9/` and `results/_archive/v9_b/` — full archive of v9/v9_b backtests, validation, and hypothesis-test outputs on the cluster.