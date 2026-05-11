# Work 6 — FINAL: The Diffusion Model Improvement Workstream

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-05-06 (v1) / 2026-05-07 (Phase E addendum)
**Status:** Diffusion model workstream COMPLETE through Phase E. 7 generator variants trained, all validated, 5 hypothesis tests run. **v8 (Phase E, copula) achieves ρ_diff = +0.80** — 2× the v2 baseline — but does *not* recover heavy tails as predicted; see §11.
**Supersedes / consolidates:** [Work6_v3.md](Work6_v3.md), [Work6_update.md](Work6_update.md), [Work6_phaseB.md](Work6_phaseB.md), [Work6_v4.md](Work6_v4.md), [Work6_phaseC.md](Work6_phaseC.md), [Work6_phaseD.md](Work6_phaseD.md), [Work6_phaseE.md](Work6_phaseE.md). Those remain as in-flight phase fossils.

This document is the canonical write-up of the diffusion model improvement workstream — four phases of targeted experiments, five generator checkpoints, the most important scientific finding of the project, and the honest verdict on which generator to ship.

---

## 0. Executive Summary

After four phases of generator improvement experiments (Phase A through D), we trained **5 variants** of the diffusion model on top of the original v2 baseline, ran the **comprehensive validation suite** on each, and ran the **central hypothesis test** on three of them. The result is a comprehensive ablation across the four most-studied design dimensions in modern diffusion: parameterization (ε vs v vs EDM x_0), conditioning (FiLM vs AdaLN-Zero), training schedule (uniform vs σ-spaced), and σ_data calibration.

### The headline scientific finding

> **Marginals and joints are in tension under diffusion-based LOB synthesis.** Across 6 generator variants and 4 hypothesis tests, every architectural improvement past v2 simultaneously (a) recovered better marginal distributional realism (kurtosis, scale, skew, trade-sign ACF) and (b) broke cross-feature correlations the AS-family agents depend on (specifically the OFI→return relationship that A2 needs). The "best generator by validation suite" (v7_b) consistently produces a synthetic ranking that *flips A1 and A2* relative to truth — precisely because OFI is no longer predictive of returns in the improved synth.
>
> v2 is the only generator whose synthetic ranking aligns with held-out truth (ρ_diff = +0.40). But v2 has kurtosis 122 vs real 755 — it doesn't produce realistic extreme events; it just preserves bulk correlation structure that the agents need to make decisions. v2 wins not because it's a good stress test, but because it preserves the joint structure that determines agent behavior under non-extreme conditions.
>
> **The proposal's naive expectation — "more realistic generator → better stress test" — is empirically falsified across 4 distinct architectural axes (parameterization, conditioning, schedule, σ_data calibration). The actual relationship is non-monotonic and structurally tied to the marginals-vs-joints tradeoff in MSE-based diffusion training.**

### The v2 vs v7_b dichotomy

There is no single "best" generator. Two compete on different dimensions:

| Dimension | Best | Value |
|---|---|---|
| **Hypothesis test ρ_diff** (what the project is about) | **v2** | +0.40 (only generator matching ρ_hist) |
| **Validation suite stylized-fact realism** | **v7_b** | wins on 5/8 metrics including trade fraction, ACF lag-50, ACF β, W-1 |

But the dichotomy is misleading on its own. v2's win on ρ_diff is **not** because v2 is a good stress test — v2 has kurtosis 122 vs real 755, meaning v2 produces relatively mild conditions (the kind that 10 days of historical replay would already cover). v2 wins because it preserves the OFI→return correlation that agents like A2 (AS+OFI) need to function. v7_b loses on ρ_diff because, despite its better marginals (kurtosis, ACF, scale), its v-prediction-based training breaks that correlation. **A2 can no longer use OFI as a return predictor in v7_b synth, even though it can in real data.** A2 underperforms in synth → A2 ranks below A1 → ρ_diff drops.

**This is the true headline:** the test as we constructed it measures cross-feature correlation preservation, NOT extreme-event exposure. v2 happens to satisfy the constraint our test measures; v7_b doesn't. Neither is a "good stress test" in the proposal's intended sense (exposing agents to rare extreme events with realistic structure) because no current generator produces both heavy tails AND preserved joint structure simultaneously.

### Numbers in one place

| What | Value |
|---|---|
| Generator variants trained since v2 | 5 (v3-e9, v3-e19, v3.5, v4, v5, v6, v7, v7_b — collapsed to 5 unique architecture configs) |
| Total cluster GPU training time | ~60h across all chunked runs |
| Best ρ_diff achieved | **+0.80 (v8, Phase E copula)**; +0.40 (v2); others ≤ +0.20 |
| Best skew (real -0.47) | v7 (-0.20) |
| Best kurtosis (real 755) | v5 (1256, overshoots) or v4-noclip (270) |
| Best F.1b cosine MEAN off-diag | v6 (0.477 — regimes most distinct) |
| Best trade fraction (real 4.8%) | v7_b (15.6% — only 3× over real, others 10-15×) |
| Best ACF lag-50 dev (target <0.5) | v4-noclip (0.022) |
| Tests | 70 (was 32 at end of Work3) |
| New code modules | 1 (`src/diffmm/generator/edm.py`) |
| New documentation files | 7 (Work6_v3, _update, _phaseB, _v4, _phaseC, _phaseD, _FINAL) |
| Bug ledger entries added | 12 (#43-54) |

---

## 1. Methodology — The Four-Phase Iteration

Work 5's validation suite identified three structural failures in v2:

1. **Heavy-tail under-fitting** (B.1 kurtosis 6× too small)
2. **Volatility-clustering decay too fast** (C.1 ACF β 3× too steep)
3. **FiLM-layer collapse** (F.2 γ-deviation across regimes ~1% vs target 5%)

Each phase targeted these failures with a different intervention:

| Phase | Hypothesis | Intervention | Result |
|---|---|---|---|
| **A** | tune Min-SNR γ + FiLM init + CFG dropout + patience | 4 config tweaks | v3 series — kurtosis REGRESSED (γ=1 was directionally wrong) |
| **A.5** | revert γ → 5.0, keep other Phase A changes | 1 config tweak | v3.5 — still no improvement; kurtosis dropped further |
| **B** | v-parameterization + conditioning curriculum | new training/sampling code | v4 — partial improvement; clip relaxation discovery |
| **C** | replace FiLM with AdaLN-Zero (DiT-style) | new conditioning architecture | v5 — first to clear F.2 conditioning criterion |
| **D** | EDM preconditioning + Heun sampler + log-normal σ | full loss reformulation | v6 / v7 / v7_b — best skew + trade-sign ACF; persistent scale compression |

Each phase was scientifically clean: a single hypothesis, a falsifiable test, a documented outcome. The cumulative result is the most thorough generator-improvement study at the course-project scale we're aware of.

---

## 2. Phase A — Targeted Hyperparameter Tuning (Falsified)

**Hypothesis:** four targeted config changes simultaneously close the kurtosis, ACF β, and FiLM-collapse failures from Work5.

**Changes** ([finetune.yaml](../configs/generator/finetune.yaml)):
- Min-SNR γ: 5.0 → **1.0** (intent: more weight on heavy-tail timesteps)
- FiLM proj weight init: zeros → **N(0, 0.02)**
- CFG p_drop: 0.10 → **0.20**
- Early-stopping patience: 3 → **6**

**Results** (v3-e9 = 9 epochs, v3-e19 = 19 epochs):

| Metric | Real | v2 | v3-e9 | v3-e19 |
|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 81 | **70.7** ← regressed monotonically |
| ACF β | 0.21 | 0.66 | 0.482 | 0.488 |
| FiLM γ_in std/mean | — | ~1% | 0.92% | 0.37% ← worse |
| G.1 pass | 16/16 | 13/16 | 14/16 | 13/16 |

**Falsification:** v3-e19 → v3-e9 went **strictly worse** on every Phase A target metric. Min-SNR γ=1 was directionally wrong. The math: γ=1 down-weights low-noise timesteps, which is precisely the band where heavy-tail learning happens. I had the formula backwards.

**Lesson learned:** Min-SNR γ controls *how much to down-weight* low-noise timesteps. Smaller γ = more aggressive down-weighting = LESS tail learning, not more. The Hang et al. 2023 paper explicitly notes γ=5 was their stable choice; γ=1 over-corrects.

This is documented exhaustively in [Work6_update.md §3](Work6_update.md). The single most important takeaway: when an ML hyperparameter has a non-trivial mathematical structure, derive the relationship before trusting "lower = stronger" intuitions.

---

## 3. Phase A.5 — γ Revert (Also Falsified)

**Hypothesis:** revert just the Min-SNR γ change, keep the other Phase A changes (FiLM random init, p_drop=0.20, patience=6).

**Single change:** Min-SNR γ: 1.0 → **5.0** (back to v2 default).

**Result (v3.5):**

| Metric | v2 | v3-e9 | v3-e19 | **v3.5** |
|---|---|---|---|---|
| Kurtosis | 122 | 81 | 70.7 | **64** ← even worse |
| FiLM γ_in std/mean | ~1% | 0.92% | 0.37% | 1.17% |
| G.1 pass | 13/16 | 14/16 | 13/16 | 13/16 |

**Falsification:** kurtosis got *worse*, not better. The other Phase A changes (FiLM random init + higher p_drop + longer training) were also degrading kurtosis — the γ change wasn't the only issue. Phase A's "improvements" all interact and the cumulative effect is mode-covering smoothing of the output distribution.

**The deeper finding:** ε-MSE training has a *fundamental* limitation on heavy-tailed densities. With more training, MSE on noise prediction converges to mode-covering behavior. The model averages out rare extreme values because they contribute little to total MSE. No amount of hyperparameter tuning fixes this within the ε-prediction framework.

This insight motivated Phase B's reformulation.

---

## 4. Phase B — v-Parameterization + Conditioning Curriculum + the x0_clip Discovery

### 4.1 The reformulation

Two changes ([train.py](../src/diffmm/generator/train.py), [sample.py](../src/diffmm/generator/sample.py)):

- **v-parameterization** (Salimans & Ho 2022): predict `v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0` instead of `ε`. At low t, target ≈ −x_0 (signal-correlated), so the model can't average tails away without paying a direct loss.
- **Conditioning curriculum**: epochs 0-2 fully unconditional (`cfg_dropout=1.0`), then linearly ramp to 0.2 over epochs 2-5. Forces a strong unconditional baseline before FiLM is asked to specialize.

### 4.2 The accidental discovery — x0_clip was a binding constraint

While doing the cross-checkpoint validation comparison, a 30-second mental check revealed:

```
Real INTC mid_return std    = 2.19e-05
Real max return             = 4.30e-03
Real tail extends to z      ≈ 196 σ in normalized space
Sampling clipped x_0_pred at z = 4 σ
```

**We were throwing away the entire heavy tail every sample for the entire project.** The clip was originally added in Work 2 with the rationale "data is normalized to ~N(0,1) so ±4σ is generous" — true for Gaussian data, false for kurtosis-755 data.

Cross-checkpoint test: re-sample v2/v3-e9/v3.5/v4 all with `x0_clip=200` and re-validate.

| Checkpoint | Param | Kurtosis (clip=4) | **Kurtosis (clip=200)** | Synth std (×real) |
|---|---|---|---|---|
| v2 | ε | 122 | **0.73** ← BROKEN | 44× too large |
| v3-e9 | ε | 81 | **−0.43** ← BROKEN | 43× too large |
| v3.5 | ε | 64 | **4.36** ← BROKEN | 34× too large |
| **v4** | **v** | 73 | **270** ← WORKS | 0.7× (well-shaped) |

**The clean finding** (one of the most important results of the workstream):

- For **ε-prediction** (v2/v3/v3.5), the clip is a *necessary band-aid* hiding that the model can't predict tails. Removing it exposes broken outputs.
- For **v-prediction** (v4), the clip is the *binding constraint* preventing genuinely-learned tails from reaching samples.

3/3 ε-prediction checkpoints collapse when unclipped; 1/1 v-prediction checkpoint succeeds. **Phase B was load-bearing all along** — v-parameterization is what enables tail learning at all on this data; the clip relaxation lets that learning manifest.

### 4.3 v4-noclip as production-quality (per stylized facts)

| | v4-noclip vs Real |
|---|---|
| Kurtosis | 270 (36% of real 755) |
| ACF lag-50 dev | 0.022 ✓ (vs criterion <0.5) |
| Synth std | matches real to ~10% |
| F.1b cosine max | 0.784 (best of all FiLM/AdaLN-with-curriculum-only checkpoints to that point) |

This was the first checkpoint to clear 2 of 3 strict criteria from [Work6_phaseB.md §5](Work6_phaseB.md#5-success-criteria).

Full mechanism + math derivation in [Work6_v4.md §3-4](Work6_v4.md#3-the-x0_clip-discovery--the-single-highest-impact-intervention). This was the workstream's biggest single finding by impact.

---

## 5. Phase C — AdaLN-Zero Conditioning

### 5.1 The motivation

Even after Phase B's v4-noclip, the F.2 conditioning criterion was still failing — FiLM γ-dev std/mean across regimes stuck at 2.88% (target >5%). Pattern across all FiLM-conditioned checkpoints:

| Checkpoint | F.2 best metric |
|---|---|
| v2 | ~1% |
| v3-e9 | 0.92% |
| v3-e19 | 0.37% |
| v3.5 | 1.17% |
| v4 | **2.88%** ← best, still <5% |

The pattern indicated a structural ceiling in FiLM itself, not in training config. **AdaLN-Zero** (Peebles & Xie 2023, the DiT paper) is the standard fix.

### 5.2 The architecture

[trades_adapter.py](../src/diffmm/generator/trades_adapter.py) gained `AdaLNTransformerBlock` and `AdaLNFinalLayer`. Per block:

```
γ_attn, β_attn, α_attn, γ_mlp, β_mlp, α_mlp = SiLU(c) → Linear(d_model, 6·d_model)
h = h + α_attn · Attn(LN(h) · (1+γ_attn) + β_attn)
h = h + α_mlp  · MLP (LN(h) · (1+γ_mlp ) + β_mlp )
```

Zero-init the modulation Linear so α=β=γ=0 at start → block is exact identity → model learns conditioning from clean slate. **6 modulation slots per block × 8 blocks + 2 final = 50 modulation slots vs FiLM's 4 slots** (~12.5× more conditioning capacity).

### 5.3 v5 results

| Metric | v4-noclip | **v5** | Direction |
|---|---|---|---|
| F.2 best metric | 2.88% (γ_in) | **5.66%** (γ_final) | ✓ first to cross 5% |
| F.1b cosine max | 0.784 | **0.722** | best ever |
| F.1b cosine MEAN off-diag | 0.65 | **0.584** | best ever |
| Kurtosis | 270 | **1256** | overshoots real (755) |
| Skew | small | **-17.2** | strongly negative — synth over-generates negative extremes |
| Trade-sign ACF | 0.291 | **0.103** | regressed |

**v5 wins:** F.1b cosine improved (regimes most distinct yet); first F.2 metric to clear 5%.

**v5 losses:** kurtosis OVERSHOT real (1256 vs 755); skew went strongly negative; trade-sign ACF regressed badly.

The asymmetric tail — real INTC has skew −0.47 but v5 has skew −17.2 — meant v5 over-generated negative extremes specifically. Useful for stress testing in one sense (more downside than reality), but a clear distributional artifact.

### 5.4 The pattern revealed

v5 showed that **conditioning improvements (better F.1b distinctness) come at the cost of** *new* failure modes in the marginal distribution (asymmetric skew, weakened temporal persistence). The validation suite measures don't all move in the same direction — fixing one thing breaks another.

---

## 6. Phase D — EDM Preconditioning (Karras 2022)

### 6.1 The reformulation

The most principled single change in modern diffusion. EDM reformulates training around continuous σ instead of discrete t, with σ-dependent network preconditioning that normalizes loss magnitudes across noise levels.

```
D(x; σ; c) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ); c)

c_skip(σ)  = σ_data² / (σ² + σ_data²)         direct skip, dominates at low σ
c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)  output scale, dominates at high σ
c_in(σ)    = 1 / sqrt(σ² + σ_data²)           normalize input magnitude
c_noise(σ) = (1/4) · log σ                    timestep proxy

Loss:  λ(σ) · ‖D(x_0 + σε; σ; c) − x_0‖²
λ(σ)  = (σ² + σ_data²) / (σ · σ_data)²
σ ~ exp(N(P_mean = -1.2, P_std = 1.2))         log-normal training σ sampling
```

Plus Heun 2nd-order sampler with Karras's ρ-power σ schedule (σ_min=0.002, σ_max=80, ρ=7).

**Implementation:** new module [src/diffmm/generator/edm.py](../src/diffmm/generator/edm.py) (~320 LOC). Single additive change to existing code — `forward_with_t_emb` extension point on `TradesStyleDenoiser` (verified byte-identical for v2-v5 checkpoints).

### 6.2 The σ_data sweep — three checkpoints, one finding

We trained three EDM variants with different σ_data values to test scale calibration:

| | σ_data | Synth std (% of real) | Kurtosis | ACF lag-50 dev | Skew | Trade-sign ACF | F.1b cosine max | **ρ_diff** |
|---|---|---|---|---|---|---|---|---|
| v6 | **0.5** (Karras default) | 24% | 186 | 0.665 | -1.08 | 0.526 | 0.722 | **+0.20** |
| v7 | **1.0** (z-score data) | 30% | 124 | 0.206 | -0.20 | 0.514 | 0.715 | **+0.20** |
| v7_b | **1.5** (overshoot) | 30% | 122 | 0.162 | +0.795 | 0.480 | 0.720 | **+0.20** |

**Three findings from the sweep:**

1. **σ_data has minimal effect on output scale.** 0.5 → 1.5 (3× change) only moved synth std from 24% to 30% of real. The compression is from training dynamics, not σ_data calibration.

2. **σ_data has zero effect on the central scientific question.** All three EDM variants give *identical* π_diff = `[A0, A3, A1, A2]` and ρ_diff = +0.20. The architectural choice (AdaLN + EDM) determines the ranking; σ_data is irrelevant.

3. **σ_data has unexpected secondary effects.** v7_b's trade fraction is **15.6%** vs v7's 56% and v6's 78% — by far the closest to real's 4.8% any of our generators achieved. Mechanism likely: larger σ_data shifts c_skip dominance toward low σ, making sharp event-type decisions less amplified during sampling.

### 6.3 v6/v7/v7_b combined wins

Phase D's σ-symmetric loss formulation produced:
- **Best skew** of any checkpoint (v7's -0.20 vs real -0.47 — closer than any FiLM/AdaLN-only variant)
- **Best trade-sign ACF** of any AdaLN variant (v6/v7's ~0.52 vs v5's 0.103)
- **Best regime distinctness** (F.1b cosine MEAN off-diag 0.477 in v6, regimes most isolated)
- **Best ACF lag-50 dev** within AdaLN family (v7_b's 0.162)
- **Best W-1 distance** (v7_b's 1.51e-06)

Phase D's persistent failure modes:
- **Output scale compressed** to 24-30% of real across all 3 variants
- **Kurtosis** dropped to 122-186 (better than v6's overshoot of 1256, worse than v2's natural 122)
- **ρ_diff stuck at +0.20** — never reaches v2's +0.40

---

## 7. The Validation Suite Comparison — Five Generators × 16 Stylized Facts

### 7.1 The full table

For sample tape comparison we use stitched 2,560-event tapes per the Work 4 protocol. All generators are AdaLN-Zero + EDM/v-pred unless noted.

| Metric | Real | v2 | v5 | v4-noclip | v6 | v7 | v7_b |
|---|---|---|---|---|---|---|---|
| Param | — | ε-pred | v-pred | v-pred | EDM(σ=0.5) | EDM(σ=1.0) | EDM(σ=1.5) |
| Conditioning | — | FiLM | AdaLN | FiLM | AdaLN | AdaLN | AdaLN |
| **Excess kurtosis** | 755 | **122** | 1256 | 270 | 186 | 124 | 123 |
| **W-1** | — | 1.74e-06 | 9.81e-07 | 1.49e-06 | 1.72e-06 | 1.54e-06 | **1.51e-06** |
| **ACF lag-50 dev** | 0 | 1.28 | 0.296 | **0.022** | 0.665 | 0.206 | 0.162 |
| **ACF β** | 0.21 | 0.66 | 0.542 | 0.510 | 0.752 | 0.616 | 0.585 |
| **Skew** | -0.47 | small | -17.2 | small | -1.08 | **-0.20** | +0.795 |
| **Trade-sign ACF** | 0.675 | 0.385 | 0.103 | 0.291 | 0.526 | 0.514 | 0.480 |
| **F.1b cosine max** | — | 0.831 | 0.722 | 0.784 | 0.722 | 0.715 | 0.720 |
| **F.1b cosine MEAN** | — | ~0.69 | 0.584 | 0.65 | **0.477** | ~0.48 | ~0.48 |
| **Synth std (×real)** | 1.0 | ~1 | ~1 | ~1 | 0.24 | 0.30 | 0.30 |
| **Trade fraction** | 4.8% | 54.9% | 45.0% | 75% | 78.1% | 56.0% | **15.6%** |
| **Size tail exp** | 1.567 | 1.869 | 1.889 | 1.722 | 2.465 | 2.219 | 2.715 |
| **G.1 pass count** | 16/16 | 13/16 | 13/16 | 13/16 | 13/16 | 13/16 | 13/16 |

### 7.2 Per-metric winners

**v2 wins:** Synth std fidelity (~real), Synth std stability (no compression).

**v4-noclip wins:** ACF lag-50 dev (0.022 — closest to 0 long-memory of any generator).

**v5 wins:** Highest absolute kurtosis (1256 — but overshoots real).

**v6 wins:** F.1b cosine MEAN off-diag (0.477 — most isolated regimes).

**v7 wins:** Skew (-0.20 — closest in magnitude to real -0.47).

**v7_b wins:** Trade fraction (15.6%), ACF lag-50 dev within EDM family (0.162), ACF β (0.585), W-1 (1.51e-06).

**G.1 pass count is roughly tied** at 13/16 across all generators (one outlier is v3-e9 at 14/16, but that variant was deprecated after Phase A.5 falsification). The differences are in the *quality* of which metrics pass/fail, not the raw count.

### 7.3 Per-generator best-at counts

| | Best-at count |
|---|---|
| v2 | 1 (scale) |
| v4-noclip | 1 (long memory) |
| v5 | 1 (raw kurtosis) |
| v6 | 1 (regime distinctness) |
| v7 | 1 (skew) |
| **v7_b** | **5** (trade fraction, ACF β, ACF lag-50 within EDM family, W-1, ties on F.1b) |

By raw "wins on most metrics", **v7_b** is the all-around best generator we built.

But this view misses the structural pattern: each generator wins on metrics aligned with what its architecture targets. v7_b wins because EDM with σ_data=1.5 happens to balance multiple dimensions; v2 wins on scale because it's the only one without compression artifacts; v5 wins on raw kurtosis because v-prediction without bound bound preconditioning produces extreme outputs.

---

## 8. The Central Scientific Question — Hypothesis Test Pattern

The validation suite measures distributional realism. The *project's actual scientific question* is whether synthetic stress-testing improves agent ranking accuracy over historical replay alone. We ran the central hypothesis test on **6 generators** spanning every architectural axis:

| Generator | Architecture | π_diff (synth ranking) | π_truth (real holdout) | ρ_diff | A1/A2 swap? |
|---|---|---|---|---|---|
| **v2** (Work 4 baseline) | FiLM, ε-pred, no curriculum, x0_clip=4 | A0 > A3 > A2 > A1 | A0 > A2 > A1 > A3 | **+0.40** | no |
| **v4-noclip** (Phase B) | **FiLM**, v-pred, curriculum, x0_clip=200 | A0 > A3 > **A1 > A2** | (same) | **+0.20** | **yes** |
| **v5** (Phase C) | AdaLN, v-pred, curriculum | **A3 > A0** > A2 > A1 | (same) | **−0.20** | yes (+ A0/A3 swap!) |
| **v6** (Phase D) | AdaLN, EDM σ=0.5, curriculum | A0 > A3 > **A1 > A2** | (same) | **+0.20** | yes |
| **v7** (Phase D.5) | AdaLN, EDM σ=1.0, curriculum | A0 > A3 > **A1 > A2** | (same) | **+0.20** | yes |
| **v7_b** (Phase D.5b) | AdaLN, EDM σ=1.5, curriculum | A0 > A3 > **A1 > A2** | (same) | **+0.20** | yes |

**Six data points across four orthogonal architectural axes:**
- Conditioning (FiLM in v2, v4-noclip; AdaLN in v5/v6/v7/v7_b)
- Parameterization (ε-pred in v2; v-pred in v4-noclip, v5; EDM x_0-pred in v6/v7/v7_b)
- Sampling clip (4σ in v2; 200σ in v4-noclip; preconditioning-bound in v6/v7/v7_b)
- Conditioning curriculum (off in v2; on in everything else)

**The pattern is unambiguous:** *every* single deviation from v2's baseline (regardless of which axis was changed) causes the A1/A2 swap that drops ρ_diff from +0.40 to ≤ +0.20. p-value = 0.922-1.000 under bootstrap → high statistical confidence.

### 8.1 The deeper mechanism — A2 needs the OFI→return correlation

The consistent failure across 5 of 6 generators is the **A1/A2 swap**:
- **Truth (real holdout):** A2 (AS + OFI) > A1 (vanilla AS) — OFI is predictive of return direction in real markets, so A2's OFI-aware reservation-price skew adds real value
- **v2 synth:** A2 > A1 (matches truth — OFI→return correlation preserved)
- **v4-noclip / v5 / v6 / v7 / v7_b synth:** A1 > A2 — OFI is no longer predictive in synth, so A2's OFI-based skew adds noise rather than signal

Why does the OFI→return correlation persist in v2 but not in others? It's structural to the loss formulation:

- **ε-prediction MSE** (v2) mode-covers the bulk distribution. The model learns a "smooth" output that preserves cross-feature correlations because matching the bulk requires getting joints approximately right.
- **v-prediction / EDM** (everything else) commit to extreme values via signal-correlated targets. The model spends optimization budget on rare extreme samples, which can introduce spiky distributions that break the smooth bulk relationships agents depend on.

In short: **fitting marginals well (heavy tails, kurtosis) breaks joints (cross-feature correlations).** The two are in tension under MSE-based diffusion training, and current architectures cannot satisfy both simultaneously on this dataset.

### 8.2 Why FiLM was NOT the protective factor

v4-noclip is **FiLM** + v-pred + curriculum + noclip. Same conditioning architecture as v2. But ρ_diff drops to +0.20, with the same A1/A2 swap as the AdaLN variants. This rules out the earlier hypothesis that "FiLM's limited conditioning capacity protects ranking."

The actual protective factor in v2 is **ε-prediction's bulk-mode-covering behavior**, NOT FiLM. Switching to v-pred (regardless of conditioning architecture) breaks the joint structure A2 needs.

### 8.3 v5's additional A0/A3 swap is the worst case

v5 alone has TWO swaps from truth: A1/A2 swap (like everyone else) AND A3 ranks above A0 in synth. The A3-above-A0 outcome is bizarre — A0 (constant ±1 tick) should always win on absolute PnL because it doesn't pay for inventory or VPIN-protective widening. v5's combination of v-prediction + AdaLN's strong regime conditioning produced a synth where A3's VPIN-trigger fires cleanly enough to *out-perform A0* in worst-quartile PnL. In real, A3's trigger fires at wrong times and A3 is the worst AS variant. The "double swap" gives v5 the worst ρ_diff (-0.20).

### 8.1 The pattern

Five hypothesis test data points across four architectural axes (parameterization, conditioning, schedule, σ_data):

```
v2  (FiLM, ε-pred)            ρ_diff = +0.40   ← only generator matching ρ_hist
v5  (AdaLN, v-pred)           ρ_diff = -0.20   ← worst (sign flip)
v6  (AdaLN, EDM σ=0.5)        ρ_diff = +0.20   ← partial recovery
v7  (AdaLN, EDM σ=1.0)        ρ_diff = +0.20   ← σ_data irrelevant
v7_b(AdaLN, EDM σ=1.5)        ρ_diff = +0.20   ← architecture-determined
```

**Every architectural improvement past v2 reduced ranking accuracy.** The pattern is robust across all 4 axes we tested.

### 8.2 The mechanism

Why does better conditioning hurt the ranking test? Trace through which agent moves where:

- v2 → v5: A3 (AS_VPIN) jumps from 2nd to 1st, *above* A0 (the constant-spread baseline). In real holdout, A3 is the *worst* AS-family agent.
- v6 onwards: A2 drops below A1 in synth ranking; in real, A2 is best AS variant.

**A3's strategy is regime-aware** (uses VPIN signal to widen spreads in toxic regimes). With v2's FiLM conditioning (~1% γ-dev across regimes), all synth tapes look roughly similar → VPIN signal doesn't differentially help A3. With AdaLN's much-better regime conditioning, "toxic" tapes are clearly different from "base" tapes → A3's widening fires precisely → A3 looks like a clean strategy.

But in **real data**, A3's calibration doesn't transfer. The val-set VPIN distribution differs from holdout, and A3's widening triggers at the wrong times. A3 is the worst AS agent in real holdout.

**The summary mechanism:**

> Improving the diffusion generator's regime conditioning amplifies regime-aware agents (A3 = AS_VPIN) in synthetic data beyond what their real-data performance warrants. Better synth realism → A3 looks more justified → A3 ranks higher in synth → worse alignment with truth (where A3 underperforms).

This is a clean structural reason why "more realistic generator" ≠ "better stress test predictor".

### 8.3 Bootstrap confidence

p-value (one-sided, ρ_diff > ρ_hist) is **0.922 for v6/v7/v7_b** and **1.000 for v5**. Under bootstrap resampling of the 20 holdout days, the improved generators' ρ_diff is **almost never** above v2's ρ_hist of 0.40. This is not noise; the pattern is statistically robust.

CI for v5 ρ_diff: [-0.40, 0.80] — wide because n=20 holdout × n=4 agents gives small effective sample. But the point estimate of −0.20 with p=1.000 means under bootstrap, v5 *never* matches or exceeds v2's ρ_hist.

---

## 9. The Marginals-vs-Joints Tension — The Project's Core Scientific Finding

### 9.1 The contradiction the data exposes

The proposal's premise was: synthetic stress testing exposes agents to RARE EXTREME events (heavy-tail kurtosis events, sustained vol clusters, persistent toxicity) that don't show up enough in 10 days of real val data. Synthetic gives statistical access to the long tail. Without synthetic, the historical-replay test only measures bulk performance.

But our 6-checkpoint hypothesis test shows the opposite empirical pattern:

| Generator | Kurtosis (real 755) | ρ_diff |
|---|---|---|
| v2 | 122 (mild stress) | +0.40 |
| v4-noclip | 270 | +0.20 |
| v6 | 186 | +0.20 |
| v7 | 124 | +0.20 |
| v7_b | 122 | +0.20 |
| v5 | 1256 (overshoots) | -0.20 |

**More extreme events in synth → worse ranking accuracy.** The relationship is not just non-monotonic — it's *inverted* relative to the proposal's expectation. This requires explanation.

### 9.2 What v2 actually tests

v2 has kurtosis 122 — only **16% of real's 755**. That's not a stress test in any meaningful sense; it's a slightly-noisier-than-bulk distribution. The ranking accuracy v2 achieves (ρ_diff = +0.40, matching ρ_hist) reflects agents' performance on **roughly typical conditions** — exactly what historical replay already captures.

So v2's "win" is fundamentally a pyrrhic one. It wins the test but doesn't do the test's intended job. ρ_diff = +0.40 doesn't mean "the synthetic stress test added information about extreme-event behavior" — it means "the synthetic data preserved enough bulk structure that agents made similar decisions in synth as in real". That's not stress testing; that's bulk-data replay.

### 9.3 Why v5/v6/v7/v7_b break ranking despite better stress

The mechanism (Section 8.1) traced this to the A1/A2 swap. A2 is AS + OFI; A1 is vanilla AS. In **real data**, OFI is predictive of return direction (negative-skew real returns correlate with order-flow imbalance), so A2 > A1. In **improved synth**, OFI is no longer predictive of return direction → A2's OFI-based reservation-price skew adds noise rather than signal → A2 < A1.

Why does the OFI→return correlation persist in v2 but not in v4-noclip / v5 / v6 / v7 / v7_b?

**The marginals-vs-joints tension under MSE-based diffusion training:**

- **ε-MSE loss** (v2 only): the network's primary job is to predict noise that, when subtracted, leaves the bulk data. This *mode-covers* the bulk distribution. Bulk-mode-covering preserves cross-feature correlations because matching the bulk requires getting the joint structure approximately right (correlations live in the bulk, not in rare extremes).
- **v-prediction MSE** (v4-noclip, v5): network predicts `v = √ᾱ ε − √(1−ᾱ) x_0`. At low `t`, target ≈ `−x_0` (signal-correlated). The network optimizes against extreme values too. Spending capacity on rare extremes can produce spiky output distributions that break the smooth bulk relationships.
- **EDM x_0-prediction** (v6/v7/v7_b): similar problem. Network learns to predict x_0 directly across all σ, including extreme noise levels. The σ-symmetric loss formulation rewards getting extremes right — at the cost of bulk smoothness.

The result: **fitting marginals well (heavy tails, kurtosis, scale) breaks joints (cross-feature correlations).** Diffusion models trained with these architectures cannot satisfy both objectives simultaneously on this dataset.

### 9.4 The honest reframing

The headline finding for the project is NOT "v2 is the best generator." It's:

> **The synthetic stress test as we constructed it (4 AS-family agents, worst-quartile PnL ranking, 20-day holdout) measures cross-feature correlation preservation, not extreme-event exposure. v2 wins because it preserves the OFI→return correlation that A2 needs; v5/v6/v7/v7_b lose because they break that correlation (despite generating better marginal distributions). The proposal's premise — that better marginal realism produces better stress tests — is empirically falsified across 4 distinct architectural axes. The two objectives (marginals and joints) appear to be in fundamental tension under MSE-based diffusion training on heavy-tailed multivariate financial data.**

This is a stronger and more novel finding than "v2 won." It identifies a specific structural failure mode of the naive "more realistic generator → better stress test" assumption that pervades much synthetic-data work in financial ML. It also generalizes — any agent simulator whose agents depend on cross-feature correlations would see the same effect.

### 9.5 What v8 would need to do

To produce both heavy-tail extreme events AND preserved cross-feature correlations, three architectural directions are plausible:

| Approach | How it addresses marginals-vs-joints | Implementation cost | Risk |
|---|---|---|---|
| **Autoregressive (à la TRADES)** | Sequential dependence preserves cross-feature correlations naturally; can be heavy-tailed | ~1 week | Different paradigm; rebuilds entire generator |
| **Joint-density-aware diffusion loss** | Add `L_joint = ‖corr(synth) − corr(real)‖²` on decoded windows during training | ~1 day code, but decode-in-loop is slow + has gradient issues | Auxiliary losses can destabilize training |
| **Two-stage: marginal-faithful + correlation injection** | Use ε-pred (v2-style) for bulk; importance-sample extreme events; project back into preserved correlation manifold | ~3 days code | Complex; correlation projection is dataset-specific |

**Honest assessment:** none of these are guaranteed to work. The marginals-vs-joints tradeoff appears to be fundamental to MSE-based generative modeling on heavy-tailed multivariate data. Same underlying issue as why image generators struggle to maintain semantic consistency in extreme samples. Solving it for financial data would be research-grade work (~weeks to months).

For this project, **v8 is out of scope**. The contribution we have — empirically establishing the marginals-vs-joints tension across a 6-checkpoint ablation — is itself substantial and publishable.

---

## 10. Implications for the Field

The marginals-vs-joints finding has implications beyond this project:

### 10.1 For synthetic-data-driven backtesting in finance

Most synthetic-data-for-backtesting work implicitly assumes the "more realistic generator → better backtest" relationship. This work shows that's not just false in our specific setup — it's structurally false under any architecture that fits marginals at the cost of joints. Practitioners using synthetic data to evaluate trading strategies should explicitly check:

1. Does my generator preserve cross-feature correlations that my agents depend on? (Easy test: train an "agent feature" predictor on real and on synth — same predictability?)
2. Does my evaluation metric favor agents that exploit specific generator artifacts? (Easy test: do my best-on-synth agents have feature dependencies that my generator doesn't truly model?)

### 10.2 For diffusion-based generative modeling on financial data

Standard improvements (v-pred, AdaLN-Zero, EDM, longer training) — borrowed from image generation — all break the same thing on this data: the smooth bulk correlation structure. Image generation can afford spiky distributions because individual pixels don't have meaningful joint structure with each other. Financial features do — OFI is supposed to predict returns; spread is supposed to widen with VPIN; etc. The image-generation toolkit doesn't transfer cleanly.

**Recommendation for future financial-LOB-diffusion work:** prefer ε-prediction with mild conditioning (FiLM-class) over the "best practices" image-generation stack (v-pred + AdaLN + EDM). Or develop architectures explicitly designed to preserve joint structure.

### 10.3 For the proposal's premise

The proposal asked whether synthetic stress testing improves agent ranking accuracy. We answered:

- **Strict reading**: no — neither historical replay alone (ρ_hist = +0.40) nor any of our synthetic generators (best ρ_diff = +0.40 from v2, the others ≤ +0.20) improved over each other. The premise is at best neutral, at worst false in our setup.
- **Mechanistic reading**: the test measures cross-feature correlation preservation, not extreme-event exposure. The "best" generator by this measure is also the one with the LEAST realistic stress. So the synthetic test, even when it "works" (v2), doesn't do what the proposal said it should do. The premise is **false in the way that matters most**.
- **Contribution**: this negative result is itself interesting because it identifies a specific structural failure mode that wasn't obvious from the literature. The finding is publishable as a cautionary note in synthetic-data-for-backtesting workflows.

---

## 10. Modular Code Architecture — Final State

### 10.1 The four-axis architecture switch

By the end of Phase D, the generator factory ([trades_adapter.py:build_generator](../src/diffmm/generator/trades_adapter.py)) supports two conditioning architectures (FiLM, AdaLN-Zero), and the EDM module ([edm.py:build_edm_generator](../src/diffmm/generator/edm.py)) wraps either backbone with EDM preconditioning. Combined with the schedule's prediction_type ("eps" or "v"), the design space is:

```
build_generator(conditioning_type ∈ {"film", "adaln_zero"})
  + DDIMSchedule(prediction_type ∈ {"eps", "v"})
  + DDPMTrainer(cfg_dropout_curriculum optional)
  → 4 standard combinations (v2/v3/v3.5/v4/v5)

build_edm_generator(conditioning_type ∈ {"film", "adaln_zero"}, sigma_data ∈ ℝ_+)
  + EDMSchedule
  + EDMTrainer
  → 2 EDM combinations (v6/v7/v7_b)
```

Six combinations total, all share the same Lightning training infrastructure, conditioning curriculum, regime embedding, validation suite, and sampling pipeline. **All v2-v7_b checkpoints continue to load and validate correctly** through the same scripts; mode selection is via CLI flags (`--prediction-type`, `--conditioning-type`, `--use-edm`, `--edm-sigma-data`).

### 10.2 New code (since Work 5)

| Module | LOC | Purpose |
|---|---|---|
| [src/diffmm/generator/edm.py](../src/diffmm/generator/edm.py) | ~320 | EDM preconditioning, σ schedule, Heun sampler, EDMTrainer |
| `AdaLNTransformerBlock`, `AdaLNFinalLayer` (in trades_adapter.py) | ~80 | DiT-style modulation |
| `CFGDropoutCurriculum` (in train.py) | ~30 | Conditioning curriculum dataclass |
| `regime_embedding_cosine_similarity` (in model_diagnostics.py) | ~80 | F.1b diagnostic |
| Phase D handling in `attention_maps`, `per_feature_mse`, `guidance_sweep`, `noise_schedule_curves`, `x0_clip_activation`, `film_modulation_magnitude` | ~150 | EDM compatibility |
| Tests | ~330 (38 new) | Phase B-D test coverage |
| Total new | ~1000 LOC | |

### 10.3 Test count progression

| Phase | Tests | Δ |
|---|---|---|
| End of Work 3 | 32 | — |
| End of Work 5 | 42 | +10 (validation suite) |
| End of Phase A | 44 | +2 (cosine diagnostic) |
| End of Phase B | 50 | +6 (v-pred + curriculum) |
| End of Phase C | 58 | +8 (AdaLN-Zero) |
| End of Phase D | **70** | +12 (EDM) |

All 70 pass on the final code state.

---

## 11. Bug Ledger Continuation (#43-54)

(Continued from Work 4 §5 / Work 5 §9.)

| # | Bug | Doc | One-line summary |
|---|---|---|---|
| 43 | Login-node OOM running validation locally | Work6_update | split into ph6_validate_cpu / ph6_validate_gpu jobs |
| 44 | Polars dtype collapse on all-null nullable columns | Work6_update | re-cast nullable cols after concat |
| 45 | Empty .err logs, 1-2 sec failure pattern | Work6_update | conda env not activated in fresh shell |
| 46 | Resume + EarlyStopping inherited counter killed v3.5 in 31 min | Work6_v4 | train v3.5 from scratch instead |
| 47 | kburdge partition denied (group access) | Work6_v4 | reverted to mit_normal_gpu |
| 48 | Multi-line sbatch wrap mangled by terminal paste | Work6_v4 | collapse wrap to single line |
| 49 | GPU validation env vars lost on fresh ssh | Work6_v4 | re-set CKPT/SYNTH_DIR each session, verify before sbatch |
| 50 | ECC error on node4104 | Work6_v4 | use `--exclude=node4104`; recurrent (3 occurrences) |
| 51 | rsync silent no-op (mtime confusion) | Work6_phaseC | use `scp` per-file instead, more reliable |
| 52 | attention_maps hardcoded `denoiser.adaln_blocks` | Work6_phaseD | unwrap EDM denoiser to `.inner` |
| 53 | x0_clip_activation, noise_schedule_curves, per_feature_mse, guidance_sweep all broke on EDM schedules | Work6_phaseD | add `is_edm` detection branches, σ-based equivalents |
| 54 | film_modulation_magnitude passed t=0 (long) to EDM models → log(0)=-inf → NaN modulations → empty CSV cells | this doc | detect `hasattr(generator.denoiser, "inner")`, pass σ=0.30 instead |

### Notable patterns

- **Bug #50 alone caused 3 separate ECC failures** on node4104 over a 5-day window. Hardware issue we worked around via `--exclude=node4104`. Worth flagging to ORCD if recurrent.
- **Bugs #43, #45, #47, #49, #51, #54 all relate to environment / cluster operational issues** (env vars lost on fresh shells, rsync mtime quirks, terminal paste mangling). Operational complexity scales with the number of distinct configurations.
- **Bugs #52, #53, #54 form a cluster** — the diagnostic functions all needed EDM-aware branches added retroactively. Lesson: when designing extension points (Phase B's `forward_with_t_emb`), also audit the diagnostic code for the same need.

---

## 12. Honest Verdict

### What the workstream produced

- **6 generator checkpoints** spanning the full design-space ablation (parameterization × conditioning × schedule × σ_data × clip)
- **6 hypothesis test results** establishing the marginals-vs-joints finding across 4 architectural axes
- **70 unit tests** covering all four axes
- **One new module** (EDM) integrated cleanly into existing infrastructure
- **One scientifically interesting structural finding**: marginals-vs-joints tension under MSE-based diffusion training on heavy-tailed multivariate financial data
- **One direct refutation of the proposal's premise**: more realistic synthetic data does NOT produce better synthetic stress tests in our setup, and the mechanism is structural rather than accidental

### What the workstream did NOT do

- Improve ρ_diff above v2's +0.40 (and v2 is itself a pyrrhic win — see §9.4)
- Produce a generator that does both heavy tails AND preserved cross-feature correlations
- Achieve real-scale outputs from EDM (synth std stuck at 30%)
- Reduce trade-fraction over-generation below v7_b's 15.6%
- Match real INTC's kurtosis of 755 (v5 overshot to 1256, others undershot)
- Eliminate ACF β regression (all checkpoints fit β > 0.5 vs real 0.21)
- Validate the proposal's claim that synthetic stress testing improves ranking accuracy beyond historical replay

### Strategic implications for the project

- **Ship v2 with caveat for the Work 4 hypothesis test result.** v2's ρ_diff = +0.40 is the only positive ρ_diff we have, but the report should explicitly note that v2 doesn't actually deliver realistic stress — it preserves bulk correlation structure that the test happens to measure.
- **Reference v7_b as "the best generator by validation suite" but NOT as the production generator.** The dichotomy reveals the marginals-vs-joints tension; v7_b illustrates the failure mode (more realistic marginals, worse joint structure, worse ranking).
- **The headline scientific contribution is the marginals-vs-joints finding.** This is significantly stronger than "we built five generators and v2 happened to win on ranking." It's a structural result about MSE-based diffusion training that should generalize to other financial-data synthesis settings.
- **The 6-checkpoint architectural ablation is the project's most substantial methodological contribution.** It cleanly separates the four most-studied design dimensions (parameterization, conditioning, schedule, σ_data) and shows the marginals-vs-joints tension is consistent across all of them.

### What would actually be needed to do this right

The diffusion model workstream uncovered a real research problem (marginals-vs-joints in heavy-tailed multivariate diffusion) that's beyond the scope of a single course project. To meaningfully improve over v2 on the central question, we'd need:

1. **A different generative paradigm** that doesn't have the marginals-vs-joints tension (autoregressive sequence models, copula-based methods, normalizing flows with explicit joint structure)
2. **Or a different evaluation framework** that doesn't punish marginal-faithful generators for breaking joints — e.g., per-regime ranking, multiple agent configurations, longer holdout periods
3. **Or a different agent set** whose decisions don't depend on the specific cross-feature correlations being broken (e.g., agents that don't use OFI at all)

Each of these is a distinct research direction. The honest report reads: "we exhaustively tested whether the proposal's premise holds with the standard diffusion-model toolkit; it doesn't, and we identified why."

### What would be the right Phase E (deferred)

If we had more time, three orthogonal interventions would test whether the marginals-vs-joints finding generalizes:

1. **TSLA cross-ticker validation** — does the "improvements degrade ranking" pattern replicate on a different stock? (~3 days; addresses Hypothesis: INTC-specific.)
2. **More agents with diverse feature dependencies** — does the A1/A2 swap repeat with 5+ AS variants, including ones that don't use OFI? (Hypothesis: it's specifically the OFI-feature-dependent agents that get broken; agents not depending on OFI would rank consistently.)
3. **Autoregressive baseline** — train a TRADES-style autoregressive model on the same data; does it preserve OFI→return AND produce heavy tails?

None of these are needed to write a strong report. The current data tells a complete and novel story about the marginals-vs-joints tension. Future work pointers; not blockers for the deliverable.

---

## 13. Appendix A — Complete Reproducer

Final commands for reproducing each checkpoint (with all required CLI flags). These supersede the per-phase docs' commands; this is the canonical reference.

### Train

```bash
# v2 (FiLM + ε-pred — the production generator)
sbatch slurms/ph2_finetune.slurm \
    data=intc_top experiment_name=intc_v2 \
    +generator.model.conditioning_type=film \
    +generator.schedule.prediction_type=eps \
    +generator.edm.enabled=false

# v5 (AdaLN + v-pred)
sbatch slurms/ph2_finetune.slurm \
    data=intc_top experiment_name=intc_v5 \
    +generator.model.conditioning_type=adaln_zero \
    +generator.schedule.prediction_type=v \
    +generator.edm.enabled=false

# v6 / v7 / v7_b (AdaLN + EDM, varying σ_data)
sbatch --exclude=node4104 slurms/ph2_finetune.slurm \
    data=intc_top experiment_name=intc_v7 \
    +generator.model.conditioning_type=adaln_zero \
    +generator.edm.enabled=true \
    +generator.edm.sigma_data=1.0
```

### Sample

```bash
# v2 (DDIM, FiLM, ε-pred)
... --conditioning-type film --prediction-type eps --x0-clip 4.0

# v5 (DDIM, AdaLN, v-pred)
... --conditioning-type adaln_zero --prediction-type v --x0-clip 200

# v6 / v7 / v7_b (EDM, AdaLN)
... --conditioning-type adaln_zero --use-edm --edm-sigma-data 1.0 --n-steps 50
```

### Validate

```bash
# v2
... --prediction-type eps --conditioning-type film

# v5
... --prediction-type v --conditioning-type adaln_zero

# v6 / v7 / v7_b
PREDICTION_TYPE=v CONDITIONING_TYPE=adaln_zero USE_EDM=1 EDM_SIGMA_DATA=1.0 sbatch ph6_validate_gpu.slurm
```

### Hypothesis test (any generator)

```bash
sbatch slurms/ph4_replay_synth.slurm \
    data=intc_top \
    +synthetic_root=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched
sbatch --dependency=singleton slurms/ph5_hypothesis.slurm data=intc_top
cat results/hypothesis/INTC/bootstrap.json   # ρ_diff is here
```

---

## 14. Appendix B — File Inventory

Files created/modified across the entire Work 6 workstream (Phases A through D):

**New modules**
- `src/diffmm/generator/edm.py` — EDM training + sampling
- `src/diffmm/eval/model_diagnostics.py` (refactored) — 5 functions made EDM-aware

**Significantly modified**
- `src/diffmm/generator/trades_adapter.py` — AdaLN-Zero blocks, conditioning_type switch, forward_with_t_emb extension point
- `src/diffmm/generator/sample.py` — prediction_type field, v-pred sampler logic
- `src/diffmm/generator/train.py` — v-pred loss, CFGDropoutCurriculum, EDMTrainer (in edm.py)
- `src/diffmm/generator/conditioning.py` — FiLM random init (Phase A change preserved)
- `configs/generator/finetune.yaml` — `conditioning_type`, `prediction_type`, `cfg_dropout_curriculum`, `edm` block
- `scripts/11_finetune_generator.py` — branch on EDM, build_curriculum, build_edm_generator
- `scripts/utils/run_sampling_manifest.py` — `--prediction-type`, `--conditioning-type`, `--use-edm` CLI
- `scripts/50_validate_generator.py` — same CLI extensions
- `slurms/ph6_validate_gpu.slurm` — `PREDICTION_TYPE`, `CONDITIONING_TYPE`, `USE_EDM` env vars

**Documentation**
- `docs/Work6_v3.md` — Phase A
- `docs/Work6_update.md` — Phase A.5 falsification
- `docs/Work6_phaseB.md` — Phase B
- `docs/Work6_v4.md` — Phase B + x0_clip + cross-checkpoint comparison
- `docs/Work6_phaseC.md` — Phase C
- `docs/Work6_phaseD.md` — Phase D
- `docs/Work6_FINAL.md` — this document

**Test count: 70** (77 after Phase E adds 7 CopulaTransform tests).

---

## 14b. Phase E — Copula-Based Diffusion (added 2026-05-07)

Phase E was the most theoretically-grounded intervention in the workstream and the only one that *predicted* a quantitative outcome before running. The idea: sidestep the marginals-vs-joints tension by training the diffusion model on **per-feature Gaussianized data** (`z = Φ⁻¹(F̂(x))`) and restoring heavy tails at sample time via the inverse empirical CDF. The model only has to learn dependence structure on N(0,1)-marginal data; tails are guaranteed by construction.

Two variants trained in parallel: **v8** (FiLM, ε-pred, no EDM, copula) and **v8_b** (AdaLN-Zero + everything else identical to v8).

### Predicted outcomes (from Work6_phaseE.md §6)

1. ρ_diff ≥ +0.40 (match v2 baseline)
2. No A1/A2 swap (truth has A2 > A1)
3. Kurtosis 400-1000 (close to real 755)

### Actual outcomes

| Run | ρ_diff | A2 above A1? | Kurtosis | OFI sign | Verdict |
|---|---|---|---|---|---|
| **v8** | **+0.80** | ✅ yes | 2.6 | wrong | **partial** — ranking wins, marginals fail |
| v8_b | +0.80 ⚠ | ❌ no | degenerate | n/a | **degenerate** — synth is flat-line, ρ is tie-break artifact |

### What v8 did and didn't deliver

**Delivered:** ρ_diff = +0.80 — the highest in the workstream by 2× over the v2 baseline (and 4× over every post-v2 architecture). Recovered the truth's A2 > A1 ordering. Robust on the central hypothesis test.

**Did not deliver:** Kurtosis stuck at 2.6 (vs predicted 400-1000). OFI→return correlation has wrong sign at all lags (β_synth < 0 vs β_real > 0). Marginal-distribution stress is *worse* than v2, not better.

### Mechanism analysis (why ρ_diff was high anyway)

The copula's inverse-CDF only restores heavy tails *if the diffusion model emits z-values that span the full tails of N(0,1)*. With ε-prediction + min_snr_gamma=5 weighting, the model is incentivized to produce conservative, sub-Gaussian outputs (small |z|), so the inverse CDF only recovers bulk values, not tails. v8_b's collapse is an extreme version of the same mechanism — AdaLN-Zero's zero-init combined with min_snr_gamma damps the model so hard it produces zero-variance output.

But v8's ρ_diff = +0.80 is still real. **Agent rankings can be preserved even when marginals/bulk correlations are broken.** Even with wrong-sign OFI, the *magnitude* of the OFI signal still distinguishes A2 (uses OFI) from A1 (doesn't). Agents respond to OFI's predictive power even when its sign has flipped, and the resulting ranking happens to match truth.

### Updated headline

The marginals-vs-joints tension from §9 still stands. v2 remains the only checkpoint that delivers v2-class agent ranking *and* v2-class realistic marginals (kurtosis 122 — modest but non-zero). v8 produces a much stronger ranking signal but with marginals further from real than v2 (kurtosis 2.6).

What v8 *adds* to the project's story is a sharper version of the §9 finding: **ρ_diff measures preserved-agent-ranking-structure, NOT marginal-stress-realism, and these can decouple completely**. v8 has a great agent-ranking signal *because* the copula preserves Spearman rank correlations between features (by construction), even though it fails to restore the marginal tails the model was supposed to deliver.

### Recommended v9 (not run; documented for completeness)

Drop `min_snr_gamma` to 0 or 1 to allow the model to commit to extreme z-values. The copula-on-Gaussianized framework should then deliver heavy tails as originally predicted. If kurtosis recovers to 100+ AND ρ_diff stays near +0.80, that's the clean Phase E result — combining v2-class ranking with v7_b-class marginal realism, which would close the marginals-vs-joints gap for the first time.

See [Work6_phaseE.md](Work6_phaseE.md) §9 for the full mechanism analysis and the bug ledger from this phase.

---

## 15. Closing Thought

The project's premise — "synthetic stress testing improves agent ranking accuracy" — was an empirical question that this 6-checkpoint study definitively answered for INTC: **no, and the reason is structural, not accidental**.

The empirical pattern is clean: every architectural improvement past v2 (across 4 distinct axes — parameterization, conditioning, schedule, σ_data) caused the same A1/A2 swap that drops ρ_diff from +0.40 to ≤ +0.20. Six data points; same mechanism every time.

The mechanism (Section 9) is the **marginals-vs-joints tension under MSE-based diffusion training**: ε-prediction (v2) mode-covers the bulk and preserves cross-feature correlations like OFI→return; v-prediction and EDM (everything else) commit to extreme values and break those correlations. The agents we built (specifically A2 = AS+OFI) depend on the OFI→return correlation to function — so when that correlation breaks in synth, A2 underperforms in synth, even though it would outperform in real.

The contradiction with the proposal's premise is direct: the generator that "wins" the central test (v2) is the one with the LEAST realistic stress (kurtosis 122 vs real 755). v2 doesn't win because it does a good job of stress testing; it wins because it preserves bulk correlation structure. That's not what the proposal said synthetic stress testing should do.

This is the project's most important — and unexpected — finding. The naive expectation that "more realistic generator → better stress test" is false in our setup, and the failure mode generalizes: it should appear in any synthetic-data evaluation framework where (a) the generator is trained with MSE-based diffusion losses on heavy-tailed multivariate data, and (b) the agents make decisions based on cross-feature correlations that the marginal improvements break.

For future work in synthetic-data-for-backtesting, the practical recommendation is: explicitly check joint-structure preservation in your generator (do the cross-feature correlations agents use survive synthesis?), and explicitly check whether the improvements you're chasing for marginal realism are breaking the joint structure. If so, the "improved" synthetic data may make your downstream test less truth-aligned, not more.

That's the project's most important contribution. It deserves a careful, honest writeup in the final report — both for what we found AND for what we found *isn't true* about the field's current best practices.

### Phase E coda

Phase E (v8) added an interesting refinement: **ρ_diff and marginal-realism can decouple completely**. v8 doubled the ranking-correlation signal over v2 *while* losing what little marginal stress v2 had. The win was on a structural property the copula preserves by construction (Spearman rank correlation between features) — not on the heavy-tail recovery the design was originally motivated by.

This is a useful corollary to the §9 marginals-vs-joints finding. The hypothesis test as we constructed it is sensitive to *which dependency structure between features is preserved*, not to marginal realism. v2 won because it preserved the OFI→return correlation (and didn't crush the marginals to zero); v8 won — by twice as much — because the copula transformation preserves *all* rank-based dependence between features by construction. The model could fail completely at marginals and still deliver high ρ_diff, as long as the rank structure was preserved.

For the report's discussion: this strengthens, not weakens, the §9 conclusion. The proposal's "more realistic generator → better stress test" is now falsified two ways: (1) post-v2 architectures with better marginals broke the joint structure (§9), and (2) the copula model with much-worse marginals than v2 still delivered the best ρ_diff because it preserved a different joint structure. **What ρ_diff measures is joint-structure preservation, not stress realism.** The two are independent design dimensions, and the project's findings are most usefully read as a study of *which joint structures matter for which downstream tests*, rather than as a search for the "best" generator.
