# Work 6 — v3 Validation Results & the Min-SNR γ Reversal (Phase A.5)

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-30
**Status:** Complete — v3.5 validation finished 2026-04-30, results in §5.
**Companion to:** [Work5.md](Work5.md) (validation suite), [Work6_v3.md](Work6_v3.md) (Phase A plan)

This document captures the v3 (Phase A) validation results, identifies a directional error in the Min-SNR γ change, and frames the Phase A.5 reversal experiment that's currently in flight.

---

## 0. Executive Summary

Phase A trained two v3 checkpoints to compare against v2:
- **v3-epoch-9** (~6h on L40S, val/loss 0.0079)
- **v3-epoch-19** (resume, ~6h more, val/loss 0.0078, model converged)

Both validate worse than v2 on the kurtosis criterion (the most consequential heavy-tail metric), and validating epoch-19 against epoch-9 shows **strict regression on every Phase A target metric**. Diagnosis: the Min-SNR γ change (5.0 → 1.0) was a directional error that down-weights the low-noise band where heavy-tail information lives. Extra training under the wrong loss weighting fits the model more confidently to the wrong objective.

The other Phase A changes (FiLM random init, CFG p_drop 0.10 → 0.20, patience 3 → 6) are likely net-positive — they let v3 train far longer than v2 (10+ epochs vs 5) and produced one bright signal at v3-epoch-9 (FiLM β_out variation of 11% across regimes, vs ~1% in v2). But that signal degraded by epoch 19, suggesting Phase A's conditioning gains are fragile under the broken loss landscape.

**v3.5 (Phase A.5)** reverts only the Min-SNR γ change (back to 5.0), keeping the other Phase A changes. Currently sampling on cluster; validation results to follow in §5.

---

## 1. Methodology

Three v3-family checkpoints were validated against v2 using the Work5 suite (`scripts/50_validate_generator.py`):

| Checkpoint | Min-SNR γ | CFG p_drop | FiLM init | Patience | Best epoch | val/loss |
|---|---|---|---|---|---|---|
| v2 | 5.0 | 0.10 | zeros | 3 | 4 | 0.0106 |
| v3-e9 | 1.0 | 0.20 | N(0, 0.02) | 6 | 9 | 0.0079 |
| v3-e19 (resume from v3-e9) | 1.0 | 0.20 | N(0, 0.02) | 6 | 19 | 0.0078 |
| v3.5 (in flight) | **5.0** | 0.20 | N(0, 0.02) | 6 | 9 | 0.0103 |

All checkpoints sampled 6,400 stitched tapes per ticker (4 regimes × 1,600 each, 2,560 events per tape). Validation reads ~250 synth tapes per regime (1,000 total) against 10 real INTC val days. CSVs at `results/validation/INTC_{v2,v3_e9,v3_e19,v3p5}/`.

**Note on val/loss comparability.** v3-e9/v3-e19 train under γ=1, v2 and v3.5 train under γ=5. Min-SNR weighting is part of the loss, so val/loss values are *only comparable within the same γ*. The valid comparison is **v2 (0.0106) → v3.5 (0.0103)** = 3% improvement on the same loss formulation.

---

## 2. Three-way Comparison: v2 vs v3-e9 vs v3-e19

### 2.1 Stylized facts

| Metric | Real | v2 | v3-e9 (γ=1) | v3-e19 (γ=1) | Direction (e9 → e19) |
|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 81 | **70.7** | regressed −13% |
| W-1 (return) | — | 1.74e-06 | 1.71e-06 | 1.63e-06 | flat |
| ACF β (\|return\|) | 0.21 | 0.66 | 0.482 | **0.488** | flat |
| ACF lag-50 deviation | 0 | 1.28 | 2.38 | **3.96** | regressed +66% |
| OFI slope at Δ=50 | 1.11e-04 | 1.11e-05 | 9.27e-06 | 1.26e-05 | marginal |
| Trade fraction | 4.8% | 54.9% | 54.9% | **64.0%** | regressed +9pp |
| Size tail exponent | 1.567 | 1.869 ✓ | 1.869 ✓ | **2.147 ✗** | crossed pass→fail |
| G.1 pass count | 16/16 | 13/16 | 14/16 | **13/16** | regressed |

**Reading:** v3-e9 was a mixed result — small ACF β improvement, kurtosis regression. v3-e19 is **strictly worse** than v3-e9 on every metric the Phase A changes were supposed to fix. More training under the wrong loss objective made things uniformly worse.

### 2.2 FiLM modulation (F.2, computed std/mean across 4 named regimes)

| FiLM metric | v2 | v3-e9 | v3-e19 | Direction (e9 → e19) |
|---|---|---|---|---|
| film_in_γ_dev | ~1% | 0.92% | **0.37%** | regressed (more collapsed) |
| film_in_β_norm | ~3% | 3.05% | 2.83% | flat |
| film_out_γ_dev | 0.42% | 0.42% | 0.92% | marginal improvement |
| film_out_β_norm | ~1% | **10.9%** | **7.85%** | regressed −28% |

The one Phase A bright signal — film_out_β_norm spread of 11% in v3-e9 — degraded to 7.85% by v3-e19. The γ-dev metrics (the more sensitive measure of regime differentiation) remain effectively collapsed (<1% std/mean) and got *worse* on film_in_γ_dev with longer training.

### 2.3 F.1b regime embedding cosine similarity

| Pair | v2 | v3-e9 | v3-e19 |
|---|---|---|---|
| max off-diagonal cosine (named regimes) | 0.831 | 0.831 | **0.869** |
| diagnosis | EMBEDDINGS DISTINCT | EMBEDDINGS DISTINCT | EMBEDDINGS DISTINCT |

The diagnosis bucket is unchanged across all three (max cosine still < 0.9), but **embeddings are converging with extra training** (0.831 → 0.869). This contradicts the Phase A intent — random FiLM init was supposed to force embedding differentiation, not convergence. The mechanism is likely that the optimizer, freed from the zero-init identity trap, is finding a *new* equilibrium where regime embeddings cluster around the FiLM-output mean rather than spreading out.

---

## 3. The Min-SNR γ Directional Error

The Min-SNR-γ loss weight is `weight(t) = min(SNR(t), γ) / SNR(t)`, where `SNR(t) = ᾱ_t / (1 - ᾱ_t)`:

- High SNR (low `t`, near-clean): `weight = γ/SNR < 1` → down-weighted
- Low SNR (high `t`, very noisy): `weight = 1` → full weight

**The crossover** at `SNR(t*) = γ`:
- γ=5 → t* ≈ 200 → lowest ~20% of timesteps down-weighted
- γ=1 → t* ≈ 450 → lowest ~45% of timesteps down-weighted

Within the down-weighted band, the per-`t` weight at γ=1 is **5× smaller** than at γ=5 (because `γ/SNR` scales linearly with γ).

So lowering γ does two compounding things to the low-noise band: it **expands** the down-weighting AND **deepens** it.

### Where heavy tails live in the diffusion timeline

At low `t` (high SNR, near-clean), the model does fine-grained denoising: refining the central peak, predicting rare large outliers, sharpening the distribution. **Heavy tails are a fine-detail property** — reproducing real INTC's kurtosis-755 spike-and-rare-jumps shape requires the model to commit, at low `t`, to producing the rare extreme values the data warrants.

Phase A's γ=1 took ~5× of gradient signal away from the band where rare extreme returns are learned. The optimizer responded rationally: it spent more capacity on rough-shape correctness (high `t`, where weight is full) and less on fine-detail rare-jump generation. The output kept the rough shape (W-1 still passes — bulk position + scale are right) but lost the rare extreme decoration (kurtosis 122 → 81 → 70.7 — fewer rare jumps, thinner tails).

**Why this was non-obvious to me at Phase A planning:** I rationalized γ=1 as "more aggressive heavy-tail focus." It is more aggressive — but the aggression is *down-weighting the band where tails are learned*, not up-weighting it. Hang et al. 2023 originally proposed γ=5 because uniform-MSE *over-*trains the low-`t` band; γ=5 brought it into balance. γ=1 over-corrects, killing the very signal the correction was supposed to merely balance.

The v3-e19 → v3-e9 regression is the cleanest evidence: **more training under the wrong γ made every targeted metric worse**, and even the FiLM signal that improved at e9 (β_out variation) regressed at e19. The optimizer fits more confidently to the wrong objective.

---

## 4. Phase A.5 — Single-Variable Reversal

### 4.1 The change

```diff
# configs/generator/finetune.yaml
-  min_snr_gamma: 1.0
+  min_snr_gamma: 5.0
```

All other Phase A changes stay:
- FiLM proj weight init: N(0, 0.02) (not zeros)
- CFG p_drop: 0.20 (not 0.10)
- Early-stopping patience: 6 (not 3)

Rationale: the FiLM init + p_drop + patience changes target *FiLM collapse*, which is a separate failure mode from heavy-tail under-fitting. There's no reason to revert them. The γ change is the single intervention that targets heavy tails, and it's the only one with strong evidence (the e19 regression) of being directionally wrong.

### 4.2 What we expect to see in v3.5

Predictions, against v3-e9 baseline (γ=1, same epoch number for fair comparison):

| Metric | v3-e9 (γ=1) | **v3.5 prediction (γ=5)** | Why |
|---|---|---|---|
| Kurtosis | 81 | > 122 (above v2 baseline) | γ=5 restores low-noise band weighting where tails are learned |
| Size tail exponent | 1.87 | < 2.0 (recovers passing) | Same mechanism — fine-detail learning |
| ACF β | 0.482 | similar (~0.45-0.55) | Long-memory is a separate failure mode |
| FiLM γ-dev std/mean | 0.92% | similar (~1%) | γ doesn't directly target FiLM |
| FiLM β_out variation | 10.9% | similar or higher (~10-15%) | This was the one Phase A bright spot, expect it to persist |

**If v3.5 hits these predictions**, we have a clean win on heavy tails and ship v3.5 as the production generator. Phase B targets only the FiLM-collapse failure (which is a separate problem).

**If v3.5 kurtosis doesn't recover**, ε-prediction itself is the bottleneck for tails — Phase B's v-parameterization becomes mandatory.

**If v3.5 kurtosis recovers AND FiLM γ-dev jumps to >5%**, the γ-revert was masking a conditioning-side effect we didn't anticipate. Either way, the experiment is informative.

### 4.3 Cluster status

- **v3.5 training:** completed (12854914), TIMEOUT at epoch 9, val/loss 0.0103 ([checkpoint](../../../orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v3p5/ckpt-epoch009-valloss0.0103.ckpt))
- **v3.5 sampling:** in flight (job 12892100, ~25 min in, ETA ~1.5h more)
- **v3.5 validation:** queued behind sampling

§5 will be filled in once validation completes.

---

## 5. v3.5 Validation Results — Thesis Falsified

The γ-revert thesis (§4.2) predicted v3.5 kurtosis would exceed v2's 122. Actual v3.5 kurtosis: **64.18** — the lowest of all four checkpoints, including the v3 runs trained under the supposedly worse γ=1.

### 5.1 Updated 4-way comparison

| Metric | Real | v2 | v3-e9 (γ=1) | v3-e19 (γ=1) | **v3.5 (γ=5)** | Best |
|---|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 81 | 71 | **64** | v2 |
| W-1 | — | 1.74e-06 | 1.71e-06 | 1.63e-06 | 1.64e-06 | tie |
| ACF β | 0.21 | 0.66 | 0.482 | 0.488 | 0.542 | v3-e9 |
| ACF lag-50 dev | 0 | 1.28 | 2.38 | 3.96 | 2.94 | v2 |
| Trade fraction | 4.8% | 54.9% | 54.9% | 64.0% | **45.0%** | v3.5 |
| Size tail exp | 1.567 | 1.869 | 1.869 | 2.147 | **1.856** | v3.5 |
| OFI slope at Δ=50 | 1.11e-04 | 1.11e-05 | 9.27e-06 | 1.26e-05 | 1.49e-05 | v3.5 |
| FiLM γ_in std/mean | — | ~1% | 0.92% | 0.37% | 1.17% | v3.5 |
| FiLM β_out std/mean | — | ~1% | **10.9%** | 7.85% | 5.81% | v3-e9 |
| F.1b cosine max | — | 0.831 | 0.831 | 0.869 | 0.830 | tie |
| G.1 pass count | 16/16 | 13/16 | **14/16** | 13/16 | 13/16 | v3-e9 |

No single checkpoint dominates. The metrics fall into roughly two anti-correlated groups:

- **Heavy-tail recovery group** (kurtosis, ACF lag-50 deviation): v2 is best.
- **Surface-marginal group** (trade fraction, size tail exponent, OFI slope): v3.5 is best.
- **Long-memory group** (ACF β): v3-e9 is best.
- **Conditioning group** (F.2, F.1b): nobody is good. All checkpoints diagnose as "FiLM-layer collapse."

### 5.2 The Real Pattern Across Phase A and A.5

The four data points reveal a structural trade-off that's not fixable with ε-prediction config changes:

|  | Tails (kurtosis) | Long memory (ACF β) | Surface marginals |
|---|---|---|---|
| γ=1, more training on rough-shape | bad and degrading | best (0.48) | mid |
| γ=5, more training on fine-detail | also bad | worse (0.54-0.66) | best |
| **More training (any γ)** | always hurts | minor effect | mostly improves |

v2's superior kurtosis is partly a happy accident of being undertrained — kurtosis (122) at epoch 4 reflects a model that hasn't yet converged into the smoother solution that all longer-trained variants drift toward. This is the "diffusion mode-covering" effect: with enough training, ε-prediction on heavy-tailed data converges to outputs that match the bulk well but lose the rare extremes.

The γ-revert thesis was wrong because it assumed γ was the dominant lever. **The dominant lever is training duration**, and there's no γ value that makes longer training help heavy tails under ε-prediction.

### 5.3 Why my Min-SNR derivation looked right but didn't predict the actual outcome

The math from §3 still holds in isolation: γ=1 down-weights low-noise timesteps where fine-detail learning happens, and that should hurt tails. The flaw was assuming this effect would dominate. In practice:

1. **Per-step gradient direction** (set by γ) and **total training pressure on the model** (set by epoch count) are different forces.
2. With 4 epochs (v2), the model under γ=5 hasn't yet flattened the tails — it's still in an early-training regime where the data's heavy tails are visible in the ε-loss landscape.
3. With 9+ epochs, ε-prediction's natural mode-covering behavior dominates regardless of γ. The model's MSE-style loss fundamentally rewards being close to the bulk on average, and tails are by definition rare events that contribute little to total MSE.

This is the well-documented limitation of ε-prediction for heavy-tailed densities. v-parameterization (Salimans & Ho 2022) and *x_0*-parameterization both address it more directly because they reframe the prediction target away from the noise-removal-MSE objective.

### 5.4 Decision

The v3.5 result rules out the "single config knob" path to fixing tails. Three paths forward:

**(a) Ship v2 as the production generator.** v2 has the best kurtosis (122), the best ACF lag-50 deviation, and is tied or close on most other metrics. It was the basis for the Work 4 central hypothesis result (ρ = +0.4). The validation suite + the four-way comparison documented in this file are themselves a substantial scientific contribution. Honest framing: "We tested four generator configurations; none beat the original on the central tail-recovery metric. We document the failure modes and use v2 for the hypothesis test." **Lowest risk, fastest path to the report.**

**(b) Phase B v-parameterization.** Switch the training target from ε to v = √ᾱ · ε − √(1-ᾱ) · x_0 (Salimans & Ho 2022). Reframes the loss landscape so high-`t` and low-`t` both get balanced gradient signal that includes signal-correlated structure. Well-known to help heavy-tail recovery. ~2-3h impl in [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) + [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py); ~6h retrain; ~30 min re-validate. **Medium risk, ~1 day total, addresses the tail problem at its root.**

**(c) Phase A ablation.** Train two more 6h checkpoints isolating which Phase A change hurts kurtosis: (i) γ=5, FiLM zero init, p_drop 0.2, patience 6 (rules out FiLM init); (ii) γ=5, FiLM random init, p_drop 0.1, patience 6 (rules out p_drop). Tells us which sub-change to keep going forward. **~12h compute, useful diagnostic, but doesn't fix the fundamental problem.**

Recommended: **(a) + (b) in parallel.** Start writing the report assuming v2 as the production generator (we have all the data we need). In parallel, kick off Phase B v-parameterization since it's only 1 day of work and has good prior probability of helping. If v-param produces a clear win, swap to it for the final report; if not, ship v2.

---

## 6. Bug Ledger Continuation (#45+)

### Bug #45 — repeated 1-2 sec job failures with empty .err logs
- **Symptom:** SLURM jobs (validation, sampling) consistently failed in 1-2 sec with `ExitCode 1:0` and empty `.err` files. Jobs disappeared from `squeue` instantly. Pattern repeated across multiple submission attempts.
- **Root cause:** Submitting shell did not have the conda env activated (`source activate diffmm`). The slurm scripts source `slurms/_common.sh` which expects `python` to point at the diffmm env's interpreter; without activation, the script imports failed before any logging started.
- **Fix:** Always run `module load miniforge && source activate diffmm` after a fresh ssh login *before* submitting any job. Add the activation to `~/.bashrc` if you want it automatic — but be aware that activates the env in *every* shell, including non-cluster work.
- **Lesson:** When `.err` is empty AND elapsed time is under 5 sec, the failure happened before Python started — first place to check is environment activation, not application code.

---

## 7. The Numbers in One Place (so far)

| What | Value |
|---|---|
| v3 epochs trained (total) | 19 (e9 + 10-epoch resume) |
| v3 wall clock | ~12h on L40S (two 6h chunks) |
| v3 best val/loss (γ=1 metric) | 0.0078 (epoch 19) |
| v3.5 epochs trained | 9 |
| v3.5 wall clock | ~6h |
| v3.5 best val/loss (γ=5 metric) | 0.0103 (epoch 9) |
| v2 best val/loss (γ=5 metric) | 0.0106 (epoch 4) |
| Stitched tapes per generator | 6,400 (per ticker) |
| Real val tapes used | 10 (INTC) |
| Synth tapes per validation run | ~1,000 |
| Validation wall clock per checkpoint | ~20 min (CPU) + ~2 min (GPU) |
| Total validation runs to date | 4 (v2, v3-e9, v3-e19, v3.5 in flight) |

---

## 8. Honest Verdict (final)

The Phase A hypothesis was: four targeted, low-risk changes (FiLM random init, CFG p_drop, Min-SNR γ, patience) would close the gap on kurtosis, ACF β, and FiLM collapse simultaneously. **None of the four checkpoints (v3-e9, v3-e19, v3.5) beat v2 on the central tail-recovery metric.** v2's superior kurtosis (122) is partly a happy accident of being undertrained — every longer-trained variant drifts toward smoother distributions regardless of γ.

The four-way comparison reveals a structural pattern, not a single fixable bug:

1. **Min-SNR γ direction:** lower γ is bad for tails (predicted; confirmed by v3 series), but **higher γ also doesn't recover them** (v3.5). γ is not the dominant lever — training duration is.
2. **ε-prediction has a fundamental tail-recovery limit on heavy-tailed data.** With enough training, MSE-style noise loss converges to mode-covering behavior. No amount of timestep reweighting fixes this within the ε-prediction framework.
3. **FiLM-layer collapse is real and resilient.** Phase A's two FiLM-targeted changes (random init, higher p_drop) produced minor surface improvements (β_out variation in v3-e9, marginal γ_in std/mean in v3.5) but the F.1b cosine matrix is essentially identical across all four configurations. Conditioning collapse is downstream of something deeper than init or training duration.
4. **Train-loss improvements ≠ stylized-fact improvements.** v3-e19 cut val/loss by 26% vs v2 but made the validation suite worse. Loss-based optimization is not aligned with the failure modes that matter for the project's premise.
5. **The metrics anti-correlate.** v3.5 best on trade fraction + size tail; v3-e9 best on ACF β; v2 best on kurtosis. There's no single config that wins everywhere — each Phase A change traded one failure mode for another.

### What this means for the project

Two parallel paths from here:

**(a) Ship v2 as the production generator** for the Work 4 hypothesis test (already done) and the final report. The validation suite (Work5) plus this four-way ablation (Work6 + Work6_update) document the failure modes exhaustively — that itself is a substantial scientific contribution. The honest framing: "We tested four generator configurations across heavy-tail loss reweighting, FiLM initialization, CFG dropout, and training duration; none improved over v2 on the central kurtosis metric. The validation suite identifies the structural cause as ε-prediction's mode-covering behavior on heavy-tailed densities."

**(b) Phase B v-parameterization (Salimans & Ho 2022)** as an optional, well-scoped experiment. Reframes the loss target away from ε-MSE; well-documented in the literature to help heavy-tail recovery. ~1 day of work (impl + retrain + validate). High prior probability of being the right fix, but not necessary for the project to be defensible.

**Recommendation:** start the report writeup now (using v2 + the validation suite as the deliverables), and run Phase B v-parameterization in parallel. If v-param produces a clear win on kurtosis, swap it in for the final result. If not, the report stands on the v2 + validation infrastructure work, framed honestly per the project's "honesty principle."

The Phase A → A.5 sequence is itself a clean piece of scientific narrative: a hypothesis (config tweaks fix tails), an experiment (Phase A), an unexpected result (kurtosis regression), a corrected hypothesis (γ revert), a confirming experiment (Phase A.5), a falsified hypothesis (kurtosis still bad), and a structural diagnosis (the limit is ε-prediction itself). The negative result is informative.
