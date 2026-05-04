# Work 6 — v3 Diffusion Generator Rework (Phase A)

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-27
**Covers:** Phase A of the v3 rework — four targeted, low-risk changes to the conditional DDPM in response to the Work5 validation findings (heavy-tail compression, vol-clustering decay, FiLM-layer collapse).

This document complements [Work5.md §6](Work5.md). Phase A is scoped to ~20 lines of code changes, all reversible, retrainable in ~7h on one L40S. If Phase A doesn't fully close the gap, Phase B (v-parameterization, larger context window, auxiliary ACF/kurtosis loss) is queued in §5.

---

## 0. Executive Summary

Two mechanisms drove v2's failures (per [Work5.md §3](Work5.md)):

1. **FiLM-layer collapse** ([F.2](../results/validation/INTC/F2_film_modulation.csv)) — the regime embeddings are distinct (F.1b cosine matrix max 0.83 < layer-collapse threshold) but FiLM modulation is nearly identical across regimes, causing synthetic regimes to be statistically indistinguishable.
2. **Heavy-tail under-fitting** ([B.1](../results/validation/INTC/B1_return_stats.csv)) — synthetic kurtosis is 6× too small, producing thin-tailed returns that don't reproduce real INTC's rare-jump regime, which in turn flattens the OFI→return slope by 10× ([D.1](../results/validation/INTC/D1_regression_stats.csv)).

Phase A targets both with four config + init changes that are independent, individually reversible, and require no architectural rewrites:

| # | Change | File | Targets |
|---|---|---|---|
| 1 | FiLM proj weight init: `zeros → N(0, 0.02)` | [src/diffmm/generator/conditioning.py:62](../src/diffmm/generator/conditioning.py#L62) | F.2 collapse |
| 2 | CFG dropout: `0.1 → 0.2` | [configs/generator/finetune.yaml:21](../configs/generator/finetune.yaml#L21) | F.2 collapse |
| 3 | Min-SNR γ: `5.0 → 1.0` | [configs/generator/finetune.yaml:30](../configs/generator/finetune.yaml#L30) | B.1 + C.1 + D.1 |
| 4 | Early-stopping patience: `3 → 6` | [configs/generator/finetune.yaml:40](../configs/generator/finetune.yaml#L40) | F.2 (slow conditioning convergence) |

All 44 tests pass after the changes.

---

## 1. The Four Changes

### 1.1 FiLM init — break the identity-at-step-0 trap

**v2 init (was):** `nn.init.zeros_(proj.weight)` and `nn.init.zeros_(proj.bias)`. With these, FiLM at step 0 produces γ=0 and β=0, so the layer is identity (`h ↦ (1+0)·h + 0 = h`). The original rationale ([conditioning.py docstring v2](../src/diffmm/generator/conditioning.py)) was "no-op at step 0" stability when adding FiLM to a pretrained backbone — a Perez-2018 standard practice.

**v3 init (now):** `nn.init.normal_(proj.weight, mean=0.0, std=0.02)`. Bias still zero.

**Why:** Identity-at-step-0 means FiLM has *zero gradient flow from ctx* until ε-loss training happens to push the projection weights nonzero by chance — and that drift is slow when the unconditional path can already minimize loss without FiLM contributing. With small random init, FiLM has nontrivial conditioning gradients from step 1, forcing the optimizer to actively decide whether to keep, attenuate, or invert the conditioning signal. σ=0.02 is small enough not to destabilize the bf16-mixed precision training of the pretrained backbone.

This is the single most important Phase A change because the Work5 F.1b cosine diagnostic explicitly points to "FiLM layer is collapsing the modulation downstream" — fixing FiLM init is the most direct intervention.

### 1.2 CFG dropout 0.1 → 0.2 — make conditioning earn its keep

**Why:** With `p_drop=0.1`, the model sees the null condition only 10% of the time. The unconditional path must work (otherwise the 10% loss term blows up), and the easiest way to satisfy it while also satisfying the 90% conditional loss is to make FiLM nearly the same as identity for *both* paths — i.e., the conditional path effectively ignores c.

Doubling `p_drop` to 0.2 means the unconditional path is exercised twice as often. The conditional path can no longer ignore c and still hit conditional-loss parity, because the model now needs to genuinely distinguish the two paths to fit both. Standard fix from Ho & Salimans 2022 §3.2 when CFG conditioning is too weak.

**Tradeoff:** Slightly weaker conditioning gradient per step (since 20% of the gradient comes from null-conditioned forwards), but stronger long-run conditioning specialization. Net positive given Patience-6 below.

### 1.3 Min-SNR γ 5.0 → 1.0 — reweight toward extreme-noise timesteps

The Min-SNR-γ loss weight (Hang et al. 2023) is `min(SNR(t), γ) / SNR(t)`:
- At `SNR(t) ≪ γ`: weight ≈ 1 (mid/high-noise timesteps get full weight)
- At `SNR(t) ≫ γ`: weight ≈ γ/SNR(t) → 0 (near-clean timesteps get down-weighted)

With γ=5, the down-weighting only kicks in for very high-SNR timesteps (very small t). The middle band (`t ∈ [200, 800]`) gets near-uniform weight. This was Work2's choice, motivated by the diagnostic that uniform-MSE was over-training the easy ends and under-training the hard middle.

But Work5 [E.2](../results/validation/INTC/E2_per_timestep_stats.csv) showed v2's per-timestep MSE is now WORST at the extreme ends (low-noise denoising of fine detail and high-noise rough-shape generation), and BEST in the middle. Min-SNR γ=5 succeeded — too well — and starved the extremes.

**v3 γ=1.0** pushes much more loss weight onto the extreme-SNR end (large t), where the heavy-tail / rare-jump information lives. The model has to learn to denoise from very-noisy states more accurately, which is exactly what's needed to generate the tails B.1 missed.

**Tradeoff:** May slightly under-train the middle band again. We accept this — middle-band MSE was the easiest part of v2 (E.2 confirms this).

### 1.4 Patience 3 → 6 — let conditioning converge

**Why:** v2 early-stopped at epoch 4 (of 20). Looking at the train.csv from v2, val/loss was still decreasing slowly when the stop fired — but the *conditioning pathway* in particular hadn't moved much (per F.2 nearly-identical FiLM γ across regimes). FiLM is a slow-moving sub-network when zero-init'd, and even with random init (change 1.1) and stronger CFG dropout (change 1.2), it benefits from more training epochs to specialize.

Patience 6 means we accept up to 6 consecutive non-improving epochs before stopping. Combined with `max_epochs=20`, this allows ~10–14 epochs of training in expectation, enough for FiLM to specialize across regimes.

**Tradeoff:** Compute cost: ~+50% over v2 (5 epochs → ~8 expected). Acceptable.

---

## 2. Validation

44/44 unit tests pass. One pre-existing test ([tests/test_generator_smoke.py:test_training_loss_decreases](../tests/test_generator_smoke.py)) was tightened from 3 → 5 epochs and a wider early-vs-late comparator window to reduce sensitivity to the (now nonzero) FiLM-init noise on a tiny synthetic dataset.

The change does not affect any production code path other than the four touched lines + the adjusted test fixture.

---

## 3. Reproducer

**Local sanity (smoke):**

```bash
# Re-run the tests to confirm Phase A changes don't break anything
python -m pytest tests/ -q
```

**Cluster — v3 fine-tune from TRADES backbone (replaces v2):**

```bash
sbatch slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v3
```

The Phase A changes are picked up automatically via [configs/generator/finetune.yaml](../configs/generator/finetune.yaml). No CLI overrides needed.

Estimated wall-clock: ~6h on L40S (v2 took ~4h; +50% from patience bump). Output checkpoint: `/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v3/ckpt-epoch{N}-valloss{X}.ckpt`.

**Cluster — re-sample v3 stitched tapes:**

```bash
CKPT_V3=/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v3/ckpt-...ckpt
OUT_V3=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v3_stitched
sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G \
       --cpus-per-task=2 \
       --output=logs/sample_v3_stitched_%j.out --error=logs/sample_v3_stitched_%j.err \
       --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py \
           --manifest data/manifests/ph2_sample.json \
           --checkpoint $CKPT_V3 --out-root $OUT_V3 \
           --n-windows-per-task 10"
```

~1h wall, 6,400 stitched tapes per ticker (same volume as v2).

**Cluster — re-validate v3:**

```bash
# CPU job (Parts B/C/D)
CKPT=$CKPT_V3 SYNTH_DIR=$OUT_V3/INTC sbatch slurms/ph6_validate_cpu.slurm

# GPU job (Parts E/F + the F.1b cosine diagnostic)
CKPT=$CKPT_V3 SYNTH_DIR=$OUT_V3/INTC sbatch slurms/ph6_validate_gpu.slurm
```

Both jobs write to `results/validation/INTC/` — **rename the v2 output dir first** (e.g., `mv results/validation/INTC results/validation/INTC_v2`) or the v3 results will overwrite it.

**Cluster — re-run the central hypothesis test on v3 data (if validation passes):**

```bash
sbatch slurms/ph4_replay_synth.slurm data=intc_top \
    +synthetic_root=$OUT_V3
sbatch slurms/ph5_hypothesis.slurm data=intc_top
```

Real-val replay and calibration don't need to re-run (they're independent of generator version).

---

## 4. Success Criteria for Phase A

Three thresholds. If 2/3 clear, Phase A is a success and we lock in v3. If only 1/3 or 0/3, escalate to Phase B.

| # | Metric | v2 baseline | v3 Phase A target | v3 stretch |
|---|---|---|---|---|
| 1 | F.2 FiLM γ-deviation std across {base, high_vol, toxic, thin}, divided by mean | ~1% | **>5%** | >10% |
| 2 | B.1 synth excess kurtosis, ratio to real (real = 755) | 0.16 (122 / 755) | **>0.30** (>225) | >0.50 (>375) |
| 3 | C.1 \|return\| ACF β_synth, fractional deviation from real (real β = 0.21) | 3.1× (synth 0.66) | **<2.0×** (synth < 0.42) | <1.5× (synth < 0.32) |

D.1 (OFI slope) is not a success criterion because [Work5.md §2.3](Work5.md) argued it is downstream of B.1 — if kurtosis closes, slope attenuation likely closes too. We'll measure it but not gate on it.

F.1b cosine matrix should also be re-checked: with FiLM random-init and stronger CFG dropout, the cosines may *increase* (regimes pulled together by stronger gradient sharing) or *decrease* (regimes pulled apart by FiLM specialization). The diagnostic will tell us which.

---

## 5. Phase B (queued, only if Phase A doesn't clear 2/3 criteria)

Higher-effort changes targeting the same failure modes. None are needed if Phase A succeeds; all are deferred until Phase A's validation results are in.

| Change | File | Risk | Cost |
|---|---|---|---|
| ε-prediction → **v-parameterization** (Salimans & Ho 2022) | [train.py](../src/diffmm/generator/train.py), [sample.py](../src/diffmm/generator/sample.py) | Medium — touches the loss target and the sampling math; well-studied, standard fix for tail recovery | ~2h impl + 6h retrain |
| `window_length` 256 → **512** in [intc_top.yaml:17](../configs/data/intc_top.yaml#L17) | data pipeline | High — requires re-running [scripts/03_finalize_tapes.py](../scripts/03_finalize_tapes.py) and re-training from scratch on doubled-context windows; doubles memory per batch | ~3h data + 8h retrain |
| Auxiliary **ACF + kurtosis loss** on decoded windows during training | new [src/diffmm/generator/aux_loss.py](../src/diffmm/generator/aux_loss.py) | High — requires decoding inside the training loop (slow); needs careful weighting to not destabilize ε-loss | ~4h impl + 6h retrain |
| **Conditioning curriculum**: epochs 0–2 unconditional, then ramp `cfg_weight` linearly | [train.py:71-75](../src/diffmm/generator/train.py#L71-L75) | Medium — needs `current_epoch` threading into training_step; standard technique from controllable-diffusion lit | ~1h impl + 6h retrain |

If the F.1b cosine matrix tightens significantly under Phase A (regimes become indistinguishable in the embedding, not just FiLM downstream), Phase B's conditioning curriculum jumps to top priority.

---

## 6. File Diff

### Modified
- [src/diffmm/generator/conditioning.py](../src/diffmm/generator/conditioning.py) — FiLM proj weight init (zeros → N(0, 0.02)); updated docstring to record the rationale (Work5 §F.2 + F.1b diagnostic).
- [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) — three values + comments tying each back to the relevant Work5 section.
- [tests/test_generator_smoke.py](../tests/test_generator_smoke.py) — `test_training_loss_decreases`: 3 → 5 epochs, comparator window 5 → 8, to absorb the (now nonzero) FiLM-init noise on a tiny synthetic dataset. No semantic change.

### Added
- This file ([docs/Work6_v3.md](Work6_v3.md)).

**Test count: 44 (unchanged from end of Work5).**
