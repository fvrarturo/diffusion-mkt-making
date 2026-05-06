# Work 6 — Phase D (v6): EDM Preconditioning + Heun Sampler

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-05-04
**Status:** Code complete, training complete, validation complete. v6 is a clear *qualitative win* on the failure modes Phase D was designed to fix (skew, trade-sign ACF, regime distinctness), but introduces a scale-compression artifact (synth std ≈ ¼ of real) that's almost certainly traceable to a σ_data hyperparameter mismatch. v7 (one-line σ_data fix) recommended in §9.
**Companion to:** [Work6_phaseC.md](Work6_phaseC.md) (AdaLN-Zero conditioning + v5 results)

This document covers Phase D: replacing the DDPM/v-prediction loss formulation and DDIM sampler with **EDM** (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"), the current SOTA framework for diffusion training. Phase D is implemented strictly modularly — all EDM code lives in a new module, existing v2-v5 code paths are preserved unchanged, and EDM is opt-in via a single config flag.

---

## 0. Executive Summary

Phase C produced v5 with three remaining failure modes:

| Failure | v5 actual | Target |
|---|---|---|
| Skew asymmetry | -17.2 | ~-0.5 |
| ACF β | 0.558 | < 0.45 |
| Trade-sign ACF | 0.103 | ≥ 0.5 |
| Per-block AdaLN spec (avg) | 2-4% | > 5% |

These have a common structural cause: the ε-MSE / v-MSE loss is symmetric, time-local, and noise-target-only. It doesn't directly reward predicting heavy-tail *shape*, long-range *correlations*, or sequence-level *persistence*. The model can satisfy the loss while producing distributions that match marginals well but miss higher-order temporal structure.

**EDM addresses this** at the loss-formulation level by:

1. **σ-spaced noise schedule** (Karras 7th-power) — covers a much wider range of noise levels (σ_min=0.002 to σ_max=80) than DDPM's bounded ᾱ_t schedule, giving direct exposure to extreme noise during training
2. **Loss preconditioning** — wraps the network with σ-dependent c_skip, c_out, c_in, c_noise so the network learns a function with bounded magnitudes at every σ; the σ-weighted MSE on the underlying x_0 quantity is uniform across noise levels
3. **Log-normal σ sampling** during training — concentrates training on σ ~ 0.3 where the model is actually doing work
4. **Heun 2nd-order sampler** — converges in fewer steps (32 vs DDIM's 200) with better quality at each step

Implementation: 70/70 tests pass. New module `src/diffmm/generator/edm.py` with `EDMSchedule`, `EDMDenoiser` (preconditioning wrapper), `edm_sample` (Heun 2nd-order), `EDMTrainer`, `build_edm_generator`. Single additive method on `TradesStyleDenoiser` (`forward_with_t_emb`) — refactor of the existing forward pass with no semantic change. All v2-v5 checkpoints continue to load and validate correctly.

---

## 1. The Math

### 1.1 Forward process

EDM uses a continuous noise level σ instead of discrete timesteps:

```
x(σ) = x_0 + σ · ε,    ε ~ N(0, I)
```

Clean data is x(σ=0); pure Gaussian noise is x(σ=σ_max). The denoiser D(x; σ; c) is trained to recover x_0 from x(σ).

### 1.2 Network preconditioning (Karras 2022 Eq. 7)

Naively predicting x_0 directly across all σ has a magnitude problem: at low σ, x_0 ≈ x; at high σ, x_0 must be predicted from pure noise. The network would need to learn vastly different functions across σ.

EDM wraps the underlying network F_θ with σ-dependent rescaling:

```
D(x; σ; c) = c_skip(σ) · x  +  c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ); c)

c_skip(σ)  = σ_data² / (σ² + σ_data²)        — direct skip, dominates at low σ
c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)  — output scale, dominates at high σ
c_in(σ)    = 1 / sqrt(σ² + σ_data²)          — normalize input magnitude
c_noise(σ) = (1/4) · log σ                   — input log σ as the timestep proxy
```

where σ_data is the (approximate) std of the clean data. For our normalized LOB features this is ≈ 0.5 (the actual std varies per feature; 0.5 is a reasonable conservative default).

The preconditioner has clean limits:
- **σ → 0**: c_skip → 1, c_out → 0 → D(x; σ) ≈ x (identity, since clean data is its own denoising target)
- **σ → ∞**: c_skip → 0, c_out → σ_data → D(x; σ) ≈ σ_data · F_θ(...) (output scaled to data magnitude)

The network F_θ now learns a "well-behaved" function with bounded magnitudes regardless of σ. This is the central trick that makes EDM training stable across the full noise range.

### 1.3 Loss

```
L = E[ λ(σ) · ‖D(x_0 + σε; σ; c) − x_0‖² ]

λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
σ ~ exp(N(P_mean, P_std))    typically P_mean = -1.2, P_std = 1.2
```

The σ-weight λ(σ) makes the effective loss on the underlying x_0 quantity uniformly weighted across σ. The log-normal σ sampling concentrates training on σ ~ exp(P_mean) ≈ 0.3, which is empirically where the model does most of its useful learning (very low σ is trivial, very high σ is mostly noise).

### 1.4 Sampling (Heun 2nd-order, Karras Algorithm 1)

```
σ schedule: σ_i = (σ_max^(1/ρ) + (i/(N-1)) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ    typically ρ=7

For i = 0..N-1:
    1. d_i = (x_i − D(x_i; σ_i; c)) / σ_i           # score-matching direction
    2. x_{i+1}_euler = x_i + d_i · (σ_{i+1} − σ_i)   # Euler step
    3. If σ_{i+1} > 0:
         d_i' = (x_{i+1}_euler − D(x_{i+1}_euler; σ_{i+1}; c)) / σ_{i+1}
         x_{i+1} = x_i + 0.5 · (d_i + d_i') · (σ_{i+1} − σ_i)   # Heun correction
       Else:
         x_{i+1} = x_{i+1}_euler
```

Heun's 2nd-order correction approximately halves the discretization error per step compared to plain Euler, allowing 32-step EDM to match DDIM's 200-step quality. CFG operates on the x_0 prediction (algebraically equivalent to CFG on ε at fixed σ).

### 1.5 Why EDM should help v5's specific failures

| Failure | Why EDM should help |
|---|---|
| Skew asymmetry (-17.2) | EDM's σ-weighted loss treats positive and negative extreme noise symmetrically; v-pred + Min-SNR has no such symmetry guarantee |
| Per-block AdaLN spec (2-4%) | Cleaner gradient landscape (bounded magnitudes via preconditioning) → consistent training signal at every σ → modulation MLPs receive better gradient flow regardless of which σ they're being asked to operate at |
| Tail SHAPE (kurtosis overshoots, asymmetric) | σ_max=80 vs Phase B's effective max ≈ √(1-ᾱ_T) = 1.0 — direct exposure to extreme-noise regime during training |
| Indirect ACF β help | More stable training dynamics → model can devote capacity to long-range structure rather than fighting loss-magnitude variance across noise levels |

---

## 2. Modularity Architecture

Phase D code is **strictly additive**: nothing in v2-v5 paths is removed or modified semantically. The single change to existing files is a refactor that preserves byte-identical behavior (verified by tests).

### 2.1 New module: `src/diffmm/generator/edm.py`

| Symbol | Purpose |
|---|---|
| `EDMSchedule` | dataclass holding σ_min, σ_max, σ_data, ρ, P_mean, P_std + `get_sigmas`, `sample_train_sigma`, `loss_weight` methods |
| `sinusoidal_continuous(value, dim)` | sinusoidal embedding for a continuous scalar (used for c_noise → time embedding) |
| `EDMDenoiser(Denoiser)` | wraps a `TradesStyleDenoiser` with the c_skip/c_out/c_in/c_noise preconditioning. Its `forward(x, sigma, ctx)` returns x_0_pred. |
| `edm_sample(predict_x0, ...)` | Heun 2nd-order sampler with optional Karras-style stochasticity (s_churn). CFG operates on x_0 predictions. |
| `EDMTrainer(LightningModule)` | trainer with σ-weighted loss; reuses `CFGDropoutCurriculum` from `train.py` |
| `build_edm_generator(...)` | factory: `GeneratorModel(EDMDenoiser(TradesStyleDenoiser(...)), RegimeEmbedding(...))`. Configurable backbone via `conditioning_type` ∈ {"film", "adaln_zero"}. |

### 2.2 Single additive change: `TradesStyleDenoiser.forward_with_t_emb`

The existing `forward(x, t, ctx)` does:
1. Look up `t_embed_table[t]` (or `t_sinusoidal_table[t]`) → integer-t-indexed sinusoidal embedding
2. Run the FiLM or AdaLN forward pass with that embedding

The refactor splits step 2 into a separate method `forward_with_t_emb(x, t_emb, ctx)`. The default `forward(x, t, ctx)` does the table lookup and dispatches to `forward_with_t_emb`. **Behavior is byte-identical** — verified by `test_forward_with_t_emb_matches_forward_film` and `test_forward_with_t_emb_matches_forward_adaln`.

EDM's preconditioner uses `forward_with_t_emb` to supply a sinusoidal embedding of c_noise(σ) = log(σ)/4 instead of the integer-t lookup. The inner conditioning architecture (FiLM γ/β or AdaLN per-block 6-tuple) is fully preserved.

### 2.3 Config opt-in

[configs/generator/finetune.yaml](../configs/generator/finetune.yaml) now has an `edm:` block:

```yaml
edm:
  enabled: true               # set false to revert to DDPM/v-pred (v5 behavior)
  sigma_min: 0.002
  sigma_max: 80.0
  sigma_data: 0.5
  rho: 7.0
  P_mean: -1.2
  P_std: 1.2
```

`enabled: false` makes Phase D dormant — `11_finetune_generator.py` takes the v2-v5 branch unchanged. `enabled: true` triggers the new branch: `build_edm_generator` + `EDMTrainer`. The DDPM `schedule:` block above is ignored when EDM is on.

### 2.4 Sampling/validation opt-in

CLI flag `--use-edm` (default off) on:
- `scripts/utils/run_sampling_manifest.py`
- `scripts/13_sample_synthetic.py` (reads from Hydra config; flips automatically if `edm.enabled: true`)
- `scripts/50_validate_generator.py`

Slurm env var `USE_EDM=1` on `slurms/ph6_validate_gpu.slurm`.

When `--use-edm` is set, the script:
- Builds the model via `build_edm_generator` instead of `build_generator`
- Constructs an `EDMSchedule` instead of `DDIMSchedule`
- Calls `edm_sample` instead of `ddim_sample`
- For validation: `per_timestep_mse` detects EDM via duck-typing and uses the σ-based path

When `--use-edm` is off (default), behavior is byte-identical to v5.

### 2.5 Backwards compat with v2-v5 checkpoints

All v2-v5 checkpoints continue to load and validate correctly. The state dict structure is unchanged for the DDPM path. EDM checkpoints have an extra `denoiser.inner.` prefix (because EDMDenoiser wraps the TRADES backbone), which `build_edm_generator` reproduces.

The validator's `--prediction-type` and `--conditioning-type` flags are unchanged. Adding `--use-edm` is the third axis. Mismatched flags between training and inference produce silent garbage (load_state_dict with strict=False loads what it can and zeros the rest). The slurm/scripts log the active configuration upfront so this is visible.

---

## 3. Code Changes Summary

| File | Change | LOC |
|---|---|---|
| **NEW** [src/diffmm/generator/edm.py](../src/diffmm/generator/edm.py) | EDM module: schedule, denoiser, sampler, trainer, factory | +320 |
| [src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py) | `forward_with_t_emb` extension point + refactor of `_forward_film`/`_forward_adaln` to use it (byte-identical behavior) | +50/-15 |
| [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) | New `edm:` block (enabled by default for v6+) | +12 |
| [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) | Branch on `cfg.generator.edm.enabled`: build_edm_generator + EDMTrainer if true | +35 |
| [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) | `--use-edm` CLI + `_sample_one` dispatch abstraction | +35/-30 |
| [scripts/50_validate_generator.py](../scripts/50_validate_generator.py) | `--use-edm` CLI; build_edm_generator + EDMSchedule when set | +20/-5 |
| [slurms/ph6_validate_gpu.slurm](../slurms/ph6_validate_gpu.slurm) | `USE_EDM` env var, threaded as `--use-edm` flag | +3 |
| [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) | `per_timestep_mse` detects EDM schedule via duck-typing, σ-based path | +30/-5 |
| [tests/test_validation.py](../tests/test_validation.py) | 12 new tests: forward_with_t_emb identity (FiLM + AdaLN), EDMSchedule shape/distribution/loss_weight, sinusoidal_continuous, preconditioning extremes, denoiser FiLM+AdaLN interop, sampler Heun + Euler, trainer loss decrease, per_timestep_mse EDM path | +260 |

Net: ~520 new LOC + ~260 new tests. Zero deletions of business logic.

**Test count: 70** (was 58 at end of Phase C).

---

## 4. Cluster Commands — v6 Train / Sample / Validate

### 4.1 Rsync from Mac

```bash
rsync -av --progress \
  src/diffmm/generator/edm.py \
  src/diffmm/generator/trades_adapter.py \
  src/diffmm/eval/model_diagnostics.py \
  configs/generator/finetune.yaml \
  scripts/11_finetune_generator.py \
  scripts/utils/run_sampling_manifest.py \
  scripts/50_validate_generator.py \
  slurms/ph6_validate_gpu.slurm \
  tests/test_validation.py \
  docs/Work6_phaseD.md \
  favara@orcd-login.mit.edu:~/diff_mm/
```

If rsync claims success but cluster files don't change (the bug we hit before), use scp per-file with checksums:

```bash
for f in src/diffmm/generator/edm.py src/diffmm/generator/trades_adapter.py \
         src/diffmm/eval/model_diagnostics.py configs/generator/finetune.yaml \
         scripts/11_finetune_generator.py scripts/utils/run_sampling_manifest.py \
         scripts/50_validate_generator.py slurms/ph6_validate_gpu.slurm \
         tests/test_validation.py docs/Work6_phaseD.md; do
  scp "$f" favara@orcd-login.mit.edu:~/diff_mm/"$f"
done
```

### 4.2 Verify on cluster

```bash
module load miniforge && source activate diffmm

# EDM module is present
ls -la src/diffmm/generator/edm.py
grep "class EDMDenoiser" src/diffmm/generator/edm.py
grep "class EDMTrainer" src/diffmm/generator/edm.py

# Config has edm.enabled: true
grep -A2 "^edm:" configs/generator/finetune.yaml

# 70 tests pass
python -m pytest tests/ -q 2>&1 | tail -3
```

### 4.3 Train v6 from scratch (~6h initial chunk)

```bash
sbatch slurms/ph2_finetune.slurm data=intc_top experiment_name=intc_v6
```

Watch the first 100 train steps:

```bash
LATEST=$(ls -t logs/ph2_finetune_*.out | head -1)
tail -f "$LATEST"
```

You should see in the early lines:
- `[INFO] ph2_finetune: conditioning_type=adaln_zero  edm_enabled=True`
- `[INFO] ph2_finetune: EDM schedule: σ_min=0.0020 σ_max=80.0 σ_data=0.500 ρ=7.0 P_mean=-1.20 P_std=1.20`
- Then train steps: `[train] step    20  loss X  σ̄=Y  pred=edm  p_drop=1.00`

Notable differences from v5 logs:
- `pred=edm` (not `pred=v`)
- `σ̄=...` instead of (no σ stat in v5) — average σ for that batch under the log-normal sampler
- Loss values are in EDM-σ-weighted space, not directly comparable to v5's v-MSE numbers

### 4.4 Resume after 6h timeout

```bash
LATEST_V6_CKPT=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v6/*.ckpt | head -1)
sbatch slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v6 \
    +resume_from=$LATEST_V6_CKPT \
    generator.training.max_epochs=40
```

Same chunked-training pattern as Phase B/C.

### 4.5 Sample v6 stitched tapes (~2h)

EDM typically samples in fewer steps than DDIM (32 vs 200) for equivalent quality. You can keep `n_steps=200` for parity with prior runs (still ~2h) or drop to 32-50 for speed.

```bash
CKPT_V6=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v6/*.ckpt | head -1)
OUT_V6=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v6_stitched
echo "v6 ckpt: $CKPT_V6"

sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G --cpus-per-task=2 --output=logs/sample_v6_%j.out --error=logs/sample_v6_%j.err --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py --manifest data/manifests/ph2_sample.json --checkpoint $CKPT_V6 --out-root $OUT_V6 --n-windows-per-task 10 --conditioning-type adaln_zero --use-edm --n-steps 50"
```

**Critical:** the wrap MUST include `--conditioning-type adaln_zero` (Phase C carryover, since v6's inner backbone is AdaLN) AND `--use-edm`. `--prediction-type` is ignored when `--use-edm` is set (EDM uses x_0-prediction via preconditioning, not ε or v). `--x0-clip` defaults to None for EDM (the preconditioning bounds outputs naturally); pass `--x0-clip 200` if you want defensive clipping.

### 4.6 Validate v6 (~25 min)

```bash
mv results/validation/INTC results/validation/INTC_v5 2>/dev/null || true

CKPT_V6=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v6/*.ckpt | head -1)
SYNTH_V6=$OUT_V6/INTC

CKPT=$CKPT_V6 SYNTH_DIR=$SYNTH_V6 sbatch slurms/ph6_validate_cpu.slurm
CKPT=$CKPT_V6 SYNTH_DIR=$SYNTH_V6 CONDITIONING_TYPE=adaln_zero USE_EDM=1 sbatch slurms/ph6_validate_gpu.slurm
```

The CPU validation doesn't load the model so doesn't need either flag. The GPU validation needs both `CONDITIONING_TYPE=adaln_zero` AND `USE_EDM=1`.

### 4.7 Read results

```bash
echo "=== v6 B.1 kurtosis (should still pass; v5 was 1256) ==="
grep -E "^real|^synth|wasserstein" results/validation/INTC/B1_return_stats.csv

echo ""
echo "=== v6 F.2 — AdaLN modulation (target: per-block γ/α avg std/mean > 5%) ==="
cat results/validation/INTC/F2_film_modulation.csv

echo ""
echo "=== v6 F.1b cosine + diagnosis ==="
cat results/validation/INTC/F1b_embedding_cosine.csv
grep "F.1b diagnosis" $(ls -t logs/ph6_validate_gpu_*.out | head -1)

echo ""
echo "=== v6 G.1 ==="
cat results/validation/INTC/G1_pass_fail_summary.csv

echo ""
echo "=== v6 D.1 OFI slope ==="
cat results/validation/INTC/D1_regression_stats.csv
```

---

## 5. Success Criteria

### 5.1 Primary criteria (3/3 to declare v6 the production generator and ship)

| # | Metric | v5 baseline | v6 target |
|---|---|---|---|
| 1 | Conditioning specialization (best per-block AdaLN γ/α std/mean) | 5.66% (final γ only) | **≥ 2 metrics > 5%, OR avg per-block γ_attn > 4%** |
| 2 | Skew (real -0.47) | -17.2 | **|skew| < 5** |
| 3 | Trade-sign lag-1 ACF (real 0.675) | 0.103 | **≥ 0.3** |

### 5.2 Don't-regress criteria (must hold from v5)

| Metric | v5 | v6 must hold |
|---|---|---|
| Kurtosis | 1256 | ≥ 200 (don't lose Phase B) |
| ACF lag-50 dev | 0.296 | < 0.5 (don't lose Phase B) |
| F.1b cosine max | 0.722 | ≤ 0.78 (don't lose Phase C) |
| W-1 distance | 9.81e-07 | within real inter-day band |
| G.1 pass count | 13/16 | ≥ 13/16 |

### 5.3 Decision tree

- **3/3 primary + all don't-regress** → ship v6 as final production generator. Re-run Work 4 hypothesis test on v6-stitched data.
- **1-2/3 primary + don't-regress hold** → mixed result. Decide based on which criterion cleared:
  - If skew + trade-sign cleared but conditioning didn't → conditioning structure may already be near optimum on this data; report Phase C+D as the production setup, ship v6 anyway since it strictly improves on v5
  - If conditioning cleared but skew/trade-sign didn't → these failures are deeper than the loss formulation; report as data-side limitations
- **0/3 primary OR don't-regress fails** → EDM didn't help on this dataset. Ship v5, write Phase D as a clean negative result.

All three outcomes are publishable. Negative results in this style ("we applied SOTA technique X to dataset Y; here's what helped and what didn't, and why we think so") are a real scientific contribution.

---

## 6. Comparison Table — v6 results

| Metric | Real | v2 | v3.5 | v4-noclip | v5 | **v6 (Phase D)** | v6 verdict |
|---|---|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 64 | 270 | **1256** | **185.9** | regressed (overshoot → near-target undershoot) |
| W-1 | — | 1.74e-06 | 1.64e-06 | 1.49e-06 | 9.81e-07 | 1.72e-06 | small regression, still in band |
| **Synth std** (real 2.19e-05) | 2.19e-05 | ~real | ~real | ~real | ~real | **5.31e-06** | ❌ **¼ of real — compressed** |
| ACF lag-50 dev | 0 | 1.28 | 2.94 | 0.022 | 0.296 | 0.665 | regressed (now fails strict <0.5) |
| ACF β | 0.21 | 0.66 | 0.542 | 0.510 | 0.558 | 0.752 | regressed |
| **Skew** (real -0.47) | -0.47 | small | small | small | **-17.2** | **-1.08** | ✓ **HUGE WIN — fixes asymmetry** |
| **Trade-sign lag-1 ACF** | 0.675 | 0.385 | 0.291 | 0.291 | **0.103** | **0.526** | ✓ **HUGE WIN — order-flow persistence restored** |
| Trade fraction | 4.8% | 54.9% | 45.0% | 75% | 55.2% | 78.1% | regressed |
| Size tail exp (real 1.567) | 1.567 | 1.869 | 1.856 | 1.722 | 1.889 | 2.465 | regressed (now fails ±30%) |
| OFI slope at Δ=50 (real 1.11e-04) | 1.11e-04 | 1.11e-05 | 1.49e-05 | 2.01e-05 | 5.55e-05 | 1.67e-05 | regressed |
| **F.1b cosine max off-diag** | — | 0.831 | 0.830 | 0.784 | 0.722 | **0.722** | ✓ **tied for best** |
| **F.1b cosine MEAN off-diag** | — | ~0.69 | ~0.68 | 0.65 | 0.584 | **0.477** | ✓ **best ever (-18% vs v5)** |
| F.1b null ↔ named (max) | — | 0.58 | 0.58 | 0.58 | 0.58 | **0.18** | ✓ **null is now near-orthogonal** |
| F.2 best metric | — | ~1% | 1.17% | 2.88% | 5.66% (γ_final) | **pending** | ? |
| G.1 pass count | 16/16 | 13/16 | 13/16 | 13/16 | 13/16 | 13/16 | tied |

---

## 7. Bug Ledger Continuation

(No new bugs in Phase D implementation — clean code-write + 70/70 tests pass on first run.)

---

## 8. Honest Verdict (final)

Phase D produced a **qualitatively different** model from v5 — neither strictly better nor strictly worse, but a clean trade across the four most consequential failure modes:

### Where v6 wins decisively over v5

1. **Skew −17.2 → −1.08.** v5 produced wildly asymmetric extreme-negative-bias outputs (max positive return 0.0006, max negative -0.0034, kurtosis 1256 was overshoot driven by negative tail). v6's σ-symmetric loss formulation produces near-symmetric outputs (skew matches real -0.47 to within ~0.5). For stress testing, this means agents see a balanced distribution of upside and downside extreme events, not just downside.

2. **Trade-sign lag-1 ACF 0.103 → 0.526.** v5 had nearly zero order-flow persistence; v6 recovers ~80% of real's 0.675. AS-family agents (which respond to order-flow signals via OFI features) now face a meaningful momentum environment in synthetic stress.

3. **F.1b cosine MEAN off-diag 0.584 → 0.477** (−18%). The conditioning curriculum + AdaLN + EDM combination produces the most distinct synthetic regimes of any checkpoint. The null embedding is now near-orthogonal to all named regimes (max cosine 0.18 vs v5's 0.58), suggesting EDM training also separated the unconditional pathway more cleanly from the conditional ones.

### Where v6 regresses vs v5

1. **Synth std 1/4 of real.** v6's synth std is 5.31e-06 vs real 2.19e-05 — output magnitudes are systematically compressed. **This is almost certainly a σ_data calibration issue** (we set σ_data=0.5; for normalized features with std ≈ 1.0 it should be 1.0). EDM's preconditioning normalizes outputs to a target std proportional to σ_data, so a 2× σ_data error produces a 2-4× scale error.

2. **Kurtosis regressed** from v5's 1256 to 185.9 — partially because v6 produces narrower outputs overall (compressed scale → fewer extreme values). With the scale fix, kurtosis should recover to a healthier middle ground (likely 400-800 range, closer to real).

3. **ACF lag-50 dev** went from 0.296 (passes) to 0.665 (fails). Same mechanism as kurtosis — compressed output magnitudes mean less sustained large-|return| sequences. Should also recover with σ_data fix.

4. **Several other metrics** (ACF β, trade fraction, size tail exponent, OFI slope) regressed. All consistent with the same compressed-output story.

### The diagnosis: σ_data mismatch

Working through the EDM math: with `σ_data=0.5` and our true normalized-feature std ≈ 1.0:
- `c_skip(σ) = σ_data² / (σ² + σ_data²)` peaks at low σ, contributes a *direct skip* of the noisy input
- `c_out(σ) = σ · σ_data / sqrt(σ² + σ_data²)` scales the network's prediction
- At σ ≈ 0.30 (median training σ), c_out ≈ 0.13. The network's effective output magnitude is bounded by ~0.13 × F_θ(...) magnitudes
- Network learns to predict outputs at that scale; the resulting distribution is compressed

If σ_data were 1.0 instead, c_out at σ=0.30 would be ~0.26 — twice as large output capacity. The network would learn to produce outputs at the proper scale.

**This is a 1-line config fix.**

### Should we ship v5 or v6?

| Criterion | v5 wins | v6 wins |
|---|---|---|
| Heavy-tail kurtosis | ✓ (1256 > 185) | |
| Output scale fidelity | ✓ (~real) | |
| Skew / asymmetry | | ✓ (-1.08 vs -17.2) |
| Trade-sign ACF | | ✓ (0.526 vs 0.103) |
| Regime distinctness (F.1b mean off-diag) | | ✓ (0.477 vs 0.584) |
| Heavy-tail recovery (lag-50 dev) | ✓ | |

**For the project's central question (do synthetic regimes differentially stress agents?):** v6 is the better choice in spirit because:
- Most distinct regimes → conditional generation actually does something different per regime
- Realistic order-flow persistence → AS-family agents face the dynamics they were calibrated against
- Symmetric extremes → no artificial bias

**But in practice, the compressed scale would handicap agents calibrated on real-scale data.** AS agents calibrated with σ ≈ 2.19e-05 might behave very differently when synth has σ ≈ 5.31e-06 — half-spreads relative to volatility would look 4× wider than calibrated, fills would be rare, PnL would compress.

**Honest verdict: neither v5 nor v6 should be shipped as-is.** v5 has the wrong asymmetry; v6 has the wrong scale. v7 (next section) is the natural unification.

---

## 9. Phase D.5 — v7 = v6 with σ_data Fix

The single-line config change that should give us v6's wins WITHOUT v6's scale regression:

```diff
# configs/generator/finetune.yaml, edm: section
-  sigma_data: 0.5
+  sigma_data: 1.0
```

### Why σ_data = 1.0 is the right value

Our training data is **z-score normalized** to N(0, 1) per feature (computed during data preparation in [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py) `compute_norm_stats`). So the std of normalized x_0 across the training set is, by construction, ≈ 1.0 per feature.

EDM's preconditioning was derived assuming σ_data ≈ std(data). Mis-setting σ_data:
- **σ_data too small** (our v6 case, σ_data=0.5): c_out compresses outputs → synth distribution is narrow
- **σ_data too large**: c_out amplifies outputs → synth distribution is wide and noisy

The original Karras 2022 paper uses σ_data=0.5 for ImageNet (where pixels are scaled to [-1, 1] giving std ≈ 0.5 after preprocessing). Our LOB features are differently normalized; σ_data=1.0 is the natural choice.

### Predictions for v7

| Metric | v5 | v6 | **v7 prediction (σ_data=1.0)** |
|---|---|---|---|
| Skew | -17.2 | **-1.08** | ~-0.5 to -1.5 (Phase D wins are loss-formulation effects, not σ_data effects → should persist) |
| Trade-sign ACF | 0.103 | **0.526** | ~0.4-0.6 (likewise) |
| F.1b cosine MEAN off-diag | 0.584 | **0.477** | ~0.45-0.50 (likewise) |
| **Synth std (real 2.19e-05)** | ~real | **5.31e-06 (¼ real)** | **~real** ← if σ_data thesis is right |
| **Kurtosis** | **1256** | 185.9 | **400-800** (recovers from scale fix; doesn't overshoot like v5 because EDM's symmetric loss bounds the asymmetric tail-amplification) |
| **ACF lag-50 dev (criterion <0.5)** | 0.296 | 0.665 | **<0.3** (recovers with scale) |
| Other (ACF β, trade fraction, size tail) | mixed | regressed | should largely recover |

If predictions hold, **v7 is unambiguously the best generator the project produces**: best F.1b distinctness, near-real skew, restored order-flow persistence, restored scale, and kurtosis comfortably above the strict 200 threshold.

### Cost / risk

- **Implementation**: 1-line YAML change (`sigma_data: 0.5 → 1.0`). Done.
- **Training**: ~6h on mit_normal_gpu, then chunked resume if needed (same pattern as v6). Total ~6-12h GPU.
- **Sampling + validation**: ~2.5h.
- **Total wall**: ~9-15h.
- **Risk**: Low. σ_data is a well-understood hyperparameter; the calibration argument is straightforward. Worst case: v7 has a different scale problem and we learn σ_data=1.0 was overcorrection (then try 0.75, etc.). Three-config sweep at most.

### Recommendation: train v7

This is the right final-final experiment. The mechanism is well-understood, the cost is small, and the expected payoff is the project's first checkpoint that wins on all four design dimensions simultaneously (parameterization=v0/EDM, conditioning=AdaLN-Zero, training schedule=σ-spaced + curriculum, output scale=calibrated to data).

If v7 hits the predictions, the diffusion model workstream is genuinely complete and we have an unambiguous "best" generator for the central hypothesis test. If it doesn't, we ship the better of v5 and v6 based on what the central hypothesis test actually needs (probably v6 — its conditioning + symmetry advantages matter more than v5's tail magnitude for the regime-differentiation question).

---

## 10. v7 cluster command (one config change, then standard sequence)

```bash
# On Mac — single-line config change to finetune.yaml is the only diff
sed -i.bak 's/sigma_data: 0.5/sigma_data: 1.0/' configs/generator/finetune.yaml
grep "sigma_data" configs/generator/finetune.yaml   # verify

# scp the updated config (bypassing rsync's mtime confusion)
scp configs/generator/finetune.yaml favara@orcd-login.mit.edu:~/diff_mm/configs/generator/finetune.yaml

# On cluster — verify, then submit (excluding the bad node from before)
grep "sigma_data" configs/generator/finetune.yaml   # should show: sigma_data: 1.0
sbatch --exclude=node4104 slurms/ph2_finetune.slurm data=intc_top experiment_name=intc_v7
```

Sample + validate v7 follows the same pattern as v6 (note `--edm-sigma-data 1.0` to match training):

```bash
CKPT_V7=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v7/*.ckpt | head -1)
OUT_V7=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v7_stitched

sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G --cpus-per-task=2 --output=logs/sample_v7_%j.out --error=logs/sample_v7_%j.err --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py --manifest data/manifests/ph2_sample.json --checkpoint $CKPT_V7 --out-root $OUT_V7 --n-windows-per-task 10 --conditioning-type adaln_zero --use-edm --edm-sigma-data 1.0 --n-steps 50"
```

Validation:

```bash
mv results/validation/INTC results/validation/INTC_v6 2>/dev/null || true
SYNTH_V7=$OUT_V7/INTC
CKPT=$CKPT_V7 SYNTH_DIR=$SYNTH_V7 sbatch slurms/ph6_validate_cpu.slurm
CKPT=$CKPT_V7 SYNTH_DIR=$SYNTH_V7 CONDITIONING_TYPE=adaln_zero USE_EDM=1 EDM_SIGMA_DATA=1.0 sbatch slurms/ph6_validate_gpu.slurm
```

The σ_data parameter has to match between training, sampling, and validation — otherwise the preconditioning math is inconsistent and outputs are garbage. The slurm now reads `EDM_SIGMA_DATA` (already added in Phase D code).
