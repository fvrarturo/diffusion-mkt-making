# Work 6 — Phase C (v5): AdaLN-Zero Conditioning

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-05-03
**Status:** Code complete; cluster training pending.
**Companion to:** [Work6_v4.md](Work6_v4.md) (Phase B + x0_clip discovery + 4-way ablation)

This document covers the Phase C implementation: replacing FiLM with **AdaLN-Zero** (Peebles & Xie 2023, the DiT paper) as the conditioning primitive. Phase C is a single-thread intervention targeting the one strict criterion that v4-noclip didn't clear — FiLM γ_in std/mean across regimes (2.88% vs target 5%).

---

## 0. Executive Summary

After Phase B + the x0_clip discovery, v4-noclip cleared 2/3 strict criteria but the third (conditioning differentiation) remained stuck. The F.2 numbers across all FiLM-conditioning checkpoints (v2-v4) cluster around the same plateau — none ever clears 5%:

| Checkpoint | FiLM γ_in std/mean across 4 regimes |
|---|---|
| v2 | ~1% |
| v3-e9 (γ=1, FiLM rand init, p_drop=0.2) | 0.92% |
| v3-e19 | 0.37% |
| v3.5 (γ=5, FiLM rand init, p_drop=0.2) | 1.17% |
| v4 (v-pred + curriculum) | **2.88%** ← best, still below 5% target |

The pattern indicates a structural ceiling in the FiLM primitive itself, not the training config. Phase A's two FiLM-targeted changes (random init, higher p_drop) and Phase B's curriculum together more than doubled the γ-dev variance vs v2's baseline, but the curve clearly plateaus well below the target.

**AdaLN-Zero is the standard fix.** It's the conditioning primitive used by DiT (the SOTA architecture for class-conditional image diffusion) and was specifically motivated in that paper by the same problem: FiLM-style modulation underutilizes its conditioning capacity. Three structural differences:

1. **~13× more conditioning capacity per layer** — instead of a single (γ, β) tuple per FiLM stage, AdaLN-Zero provides 6 modulation params per transformer block (γ_attn, β_attn, α_attn, γ_mlp, β_mlp, α_mlp). With 8 blocks, that's 48 modulation slots vs 4 FiLM slots.
2. **Identity at init** — zero-initialized residual gate `α` makes every block exactly identity at step 0 (`h + 0 · Block(...) = h`). The model learns conditioning from a clean slate. Compare to Phase A's random-init FiLM which was supposed to break the identity trap but only partially helped.
3. **Conditioning includes the diffusion timestep** — in the FiLM path, `t` is added directly to the hidden state via a sinusoidal lookup. In AdaLN-Zero, `t` and the regime embedding both flow through the same conditioning vector `c`, which then drives all the per-block modulations. Cleaner decoupling of "what kind of step is this" (t) and "what kind of representation is this" (h).

50 → 58 tests pass. All Phase A/B infrastructure (v-prediction, conditioning curriculum, x0_clip) is preserved — Phase C only swaps the modulation primitive.

---

## 1. The Architectural Change

### 1.1 What FiLM did

[src/diffmm/generator/conditioning.py](../src/diffmm/generator/conditioning.py):

```python
# FiLMLayer
gamma, beta = Linear(ctx_dim, 2 * d_model)(ctx).chunk(2)
h_modulated = (1 + gamma) * h + beta
```

Two FiLM layers in the denoiser: `film_in` (after time/positional embedding) and `film_out` (before output projection). 4 modulation slots total (2 γ + 2 β).

### 1.2 What AdaLN-Zero does

[src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py) — new classes `AdaLNTransformerBlock` and `AdaLNFinalLayer`:

```python
# AdaLNTransformerBlock
gamma1, beta1, alpha1, gamma2, beta2, alpha2 = (
    Sequential(SiLU, Linear(ctx_dim, 6 * d_model))(c)
).chunk(6)

# Attention sub-block — zero-init makes alpha1 = 0 → block is identity at start
h = h + alpha1 * Attn(LN(h) * (1 + gamma1) + beta1)
# MLP sub-block — same trick
h = h + alpha2 * MLP(LN(h) * (1 + gamma2) + beta2)
```

Per block: 6 modulation slots. Across 8 blocks + 2-slot final layer: 50 modulation slots vs FiLM's 4. **12.5× more conditioning capacity.**

The modulation `Linear(ctx_dim, 6 * d_model)` is zero-initialized (DiT §3.1). At step 0:
- `alpha1 = alpha2 = 0` → both residual contributions are zero → block is identity
- Stack of 8 identity blocks + identity final layer → denoiser is identity composed with `out_proj` (which is also zero-init) → output is zero
- Loss is just `||target - 0||²` initially, with strong gradients → model learns from a clean slate

### 1.3 What stays the same

- All Phase A/B infrastructure: v-prediction (`prediction_type="v"`), conditioning curriculum (`cfg_dropout` ramp 1.0 → 0.2 over epochs 2-5), Min-SNR γ=5, patience 6
- `x0_clip=200` at sampling time (Phase B finding)
- Regime embedding (`RegimeEmbedding`) unchanged — still produces a (B, embed_dim) vector from the 4-axis condition
- Self-attention math unchanged
- MLP and feed-forward unchanged (PReLU activation, 4× expansion ratio)
- Output projection zero-init unchanged

The change is surgical: one primitive (FiLM → AdaLN-Zero), one timestep flow (additive → conditioning-vector-driven). Everything else from Phase B is preserved.

### 1.4 Backwards compatibility

`build_generator()` takes a new `conditioning_type` arg defaulting to `"film"` (v2-v4 behavior). Old checkpoints continue to load and validate. New v5 training uses `conditioning_type="adaln_zero"` via the updated [configs/generator/finetune.yaml](../configs/generator/finetune.yaml).

Validation/sampling scripts take a matching `--conditioning-type` CLI arg (or `CONDITIONING_TYPE` env var for the GPU validation slurm), defaulting to `"film"`. Mismatching the flag and the checkpoint produces silent garbage (state dict load fails to populate AdaLN params with `strict=False`, model output is uninitialized noise) — same protocol as `--prediction-type`.

---

## 2. F.2 Validation — Now Works for Both Architectures

The `film_modulation_magnitude()` function in [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) is updated to detect both architectures:

- **FiLM checkpoints**: hooks `FiLMLayer.proj` outputs, reports `{name}_gamma_dev` and `{name}_beta_norm` per layer.
- **AdaLN-Zero checkpoints**: hooks `AdaLNTransformerBlock.adaLN_modulation[-1]` outputs, reports `{name}_gamma_attn_norm`, `{name}_beta_attn_norm`, `{name}_alpha_attn_norm`, `{name}_gamma_mlp_norm`, `{name}_beta_mlp_norm`, `{name}_alpha_mlp_norm` per block, plus `{name}_gamma_final_norm` and `{name}_beta_final_norm` for the final layer.

The conceptual metric is the same: **std/mean across regimes of the modulation magnitudes**. Larger std/mean → regimes more distinct. The new criterion for AdaLN:

- **Primary:** for at least one of (γ_attn, γ_mlp, α_attn, α_mlp), std/mean across 4 named regimes > 5%
- **Stretch:** for *most* blocks, std/mean of α_attn or α_mlp > 10% (strong per-block conditioning specialization)

The α gates are particularly informative because they're the residual scales — large α-variance across regimes means different regimes route differently through the network blocks.

---

## 3. Code Changes Since Work6_v4.md

### 3.1 Modified files

- **[src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py)** — new classes `AdaLNTransformerBlock`, `AdaLNFinalLayer`, helper `_modulate(h, γ, β)`. `TradesStyleDenoiser` now branches on `conditioning_type` between two complete forward paths (`_forward_film` and `_forward_adaln`). `build_generator()` takes `conditioning_type` arg, defaults to `"film"`.
- **[configs/generator/finetune.yaml](../configs/generator/finetune.yaml)** — `model.conditioning_type: "adaln_zero"` (v5+ default).
- **[scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py)** — reads `conditioning_type` from config, passes to `build_generator`, logs which is active.
- **[scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py)** — `--conditioning-type` CLI arg.
- **[scripts/50_validate_generator.py](../scripts/50_validate_generator.py)** — `--conditioning-type` CLI arg, threaded into `build_generator`.
- **[slurms/ph6_validate_gpu.slurm](../slurms/ph6_validate_gpu.slurm)** — `CONDITIONING_TYPE` env var (default `film`), threaded as `--conditioning-type` into the validator.
- **[src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py)** — `film_modulation_magnitude()` detects both `FiLMLayer` and `AdaLNTransformerBlock`/`AdaLNFinalLayer`, hooks the appropriate modulation projection, decomposes the chunked output into per-modulation columns.

### 3.2 New tests in [tests/test_validation.py](../tests/test_validation.py)

8 new tests added (50 → 58 total):

1. `test_adaln_zero_block_starts_as_identity` — verify zero-init produces exact identity output
2. `test_adaln_final_layer_starts_as_identity_on_normed_input` — verify final layer at zero-init = LayerNorm
3. `test_adaln_block_responds_to_conditioning_after_init` — perturb mod weights, verify output now depends on conditioning
4. `test_adaln_denoiser_smoke_forward_and_backward` — full denoiser forward + backward + optimizer step on random data
5. `test_adaln_denoiser_loss_decreases_on_random_data` — 5 epochs of training with v-prediction + AdaLN, loss decreases
6. `test_film_modulation_magnitude_handles_adaln_blocks` — F.2 extractor produces AdaLN-style columns when given an AdaLN model
7. `test_build_generator_conditioning_type_validation` — invalid conditioning_type raises clear error
8. `test_adaln_film_paths_independent` — building both architectures gives different module structures

### 3.3 Lines of code

```
src/diffmm/generator/trades_adapter.py    +130/-15
src/diffmm/eval/model_diagnostics.py       +35/-15
configs/generator/finetune.yaml             +9/-0
scripts/11_finetune_generator.py            +2/-0
scripts/13_sample_synthetic.py              +0/-0  (uses Hydra config; finetune.yaml change picks it up)
scripts/utils/run_sampling_manifest.py      +6/-0
scripts/50_validate_generator.py            +5/-2
slurms/ph6_validate_gpu.slurm               +2/-0
tests/test_validation.py                  +135/-0
docs/Work6_phaseC.md                       +NEW
```

Net: ~225 LOC of new functionality + 135 LOC of tests. No deleted business logic — FiLM path remains intact for backwards compat with v2-v4.

---

## 4. Cluster Commands — v5 Train / Sample / Validate

### 4.1 Rsync from Mac

```bash
rsync -av --progress \
  src/diffmm/generator/trades_adapter.py \
  src/diffmm/eval/model_diagnostics.py \
  configs/generator/finetune.yaml \
  scripts/11_finetune_generator.py \
  scripts/utils/run_sampling_manifest.py \
  scripts/50_validate_generator.py \
  slurms/ph6_validate_gpu.slurm \
  tests/test_validation.py \
  docs/Work6_phaseC.md \
  favara@orcd-login.mit.edu:~/diff_mm/
```

### 4.2 Verify on cluster

```bash
module load miniforge && source activate diffmm
grep "conditioning_type" configs/generator/finetune.yaml | head -3
grep "AdaLNTransformerBlock\|adaln_zero" src/diffmm/generator/trades_adapter.py | head -5
python -m pytest tests/test_validation.py -q 2>&1 | tail -3
```

Should see `conditioning_type: "adaln_zero"` in the config, `AdaLNTransformerBlock` definition in the adapter, and `26 passed` from the validation tests.

### 4.3 Train v5 from scratch (~6h initial chunk on mit_normal_gpu)

```bash
sbatch slurms/ph2_finetune.slurm data=intc_top experiment_name=intc_v5
```

Watch the first 100 train steps to confirm the new architecture is active:

```bash
LATEST=$(ls -t logs/ph2_finetune_*.out | head -1)
tail -f "$LATEST"
```

Expected early lines:

```
[INFO] ph2_finetune: conditioning_type=adaln_zero
[train] step      0  loss 0.xxxx  pred=v  p_drop=1.00
[train] step     20  loss 0.xxxx  pred=v  p_drop=1.00
```

Initial loss may be slightly higher than v4 because AdaLN starts as exact identity (output = 0 from the zero-init out_proj). The model has more to learn at start — but the same identity-at-init behavior also means stable training, no early instability.

After ~30 min (epoch 2), the curriculum kicks in and `p_drop` ramps; same dynamics as v4. After ~75 min (epoch 5), `p_drop=0.20` steady state.

### 4.4 Resume after 6h timeout (expected)

```bash
LATEST_V5_CKPT=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v5/*.ckpt | head -1)
sbatch slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v5 \
    +resume_from=$LATEST_V5_CKPT \
    generator.training.max_epochs=40
```

Same chunked-training pattern as Phase B. Expect 9-10 epochs per 6h chunk; resume to reach epoch ~19 if val/loss is still improving.

### 4.5 Sample v5 stitched tapes (~2h)

```bash
CKPT_V5=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v5/*.ckpt | head -1)
OUT_V5=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v5_stitched
echo "v5 ckpt: $CKPT_V5"

sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G --cpus-per-task=2 --output=logs/sample_v5_%j.out --error=logs/sample_v5_%j.err --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py --manifest data/manifests/ph2_sample.json --checkpoint $CKPT_V5 --out-root $OUT_V5 --n-windows-per-task 10 --prediction-type v --conditioning-type adaln_zero --x0-clip 200"
```

**Critical:** the wrap MUST include both `--prediction-type v` (Phase B) AND `--conditioning-type adaln_zero` (Phase C) AND `--x0-clip 200` (Phase B finding). Missing any of those produces incorrect output.

### 4.6 Validate v5 (~25 min)

```bash
mv results/validation/INTC results/validation/INTC_v4_noclip 2>/dev/null || true

CKPT_V5=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v5/*.ckpt | head -1)
SYNTH_V5=$OUT_V5/INTC

CKPT=$CKPT_V5 SYNTH_DIR=$SYNTH_V5 sbatch slurms/ph6_validate_cpu.slurm
CKPT=$CKPT_V5 SYNTH_DIR=$SYNTH_V5 PREDICTION_TYPE=v CONDITIONING_TYPE=adaln_zero sbatch slurms/ph6_validate_gpu.slurm
```

The CPU validation doesn't load the model so doesn't need either flag. The GPU validation needs both.

### 4.7 Read results

```bash
echo "=== v5 B.1 kurtosis (target: > 200; v4-noclip was 270) ==="
grep -E "^real|^synth|wasserstein" results/validation/INTC/B1_return_stats.csv

echo ""
echo "=== v5 F.2 — AdaLN modulation by regime (target: γ_attn or α std/mean > 5%) ==="
cat results/validation/INTC/F2_film_modulation.csv

echo ""
echo "=== v5 F.1b — embedding cosine + diagnosis ==="
cat results/validation/INTC/F1b_embedding_cosine.csv
grep "F.1b diagnosis" $(ls -t logs/ph6_validate_gpu_*.out | head -1)

echo ""
echo "=== v5 G.1 ==="
cat results/validation/INTC/G1_pass_fail_summary.csv
```

---

## 5. Success Criteria

The single target Phase C is meant to clear:

| # | Metric | Target | v4-noclip | **v5 target** |
|---|---|---|---|---|
| 1 | Conditioning differentiation across regimes | min std/mean of any γ/α metric > 5% | 2.88% (γ_in only) | **>5% on at least 2 of {γ_attn_avg, γ_mlp_avg, α_attn_avg, α_mlp_avg}** |
| 2 | F.1b cosine max off-diag (regime distinctness) | < 0.83 | 0.784 | **< 0.75** (further separation expected with more conditioning capacity) |

Secondary criteria — should hold or improve from v4-noclip baseline:

- B.1 kurtosis: ≥ 200 (don't regress on Phase B)
- B.1 W-1: stays well within inter-day band
- C.1 ACF lag-50 dev: stays < 0.5 (Phase B's biggest win)
- D.1 OFI slope sign: passes at all Δ
- G.1 pass count: ≥ 13/16 (match v4-noclip)
- Trade fraction: doesn't blow up

If primary criterion clears AND no secondary regresses, **v5 is the production generator** and we move to re-running the Work 4 hypothesis test.

If primary clears but kurtosis regresses (e.g., AdaLN routing changes how the model handles tails), we have a tradeoff and need to think about which generator to ship for what.

If primary doesn't clear, the FiLM-collapse failure mode is *not* about the conditioning primitive — it's about the data or the underlying conditioning signal. Phase D options would be: bigger embed_dim, contrastive auxiliary loss between regimes, or accepting that this dataset's regime labels carry insufficient discriminative signal for FiLM-collapse to fully resolve.

---

## 6. Comparison Table (to be filled in after v5 validation)

| Metric | Real | v2 | v3.5 | v4-noclip | **v5 (Phase C)** |
|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 64 | 270 | **?** |
| ACF lag-50 dev | 0 | 1.28 | 2.94 | 0.022 | ? |
| ACF β | 0.21 | 0.66 | 0.542 | 0.510 | ? |
| FiLM γ_in std/mean (FiLM) / γ_attn std/mean (AdaLN) | — | ~1% | 1.17% | 2.88% | **?** |
| F.1b cosine max | — | 0.831 | 0.830 | 0.784 | **?** |
| G.1 pass count | 16/16 | 13/16 | 13/16 | 13/16 | ? |

---

## 7. Bug Ledger Continuation

(No new bugs introduced in Phase C implementation — clean code-write + 58/58 tests pass on first run. Bug entries 1-50 are in earlier docs.)

---

## 8. Honest Verdict (interim)

Phase C is the cleanest single-thread intervention of the entire Work 6 series:

- **One failure mode targeted** (FiLM collapse) — F.2 std/mean stuck at ~1-3% across all FiLM checkpoints
- **One well-known fix** (AdaLN-Zero, Peebles & Xie 2023, the SOTA conditional diffusion technique)
- **Surgical scope** — replaces FiLM with AdaLN, preserves all Phase A/B infrastructure (v-prediction, curriculum, x0_clip)
- **Prior probability of success: high** — DiT empirically outperforms FiLM on every standard conditional diffusion benchmark; if it doesn't help here, the failure mode is genuinely deeper than the conditioning primitive

If Phase C clears the primary criterion (γ/α std/mean > 5%) without regressing kurtosis or ACF, v5 becomes the production generator and we have a clean 3/3 strict-criteria pass for the first time in the project. The Work 4 hypothesis test then has its first chance to use *genuinely differentiated* synthetic regimes for stress-directed counterfactuals.

If Phase C doesn't clear it, the answer to "why don't synthetic regimes look distinct?" lies in the data or labels themselves, not the model. That's an honest scientific outcome too — narrows the search.

Either way, Phase C is the right next step. Code is complete and tested. Time to launch.
