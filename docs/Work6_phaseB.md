# Work 6 — Phase B (v4): v-parameterization + Conditioning Curriculum

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-30
**Status:** Code complete; cluster training pending.
**Companion to:** [Work6_update.md](Work6_update.md) (Phase A.5 falsification + Phase B scoping)

This document covers the Phase B implementation. Two surgical changes targeting two distinct failure modes identified in [Work6_update.md §5](Work6_update.md):

1. **v-parameterization** (Salimans & Ho 2022) — predict `v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0` instead of `ε`. Targets the heavy-tail under-fitting (B.1 kurtosis 81-122 across v2-v3.5 vs real 755).
2. **Conditioning curriculum** — start fully unconditional for 2 epochs, ramp `cfg_dropout` 1.0 → 0.2 over epochs 2-5. Targets the FiLM-layer collapse (F.2 γ-dev std/mean ~1% across all four prior runs).

Items deferred from the original Phase B scope:
- ~~`window_length` 256 → 512~~ — too disruptive (data pipeline rerun + retrain from scratch on doubled-context windows). Defer to a separate workstream if Phase B alone doesn't close the long-memory gap.
- ~~Auxiliary ACF + kurtosis loss~~ — high complexity (decoding inside the training loop), uncertain payoff. v-parameterization alone has good prior probability of fixing tails; revisit only if it doesn't.

---

## 0. Executive Summary

Phase B implementation is complete and tested. **50/50 tests pass** (was 44/44 at end of Phase A.5; +6 new tests covering v-prediction round-trip, ddim_sample with v, curriculum ramp, and trainer integration).

| Change | File | Risk | Backwards-compatibility |
|---|---|---|---|
| `prediction_type` field on `DDIMSchedule` (default `"eps"`) | [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py) | Low | Default keeps v2/v3/v3.5 sampling identical |
| `ddim_sample` interprets model output as v when `prediction_type="v"` | [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py) | Low | Behind the schedule's prediction_type flag |
| `DDPMTrainer.prediction_type` + Min-SNR weight adjustment for v | [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) | Medium | Defaults to schedule.prediction_type → "eps" if unset |
| `CFGDropoutCurriculum` dataclass + epoch-based ramp | [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) | Low | Disabled by default (`curriculum=None`) |
| `prediction_type` and `cfg_dropout_curriculum` in finetune.yaml | [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) | Low | Set to `"v"` and `enabled: true` for v4+ |
| `per_timestep_mse` (validation Part E.2) builds correct target | [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) | Low | Reads from `schedule.prediction_type` |
| `--prediction-type` CLI arg in sampling + validation scripts | [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py), [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py), [scripts/50_validate_generator.py](../scripts/50_validate_generator.py), [slurms/ph6_validate_gpu.slurm](../slurms/ph6_validate_gpu.slurm) | Low | Default `"eps"` for backwards compat |

**Old checkpoints (v2 / v3 / v3.5) continue to work** as long as the user passes `--prediction-type eps` (the default). New v4 checkpoints require `--prediction-type v`. There is no automatic migration — passing the wrong flag silently produces garbage at sample time. The slurms and scripts log the active `prediction_type` upfront so this is visible in SLURM logs.

---

## 1. The Two Changes — Detailed Rationale

### 1.1 v-parameterization (Salimans & Ho 2022)

**The math:**
```
Forward:   x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
v:         v   = √ᾱ_t · ε   − √(1−ᾱ_t) · x_0       (definition)
Reverse:   x_0 = √ᾱ_t · x_t − √(1−ᾱ_t) · v
           ε   = √(1−ᾱ_t) · x_t + √ᾱ_t · v
```

`v` interpolates between `ε` (at high `t`, where √ᾱ → 0) and `−x_0` (at low `t`, where √(1−ᾱ) → 0). The training target is the model's prediction of `v`, regressed against the true `v` computed from the (x_0, ε) sample.

**Why this should help heavy tails:**

ε-prediction's failure mode (per [Work6_update.md §5.3](Work6_update.md)) is that with enough training, MSE on noise prediction converges to mode-covering: the model averages out the rare extreme values because they contribute little to total MSE. The bulk fits well, the tails compress.

v-prediction breaks this in two ways:
1. **Signal-correlated target at low `t`.** When `t` is small, the target `v ≈ −x_0`, i.e., the model is being asked to predict the (negated) clean data directly. Heavy tails appear *in the target*, so the model can't average them out without paying a direct loss. This is the "fine-detail commit" the model needs to make for tails.
2. **Uniform loss-vs-SNR profile.** ε-prediction loss naturally collapses at low t (because ε is "easy" to predict when most of x_t is x_0). v-prediction's per-`t` loss is closer to uniform, so the model spends more training cycles on the timesteps where heavy-tail recovery happens.

**Min-SNR weighting under v-prediction:** The standard Hang et al. 2023 formula for ε-prediction is `weight = min(SNR, γ) / SNR`. For v-prediction, the per-sample MSE in v-space is `(SNR + 1)×` the underlying x_0-MSE (Salimans & Ho 2022 §3.2), so to match ε-prediction's effective weighting on the underlying x_0 quantity, the formula becomes `weight = min(SNR, γ) / (SNR + 1)`. Implementation: [train.py:_step](../src/diffmm/generator/train.py).

### 1.2 Conditioning curriculum

**The mechanism:**
- Epochs 0..1: `cfg_dropout = 1.0` (every batch is unconditional — model sees only the null condition).
- Epochs 2..5: linear ramp from 1.0 → 0.2.
- Epoch 5+: steady state at 0.2 (matches Phase A's CFG dropout setting).

**Why this should help FiLM collapse:**

Under v3/v3.5, the model trained simultaneously on conditional + unconditional from step 0 with `cfg_dropout = 0.10` or `0.20`. The optimizer found a shortcut: make FiLM nearly identity-ish, so the conditional and unconditional paths produce similar outputs. This satisfies both losses simultaneously without requiring FiLM to specialize per-regime. Result: F.2 γ-dev std/mean ~1% across all four checkpoints — FiLM modulates uniformly regardless of `c`.

The curriculum forces a temporal ordering: epochs 0-1 establish a strong unconditional baseline (no shortcut available — there's no conditional path to "match"). Then conditioning is introduced gradually, and FiLM's job becomes "what should I add to the established unconditional baseline for each regime?" rather than "find a pair of paths that both work." Standard technique from controllable-diffusion lit (e.g. Karras 2024).

**Risk:** The unconditional-only phase could produce a generator that's *too* committed to the unconditional pathway, leaving no headroom for conditioning. We mitigate by (a) keeping the ramp short (2 → 5 epochs is 3 ramp epochs out of ~20 total) and (b) keeping the steady-state `cfg_dropout = 0.20` (so the conditional path still gets 80% of training pressure after ramp).

---

## 2. Code Changes Summary

```
configs/generator/finetune.yaml                           +24/-6
scripts/11_finetune_generator.py                          +20/-2
scripts/13_sample_synthetic.py                             +5/-2
scripts/50_validate_generator.py                           +6/-1
scripts/utils/run_sampling_manifest.py                     +6/-1
slurms/ph6_validate_gpu.slurm                              +5/-1
src/diffmm/eval/model_diagnostics.py                      +14/-7
src/diffmm/generator/sample.py                            +30/-9
src/diffmm/generator/train.py                             +73/-12
tests/test_validation.py                                 +130/-0
docs/Work6_phaseB.md                                      +NEW
```

Net: ~280 LOC added, mostly new functionality + 130 lines of tests. No deleted business logic.

---

## 3. Tests

50/50 tests pass. New tests added (in [tests/test_validation.py](../tests/test_validation.py)):

- `test_v_parameterization_round_trip` — algebraic identity x_0 ↔ v ↔ ε holds
- `test_ddim_sample_with_v_prediction_runs_and_produces_finite_output` — sampler doesn't NaN under v-prediction
- `test_ddpm_trainer_v_prediction_loss_decreases` — v-prediction trains successfully on synthetic data
- `test_cfg_dropout_curriculum_ramps_correctly` — linear ramp math is correct at boundaries and midpoints
- `test_trainer_uses_curriculum_dropout_when_enabled` — constructor's `cfg_dropout` arg is correctly overridden when curriculum is active
- `test_schedule_prediction_type_defaults_and_round_trips` — `DDIMSchedule.cosine`/`linear` factories preserve prediction_type

Pre-existing tests still pass — the changes are backwards-compatible at the API surface (new args are keyword-only with sensible defaults).

---

## 4. Cluster Commands — v4 Train / Sample / Validate

These are designed to run sequentially. Each is a single tight command — no multi-line backslashes (they get mangled by terminal paste).

### 4.1 Rsync the Phase B code from Mac to cluster

```bash
rsync -av --progress \
  src/diffmm/generator/sample.py \
  src/diffmm/generator/train.py \
  src/diffmm/eval/model_diagnostics.py \
  scripts/11_finetune_generator.py \
  scripts/13_sample_synthetic.py \
  scripts/50_validate_generator.py \
  scripts/utils/run_sampling_manifest.py \
  slurms/ph6_validate_gpu.slurm \
  configs/generator/finetune.yaml \
  tests/test_validation.py \
  docs/Work6_phaseB.md \
  favara@orcd-login.mit.edu:~/diff_mm/
```

(`--relative` flag is intentionally omitted so files land at the right paths via the source dir layout.)

### 4.2 Verify env active and code is current

```bash
module load miniforge && source activate diffmm
grep prediction_type configs/generator/finetune.yaml | head -3
grep "prediction_type" src/diffmm/generator/sample.py | head -3
python -m pytest tests/test_validation.py -q 2>&1 | tail -3
```

Should see `prediction_type: "v"` in the config, `prediction_type` field in the schedule dataclass, and `18 passed` from the validation tests.

### 4.3 Train v4 from scratch (~6h on mit_normal_gpu, can chunk)

```bash
sbatch slurms/ph2_finetune.slurm data=intc_top experiment_name=intc_v4
```

The slurm picks up `prediction_type: "v"` and the `cfg_dropout_curriculum` from the updated `finetune.yaml`. Watch the first 100 train steps to confirm:

```bash
LATEST=$(ls -t logs/ph2_finetune_*.out | head -1)
tail -f "$LATEST"
```

You should see lines like:
```
[train] step    20  loss 0.0xxx  pred=v  p_drop=1.00
[train] step    40  loss 0.0xxx  pred=v  p_drop=1.00
```

The `pred=v` confirms v-prediction is active. The `p_drop=1.00` confirms the curriculum is in its initial unconditional phase (epochs 0-1).

After ~30 min (2 epochs in), it should transition to ramp:
```
[train] step xxxxx  loss 0.0xxx  pred=v  p_drop=0.73
```

After ~75 min (5 epochs in), steady state:
```
[train] step xxxxx  loss 0.0xxx  pred=v  p_drop=0.20
```

If you see `pred=eps` or `p_drop=0.10` (or any constant non-curriculum value), something didn't pick up — check that `configs/generator/finetune.yaml` got rsync'd.

If the 6h SLURM timeout cuts training off early (likely — v3 took 19 epochs in 12h), resume from the latest checkpoint:

```bash
LATEST_V4_CKPT=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v4/*.ckpt | head -1)
sbatch slurms/ph2_finetune.slurm data=intc_top experiment_name=intc_v4 +resume_from=$LATEST_V4_CKPT generator.training.max_epochs=40
```

### 4.4 Sample v4 stitched tapes (~2h on mit_normal_gpu)

```bash
CKPT_V4=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v4/*.ckpt | head -1)
OUT_V4=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v4_stitched
echo "best ckpt: $CKPT_V4"
sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G --cpus-per-task=2 --output=logs/sample_v4_%j.out --error=logs/sample_v4_%j.err --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py --manifest data/manifests/ph2_sample.json --checkpoint $CKPT_V4 --out-root $OUT_V4 --n-windows-per-task 10 --prediction-type v"
```

The `--prediction-type v` is critical. Without it, the schedule defaults to `eps`-mode and interprets v-prediction model outputs as ε, producing garbage (random-noise-looking tapes).

Verify when complete:

```bash
ls -d $OUT_V4/INTC/*/ && ls $OUT_V4/INTC/base/ | wc -l   # should show 4 regime dirs, ~3200 files in base
```

### 4.5 Stash prior validation results, run v4 validation

```bash
mv results/validation/INTC results/validation/INTC_v3p5 2>/dev/null || true
ls -d results/validation/*

CKPT_V4=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v4/*.ckpt | head -1)
SYNTH_V4=$OUT_V4/INTC

CKPT=$CKPT_V4 SYNTH_DIR=$SYNTH_V4 sbatch slurms/ph6_validate_cpu.slurm
CKPT=$CKPT_V4 SYNTH_DIR=$SYNTH_V4 PREDICTION_TYPE=v sbatch slurms/ph6_validate_gpu.slurm
```

The CPU validation doesn't need `prediction_type` (it doesn't load the model). The GPU validation does — pass `PREDICTION_TYPE=v` as an env var; the slurm reads it.

### 4.6 Read v4 results and compare to v2/v3/v3.5

```bash
echo "=== B.1 — kurtosis ==="
grep -E "^real|^synth|wasserstein|pass_kurt" results/validation/INTC/B1_return_stats.csv

echo ""
echo "=== F.2 — FiLM γ/β by regime ==="
cat results/validation/INTC/F2_film_modulation.csv

echo ""
echo "=== F.1b — embedding cosine + diagnosis ==="
cat results/validation/INTC/F1b_embedding_cosine.csv
grep "F.1b diagnosis" logs/ph6_validate_gpu_*.out | tail -1

echo ""
echo "=== G.1 — consolidated pass/fail ==="
cat results/validation/INTC/G1_pass_fail_summary.csv

echo ""
echo "=== D.1 — OFI slope ==="
cat results/validation/INTC/D1_regression_stats.csv
```

Then paste these and we'll fill in the v4 column of the comparison table in §6 (and decide ship/iterate/Phase C).

---

## 5. Success Criteria

This is a hypothesis test, not a fishing expedition. Three criteria; **2/3 must clear** to declare Phase B a success and ship v4.

| # | Metric | v2 baseline | v3.5 (current best on conditioning) | **v4 target** | Stretch |
|---|---|---|---|---|---|
| 1 | B.1 kurtosis | 122 | 64 | **> 200** | > 400 |
| 2 | F.2 γ_in std/mean across 4 regimes | ~1% | 1.17% | **> 5%** | > 10% |
| 3 | C.1 ACF β | 0.66 | 0.542 | **< 0.45** | < 0.30 |

Secondary (informational, not gating):
- F.1b cosine matrix max off-diagonal — should *decrease* (regimes more distinct) under the curriculum
- D.1 OFI slope — expected to improve as a downstream consequence of (1)
- Trade fraction — likely to stay around 45-55% (separate failure mode from what Phase B targets)
- G.1 pass count — should be ≥ 14/16

If 2/3 clear → ship v4 and re-run the central hypothesis test (Work 4 §3) on v4-stitched data. If 1/3 → diagnose which thread (v-pred or curriculum) didn't fire and consider an A/B follow-up. If 0/3 → the failures may be data-side (window length too short for ACF; trade-fraction imbalance in the source data) rather than model-side. Pivot to writing the report on v2 + the four-way ablation.

---

## 6. Comparison Table (to be filled in after v4 validation)

| Metric | Real | v2 | v3-e9 | v3-e19 | v3.5 | **v4 (Phase B)** |
|---|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 81 | 71 | 64 | **?** |
| W-1 | — | 1.74e-06 | 1.71e-06 | 1.63e-06 | 1.64e-06 | ? |
| ACF β | 0.21 | 0.66 | 0.482 | 0.488 | 0.542 | **?** |
| ACF lag-50 dev | 0 | 1.28 | 2.38 | 3.96 | 2.94 | ? |
| OFI slope at Δ=50 | 1.11e-04 | 1.11e-05 | 9.27e-06 | 1.26e-05 | 1.49e-05 | ? |
| Trade fraction | 4.8% | 54.9% | 54.9% | 64.0% | 45.0% | ? |
| Size tail exp | 1.567 | 1.869 | 1.869 | 2.147 | 1.856 | ? |
| FiLM γ_in std/mean | — | ~1% | 0.92% | 0.37% | 1.17% | **?** |
| FiLM β_out std/mean | — | ~1% | 10.9% | 7.85% | 5.81% | ? |
| F.1b cosine max | — | 0.831 | 0.831 | 0.869 | 0.830 | ? |
| G.1 pass count | 16/16 | 13/16 | 14/16 | 13/16 | 13/16 | ? |

---

## 7. Bug Ledger Continuation (#46+)

(No new bugs in Phase B implementation — clean code-write + test pass on first run. Bug entries 1-45 are in earlier docs.)

---

## 8. File Inventory Diff Since Work6_update.md

### Added
- [docs/Work6_phaseB.md](Work6_phaseB.md) — this file.

### Modified
- [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py) — `prediction_type` on `DDIMSchedule`, v-mode in `ddim_sample`.
- [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) — `prediction_type` arg, v-prediction loss target, Min-SNR weight adjustment for v, `CFGDropoutCurriculum` dataclass + epoch-driven dropout.
- [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) — `per_timestep_mse` builds correct target based on `schedule.prediction_type`.
- [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) — `_build_schedule` reads prediction_type, `_build_curriculum` constructs from config, `DDPMTrainer` receives both.
- [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py) — schedule constructed with prediction_type from config.
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) — `--prediction-type` CLI arg.
- [scripts/50_validate_generator.py](../scripts/50_validate_generator.py) — `--prediction-type` CLI arg.
- [slurms/ph6_validate_gpu.slurm](../slurms/ph6_validate_gpu.slurm) — `PREDICTION_TYPE` env var, threaded as `--prediction-type` to the validator.
- [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) — `prediction_type: "v"`, `cfg_dropout_curriculum` block.
- [tests/test_validation.py](../tests/test_validation.py) — 6 new tests for Phase B.

**Test count: 50** (was 44 at end of Phase A.5).
