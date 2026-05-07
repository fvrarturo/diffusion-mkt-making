# Lab notes — Phase E (v8) launch

**Date:** 2026-05-06 → 2026-05-07
**Branch:** `work_4may`
**Goal:** Implement and run Phase E (copula-based diffusion) — addresses the marginals-vs-joints tension from `docs/Work6_FINAL.md` §9.

## TL;DR

- Built `CopulaTransform` as a drop-in replacement for `NormStats`. Both share the `.normalize() / .denormalize()` interface so `LOBWindowDataset`, `decode_*`, and the sampler are agnostic.
- Trained two variants in parallel:
  - **v8** = copula + FiLM + ε-pred + no EDM (the "all-changes-vs-v7_b reverted" baseline for copula).
  - **v8_b** = copula + AdaLN-Zero + ε-pred + no EDM (clean A/B on the conditioning module).
- Both finished training cleanly. Best ckpts: `intc_v8/ckpt-epoch009-valloss0.0114.ckpt` and `intc_v8_b/ckpt-epoch009-valloss0.0111.ckpt`. v8_b's val loss edged v8 by ~3%.
- v8 sampling underway on GPU; v8_b will switch to CPU partition to avoid the throughput cliff (see "Cluster lessons" below).

## What we built

### 1. `CopulaTransform` (`src/diffmm/data/copula_transform.py`)
- Per-feature empirical CDF via `np.searchsorted` on sorted training values + `scipy.stats.norm.ppf` (probit).
- Discrete features (auto-detected: ≤32 unique values) fall back to z-score because `Φ⁻¹` of a degenerate CDF blows up.
- Inverse via `Φ` then linear-interpolated inverse-CDF lookup.
- Tail winsorization at `clip_quantile=0.001` to avoid `Φ⁻¹(0)=-∞`.
- Saves as JSON metadata + `.npz` of sorted-value arrays (~330 KB for 8 features × 60 INTC tapes).
- Auto-fits in ~32s on training tapes.

### 2. Wired through every pipeline stage
- `dataset.py`: replaced inline `(arr - mean)/std` with `norm_stats.normalize(arr)` (duck-typed).
- `11_finetune_generator.py`: branches on `cfg.generator.copula.enabled` to fit/load CopulaTransform vs NormStats.
- `13_sample_synthetic.py`: same branch — loads `CopulaTransform.load(.copula.json)` if enabled.
- `50_validate_generator.py`: same branch (already had `--use-copula` CLI flag wired earlier).
- `decode.py`: works unchanged because both classes have the same interface plus `anchor_mid`.

### 3. Config additions
`configs/generator/finetune.yaml`:
```yaml
copula:
  enabled: true
  clip_quantile: 0.001
  max_samples_per_feature: 5000000
  discrete_features: null   # auto-detect
```

For v8 we also reverted finetune defaults from v7_b's stack:
- `prediction_type: v` → `eps`
- `edm.enabled: true` → `false`
- `conditioning_type: adaln_zero` → `film`

(v8_b overrides `conditioning_type` back to `adaln_zero` on the CLI.)

## Bugs hit and fixed (in order)

The "easy" plumbing accounted for ~60% of the wall time today. Worth recording so we move faster next time.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | First two training jobs ran with `conditioning_type: adaln_zero`, `prediction_type: v`, `edm: true`, no `copula:` block | rsync put the Phase E files at `~/diff_mm/` flat, not in subdirs | Re-rsync with `-R` flag to preserve relative paths |
| 2 | Sampler ignored `copula:` config | `13_sample_synthetic.py` had no copula branch — only `run_sampling_manifest.py` did | Added `OmegaConf.select(cfg, "generator.copula.enabled")` branch |
| 3 | `Could not override 'generator.checkpoint_path'` | Hydra strict mode — key not in struct of `finetune.yaml` | Use `+generator.checkpoint_path=...` (append) — but better: pass `generator=ddim_sample` (next bug) |
| 4 | `Key 'regime_to_condition' is not in struct` | `ddim_sample.yaml` config wasn't being selected, so sampling-specific keys absent | Pass `generator=ddim_sample` as a Hydra group override; drops the `+` on `checkpoint_path` |
| 5 | `mat1 and mat2 shapes cannot be multiplied (4096x9 and 8x256)` | `ddim_sample.yaml` had stale `n_features: 9` while `FEATURE_COLUMNS` has 8 entries | One-line yaml fix: `n_features: 8` |
| 6 | `bid_px: expected dtype Float64, got Float32` | `CopulaTransform.denormalize` returned Float32; `NormStats.denormalize` happens to return Float64 because `mean/std` are Float64 → numpy upcasts | Change CopulaTransform output dtype to Float64 explicitly |
| 7 | `trade_px: expected dtype Float64, got Null` | When all 256 events in a window have `trade_sz=0`, `trade_px_list` is all `None` → Polars infers `Null` dtype. Stitched decode already had a fix (Null→Float64 recast) but per-window decode didn't | Mirror the stitched recast in `decode_window_to_dataframe` |

The v8 model is more likely to produce trade-less windows than NormStats-based models — copula `denormalize` of `trade_sz` (heavy-tailed with mass at 0) maps moderate z to 0 by construction. Worth flagging in validation.

## Cluster lessons (the big one)

`mit_normal_gpu` has a tight QOS cap — we got `QOSMaxSubmitJobPerUserLimit` even submitting `--array=0-199`. Concurrency was forced down to `%2`.

Per-task GPU compute is ~10s. But each new SLURM allocation between `%2` task pairs takes ~3 min (GPU re-allocation + Python startup + checkpoint reload). At `%2`: 200 sequential pairs × 3 min ≈ **10h per model**.

Prior runs (v2-v7) sampled in ~20 min. We assumed they were on GPU with high concurrency; turns out they were on **CPU partition** (`mit_normal`), which has way more concurrent slots. Per-task is 3-4× slower on CPU but the throughput at `%20` swamps that.

**Action for v8_b:** wrote `slurms/ph2_sample_cpu.slurm` (CPU partition, no `--gres=gpu`, sets `cluster.device=cpu`). Will sanity-test then run v8_b through it.

**Lesson for future projects:** check the cluster's actual concurrency limits before defaulting to GPU. If the model fits in <1 min per task on CPU and the partition is permissive, CPU+high-concurrency beats GPU+constrained-concurrency by orders of magnitude on wall time.

## Open questions / next steps

- [ ] Confirm CPU sampling works end-to-end (1-task sanity submit currently being tested)
- [ ] Run v8_b sampling on CPU once v8 finishes
- [ ] Validation (Phase F): per-feature kurtosis recovery, OFI→return Spearman, A1/A2 ranking, regime-conditional copula
- [ ] Hypothesis test (Phase 5): ρ_diff between v8/v8_b synth ranking and held-out-truth ranking
- [ ] Side-by-side: v2 / v6 / v7_b / v8 / v8_b on the headline metrics (kurtosis, OFI corr, A1/A2 swap, ρ_diff)

## Predicted outcomes (record so we can verify against them)

From `docs/Work6_phaseE.md`:
- ρ_diff = +0.40 (vs +0.10 for v6/v7_b)
- No A1/A2 swap (truth has A2 > A1; we want to recover this — v4-noclip / v5 / v6 / v7 / v7_b all swapped)
- Kurtosis 400-1000 (vs ~120 for v2, ~600-2000 for v7_b but with broken correlations)

The ablation between v8 (FiLM) vs v8_b (AdaLN-Zero) tells us whether the copula's value-add depends on the conditioning module.

## File index

New / modified this session:
- `src/diffmm/data/copula_transform.py` (new, ~190 LOC)
- `src/diffmm/data/dataset.py` (5-line edit — duck-typed normalize)
- `scripts/11_finetune_generator.py` (branch on copula.enabled)
- `scripts/13_sample_synthetic.py` (branch on copula.enabled)
- `scripts/utils/run_sampling_manifest.py` (--use-copula CLI)
- `scripts/50_validate_generator.py` (--use-copula CLI + _load_normalizer helper)
- `scripts/utils/run_v8_sampling.sh` (chunked-submit loop for QOS-constrained submissions)
- `slurms/ph2_sample_cpu.slurm` (CPU variant of sampling slurm)
- `slurms/ph6_validate_gpu.slurm` (USE_COPULA env var threading)
- `configs/generator/finetune.yaml` (copula block + reverted to FiLM/eps/no-EDM for v8)
- `configs/generator/ddim_sample.yaml` (n_features: 9 → 8 — stale value)
- `src/diffmm/generator/decode.py` (per-window Null→typed recast)
- `tests/test_validation.py` (7 new tests for CopulaTransform)
