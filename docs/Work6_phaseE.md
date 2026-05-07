# Work 6 — Phase E (v8): Copula-Based Diffusion (Empirical-CDF Gaussianization)

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-05-06
**Status:** Code complete, modular, 77/77 tests passing. Cluster training pending.
**Companion to:** [Work6_FINAL.md](Work6_FINAL.md) (the marginals-vs-joints finding that motivated this phase)

This document covers Phase E: addressing the marginals-vs-joints tension identified in Work6_FINAL §9 by **Gaussianizing the training data via per-feature empirical-CDF transform**. The diffusion model never sees heavy-tailed data; it trains on Gaussianized marginals and learns only the dependence structure. Heavy tails are restored mechanically at sampling time via the inverse empirical CDF.

---

## 0. Executive Summary

The 6-checkpoint study from Work6_FINAL §8-9 established a clean tension:

- **ε-MSE training (v2)** mode-covers the bulk distribution → preserves cross-feature correlations → **truth-aligned ranking** (ρ_diff = +0.40)
- **v-prediction / EDM training (v4-noclip / v5 / v6 / v7 / v7_b)** commits to extreme values → spikier output distributions → breaks correlations → **broken ranking** (ρ_diff ≤ +0.20, with consistent A1/A2 swap)
- BUT: v2 has kurtosis 122 vs real 755 — it's not actually delivering realistic stress

**Phase E sidesteps the tension entirely.** By Gaussianizing the training data:
- The diffusion model trains on **per-feature ~N(0, 1)** distributions — there are no heavy tails to learn or amplify
- ε-prediction's mode-covering becomes an *advantage* not a limitation — preserving Gaussian covariance preserves rank correlations
- Heavy tails are restored mechanically at sampling time via `x_j = F̂_j⁻¹(Φ(z_j))` — no model involvement

**The hypothesis:** v8 should have v2's joint-structure preservation (ε-prediction on Gaussian targets preserves covariance) + v5/real-level kurtosis (inverse CDF restores any marginal shape) + v7_b's temporal quality (rank-preserving transforms preserve rank-autocorrelations).

Code complete in 8 files, 77/77 tests passing. Cluster training/sampling/validation pending.

---

## 1. The Math

### 1.1 Forward transform (training-time)

For each continuous feature `j`:

```
u_j = F̂_j(x_j)          # empirical CDF of x_j among training samples → Uniform(0, 1)
z_j = Φ⁻¹(u_j)          # probit (inverse standard-normal CDF) → ~N(0, 1)
```

Implementation: `np.searchsorted` on the sorted training values gives the rank in O(log n); `(rank + 0.5) / (N + 1)` converts to a uniform; clip to `[ε, 1−ε]` (default ε=0.001) to avoid `Φ⁻¹(0) = −∞`; `scipy.stats.norm.ppf` does the probit.

### 1.2 Inverse transform (sampling-time)

```
u_j = Φ(z_j)             # standard normal CDF → Uniform(0, 1)
x_j = F̂_j⁻¹(u_j)        # inverse empirical CDF → original-scale value
```

Implementation: `scipy.stats.norm.cdf` for Φ; linear interpolation between adjacent sorted training values for `F̂⁻¹`. The reconstruction is monotone and recovers the empirical training distribution in expectation.

### 1.3 Discrete features

`trade_sign ∈ {-1, 0, +1}` has a degenerate empirical CDF (3 step-jumps). Auto-detected by `≤32 unique values` rule; falls back to plain z-score for both forward and inverse.

### 1.4 Why this preserves what we need

**Spearman rank correlation is invariant under any per-feature monotone transform** — including our CDF transform. So if real INTC has Spearman ρ(OFI, return) = X in original space, the Gaussianized data has Spearman ρ(OFI, return) = X in Gaussian space. Diffusion training on Gaussianized data must preserve this rank correlation if it preserves the bulk distribution at all.

**Pearson correlation is approximately preserved** for jointly-Gaussian-like distributions: `ρ_Spearman ≈ (6/π) arcsin(ρ_Pearson / 2)`. So the OFI→return relationship that A2 needs is preserved through the round-trip.

**Heavy tails are preserved by construction**: the inverse CDF reconstructs the empirical training distribution per feature, including all its tail structure. A diffusion model that just produces N(0, 1) z-space samples and applies the inverse CDF will produce real-INTC-distributed outputs — with kurtosis 755, asymmetric skew, and all the heavy-tail structure.

The only thing we lose: **resolution at the very extreme tails** (beyond clip_quantile=0.001 = 0.1% on each tail). For 5M training samples, that's the most-extreme ~5,000 values per feature getting collapsed to the same Gaussian quantile. Compare to v2's `x0_clip=4.0` which rejected everything beyond 99.99% — Phase E loses much less tail resolution.

---

## 2. Modular Code Architecture

The fundamental design decision: **CopulaTransform is a drop-in replacement for NormStats.** Both expose the same `.normalize()` / `.denormalize()` interface and `.anchor_mid` field. Existing code (`dataset.py`, `decode.py`, `sample.py`, etc.) works without per-class branches — Python's duck-typing handles dispatch.

This means **virtually all existing infrastructure is reusable**. The only changes are:

1. **NEW `src/diffmm/data/copula_transform.py`** (~190 LOC) — `CopulaTransform` class implementing `.normalize()` (CDF + probit) and `.denormalize()` (inverse CDF) plus fit/save/load.
2. **`src/diffmm/data/dataset.py`** (5-line edit) — replace inline `(arr - mean) / std` with `norm_stats.normalize(arr)`. Now works with both NormStats and CopulaTransform.
3. **`configs/generator/finetune.yaml`** — new `copula:` block (default-on for v8); switched `prediction_type` back to `eps`, `conditioning_type` back to `film`, `edm.enabled` to `false` (since data is Gaussian, EDM/v-pred don't help).
4. **`scripts/11_finetune_generator.py`** — branch on `cfg.generator.copula.enabled`: `CopulaTransform.fit()` or `NormStats.compute()`. Save copula sidecar at `{norm_stats_file}.copula.json`.
5. **`scripts/utils/run_sampling_manifest.py`** + **`scripts/50_validate_generator.py`** — new `--use-copula` CLI flag; load `CopulaTransform` instead of `NormStats` when set.
6. **`slurms/ph6_validate_gpu.slurm`** — new `USE_COPULA` env var, threaded as `--use-copula` flag.
7. **`tests/test_validation.py`** — 7 new tests covering forward/inverse round-trip, Gaussianization of marginals, discrete-feature passthrough, rank-correlation preservation, heavy-tail recovery, save/load, dataset integration.

### Modularity guarantees verified

- **77/77 tests pass** including all the prior phases (Phase A-D code paths unchanged)
- v2-v7 checkpoints continue to load + sample + validate without modification (no `--use-copula` flag, behavior identical)
- All Phase E logic conditional on `--use-copula` / `USE_COPULA=1` / `copula.enabled: true`
- The decode pipeline (`src/diffmm/generator/decode.py`) was *not modified* — duck-typing through `.denormalize()` works for both classes

---

## 3. v8 Configuration — Architecturally Closest to v2

Per Work6_FINAL §8.1's mechanism ("ε-MSE mode-covers the bulk; preserves cross-feature correlations"), v8 uses the v2 architecture with the copula transform on top:

| Setting | v2 | v8 | Why |
|---|---|---|---|
| Conditioning | FiLM | **FiLM** | v2's FiLM uniquely preserved OFI→return correlation. AdaLN's extra capacity helped marginals but hurt joints. With copula handling marginals, FiLM's joint-preserving conditioning is the right architecture. |
| Parameterization | ε-pred | **ε-pred** | Mode-covering bulk-fitting on Gaussianized data is exactly what we want. v-prediction's signal-correlated target adds nothing on N(0,1) inputs. |
| EDM | off | **off** | Preconditioning math assumes σ_data ≈ data std. With copula, data is exactly N(0,1); EDM's c_skip/c_out reduce to identity-ish. No benefit, more complexity. |
| x0_clip | 4.0 | **4.0** | Standard for ~N(0,1) data. Inverse CDF restores extremes regardless of clip. |
| FiLM init | zeros | **zeros** | v2 default; the conditioning-curriculum-with-random-init combination introduced in Phase A interacts unpredictably with mode-covering training. |
| CFG p_drop | 0.1 | **0.1** | v2 default; curriculum disabled. |
| Patience | 3 | **3** | v2 default; we want the same early-stopping behavior that gave v2 its mode-covering goodness. |
| Min-SNR γ | 5.0 | **5.0** | Same as v2. |
| Curriculum | off | **off** | Same as v2. |
| **Copula transform** | n/a | **on** | THE single architectural change. |

The v8 hypothesis is sharply scoped: **copula-Gaussianization is the only change from v2.** If v8 has v2's ρ_diff *and* better marginals (kurtosis, skew, scale), the copula approach is validated. If v8 has v2's ρ_diff but no better marginals, the copula transform isn't carrying the load. If v8 loses ρ_diff, something subtle in the training-on-Gaussianized-data dynamics breaks the joint preservation.

---

## 4. Predicted v8 Results

| Metric | Real | v2 | v5 | v7_b | **v8 prediction** | Why |
|---|---|---|---|---|---|---|
| **Excess kurtosis** | 755 | 122 | 1256 | 122 | **~700-800** | Inverse CDF restores empirical training kurtosis. Not 100% match because inverse CDF samples uniformly across quantiles whereas N(0,1) sampling concentrates on bulk; small underestimate expected. |
| **Synth std (× real)** | 1.0 | ~1 | ~1 | 0.30 | **~1** | Inverse CDF restores real scale by construction. |
| **Skew** (real -0.47) | -0.47 | small | -17.2 | +0.795 | **~-0.47** | Inverse CDF preserves marginal shape including asymmetry. |
| **OFI→return slope** at Δ=50 | 1.11e-04 | 1.11e-05 | similar | degraded | **~1.11e-05** (matches v2) | ε-prediction on Gaussian targets preserves covariance → preserves Spearman ρ(OFI, return) → preserves the slope after inverse CDF (which is monotone). |
| **ACF \|return\| β** (real 0.21) | 0.21 | 0.66 | 0.542 | 0.585 | **~0.4-0.6** | Rank-autocorrelations are preserved exactly through monotone transforms. ACF improvement over v2 likely. |
| **Trade-sign lag-1 ACF** (real 0.675) | 0.675 | 0.385 | 0.103 | 0.480 | **~0.4-0.5** | Trade sign is discrete (z-score fallback) but rank correlations preserved — should match v2's level. |
| **F.1b cosine** | — | 0.831 | 0.722 | 0.720 | **~0.83** (FiLM-class) | Same conditioning architecture as v2. |
| **ρ_diff** (the critical number) | — | **+0.40** | -0.20 | +0.20 | **+0.40** | OFI→return preserved → A2 ranks correctly. The whole point of Phase E. |

### What's mathematically guaranteed vs. empirically uncertain

**Guaranteed by construction** (inverse CDF math):
- Synth marginal kurtosis ≈ training kurtosis
- Synth marginal scale ≈ training scale
- Synth skew ≈ training skew
- Spearman rank correlations preserved exactly

**Likely from theory + empirical evidence**:
- ε-prediction on Gaussianized data preserves Pearson correlation (because joint Gaussian is fully determined by mean + covariance, and ε-MSE fits covariance well)
- OFI→return predictability persists in synth (follows from rank correlation preservation)

**Empirically uncertain**:
- Whether the diffusion model's mode-covering on Gaussianized data is good ENOUGH to preserve correlations strongly enough for A2's OFI signal to remain useful
- Whether the FiLM conditioning architecture interacts well with Gaussianized inputs (no obvious reason it wouldn't, but not tested)
- Whether the curriculum-disabled v2-style training actually learns regimes well enough for the regime-conditional copula shifts (toxic vs base have different dependence structures in real)

**Cost:** ~6h training + ~2h sampling + ~30 min validate + ~40 min hypothesis test = ~9h wall.

---

## 5. Cluster Commands — v8 Train / Sample / Validate

### 5.1 Rsync from Mac

```bash
rsync -av --progress \
  src/diffmm/data/copula_transform.py \
  src/diffmm/data/dataset.py \
  configs/generator/finetune.yaml \
  scripts/11_finetune_generator.py \
  scripts/utils/run_sampling_manifest.py \
  scripts/50_validate_generator.py \
  slurms/ph6_validate_gpu.slurm \
  tests/test_validation.py \
  docs/Work6_phaseE.md \
  favara@orcd-login.mit.edu:~/diff_mm/
```

If rsync misbehaves (it has before), fall back to per-file scp:

```bash
for f in src/diffmm/data/copula_transform.py src/diffmm/data/dataset.py \
         configs/generator/finetune.yaml scripts/11_finetune_generator.py \
         scripts/utils/run_sampling_manifest.py scripts/50_validate_generator.py \
         slurms/ph6_validate_gpu.slurm tests/test_validation.py \
         docs/Work6_phaseE.md; do
  scp "$f" favara@orcd-login.mit.edu:~/diff_mm/"$f"
done
```

### 5.2 Verify on cluster

```bash
module load miniforge && source activate diffmm

# Copula module is present
ls -la src/diffmm/data/copula_transform.py
grep "class CopulaTransform" src/diffmm/data/copula_transform.py

# Config has copula.enabled: true and v2-style settings
grep -A2 "^copula:" configs/generator/finetune.yaml
grep "conditioning_type\|prediction_type\|enabled:" configs/generator/finetune.yaml | head -10

# 77 tests pass
python -m pytest tests/ -q 2>&1 | tail -3
```

### 5.3 Train v8 from scratch (~6h on mit_normal_gpu, may chunk)

```bash
sbatch --exclude=node4104 slurms/ph2_finetune.slurm \
    data=intc_top experiment_name=intc_v8
```

Watch the first 100 train steps:

```bash
LATEST=$(ls -t logs/ph2_finetune_*.out | head -1)
tail -f "$LATEST"
```

Expected early lines:
- `[INFO] ph2_finetune: fitting copula transform on training tapes (60 files)` — fits per-feature CDFs from 60 train tapes (~60 sec)
- `[INFO] ph2_finetune: fit copula in NN.Ns → /home/.../norm_stats_INTC.copula.json (discrete_features=(...,))`
- `[INFO] ph2_finetune: conditioning_type=film  edm_enabled=False`
- Train steps: `[train] step 20 loss X.XXXX  pred=eps  p_drop=0.10`

Notable differences from prior runs:
- `pred=eps` (not pred=v or pred=edm)
- `p_drop=0.10` constant (no curriculum ramp)
- Loss values: smaller than v2's because the targets are Gaussian (mode-covering MSE on N(0,1) is much easier than on heavy-tailed data). Expect ~0.3-0.5 → ~0.05 over the run.

### 5.4 Sample v8 stitched tapes (~2h)

```bash
CKPT_V8=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v8/*.ckpt | head -1)
OUT_V8=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v8_stitched
echo "v8 ckpt: $CKPT_V8"

sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G --cpus-per-task=2 --output=logs/sample_v8_%j.out --error=logs/sample_v8_%j.err --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py --manifest data/manifests/ph2_sample.json --checkpoint $CKPT_V8 --out-root $OUT_V8 --n-windows-per-task 10 --conditioning-type film --prediction-type eps --use-copula --x0-clip 4.0"
```

**Critical:** the wrap MUST include `--use-copula`. Without it, the sampler builds NormStats and the inverse transform never runs → samples stay in z-space (wrong scale). Plus `--conditioning-type film` and `--prediction-type eps` to match training; `--x0-clip 4.0` is the v2-style Gaussian-data clip.

### 5.5 Validate v8 (~25 min)

```bash
mv results/validation/INTC results/validation/INTC_v7_b 2>/dev/null || true

CKPT_V8=$(ls -t /orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v8/*.ckpt | head -1)
SYNTH_V8=$OUT_V8/INTC

CKPT=$CKPT_V8 SYNTH_DIR=$SYNTH_V8 sbatch slurms/ph6_validate_cpu.slurm
CKPT=$CKPT_V8 SYNTH_DIR=$SYNTH_V8 CONDITIONING_TYPE=film USE_COPULA=1 sbatch --exclude=node4104 slurms/ph6_validate_gpu.slurm
```

### 5.6 v8 hypothesis test (the project-defining number)

```bash
sbatch slurms/ph4_replay_synth.slurm \
    data=intc_top \
    +synthetic_root=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v8_stitched
sbatch --dependency=singleton slurms/ph5_hypothesis.slurm data=intc_top
```

When done:

```bash
cat results/hypothesis/INTC/bootstrap.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'rho_diff = {d[\"rho_diff_point\"]:+.3f}')
print(f'rho_hist = {d[\"rho_hist_point\"]:+.3f}')
print(f'p_value = {d[\"p_value_one_sided\"]:.3f}')
print(f'pi_diff (v8): {d[\"pi_diff\"]}')
"
cp results/hypothesis/INTC/bootstrap.json results/hypothesis/INTC/bootstrap_v8.json
```

If `rho_diff = +0.40` and `pi_diff = [A0, A3, A2, A1]` (matching v2 pattern), Phase E is validated and **v8 becomes the production generator** with v2's ranking + better marginals.

---

## 6. Success Criteria

The decisive question for v8 is whether the OFI→return correlation survives diffusion training on Gaussianized data:

### 6.1 Primary criteria (3/3 to ship v8)

| # | Metric | Target | Why |
|---|---|---|---|
| 1 | ρ_diff | **+0.40** | Same as v2; the central scientific question |
| 2 | π_diff = [A0, A3, A2, A1] | (no A1/A2 swap) | Direct test of OFI→return preservation |
| 3 | Kurtosis | **400-1000** | Inverse CDF should restore tail; in real range |

### 6.2 Secondary criteria (informational)

| Metric | Target |
|---|---|
| Synth std / real std | within 0.8-1.2× (~real scale) |
| Skew | within 0.3 of real -0.47 |
| ACF lag-50 dev | < 0.5 (passes) |
| Trade-sign ACF | > 0.4 (close to real 0.675) |
| F.1b cosine max off-diag | < 0.85 (FiLM-class, similar to v2's 0.83) |
| G.1 pass count | ≥ 14/16 |

### 6.3 Decision tree

- **3/3 primary clear** → **ship v8 as the production generator.** First checkpoint to combine v2's ranking with realistic marginal stress.
- **ρ_diff = +0.40 but kurtosis < 400** → copula transform isn't restoring enough tail. Reduce clip_quantile to 0.0001 or fit GPD tails. Modest fix.
- **ρ_diff = +0.20 (broken ranking)** → Gaussianization wasn't enough; ε-prediction with FiLM on Gaussianized data still breaks something. Diagnose by examining synth OFI→return slope directly. Possibly switch to FiLM zero-init exact v2 settings or revert curriculum/patience changes inherited from earlier phases.
- **ρ_diff < +0.20** → unexpected. Phase E broke something we didn't anticipate. Falls to detailed mechanism diagnosis.

---

## 7. The 7-Checkpoint Comparison (to be filled in)

| Metric | Real | v2 | v5 | v6 | v7 | v7_b | **v8 (Phase E)** |
|---|---|---|---|---|---|---|---|
| Excess kurtosis | 755 | 122 | 1256 | 186 | 124 | 122 | **?** |
| W-1 | — | 1.74e-06 | 9.81e-07 | 1.72e-06 | 1.54e-06 | 1.51e-06 | ? |
| ACF lag-50 dev | 0 | 1.28 | 0.296 | 0.665 | 0.206 | 0.162 | ? |
| ACF β | 0.21 | 0.66 | 0.542 | 0.752 | 0.616 | 0.585 | ? |
| Skew | -0.47 | small | -17.2 | -1.08 | -0.20 | +0.795 | ? |
| Trade-sign ACF | 0.675 | 0.385 | 0.103 | 0.526 | 0.514 | 0.480 | ? |
| F.1b cosine max | — | 0.831 | 0.722 | 0.722 | 0.715 | 0.720 | ? |
| Synth std (× real) | 1.0 | ~1 | ~1 | 0.24 | 0.30 | 0.30 | **?** |
| **ρ_diff** | — | **+0.40** | -0.20 | +0.20 | +0.20 | +0.20 | **?** |
| G.1 pass | 16/16 | 13/16 | 13/16 | 13/16 | 13/16 | 13/16 | ? |

---

## 8. Honest Verdict (interim)

Phase E is the most theoretically-grounded intervention in the workstream. The marginals-vs-joints tension we identified in Work6_FINAL §9 is sidestepped by construction: train on Gaussian data (preserve joints via ε-MSE mode-covering), restore marginals at sample time (via mathematically-guaranteed inverse CDF). The risk isn't in the math; it's in whether MSE-based diffusion training on Gaussianized LOB data preserves Pearson correlations strongly enough for A2's OFI signal to remain useful.

**Three outcomes have different scientific contributions:**

1. **v8 succeeds** (ρ_diff = +0.40 + good marginals) → first generator combining truthful ranking with realistic stress. The copula-diffusion approach is novel for financial LOB data and would be the project's most concrete methodological contribution.

2. **v8 fails on ρ_diff** (still +0.20) → Gaussianization isn't sufficient; the marginals-vs-joints tension goes deeper than just data preprocessing. Falsifies a clean hypothesis; meaningful negative result.

3. **v8 partial** (ρ_diff between +0.20 and +0.40) → suggestive but not definitive; would motivate a v9 with tighter v2-style settings.

All three outcomes strengthen the report's discussion section. The negative result (mode 2) would actually be quite interesting — it would point at a deeper architectural failure in how MSE-based diffusion handles cross-feature dependence, beyond just the heavy-tail issue.

The diffusion model workstream is genuinely complete after Phase E regardless of outcome — we'll have systematically tested every reasonable intervention against the marginals-vs-joints problem.
