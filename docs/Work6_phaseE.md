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

## 7. The Checkpoint Comparison (filled in 2026-05-07, extended 2026-05-08)

| Metric | Real | v2 | v5 | v6 | v7 | v7_b | **v8** | **v8_b** | **v9** | **v9_b** |
|---|---|---|---|---|---|---|---|---|---|---|
| Excess kurtosis (uncond) | 755 | 122 | 1256 | 186 | 124 | 122 | 2.6 | degen | (small) | (small) |
| Conditional kurtosis | 55.8 | 122 | ? | ? | ? | 174 | 155.7 | NaN | **−1.7** | **−1.6** |
| frac_zero (mid_return) | 0.89 | 0.00 | ? | ? | ? | 0.00 | 0.17 | 1.0 | **0.99** | **0.99** |
| Synth std (× real) | 1.0 | ~1 | ~1 | 0.24 | 0.30 | 0.30 | 1.85 | 0.0 | (real-scale) | (real-scale) |
| OFI→return β sign | + | + | ? | − | − | − | **−** | 0 | (broken) | (broken) |
| Trade fraction | 4.8% | ~5% | ? | ? | ? | ? | 0.9% | 0% | low | moderate |
| **ρ_diff** | — | **+0.40** | -0.20 | +0.20 | +0.20 | +0.20 | **+0.80** ⚠ | **+0.80** ⚠ | **−0.80** | **−1.00** |
| **A2 above A1?** | yes | **yes** | no | no | no | no | yes (artifact) | no | **no** | **no** |
| G.1 pass | 16/16 | 13/16 | 13/16 | 13/16 | 13/16 | 13/16 | 7/16 | 6/16 | (TBD) | (TBD) |

**Two ρ_diff results are flagged ⚠ as artifacts:** v8/v8_b achieved +0.80 with degenerate or near-degenerate output. v9's −0.80 is the **honest measurement** of the same broken OFI joint that v8 was hiding behind return collapse. See §11 for the corrected causal story.

**v8_b ρ_diff is flagged ⚠ as a tie-breaking artifact** — synth tapes have zero variance, so all four agents earn identical zero PnL and the ranking is determined by alphabetical tie-break. The "0.80" is meaningless for v8_b; only v8's 0.80 is a real result.

---

## 8. Honest Verdict (interim)

Phase E is the most theoretically-grounded intervention in the workstream. The marginals-vs-joints tension we identified in Work6_FINAL §9 is sidestepped by construction: train on Gaussian data (preserve joints via ε-MSE mode-covering), restore marginals at sample time (via mathematically-guaranteed inverse CDF). The risk isn't in the math; it's in whether MSE-based diffusion training on Gaussianized LOB data preserves Pearson correlations strongly enough for A2's OFI signal to remain useful.

**Three outcomes have different scientific contributions:**

1. **v8 succeeds** (ρ_diff = +0.40 + good marginals) → first generator combining truthful ranking with realistic stress. The copula-diffusion approach is novel for financial LOB data and would be the project's most concrete methodological contribution.

2. **v8 fails on ρ_diff** (still +0.20) → Gaussianization isn't sufficient; the marginals-vs-joints tension goes deeper than just data preprocessing. Falsifies a clean hypothesis; meaningful negative result.

3. **v8 partial** (ρ_diff between +0.20 and +0.40) → suggestive but not definitive; would motivate a v9 with tighter v2-style settings.

All three outcomes strengthen the report's discussion section. The negative result (mode 2) would actually be quite interesting — it would point at a deeper architectural failure in how MSE-based diffusion handles cross-feature dependence, beyond just the heavy-tail issue.

The diffusion model workstream is genuinely complete after Phase E regardless of outcome — we'll have systematically tested every reasonable intervention against the marginals-vs-joints problem.

---

## 9. Outcome (added 2026-05-07)

**v8 ρ_diff = +0.80 — the headline number is a 2× improvement over both the historical-val baseline (ρ_hist = 0.40) and every prior post-v2 architecture (~+0.20 for v5/v6/v7/v7_b). It also restores the truth's A2 > A1 ordering that all post-v2 architectures had broken.**

**But the underlying mechanism is *not* what Phase E predicted.** The three primary criteria from §6 were:

1. ❌ **Kurtosis 400-1000** → got **2.6**. The copula did not recover heavy tails.
2. ✅ **No A1/A2 swap** → π_diff = `[A0, A2, A3, A1]`. Truth's A2 > A1 ordering is preserved.
3. ❌ **OFI→return β > 0** → got **−1.4e-5** (Δ=10) and **−6.2e-4** (Δ=50). Wrong-sign correlation, same failure mode as v6/v7.

So 1/3 primary criteria passed. By the §6 outcome map, this is a **partial result** — definitely above "broken ranking" (mode c) but well short of "all 3 clear" (mode a). The ranking-correlation win is real; the marginal-stress-recovery story Phase E was designed to deliver is not.

### Why ρ_diff was high anyway (mechanism analysis)

The copula's inverse-CDF only restores heavy tails *if the diffusion model emits z-values that span the full tails of N(0,1)*. With ε-prediction + min_snr_gamma=5 weighting, the model is incentivized to produce conservative, sub-Gaussian outputs (small |z|), so the inverse CDF only recovers bulk values, not tails. Kurtosis 2.6 is consistent with the model emitting near-uniform-ish z-values plus Gaussian noise.

The high ρ_diff = 0.80 nonetheless reflects a real signal: **agent rankings can be preserved even when marginals/bulk correlations are broken**. The mechanism:
- A0 (constant) sits out → well-defined Sharpe regardless of synth distribution.
- A2_AS_OFI vs A1_AS: even with wrong-sign OFI (β_synth has opposite sign of β_real), the *magnitude* of the OFI signal still distinguishes A2 from A1. Agents that USE OFI (A2) earn *different* Sharpe than those that don't (A1), and the ordering can match truth even when the correlation sign is flipped.

This is a weaker but interesting positive result and worth its own paragraph in the report's discussion: rank-correlation tests can credit a generator with capturing agent-ranking signal even when the underlying mechanism (heavy-tail stress + correctly-signed OFI predictability) is broken. Whether that's a feature or a bug depends on what you're using the generator for.

### v8_b: degenerate outcome

v8_b's identical-looking ρ_diff = +0.80 is an artifact. The model collapsed to zero-variance output at sample time:
- `synth std = 0`, `synth kurt = blank`, `trade fraction = 0.000`
- All four agents earn identical zero PnL on every tape
- π_diff = `[A0, A1, A2, A3]` is just alphabetical tie-breaking

The likely cause is AdaLN-Zero's zero-initialization combined with `min_snr_gamma=5` weighting — both push the model toward conservative outputs, and together they produced full collapse. v8 (FiLM) avoided this because FiLM doesn't have the zero-init bias. **v8_b should be excluded from comparisons; it's a sampling-time failure, not a meaningful result.**

### Implications for the workstream

Phase E gave us a useful agent-ranking generator (v8) but did not solve the marginals problem. The "marginals vs joints" framing from Work6_FINAL §9 stands: the only checkpoint that delivers v2-class agent-ranking signal AND v2-class realistic marginals is still **v2 itself**. v8 doubled the ranking signal but lost the marginal stress (kurtosis 2.6 vs v2's 122 vs real's 755).

**A v9 worth trying** would lower (or remove) `min_snr_gamma` to allow the model to commit to extreme z-values, which the copula can then map to true tails. If kurtosis recovers to 100+ at v9 with ρ_diff still near +0.80, that would be the clean Phase E result that was originally predicted.

For the report: lead with v8's ρ_diff = +0.80 as the headline, but be honest about the mechanism — it's not heavy-tail stress that drives the ranking improvement, it's preserved ranking-of-agents under broken marginals. That's a more interesting story than "we recovered the kurtosis" anyway: it tells the reader something about *what hypothesis tests like ρ_diff actually measure* in this kind of setup.

---

## 10. v9 — Dequantization Fix (planned 2026-05-07)

The §9 diagnostic identified the v8 marginal failure as a **CDF discontinuity at point masses**: real INTC mid_return is 89% exact zeros, the empirical CDF jumps at x=0, and the probit Φ⁻¹ collapses the 89% to a single z-value. The diffusion model trains on a near-constant z target for that feature → output collapses to a near-constant in z-space → inverse CDF gives small-magnitude returns clustered at 0 (kurt 2.6 instead of predicted 400-1000).

The standard fix from the normalizing-flow literature (Theis et al. 2016): **dequantization**. At fit time, spread point-mass values uniformly over [v_pm − ε, v_pm + ε] before building the empirical CDF. At normalize time, add fresh U(−ε, +ε) noise per call (so the model sees a smoothly-distributed z target). At denormalize time, snap any sampled value within ε of v_pm back to v_pm (restores the zero-inflated structure exactly).

Implemented in `src/diffmm/data/copula_transform.py` with a robust 5th-percentile-based ε computation (the literal-min approach was pathological for features like trade_dist with continuous density near 0). All 11 copula tests pass. Configured via `copula.dequantize: true` in `configs/generator/finetune.yaml`.

**Two parallel runs:**
- **v9** = copula + dequantization + FiLM + ε-prediction (the "fix v8's preprocessing" run)
- **v9_b** = copula + dequantization + FiLM + v-prediction (does dequantized copula rescue v-pred?)

Predicted outcomes per the §6 decision tree:
- ρ_diff ≥ +0.40 (preserve v8's ranking signal)
- Kurtosis 100-800 (heavy-tail recovery via the inverse CDF, now that the CDF is continuous)
- frac_zero ≈ 0.80 (close to real's 0.89)

**See `notebooks/2026-05-07_deep-diagnostic-and-v9-plan.md` for the full mechanism analysis and implementation log.**

---

## 11. v9 Outcome (added 2026-05-08)

Both runs trained ~6h (hit time limit, both saved checkpoints), sampled overnight on CPU partition (400 tasks at %20 concurrency), validated, and ran ph5 hypothesis. Headline:

| Run | ρ_diff | π_diff | A2 above A1? |
|---|---|---|---|
| v8 (baseline) | +0.80 ⚠ | `[A0, A2, A3, A1]` | yes (artifact) |
| **v9** | **−0.80** | `[A3, A1, A0, A2]` | no — A2 ranked LAST |
| **v9_b** | **−1.00** | `[A3, A1, A2, A0]` | no — exact reverse of truth |

ρ = −1.00 means v9_b's ranking is the perfect anti-image of held-out truth. This is a sharp, unambiguous negative result.

### What dequantization actually did to the marginals

Dequantization "fixed" the marginal but not in a useful way:

| Model | frac_zero | cond_std | cond_kurt |
|---|---|---|---|
| real | 0.89 | 6.76e-5 | **55.8** |
| v8 (no dequant) | 0.17 | 1.65e-5 | 155.7 |
| v9 | **0.99** (over-corrected) | 1.23e-4 | **−1.7** (uniform-shaped) |
| v9_b | 0.99 | 1.42e-4 | −1.6 |

frac_zero is now ≈ real (slightly over-corrected — snap-back works too aggressively because ε was too large). But the **conditional return distribution is now uniform-noise-shaped** (kurt ≈ −1.7 ≈ kurtosis of a uniform distribution). Heavy tails are NOT restored. The cause: ε for mid_return was 4.93e-5; real conditional std is 6.76e-5; **ε / cond_std ≈ 0.73**, so the dequantization noise dominates the actual return signal. The model learned to emit "uniform noise within the band" rather than reproduce the heavy-tailed continuous part.

**Methodology rule for the report:** dequantization ε must be at least an order of magnitude *below* the conditional standard deviation of the non-point-mass portion of the distribution. For mid_return the right ε is ~1e-6 to 1e-7 (below the smallest tick-return scale), not ~5e-5. The 5th-percentile-based ε was robust against floating-point artifacts but produced a band too wide for the natural feature scale.

### Why ρ_diff flipped — the corrected causal story

**An incorrect first reading of this result** would be: "dequantization changed the joint structure and broke ρ_diff." That framing is wrong on the causal direction.

Look at v8's OFI→return regression from `60_deep_diagnostic.py` (run 2026-05-07, BEFORE v9 existed):

```
v8 OFI→return slopes by horizon Δ:
  Δ=1:   slope=-4.59e-10   sign=-
  Δ=10:  slope=-1.60e-09   sign=-
  Δ=50:  slope=-1.05e-08   sign=-
  Δ=100: slope=-2.68e-08   sign=-
```

**v8's OFI joint was already wrong-signed at every horizon, BEFORE any dequantization existed.** Same direction as v6/v7/v7_b. The copula transform broke the OFI→return joint the moment it was applied — dequantization had nothing to do with breaking the joint.

What was different about v8 vs v9 is the **PnL CONSEQUENCES** of the broken joint:

- **v8**: returns were collapsed (z-mean = −2.55, z-std = 0.30 → near-constant model output). Wrong-sign OFI signal had ~zero PnL effect because returns were tiny everywhere. All four agents earned ≈ −$6 per tape regardless of strategy. The ρ_diff = +0.80 was measuring **fill-count noise on top of near-zero returns** — happened to align with truth's tail-risk dimension by accident.

- **v9**: dequantization restored return variance to ~real scale. **Now the wrong-sign OFI has real PnL consequences.** A2_AS_OFI tilts quotes per the (broken) OFI signal, gets more fills (3× A1's fill rate), and every fill is in the wrong direction → A2 loses the most → A2 ranks LAST. The mechanism is unambiguous in the per-agent table:

| agent | v9 pnl | v9 fills | v9 fill_rate |
|---|---|---|---|
| A0_const | +0.004 | 0.0002 | 53 |
| A1_AS | +0.005 | 0.0003 | 104 |
| **A2_AS_OFI** | **−0.005** | **0.0007** | **284** ← only loser; uses OFI |
| A3_AS_VPIN | +0.006 | 0.0002 | 68 |

**The ρ_diff = −0.80 in v9 is the HONEST measurement of the same broken OFI joint that v8 was hiding.** v8's +0.80 was an artifact; v9's −0.80 is the truth.

### Why a v9.1 with smaller ε would NOT fix ρ_diff

A v9.1 with ε ~ 1e-7 (an order of magnitude below cond_std) would fix the marginal pathology — frac_zero should land at ~0.89, conditional kurt should recover toward 55. **But it would not recover ρ_diff.** The OFI→return joint is broken by the copula transform itself, not by the dequantization band-width. Smaller ε won't reverse the wrong sign on the OFI slope.

The deeper limitation: **the copula approach is incompatible with preserving weak cross-feature correlations.** Real-data Spearman ρ(OFI, mid_return) = 0.071 — that's noise-level. A rank correlation that weak cannot survive the round-trip through:
1. forward Gaussianization (transforms each feature to z-space)
2. diffusion training on Gaussianized data (model learns smoothed joints with MSE loss)
3. sampling (DDIM with ε-prediction)
4. inverse CDF (back to original scale)

Each stage adds noise. A 0.071 rank correlation is below the noise floor of this pipeline. The copula's *mathematical* guarantee that monotone transforms preserve Spearman ρ is theoretically true and practically irrelevant when ρ ≈ 0 to begin with.

### Reframed §6 outcome map (post-hoc)

| Predicted outcome | Actual classification |
|---|---|
| 3/3 primary clear (ρ_diff ≥ +0.40, no swap, kurt 400-1000) | DID NOT OCCUR for any v8/v9/v9_b run |
| ρ_diff = +0.40 but kurt < 400 ("modest fix") | (v8 looked like this, but +0.80 was artifact) |
| ρ_diff = +0.20 (broken ranking) | (most prior post-v2 runs) |
| ρ_diff < +0.20 ("Phase E broke something") | **v9 / v9_b — but this is the HONEST measurement, not breakage** |

The decision tree assumed v8's +0.80 was real. With that artifact removed, the actual Phase E outcome is mode (c)/(d): the copula approach cannot deliver v2-class ranking signal, and we now know why (weak Spearman correlations don't survive the copula pipeline).

---

## 12. Final Verdict on Phase E

**Phase E is conclusively a negative result.** The copula-based approach:
- Did not recover heavy tails (v8 kurt=2.6 vs predicted 400-1000)
- Did not preserve OFI→return joint (wrong sign at every horizon, in every variant)
- Produced apparent +0.80 ρ_diff in v8 only as an artifact of return collapse
- Confirmed the artifact in v9, where dequantization restored variance and revealed the broken joint as ρ_diff = −0.80

**The methodological contribution that survives is sharper than "dequantize before copula":**

> **Copula-based generative models are inadequate for financial microstructure data because the agent-relevant cross-feature relationships are noise-level (Spearman ρ ~ 0.07) and cannot survive the Gaussianization → diffusion training → inverse CDF round-trip. The mathematical guarantee that copula transforms preserve rank correlations is meaningful only when those correlations are well above the noise floor of the diffusion pipeline. For zero-inflated data with weak joints (a common combination in market microstructure), copula-diffusion approaches will fail by construction, regardless of dequantization or other preprocessing fixes.**

This is the first systematic demonstration of this limitation we're aware of, and belongs in the report's discussion as a methodological note for anyone considering copula-based diffusion for similar data.

### Updated workstream-level position

The marginals-vs-joints story from Work6_FINAL §9 is now refined, not refuted:
- **v2** remains the only generator that delivers v2-class ranking (ρ_diff = +0.40) AND v2-class realistic marginals (kurt 122). It does so because it doesn't transform the data — it just z-scores. Joint structure is preserved by being preserved-by-default.
- **v6/v7/v7_b/v8/v9** all break the OFI joint via different mechanisms (v-prediction, EDM σ-weighting, copula). No single intervention rescues the joint while improving marginals.
- **v8's apparent +0.80 ρ_diff was a false positive** caused by return collapse hiding the broken joint.
- **The "right" answer doesn't exist within the diffusion-based generator framework we tested.** A fundamentally different approach (e.g., autoregressive event-by-event generation, or a generator trained directly on multi-feature joint targets like a copula-aware loss) would be needed to deliver both realistic marginals AND preserved joints for this kind of data.

The honest report conclusion: **the proposed approach (synthetic stress testing via diffusion-generated counterfactual LOBs) faces a structural limitation we have characterized but not resolved.** v2 is the practical recommendation IF a user must choose one of our generators today; the field-level conclusion is that the framework needs a different kind of generator for this class of problem.
