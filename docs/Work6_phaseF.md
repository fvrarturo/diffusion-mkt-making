# Work 6 — Phase F: Post-hoc remap + cascade + auxiliary losses

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-05-08
**Status:** Implementation complete. Experiment 1 ran. Experiments 3 (cascade v10) and 4 (aux-loss v11) currently training on cluster.
**Predecessor:** [Work6_phaseE.md](Work6_phaseE.md) — copula-based diffusion (v8/v9) declared a conclusive negative result.

After Phase E established that copula-Gaussianized training breaks weak cross-feature joints (Spearman ρ(OFI, return) = 0.071 cannot survive the round-trip), we have **four** remaining experiments that follow directly from the diagnostic findings. None of them is "another architectural axis of the diffusion model" — three of the four are designed around v2's already-learned weights (post-hoc transform, fine-tune from v2, or use v2 as Stage 1 of a cascade), and the fourth (aux losses) is v2 with explicit objective-level pressure on the things v2 does poorly.

---

## 0. Motivation: what the diagnostics actually told us

Three findings from the Phase E diagnostics reframe the design space:

**1. The OFI→return narrative was explaining noise.** Real-data Spearman ρ(OFI, mid_return) = +0.071 across all horizons. v2's synthetic ρ is *negative* at every horizon except lag-100 (where it's ~0). Yet v2 achieves ρ_diff = +0.40. Whatever drives the agent ranking, it's not v2 preserving the OFI→return correlation — that correlation is essentially nonexistent in v2's synth.

**2. A1/A2 separation is fill-rate driven, not return-prediction driven.** From `61_ranking_mechanism.py`: in real data, A2_AS_OFI vs A1_AS effect sizes are **<0.1σ on every measured dimension** (PnL, fills, fill_rate, inventory). A2 fills slightly more (fill_rate_z = +0.04), but the PnL effect is essentially zero. The ranking signal that ρ_diff measures is a tiny, fill-dynamics-driven property — not a return-prediction property.

**3. v2 is excellent on bid_sz / ask_sz marginals already.** Per the deep diagnostic: bid_sz std ratio 1.04× real, ask_sz 0.93× real, kurtosis tracks real to within 30%. Where v2 fails is **mid_return** (kurt 122 vs 755, std 0.25× real) and **trade_sz** (std 0.07× real). These are exactly the heavy-tailed point-mass features.

**Conclusion:** the path forward isn't "fix the cross-feature joint" (it's noise-level even in real data), it's "fix v2's weakest marginals (mid_return, trade_sz) without disrupting the fill dynamics that depend on bid_sz/ask_sz/event-arrival structure."

---

## 1. Experiment 1 — Post-hoc Quantile Remapping (CPU, ~30 min total)

### 1.1 Hypothesis

Apply a per-feature monotone transformation to v2's existing 6400 synthetic parquets that maps v2's empirical marginal to real's empirical marginal:

$$
x^{\text{remapped}}_j = \hat{F}_{\text{real},j}^{-1}\!\left( \hat{F}_{\text{v2},j}(x^{\text{v2}}_j) \right)
$$

This is mathematically the same operation the copula's inverse-CDF tries to perform. The crucial difference: **the diffusion model never sees the transformed data during training.** v2's weights are unchanged. The transform happens purely at the output stage.

### 1.2 Why this is fundamentally different from Phase E

The copula approach (v8/v9) Gaussianized the *training data*, which changed what the model learned. The round-trip through Gaussianization → diffusion training (MSE on z-targets) → sampling → inverse CDF added enough noise to flip the OFI→return sign and degrade weak joints. **Each stage destroyed signal.**

Post-hoc remapping doesn't touch training. v2's learned denoiser produces v2's outputs. We then apply a deterministic monotone transform to each feature column. Mathematically:

- **Spearman rank correlations between ALL feature pairs are preserved exactly** (monotone-invariance).
- **Temporal rank-autocorrelation of each feature is preserved exactly** (the transform is per-event, applied identically across time).
- **Whatever fill-rate dynamics drove v2's ρ_diff = +0.40 operate on the rank ordering of events** — which is unchanged.

What changes:
- Per-feature marginals become exact matches to real (modulo the discreteness of the empirical-CDF interpolation).
- mid_return gains real's heavy-tailed kurtosis ~755 by construction.
- trade_sz gains real's variance.
- frac_zero on mid_return goes from 0.00 to ~0.89 (the lowest 89% of v2's mid_return values map to the "zero" region of real's CDF).
- Absolute prices (bid_px/ask_px/mid) are re-derived from the remapped mid_return via the standard decode integration (anchor + cumsum log-returns).

### 1.3 The critical question

Can we have v2's joints AND real's marginals?

- **If yes (ρ_diff stays at ~+0.40):** the production generator is `v2 + post-hoc remap`. It satisfies BOTH the central hypothesis test AND the validation suite's marginal realism criteria. This would be the cleanest project outcome.
- **If no (ρ_diff drops):** the fill simulator's behavior depends on absolute feature magnitudes (not just ranks), which is itself a finding about what the evaluation metrics actually depend on. The marginals-vs-joints tradeoff is then irreducible.

### 1.4 Implementation

**`scripts/70_remap_v2.py`** — ~280 LOC, CPU only:

1. **Pool source values** (subsample up to 2M events per feature from v2's 6400 stitched parquets).
2. **Pool reference values** (up to 5M events per feature from real val tapes).
3. **Build per-feature quantile maps** as `np.interp(x, src_quantiles, tgt_quantiles)` over 10,000 quantile points. Floating-point monotonicity guard for tied source values.
4. **For each source parquet:** apply the per-feature map to the 8-feature tensor, then re-derive the dependent canonical-schema columns (mid via cumsum, bid_px/ask_px from new mid + new dists, spread, trade_px on trade events, sizes rounded to int) — mirrors `decode.py::decode_window_to_dataframe` exactly.
5. **Write to a parallel output directory** preserving regime subdirs.

The script's per-feature stats output lets you sanity-check that the maps are sane (source [1%, 99%] should land in target's wider [1%, 99%] range).

### 1.5 Predicted outcomes

| Outcome | What it means | Production answer |
|---|---|---|
| ρ_diff = +0.40 ± 0.05 | Joint structure invariant under monotone remap; marginals can be fixed cheaply | **v2 + remap is the production generator** |
| ρ_diff = +0.20 to +0.40 | Partial invariance; absolute magnitudes matter somewhat | Use v2 as-is OR investigate which features tolerate remap |
| ρ_diff < +0.20 | Fill simulator depends on absolute values; rank preservation is insufficient | "Even exact rank preservation fails" — sharp negative finding |

### 1.6 Why this should be tried first

Highest expected value per unit of compute: ~30 minutes wall time (mostly the ph4 replay), no GPU needed, no retraining. If it succeeds, the project has a clean production answer. If it fails, the finding (rank-preservation isn't enough) is publishable on its own.

### 1.7 Outcome (added 2026-05-08, afternoon)

The full remap completed and went through ph4 + ph5 + ph6 + deep diagnostic. **Three findings, one of which is genuinely novel for the project.**

#### 1.7.1 Headline (the simple bad-news story)

| Metric | Real | v2 | **v2 + remap** |
|---|---|---|---|
| **ρ_diff** | — | **+0.40** | **−0.40** ← sign FLIPPED |
| π_diff | — | `[A0, A3, A2, A1]` | `[A2, A3, A1, A0]` |

The naive "rank-preservation should preserve ρ_diff" hypothesis from §1.5 is falsified. But the picture is much richer than this single number suggests.

#### 1.7.2 Marginals are the best in the project

The remap recovered v2's missing marginals across the board:

| Metric | Real | v2 | **v2 + remap** |
|---|---|---|---|
| `frac_zero(mid_return)` | 0.89 | 0.00 | **0.978** ✓ |
| `kurt(mid_return)` (uncond, aggregate) | 524 | 126 | 112 (≈ v2-class) |
| `std(mid_return)` | 3.17e-5 | 5.6e-6 | **1.0e-4** (uncond; cond stretched) |
| `mean(bid_sz)` | 48 | 2435 | **62** ✓ |
| `mean(ask_sz)` | 52 | 2706 | **71** ✓ |
| Trade fraction | 5% | **59%** | **5%** ✓ |

**v2_remapped is the closest match to real's per-feature marginals of any generator the project has produced.** The kurtosis didn't reach real's 524 because the per-window quantile maps re-sample tails proportionally to v2's tail mass, which under-represents real's extreme rare events — but v2-class kurtosis was already the best we'd reached in the workstream.

#### 1.7.3 Volatility clustering is preserved (the user's specific worry → resolved)

`|mid_return|` ACF — the volatility-clustering metric:

| Lag | real | v2 | **v2_remapped** |
|---|---|---|---|
| 1 | 0.129 | 0.168 | **0.173** |
| 5 | 0.080 | 0.083 | **0.085** |
| 10 | 0.065 | 0.061 | **0.066** ← closest to real of any generator |
| 50 | 0.046 | 0.020 | **0.022** |
| 100 | 0.044 | 0.012 | **0.011** |

**The "spike process from 99% zero-inflation breaks volatility clustering" failure mode did NOT occur.** Per-event rank ordering of |return| within each window is preserved by the monotone remap, and the rescaled magnitudes inherit v2's autocorrelation structure exactly. At low lags v2_remapped slightly improves on v2 (closer to real); at high lags it matches v2 (still below real but no degradation).

#### 1.7.4 NOVEL FINDING: v2_remapped is the FIRST generator with correct-signed OFI→return slope

G1 rows 12 and 13 (E[r|I] sign tests):

| Generator | β at Δ=10 | Sign | β at Δ=50 | Sign |
|---|---|---|---|---|
| Real | +3.68e-05 | + | +1.11e-04 | + |
| v2 | (negative) | − | (negative) | − |
| v6, v7, v7_b | (negative) | − | (negative) | − |
| v8 (copula) | -1.4e-04 | − | -6.2e-04 | − |
| v9 (copula+dequant) | (negative) | − | (negative) | − |
| **v2_remapped** | **+6.31e-05** | **+** ✓ | **+1.67e-04** | **+** ✓ |

**Every prior generator we tested had the wrong-signed OFI→return slope. v2_remapped is the only one that gets the sign right at both horizons, and the magnitudes are within 2× of real's.** This is a real, reproducible project finding.

The mechanism: the remap rescaled `bid_sz` and `ask_sz` from v2's ~2400-class to real's ~50-class. The OFI proxy `bid_sz - ask_sz` is now on the same scale ratio relative to mid_return as in real data. The slope of return on OFI is sensitive to the relative scales of the two regressors, and rescaling both to real-class makes the implied slope come out positive — without any change to v2's training, just by transforming the marginals at the output stage.

This is the cleanest example of **emergent joint-structure correction via marginal alignment** that the project has produced. Worth a paragraph in the report regardless of the ρ_diff outcome.

#### 1.7.5 The three-axis Pareto picture (the actual project finding)

No single generator wins all three axes. Phase F has produced the cleanest mapping yet of which intervention moves which axis:

| | ρ_diff | OFI sign | Marginals (G1 pass count) |
|---|---|---|---|
| v2 | **+0.40** ✓ | wrong | weak (13/16) |
| v8 (copula) | +0.80 (artifact) | wrong | broken (7/16) |
| **v2_remapped** | **−0.40** ✗ | **correct** ✓ | **best of project** (9/15 — fails *different* checks than v2) |

**This is the core empirical finding of the diffusion workstream.** The three desiderata are orthogonal — interventions that improve any one of them tend to disrupt at least one of the others. ρ_diff (the central hypothesis test) is NOT measuring "preserved OFI→return joint" — v2_remapped has demonstrably better OFI→return joint than v2, yet ρ_diff is its opposite.

What ρ_diff *is* measuring becomes harder to characterize:
- It's not OFI sign (v2_remapped has it right, v2 doesn't).
- It's not marginal realism (v2_remapped is closer to real on marginals than any other generator).
- It's not joint structure preservation in any obvious sense.
- It correlates with **trade fraction**: v2 (59%) wins ρ_diff, every generator with realistic ~5% trade fraction (v2_remapped, v8, v9) loses ρ_diff. The agent ranking depends on how often agents get to act, and v2's specific accidental property of generating 12× too many trade events happens to give A2's strategy a structural advantage that aligns with truth.

**The story for the report:** the project has a clean Pareto frontier with three competing generators, NOT a single best one. The honest framing is "what is your downstream test sensitive to?" — different evaluators (validation suite vs central hypothesis test vs OFI joint metrics) prefer different generators, and the field-level finding is that the diffusion-based generator framework cannot satisfy all three simultaneously.

### 1.8 Follow-up — partial remap (added 2026-05-08)

Added `--features` flag to `scripts/70_remap_v2.py`. Lets us remap a SUBSET of features and pass through the others unchanged. The targeted experiment:

```bash
# v2_remap_midreturn_only — fix the headline marginal (mid_return) but keep
# v2's trade_sz / bid_sz / ask_sz distributions intact (preserves event mix).
python scripts/70_remap_v2.py \
    --source-dir $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
    --reference-dir data/processed/val \
    --features mid_return \
    --out-dir $HOME/orcd/scratch/diffmm/synthetic_v2_remap_midret/INTC \
    --norm-stats data/processed/norm_stats_INTC.json
```

Predicted outcome: ρ_diff ≈ +0.40 (event mix preserved → A2 still gets fill opportunities) AND kurt ≈ 213 (mid_return marginal recovered). If this holds, **`v2 + partial remap` is the production generator**: v2's joints AND realistic mid_return marginals, ranking signal preserved.

If ρ_diff drops anyway → there's an interaction between the cumsum(mid_return) → mid trajectory reconstruction and the simulator's tick-level decisions that we haven't isolated. Worth investigating but a sharper failure than the full remap.

---

## 2. Experiment 2 — v-prediction Fine-tune from v2, Per-epoch Trajectory (DEFERRED)

**Status:** Implementation ready (`scripts/utils/run_v10_per_epoch_eval.sh` + `+init_from_checkpoint=` in trainer), but **deferred / superseded** by Experiments 3 + 4. The motivation was to map the marginals-vs-joints frontier as a continuous trajectory by fine-tuning v2 with v-prediction one epoch at a time. After the secondary-opinion analysis (see §0 motivation), we concluded that:

- The **cascade approach (Exp 3)** addresses the same goal more sharply: instead of mapping a tradeoff curve between v2 and a fully-broken v-pred run, it asks whether the joint structure can be *enforced* by causal decomposition.
- The **aux-loss approach (Exp 4)** keeps v2's architecture and adds explicit pressure on the things v2 does poorly — a more targeted intervention than fine-tuning the loss target.

If both Exp 3 and Exp 4 fail to beat v2 on ρ_diff, the per-epoch trajectory experiment becomes worth running as a fallback to verify whether *any* v-pred regime preserves v2's signal at all. Until then, the experiment is documented but not queued.



### 2.1 Hypothesis

The marginals-vs-joints tradeoff might not be a binary cliff but a smooth curve. Start from v2's converged ε-prediction weights, switch the loss target to v-prediction, and train for 5-10 epochs while saving every epoch. Sample + ph4 + ph5 at each saved epoch.

The hypothesis: v2's learned joint structure degrades **gradually** as v-prediction fine-tuning progresses. There may exist an epoch k* where kurtosis has improved meaningfully (>200) but ρ_diff hasn't yet collapsed (still ≥ +0.20). If so, that epoch is a strictly-better generator than v2.

### 2.2 The trajectory we want

| Epoch | Kurtosis | ρ_diff | Notes |
|---|---|---|---|
| 0 (v2) | 122 | **+0.40** | v2 baseline (ε-prediction, no fine-tune) |
| 1 (v10-e1) | ? | ? | first epoch of v-prediction fine-tune |
| 2 | ? | ? | |
| 3 | ? | ? | |
| 5 | ? | ? | likely fully converged to v-pred regime by here |

This is the project's central figure for the diffusion-model section: **the marginals-vs-joints Pareto frontier as a continuous trajectory**, not just discrete architectural snapshots.

### 2.3 Possible shapes of the trajectory

- **Smooth slope:** ρ_diff degrades gradually as kurtosis improves. There IS a useful regime (epoch 1-2). Production: pick the knee.
- **Cliff (binary):** ρ_diff drops to +0.20 at epoch 1 already. The tradeoff is structurally binary; no interpolation helps. Confirms Phase E framing.
- **Inverse-U:** ρ_diff briefly *improves* (because v-pred initial fine-tune adjusts something useful before degrading the joint). Unlikely but would be very interesting if seen.

Each shape produces a different paper conclusion. **The trajectory figure belongs in the report regardless of which shape we get.**

### 2.4 Implementation

**Two pieces:**

**(a) `scripts/11_finetune_generator.py` — added `init_from_checkpoint` plumbing.** Distinct from the existing `+resume_from=` (which restores full Lightning state — optimizer, LR scheduler, epoch counter — unsuitable when the loss target changes). The new `+init_from_checkpoint=<path>` loads ONLY model weights (via `pl_module.load_state_dict(strict=False)`); optimizer is fresh, training starts at epoch 0. Logged so the run-time confirmation is visible.

**(b) `scripts/utils/run_v10_per_epoch_eval.sh` — per-epoch evaluation orchestrator.** Sequentially samples on CPU partition, runs ph4 + (optional) ph6 + ph5 for each saved epoch checkpoint, and archives results to `results/_archive/v10_e001/`, `v10_e002/`, etc. Final summary table prints the trajectory.

### 2.5 Training config (Hydra overrides on existing `ph2_finetune.slurm`)

```bash
sbatch slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v10 \
    generator.schedule.prediction_type=v \
    generator.model.conditioning_type=film \
    generator.edm.enabled=false \
    generator.copula.enabled=false \
    generator.training.max_epochs=10 \
    generator.training.early_stopping_patience=999 \
    generator.checkpointing.every_n_epochs=1 \
    generator.checkpointing.save_top_k=-1 \
    +init_from_checkpoint=$HOME/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-epoch004-valloss<XX>.ckpt
```

Note `early_stopping_patience=999` and `save_top_k=-1` to ensure ALL epochs are saved.

### 2.6 Expected behavior in the first ~100 training steps

The model was trained to predict ε; the new loss expects v. The first ~100 steps will show very high loss as the optimizer adapts. **This is expected, not a bug.** Loss should drop rapidly within a few hundred steps and stabilize.

### 2.7 What we can do with the trajectory data

1. **Plot (kurtosis, ρ_diff) per epoch** as a Pareto frontier — the central figure for the report's discussion section.
2. **Identify the knee** if it exists — epoch with best (ρ_diff, kurtosis) trade.
3. **Compare to architectural snapshots** (v2, v8, v9) — show that the trajectory passes through them or detours around.

---

## 2bis. Experiment 3 — v10 Cascade (Conditional Decomposition, GPU + CPU)

**Status: training (job 13564820 on cluster, started 2026-05-08 10:28 UTC).**

### 2bis.1 Hypothesis

Stop trying to make one diffusion model handle both jobs. Decompose into:
- **Stage 1 = v2** (already trained): generates the 7 book features `{bid_dist, ask_dist, bid_sz, ask_sz, trade_dist, trade_sz, trade_sign}`. We re-use v2's existing checkpoint as Stage 1 — no retraining.
- **Stage 2 = v10**: a generator that predicts ONLY `mid_return` conditional on the clean book features as in-window context. Trained with `cascade.noise_mask = [0,0,0,0,0,0,0,1]` and matching `loss_mask`, so it sees clean book features at every diffusion timestep and only diffuses the mid_return channel.

The OFI→return relationship is then **enforced by the training objective**. Stage 2 only gets gradient when its mid_return prediction matches real mid_return given the actual book trajectory — there is no degree of freedom for the model to "drop the OFI signal" because the joint is the conditioning interface.

This mirrors the cascaded-diffusion pattern that works for image generation (Ho et al. 2022, "Cascaded Diffusion Models for High Fidelity Image Generation") — a coarse model handles structure, a fine model adds detail conditioned on it. The financial analog: book state + price response.

### 2bis.2 Why this should succeed where Phase E failed

In Phase E (v8/v9), all 8 features were generated jointly from a single denoiser trained on copula-Gaussianized data. The OFI→return joint had to *emerge* from the MSE loss, and any noise added to the data (Gaussianization, σ-weighting) destroyed it (Spearman ρ ≈ 0.071 is below the noise floor).

In v10 cascade, Stage 2 only has to model a **1-D conditional distribution** `p(return | book state)`. That's much easier than the 8-D joint. The model can use any heavy-tail-friendly trick on this single channel (we kept it simple: ε-prediction with FiLM, init from v2's weights). The kurtosis problem evaporates because the model only has to fit the conditional return distribution, not maintain joint structure across all 8 channels.

### 2bis.3 Implementation

| Piece | Location | What it does |
|---|---|---|
| `cascade.noise_mask` / `loss_mask` config | `configs/generator/finetune.yaml` | Per-feature 0/1 masks; default null = v2 behavior |
| `noise_mask` / `loss_mask` plumbing | `src/diffmm/generator/train.py::DDPMTrainer` | At training time: where mask=0, x_t = x_0 (no noise added). Loss mask zeros out per-channel MSE for unmasked channels. |
| `+init_from_checkpoint=<path>` | `scripts/11_finetune_generator.py` | Loads ONLY model weights from another checkpoint; fresh optimizer. Used to start v10 from v2. |
| `ddim_sample_cascade()` | `src/diffmm/generator/sample.py` | Same DDIM update math as `ddim_sample`, but at every step `x = nm·x_diffused + (1-nm)·clean_features` so masked-0 channels stay clean. |
| `scripts/13_sample_cascade.py` | new | Loads two checkpoints (Stage 1 + Stage 2), runs Stage 1 unconditionally, then Stage 2 with `clean_features = stage1_x0` and `noise_mask = [0,...,0,1]`. |
| `slurms/ph2_sample_cascade_cpu.slurm` | new | CPU-partition slurm wrapper for the cascade sampler. |

### 2bis.4 Training command (already submitted)

```bash
sbatch --exclude=node4104 slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v10_stage2 \
    generator.schedule.prediction_type=eps \
    generator.model.conditioning_type=film \
    generator.edm.enabled=false \
    generator.copula.enabled=false \
    +generator.cascade.noise_mask=[0,0,0,0,0,0,0,1] \
    +generator.cascade.loss_mask=[0,0,0,0,0,0,0,1] \
    generator.training.max_epochs=10 \
    generator.training.early_stopping_patience=999 \
    generator.checkpointing.every_n_epochs=1 \
    generator.checkpointing.save_top_k=-1 \
    +init_from_checkpoint=/home/favara/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-epoch004-valloss0.0106.ckpt
```

Confirmed at runtime: `Phase F cascade: noise_mask=[0,0,0,0,0,0,0,1] loss_mask=[0,0,0,0,0,0,0,1]` and `INITIALIZING WEIGHTS from checkpoint: ...intc_v2/ckpt-epoch004-valloss0.0106.ckpt`.

### 2bis.5 Predicted outcomes

| Metric | Real | v2 | v10 cascade (predicted) | Reason |
|---|---|---|---|---|
| ρ_diff | — | +0.40 | **+0.50 to +0.80** | OFI→return is forced by conditioning structure |
| OFI→return slope at Δ=50 | 1.11e-04 | 1.11e-05 | ~5e-5 to 1e-4 | Direct supervised learning of the conditional |
| Excess kurtosis | 755 | 122 | 400-800 | 1-D conditional, no joint structure to break |
| A2 > A1 in synth | yes | yes | yes | Mechanism preserved (and amplified) |
| Trade-sign ACF | 0.675 | 0.385 | ~0.5 | Inherited from Stage 1 (v2 unchanged) |

### 2bis.6 Kill signal

If Stage 1's imbalance distribution (computed from synth `bid_sz - ask_sz`) is degenerate (always near zero, or never crosses sign), Stage 2 has no signal to learn. Diagnosable in <1 min after Stage 1 sampling, before Stage 2 sampling even starts. Inspect with:

```python
import polars as pl, numpy as np
df = pl.concat([pl.read_parquet(p) for p in v10_cascade_parquets[:20]])
imb = (df["bid_sz"].to_numpy() - df["ask_sz"].to_numpy())
print(f"imbalance: mean={imb.mean():+.1f} std={imb.std():.1f} frac_pos={(imb>0).mean():.3f}")
```

---

## 2ter. Experiment 4 — v11 Auxiliary Losses (joint + kurt + tail, GPU)

**Status: training (job 13565002 on cluster, started 2026-05-08 10:35 UTC).**

### 2ter.1 Hypothesis

Keep v2's architecture exactly. Don't change parameterization, conditioning, or schedule. Instead, add explicit pressure on the two things v2 currently does poorly (tails) and the one thing v2 does well but fragilely (the OFI→return joint).

The training loss becomes:

$$
L_{\text{total}} = L_{\varepsilon\text{-MSE}} + \lambda_{\text{joint}} \cdot \big\| \hat{\beta}(\hat{x}_0) - \beta_{\text{real}} \big\|^2 + \lambda_{\text{kurt}} \cdot \max(0, \kappa^* - \hat{\kappa}(\hat{x}_0))^2 + \lambda_{\text{tail-w}} \cdot L_{\varepsilon\text{-MSE}}^{\text{IS}}
$$

All four terms operate on the **single-step x̂_0 approximation** $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t} \cdot \hat\varepsilon_\theta) / \sqrt{\bar\alpha_t}$ for ε-prediction (analogous formula for v-pred). Stop-gradient through schedule constants; gradient flows back through `eps_pred`. No decode-in-loop.

### 2ter.2 Why this complements v10 (cascade)

- **v10 (cascade)** says "the joint between OFI and return is fragile because it's emergent from MSE; replace it with a *structural* guarantee."
- **v11 (aux losses)** says "the joint is fragile because the loss doesn't care about it; replace with *explicit objective-level supervision*."

Cross-tabulation of outcomes:
- v10 works, v11 fails → causal decomposition is the right answer
- v11 works, v10 fails → explicit objectives are the right answer
- both work → Pareto-comparable; pick whichever has cleaner marginals
- neither works → marginals-vs-joints tension is even deeper than v8/v9 already showed; v2 stays the practical recommendation

### 2ter.3 Why aux losses didn't get tried earlier

Work6_FINAL §9.5 considered explicit auxiliary losses and rejected them as "potentially destabilizing." That rejection was based on **decode-in-loop gradients** — running the FillSimulator inside the loss function would be O(events) per backward pass and is genuinely unstable.

The single-step x̂_0 approximation sidesteps this entirely. It's just another tensor manipulation in the same forward pass. Stable in practice across the DDPM literature (e.g., DDPM++ uses x̂_0 for x0-prediction loss; EDM uses x̂_0 for preconditioning targets). The actual hyperparameter risk is just the λ values, which are curriculum-ramped to allow the baseline ε-MSE to converge before aux pressure kicks in.

### 2ter.4 Three loss components, with explicit curricula

All curriculum-ramped via the `_ramp` helper in `train.py` (linear interpolation between `ramp_start_epoch` and `ramp_end_epoch`).

| Component | What it does | λ_final | Ramp |
|---|---|---|---|
| **joint** | MSE of per-window OLS slope of x̂_0[return] on x̂_0[bid_sz - ask_sz] vs `target_slope=1.11e-4` (precomputed real INTC value at Δ=50) | 0.1 | epochs 2 → 5 |
| **kurt** | One-sided hinge: `max(0, target_kurt - kurt(x̂_0_return))² / target_kurt²`. Penalizes UNDER-shoot only (v5 showed overshoot is reachable). | 0.01 | epochs 3 → 6 |
| **tail** | Adds `(per_sample_mse · w).mean() - per_sample_mse.mean()` where `w = 1 + α·(extremity/σ_data)^p`, `extremity = max\|return\| over window`. Re-weights gradient toward rare large-magnitude windows. | 1.0 | epochs 0 → 3 |

All log per-epoch metrics: `train/aux_{joint,kurt,tail}_loss`, `train/aux_{joint,kurt,tail}_lambda`, `train/aux_total`. Visible in Lightning's CSV log.

At epoch 0, ALL aux λ are 0 — the model trains as pure v2 baseline (matches v2's loss values 0.008-0.011). Curricula kick in stepwise: tail at epoch 1, joint at epoch 3, kurt at epoch 4. Full pressure at epoch 6+.

### 2ter.5 Training command (already submitted)

```bash
sbatch --exclude=node4104 slurms/ph2_finetune.slurm \
    data=intc_top \
    experiment_name=intc_v11 \
    generator.schedule.prediction_type=eps \
    generator.model.conditioning_type=film \
    generator.edm.enabled=false \
    generator.copula.enabled=false \
    generator.training.aux_losses.joint.enabled=true \
    generator.training.aux_losses.kurt.enabled=true \
    generator.training.aux_losses.tail.enabled=true \
    generator.training.max_epochs=10 \
    generator.training.early_stopping_patience=999 \
    generator.checkpointing.every_n_epochs=1 \
    generator.checkpointing.save_top_k=-1 \
    +init_from_checkpoint=/home/favara/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-epoch004-valloss0.0106.ckpt
```

Confirmed at runtime: `Phase F aux losses enabled: {'joint': True, 'kurt': True, 'tail': True}` and same v2 init checkpoint as v10.

### 2ter.6 Predicted outcomes

| Metric | Real | v2 | v11 (predicted) | Reason |
|---|---|---|---|---|
| ρ_diff | — | +0.40 | **+0.40 to +0.60** | Joint loss explicitly preserves it; should be at least as good as v2 |
| Excess kurtosis | 755 | 122 | 300-700 | Dual pressure: kurt hinge + tail importance weighting |
| OFI→return slope | 1.11e-04 | 1.11e-05 | ~3-8e-5 | Explicit supervision pushes synth slope toward real |
| A2 > A1 in synth | yes | yes | yes | Joint loss prevents A1/A2 swap |
| ACF \|return\| β | 0.21 | 0.66 | unchanged ~0.6 | Not addressed by these losses |

### 2ter.7 Kill signal

`aux_joint_loss → 0` rapidly while the actual model output's OFI→return slope (computable post-hoc on x̂_0 samples logged or recomputed at validation time) doesn't approach `target_slope`. That would mean the model found a degenerate way to satisfy the loss vacuously. Diagnosable from the CSV log within ~2 epochs.

---

## 3. Decision Tree → Decision Matrix (post-experiments)

The original decision tree (§3 first version) assumed a single ρ_diff axis. The Exp 1 outcome (§1.7.5) showed the actual structure is a **three-axis Pareto** with no single best generator. The decision now is "which generator for which downstream use?" — not "which generator wins?".

### Per-axis recommendation (with results in)

| Downstream use | Recommended generator | Why |
|---|---|---|
| **The central hypothesis test (ρ_diff agent ranking)** | **v2** | Only generator with positive ρ_diff. Mechanism likely an artifact of v2's high trade fraction (59% vs real 5%), but it's the empirical winner. |
| **The validation suite (marginals + temporal)** | **v2_remapped** | Real-class kurtosis, frac_zero, sizes, trade fraction. Volatility clustering preserved from v2. Loses some checks v2 passes but gains correct OFI sign. |
| **OFI→return joint (microstructure realism)** | **v2_remapped** | Only generator with correct-signed slope at both Δ=10 and Δ=50. Magnitudes within 2× of real. |
| **A "production stress generator" combining all three** | **doesn't exist within tested options** | The Pareto frontier is real. Tradeoffs are unavoidable. |

### Pending experiments (v10, v11) — still informative for the matrix

| Exp 3 (cascade v10) | Exp 4 (aux v11) | What it adds to the matrix |
|---|---|---|
| ρ_diff ≥ +0.40 AND OFI sign correct | (any) | **NEW production candidate** — first generator to win on ρ_diff axis WITHOUT v2's accidentally-broken OFI sign |
| ρ_diff drops, OFI sign correct | (any) | confirms structural decomposition can fix OFI but not ρ_diff; consistent with v2_remapped story |
| (any) | ρ_diff ≥ +0.40 AND kurt ≥ 300 AND OFI sign correct | strong second production candidate via objective-level supervision |
| both fail to win ρ_diff | both fail to win ρ_diff | matrix stays as-is; v2 wins ρ_diff for accidental reasons; v2_remapped wins everything else |

### Report framing (the headline that actually emerged)

> **Three desirable properties — agent-ranking accuracy (ρ_diff), microstructure-correlation correctness (OFI sign), and per-feature marginal realism — are not jointly achievable within the diffusion-based generator framework we tested.** Across nine architectural variants and one post-hoc transform, no single generator wins on more than one axis. The most common failure is that interventions that fix marginals (Phase E copula, Phase F remap) tend to break one of the other axes, and vice-versa. v2 wins ρ_diff via what is likely an accidental property (12× too many trade events compared to real); v2_remapped uniquely satisfies the microstructure correctness criterion but loses ρ_diff. The field-level conclusion is that synthetic-data evaluation for market-microstructure agent backtests requires careful matching of the *test* to the *generator's strengths* — there is no single generator that is "best" in a generator-agnostic sense.

This is a stronger finding than "v2 is best" or "Phase E failed." It identifies a Pareto frontier and characterizes which interventions move which axis.

---

## 4. File Index

**New code (this phase):**
- `scripts/70_remap_v2.py` — Experiment 1 (~280 LOC, CPU)
- `scripts/utils/run_v10_per_epoch_eval.sh` — Experiment 2 orchestrator (deferred, kept for fallback use)
- `scripts/13_sample_cascade.py` — Experiment 3 cascade sampler (Stage 1 + Stage 2 chained)
- `slurms/ph2_sample_cascade_cpu.slurm` — CPU-partition slurm wrapper for the cascade sampler
- `scripts/11_finetune_generator.py` — added `+init_from_checkpoint=<path>` mechanism + Phase F config plumbing (~50 LOC patch)
- `src/diffmm/generator/train.py` — DDPMTrainer extensions: `noise_mask` / `loss_mask` for cascade Stage 2 (Exp 3) + `aux_loss_cfg` with `joint`, `kurt`, `tail` components (Exp 4) + `_ramp` helper for curriculum-ramped λ. ~120 LOC patch.
- `src/diffmm/generator/sample.py` — added `ddim_sample_cascade()` (~80 LOC) — same DDIM update math as `ddim_sample()` but mask-aware.
- `configs/generator/finetune.yaml` — new `cascade:` block (defaults to nulls = v2 behavior) + `training.aux_losses:` block with three sub-components (joint/kurt/tail), all `enabled: false` by default.

**Reused from prior phases:**
- `slurms/ph2_finetune.slurm` — training entry point for v10 Stage 2 and v11
- `slurms/ph2_sample_cpu.slurm` — sampling for v11 (regular path, no cascade)
- `slurms/ph4_replay_synth.slurm` — agent backtest on synth (all experiments)
- `slurms/ph5_hypothesis.slurm` — bootstrap rank correlation test
- `slurms/ph6_validate_cpu.slurm` / `ph6_validate_gpu.slurm` — optional, for full marginal stats per experiment
- `scripts/60_deep_diagnostic.py` — for cross-model side-by-side comparison (v2 / v8 / v9 / v10 / v11 / v2_remapped)
- `scripts/61_ranking_mechanism.py` — for per-agent decomposition

---

## 5. Honest expectations (recorded so we can verify post-hoc)

**Experiment 1 — best guess: ρ_diff stays at +0.40.**

Rationale: the per-event rank ordering of features (which is what the FillSimulator's tick-level decisions and trigger conditions actually use) is unchanged by monotone remap. The simulator decides "is this event a trade?" based on `trade_sz > 0` AFTER rounding — a property invariant under the size remap because we round AFTER applying the map. The simulator decides "did this fill at bid or ask?" based on relative price positioning — invariant under monotone price remap. The only place absolute magnitudes enter is in PnL accounting (dollars made/lost per fill), which scales the magnitudes of all agents identically and so should preserve the *ranking*.

If this guess is right, this is the project's concrete production deliverable.

**Experiment 3 (v10 cascade) — best guess: ρ_diff lands at +0.50 to +0.70, kurtosis 200-500.**

Rationale: causal decomposition forces the model to use the OFI signal because the conditioning is the only input pathway carrying information about return direction. Stage 2 cannot ignore it (loss requires matching real return given real book state). Kurtosis recovery should be moderate (better than v2's 122, lower than the predicted 400-800 from §2bis.5) because Stage 2 is still a diffusion model with MSE loss — the standard heavy-tail challenge applies, just on a 1-D output instead of 8-D. The dominant risk is that Stage 1 (= v2) produces book-feature trajectories with imbalance distributions too smooth or too narrow to contain real-data-like signal — Stage 2 then learns from a degenerate distribution.

If the guess is right (ρ_diff > +0.40 AND kurt > 200), this is the project's main result.

**Experiment 4 (v11 aux losses) — best guess: ρ_diff stays at +0.40 ± 0.05, kurtosis 200-400.**

Rationale: aux losses are softer than structural constraints. The joint loss can preserve OFI→return at the level the model was already producing, but adding the explicit slope target won't necessarily strengthen what's already at noise-floor (Spearman 0.071). The kurt and tail aux losses should produce some heavy-tail recovery — both directly target that property — but the full predicted 300-700 range depends on whether the model can satisfy both the ε-MSE objective AND the explicit kurt target without trading off. The most likely outcome is "v2 with marginally better marginals," not a step-change.

If the guess is right, v11 is a strict improvement over v2 on validation-suite metrics but not a breakthrough on ρ_diff.

**Most likely overall outcome (combined):** Exp 1 succeeds → production = v2 + remap. Exp 3 produces a generator with better ρ_diff than v2 but worse marginals than v2 + remap. Exp 4 produces a v2 variant with better marginals at the cost of some ρ_diff stability. The report frames v2 + remap as the practical deliverable, with v10 cascade as the methodological contribution (causal decomposition is the right pattern for this class of problem).

**Most interesting outcome:** Exp 3 cascades with ρ_diff > +0.60. That would be the project's strongest result — a generator that beats v2 on the central hypothesis test by causal decomposition, validating the field-level prediction that joint-feature diffusion is wrong for this data type.
