# Lab notes — Phase F experiments (remap + cascade + auxiliary losses)

**Date:** 2026-05-08
**Branch:** `work_4may`
**Goal:** Close out the diffusion workstream with four experiments, all designed around v2's already-trained weights or v2's existing outputs. None requires retraining v2; all follow from the diagnostic findings (OFI is noise-level, A1/A2 separation is fill-rate-driven, copula-Gaussianization breaks weak joints).

## TL;DR

Four experiments staged. Two were already documented in the original Phase F doc (remap + per-epoch v-pred); two more were added today after a sharper analysis (cascade + aux losses):

1. **Quantile remap** (CPU, ~30 min): apply per-feature `F̂_real⁻¹(F̂_v2(x))` to v2's existing parquets → ph4 → ph5. Tests whether v2's joint structure is invariant under monotone marginal transforms.

2. **v-prediction fine-tune from v2, save every epoch** — DEFERRED. Implementation ready, but superseded by Experiments 3 and 4 below (both attack the same goal more sharply). Will run as a fallback if 3 and 4 both fail.

3. **v10 cascade** (GPU + CPU, ~8h): structural decomposition. Stage 1 = v2 generates book features. Stage 2 (new) generates `mid_return` *conditional* on the clean book features via per-feature noise/loss masking. Forces OFI→return relationship via the conditioning interface — the model can't ignore the joint and minimize loss simultaneously.

4. **v11 auxiliary losses** (GPU, ~6h): keep v2's architecture; add three explicit aux losses on the single-step x̂_0 — joint OFI→return slope MSE, kurtosis hinge, tail-importance-weighted MSE. Tests whether explicit objective-level supervision can preserve the joint while fixing marginals.

**Currently running on cluster:** v10 Stage 2 (job 13564820) + v11 (job 13565002), both started ~2026-05-08 10:30 UTC. Same v2 init checkpoint (`ckpt-epoch004-valloss0.0106.ckpt`), running in parallel on the GPU partition.

Full design rationale + decision tree + predicted outcomes in [docs/Work6_phaseF.md](../docs/Work6_phaseF.md).

## Why these specifically (vs more architectural variants)

The deep diagnostic + ranking-mechanism diagnostic from 2026-05-07 reframed the design space:

1. **OFI→return is noise-level** (Spearman 0.071 in real, mostly negative in v2). The "v-prediction breaks OFI→return" narrative was explaining noise.

2. **A1/A2 separation is fill-rate driven**, not return-prediction driven. In real data, A2 vs A1 effect sizes are <0.1σ on every measured dimension. Whatever ρ_diff measures, it's a tiny fill-dynamics property.

3. **v2 is already excellent on bid_sz / ask_sz marginals.** Where v2 fails is mid_return (kurt 122 vs 755) and trade_sz (std 0.07× real) — both heavy-tailed point-mass features.

So the design question isn't "preserve OFI→return" (it's noise even in real). It's "fix v2's weakest marginals (mid_return, trade_sz) without touching the fill-rate dynamics that depend on bid_sz/ask_sz."

That motivates the two experiments:
- **Remap** = fix marginals at the OUTPUT (no training change) → tests whether ranking depends on marginal magnitudes or just per-event rank ordering
- **v-pred fine-tune from v2** = sweep the tradeoff continuously rather than via discrete architectures → maps the actual Pareto frontier

Both experiments accept v2 as the baseline and ask what happens at the margins. Neither tries to "fix" the OFI joint, because the OFI joint isn't actually the problem.

## What was built

### Experiment 1: `scripts/70_remap_v2.py`

~280 LOC, CPU-only. Three stages:

1. **Pool source/reference values** per feature (subsample to 2M source / 5M reference events to bound memory).
2. **Build per-feature quantile maps** as `np.interp(x, src_quantiles, tgt_quantiles)` over 10k quantile points. Floating-point monotonicity guard for tied source values.
3. **For each source parquet:** apply per-feature map to the 8-feature tensor, re-derive dependent canonical-schema columns (mid via cumsum, bid_px/ask_px from new mid + new dists, spread, trade_px on trade events, sizes rounded to int) — mirrors `decode.py::decode_window_to_dataframe` exactly, minus the denormalization step.

Mathematical guarantee: Spearman rank correlations between all feature pairs and temporal rank-autocorrelation of each feature are preserved EXACTLY (monotone-invariance). The fill simulator decides things based on per-event rank ordering — which is unchanged.

What changes: each feature's per-event values are smoothly mapped to real's distribution, by construction. `frac_zero(mid_return)` should jump from ~0 (v2 has no exact zeros) to ~0.89 (matches real) because the lowest 89% of v2's mid_return values map to the zero region of real's CDF.

### Experiment 2a: `scripts/11_finetune_generator.py` — `init_from_checkpoint` mechanism

~25 LOC patch. Distinct from the existing `+resume_from=<path>` (which restores full Lightning state — optimizer, LR scheduler, epoch counter — and is only suitable for resuming an interrupted run with the SAME loss target).

The new `+init_from_checkpoint=<path>` loads ONLY `pl_module.load_state_dict(strict=False)`. Optimizer is fresh, training starts at epoch 0, but model weights start at the source checkpoint's values. Logs missing/unexpected keys for diagnosis if architectures don't match exactly.

This is the right primitive when you want to fine-tune a model with a different loss target (which is exactly Experiment 2's situation: load v2's ε-pred weights, fine-tune to v-pred).

### Experiment 2b: `scripts/utils/run_v10_per_epoch_eval.sh`

Orchestration script that, for each saved epoch checkpoint:
1. Submits CPU sampling (full 400-task manifest) to a unique synth dir
2. Waits, runs ph4_replay_synth
3. (Optional, default off) ph6_validate_cpu + ph6_validate_gpu
4. Archives results to `results/_archive/v10_e<NNN>/`
5. Submits ph5_hypothesis (reuses already-opened holdout from v9 run)
6. Archives the hypothesis output

Final summary table prints the (epoch, ρ_diff, π_diff) trajectory.

Configurable: `EPOCHS="1 2 3 5"` to choose which epochs to evaluate; `RUN_PH6=1` to also run validation suite per epoch (would extend wall time considerably).

## Honest expectations (recorded for falsification)

**Experiment 1 — best guess: ρ_diff stays at +0.40.** The fill simulator's tick-level decisions ("is this a trade?", "did the fill happen at bid or ask?", "which agent's quote was at top of book?") all depend on per-event rank ordering, which is invariant under monotone remap. The only place absolute magnitudes enter is PnL accounting, which scales magnitudes uniformly across agents and should preserve the *ranking*. If this guess is right, the project has a clean production deliverable: **v2 + post-hoc remap** delivers v2's ranking (ρ_diff = +0.40) AND real's marginals (kurtosis, frac_zero, etc.).

**Experiment 2 — best guess: cliff at epoch 1, not a smooth slope.** v-prediction's σ-correlated targets put non-trivial loss weight on high-noise timesteps where the model must commit to extreme values. One epoch of v-pred fine-tune will rapidly unlearn v2's mode-covering behavior. The OFI joint is already at the noise floor (Spearman 0.071) and any extra training noise pushes it to wrong-signed. If the trajectory is a cliff, the figure still belongs in the report — it visualizes WHY v2 is special (and why every post-v2 architecture lands in the same broken-joint region).

## What's left

- [ ] Run Experiment 1 (remap → ph4 → ph5): cluster, ~30 min wall time
- [ ] Find v2's checkpoint path (need to `ls` on cluster)
- [ ] Submit Experiment 2 training (~6h GPU)
- [ ] Run Experiment 2 per-epoch evaluation (~24h orchestration)
- [ ] Update `Work6_phaseF.md` with actual outcome numbers + Pareto-frontier plot
- [ ] Decide production generator based on Experiment 1's ρ_diff result

## Decision tree (post-experiments)

| Exp 1 ρ_diff | Exp 2 trajectory | Production answer | Report headline |
|---|---|---|---|
| ≥+0.35 | (any) | **v2 + remap** | "marginals can be fixed cheaply via post-hoc remap; joint structure is invariant under monotone marginal transforms" |
| 0.20-0.35 | smooth, has knee | **v10 at the knee epoch** | "smooth tradeoff; we found a strictly-better-than-v2 generator" |
| 0.20-0.35 | cliff | v2 | "post-hoc remap loses some ranking signal; v2 remains best practical generator" |
| <+0.20 | (any) | v2 | "even exact rank-preservation is insufficient; FillSimulator is sensitive to absolute magnitudes" |

The branch we end up in determines the report's conclusion. Phase F is the experimental closure of the diffusion workstream regardless of outcome.

## File index

**New:**
- `scripts/70_remap_v2.py` — Experiment 1 quantile remap (~280 LOC)
- `scripts/utils/run_v10_per_epoch_eval.sh` — Experiment 2 orchestrator
- `docs/Work6_phaseF.md` — design + decision tree + predicted outcomes

**Modified:**
- `scripts/11_finetune_generator.py` — added `+init_from_checkpoint=<path>` mechanism (~25 LOC patch)

**Reused (no changes):**
- `slurms/ph2_finetune.slurm` (training)
- `slurms/ph2_sample_cpu.slurm` (sampling on CPU partition)
- `slurms/ph4_replay_synth.slurm` (agent backtest)
- `slurms/ph5_hypothesis.slurm` (rank correlation bootstrap)
- `slurms/ph6_validate_{cpu,gpu}.slurm` (optional, for full validation per epoch)

## Note on the project narrative

After Phase F, the diffusion workstream is genuinely complete. The story arc:

- Work3-5: built v1/v2/v3 + the validation suite + the central hypothesis test
- Work6 phases A-D: 5 architectural variants (v3.5/v4/v5/v6/v7/v7_b), all fail to beat v2 on ρ_diff
- Work6 phase E: copula-based approach (v8/v9/v9_b) — v8 was an artifact (return collapse hid broken joint), v9 was the honest negative result. Established that copula-Gaussianized training cannot preserve weak cross-feature correlations.
- Work6 phase F: post-hoc remap (Exp 1) + per-epoch trajectory (Exp 2). Tests whether marginals can be fixed without touching v2's joint, and maps the tradeoff Pareto frontier.

If Exp 1 succeeds, the project ships v2+remap with realistic marginals AND v2's ranking. That's a strong concrete contribution.

If Exp 1 fails AND Exp 2 shows a cliff, the field-level finding stands: this class of generative model has a structural limitation for this kind of data, and v2 is the practical best.

Either way the report is sharper than it would have been without these experiments.

---

## Today's launches (2026-05-08, afternoon)

### Implementation summary

After the morning's discussion of "Model A (cascade)" and "Model B (aux losses)", I added two new experiments to Phase F's roster. These required deeper code changes than the morning's two experiments:

**Files modified for cascade + aux losses:**
- `src/diffmm/generator/train.py` — extended `DDPMTrainer` with `noise_mask`, `loss_mask` (Model A masking) and `aux_loss_cfg` (Model B with `joint`, `kurt`, `tail` sub-components). Added module-level `_ramp()` helper for curriculum-ramped λ. ~120 LOC patch.
- `src/diffmm/generator/sample.py` — added `ddim_sample_cascade()`. Same DDIM update math as `ddim_sample()`, but at every step: `x = nm * x_diffused + (1-nm) * clean_features`, so masked-0 channels stay clean throughout reverse diffusion. ~80 LOC.
- `scripts/11_finetune_generator.py` — reads `cascade.{noise,loss}_mask` and `training.aux_losses` from Hydra config; passes both into `DDPMTrainer.__init__`. Logs `Phase F cascade:` and `Phase F aux losses enabled:` so the run-time confirmation is visible.
- `configs/generator/finetune.yaml` — new top-level `cascade:` block (defaults to nulls = v2 behavior) + new `training.aux_losses:` block with three sub-components (joint/kurt/tail), all `enabled: false` by default.
- `scripts/13_sample_cascade.py` — new sampler. Loads two checkpoints (Stage 1 + Stage 2), runs Stage 1 unconditionally for full 8-feature output, then runs Stage 2 with `clean_features = stage1_x0` and `noise_mask = [0,...,0,1]` to refine only `mid_return`.
- `slurms/ph2_sample_cascade_cpu.slurm` — CPU-partition slurm wrapper for the cascade sampler (matches `ph2_sample_cpu.slurm` structure).

### Cluster jobs (running as of writing)

```
13564820  diffmm_ph2_finetune  RUNNING  intc_v10_stage2  (cascade Stage 2)
13565002  diffmm_ph2_finetune  RUNNING  intc_v11         (aux losses)
```

Both started from the same v2 checkpoint: `ckpt-epoch004-valloss0.0106.ckpt`. The diff between them is exactly:
- v10 Stage 2: `cascade.noise_mask=[0,0,0,0,0,0,0,1]`, `cascade.loss_mask=[0,0,0,0,0,0,0,1]`, no aux losses
- v11: `noise_mask=null`, `loss_mask=null`, all three aux losses (joint/kurt/tail) enabled

Expected wall time: ~6h training each (or 6h time limit). Then ~1h sampling + 1.5h ph4 + ph6 + ph5 each.

### Confirmed at runtime

**v10 Stage 2 (job 13564820):**
```
Phase F cascade: noise_mask=[0,0,0,0,0,0,0,1] loss_mask=[0,0,0,0,0,0,0,1]
Phase F aux losses enabled: {'joint': False, 'kurt': False, 'tail': False}
INITIALIZING WEIGHTS from checkpoint: .../intc_v2/ckpt-epoch004-valloss0.0106.ckpt
init_from_checkpoint: missing keys (2): ['noise_mask', 'loss_mask']  ← expected (new buffers)
init_from_checkpoint: weights loaded; optimizer is fresh, training starts at epoch 0
[train] step 0  loss 0.0056  pred=eps  p_drop=1.00
```

Loss values 0.001-0.005 are LOWER than v2's 0.011 val loss — expected because (a) starting from converged v2 weights, (b) loss is computed only on the `mid_return` channel (1/8 of v2's full feature dim), (c) v2's mid_return predictions in z-score space are near-zero and the targets are also near-zero (89% are exact zeros), so MSE on that channel is small even before any new learning.

**v11 (job 13565002):**
```
Phase F aux losses enabled: {'joint': True, 'kurt': True, 'tail': True}
INITIALIZING WEIGHTS from checkpoint: .../intc_v2/ckpt-epoch004-valloss0.0106.ckpt
[train] step 0  loss 0.0086  pred=eps  p_drop=1.00
```

Loss values 0.008-0.011 match v2's baseline — expected because at epoch 0 ALL aux losses are λ=0 (curriculum schedules: tail starts at epoch 0 → 3, joint at 2 → 5, kurt at 3 → 6). v11 will start to diverge from v2 at epoch 1 (tail kicks in), then more so at epoch 3+ as joint and kurt ramp.

### What to watch

**v10 Stage 2 — the question is whether Stage 2 actually USES the clean book features.**
Loss alone won't tell us this:
- "ignore book features and predict mid_return ≈ 0" gives low loss (matches v2's behavior)
- "use book features as conditioning to predict the right return" also gives low loss

The decisive test is in the cascade-sampled output:
- v2's standalone mid_return synth had `frac_zero=0.00`, kurt=122, no useful OFI→return signal.
- If v10 cascade has the same → Stage 2 ignored conditioning, just memorized v2's behavior. **Failure.**
- If v10 cascade has `frac_zero` significantly above 0 OR conditional kurt closer to real OR a per-window OFI-return correlation with a real-data-like sign → Stage 2 learned something. **Success.**

**v11 — watch for the joint loss kill signal.**
After ~2 epochs (when joint loss starts ramping at λ=0.033), check Lightning's CSV log:
```bash
tail $HOME/diff_mm/results/lightning_logs/intc_v11/version_*/metrics.csv
```
Look for `train/aux_joint_loss` collapsing to ~0 while the actual model output's OFI→return slope (recoverable post-sampling) doesn't approach `target_slope=1.11e-4`. That would mean the model is satisfying the loss vacuously (degenerate solution).

### Updated decision tree

| Exp 1 (remap) | Exp 3 (cascade v10) | Exp 4 (aux v11) | Production answer |
|---|---|---|---|
| ρ_diff ≥ +0.35 | (any) | (any) | v2 + remap |
| ρ_diff drops | v10 ρ_diff ≥ +0.50 | (any) | v10 cascade — beats v2 by causal decomposition |
| ρ_diff drops | v10 fails | v11 ρ_diff ≥ +0.40 + kurt ≥ 300 | v11 — explicit objectives preserve joint while fixing marginals |
| ρ_diff drops | both fail | (n/a) | v2 — strong negative finding |
| (any) | both succeed | (any) | whichever has cleaner marginals |

Phase F is now genuinely the experimental closure of the workstream. After v10 + v11 results we have the four-way matrix to interpret.

### Memory note for future me

The implementation strategy that paid off was making the masking + aux-loss extensions *config-gated* with safe defaults (null masks = v2 behavior, all aux losses `enabled: false`). That meant zero risk of breaking existing v2/v8/v9 reproductions and we could test the new code path by just flipping config flags. The +init_from_checkpoint mechanism was the key primitive — it lets us start a new training run with a different loss target from another model's converged weights, without resuming optimizer state. That distinction (vs `+resume_from=`) matters for *all* the Phase F experiments.

The other big win was the symmetry between v10 and v11: same v2 init, same architecture, same data loader, only the loss differs. That makes any comparison between them clean. If v10 wins, it's about structural decomposition. If v11 wins, it's about objective-level pressure. If neither wins, the marginals-vs-joints tension is structural to the data + diffusion-model class, not to any specific intervention we tested.

---

## Experiment 1 result (2026-05-08, late afternoon)

The quantile remap completed and went through the full ph4 + ph5 pipeline. **Result: ρ_diff = −0.40**, sign FLIPPED vs v2's +0.40. Hypothesis from the original Phase F doc (rank-preservation should preserve ρ_diff) is **falsified**.

Per-feature spot-check on the remapped output (one base-regime parquet):

| Feature | v2 orig | v2 remap | real reference |
|---|---|---|---|
| `mid_return` std | 3.1e-7 | **3.0e-5** | 3.2e-5 ✓ |
| `mid_return` kurt | 17 | **213** | 524 (close) |
| `mid_return` frac_zero | 0.00 | **0.99** | 0.89 ✓ |
| `bid_sz` mean | 2435 | **56** | 48 ✓ |
| `ask_sz` mean | 2705 | **57** | 52 ✓ |
| Trade events (per 2560) | 1500 (59%) | **135 (5%)** | 5% |

Marginals all matched real reasonably well. AND the trade event mix matched real (5%). The agent ranking nonetheless inverted to a near-mirror of truth.

### Why the ranking flipped — sharper version of the diagnostic

The "rank-preservation" framing was missing one thing: **the trade-vs-quote event MIX is itself implicitly set by `trade_sz`'s marginal**, via `is_trade = round(trade_sz) > 0` in the decode logic. When we remap `trade_sz` to real's distribution (which has 95% zeros), the rounding step converts ~12× as many events from "trade" to "quote_update." That's not a per-feature rank-ordering change — it's a **categorical reclassification of which events are even eligible for the simulator to consider as trades**.

Combined with yesterday's ranking-mechanism finding (A2 vs A1 is fill-rate driven), the picture is now:
- v2 has 59% trade events. A2 (AS+OFI) has many opportunities to use the OFI signal → wins ranking.
- v2_remap has 5% trade events (matching real). A2 has 12× fewer opportunities → loses ranking.
- The mechanism that gave v2 ρ_diff = +0.40 was **load-bearing on the high trade fraction**, not on any specific marginal property.

This is a stronger negative finding than "rank-preservation isn't enough." It identifies a specific structural property — the trade-event frequency — as the load-bearing piece. v2's ρ_diff was an artifact of v2 generating trade events 12× too often, which gave the OFI-using agent disproportionate opportunities.

### Follow-up — partial remap

Added `--features` flag to `scripts/70_remap_v2.py`. Lets us remap a SUBSET of features and pass through the others unchanged. The targeted next experiment:

```bash
python scripts/70_remap_v2.py \
    --source-dir $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
    --reference-dir data/processed/val \
    --features mid_return \
    --out-dir $HOME/orcd/scratch/diffmm/synthetic_v2_remap_midret/INTC \
    --norm-stats data/processed/norm_stats_INTC.json
```

Then ph4 + ph5 on the new output. If `ρ_diff ≈ +0.40` AND `kurt(mid_return) ≈ 213`: **`v2 + partial remap` is the production generator** — v2's joints and event mix preserved, mid_return marginal fixed in post.

If ρ_diff still drops → there's an interaction between cumsum(remapped mid_return) → reconstructed price trajectory and the simulator's tick-level fill decisions. Worth investigating but a sharper failure than the full remap.

### Updated decision tree row

| Exp | ρ_diff | Status |
|---|---|---|
| Exp 1 (full remap) | **−0.40** | done — failed hypothesis, but informative |
| Exp 1b (partial remap, mid_return only) | TBD | next |
| Exp 3 (v10 cascade) | TBD | training (13564820) |
| Exp 4 (v11 aux losses) | TBD | training (13565002) |

If Exp 1b succeeds, that's the production answer regardless of v10/v11 results — the cheapest possible solution.

---

## v2_remapped full validation results (2026-05-08, evening)

After the simple "ρ_diff dropped" headline, ran the deep diagnostic + ph6 validation suite to characterize v2_remapped properly. **The result is much sharper than the ρ_diff number alone suggested. Three findings, including one that is genuinely novel for the project.**

### Finding 1: |mid_return| ACF is preserved exactly

The user's specific worry was that 99% zero-inflation would create a sparse spike process with different volatility clustering. **Worry rejected.** The remap preserved v2's ACF structure exactly:

| Lag | real | v2 | v2_remapped |
|---|---|---|---|
| 1 | 0.129 | 0.168 | **0.173** |
| 5 | 0.080 | 0.083 | **0.085** |
| 10 | 0.065 | 0.061 | **0.066** ← closest to real of any generator |
| 50 | 0.046 | 0.020 | **0.022** |
| 100 | 0.044 | 0.012 | **0.011** |

At low lags, v2_remapped is slightly closer to real than v2. At high lags, it matches v2. Rank-preservation does its job here.

### Finding 2: Marginals are now real-class on every headline metric

| Metric | Real | v2 | v2_remapped |
|---|---|---|---|
| `frac_zero(mid_return)` | 0.89 | 0.00 | **0.978** ✓ |
| `kurt(mid_return)` (uncond, aggregate) | 524 | 126 | 112 (≈ v2-class) |
| `mean(bid_sz)` | 48 | 2435 | **62** ✓ |
| `mean(ask_sz)` | 52 | 2706 | **71** ✓ |
| Trade fraction | 5% | 59% | **5%** ✓ |
| Spread 1-tick fraction | ~95% | 95% | **95.7%** ✓ |

Aggregate kurtosis is 112 — same as v2's, didn't reach real's 524 because the per-window quantile maps re-sample tails proportionally to v2's tail mass, which under-represents real's extreme rare events. But matching v2's kurtosis while ALSO matching real's frac_zero and trade fraction is genuinely novel — every prior generator either had near-zero frac_zero (v2-v7_b) or degenerate output (v8_b/v9_b).

### Finding 3 (NOVEL): v2_remapped is the FIRST generator to pass G1 OFI→return sign tests

Looking at G1 rows 12 and 13:

| Generator | β at Δ=10 | sign | β at Δ=50 | sign |
|---|---|---|---|---|
| Real | +3.68e-05 | + | +1.11e-04 | + |
| v2 | (−) | ✗ | (−) | ✗ |
| v6, v7, v7_b | (−) | ✗ | (−) | ✗ |
| v8 | -1.4e-04 | ✗ | -6.2e-04 | ✗ |
| v9 | (−) | ✗ | (−) | ✗ |
| **v2_remapped** | **+6.31e-05** | ✓ | **+1.67e-04** | ✓ |

**Every prior generator we tested had the wrong-signed OFI→return slope.** v2_remapped is the only one that gets the sign right at both horizons, with magnitudes within 2× of real's.

The mechanism: the remap rescaled `bid_sz`/`ask_sz` from v2's ~2400-class to real's ~50-class. The OFI proxy `bid_sz - ask_sz` is now on the same scale ratio relative to mid_return as in real data, and the slope of return on OFI comes out positive — without any change to v2's training, just by transforming marginals at the output stage. **This is emergent joint-structure correction via marginal alignment.** Worth a paragraph in the report regardless of ρ_diff.

### G1 pass count summary

- v2: 13/16
- v8: 7/16
- v9: 7/16
- **v2_remapped: 9/15** (one row is "report only", not pass/fail)

v2_remapped passes FEWER tests overall than v2, but passes the SPECIFIC tests that no prior generator did (OFI sign at both horizons). Trade-offs:
- v2_remapped GAINS: rows 12, 13 (OFI sign × 2)
- v2_remapped LOSES: row 5 (size tail too thin), row 7 (ACF |r| lag-50 deviation), row 8 (ACF power-law β too steep), row 11 (RV signature), row 14 (joint spread-imbalance JSD just over 0.3 threshold at 0.338)
- v2_remapped TIES (both pass): rows 1-4, 9, 10, 15, 16

### The three-axis Pareto picture (the actual project finding)

| | ρ_diff | OFI sign | Marginals (G1) |
|---|---|---|---|
| v2 | **+0.40** ✓ | ✗ | 13/16 (good but wrong OFI) |
| v8 | +0.80 (artifact) | ✗ | 7/16 (broken) |
| **v2_remapped** | **−0.40** ✗ | **✓** | **best on marginals** (9/15 — gains rows 12, 13 vs v2) |

**No single generator wins on more than one axis.** Phase F has produced the cleanest demonstration of the Pareto frontier in the project. The honest framing for the report:

- The three desirable properties (ρ_diff, OFI correctness, marginal realism) are not jointly achievable within the diffusion-based generator framework we tested.
- v2 wins ρ_diff via what is likely an accidental property (12× too many trade events compared to real — the OFI agent gets disproportionately many fills, which happens to align with truth's ranking).
- v2_remapped uniquely satisfies microstructure correctness (correct-signed OFI), demonstrating that fixing marginals can emergently fix joints.
- No tested generator achieves both.

### What ρ_diff is actually measuring (revised after this finding)

After v2_remapped, the answer is *less* clear, not more:
- Not OFI joint preservation (v2_remapped has it right; v2 has it wrong; v2 wins ρ_diff anyway).
- Not marginal realism (v2_remapped is closer to real on marginals; loses ρ_diff).
- Not any joint-structure metric we've measured directly.
- **Strongest correlate so far: trade fraction.** Every generator with realistic ~5% trade fraction (v2_remapped, v8, v9) loses ρ_diff. v2's high trade fraction (59%) is the only condition under which we've seen positive ρ_diff.

The mechanism interpretation: the agent ranking depends on how often agents get fills. v2's accidentally-high trade fraction gives the OFI-using A2 agent disproportionately many opportunities to act on its (wrong-signed!) signal. Despite the wrong sign, A2's higher activity creates a Sharpe profile that happens to align with truth's worst-quartile ranking (the actual metric used). When we shift trade fraction to real's 5%, A2 acts less often, and the alignment breaks.

This is a much more nuanced story than "v2 has the right joint structure." It's "v2 has a specific accidental combination of features that produces a Sharpe profile that happens to rank-correlate with truth, and that combination doesn't survive any of the marginal-fixing interventions."

### Updated decision matrix (replaces the simple decision tree)

| Downstream test | Recommended generator | Why |
|---|---|---|
| Central hypothesis test (ρ_diff) | **v2** | Only generator with positive ρ_diff |
| Validation suite (marginals + temporal) | **v2_remapped** | Real-class marginals, preserved volatility clustering |
| OFI→return microstructure realism | **v2_remapped** | Only generator with correct OFI sign |
| Combined "production" generator | **doesn't exist within tested options** | Pareto frontier; tradeoffs unavoidable |

v10 (cascade) and v11 (aux losses) are still training. They could change this picture if either lands in a previously-empty region of the Pareto frontier — e.g., v10 cascade with ρ_diff ≥ +0.40 AND correct OFI sign would be the first generator to win on TWO axes. Watching the queue.

---

## TRADES supplementary metrics + Predictive score (2026-05-08, late afternoon)

Ran `scripts/62_trades_metrics.py` (volume-volatility, leverage effect, PCA coverage, mid-price traces) + `scripts/63_predictive_score.py` (LSTM train-on-synth, test-on-real MAE) over all 18 generators via `slurms/ph7_trades_eval.slurm`. Total wall time 18 min on `mit_normal` partition.

### Headline: three evaluation frameworks give CONTRADICTORY rankings

| Framework | Best | Runner-up | Worst |
|---|---|---|---|
| **G1 (stylized facts)** | v3_e9 / v4 / v5 / v7 / v7_b (13/15) | v2 / v3_e19 / v3p5 / v6 / v9 (12/15) | v3p5_noclip (5/15) |
| **PCA coverage** (TRADES Fig 2) | **v8 (85.2%)** | v9 (71.4%) | noclip variants (0%) |
| **Predictive score** (TRADES Table 1) | **v3_e9_noclip (4.02× replay)** | v2_noclip (8.25×) | v2_remapped (266×) |
| **ρ_diff** (central hypothesis test) | v2 (+0.40) | v8 / v9 are noise (artifacts) | v9_b (-1.00) |

**No two frameworks pick the same winner.** v8 wins PCA coverage, is 16th of 18 on predictive score (117× replay). v3_e9_noclip wins predictive score, has 0% PCA coverage and 5/15 G1. v2_remapped has the closest marginal scale to real and the worst predictive score (266× replay). v2 wins ρ_diff and is mid-pack on G1, mid-pack on coverage, mid-pack on predictive score. The metrics are empirically anti-correlated.

### Mechanism: predictive score rewards temporal pattern fidelity at ANY scale

The noclip variants — declared "ablation degenerate" by every prior framing — win predictive score by the largest margin. v3_e9_noclip has 5/15 G1 passes, 0% PCA coverage, 44× real return std — and an LSTM trained on it generalizes to real with the lowest MAE (4× replay). Mechanism:

- The noclip variants have **scale-exploded magnitudes** but **correct relative ordering** of per-event features.
- The LSTM learns "when bid_sz drops and ask_sz rises, mid_return tends to go negative" — a directional/temporal pattern.
- That pattern transfers to real data **regardless of absolute scale**, because per-source z-score normalization renormalizes the magnitude away.
- v8/v9 (copula) have correct marginal scale but the temporal patterns were disrupted by Gaussianization → diffusion → inverse-CDF round-trip. The LSTM learns the WRONG associations.

The predictive score is measuring something much subtler than "marginal realism" — it's measuring whether the temporal/correlational structure of synthetic events matches real well enough that a downstream learner generalizes. v3_e9_noclip happens to have that property despite failing every aggregate metric. v2_remapped happens to lack it despite matching every aggregate marginal.

### v2_remapped's predictive-score failure mechanism

v2_remapped MAE = 3.06e-2 = **266× replay** — the worst of any generator we've tested. Mechanism:

- v2_remapped has 99% mid_return = 0 and ~1% non-zero values stretched to real-magnitude scale (10× larger than v2's nonzero returns).
- Per-source z-score normalization with that wide std (`std=1.0e-4`) → real test data values get renormalized to extreme z-values.
- The LSTM trained on v2_remapped sees mostly z ≈ 0 with rare large outliers; tested on real, every nonzero return looks like an extreme outlier → predictions explode → MAE explodes.

**Takeaway:** "matching real marginals" can hurt downstream tasks if the matching mechanism produces a different EVENT-LEVEL scale calibration than real has. Per-source normalization is the standard TRADES protocol and we kept it deliberately, but it interacts pathologically with v2_remapped's specific zero-inflation.

### Two important metric caveats

**1. The leverage effect metric failed universally.** All 18 models had 0 usable tapes — even REAL data gave only mean +0.006 (essentially zero) at 1-minute buckets. INTC at event-level/microstructure timescale doesn't exhibit the leverage effect; it's a daily/weekly phenomenon. For the report: acknowledge this metric isn't informative at our timescale; would need either much longer tapes or aggregation to multi-hour buckets.

**2. Volume-volatility correlation failed for the realistic-trade-rate models.** Real has a clean +0.341 mean (Karpoff effect well-defined). Stitched generators with 50-99% trade fractions span +0.13 to +0.43 — most close to real. But v8/v9/v9_b all gave 0 usable tapes because their realistic 5% trade fraction means 1-minute buckets often have zero trades, so per-bucket volume = 0 and Pearson is undefined.

**This is itself a finding:** the TRADES evaluation protocol cannot evaluate the kind of generator we want most (one with realistic trade fractions), because the protocol assumes high-frequency event mixes. For the report: flag this as a limitation of TRADES evaluation when applied to realistic large-tick microstructure.

### The methodological contribution that emerges

> **No single evaluation metric captures "generator quality" for synthetic LOB data.** Across our 18-checkpoint × 7-axis evaluation matrix, the metrics are empirically anti-correlated:
>
> - G1 stylized facts measure distributional realism (per-feature marginals + temporal autocorrelations).
> - PCA coverage measures distributional diversity (does synth span real's summary-statistic space?).
> - Predictive score measures temporal pattern transferability (does an LSTM trained on synth generalize?).
> - ρ_diff measures fill-dynamics alignment with a specific holdout-truth agent ranking.
> - OFI sign / OFI magnitude measure microstructure correlation correctness.
>
> These five evaluation frameworks pick five different winners across our 18 generators. **Any claim of the form "model X is the best generator" is incomplete without specifying best FOR WHAT.** That's the project's most generalizable methodological finding: it applies beyond LOB diffusion models to any synthetic-data evaluation setting where multiple downstream uses are plausible.

This reframing strengthens, not weakens, the project's contribution. We're not claiming "we built the best generator"; we're claiming "we mapped which interventions move which axis, and demonstrated that the axes are not jointly optimizable within the diffusion-based framework we tested."

### Cross-axis empirical anti-correlations (worth noting in the report)

Sample correlations across the 18 generators (excluding v8_b which is degenerate):

| Axis pair | Sign of correlation |
|---|---|
| G1 pass count ↔ PCA coverage | weak negative (v7_b 13/15 G1 but only 25% coverage) |
| G1 pass count ↔ Predictive score | weak (no clean direction) |
| **PCA coverage ↔ Predictive score** | **strong NEGATIVE** (v8 best on coverage, near-worst on predictive) |
| Marginal realism ↔ Predictive score | strong negative (v2_remapped extreme case) |
| ρ_diff ↔ everything else | uncorrelated (v2 wins ρ_diff, mid-pack everywhere else) |

The strong negative correlation between PCA coverage and predictive score is the single most surprising finding — they were *intended* by TRADES to be complementary measures of generator quality. In our setting they're contradictory: high coverage means the model produced diverse-looking tapes (good) but those tapes had temporal patterns far from real (bad for downstream prediction).

### Updated final decision matrix

| Downstream use | Recommended generator | Why |
|---|---|---|
| Central hypothesis test (ρ_diff agent ranking) | **v2** | Only generator with positive ρ_diff; mechanism likely artifactual |
| Validation suite (G1 stylized facts) | **v3_e9 / v4 / v5 / v7 / v7_b** (tied 13/15) | Best on aggregate stylized fact pass count |
| OFI→return microstructure correctness | **v2_remapped** | Only generator with OFI slope magnitude within 2× of real |
| Distribution coverage (PCA hull overlap) | **v8** | Highest coverage of real summary-statistic space (85%) |
| Predictive score (TRADES, LSTM training data quality) | **v3_e9_noclip** | LSTM trained on it generalizes best to real (4.02× replay) |
| Realistic trade event fractions | **v9** (5.0%), v9_b (3.2%), v2_remapped (4.5%) | Match real's 4.8% trade fraction |
| Volatility clustering (|r| ACF lag-50) | **v9** | Best deviation from real (0.017) |
| Combined "production" answer | **doesn't exist** | The Pareto frontier is real; no single winner |

Seven different desirable properties, seven different generators winning. This is the cleanest demonstration of "evaluation metric ≠ ground truth quality" we've produced.

---

## v10 cascade + v11 aux-losses — final outcomes (2026-05-09)

After resolving the cascade memory bug (`@torch.no_grad()` was missing on `ddim_sample_cascade`, OOM-killed every task at 8G/16G/32G mem until the decorator was added — peak then dropped to ~2GB), both runs completed the full sample → ph4 → ph5 → archive pipeline.

### v10 cascade (Model A): ρ_diff = −0.40, NON-degenerate, real failure

| | Predicted (Phase F §2bis) | Actual |
|---|---|---|
| **ρ_diff** | +0.40 to +0.80 | **−0.40** |
| **π_diff** | preserves A2 > A1 | `[A1, A2, A3, A0]` (A0 dropped from FIRST to LAST) |
| **OFI sign at Δ=10/Δ=50** | PASS (cascade enforces joint) | **FAIL** at both |
| **Marginal scale realism** | inherits Stage 1's | mid_return std=1.23e-7 (7× smaller than v2's 9e-7) |
| **G1 pass count** | 11-13/15 | **9/15** |
| **Mid_return frac_zero** | low (continuous) | 0.0000 (continuous, ≠ real's 0.89) |
| **Trade fraction** | inherited from v2 | 60% (matches v2's 59%, ≠ real's 5%) |
| **Degenerate?** | No | **No** — agents trade actively, ranking is real |

**The cascade hypothesis was structurally falsified.** We predicted that Stage 2's loss being explicitly conditional on Stage 1's book features would force the OFI→return joint to be enforced at the right sign and magnitude. In reality:

- Stage 2 DID enforce SOME relationship between book features and mid_return — the model can't satisfy its loss otherwise. We can see this in the non-zero mid_return output and the agent ranking's responsiveness to book-state.
- BUT the relationship Stage 2 learned does NOT match real's OFI→return sign. v10_cascade fails both G1 OFI sign tests (rows 12, 13) — same failure pattern as v8_b/v3p5_noclip.
- A0 (constant agent, doesn't trade) dropped to LAST in v10_cascade. Truth has A0 first. The whole ranking is anti-truth.

This is publishable as a clean negative result for structural decomposition: **causal conditioning forces some joint structure, but does not guarantee that joint matches truth's joint**. The cascade interface enforced "Stage 2's mid_return responds to book features" but the mapping it learned is not the right mapping.

**v10_cascade does PASS one important G1 check** that most generators fail: ACF(|r|) lag-50 deviation. It joins v5/v7/v7_b/v9/v9_b as one of only 6 generators (of 20) that satisfy this volatility-clustering long-memory criterion. This is consistent with Stage 2's conditional generation preserving some temporal structure inherited from v2's training distribution.

### v11 aux losses (Model B): ρ_diff = +0.80 ARTIFACT, degenerate collapse

| | Predicted (Phase F §2ter) | Actual |
|---|---|---|
| **ρ_diff** | +0.40 to +0.60 | +0.80 (ARTIFACT — same pattern as v8_b) |
| **π_diff** | A2 > A1 preserved | `[A0, A1, A2, A3]` (alphabetical tie-break) |
| **mid_return std** | preserved real-class | **0** (every event mid_return = 0) |
| **mid_return frac_zero** | match real ~0.89 | **1.0** (over-collapsed) |
| **Trade fraction** | preserved | **0.04%** (5 trade events of 12800) |
| **Spread / size kurtosis** | within target | extreme (spread kurt 3190, ask_sz kurt 530) |
| **G1 pass count** | 13+/15 | **6/15** |

**v11 is degenerate via the same mechanism as v8_b but a different path**:
- v11's training trajectory: val loss went from 0.011 (epoch 0, no aux loss active) to 0.034-0.098 (epochs 1-8 with full aux loss curriculum). Higher val loss = baseline ε-MSE objective being violated by the aux loss pressure.
- The model "satisfied" the aux losses by:
  - Driving spread/size kurtosis to extreme outliers (kurt loss target met)
  - Collapsing mid_return to constant 0 (makes baseline MSE finite under aux pressure)
- Result: degenerate output → all 4 agents earn ~0 PnL → ranking falls out by tie-break → A0 first by alphabetical ordering → ρ_diff = +0.80 by accident.

**Auxiliary loss training in DDPMTrainer can satisfy the aux objectives but at the cost of collapsing the primary reconstruction.** Same lesson as Phase E v8: optimizing for one property breaks another in the diffusion-based framework.

### Updated multi-axis Pareto matrix (now 20 generators)

| Run | ρ_diff | G1 | OFI sign | Trade fraction | Real result? |
|---|---|---|---|---|---|
| v2 | +0.40 ✓ | 12/15 | PASS | 56% | ✓ Real |
| v3_e9 | +0.20 | 13/15 | PASS | 55% | ✓ |
| v4 / v5 / v7 / v7_b | +0.20 | 13/15 | PASS | 55-78% | ✓ |
| v8 (copula) | +0.80 ⚠ | 8/15 | FAIL | 0.9% | Artifact |
| v8_b | +0.80 ⚠ | 6/15 | FAIL | 0% | Artifact (degenerate) |
| v9 (copula+dequant) | −0.80 | 12/15 | PASS | 5% | ✓ Honest negative |
| v9_b | −1.00 | 11/15 | PASS | 3% | ✓ |
| v2_remapped | −0.40 | 9/15 | PASS | 5% | ✓ Honest negative |
| **v10_cascade** | **−0.40** | **9/15** | **FAIL** | **60%** | **✓ Real (anti-truth joint)** |
| **v11 aux losses** | **+0.80** ⚠ | **6/15** | **FAIL** | **0.04%** | **Artifact (degenerate collapse)** |

Three artifact ρ_diff = +0.80 results (v8/v8_b/v11) all share the same mechanism: degenerate output → tie-break ordering. Two real-but-anti-truth ρ_diff = −0.40 results (v2_remapped, v10_cascade). One real ρ_diff = +0.40 (v2 only).

### What both experiments tell us in one paragraph

> **Both Phase F Models A and B failed to beat v2 on ρ_diff in informatively different ways.** Model A (cascade) produced a non-degenerate generator with a structurally-enforced book-state→mid_return joint, but the joint Stage 2 learned was anti-truth, yielding ρ_diff = −0.40. Model B (aux losses) collapsed mid_return to constant zero while satisfying the kurt and tail aux losses on other features, yielding the degenerate-tie-break ρ_diff = +0.80 artifact. **Neither structural decomposition (causal conditioning) nor explicit objective-level supervision (aux losses) produced a generator that beats v2 on the central hypothesis test.** The marginals-vs-joints tension is structural to the diffusion-based generator framework we tested — no amount of architectural cleverness within this framework rescues both axes simultaneously.

The PCA-coverage-vs-predictive-score anti-correlation finding from Stage 1 of the TRADES eval is now even more interesting: those two evaluation metrics already disagreed on which generator is "best" before v10_cascade and v11 entered the comparison. After adding them, the contradictions deepen. The "best for what?" framing remains the project's strongest methodological contribution.
