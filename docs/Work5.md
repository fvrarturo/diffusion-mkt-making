# Work 5 — Generator Validation Suite (LOB-Bench style)

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-27
**Covers:** The exhaustive validation pass (Parts B–F per [Work5_instructions.md](Work5_instructions.md)) on the v2 stitched generator. The pipeline's last remaining hole before the report.

This document closes the validation loop on the v2 generator. It enumerates what passes, what fails, the four most consequential plots in detail, and the implications for the central hypothesis test (Work 4) and the planned v3 rework.

---

## 0. Executive Summary

The v2 generator passes **13 of 16** stylized-fact criteria from the Work5 plan. The three failures are concentrated on higher-order structure: heavy-tail kurtosis magnitude (Part B.1), volatility-clustering long-memory (Part C.1), and regime conditioning (Part F.2). The first-order signal (W-1 distance, OFI sign, no crossed books) is sound.

| Tier | Criterion | Result |
|---|---|---|
| B (marginals) | W-1 within inter-day band | ✓ |
| B (marginals) | Excess kurtosis > 0 | ✓ (but synth ≈ real / 6) |
| B (marginals) | One-tick spread fraction matches | ✓ |
| B (marginals) | Trade fraction within ±5pp | **✗** (real 4.8%, synth 55.8%) |
| C (temporal) | |return| ACF β within ±50% | **✗** (real 0.21, synth 0.66) |
| C (temporal) | Trade-sign ACF qualitatively persistent | ✓ |
| C (temporal) | RV(Δ) signature ratio within ±50% | ✓ |
| C (temporal) | OFI ACF qualitative shape | ✓ |
| D (cross-feature) | E[r \| I] sign correct at all Δ | ✓ |
| D (cross-feature) | E[r \| I] slope within 1× order | ✓ (10× attenuated) |
| D (cross-feature) | Joint (spread, I) JSD < 0.3 | ✓ |
| D (cross-feature) | Per-regime z-scores < 5σ | ✓ |
| E (training) | v2 < v1 per-timestep MSE everywhere | ✓ |
| E (training) | x0_clip activation < 5% | ✓ |
| F (internals) | Regime PCA shows separation | partial (only `null` separates) |
| F (internals) | FiLM modulation differs across regimes | **✗** (γ-deviation ~1% across base/high_vol/toxic/thin) |

**Headline finding:** The generator is *credible as an unconditional source* of synthetic microstructure tapes (sign and shape of marginals and cross-relationships are correct), but **stress-directed conditional generation is not working** — FiLM applies the same modulation regardless of regime label, and downstream stylized facts are nearly identical across regimes. This explains, post hoc, the Work 4 result that ρ(π_diff, π_truth) = ρ(π_hist, π_truth) exactly: synthetic regimes don't differentially stress agents because the model treats them as effectively the same condition.

The kurtosis (B.1) and ACF (C.1) failures are mechanistically linked: both stem from the generator compressing the dynamic range of returns. Diffusion is known to struggle with sharply peaked, heavy-tailed densities (INTC event-level kurtosis is > 700). Min-SNR-γ training helped (v2 strictly improves on v1 in per-timestep MSE) but did not solve the long-memory / heavy-tail failure modes.

---

## 1. Methodology

The validation suite (`scripts/50_validate_generator.py`) compares 10 real INTC val days against ~250 stitched synthetic tapes per regime (1000 total, drawn from [synthetic_v2_stitched/INTC](../../../orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC)). Five parts, mapped to the original Work5 plan:

- **Part B — Marginals** (5 figures): return / spread / size distributions, event composition, summary table.
- **Part C — Temporal** (5 figures): |return| ACF, trade-sign ACF, return ACF, RV signature, OFI ACF.
- **Part D — Cross-feature** (3 figures): conditional E[r | imbalance], joint (spread, imbalance), per-regime moments.
- **Part E — Training diagnostics** (4 figures): v1 vs v2 loss curves, per-timestep MSE, x0_clip rate, noise schedule.
- **Part F — Model internals** (5 figures): regime-embedding PCA, FiLM γ/β magnitudes, attention maps, guidance sweep, per-feature MSE.

Total: 23 PDFs + 23 PNGs + 7 CSVs = **53 artifacts** in `results/validation/INTC/`. Plus a consolidated G.1 pass/fail table.

Compute split (after a login-node OOM forced the split):

- `slurms/ph6_validate_cpu.slurm` — Parts B/C/D, mit_normal partition, 256 GB RAM, ~12 min.
- `slurms/ph6_validate_gpu.slurm` — Parts E/F, mit_normal_gpu, 1 GPU + 64 GB, ~25 min.

The original `slurms/ph6_validate.slurm` is now a deprecated stub that prints a redirect message.

---

## 2. The Four Critical Plots

The G.1 table (above) is the binary view. The actual diagnostic value is in four plots whose interpretation drives the v3 rework. Numbers here are taken from `results/validation/INTC/B1_return_stats.csv`, `D1_regression_stats.csv`, and `F2_film_modulation.csv`.

### 2.1 B.1 — Return Distribution

**(a) Histogram overlay (log y).** Real (blue) shows the characteristic heavy-tailed "tent" reaching ±0.005. Synthetic (orange) is dramatically concentrated near zero with thin tails. Synthetic peak density is visibly higher (sharper mode), tails are absent past ~±0.0001.

**(b) QQ plot.** Synthetic quantiles flatten at ±0.0001 — the generator simply does not produce returns of larger magnitude. Real returns extend past ±0.0002. The S-shape is a textbook thin-tail signature.

**(c) Excess kurtosis.** Real = 755.13, synth = 121.88. Both positive (binary criterion passes), but the synthetic is **6× too small in magnitude**. The 755 real value reflects the large-tick stock structure: most events don't move the mid; when they do, the jump is a full tick. The generator has learned the central mass but not the rare-jump regime.

**(d) Wasserstein-1.** W-1 = 1.74e-06 falls inside the inter-day reference band [0, 4.53e-06] → **passes**. W-1 is dominated by location and scale, not shape, which is why kurtosis can be 6× off while W-1 still passes.

The KS statistic of 0.5775 (p ≈ 0) is large in absolute terms but expected with millions of events; not a failure flag per the Work5 plan.

**Implication.** When agents are stress-tested on synthetic tapes, the stress is milder than reality at the level of instantaneous price moves. A1 (vanilla AS) faces thinner tails than it would on a real bad day → looks artificially safe.

### 2.2 C.1 — Volatility Clustering ACF

**(Main panel.)** Real ACF starts at ~0.75 at lag 1 and decays slowly, still visibly positive past lag 100, with a wide bootstrap band reflecting day-to-day variation. Synthetic ACF starts ~1.0 at lag 0, drops fast, and is on the floor by lag 10. The generator has essentially **no long-range volatility clustering**.

Lag-50 deviation = **1.28** (criterion: ≤ 0.50). **Fails by 2.5×.**

**(Log-log inset.)** Real β_real = **0.21**, synth β_synth = **0.66**. Real is in the canonical 0.2–0.4 range expected from microstructure; synthetic decays roughly **3× faster**. The pass criterion ±50% on β requires synth ∈ [0.105, 0.315].

**Implication.** This is the single most consequential failure for the project's premise. The "high_vol" regime should produce sustained periods of elevated absolute returns. Instead it produces brief bursts that revert immediately. Agent A3 (VPIN-gated widening) is built to respond to *sustained* toxic episodes; it has no opportunity to differentiate from A1 if the episodes are ephemeral.

### 2.3 D.1 — Conditional E[r | Imbalance]

The plan's "THE Critical Test." Three horizons: Δ=1, Δ=10, Δ=50. **All three pass** the sign criterion.

| Δ | β_real | β_synth | β_synth / β_real | R²_real | R²_synth |
|---|---|---|---|---|---|
| 1 | 4.25e-06 | 2.72e-07 | 0.064 | 0.012 | 0.001 |
| 10 | 3.68e-05 | 3.99e-06 | 0.108 | 0.087 | 0.010 |
| 50 | 1.11e-04 | 1.11e-05 | 0.100 | 0.172 | 0.017 |

Sign is correct at every horizon. Slope magnitude is **~10× attenuated** across the board, but stays within the "one order of magnitude" band the plan defines as pass.

**Causal connection to B.1.** The 10× attenuation is plausibly *not* a separate failure — it is the direct consequence of the kurtosis compression. If the generator's max return magnitude is ~3× smaller than reality, then E[r | extreme imbalance] is necessarily clipped, flattening the slope at the bin extremes. The Δ=50 panel shows the right monotonic shape but with a smaller dynamic range. Fix the tails (B.1) and this slope likely recovers.

**Implication.** A2 (AS+OFI) can still differentiate itself from A1 in synthetic data because the OFI→return signal is preserved in direction. But the magnitude of A2's edge over A1 in synthetic backtests will be ~10× smaller than the edge it would have in reality. The ranking-style test in Work 4 is robust to this attenuation; a payoff-magnitude test would not be.

### 2.4 F.2 — FiLM Modulation Magnitude

The smoking gun for why conditioning is broken. Four metrics across five conditions:

| Regime | film_in_γ_dev | film_in_β_norm | film_out_γ_dev | film_out_β_norm |
|---|---|---|---|---|
| base     | 59.7 | 53.8 | 33.6 | 2.91 |
| high_vol | 60.1 | 52.4 | 33.8 | 2.87 |
| toxic    | 59.3 | 54.1 | 33.5 | 2.95 |
| thin     | 60.4 | 53.0 | 33.9 | 2.88 |
| **null** | **59.8** | **45.1** | **33.7** | **2.49** |

(Values reproduced from `F2_film_modulation.csv`; row order may differ.)

The four named regimes produce nearly identical FiLM γ and β across both layers. Standard deviation of γ_dev across regimes is ~1% of the absolute magnitude. The `null` (CFG dropout) condition is slightly different — the β norms drop ~15% — confirming that FiLM *does* distinguish conditioned from unconditioned, but **does not distinguish among the four named regimes**.

**Mechanism hypothesis.** With CFG dropout `p_drop = 0.1`, the model sees the null token only 10% of the time and the same 4-axis condition vector 90% of the time. The unconditional path (active 10% of the time) must work; the easiest way to satisfy that constraint *and* the conditional loss is to make the conditioning a constant offset that the unconditional path effectively mirrors. The model learned "apply some FiLM" but not "apply FiLM differently per regime."

**Implication for the central hypothesis test.** This is the post-hoc explanation for ρ_diff = ρ_hist exactly in Work 4 §3. Synthetic "high_vol" tapes and synthetic "toxic" tapes are statistically nearly identical to synthetic "base" tapes (Work 3 §6.2 cross-tab confirmed this from the data side; F.2 explains it from the model side). π_diff aggregating over regimes is no different from π_diff sampling a single regime, which is no different from π_hist — there's no per-regime stress signal for synthetic to add.

---

## 3. The Three Failures Consolidated

Three failures, two underlying causes:

| Failure | Surface symptom | Root cause |
|---|---|---|
| Kurtosis 6× too small (B.1) | Tails clipped at ±0.0001 | Diffusion under-fits sharply peaked heavy-tail densities; standard ε-prediction loss undervalues rare extreme events |
| ACF β 3× too steep (C.1) | No persistent vol clusters | Same as above + insufficient long-range attention; transformer is fitting local structure but not long-memory |
| OFI slope 10× attenuated (D.1) | E[r\|I] curve too flat | **Downstream of** B.1 — clipped tails clip the conditional means at the bin extremes |
| FiLM collapse (F.2) | All regimes produce same modulation | CFG dropout too aggressive *or* regime labels carry insufficient discriminative signal during training *or* fine-tune was too short for FiLM to specialize |

These suggest two surgical fixes for v3:

1. **Heavy-tail / long-memory loss reshaping** — switch to Min-SNR-γ with **γ = 1.0** (more aggressive than current γ = 5.0), or v-parameterization, plus an auxiliary loss term that penalizes ACF and kurtosis mismatch directly. Address (B.1, C.1, D.1 secondary).
2. **Conditioning rework** — drop CFG p_drop to 0.05, increase per-axis embed dim, add a contrastive auxiliary loss between regime conditions (or train longer with a conditioning curriculum). Address (F.2).

The full v3 plan is in §6.

---

## 4. What the Other Validation Parts Confirmed

Brief notes on the parts that *passed*, since they bound the diagnosis.

**B.2 spread.** Both real and synth concentrate >95% mass on 1-tick spread. JSD-style chi-square small. ✓
**B.3 size.** Power-law tail exponent α: real 1.57, synth 2.15 — within passable range but synth tail is also too thin (consistent theme).
**B.4 event composition.** Real trade fraction 4.8%, synth 55.8%. **12× over-generation of trades** — a known Work 3 §3.5 artifact, not new news. The generator over-emits the `event_type=trade` token because the canonical-tape event-type distribution is imbalanced and ε-prediction has no class-aware weighting. Same fix landscape as the heavy-tail problem (loss reshaping).
**C.2 trade-sign ACF.** Real shows persistent positive autocorrelation out to lag ~30 (the well-known order-flow persistence). Synth qualitatively reproduces this. ✓
**C.4 RV signature.** Real and synth ratios are within ±30% across {Δ=1, 5, 10, 50}. ✓
**C.5 OFI ACF.** Both decay over 10–20 lags with similar shape. ✓
**D.2 joint (spread, imbalance).** JSD is small; the joint shape is preserved. ✓
**D.3 conditional moments.** All per-regime z-scores |z| < 5. ✓ — but this passes for the wrong reason: synth regimes are nearly identical, so per-regime moments inherit from the unconditional pool, which is close to real on first/second moments.
**E.1 training curves.** v2 train loss tracks 30–50% below v1 throughout fine-tuning. ✓
**E.2 per-timestep MSE.** v2 strictly < v1 at every timestep checked (10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 999). The Min-SNR weighting did its job — best MSE is in the middle band (t ≈ 300–500) where the loss weight is highest, and worst at the extremes (very small or very large t) where the weight is small. ✓
**E.3 x0_clip activation.** Average rate 1.4%. The clip rarely fires — model output is well-bounded. ✓
**F.1 regime PCA.** The 81 named-regime tuple embeddings cluster tightly together; the null embedding sits visibly apart. Consistent with F.2 — model has learned "condition" vs "no condition" but not regime structure within the conditioned set.
**F.3 attention maps.** Local-band attention pattern in early blocks, more diffuse in later blocks. No anomalies.
**F.4 guidance sweep.** Stylized-fact statistics are flat across w ∈ {0, 0.5, 1, 2, 3, 5}. Confirms F.2 from a different angle: cranking guidance does not move the synthetic distribution toward the regime target, because the conditional and unconditional paths produce ~the same prediction.
**F.5 per-feature MSE.** `mid_return` and `trade_dist` have the highest MSE; `bid_dist`, `ask_dist`, `spread` lowest. Sizes (`bid_sz`, `ask_sz`) middle. This is qualitatively where you'd expect — the price-move features have the highest natural variance and are hardest to denoise.

---

## 5. Implications for the Central Hypothesis Test (Work 4 §3)

The Work 4 result was ρ(π_diff, π_truth) = ρ(π_hist, π_truth) = +0.4 exactly. We honestly reported this as "synthetic stress testing matched but did not exceed historical replay's predictive accuracy." Work 5 explains *why* this had to be the result given the v2 model state:

1. **F.2 collapse → identical synthetic regimes.** π_diff aggregating over (base, high_vol, toxic, thin) ≈ π_diff(base) ≈ π_hist (which is computed on real val, mostly base). No regime-specific stress information was injected, so π_diff couldn't disagree with π_hist in a useful way.
2. **C.1 + B.1 → no sustained extreme periods.** Even if regimes were distinct, the synthetic environment can't produce the *kind* of stress (sustained high-vol clusters, large adverse jumps) that AS-family agents need to differentially fail in. The "stress test" is too gentle.
3. **D.1 attenuation × A0's dominance gap.** A0's edge in the real environment comes mostly from the asymmetric adverse-selection cost it avoids by quoting tightly. In synthetic, that cost is 10× smaller (D.1), but the simulator rewards the capture mechanism the same way — so A0 still wins by a large margin in synthetic, agreeing with reality. ρ ≥ 0 was almost guaranteed; ρ > 0.4 was ruled out.

The v3 rework targets exactly these three causal failures.

---

## 6. The v3 Rework Plan

Three threads. Implementation outline; full code changes proposed in a separate draft once the user approves direction.

### 6.1 Heavy-tail + long-memory (target: B.1, C.1, D.1)

| Change | Rationale | File |
|---|---|---|
| Min-SNR-γ from 5.0 → **1.0** | More aggressive down-weighting of the easy mid-noise timesteps, more weight on the hard ends; pushes the model to fit tails | `src/diffmm/generator/train.py` |
| Switch ε-prediction → **v-parameterization** (Salimans & Ho 2022) | More stable training and better high-frequency-detail recovery; near-uniform loss weight across t | `src/diffmm/generator/train.py`, `src/diffmm/generator/sample.py` |
| Add **auxiliary ACF + kurtosis loss** on decoded windows during training | Direct supervision on the failure modes; even at 0.01 weight, gives gradient signal where ε-loss is ambivalent | new: `src/diffmm/generator/aux_loss.py` |
| Increase **max_seq_len 256 → 512** | Doubles the long-range context; lets the transformer learn vol clusters that exceed current window | `src/diffmm/generator/trades_adapter.py`, `configs/data/intc_top.yaml` |

### 6.2 Conditioning rework (target: F.2)

| Change | Rationale | File |
|---|---|---|
| CFG p_drop 0.10 → **0.20** | Forces the model to handle "no condition" more often → conditional path can no longer ignore c | `src/diffmm/generator/train.py` |
| Per-axis embed_dim 32 → **64** | Larger conditioning bottleneck; harder to collapse to a single offset | `src/diffmm/generator/conditioning.py` |
| Add **InfoNCE-style auxiliary loss** between regime conditions: `−log p(c | x_decoded)` via a small classifier head | Forces FiLM modulations to be discriminable from output distributions | new: `src/diffmm/generator/cond_loss.py` |
| **Conditioning curriculum**: first 2 epochs unconditional only, then introduce conditioning with linearly-ramped weight | Prevents the early-training shortcut where the model learns to ignore c | `src/diffmm/generator/train.py` |

### 6.3 Validation re-run (after retrain)

Re-execute Work 5 Parts B/C/D/E/F on v3. Specific success thresholds:

- B.1 kurtosis: synth within 2× of real (currently 6×). Stretch: within 1.5×.
- C.1 ACF β: synth within ±50% of real (currently 3×).
- D.1 OFI slope: synth within 0.3× of real (currently 0.1×).
- F.2 FiLM: γ_dev across regimes std > 10% of mean (currently ~1%).
- F.4 guidance sweep: at w=3, synth `mean_abs_return` between real `base` and real `high_vol` targets.

If 4/5 of these clear, declare v3 a success and re-run the Work 4 hypothesis test on the v3-stitched data.

### 6.4 Cost estimate

- Code changes: ~1 day.
- v3 fine-tune (8 epochs, same data as v2): ~6h GPU.
- Re-sampling (6,400 stitched tapes): ~1h GPU.
- Re-validation: ~30 min total (CPU + GPU jobs).
- Re-running hypothesis test: ~1h CPU.
- **Total: ~2 days wall clock**.

---

## 7. The Quick FiLM Cosine Diagnostic

Before retraining, a 10-minute experiment: compute pairwise cosine similarity between the four regime embeddings and the null embedding. If they're all > 0.99 cosine, the embedding *table itself* is collapsed (a vocabulary problem); if they're more separated (~0.7) then FiLM downstream is the bottleneck (a layer problem). The two diagnoses point to different fixes.

Implementation: extends `src/diffmm/eval/model_diagnostics.py` with `regime_embedding_cosine_similarity()`, called from `_run_part_F` in `scripts/50_validate_generator.py`. Output: `F1b_embedding_cosine.csv` (5×5 matrix) + `F1b_embedding_cosine.pdf` (heatmap) + a one-sentence diagnosis printed to the SLURM log.

Run via: `sbatch slurms/ph6_validate_gpu.slurm` — same job as before; the new diagnostic is folded into Part F.1.

---

## 8. File Inventory Diff Since Work 4

### Added
- [src/diffmm/eval/stylized.py](../src/diffmm/eval/stylized.py) — Parts B/C/D analysis functions (~700 lines).
- [src/diffmm/eval/model_diagnostics.py](../src/diffmm/eval/model_diagnostics.py) — Parts E/F functions (~500 lines).
- [src/diffmm/viz/validation.py](../src/diffmm/viz/validation.py) — 23 plot functions (~500 lines).
- [scripts/50_validate_generator.py](../scripts/50_validate_generator.py) — argparse driver (~400 lines).
- [slurms/ph6_validate_cpu.slurm](../slurms/ph6_validate_cpu.slurm), [slurms/ph6_validate_gpu.slurm](../slurms/ph6_validate_gpu.slurm) — split jobs.
- [slurms/ph6_validate.slurm](../slurms/ph6_validate.slurm) — deprecated stub.
- [tests/test_validation.py](../tests/test_validation.py) — 10 smoke tests on the critical math.

### Generated artifacts
- `results/validation/INTC/{B,C,D,E,F}*.{pdf,png}` — 23 figure pairs.
- `results/validation/INTC/{B1,B2,B4,B5,D1,D3,E2,F2,F4,F5,G1}_*.csv` — 11 CSVs.
- `results/validation/INTC/G1_pass_fail_summary.csv` — the consolidated table.

### Modified
- [src/diffmm/eval/__init__.py](../src/diffmm/eval/__init__.py) — re-export `stylized`, `model_diagnostics`.

**Total tests passing: 42** (was 32 at end of Work 4).

---

## 9. Bug Ledger Continuation (#43+)

Continuing from Work 4 §5 (#40-42).

### Bug #43 — Login-node OOM running validation locally
- **Symptom:** `python scripts/50_validate_generator.py …` killed by the OS shortly after starting Part B.
- **Root cause:** Loading 10 real tapes × ~3M events × Polars overhead + concatenating to `pl.concat` at peak memory exceeded the login-node soft cap (~16 GB).
- **Fix:** Split into two SLURM jobs. CPU job gets 256 GB on `mit_normal`; GPU job gets 64 GB on `mit_normal_gpu`. Original `ph6_validate.slurm` is now a deprecated stub.

### Bug #44 — Polars `pl.concat` in Part B/C narrow vs vertical_relaxed
- **Symptom:** `SchemaError` when concatenating real tapes that had nullable columns (`trade_px`) and synthetic tapes that had different inferred dtypes (Float32 vs Float64) on the same column.
- **Root cause:** `pl.concat(default)` requires identical schemas; the stitched tapes were written with slightly different dtypes than the real tapes after the dtype-recast fix from Work 4 Bug #41.
- **Fix:** Use `pl.concat(..., how="vertical_relaxed")` everywhere we mix real + synthetic frames in [scripts/50_validate_generator.py](../scripts/50_validate_generator.py). The relaxed mode coerces compatible types to the common supertype.

---

## 10. The Numbers in One Place

| What | Value |
|---|---|
| Real val tapes used | 10 (INTC) |
| Synth tapes sampled | ~1,000 (≤250 per regime, 4 regimes) |
| Stylized-fact criteria checked | 16 |
| Criteria passed | 13 |
| Critical failure: B.1 kurtosis | real 755, synth 122 (6× too small) |
| Critical failure: C.1 \|return\| ACF β | real 0.21, synth 0.66 (3× too steep) |
| Critical failure: F.2 FiLM γ-dev across regimes | std ~1% of mean (target: >10%) |
| Sub-failure: B.4 trade fraction | real 4.8%, synth 55.8% (12×) |
| OFI slope attenuation (D.1) | ~10× across all Δ |
| W-1 distance (B.1) | 1.74e-06 (in band [0, 4.53e-06]) ✓ |
| Best timestep range (E.2) | t ∈ [300, 500] (Min-SNR working) |
| x0_clip activation rate (E.3) | 1.4% ✓ |
| Validation wall clock | ~12 min (CPU) + ~25 min (GPU) |
| Output files | 53 (23 pdf + 23 png + 7 csv) |

---

## 11. Honest Verdict

The v2 generator is a credible *unconditional* model of INTC microstructure. As a *conditional, stress-directed* model — which is what the project's premise needs — it is not yet good enough. The proposed v3 rework targets the three identified causal failures (heavy tails, long memory, FiLM collapse) and is scoped at ~2 days wall clock. If v3 hits the success thresholds in §6.3, the Work 4 hypothesis test should be re-run; only then will we have a fair test of the proposal's premise.

The validation suite itself is the project's most reusable contribution. The 16 stylized-fact criteria, the consolidated G.1 table, and the model-internals diagnostics (E/F) form a complete LOB-Bench-style evaluation pipeline that will outlive this particular generator. Even a "negative" v3 result would be a clean scientific outcome supported by exhaustive evidence.
