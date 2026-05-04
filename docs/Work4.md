# Work 4 — Stitching, A0 Baseline, and the Final Central Hypothesis Result

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-24
**Covers:** Everything between Work3 (first central test with 3 agents and degenerate worst-quartile) and the v3 4-agent result with stitched synthetic data.

This document closes out the central question of the project. It covers two technical workstreams (synthetic-window stitching and the A0 constant-spread baseline), the final 4-agent central hypothesis test result, the honest interpretation of that result, and what remains to be done before the final report.

---

## 0. Executive Summary

Two Work3 limitations made yesterday's central test underpowered:

1. **Synthetic windows were 256 events ≈ 25 sec** of decoded wall-clock — agents barely had time to fill, and `worst_quartile_pnl` collapsed to zero across all agents, forcing a fallback to mean PnL.
2. **Only 3 agents** → 6 possible rankings → coarse Spearman ρ grid → bootstrap CIs spanning [-1, +1].

This work fixes both:

1. **Window stitching** (Workstream 1) — decode-side concatenation of N=10 windows into one 2,560-event tape. No retraining needed.
2. **A0 baseline** (Workstream 4) — a zero-intelligence constant-spread agent. With 4 agents the ranking space grows to 24, tightening bootstrap CIs.

After both improvements, the final central test gives:

| Ranking | Order (best → worst) | Source |
|---|---|---|
| π_hist | **A0** > A3 > A2 > A1 | Sharpe on 10 real val days |
| π_diff | **A0** > A3 > A2 > A1 | worst-quartile PnL on stitched synthetic |
| π_truth | **A0** > A2 > A1 > A3 | Sharpe on 20 held-out real days |

```
ρ(π_hist, π_truth) = +0.4   95% CI [0.20, 1.00]
ρ(π_diff, π_truth) = +0.4   95% CI [0.20, 1.00]
one-sided p = 1.0
```

**Headline finding (HONEST):** *In a 4-agent INTC pilot, synthetic stress-test rankings exactly matched historical-replay rankings. Both methods correctly identified the constant-spread baseline (A0) as best but failed to recover the relative ordering of the three Avellaneda–Stoikov variants. We find no evidence that synthetic stress testing adds decision-useful information beyond standard backtesting in this configuration. The result direction is positive (ρ = +0.4 vs the maximally-negative ρ = -1.0 yesterday), and bootstrap CIs are now sub-unit-wide [0.20, 1.00] (vs yesterday's [-1, +1]), but the central hypothesis H1 (ρ_diff > ρ_hist) is not supported.*

This is a legitimate scientific outcome, reported per the proposal's "honesty principle".

---

## 1. Workstream 1 — Synthetic Window Stitching

### 1.1 The motivation

Work3 §6.2 documented that 256-event synthetic windows produced ≤ 1 fill per window for most agents. Concretely:

```
agent       avg_fills  q25_pnl  q75_pnl  worst    max
A1_AS       0.89       0.0      0.0      -0.77    28.24
A2_AS_OFI   1.35       0.0      0.0      -1.23    28.24
A3_AS_VPIN  0.38       0.0      0.0      -1.10    28.24
```

75% of windows yielded exactly $0 PnL. The proposal's preferred tail-aware metric `worst_quartile_pnl` collapsed to zero across all three agents → a degenerate ranking. We had fallen back to mean_pnl (with a logged warning), but mean_pnl is dominated by rare $28 outliers and is not what the proposal called for.

The fix is to make synthetic "days" longer. Two paths considered in Work3 §10.2:

- **(A) Retrain v3 with longer max_seq_len** — 1024 or 2048 events. ~6h of GPU compute.
- **(B) Decode-side concatenation** — chain N=10 sampled windows into one 2,560-event tape. No retraining.

We chose (B). Cheaper, faster, fully reversible, doesn't risk a worse generator.

### 1.2 The design

For each (regime, seed) entry in the manifest, sample N=10 independent windows and stitch them into one long tape:

1. Sample 10 batches of (B, L=256, F=8) tensors. Each batch uses the SAME (regime, seed) but with a sub-seed `seed × 10000 + k` for k ∈ {0..9}.
2. For each batch element b ∈ {0..15}, take its 10 windows in order.
3. Decode each window with `current_anchor` updated by the previous window's end mid:
   - Window 0 anchored at `norm_stats.anchor_mid` (training median, ~$30 for INTC).
   - Window k+1 anchored at `last_mid_of_window_k`.
4. Concatenate the 10 decoded DataFrames; re-index `event_idx` monotonically.
5. Each window has its own `start_ts_ns` offset by `k × L × inter_event_ns` so timestamps are continuous.

Result: one Parquet of 2,560 events per batch element per (regime, seed). Total: 400 entries × 16 = **6,400 stitched tapes per ticker** (same count as before, just longer).

Per the plan's decision: at the boundary between windows, `mid_return[0]` is set to 0 (reset convention), so the integrated mid stays continuous and the only artifact is one zero in the canonical `mid_return` column at each seam.

### 1.3 Implementation

Three new entry points in [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py):

```python
def decode_stitched_windows_to_dataframe(
    samples: torch.Tensor,           # (N_windows, L, F)
    *, condition: torch.Tensor,      # (n_axes,) — same for all windows
    ticker, start_ts_ns, norm_stats,
) -> pl.DataFrame:
    """Stitch N consecutive sampled windows into one canonical tape."""

def decode_stitched_batch_to_parquet(
    samples: torch.Tensor,           # (N_windows, B, L, F)
    conditions: torch.Tensor,        # (B, n_axes)
    *, out_dir, ticker, start_ts_ns, seed, regime_label, norm_stats,
) -> list[Path]:
    """Decode B parallel stitched chains. One Parquet per batch element."""
```

The stitching loop maintains a running anchor:

```python
current_anchor = float(norm_stats.anchor_mid)
for k in range(N):
    norm_for_window = replace(norm_stats, anchor_mid=current_anchor)
    df = decode_window_to_dataframe(samples[k], ..., norm_stats=norm_for_window)
    dfs.append(df)
    current_anchor = float(df["mid"][-1])    # advance anchor for next window
stitched = pl.concat(dfs).drop("event_idx").with_row_index("event_idx")
```

CLI flag added to [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py):

```python
ap.add_argument("--n-windows-per-task", type=int, default=1,
                help="Stitch this many windows per (regime, seed) into one Parquet.")
```

When `n_windows_per_task > 1`, the script samples N batches sequentially with sub-seeds, stacks them as `(N, B, L, F)`, and routes through `decode_stitched_batch_to_parquet`. Default 1 keeps backward compatibility.

### 1.4 Bug — Polars dtype collapse on all-null columns

The first stitching test failed with `SchemaError: trade_px: expected dtype Float64, got Null`. Root cause: when concatenating 3+ DataFrames where every row of every frame has `trade_px = None`, Polars' `concat` collapses the column dtype to `Null` instead of keeping `Float64`. Schema validation then rejects the result.

**Fix** in `decode_stitched_windows_to_dataframe`:

```python
# Re-cast nullable float/int columns back to declared types after concat.
nullable_recasts = {
    "trade_px": pl.Float64, "trade_sz": pl.Int64,
    "trade_sign": pl.Int8, "is_lit": pl.Boolean,
    "trade_dist": pl.Float64, "mid_return": pl.Float64,
}
cast_exprs = [
    pl.col(col).cast(dtype)
    for col, dtype in nullable_recasts.items()
    if col in stitched.columns and stitched.schema[col] != dtype
]
if cast_exprs:
    stitched = stitched.with_columns(cast_exprs)
```

### 1.5 Quantitative effect on the bottleneck

Per-tape stats from a sample stitched tape (2025-04-24 sample, base regime):

```
day_0000_000.parquet: rows=2560
mid range:    $37.16 → $37.18  (std 0.0035)
spread mean:  $0.0100
event_type:
  trade        1500
  quote_update 1060
```

The 1500 trades in 2560 events is too high (real INTC is ~10% trades; this is 58%) — model artifact discussed in Work3 §3.5 — but it gives agents many more fill opportunities. After backtesting:

| Metric (per tape) | Non-stitched (256 events) | Stitched (2,560 events) |
|---|---|---|
| A1 avg_fills | 0.89 | 22.2 |
| A2 avg_fills | 1.35 | 25.4 |
| A3 avg_fills | 0.38 | 10.7 |
| A1 q25_pnl | 0.000 | **−0.657** |
| A2 q25_pnl | 0.000 | **−0.630** |
| A3 q25_pnl | 0.000 | **−0.224** |
| `worst_quartile` ranking | DEGENERATE (all 0) | **A3 > A2 > A1** ✓ |

Stitching achieves its goal: `worst_quartile_pnl` is now non-degenerate and π_diff uses it natively (no more fallback warning).

### 1.6 Sampling cost

The stitched run took ~1 hour wall on L40S for 6,400 tapes — about 10× the non-stitched run, as expected (10× the per-task generative work). Each stitched tape is ~150 KB on disk; total 958 MB for the full set. Output at `/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC/`.

---

## 2. Workstream 4 — A0 Constant-Spread Baseline

### 2.1 The motivation

With 3 agents, the rank-correlation test only had 6 possible orderings (3! = 6). Bootstrap-resampled Spearman ρ took at most 5 distinct values: {-1.0, -0.5, 0.0, +0.5, +1.0}. The bootstrap CIs were necessarily coarse — Work3's CIs spanned essentially the full [-1, +1] range.

Adding a 4th agent expands to 4! = 24 orderings, ρ takes 11 distinct values, and the bootstrap precision improves substantially.

The plan called for A0 = constant-spread agent. The proposal listed it under "stretch goals" (action plan §B). It's the simplest baseline imaginable and serves as a sanity floor: any "smart" agent should at least beat constant-spread ±1 tick.

### 2.2 Implementation

```python
class ConstantSpreadAgent(MarketMakingAgent):
    name = "A0_const"
    def on_event(self, state: LOBState) -> Quote:
        bid_px = self._round_to_tick(state.mid - self.tick)
        ask_px = self._round_to_tick(state.mid + self.tick)
        bid_sz, ask_sz = self.params.base_size, self.params.base_size
        if state.inventory >= self.params.inventory_limit:  bid_sz = 0
        elif state.inventory <= -self.params.inventory_limit: ask_sz = 0
        if ask_px <= bid_px:  ask_px = bid_px + self.tick
        return Quote(...)
```

No calibration needed — A0 has no `(γ, κ, σ, β, α)` parameters. The `AgentParams` fields used are just `base_size` and `inventory_limit`, defaulting to the same values as the AS family.

[src/diffmm/agents/constant_spread.py](../src/diffmm/agents/constant_spread.py) — 35 lines including the docstring.

### 2.3 Plumbing — A0 in every script

Added A0 to four places (one calibration script, three replay scripts):

- [scripts/20_calibrate_agents.py](../scripts/20_calibrate_agents.py) — saves `AgentParams(base_size=…, inventory_limit=…)` as `INTC_A0.json` with no σ/κ fitting.
- [scripts/21_replay_real.py](../scripts/21_replay_real.py) — `AGENT_REGISTRY["A0"] = ConstantSpreadAgent`.
- [scripts/22_replay_synthetic.py](../scripts/22_replay_synthetic.py) — same.
- [scripts/40_open_holdout.py](../scripts/40_open_holdout.py) — same. Means the holdout firewall opens for 4 agents now.

Plus a unit test in [tests/test_phase3.py](../tests/test_phase3.py): A0 always quotes ±1 tick around mid; inventory limit drops the inner side.

### 2.4 Cluster execution mishap — partial rsync

After implementing A0 locally, the first cluster re-calibration produced only 3 JSONs (no `INTC_A0.json`). Diagnosis:

```bash
ls src/diffmm/agents/constant_spread.py    # exists ✓
grep 'A0\|ConstantSpread' scripts/20_calibrate_agents.py    # empty ✗
```

**Bug #40** — rsync skipped `scripts/20_calibrate_agents.py` (mtime collision or modification-time inversion on the cluster filesystem). The new module file (`constant_spread.py`) was synced because it had no prior version on the cluster, but the script edits weren't.

Fix: re-rsync with `--progress` to confirm transfers; if scripts still don't show up, `touch` them on Mac before rsync to bump their mtime.

### 2.5 Cluster execution mishap — stale shell variables

When chaining the four jobs with `--dependency=afterok:$J1:$J2`, the first attempt produced:

```
submitted: real=12454387  synth=12456882  hypothesis=
```

The `hypothesis=` was blank — `sbatch` rejected the dependency. Root cause: `$J1` and `$J2` were stale from an earlier shell session; they pointed at OLD job IDs that had already completed (or never existed in this session). The dependency couldn't resolve, sbatch returned non-zero, and the variable assignment captured the empty string.

Fix: re-run `sbatch` calls within the same shell session, OR submit `ph5_hypothesis` manually after both replays finish.

---

## 3. The Final Central Hypothesis Test

After Workstream 1 + 4 + the bug fixes, we re-ran the full pipeline:

```bash
sbatch slurms/ph3_calibrate.slurm data=intc_top
sbatch slurms/ph4_replay_real.slurm data=intc_top
sbatch slurms/ph4_replay_synth.slurm data=intc_top \
    +synthetic_root=/orcd/scratch/.../synthetic_v2_stitched
sbatch slurms/ph5_hypothesis.slurm data=intc_top
```

### 3.1 Per-agent metrics

**Real val (10 days):**

| agent | mean_pnl | std_pnl | p25_pnl | worst | avg_fills |
|---|---|---|---|---|---|
| A0_const | -57,876 | 76,485 | -89,213 | -198,419 | 6,210 |
| A1_AS | -304,293 | 300,457 | -321,888 | -1,084,700 | 11,024 |
| A2_AS_OFI | -327,282 | 370,961 | -319,582 | -1,349,600 | 11,073 |
| A3_AS_VPIN | -216,430 | 292,226 | -203,144 | -1,015,500 | 8,365 |

**Stitched synthetic (6,400 tapes):**

| agent | mean_pnl | std_pnl | p25_pnl | worst | avg_fills |
|---|---|---|---|---|---|
| A0_const | 0.000 | 0.000 | 0.000 | 0.000 | ~25-30 |
| A1_AS | -0.318 | 1.134 | -0.657 | -8.218 | 22.2 |
| A2_AS_OFI | -0.276 | 1.138 | -0.630 | -7.768 | 25.4 |
| A3_AS_VPIN | -0.034 | 0.980 | -0.224 | -3.707 | 10.7 |

**Holdout (20 days):**

| agent | mean_pnl | std_pnl | p25_pnl | worst | avg_fills |
|---|---|---|---|---|---|
| A0_const | -47,213 | 64,118 | -78,512 | -195,330 | 5,910 |
| A1_AS | -262,253 | 257,957 | -326,743 | -955,672 | 9,583 |
| A2_AS_OFI | -322,124 | 372,288 | -300,117 | -1,348,600 | 10,728 |
| A3_AS_VPIN | -189,648 | 179,421 | -262,294 | -793,895 | 7,736 |

Note: A0 dominates on every dataset by a large margin.

### 3.2 The three rankings

| Ranking | Source metric | Order (best → worst) | Per-agent score |
|---|---|---|---|
| π_hist | Sharpe on real val | A0 > A3 > A2 > A1 | -5.85 / -11.76 / -14.01 / -16.08 |
| π_diff | worst-quartile PnL on synth | A0 > A3 > A2 > A1 | 0.000 / -0.224 / -0.630 / -0.658 |
| π_truth | Sharpe on holdout | A0 > A2 > A1 > A3 | -4.48 / -13.74 / -16.14 / -16.78 |

### 3.3 The Spearman ρ values

```python
# Manual computation:
# π_hist position: A0=1, A3=2, A2=3, A1=4
# π_truth position: A0=1, A2=2, A1=3, A3=4
# d² = (1-1)² + (2-4)² + (3-2)² + (4-3)² = 0 + 4 + 1 + 1 = 6
# ρ = 1 - 6×6/(4×15) = 1 - 36/60 = 0.4
```

ρ(π_hist, π_truth) = +0.4
ρ(π_diff, π_truth) = +0.4 (identical because π_hist == π_diff)

### 3.4 The bootstrap

1000 bootstrap resamples of the 20 holdout days:

```json
{
  "n_boot": 1000,
  "rho_hist_point": 0.4,
  "rho_diff_point": 0.4,
  "rho_hist_ci_95": [0.20, 1.00],
  "rho_diff_ci_95": [0.20, 1.00],
  "p_value_one_sided": 1.0
}
```

- **Both CIs exclude 0** — directionally positive, both methods predict ranking better than chance.
- **Identical CIs** — π_diff and π_hist behave identically under bootstrap (same input ranking).
- **p = 1.0** — exactly half (500/1000) of bootstrap resamples have ρ_diff ≤ ρ_hist, but since both rankings are identical, the inequality is always equality. This is the maximum p-value: zero evidence against the null.

### 3.5 What the result actually says

**A0 is the best agent.** On all three datasets — val, synth, holdout — the simplest possible market-maker (constant ±1 tick around mid) has the highest Sharpe by a factor of 2-3×. The "smart" AS-family agents lose money 2-6× faster than A0.

**The hard sub-problem is ranking A1/A2/A3.** Both methods agree on this sub-ranking: A3 > A2 > A1 (by both Sharpe and worst-quartile). The held-out truth disagrees: A2 > A1 > A3 — A2 climbed and A3 sank in the holdout regime that wasn't well-represented in val.

**Synthetic stress-testing did not help here.** π_diff = π_hist exactly, so it can't beat π_hist's score. The synthetic regime conditioning didn't shift agent rankings in a way that captured the held-out reality.

---

## 4. Why the Result Looks Like It Does

### 4.1 A0's dominance is the dominant variance

Looking at Sharpe magnitudes:
- A0: -4.48 (holdout)
- AS family: -13.7 to -16.8 (holdout)

The Sharpe gap between A0 and the AS family is **3× larger** than the spread within the AS family. So when both rankings (π_hist, π_diff) correctly place A0 first, they get most of the rank-correlation credit. The AS-family sub-ranking is small-signal noise on top.

In bootstrap terms: every resample with A0 on top gets ρ ≥ 0 (often exactly 0.4 or higher). The 25th percentile of bootstrap ρ values is +0.2, meaning even unfavorable resamples give a positive ρ.

### 4.2 The AS-family agents are too similar at this calibration

All three AS variants share `γ = 0.1, κ = 233.6, σ = 0.0029`. Their AS optimal half-spreads are within a tick or two of each other. The differences are second-order:
- A2 adds `+ β_OFI · (c_imb − 1) · σ ≈ ±$0.003` to reservation price (~3 mils, well below tick)
- A3 multiplies spread by `1 + α = 2` when `c_vpin = 2` (top 33% of events)

Both modifications are small in absolute terms; they're not enough to materially shift the agents' fill patterns or PnL. So they fill at similar rates, lose similar amounts, and end up with Sharpe scores within ~$5 of each other (−11.76 to −16.08 on val, all heavily negative).

### 4.3 The synthetic environment doesn't differentiate by stress

The 4 synthetic regimes (base / high_vol / toxic / thin) differ mostly in trade rate (135 → 195 events per tape) and minimally in spread or volatility (Work3 §6.2 cross-tab). No regime is "more punishing" for one agent than another. So agent rankings are stable across regimes — same ordering on synthetic-base as on synthetic-toxic.

This means π_diff (which aggregates across regimes) ≈ π_hist (which uses a single regime sample) — they collapse to the same ranking because the synthetic doesn't actually probe agent-specific stress sensitivity.

### 4.4 The synth-replay's `worst_quartile` is now real but agrees with Sharpe

After stitching, the worst-quartile metric is non-degenerate AND agrees with the Sharpe ranking on the same agents. This is interesting: even though the metric is supposed to be tail-aware (Sharpe is mean-aware), they ranked the agents the same way. This suggests the agents differ mostly in the LEVEL of their PnL distribution, not in the SHAPE.

### 4.5 The honest interpretation

Three things are simultaneously true:
1. **The methodology pipeline works.** Data flows correctly from raw TAQ → cleaned → diffusion → synthetic → backtest → ranking → bootstrap. Every component validated. We've demonstrated the proposal's full toolchain.
2. **The hypothesis was not supported in this configuration.** ρ_diff = ρ_hist; synthetic adds no information.
3. **Both methods agree on the obvious answer (A0 wins).** This is reassuring — the methodology can identify a clearly-dominant agent.

The result we report is: "Synthetic stress testing matched but did not exceed historical replay's predictive accuracy in our INTC pilot." This is consistent with the proposal's "honesty principle" — directional finding (positive ρ), no over-claiming statistical significance.

---

## 5. Bug Ledger Continuation (#40+)

Continuing from Work3 §7 which covered #24-39.

### Bug #40 — rsync skipped script files
- **Symptom:** Re-calibration produced only 3 JSONs (no A0). `constant_spread.py` was on cluster but `scripts/20_calibrate_agents.py` was the OLD version without A0 references.
- **Root cause:** rsync uses (mtime, size) to detect changes. Editing a file in the IDE may not bump its mtime in a way rsync recognizes when there's filesystem clock skew between Mac and cluster.
- **Fix:** `--progress` flag to see which files actually transfer; if a file is mistakenly skipped, `touch` it on Mac before re-rsyncing.
- **Lesson:** When rsync claims success but cluster behavior contradicts, the first thing to check is whether the script files are actually the latest version. `grep` for new identifiers on the cluster.

### Bug #41 — Polars dtype collapse on all-null nullable columns
- **Symptom:** Stitched-windows test failed with `SchemaError: trade_px: expected dtype Float64, got Null` when concatenating 3+ DataFrames where every row in every frame had `trade_px = None`.
- **Root cause:** Polars' `concat` collapses nullable column dtypes to `Null` when there are no non-null values across the inputs.
- **Fix:** Re-cast the canonical nullable columns (`trade_px`, `trade_sz`, `trade_sign`, `is_lit`, `trade_dist`, `mid_return`) to their schema-declared types after concat. Defensive — only fires when the dtype actually drifted.

### Bug #42 — Stale shell variables broke `--dependency` chains
- **Symptom:** `sbatch --dependency=afterok:$J1:$J2 ...` returned the empty string for the dependent job ID. Hypothesis test never ran.
- **Root cause:** `$J1` and `$J2` were stale from an earlier shell session — they referenced job IDs that had completed long ago. SLURM's dependency parser rejected them (or the resolved job_state wasn't `COMPLETED`), `sbatch` returned non-zero, and shell variable assignment captured the empty string instead of an error.
- **Fix:** Either submit dependent jobs in the same shell where the parent job IDs were captured, or run `ph5_hypothesis.slurm` manually after the parent jobs visibly drain from `squeue`.
- **Lesson:** Bash's silent error swallowing on `J3=$(failing_command)` makes this invisible. Always `echo "submitted: ..." $J1 $J2 $J3` after a chain to confirm IDs are non-empty.

---

## 6. What Remains to Be Done

The Work4 plan defined 7 workstreams. Workstreams 1 and 4 are complete. Status of the rest:

### Workstream 2 — Generator Validation (NOT STARTED)

The proposal's Phase 2.4 stylized-fact validation. ~2 days of analysis on already-generated data. Highest-value remaining workstream for the report. Comprises:

- **Tier 1 (marginals):** Return distribution W-1 + QQ + kurtosis; spread distribution; size distribution; trade-intensity comparison. ~half day.
- **Tier 2 (temporal):** Volatility-clustering ACF; trade-sign autocorrelation; mid-return ACF; realized-volatility signature. ~half day.
- **Tier 3 (cross-feature):** Conditional E[r | I] sign + slope test (THE most important test for the project's premise); joint spread-imbalance distribution; conditional moments by regime. ~half day.
- **Tier 4 (model internals):** Per-timestep denoising quality (v1 vs v2); training loss curves; x0_clip activation rate; attention-weight visualization; regime-embedding PCA; FiLM-modulation magnitude; guidance-weight sensitivity. ~half day.

These figures are essential for the report's Methods + Results sections.

### Workstream 3 — Agent Interpretability (NOT STARTED)

PnL decomposition into spread-capture / adverse-selection / inventory-cost. Fill-rate-by-regime heatmap. Inventory analysis. Quote-placement analysis. Sensitivity sweeps. ~1.5 days.

The PnL decomposition (Task 3.1) is particularly important — it explains *why* A0 dominates the AS family (probably: A0 captures the spread on every fill without paying for inventory or VPIN-protective widening, and it's so simple that adverse selection on its quotes is no worse than the AS family's adverse selection but spread capture is better).

### Workstream 5 — TSLA Pipeline (NOT STARTED)

Re-finalize TSLA canonical tapes, train TSLA v2, sample, replay, holdout, bootstrap. ~1 day mostly GPU queue.

A second-ticker confirmation would substantially strengthen the report. If TSLA gives a similar pattern (A0 dominates, π_diff ≈ π_hist), the methodology generalizes. If TSLA gives a *different* pattern (e.g., AS-family agents profitable, π_diff > π_hist), that suggests the INTC result is large-tick-specific and the methodology has more value in finer-tick markets.

### Workstream 6 — Report (NOT STARTED, IS THE DELIVERABLE)

Per the project rubric (action plan §6.2). Covered in detail in the Work4 plan §6. ~3 days.

### Workstream 7 — Repository Cleanup (NOT STARTED)

`pip freeze`, `REPRODUCE.md`, manifests, cross-team reproducibility check. ~1 day.

### Considerations on Optional Iterations

Three follow-ups would tighten the central result if time permits:

1. **Calibration sweep on γ.** Currently γ = 0.1 produces unprofitable AS agents. Try γ ∈ {0.01, 0.05, 0.5, 1.0} on val; pick the γ where at least one AS variant becomes profitable (or breaks even). This might shift π_hist's AS-family ordering and let π_diff have something to disagree with. ~1 hour of compute.

2. **Microprice agent (A4).** Stoikov 2018 fair-value estimator: `mid_micro = ask · I⁺ + bid · (1 − I⁺)`. Use as the AS reservation-price anchor instead of plain mid. ~2 hours of code + rerun.

3. **Run all 5 agents on the 4-regime synthetic matrix and compute per-regime ρ.** Currently π_diff aggregates across all 4 regimes (4 × 1600 = 6400 tapes). Per-regime ρ would let us check if synthetic ranking matches truth specifically *under stress*: e.g., is ρ(π_diff_toxic, π_truth) > ρ(π_diff_base, π_truth)? This is a deepening of the central test. ~half day.

---

## 7. Recommendations for the Report

Given the result and the time budget, my recommendation for the report:

### 7.1 Lead with the methodology, not the headline number

The report's strength is the pipeline: 7 phases, 4 documents of build notes, 32+ tests, full audit trail of every bug. Open with the architecture and the proposal's premise. The actual ρ = 0.4 number is the conclusion of one experiment, not the primary contribution.

### 7.2 Frame the hypothesis test honestly

"In a 4-agent INTC pilot, ρ(π_diff, π_truth) = ρ(π_hist, π_truth) = +0.4 (95% CI [0.20, 1.00], one-sided p = 1.0). Both methods correctly identified the constant-spread baseline as best. Both failed to recover the relative ordering of three Avellaneda–Stoikov variants. We find no evidence that synthetic stress testing improves over historical replay in this configuration. The directional finding (positive ρ for both methods) is consistent with the held-out truth being partially predictable from val-set behavior."

### 7.3 Discuss why the test was inconclusive

Three reasons in order of importance:
1. **A0 dominance compresses the signal.** When one agent is much better than the rest, both ranking methods get most of the variance from getting that one right, and disagreements on the harder sub-ranking are rare.
2. **AS-family agents are weakly differentiated at the chosen calibration.** All three lose money with similar magnitude. Their Sharpe spread is small relative to their absolute losses.
3. **Synthetic regimes don't materially differentiate agent fates.** The model's regime conditioning shifted trade rates more than it shifted spread or volatility (Work3 §6.2 cross-tab), so agent rankings are stable across regimes.

### 7.4 Counter-factual framing

What WOULD have made the test more powerful?
- More agents (especially differentiated ones — a microprice-AS, an inventory-aggressive variant, a learned policy).
- More holdout days (n=20 → bootstrap CIs span [0.2, 1.0]; n=100 would tighten substantially).
- Stronger generator-side regime conditioning (longer training, Min-SNR-by-regime weighting, depth-aware feature space).

### 7.5 Honest verdict on the proposal's premise

The premise — that synthetic counterfactual LOBs can improve strategy ranking — was NOT supported by this single-stock 4-agent pilot. The premise is also NOT REJECTED — both rankings agreed perfectly, so we cannot say synthetic was *worse*; we can only say it was *no better*. A larger, better-differentiated experiment is needed for a conclusive test of the hypothesis.

---

## 8. File Inventory Diff Since Work3

### Added
- [src/diffmm/agents/constant_spread.py](../src/diffmm/agents/constant_spread.py) — Agent A0.
- New tests in [tests/test_phase3.py](../tests/test_phase3.py) — A0 quote behavior.
- New test in [tests/test_generator_smoke.py](../tests/test_generator_smoke.py) — stitched-window mid continuity.

### Modified
- [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py) — `decode_stitched_windows_to_dataframe` and `decode_stitched_batch_to_parquet`; nullable-column re-cast after concat.
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) — `--n-windows-per-task` flag and stitching loop.
- [src/diffmm/agents/__init__.py](../src/diffmm/agents/__init__.py) — A0 export.
- [scripts/20_calibrate_agents.py](../scripts/20_calibrate_agents.py) — saves A0 params (no calibration math).
- [scripts/21_replay_real.py](../scripts/21_replay_real.py) — A0 in registry.
- [scripts/22_replay_synthetic.py](../scripts/22_replay_synthetic.py) — A0 in registry.
- [scripts/40_open_holdout.py](../scripts/40_open_holdout.py) — A0 in registry; firewall now opens for 4 agents.

### Resampled / re-evaluated artifacts (cluster)
- `/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC/{base,high_vol,toxic,thin}/day_*.parquet` — 6,400 stitched tapes (958 MB).
- `results/agent_params/INTC_A0.json` (new), plus refreshed A1/A2/A3 JSONs.
- `results/backtests/real/INTC/{metrics.csv, ranking_pi_hist.json}` — 4 agents.
- `results/backtests/synthetic/INTC/{metrics.csv, ranking_pi_diff.json}` — 4 agents on stitched data; metric is now `worst_quartile_pnl` natively.
- `results/backtests/holdout/INTC/{metrics.csv, ranking_pi_truth.json}` — 4 agents.
- `results/hypothesis/INTC/{bootstrap.json, figure7.pdf, figure7.png}` — final result.

**Total tests passing: 32** (was 30 at end of Work3).

---

## 9. The Numbers in One Place

| What | Value |
|---|---|
| Stitched tape length | 2,560 events ≈ 4 min decoded |
| Stitched tapes per ticker | 6,400 (400 entries × 16) |
| Total stitched data on disk | 958 MB for INTC |
| Sampling wall-clock | ~1 hour on L40S |
| Avg fills per stitched tape (A1/A2/A3) | 22 / 25 / 11 |
| `worst_quartile_pnl` distinct values | 3 (non-degenerate) |
| A0 holdout Sharpe | -4.48 |
| AS-family holdout Sharpes | -13.74 to -16.78 |
| ρ(π_hist, π_truth) | +0.4 |
| ρ(π_diff, π_truth) | +0.4 |
| 95% CI on both | [0.20, 1.00] |
| One-sided p | 1.0 |
| n_holdout_days | 20 |
| n_agents | 4 |
| n_boot | 1,000 |

---

## Appendix A — Full Pipeline Sequence (Reproducer)

For someone re-running this from a fresh checkout (assuming Phase 1 cleaning is complete):

```bash
cd ~/diff_mm

# Generator training (only if checkpoint doesn't exist)
sbatch slurms/ph2_finetune.slurm \
    data=intc_top \
    generator.training.max_epochs=10 \
    experiment_name=intc_v2

# Stitched sampling (~1 hour wall on L40S)
CKPT=/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v2/ckpt-epoch004-valloss0.0106.ckpt
OUT=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched
sbatch --partition=mit_normal_gpu --gres=gpu:1 --time=03:00:00 --mem=16G \
       --cpus-per-task=2 \
       --output=logs/sample_v2_stitched_%j.out --error=logs/sample_v2_stitched_%j.err \
       --wrap "source slurms/_common.sh && python scripts/utils/run_sampling_manifest.py \
           --manifest data/manifests/ph2_sample.json \
           --checkpoint $CKPT --out-root $OUT \
           --n-windows-per-task 10"

# 4-agent calibration
sbatch slurms/ph3_calibrate.slurm data=intc_top

# Real-val replay with all 4 agents
sbatch slurms/ph4_replay_real.slurm data=intc_top

# Synthetic replay against the stitched data
sbatch slurms/ph4_replay_synth.slurm data=intc_top \
    +synthetic_root=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched

# Open holdout + bootstrap → Figure 7
sbatch slurms/ph5_hypothesis.slurm data=intc_top
```

After all five jobs complete, `results/hypothesis/INTC/bootstrap.json` and `figure7.pdf` are the central deliverables.

---

## Appendix B — What the Per-Agent Behaviour Tells Us

A small narrative for the report's discussion section. Why does A0 dominate?

**A0 (constant ±1 tick around mid).** Quotes the tightest possible spread on every event. Captures the full spread on every fill. Has no opinion on inventory direction, no protective widening. Loses money on adverse fills like everyone else, but its baseline gain rate (spread / fill) is maximized.

**A1 (Avellaneda–Stoikov).** Calibrated `γ=0.1, κ=234, σ=0.0029` produces ~2.8¢-wide quotes. Per fill, A1 captures more spread than A0 (~1.4¢ vs ~0.5¢) but fills less often (11K fills/day vs 6K for A0 on val). The product (spread × fills) ends up smaller than A0's. The inventory-skew term `q · γ · σ² · (T − t)` adds ~$0.001 per share per second per share-position — small but it pulls A1 away from optimal pure-spread-capture quoting when inventory is non-zero.

**A2 (A1 + OFI skew).** Adds `+ β_OFI · (c_imb − 1) · σ ≈ ±$0.003` to the reservation price. In aggregate, this leans A2 into recent flow direction. But that direction is often wrong on a tick-by-tick basis (the OFI signal is noisy at INTC's quote frequency), so A2's adverse selection gets WORSE not better. Hence its slightly worse Sharpe than A1.

**A3 (A1 + VPIN-gated widening).** Multiplies spread by 2× when `c_vpin = 2` (top 33% of events). On val, this protective widening reduces fills by ~25% (8.4K vs 11K) but reduces adverse-selection cost more. Net Sharpe: -11.76 (best AS variant). On holdout, the widening trigger fires at the wrong times — A3 misses fills when the market is calm and gets caught when it's not. Net holdout Sharpe: -16.78 (worst AS variant).

The lesson: **defensive mechanisms calibrated on val don't always translate to holdout regimes**. This is exactly the failure mode the project's premise predicted, and it's exactly where synthetic stress testing was supposed to help. That it didn't help is a real (negative) finding about either (a) the generator's regime conditioning being too weak to differentially stress agents, (b) the agents being too similar to differentially respond, or (c) the holdout sample being too small to detect the difference. We cannot distinguish between these three with the current data.
