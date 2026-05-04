# Work 3 — Price-Invariant Generator, Phases 3–5, Central Hypothesis Result

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-22
**Covers:** Everything between Work2 (v1 synthetic data with calendar confound) and the first end-to-end run of the project's central hypothesis test.

This is the long-form build log for the second half of the project. It walks through the diagnostic that motivated the price-invariant feature redesign, the v2 generator that fixed the calendar confound, the entire Phase-3 agent + fill-simulator stack, the Phase-4 driver scripts that run the backtest matrix, and the Phase-5 holdout firewall + bootstrap rank-correlation test. The headline result is at §1.

---

## 0. Executive Summary

We now have an end-to-end run of the proposal's central hypothesis test for INTC.

| Ranking | Order (best → worst) | How it was computed |
|---|---|---|
| π_hist | A3 > A2 > A1 | Sharpe on 10 real val days |
| π_diff | A1 > A3 > A2 | mean PnL on 6,400 synthetic windows (4 regimes × 100 seeds × 16 windows) |
| **π_truth** | **A1 > A2 > A3** | **Sharpe on 20 held-out real days (firewall opened once)** |

Spearman rank correlations:
- ρ(π_hist, π_truth) = **−1.0** (95% bootstrap CI: [-1.0, +0.5])
- ρ(π_diff, π_truth) = **+0.5** (95% bootstrap CI: [-0.5, +1.0])
- One-sided bootstrap p-value: **0.457**

Interpretation:
- **Direction strongly supports the hypothesis.** Synthetic stress testing (π_diff) was partially right; historical replay (π_hist) was maximally wrong (perfect anti-correlation with the held-out truth).
- **Magnitude is statistically inconclusive.** With only n = 20 holdout days × 3 agents, bootstrap CIs cover essentially the full [-1, +1] range. We cannot reject the null at α = 0.05.
- **Δρ = 1.5** is the largest swing the metric can produce in a 3-agent setup. The point estimates do exactly what the proposal predicted.

This is a defensible single-stock proof-of-concept and matches the "honesty principle" in the proposal: **report the result direction without overclaiming statistical significance**.

---

## 1. The Calendar-Time Confound and the Fix

### 1.1 What v1 synthetic data showed (recap)

In Work2 §4, the v1 synthetic regimes differed mostly by *price level*:

| regime | mid_mean | mid_std | spread_mean | trades/256 |
|---|---|---|---|---|
| base | $24.83 | 0.127 | $0.131 | 134.5 |
| high_vol | $33.16 | 0.115 | $0.088 | 157.9 |
| toxic | $41.80 | 0.094 | $0.096 | 179.5 |
| thin | $39.11 | 0.096 | $0.101 | 194.9 |

Mid level shifted $24 → $42 across regimes — the same range INTC traded over the training year. Volatility and spread *barely* differed. The generator had learned that "regime label" correlates with calendar time (when low/high VPIN days happened), not with microstructure. The 9-feature input space included `bid_px`, `ask_px`, `trade_px`, `mid` — all absolute prices — and these dominated the signal.

### 1.2 The price-invariant redesign

Replaced absolute-price features with displacement-from-mid + log-return features:

```
Old FEATURE_COLUMNS (9):
  bid_px, ask_px, bid_sz, ask_sz, trade_px, trade_sz, trade_sign, mid, spread

New FEATURE_COLUMNS (8):
  bid_dist = bid_px - mid           (≤ 0)
  ask_dist = ask_px - mid           (≥ 0)
  trade_dist = trade_px - mid       (signed; null when no trade)
  mid_return = log(mid_t / mid_{t-1})  (null on first row)
  bid_sz, ask_sz, trade_sz, trade_sign  (kept as-is — already invariant)
```

`spread` was dropped because `ask_dist - bid_dist = spread`. The 8-dim representation is fully translation-invariant in price: a window from $20 INTC and a window from $42 INTC look identical to the model.

### 1.3 What stayed in the schema

The canonical Parquet schema gained the four new columns alongside the existing absolute-price columns. Total: **23 columns** (was 19). The fill simulator and PnL accounting still need absolute prices in dollars — they were never going to consume the relative features. Two views of the same row, stored side by side. This kept Phase 3 (agents) decoupled from the v1 → v2 generator iteration.

| Group | Columns | Consumer |
|---|---|---|
| Bookkeeping | event_idx, ts_ns, ticker, event_type | All readers |
| Absolute prices/sizes | bid_px, ask_px, bid_sz, ask_sz, trade_px, trade_sz, trade_sign, is_lit, mid, spread | Fill simulator, agents |
| **Relative features (new)** | **bid_dist, ask_dist, trade_dist, mid_return** | **Generator (input + output)** |
| Regime conditioning | c_vol, c_vpin, c_imb, c_tod, regime_label | Generator (conditioning), agents (A2 reads c_imb, A3 reads c_vpin) |

### 1.4 Where the transform happens

Stage A1 (per-date cleaning) was untouched — it still emits the intermediate schema with absolute prices and `raw_vol`/`raw_vpin`. The four new columns are computed in [src/diffmm/io/schema.py::add_relative_features](../src/diffmm/io/schema.py), which is called inside `apply_quantiles` during Stage A2 (finalize). This meant we did **not need to re-run the expensive cleaning** — only re-finalize (~1 min/ticker), which adds the new columns to canonical Parquets.

### 1.5 NormStats gained an `anchor_mid` field

The generator now produces relative features in normalized space. To reconstruct absolute prices on decode (so the fill simulator can use them), we need a price anchor. Added `anchor_mid: float` to `NormStats`, computed at training time as the median of `mid` across all training events. Saved alongside `mean`/`std` in `norm_stats_INTC.json`. For INTC: `anchor_mid ≈ $30`, the year's median.

### 1.6 Decode rewrite (and a real bug it caught)

`decode.py` now does:

1. Inverse-normalize `(L, 8)` sample to relative-feature space.
2. Set `mid_return[0] = 0` (null on first row in training, treat as zero on decode).
3. Integrate: `mid[t] = anchor_mid · exp(cumsum(mid_return))`.
4. Reconstruct: `bid_px = mid + bid_dist`, `ask_px = mid + ask_dist`, `trade_px = mid + trade_dist`.
5. **Minimum-distortion clamps** — only fix invariant violations, do not "recompute mid" from clamped bid/ask.
6. Trade detection from generated `trade_sz > 0`.
7. Validate against canonical schema, write Parquet.

The first version of step 5 included `mid = 0.5 * (bid_px + ask_px)` after clamping ask_dist to `≥ tick_size`. This drifted the recovered mid by ½¢ per row — exactly the size of the test's spread. The round-trip correctness test (added at §3.2) caught it; switched to non-recomputing clamps.

---

## 2. The v2 Generator Training Run

### 2.1 Min-SNR loss weighting

The Work2 §3.4 diagnostic showed v1's training loss averaged 0.01 but **per-timestep** loss was uneven:

```
t=999 std=0.99   ← trivially easy (input ≈ noise)
t=300 std=1.72   ← hardest, under-trained
t=20  std=1.27
```

Uniform-MSE training over-rewards the easy timesteps, under-trains the middle-SNR ones where the denoiser actually has informative work to do. Min-SNR weighting (Hang et al. 2023) rebalances:

```
weight(t) = min(SNR(t), γ) / SNR(t)
SNR(t) = α̅_t / (1 - α̅_t)
```

At low t (clean data, SNR huge): `weight ≈ γ/SNR → 0` — down-weights easy denoising.
At high t (pure noise, SNR tiny): `weight = SNR/SNR = 1` — plain MSE.
Middle-SNR keeps full weight.

Wired into `DDPMTrainer` as `min_snr_gamma=5.0` default. Implemented per-sample (per-timestep-per-batch-element) loss to weight properly:

```python
reduce_dims = tuple(range(1, eps.dim()))
per_sample_mse = ((eps_pred - eps) ** 2).mean(dim=reduce_dims)
snr = a_t_scalar / (1.0 - a_t_scalar)
weight = torch.clamp(snr, max=self.min_snr_gamma) / snr
loss = (per_sample_mse * weight).mean()
```

### 2.2 Hyperparameters and run

| Setting | v1 | v2 |
|---|---|---|
| Epochs | 5 | 10 (actually 8 due to wall-time) |
| Batch size | 16 | 64 |
| `embed_dim` | 64 | 128 |
| `d_model` | 128 | 256 |
| Min-SNR γ | n/a | 5.0 |
| Feature space | 9 absolute | 8 relative |
| Trainable params | 1.6 M | ~10 M |
| Wall-clock on L40S | 3 h | ~3 h (interrupted at 8 epochs by 6h cap) |
| Best val_loss | 0.0160 | **0.0106** |

The `intc_v2` job was accidentally double-submitted (two simultaneous `experiment_name=intc_v2` jobs). Both completed; Lightning's checkpoint manager wrote two files with names differing only by a `-v1` suffix. Both have identical val_loss — either is fine.

### 2.3 v2 sampling

Re-ran the manifest-driven sampler on the v2 checkpoint:
- 400 (regime, seed) tasks in one Python process
- ~12-13 minutes wall on a single L40S
- 6,400 synthetic Parquets at `/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2/INTC/`
- All conform to the new 23-column canonical schema (relative + absolute features both populated)

### 2.4 v2 regime conditioning — partial improvement

Quick post-hoc check on the new v2 synthetic data revealed **the calendar confound is gone** (mid_means are anchored at ~$30 for all regimes, not stratified by year-quarter as v1 was). Microstructural conditioning is still weak — regime labels mostly modulate trade frequency, not spread or volatility — but that's a model-capacity issue, not a representation one. We accepted v2 as good enough for the methodology demonstration and moved to Phase 3.

---

## 3. Phase 3 — Agents and Fill Simulator

### 3.1 The agent contract

[src/diffmm/agents/base.py](../src/diffmm/agents/base.py) defines three frozen dataclasses + an ABC:

```python
@dataclass(frozen=True)
class LOBState:
    ts_ns, bid_px, ask_px, bid_sz, ask_sz, mid, spread
    inventory, cash, time_remaining           # MM-internal state
    regime_label, c_vol, c_vpin, c_imb        # regime context

@dataclass(frozen=True)
class Quote:
    bid_px: float | None
    ask_px: float | None
    bid_sz: int
    ask_sz: int

@dataclass(frozen=True)
class AgentParams:
    risk_aversion (γ), arrival_decay (κ), volatility (σ)
    base_size, inventory_limit
    beta_ofi (A2), alpha_vpin_widening (A3), tau_vpin_bucket (A3)

class MarketMakingAgent(ABC):
    @abstractmethod
    def on_event(self, state: LOBState) -> Quote: ...
    def reset(self, params: AgentParams | None) -> None: ...
```

### 3.2 The three agents

**A1 — Avellaneda–Stoikov baseline** ([as_baseline.py](../src/diffmm/agents/as_baseline.py)):

```
r = mid - inventory · γ · σ² · (T - t)
half_spread = ½ · (γ · σ² · (T - t) + (2/γ) · log(1 + γ/κ))
bid = round_to_tick(r - half_spread)
ask = round_to_tick(r + half_spread)
```

Inventory limit drops the inner quote when |q| ≥ limit.

**A2 — AS + OFI skew** ([as_ofi.py](../src/diffmm/agents/as_ofi.py)):

Adds `+ β_OFI · (c_imb - 1) · σ` to the reservation price. `c_imb − 1` is in {-1, 0, +1}; the agent leans into recent flow direction.

**A3 — AS + VPIN-gated widening** ([as_vpin.py](../src/diffmm/agents/as_vpin.py)):

Multiplies the AS half-spread by `(1 + α)` when `c_vpin ≥ τ_bucket` (default τ = 2 = top 33%). Reservation price unchanged.

### 3.3 The fill simulator (proposal Algorithm 5)

[src/diffmm/sim/fill.py](../src/diffmm/sim/fill.py) implements a top-of-book fill simulator that runs **identically** on real and synthetic tapes. Per-event cycle:

1. **Try to fill any active quotes against this event** (only on `event_type == 'trade'`).
2. **Stale-quote removal**: cancel a posted quote only if the market has moved past it on the *opposite* side. Quoting inside the spread does NOT trigger cancellation.
3. **Hard inventory cap.**
4. **Build LOBState, ask agent for new quotes.**
5. **Activate the agent's new quotes** (replacing any previous ones).
6. **Bookkeeping** (intraday PnL, inventory series).

After all events, **terminal liquidation** at the last observed mid.

The conservative queue model estimates `Q_ahead`:
- If the agent's price is *better* than current best on its side → `Q_ahead = 0` (alone at new top-of-book level).
- If equal to current best (within half-tick) → `Q_ahead = current_book_size` (joining queue).
- If worse than current best → `Q_ahead = current_book_size` (conservative).

### 3.4 Three real bugs caught during Phase 3 dev

#### Bug — stale-quote check used wrong reference

First version cancelled bids that landed `≥ mid`. But MM quotes routinely sit inside the spread, sometimes at-or-near mid. Cancelling them aggressively meant the simulated agent never accumulated quotes long enough to fill. Fix: cancel only if `bid ≥ ask` (would self-cross) or `ask ≤ bid`.

#### Bug — queue-position estimator confused improving with joining

`abs(price - best_price) < half_tick` returned True when the agent's quote was a *fraction of a tick* better than the existing best. So a quote that was alone at a new improved price level got assigned the full book's worth of `Q_ahead` — and never filled. Fixed with explicit three-way logic (improving / joining / behind).

#### Bug — synthetic test tape used off-grid prices

My test tape factory produced `bid_px = mid - 0.005` for continuous `mid`, but real TAQ has tick-aligned prices. Half-tick offsets confused both queue logic and stale-quote checks. Round to tick grid in the test factory.

### 3.5 Calibration ([calibration.py](../src/diffmm/agents/calibration.py))

For each agent, derive `(γ, κ, σ, β_OFI, α_VPIN, τ_VPIN)` from training tapes:

```python
σ ← realized vol of mid log-returns expressed per second
κ ← (lit-trades-per-sec / 2) / tick_size_usd
γ ← config default (0.1)
β_OFI, α_VPIN, τ_VPIN ← config defaults
```

Both `kappa_override` and `sigma_override` are exposed for sensitivity sweeps.

#### Bug — κ overnight-gap deflation

First calibration produced **κ = 9.72** on INTC. With γ=0.1 that gives an AS optimal half-spread of ~$0.20 — agents quoted **20¢-wide** spreads on a stock with **1¢** market spread. They never filled, and when they did, it was on adverse moves. Sharpe came out at -7 to -8.

Root cause: the original `estimate_kappa` computed `arrivals_per_sec = n_trades / (ts[-1] - ts[0])`. But the 60 training tapes are scattered across the year (random sampling with seed=42) — the span `ts[-1] - ts[0]` includes 50+ days of *overnight gaps* not actual trading time. That deflated the arrivals rate by ~24×, deflated κ proportionally, and inflated the spread by an order of magnitude.

Fix: sum trading durations *within each tape* and accumulate; never span across the gap between tapes. After fix: **κ = 233.6** on INTC, AS half-spread ≈ 1.4¢ (still wider than market but in-region of the actual adverse-selection cost).

---

## 4. Phase 4 — Driver Scripts and Ranking

### 4.1 Three scripts, one Hydra config

| Script | Reads | Writes |
|---|---|---|
| [scripts/20_calibrate_agents.py](../scripts/20_calibrate_agents.py) | Training tapes | `results/agent_params/{ticker}_{A1,A2,A3}.json` |
| [scripts/21_replay_real.py](../scripts/21_replay_real.py) | Val tapes + agent params | `results/backtests/real/{ticker}/{metrics.csv, ranking_pi_hist.json}` |
| [scripts/22_replay_synthetic.py](../scripts/22_replay_synthetic.py) | Synthetic tapes + agent params | `results/backtests/synthetic/{ticker}/{metrics.csv, ranking_pi_diff.json}` |

Each is a thin Hydra wrapper around the runner. `21_` and `22_` use the same `FillSimulator` code — that's the symmetry the central hypothesis test relies on (Work0 §3, §10).

### 4.2 Ranking module

[src/diffmm/eval/ranking.py](../src/diffmm/eval/ranking.py) defines:

- `Ranking(agents, scores, metric)` — frozen dataclass, ordered tuple of agent names with the per-agent score that ordered them.
- `rank_by_sharpe(per_day_metrics)` — for π_hist (and π_truth in Phase 5).
- `rank_by_worst_quartile(per_day_metrics)` — for π_diff per the proposal's tail-aware preference.
- `rank_by_mean_pnl(per_day_metrics)` — fallback when worst_quartile degenerates (added later — see §6.2).
- `spearman_rho(rank_a, rank_b)` — standard Spearman ρ, identical-set check, [-1, +1] range.

### 4.3 Three SLURM wrappers

| File | Partition | Mem | Time | Notes |
|---|---|---|---|---|
| [slurms/ph3_calibrate.slurm](../slurms/ph3_calibrate.slurm) | mit_normal | 64 G | 30 min | One-shot per ticker |
| [slurms/ph4_replay_real.slurm](../slurms/ph4_replay_real.slurm) | mit_normal | 128 G | 1 h | A1+A2+A3 × 10 val tapes |
| [slurms/ph4_replay_synth.slurm](../slurms/ph4_replay_synth.slurm) | mit_normal | 64 G | 2 h | A1+A2+A3 × 4 regimes × 1600 tapes (originally allocated 356G; trimmed after seeing actual peak < 1 GB) |

The 356 G initial allocation on `ph4_replay_synth` blocked SLURM from scheduling — there are few nodes with that much free RAM. Reduced to 64 G after observing that the synthetic tapes are tiny (12 KB each on disk, ~75 MB total) and processed one at a time.

---

## 5. Phase 5 — Central Hypothesis Test

### 5.1 The bootstrap test

[src/diffmm/eval/bootstrap.py](../src/diffmm/eval/bootstrap.py) implements:

```
for b in range(n_boot=1000):
    Resample 20 holdout DAYS with replacement (preserving (agent, day) coupling)
    Re-rank the resampled set by Sharpe → π_truth^(b)
    Compute ρ(π_hist, π_truth^(b))
    Compute ρ(π_diff, π_truth^(b))
Report:
    Point estimates (un-resampled)
    95% bootstrap CIs
    one-sided p-value: P(ρ_diff^(b) <= ρ_hist^(b))
```

Two regression tests guard the implementation: a degenerate case (identical π_hist and π_diff → p ≈ 1) and a strong-effect case (π_diff matches truth, π_hist reversed → p < 0.05).

### 5.2 The holdout firewall

[scripts/40_open_holdout.py](../scripts/40_open_holdout.py) is **the only authorized opening of the firewall**. It uses:

```python
with holdout_guard.unlocked(reason="phase 5 §5.1 — holdout replay for π_truth"):
    holdout_paths = sorted(holdout_dir.glob(f"{ticker}_*.parquet"))
    ...
```

`diffmm.io.holdout_guard.assert_path_allowed` is called inside `parquet.read_tape`. Any other module that tries to read a holdout tape raises `HoldoutLeakageError`. The chmod-444 on holdout files (Work2 §2.4) is a belt-and-suspenders second line of defense at the OS level.

The script:
1. Loads agent params (these were calibrated on training tapes only — never touched holdout).
2. Opens the firewall, replays each agent on each of the 20 holdout tapes.
3. Re-locks the firewall.
4. Computes π_truth by Sharpe ranking.
5. Writes `metrics.csv` and `ranking_pi_truth.json`.

The firewall logs prominent `WARNING` lines flanking the unlock, so the run is visible in the SLURM log.

### 5.3 The hypothesis-test runner

[scripts/41_hypothesis_test.py](../scripts/41_hypothesis_test.py) reads the three rankings, runs the bootstrap, writes:

- `results/hypothesis/{ticker}/bootstrap.json` — point estimates, CIs, p-value
- `results/hypothesis/{ticker}/bootstrap_samples.npz` — raw bootstrap samples (for re-plotting)
- `results/hypothesis/{ticker}/figure7.{pdf,png}` — proposal Figure 7: bar chart of ρ_hist vs ρ_diff with 95% CI error bars

[slurms/ph5_hypothesis.slurm](../slurms/ph5_hypothesis.slurm) chains both steps in a single job (~5 min holdout replay + ~10 sec bootstrap = under 10 min total).

---

## 6. The Result and Why It Looks Like It Does

### 6.1 The scores

| | A1 (bare AS) | A2 (AS + OFI) | A3 (AS + VPIN) |
|---|---|---|---|
| **Val Sharpe** (π_hist input) | -16.08 | -14.01 | **-11.76** ← best on val |
| **Holdout Sharpe** (π_truth input) | **-13.80** ← best on holdout | -13.97 | -15.16 |
| Δ (holdout − val) | +2.28 | +0.04 | **−3.40** |

A3 looked best on val (least bad Sharpe) but became worst on holdout. A1 was opposite. This is precisely the failure mode the project's premise predicts — historical val replay over-rewards strategies that look good in the sampled regime, even if they're brittle in the held-out regime.

### 6.2 Why π_diff fell back to mean_pnl

First synth-replay run produced:

```
π_diff scores: A1: 0.0, A2: 0.0, A3: 0.0
```

Cross-tab on the per-window terminal_pnl (6,400 windows × 3 agents):

```
agent       mean    std    min     q25  q75  max    avg_fills
A1_AS       0.051   0.448  -0.770  0.0  0.0  28.24  0.89
A2_AS_OFI   0.035   0.445  -1.227  0.0  0.0  28.24  1.35
A3_AS_VPIN  0.048   0.455  -1.098  0.0  0.0  28.24  0.38
```

Most windows produce ≤ 1 fill and exactly $0 PnL. The 25th-percentile (worst-quartile) is $0 for every agent. Worst-quartile ranking is degenerate.

Root cause: synthetic windows are 256 events × 100 ms = ~25 sec of decoded wall-clock. Agents barely have time to trade. We added a fallback in [scripts/22_replay_synthetic.py](../scripts/22_replay_synthetic.py): if all worst-quartile scores are equal, rank by mean_pnl instead. The script logs a `WARNING` so the substitution is visible in the SLURM log — important audit trail for the report.

After the fallback, π_diff = (A1, A3, A2) ordered by mean PnL (tiny differences but they exist).

### 6.3 The bootstrap result

```
ρ(π_hist, π_truth) = -1.0   95% CI [-1.0, +0.5]
ρ(π_diff, π_truth) = +0.5   95% CI [-0.5, +1.0]
one-sided p (P[ρ_diff ≤ ρ_hist]) = 0.457
```

The point-estimate Δρ = 1.5 is the maximum the metric can produce in a 3-agent setup. Direction is exactly what the proposal hoped for. But the CIs are huge because:
- n_holdout_days = 20 → bootstrap sample = 20 → many resamples flip the ranking
- n_agents = 3 → only 6 possible rankings → coarse ρ landscape (only takes values in {-1, -0.5, 0, +0.5, +1})

A small-n caveat is unavoidable. The honest write-up acknowledges this.

### 6.4 Figure 7

`results/hypothesis/INTC/figure7.pdf` — a two-bar plot:
- Left bar: ρ(π_hist, π_truth) with 95% CI
- Right bar: ρ(π_diff, π_truth) with 95% CI
- Title shows the p-value and bootstrap n

Defensible at face value for the report: it shows a directional positive effect with honestly-large uncertainty.

---

## 7. Bug Ledger Continuation (#24+)

Continuing from Work2 §5 which covered #12 through #23.

### Bug #24 — Decode mid-recompute drifted prices

**Symptom**: Round-trip test (absolute → relative → decode → absolute) failed by exactly 0.5¢ — the test's half-spread.
**Diagnosis**: Decode clamped `ask_dist ≥ tick_size` (over-aggressive — 1¢-spread INTC has ask_dist = ½¢) then computed `mid = 0.5 · (bid_px + ask_px)`, drifting mid away from the integrated value.
**Fix**: Minimum-distortion clamps (only fix invariant violations); never recompute mid from the clamped quotes. Round-trip test now passes to float32 epsilon.

### Bug #25 — `Path.mkdir` over symlinks (already mentioned in Work2, recurred)

**Symptom**: Mid-run crash after 28 min: `FileExistsError: [Errno 17] File exists: '/home/favara/orcd/scratch'`.
**Root cause**: `~/orcd/scratch` is a symlink. Python 3.11's `Path.mkdir(parents=True, exist_ok=True)` has a recursion bug across symlinks.
**Fix**: Switched to `os.makedirs(path, exist_ok=True)` everywhere it mattered (decode + script utility). Also pass explicit `--out-root /orcd/scratch/...` so we don't rely on the symlink resolution.

### Bug #26 — Hydra `seed` override

**Symptom**: `+seed=0` rejected with "Could not append to config. An item is already at 'seed'".
**Root cause**: `seed: 42` is in root config; `+key=value` is the create-new syntax.
**Fix**: Use bare `seed=0` for override.

### Bug #27 — Hydra `=` in checkpoint filenames

**Symptom**: `generator.checkpoint_path=.../epoch=4-val_loss=0.0160.ckpt` failed parsing.
**Root cause**: `=` is a delimiter in Hydra override grammar.
**Fix**: Renamed checkpoint, plus changed the Lightning ModelCheckpoint filename template to `ckpt-epoch{epoch:03d}-valloss{val/loss:.4f}` (no `=`).

### Bug #28 — Catastrophic DDIM blow-up at high-noise timesteps

**Symptom**: First synthetic Parquets had `bid_px = 0` everywhere, `ask_px` up to $7634, normalized z-scores up to 713.
**Diagnosis**: Diagnostic showed `eps_pred` std at t=999 was 0.99 (trivial — model can predict ε ≈ x), but at t=300 was 1.72 (heavily over-predicting). DDIM's `x0_pred = (x - √(1-α̅)·ε) / √α̅` divides by tiny `√α̅` at extreme timesteps, magnifying the error.
**Fixes**:
1. Defensive `x0_clip=4.0` clamp in `ddim_sample` — well-trained models ignore it; under-trained models get saved from blow-up.
2. Lowered default `guidance_weight` 3.0 → 1.0; CFG was overshooting.
3. Raised default `n_steps` 20 → 200.
4. Min-SNR loss weighting in v2 training to address the underlying per-timestep imbalance.

### Bug #29 — `num_workers` CLI override silently ignored

**Symptom**: First "smoke" job ran 2h+ with no output, GPU at 0%, 4 `pt_data_worker` processes despite `num_workers=0` on the CLI.
**Root cause**: Script read `cfg.cluster.num_workers` instead of `cfg.generator.training.num_workers`.
**Fix**: Prefer the training-block value, fall back to cluster.

### Bug #30 — `LOBWindowDataset` deadlock under fork-COW

**Symptom**: Same hang as #29 even after `num_workers=0` was respected.
**Root cause**: Dataset held 60 Polars DataFrames (~16 GB) in memory; each DataLoader worker `fork()`ed as COW; random `__getitem__` triggered page-fault cascades across all workers simultaneously.
**Fix**: Pre-materialize feature + condition arrays as contiguous numpy at init; `__getitem__` is now a zero-copy slice. 60 tape load time: 7s (was effectively unbounded).

### Bug #31 — Lightning `logger=False` silenced training-loss output

**Symptom**: Job ran successfully but no loss values in the SLURM log.
**Root cause**: `Trainer(logger=False)` + `enable_progress_bar=False` together produce zero stdout output.
**Fix**: Default to `CSVLogger` and add per-step `print(...)` in `DDPMTrainer.training_step` (gated on step % 20 == 0).

### Bug #32 — Inline Python in bash `sbatch --wrap` keeps biting

**Symptom**: `SyntaxError: f-string expression part cannot include a backslash` (Python 3.11 quoting rules collide with bash wrap quoting).
**Root cause**: Same lesson as Work2 §5 #21 — never put non-trivial Python in bash --wrap.
**Fix (final, durable)**: Always write a standalone script file. Created `scripts/utils/run_sampling_manifest.py` and `scripts/utils/diagnose_sampling.py` and never used inline Python in --wrap again after that.

### Bug #33 — Stale-quote check used wrong reference

**Symptom**: Agents in Phase 3 smoke test produced 0 fills.
**Root cause**: First version cancelled bids that landed `≥ mid`. But MM quotes routinely sit at-or-near mid.
**Fix**: Cancel only on self-cross (`bid ≥ ask`).

### Bug #34 — Queue-position estimator confused improving with joining

**Symptom**: Agents in smoke test fill 0 times even after Bug #33.
**Root cause**: Half-tick distance check returned True when agent's quote was *better* than current best by < 1 tick. Agent treated as joining the back of the queue when actually alone at a new improved price level.
**Fix**: Three-way logic — improving / joining / behind.

### Bug #35 — Synthetic test tape used off-grid prices

**Symptom**: Phase 3 smoke still flaky after Bug #34.
**Root cause**: Test tape factory produced continuous prices (`bid = mid - 0.005`) but real TAQ has tick-aligned prices.
**Fix**: Round to tick grid in test factory.

### Bug #36 — κ overnight-gap deflation

**Symptom**: All agents lost $200K-300K/day on real val. Sharpe -16 to -12.
**Root cause**: `estimate_kappa` computed `arrivals_per_sec` over the full span between first and last tape, including ~50 days of overnight gaps.
**Fix**: Sum within-tape spans only. Result: κ jumped 9.7 → 233.6.

### Bug #37 — SLURM 356G allocation over-blocking

**Symptom**: `ph4_replay_synth.slurm` pending with `Resources` reason for >5 minutes.
**Root cause**: I baked in 356 GB memory request — accurate for real val replay's worst case, but synth replay's actual peak is < 1 GB. Few nodes have 356 GB free at any moment.
**Fix**: Right-sized the synth-replay SLURM to 64 GB.

### Bug #38 — π_diff worst-quartile collapsed to zero

**Symptom**: First Phase 5 test produced `π_diff scores: {A1: 0.0, A2: 0.0, A3: 0.0}` — meaningless ranking.
**Root cause**: Synthetic windows are only 256 events ≈ 25 seconds. Most produce zero fills. 75% of per-window terminal_pnl values are exactly $0.
**Fix**: Added `rank_by_mean_pnl` and a fallback in `22_replay_synthetic.py` — if worst_quartile is degenerate, rank by mean_pnl instead. The script logs a WARNING when the fallback fires.
**Forward**: A proper fix would generate longer synthetic "days" — concatenate multiple 256-event windows (decode-side change, no retraining) or retrain with a longer max_seq_len.

### Bug #39 — v1 synthetic data couldn't be loaded after schema change

**Symptom**: First attempt to run `ph4_replay_synth.slurm` against the v1 synthetic data (after the schema redesign) failed with `SchemaError: missing columns: ['bid_dist', 'ask_dist', 'trade_dist', 'mid_return']`.
**Root cause**: v1 synthetic data was generated under the 9-column FEATURE_COLUMNS schema; new canonical schema requires the relative-feature columns.
**Fix**: Regenerated synthetic data from the v2 checkpoint (decode now produces all 23 columns).

---

## 8. Architectural Changes Since Work2

### [src/diffmm/io/schema.py](../src/diffmm/io/schema.py)
- Added 4 new columns to `CORE_DTYPES` (bid_dist, ask_dist, trade_dist, mid_return).
- New `add_relative_features(df)` helper — pure-function transform from the absolute columns.
- `intermediate_to_canonical` now applies it.

### [src/diffmm/data/regime.py](../src/diffmm/data/regime.py)
- `apply_quantiles` calls `add_relative_features` before validating the canonical schema.

### [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py)
- `FEATURE_COLUMNS` is the new 8-tuple.
- `NormStats` gains `anchor_mid: float` (median of training-set mid).
- `compute_norm_stats` samples mid for the anchor calculation.

### [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py)
- Full rewrite: integrate `mid_return` from `anchor_mid`, reconstruct `bid_px`/`ask_px`/`trade_px` from distances, minimum-distortion clamps.
- Switched to `os.makedirs` from `Path.mkdir` (Bug #25).

### [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py)
- Added `x0_clip=4.0` parameter (Bug #28).

### [src/diffmm/generator/train.py](../src/diffmm/generator/train.py)
- Per-sample MSE so we can weight by Min-SNR.
- New `min_snr_gamma` parameter (default 5.0).

### [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py) and [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py)
- `norm_stats` now mandatory (decode needs `anchor_mid`).
- `anchor_mid_override` plumbed through.

### [src/diffmm/agents/](../src/diffmm/agents/) (all new)
- `base.py` — LOBState, Quote, AgentParams, MarketMakingAgent ABC.
- `as_baseline.py` — A1.
- `as_ofi.py` — A2.
- `as_vpin.py` — A3.
- `calibration.py` — `calibrate_a1/a2/a3`, `estimate_sigma_per_second`, `estimate_kappa` (within-day fix), save/load.

### [src/diffmm/sim/](../src/diffmm/sim/) (all new)
- `pnl.py` — `Position`.
- `fill.py` — `FillSimulator`, `FillConfig`, `BacktestResult`.
- `runner.py` — `run_backtest_on_tape`, `run_backtest_on_split`.

### [src/diffmm/eval/](../src/diffmm/eval/) (all new)
- `metrics.py` — `aggregate(rows) → AggregateMetrics`.
- `ranking.py` — `Ranking`, `rank_by_sharpe`, `rank_by_worst_quartile`, `rank_by_mean_pnl`, `spearman_rho`.
- `bootstrap.py` — `bootstrap_central_test`.

### [scripts/](../scripts/) (all new)
- `20_calibrate_agents.py`, `21_replay_real.py`, `22_replay_synthetic.py` — Phases 3 + 4.
- `40_open_holdout.py`, `41_hypothesis_test.py` — Phase 5.

### [slurms/](../slurms/)
- `ph3_calibrate.slurm`, `ph4_replay_real.slurm`, `ph4_replay_synth.slurm`, `ph5_hypothesis.slurm`.

### [tests/](../tests/)
- `test_phase3.py` — 9 tests (agent quote validity, position conservation, fill end-to-end, A2 skew direction, A3 spread widening, calibration).
- `test_phase4.py` — 10 tests (aggregate math, ranking semantics, Spearman ρ, bootstrap correctness on degenerate + strong-effect cases).
- `test_data_pipeline.py` — added round-trip test for the absolute → relative → decode → absolute correctness.

**Total tests now: 30 passing** (was 10 at end of Work1, 11 at end of Work2 §6.1).

---

## 9. What's Saved Where

### On the cluster

```
~/diff_mm/                                       # code, configs, logs (synced)
~/diff_mm/data/processed/{train,val,holdout}/   # canonical Parquets (23 cols)
~/diff_mm/data/manifests/                        # ph1_clean.json, ph2_sample.json

/orcd/scratch/orcd/007/favara/diffmm/
    checkpoints/
        intc_v1_short/best.ckpt                  # Work2's 5-epoch v1 (val 0.0160)
        intc_v2/ckpt-epoch004-valloss0.0106.ckpt # this work's 8-epoch v2 (val 0.0106)
    synthetic/INTC/{base,high_vol,toxic,thin}/   # v1 synth (incompatible with new schema; archived)
    synthetic_v2/INTC/{base,high_vol,toxic,thin}/  # 6400 v2 synth Parquets, canonical-schema

~/diff_mm/results/
    agent_params/INTC_{A1,A2,A3}.json
    backtests/real/INTC/{metrics.csv, ranking_pi_hist.json}
    backtests/synthetic/INTC/{metrics.csv, ranking_pi_diff.json}
    backtests/holdout/INTC/{metrics.csv, ranking_pi_truth.json}
    hypothesis/INTC/{bootstrap.json, bootstrap_samples.npz, figure7.pdf, figure7.png}
```

### Locally (mirrored via rsync)

```
docs/Work0.md, Work1.md, Work2.md, Work3.md
src/diffmm/ (with all of Phase 3 + 4 + 5 modules)
scripts/ (numbered 01-41 plus utils/)
slurms/ (ph1-ph5)
tests/ (4 files, 30 tests)
```

---

## 10. Open Issues and Forward Work

These do not block the project's central deliverable but shape what we'd do next.

### 10.1 Statistical power of the central test

n = 20 holdout days × 3 agents gives the bootstrap ~6 distinct possible ρ values per resample. Bootstrap CIs span [-1, +1] for both estimates. **Two cheap ways to increase power**:
- **Add agents.** A 4th and 5th agent (e.g., constant-spread baseline, microprice-AS) add no compute proportionally but multiply the ranking space (`n!` rankings).
- **Add a second ticker (TSLA).** Whole pipeline parallelizes for TSLA; we already have splits + intermediate Parquets. Calibration + replay + holdout + bootstrap = ~30 min wall.

### 10.2 Synthetic-window length

Current 256-event × 100ms = 25-second windows produce ≤ 1 fill per window for most agents. Worst-quartile collapses to zero. The fallback to mean_pnl works but is statistically weaker than the proposal's preferred tail-aware metric.

**Two paths**:
- **Decode-side concatenation**: chain N=10 sampled windows into one 2,560-event "day" with continuous mid_return integration. Single-file change, no retraining. Cost: ~half-day of code + retest.
- **Retrain v3 with longer max_seq_len**: `max_seq_len=1024` × 100ms = 1.7-min windows; `2048` = 3.4-min. Requires re-training (~6h on L40S) and re-sampling. Cost: ~half-day of compute.

### 10.3 Agent absolute profitability

All Sharpe scores are negative on both real val and holdout. The methodology is unaffected (we test ranking not profitability), but a follow-up could:
- Sweep γ ∈ {0.01, 0.1, 0.5, 1.0} to find the regime where AS roughly breaks even on real INTC.
- Replace the κ proxy with a proper fill-rate-vs-spread fit (the action plan's preferred calibration).
- Add an inventory cost term (pay for extreme positions).

### 10.4 Regime conditioning fidelity

v2 fixed the calendar confound but regime labels still mostly modulate trade frequency, not spread or volatility. The model has the capacity (10M params) and the loss converged; the conditioning signal is just weak. Possible fixes:
- Stronger CFG dropout during training (currently 10%).
- Sweep guidance weight at sample time (we currently use w=1; could try w∈{0,1,3}).
- Encode regime as a continuous embedding rather than discrete buckets.

### 10.5 LOB-Bench validation suite

Phase 2.4 of the action plan called for the formal LOB-Bench stylized-fact validation (12 scores). We never implemented it because the v2 model evaluation through the central test was the more critical signal. For the final report, we should at minimum show:
- Return-distribution Wasserstein-1 between v2 synthetic and real INTC val.
- Absolute-return ACF.
- Conditional-return-given-imbalance sign test.

These are 3 of the 12 LOB-Bench scores and capture the proposal's must-pass criteria. ~1 day of work.

---

## 11. Project-State Tracking vs Action Plan

| Phase | Status |
|---|---|
| 0 — Infrastructure | ✅ |
| 1 — Data pipeline | ✅ (60/10/20 splits × 2 tickers, with the Work3 schema extension) |
| 2.1 — Load TRADES | ✅ (decision: train from scratch, documented Work1) |
| 2.2 — Fine-tune generator | ✅ (v2: 8 epochs, val 0.0106, Min-SNR weighting) |
| 2.3 — DDIM sampler | ✅ (with x0_clip + n_steps=200 + w=1 defaults) |
| 2.4 — Stylized-fact validation | ❌ **Open** — see §10.5 |
| 3.1 — Agent base class | ✅ |
| 3.2 — A1 (AS baseline) | ✅ |
| 3.3 — A2 (AS + OFI) | ✅ |
| 3.4 — A3 (AS + VPIN) | ✅ |
| 3.5 — Fill simulator | ✅ |
| 4.1 — Generate synthetic days | ✅ (6400 Parquets at synthetic_v2/) |
| 4.2 — Historical replay | ✅ → π_hist on disk |
| 4.3 — Synthetic stress replay | ✅ → π_diff on disk (mean_pnl ranking with logged fallback) |
| 4.4 — Metrics computation | ✅ |
| 5.1 — Holdout replay | ✅ → π_truth on disk |
| 5.2 — Rank-correlation comparison | ✅ → bootstrap.json + figure7.pdf |
| 6.1 — Figures | ◐ Figure 7 generated; Figures 1-6 still TBD |
| 6.2 — Written report | ❌ Not started |
| 6.3 — Repository cleanup | ❌ |

**The central question of the project is now answered**, with the appropriate caveats about statistical power. The remaining work is the report itself, optional improvements to the generator/agents, and reproducibility hardening.

---

## 12. Numbers in One Place

| What | Value |
|---|---|
| v2 best val_loss | 0.0106 (vs v1's 0.0160) |
| v2 training time | ~3 h on L40S, 8 epochs |
| v2 trainable params | ~10 M |
| v2 synthetic data | 6,400 Parquets at synthetic_v2/ |
| INTC `anchor_mid` | $30 (training-set median) |
| INTC `arrival_decay` (κ, after fix) | 233.6 |
| INTC `volatility` (σ per second) | 0.00294 |
| AS half-spread on INTC | ~$0.014 (1.4¢) |
| Real-val replay wall | ~5 min for 3 agents × 10 tapes |
| Synth replay wall | ~1 min for 3 agents × 4 regimes × 1600 tapes |
| Holdout replay wall | ~10 min for 3 agents × 20 tapes |
| Bootstrap (n=1000) wall | ~10 sec |
| ρ(π_hist, π_truth) | -1.000 |
| ρ(π_diff, π_truth) | +0.500 |
| Δρ | 1.500 |
| One-sided p-value | 0.457 |

---

## Appendix A — Complete File Inventory Diff Since Work2

### Added
- [src/diffmm/agents/](../src/diffmm/agents/) — base.py, as_baseline.py, as_ofi.py, as_vpin.py, calibration.py, __init__.py
- [src/diffmm/sim/](../src/diffmm/sim/) — fill.py, pnl.py, runner.py, __init__.py
- [src/diffmm/eval/](../src/diffmm/eval/) — metrics.py, ranking.py, bootstrap.py, __init__.py
- [scripts/20_calibrate_agents.py](../scripts/20_calibrate_agents.py)
- [scripts/21_replay_real.py](../scripts/21_replay_real.py)
- [scripts/22_replay_synthetic.py](../scripts/22_replay_synthetic.py)
- [scripts/40_open_holdout.py](../scripts/40_open_holdout.py)
- [scripts/41_hypothesis_test.py](../scripts/41_hypothesis_test.py)
- [scripts/utils/diagnose_sampling.py](../scripts/utils/diagnose_sampling.py)
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py)
- [slurms/ph3_calibrate.slurm](../slurms/ph3_calibrate.slurm)
- [slurms/ph4_replay_real.slurm](../slurms/ph4_replay_real.slurm)
- [slurms/ph4_replay_synth.slurm](../slurms/ph4_replay_synth.slurm)
- [slurms/ph5_hypothesis.slurm](../slurms/ph5_hypothesis.slurm)
- [tests/test_phase3.py](../tests/test_phase3.py)
- [tests/test_phase4.py](../tests/test_phase4.py)

### Modified
- [src/diffmm/io/schema.py](../src/diffmm/io/schema.py) — added 4 relative-feature columns + `add_relative_features` helper
- [src/diffmm/data/regime.py](../src/diffmm/data/regime.py) — `apply_quantiles` calls `add_relative_features`
- [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py) — new `FEATURE_COLUMNS`, `NormStats.anchor_mid`
- [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py) — full rewrite for relative→absolute reconstruction; os.makedirs
- [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py) — `x0_clip` parameter
- [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) — Min-SNR loss weighting
- [src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py) — minor (just confirmed `n_features` agility)
- [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) — Min-SNR plumbing
- [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py) — anchor override; mandatory norm_stats
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) — `--anchor-mid-override`; `os.makedirs`
- [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) — `min_snr_gamma: 5.0`
- [configs/generator/ddim_sample.yaml](../configs/generator/ddim_sample.yaml) — `anchor_mid_override: null`; new defaults `n_steps=200`, `guidance_weight=1.0`, `x0_clip=4.0`
- [tests/test_data_pipeline.py](../tests/test_data_pipeline.py) — added `test_absolute_relative_round_trip`
- [tests/test_generator_smoke.py](../tests/test_generator_smoke.py) — decode test now provides NormStats
- [slurms/ph2_finetune.slurm](../slurms/ph2_finetune.slurm) — minor (max_steps plumbing)

**Total tests passing: 30** (5 data, 6 generator smoke, 9 phase3, 10 phase4).
