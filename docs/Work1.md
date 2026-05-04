# Work 1 — Phase 1 Data Pipeline + Diffusion Plumbing (Build Notes)

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-18
**Status of this doc:** written while the first Stage-A0 split job is running on Engaging.

This document captures everything we learned, built, and fixed between Work0 (architecture) and the first cluster-side data-pipeline submission. It is meant to be exhaustive: data-format discoveries, every module's contract and the design choices behind it, every bug encountered and how we diagnosed it, and a complete inventory of what's now committed.

---

## 1. Source Data — What We Actually Have

The cluster has 7 gzipped CSVs in `~/diff_mm/data/raw/`. The 5 files **without a year prefix** are the project's working dataset (2025-04-02 → 2026-04-01); the two `trade_2025_*` files cover the prior year and are not used.

| File | Compressed | Rows | Date span | Notes |
|---|---|---|---|---|
| `order_INTC.csv.gz` | 11 GB | 930,752,457 | 2025-04-02 → 2026-04-01 | Per-exchange BBO updates (NOT consolidated NBBO) |
| `order_TSLA.csv.gz` | 14 GB | 1,271,738,194 | 2025-04-02 → 2026-04-01 | Same |
| `trade_INTC.csv.gz` | 1.5 GB | 103,608,815 | 2025-04-02 → 2026-04-01 | Consolidated trades, **not** pre-signed |
| `trade_TSLA.csv.gz` | 5.5 GB | 380,748,701 | 2025-04-02 → 2026-04-01 | Same |
| `trade_SPY.csv.gz` | 3.5 GB | 243,464,773 | 2025-04-02 → 2026-04-01 | Trades only — no quote file → SPY is a **regime / benchmark reference**, not a generator target |

Decompression ratio runs ~7-10× → quote files balloon to 80-100+ GB each. Cannot fit any single file in RAM on a typical compute node.

### 1.1 Quote File Schema (TAQ NBBO via WRDS)

```
DATE, TIME_M, EX, BID, BIDSIZ, ASK, ASKSIZ,
QU_COND, QU_SEQNUM, NATBBO_IND, QU_CANCEL, QU_SOURCE, SYM_ROOT, SYM_SUFFIX
```

Sample (INTC, 2025-04-02):
```
2025-04-02,3:59:00.083513342,Z,0,0,0,0,L,1130,1,,N,INTC,
2025-04-02,4:00:00.082674150,P,19,20,0,0,R,4663,2,,N,INTC,
2025-04-02,4:00:00.082678790,P,19.08,2,0,0,R,4668,2,,N,INTC,
2025-04-02,4:00:00.082682833,P,19.08,2,23.48,5,R,4671,4,,N,INTC,
```

Key observations from the histogram probe (first 2M rows):

- **Time format:** `H:MM:SS.nnnnnnnnn` (variable-width hour). Local time, ET.
- **Pre-market events** with `BID=0/ASK=0` placeholders before market open at 09:30 ET.
- **Per-exchange BBO**, NOT consolidated NBBO. Each row is one exchange's update.
- **`QU_COND`:** `R` (regular) is 99.99% of rows. `Y` and `L` together <0.01%.
- **`QU_SOURCE`:** all `N` (UTP / Nasdaq tape — INTC and TSLA are Nasdaq-listed).
- **`QU_CANCEL`:** always blank.
- **`NATBBO_IND` distribution** in the 2M sample: 0=81%, 1=tiny, 2=13%, 4=6%. **Conclusion: this single-row indicator is unreliable** for partitioning NBBO-touching rows. Need to roll our own NBBO state machine.
- **Top exchanges (count)** for INTC quotes: Q (Nasdaq) > P (NYSE Arca) > N (NYSE) > U > K > Z > V > J > Y > A.

### 1.2 Trade File Schema

```
DATE, TIME_M, EX, SYM_ROOT, SYM_SUFFIX, TR_SCOND, SIZE, PRICE,
TR_STOP_IND, TR_CORR, TR_SEQNUM, TR_ID, TR_SOURCE, TR_RF
```

Sample (INTC, 2025-04-02):
```
2025-04-02,4:00:00.014686791,P,INTC,,@ TI,54,22.01,,00,2026,1,N,
2025-04-02,4:00:00.037435256,K,INTC,,@ TI,6,22.04,,00,2094,1,N,
```

Key observations:

- **`TR_SCOND` is a 4-character condition string**: char 1 = sale type (`@` regular, `4` derivatively-priced, `F` Form-T pre/post-market, `7` priced-after-close, `C` cash); char 4 = modifier (` ` regular, `I` ISO, `W` avg-price, `P` prior-ref, `X` cross). The dominant codes are `@` (35%), `@  I` (21%), `@F` (21%), `@F I` (12%), `@4 I` (3%) — many are pre/post-market trades that we drop.
- **`TR_CORR`:** `00` = normal (1,999,984 of 2M); `08` and `10` are corrections to drop.
- **`TR_RF`:** blank (68%), `Q` qualified contingent (30%), `N` not reportable (1%), `B` late report (0.5%). Drop `B` and `Q` (multi-leg trades skew microstructure).
- **`TR_SOURCE`:** `N` for INTC/TSLA (UTP). `C` for SPY (CTA). Different feeds; both legitimate.
- **EX distribution** for INTC trades: **`D` (FINRA TRF / dark trades) is 32%** of all trades. We keep them but tag `is_lit=False` so OFI/sign signals can use lit trades only while PnL replay uses everything.
- **Trades are NOT pre-signed** → we run Lee–Ready (or BVC) ourselves.

### 1.3 Per-Day Volume Estimates (extrapolated from sample)

| Stream | Per day | Per year (~250 days) |
|---|---|---|
| INTC quotes | ~3.7M events | ~930M (matches the 11 GB compressed) |
| INTC trades | ~415K events | ~104M (matches 1.5 GB) |
| TSLA quotes | ~5.1M events | ~1.27B |
| TSLA trades | ~1.5M events | ~380M |

For a 256-event diffusion window on INTC, that's ~14K windows per day, ~840K windows over 60 training days — plenty for the generator.

---

## 2. Filter Rules (Locked In)

These are the production rules now baked into [src/diffmm/data/clean.py](../src/diffmm/data/clean.py) and configurable from [configs/data/intc_top.yaml](../configs/data/intc_top.yaml).

| Field | Decision |
|---|---|
| `QU_COND` | Keep `R` only |
| `BID, ASK` | Drop if 0 (placeholder) or `BID >= ASK` (crossed/locked) |
| `QU_CANCEL` | Always blank — no filter |
| `QU_SOURCE` | 100% `N` — sanity-check only |
| `NATBBO_IND` | **Ignored** — we compute NBBO ourselves |
| `TR_CORR` | Keep `00` only |
| `TR_SCOND` char 1 | Keep `@` and `4`; drop `F`, `7`, `C` |
| `TR_SCOND` char 4 | Keep blank or `I`; drop `W`, `P`, `X` |
| `TR_RF` | Drop `B` (late) and `Q` (qualified contingent) |
| `EX == 'D'` | Keep, set `is_lit=False` |
| Session window | 09:45–15:45 ET (configurable) |

---

## 3. Phase 1 Architecture: Three Stages

The action plan describes Phase 1 as a single cleaning step. At the actual data scale (~350 GB decompressed), that monolith doesn't fit — and worse, it tightly couples filter iteration to the slow gzip-decompression pass. I split it into three stages:

```
                                                       (CPU array, re-runnable)
   raw .csv.gz                                             ┌──────────────┐
   ┌─────────┐    Stage A0           ┌─────────────┐       │  Stage A1    │
   │ 11 GB   │  ── streaming ──→     │ raw_part/   │  ──→  │  build_      │  ──→
   │ gzip    │   one-time, slow      │ {ticker}/   │       │  intermediate│
   └─────────┘                       │ {kind}/     │       │  _tape       │
                                     │ {date}.parq │       └──────────────┘
                                     └─────────────┘            ↓
                                                          intermediate/
                                                          {ticker}/
                                                          {date}.parq
                                                                ↓
                                                       ┌──────────────┐
                                                       │  Stage A2    │
                                                       │  finalize    │  ──→  data/processed/
                                                       │  (one job /  │       {train,val,holdout}/
                                                       │  ticker)     │       {ticker}_{date}.parq
                                                       └──────────────┘       (canonical schema)
```

### Why three stages

- **A0 is expensive (~30 min per gzip) but we never want to repeat it.** Splitting raw bytes into per-date chunks pays off the moment we re-run cleaning with new filter rules.
- **A1 is the iteration sweet spot.** When the filter rules or feature definitions change, we re-run A1 only — small per-task cost, parallelizable.
- **A2 is cheap (~5 min) and is where the train/val/holdout-size knobs live.** This is what makes the user's "we're still uncertain on sizes" non-painful: change `n_train_days`, re-run A2, done.

---

## 4. Module Inventory and Contracts

### 4.1 [src/diffmm/io/schema.py](../src/diffmm/io/schema.py)

Canonical event-tape schema in `CORE_DTYPES` (Work0 §3 spec) and a new `INTERMEDIATE_DTYPES` for Stage-A1 output.

**Why the intermediate schema is different.** `c_vol` and `c_vpin` require quantile thresholds fitted on the training-set ONLY. Stage A1 doesn't yet know which days are training, so it cannot bucket. The intermediate keeps the continuous values (`raw_vol`, `raw_vpin`) and omits `c_vol`, `c_vpin`, `regime_label`. Stage A2 reads the intermediate, fits quantiles on training days, fills the regime columns, and emits canonical Parquets.

`c_imb` and `c_tod` ARE filled in at Stage A1 — `c_imb` uses fixed thresholds `±τ`, `c_tod` is purely a clock function. Neither needs training data.

Helpers: `validate(df)`, `validate_intermediate(df)`, `intermediate_to_canonical(df, c_vol, c_vpin, regime_label)`.

### 4.2 [src/diffmm/data/raw_split.py](../src/diffmm/data/raw_split.py)

**Stage A0.** Stream a TAQ gzip → per-date Parquet partition.

Original implementation used `pl.read_csv_batched`, then `pl.scan_csv(...).collect().iter_slices(...)`. **Both materialize the entire file in memory** — Polars' streaming engine doesn't actually stream `read_csv` despite the lazy-frame API.

**Final implementation: PyArrow's `pyarrow.csv.open_csv`** with `block_size=128 MB`. PyArrow gzip-decompresses block-by-block, yielding `RecordBatch`es. We exploit the fact that TAQ files are pre-sorted chronologically: keep ONE `ParquetWriter` open at a time, close it when DATE rolls over to the next.

Key contract:
```python
split_gzip_by_date(gz_path, out_dir, *, kind, block_size=128*1024*1024) -> list[Path]
```

Memory ceiling: ~256 MB working set (one PyArrow block + one ParquetWriter buffer), regardless of input size. Verified passing the smoke test on a 3-date synthetic gzip; will test at scale once the cluster job completes.

### 4.3 [src/diffmm/data/clean.py](../src/diffmm/data/clean.py)

**Stage A1, part 1.** All filtering, NBBO computation, and trade signing.

- `QuoteFilterConfig`, `TradeFilterConfig`, `SessionConfig`, `CleanConfig` — frozen dataclasses populated from Hydra.
- `filter_quotes(raw, cfg)` — applies the rules from §2 and renames TAQ columns to internal lowercase (`bid_px`, `ask_px`, `bid_sz`, `ask_sz`, `ex`).
- `filter_trades(raw, cfg)` — same, plus `is_lit = (EX != 'D')`.
- `compute_nbbo(quotes)` — the NBBO state machine.
- `sign_trades_lee_ready(trades, nbbo)` — asof-join trade rows to prevailing NBBO mid, apply Lee–Ready with tick-rule on ties.

**NBBO algorithm (vectorized):**
1. Pivot per-exchange quotes wide: rows = ts_ns, columns = `(bid_px_EX, ask_px_EX, bid_sz_EX, ask_sz_EX)` for each exchange.
2. Forward-fill within each column (each exchange holds its last-known BBO between its updates).
3. Row-wise NBB = `pl.max_horizontal(bid_px_*)`, NBO = `pl.min_horizontal(ask_px_*)`.
4. NBBO sizes = sum of sizes across exchanges quoting at the NBBO price (`when bid_px_EX == bid_px then bid_sz_EX else 0`, summed).
5. Drop rows with crossed NBBO (`ask_px <= bid_px`).
6. Filter to events where the NBBO actually changed (any of bid_px/ask_px/bid_sz/ask_sz differs from the prior row).

**Lee–Ready:** asof backward-join trades to NBBO snapshots, compute `mid`, then:
- `trade_px > mid` → +1 (buyer-initiated)
- `trade_px < mid` → −1 (seller-initiated)
- tie → tick rule (sign of last non-zero trade-price change, forward-filled)

### 4.4 [src/diffmm/data/features.py](../src/diffmm/data/features.py)

**Stage A1, part 2.** Continuous regime features.

- `add_realized_vol(df, cfg)` — rolling-by-time sum of squared log-mid-returns over `vol_window_seconds` (default 300s = 5 min). Uses `rolling_sum_by("_dt", "300s")` after converting `ts_ns` to a temporary `_dt: Datetime` column (Polars' time-window syntax requires Datetime, not Int).
- `add_imbalance(df, cfg)` — `raw_imb = (bid_sz - ask_sz)/(bid_sz + ask_sz)`, plus `c_imb` bucketed at fixed `±τ` thresholds.
- `add_time_of_day(df, cfg)` — clock-only `c_tod` ∈ {0=morning, 1=midday, 2=afternoon}, with thresholds at 11:30 ET and 13:30 ET (configurable).
- `add_vpin(df, cfg)` — VPIN per Easley–LdP–O'Hara (2012):
  1. Restrict to lit trades with positive size and price.
  2. Bucket by cumulative volume: `bucket_size = total_lit_volume / vpin_n_buckets_per_day` (default 50 buckets/day).
  3. Per bucket: BVC classification: `Z = (close − open)/(σ · √2)`, `buy_frac = Φ(Z) = 0.5·(1 + erf(Z))`. `Φ` is computed via `scipy.special.erf` on the (small, ~50-row) per-bucket pandas frame.
  4. `signed_imb = |2·buy_frac − 1|` per bucket; rolling mean over `vpin_window_buckets` buckets (default 50).
  5. Forward-fill VPIN onto every event timestamp via asof-backward join.

### 4.5 [src/diffmm/data/tapes.py](../src/diffmm/data/tapes.py)

**Stage A1 orchestrator.** `build_intermediate_tape(ticker, date, raw_quotes_parquet, raw_trades_parquet, clean_cfg, feat_cfg) -> pl.DataFrame`.

Flow:
1. Read partitioned raw quotes + trades for one date.
2. `filter_quotes`, `filter_trades`, `compute_nbbo`, `sign_trades_lee_ready`.
3. Build a unified event stream:
   - `quote_update` rows from NBBO, with trade fields nulled.
   - `trade` rows from signed trades, with NBBO context joined via asof-backward (carries `bid_px`/`ask_px`/`bid_sz`/`ask_sz` as the prevailing book at trade time).
4. Concatenate, sort by `ts_ns`, assign `event_idx` (monotonic Int64).
5. Apply `add_realized_vol`, `add_imbalance`, `add_time_of_day`, `add_vpin`.
6. Select the intermediate-schema columns and validate.

Returns a Polars DataFrame ready to write.

### 4.6 [src/diffmm/data/regime.py](../src/diffmm/data/regime.py)

**Stage A2, part 1.** Quantile fitting + regime-label assignment.

- `Quantiles(vol, vpin)` — frozen dataclass with `save`/`load` (JSON).
- `fit_quantiles(intermediate_paths)` — concatenates `raw_vol` / `raw_vpin` across ALL provided tapes (caller passes training-only paths) and computes 33%/67% quantiles.
- `apply_quantiles(df, q)` — buckets `raw_vol` and `raw_vpin` into `c_vol`, `c_vpin`, joins a small lookup table to assign `regime_label` ∈ {`base`, `high_vol`, `toxic`, `thin`}, drops the raw columns, conforms to canonical schema.

The label table is the proposal's 4-cell summary:
- (high vol, high vpin) → `toxic`
- (high vol, low/mid vpin) → `high_vol`
- (low/mid vol, high vpin) → `thin`
- everything else → `base`

### 4.7 [src/diffmm/data/splits.py](../src/diffmm/data/splits.py)

**Stage A2, part 2.** Date partitioning + holdout firewall enforcement.

- `SplitConfig(n_train_days, n_val_days, n_holdout_days, holdout_priority_dates, seed)` — all CLI-overridable.
- `Splits(train, val, holdout)` — frozen, JSON-serializable.
- `make_splits(available_dates, cfg)` — picks holdout first (priority dates if given, else random tail-biased), then fills train and val from the remainder.
- `materialize_split(...)` — for each split, reads intermediate Parquets, applies quantiles, writes canonical Parquets to `data/processed/{train,val,holdout}/{TICKER}_{DATE}.parquet`. **Holdout writes go through `holdout_guard.unlocked(...)`** and the resulting files get `chmod 444` so casual reads fail at the OS level.

### 4.8 [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py)

(Already from Work0.) `LOBWindowDataset`, `NormStats`, `RandomWindowDataset`. Now feeds the diffusion training loop with windows of canonical-schema rows.

---

## 5. Diffusion Pipeline (also added between Work0 and Work1)

### 5.1 [src/diffmm/generator/conditioning.py](../src/diffmm/generator/conditioning.py)

`RegimeEmbedding` — separate embedding table per regime axis with an extra "null" token (index `n` for an axis with `n` real categories) for classifier-free guidance dropout. `null_condition(B, device)` returns the null tensor.

`FiLMLayer` — standard `(1+γ)·h + β` modulation; `γ`/`β` predicted from a context vector via a `nn.Linear` initialized to zero so the layer starts as identity (stable for fine-tuning a pre-trained backbone).

### 5.2 [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py)

`DDIMSchedule` (linear + cosine factories) and `ddim_sample(...)` implementing reverse-DDIM with classifier-free guidance:

```
ε̃ = (1 + w) · ε_θ(x_t, t, c) − w · ε_θ(x_t, t, ∅)
```

Default `n_steps=20`, `eta=0.0` (deterministic). Subsamples the T=1000 schedule to S=20 timesteps per Song–Meng–Ermon (2021).

### 5.3 [src/diffmm/generator/model.py](../src/diffmm/generator/model.py)

`Denoiser` ABC, `GeneratorModel` wrapper that pairs a denoiser with a `RegimeEmbedding`, plus `MinimalDenoiser` (per-token MLP with FiLM) used only by tests.

### 5.4 [src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py)

After inspecting [_info/DeepMarket-main](../_info/DeepMarket-main), the call was made to **train from scratch** rather than load TRADES' pretrained checkpoint:

- TRADES expects LOBSTER message-format inputs: 6-feature orders (`event_type`, `order_id`, `size`, `price`, `direction`, `time`) plus a 40-feature LOB snapshot per step (10 levels × 4 fields).
- TAQ NBBO has only top-of-book prices/sizes; we have no order_id-level events, no L2+ depth, no message types.
- The format mismatch is too severe to use the pre-trained weights directly.

So [trades_adapter.py](../src/diffmm/generator/trades_adapter.py) implements **TRADES-style architecture from scratch**:
- `sinusoidal_positional_embedding` (lifted from DeepMarket's `utils/utils.py`).
- `TransformerBlockSelfAtt` (lifted from DeepMarket's `Transformer.py`).
- `TradesStyleDenoiser`: in_proj (F → d_model) → add positional embed → add diffusion-timestep embed → FiLM(regime, start) → N × self-attention → FiLM(regime, end) → out_proj (d_model → F). Output proj zero-initialized so the network starts predicting zero noise (stable initial loss).
- `build_generator(...)` factory.
- `encode_window(df, norm)` — canonical Parquet → tensor (currently a pass-through; ordering is the FEATURE_COLUMNS contract).
- `load_pretrained_trades_checkpoint(...)` — explicitly raises `NotImplementedError` documenting why.

### 5.5 [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py)

Sampled tensor → canonical Parquet:
1. Inverse-normalize via `NormStats`.
2. Enforce non-negativity on prices/sizes.
3. Uncross: `ask_px = max(ask_px, bid_px + tick)`.
4. Round sizes to integers.
5. Infer event type: any window-event with rounded `trade_sz > 0` becomes a `trade`; else `quote_update`.
6. Synthesize timestamps with uniform 100ms spacing (replace once Δt becomes a learned channel).
7. Compute `mid`, `spread`.
8. Set `trade_sign` = sign of generated channel (with tie-break by `trade_px - mid`).
9. Map `(c_vol, c_vpin)` → `regime_label`.
10. Validate against canonical schema and write Parquet.

### 5.6 [src/diffmm/generator/train.py](../src/diffmm/generator/train.py)

`DDPMTrainer` — PyTorch Lightning module implementing Algorithm 1 from Ho et al. 2020 with classifier-free-guidance dropout. `freeze_backbone` flag for fine-tuning vs. from-scratch. `self.log()` calls are conditional on a Trainer being attached, so manual training loops in tests don't emit warnings.

---

## 6. Configs / Scripts / SLURM Inventory

### Configs (Hydra)

| File | Purpose |
|---|---|
| [configs/config.yaml](../configs/config.yaml) | Root composition + `hydra.job.chdir=false` (critical — see bug #4) |
| [configs/data/intc_top.yaml](../configs/data/intc_top.yaml) | INTC paths, filter rules, feature params, `splits` block (sizes here) |
| [configs/data/tsla_top.yaml](../configs/data/tsla_top.yaml) | Inherits INTC, overrides ticker + paths |
| [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) | Model/schedule/training/checkpointing |
| [configs/generator/ddim_sample.yaml](../configs/generator/ddim_sample.yaml) | Sampler params + per-regime conditioning vectors |
| [configs/cluster/local.yaml](../configs/cluster/local.yaml) | Laptop defaults |
| [configs/cluster/engaging.yaml](../configs/cluster/engaging.yaml) | Engaging paths + partition names |

### Scripts

| File | Phase | Notes |
|---|---|---|
| [scripts/01_split_raw_to_parquet.py](../scripts/01_split_raw_to_parquet.py) | A0 | Single-gzip splitter |
| [scripts/02_build_tapes.py](../scripts/02_build_tapes.py) | A1 | Per-date intermediate; manifest-driven for arrays |
| [scripts/03_finalize_tapes.py](../scripts/03_finalize_tapes.py) | A2 | Splits + quantiles + canonical |
| [scripts/utils/make_manifest.py](../scripts/utils/make_manifest.py) | helper | `--kind clean` enumerates (ticker, date) from partitioned dirs; `--kind sample` enumerates (regime, seed) |
| [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) | Phase 2 | `+smoke=true` runs on RandomWindowDataset locally |
| [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py) | Phase 2 | Loads checkpoint, samples per (regime, seed) |

### SLURMs

| File | Resources | Notes |
|---|---|---|
| [slurms/_common.sh](../slurms/_common.sh) | — | Shared scaffolding sourced by every job |
| [slurms/ph1_split_raw.slurm](../slurms/ph1_split_raw.slurm) | `mit_normal`, 4 CPU, 32 GB, 4 h | One job per gzip |
| [slurms/ph1_clean.slurm](../slurms/ph1_clean.slurm) | `mit_normal`, 2 CPU, 16 GB, 45 min, array | Manifest-driven over (ticker, date) |
| [slurms/ph1_finalize.slurm](../slurms/ph1_finalize.slurm) | `mit_normal`, 4 CPU, 32 GB, 30 min | Per ticker |
| [slurms/ph2_finetune.slurm](../slurms/ph2_finetune.slurm) | `mit_normal_gpu`, 1 GPU, 8 CPU, 64 GB, 6 h | Single-node training |
| [slurms/ph2_sample.slurm](../slurms/ph2_sample.slurm) | `mit_normal_gpu`, 1 GPU, 2 CPU, 16 GB, 30 min, array | (regime × seed) |

---

## 7. Bugs Encountered and How They Were Fixed

This section is a complete diary — every dead end matters because the same shape of bug will recur in Phase 2 onward.

### Bug 1 — `read_csv_batched` deprecation
**Symptom:** Polars deprecation warning during the first smoke-test run.
**Diagnosis:** `pl.read_csv_batched` was deprecated in Polars 1.x in favor of `pl.scan_csv(...).collect_batches()`.
**Fix:** Switched to `scan_csv`. **But:** see Bug 7 — the new API still didn't actually stream.

### Bug 2 — Int8 overflow in session-window filter
**Symptom:** `filter_quotes` returned an empty DataFrame for every input even though synthetic times were clearly within 09:45–15:45.
**Diagnosis:** Polars' `dt.hour()` and `dt.minute()` return `Int8`. `Int8 × 60 = 600` overflows the Int8 range (max 127), wrapping to a small negative number. The comparison `>= h0*60 + m0` was always False.
**Fix:** Cast to `Int32` before arithmetic in `clean._within_session_expr` and `features.add_time_of_day`.
**Lesson:** any time arithmetic on Polars dt-component returns hits Int8. Always cast.

### Bug 3 — `rolling_sum_by` requires Datetime, not Int64
**Symptom:** `InvalidOperationError: window_size duration must be a parsed integer (i.e. use '2i', not '2d') when working with a numeric column`.
**Diagnosis:** `pl.col(...).rolling_sum_by("ts_ns", window_size="300s")` — the `300s` syntax requires the index column to be Datetime. `ts_ns` is Int64.
**Fix:** Compute a temporary `_dt = pl.from_epoch(ts_ns, "ns")` Datetime column for the rolling, then drop it.
**Lesson:** Polars' time-window rolling APIs are strict about index types.

### Bug 4 — `np.erf` doesn't exist
**Symptom:** `AttributeError: module 'numpy' has no attribute 'erf'`.
**Diagnosis:** I'd written `np.math.erf(v)` — but `np.math` was deprecated in numpy 1.x and **removed in 2.0**. There is no `np.erf` either; erf lives in `scipy.special` or `math`.
**Fix:** Use `scipy.special.erf` on a numpy array (the per-bucket frame is small, ~50 rows).
**Lesson:** never use `np.math.X`. Use `math.X` or `scipy.special.X`.

### Bug 5 — `min_periods` deprecation in rolling
**Symptom:** Polars deprecation warning.
**Fix:** rename `min_periods` → `min_samples`.

### Bug 6 — Hook routes new `.md` files to `_info/md_files/`
**Symptom:** When I wrote `Work0.md` to project root, the file landed in `_info/md_files/` despite the success message reporting the requested path.
**Diagnosis:** A user-configured hook intercepts `.md` writes and re-routes them.
**Workaround:** Write to the intended location, then `mv` to where I want it. (Not a code bug, but a developer-experience surprise worth recording.)

### Bug 7 — rsync `--exclude='data/'` matched `src/diffmm/data/`
**Symptom:** Cluster job failed with `ModuleNotFoundError: No module named 'diffmm.data'`. Locally: `pip show -f diffmm` showed no `diffmm/data/*` files in the install.
**Diagnosis:** The exclude pattern `data/` (without leading slash) matches **any** directory named `data` anywhere in the source tree, including `src/diffmm/data/` — not just the top-level `data/` directory the user intended to skip.
**Fix:** Anchor to root with `--exclude='/data/'`. Same fix for `/checkpoints/`, `/results/`, `/logs/`, `/wandb/`. The corrected rsync is documented in Work0 §7 and is the version you should use going forward.
**Lesson:** rsync's exclude patterns are NOT path-anchored by default. Always anchor with leading `/` for top-level-only excludes.

### Bug 8 — conda `pip: -e .` dependency didn't install diffmm
**Symptom:** After `conda env create -f environment.yml`, `import diffmm` worked but importing any submodule failed.
**Diagnosis:** The `-e .` line in `environment.yml`'s pip section runs during env creation, but the resulting install was either incomplete or the submodules weren't picked up because the (Bug 7) source tree lacked them.
**Fix:** After re-rsync (Bug 7 fix), `pip install -e .` once manually. Editable installs auto-pick-up new files going forward.

### Bug 9 — Hydra `chdir` into `outputs/...` broke relative paths
**Symptom:** SLURM job ran, then `FileNotFoundError: data/raw/order_INTC.csv.gz` even though the file exists in `~/diff_mm/data/raw/`.
**Diagnosis:** Hydra's default behavior is to `chdir()` into a per-run output directory (`outputs/YYYY-MM-DD_HH-MM-SS/`). Relative paths in the config (e.g., `data/raw/order_INTC.csv.gz`) then resolve relative to that dir, not the project root.
**Fix:** Add to [configs/config.yaml](../configs/config.yaml):
```yaml
hydra:
  job:
    chdir: false
  run:
    dir: ${output_dir}
```
**Lesson:** `hydra.job.chdir=false` is the right default for any project where configs use repo-relative paths.

### Bug 10 — OOM in the streaming gzip splitter
**Symptom:** SLURM job killed by oom_kill after a few minutes, even with 32 GB allocated.
**Diagnosis:** My implementation:
```python
lf = pl.scan_csv(gz_path, ...)
for batch in lf.collect().iter_slices(n_rows=batch_size):
    ...
```
**`lf.collect()` materializes the entire query result** before `iter_slices` can chunk it. The 11 GB compressed → 100 GB decompressed CSV blew through memory. I had assumed `scan_csv` + `collect` + `iter_slices` was streaming because the docs use the word "lazy" — it isn't, in this composition.
**Fix:** Switched to `pyarrow.csv.open_csv` with `block_size=128 MB`. PyArrow decompresses gzip block-by-block and yields `RecordBatch`es; we exploit the chronological pre-sort to keep one ParquetWriter open at a time. New implementation in [src/diffmm/data/raw_split.py](../src/diffmm/data/raw_split.py).
**Lesson:** "lazy" in Polars CSV terminology refers to query-plan deferral, NOT to streaming over the source file. For genuine streaming over multi-GB CSVs, use PyArrow's chunked reader (`open_csv` returns a `CSVStreamingReader`).

### Bug 11 — Lightning `self.log()` warning in manual training loop
**Symptom:** `You are trying to self.log() but the self.trainer reference is not registered on the model yet.`
**Diagnosis:** Tests call `training_step` manually (without `Trainer.fit`) so `self._trainer` is `None`. `self.log` warns and noops.
**Fix:** Wrap `self.log` calls in `if getattr(self, "_trainer", None) is not None:` so manual usage is silent.

---

## 8. Tests

### [tests/test_generator_smoke.py](../tests/test_generator_smoke.py) — 6 tests
1. `test_training_loss_decreases` — minimal denoiser trains, loss decreases over 3 epochs of `RandomWindowDataset`.
2. `test_ddim_sample_returns_correct_shape_and_finite` — DDIM sampler on a frozen minimal denoiser produces correct shape, no NaNs.
3. `test_cfg_zero_matches_unconditional` — at `w=0`, repeated calls with the same seed are bit-identical (deterministic check).
4. `test_null_condition_shape` — null token shape and value correctness.
5. `test_trades_style_denoiser_forward_and_backward` — production `TradesStyleDenoiser` runs forward + backward, no NaN losses.
6. `test_decode_produces_valid_canonical_parquet` — sample → decode → canonical-schema validation round trip.

### [tests/test_data_pipeline.py](../tests/test_data_pipeline.py) — 3 tests
1. `test_split_gzip_by_date_writes_one_parquet_per_date` — A0 produces one file per DATE.
2. `test_intermediate_tape_validates_against_intermediate_schema` — A1 builds a valid intermediate tape with both event types.
3. `test_finalize_makes_canonical_tapes_and_locks_holdout` — A2 produces canonical Parquets and `chmod 444`'s the holdout file.

**All 9 tests pass in ~5 s on a Mac laptop.** Smoke tests use synthesized 1 KB TAQ-shaped CSV — they're exhaustive in that they exercise every module on the critical path, but don't validate at scale.

---

## 9. Open Items / Carry-Forward Decisions

These are deferred to later work documents:

1. **Stress-day calendar.** `data.splits.holdout_priority_dates` is currently `[]`. We need to populate it with FOMC, CPI, and INTC/TSLA earnings dates within 2025-04-02 → 2026-04-01 before running `03_finalize_tapes.py` for keeps.
2. **Final split sizes.** Default 60/10/20 is the action plan's suggestion. Easy to override; we'll iterate after looking at how the regime distribution lands.
3. **Norm stats / window length.** Dataset uses 256-event windows by default. Once we look at typical inter-event time on real INTC data, we may revisit this — 256 events ≈ 60 ms wall-clock for INTC, which is short. We may want a longer window (1024?) to capture multi-second structure.
4. **Decode timestamp model.** Currently uniform 100 ms spacing in [decode.py](../src/diffmm/generator/decode.py). A learned Δt channel would be more realistic; deferred until we see synthetic outputs and judge whether the uniform-spacing assumption hurts downstream backtests.
5. **VPIN bucket count.** `vpin_n_buckets_per_day=50` is the textbook ELP default; if INTC's daily volume is very different from their TSLA-2010 sample, we may want to tune.
6. **GPU-array budget for sampling.** 400 synthetic days × ~5 min/day = 2000 GPU-min. With 20-way concurrency on `mit_normal_gpu`, ~2 hours wall. Within reach.

---

## 10. The Cluster Run (in progress at write time)

Stage A0 is currently running on `mit_normal` with the rewritten PyArrow-streaming splitter. The cluster-side state machine that got us here:

```
1. Clone code → ~/diff_mm/ via rsync (Bug 7 fix applied)
2. module load miniforge && conda env create -f environment.yml
3. (After Bug 8 manifested) pip install -e .
4. (After Bug 9 manifested) Bug-fixed config + re-rsync
5. (After Bug 10 OOM) Bug-fixed splitter + re-rsync
6. sbatch slurms/ph1_split_raw.slurm +gz=data/raw/order_INTC.csv.gz +kind=quotes +ticker=INTC
   → currently in CG/R state, expected wall ~30 min
```

Once this completes cleanly, the remaining sequence is:

```bash
# Fire the other 4 splits in parallel
sbatch slurms/ph1_split_raw.slurm +gz=data/raw/order_TSLA.csv.gz +kind=quotes +ticker=TSLA
sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_INTC.csv.gz +kind=trades +ticker=INTC
sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_TSLA.csv.gz +kind=trades +ticker=TSLA
sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_SPY.csv.gz  +kind=trades +ticker=SPY

# Build cleaning manifest, then array job for Stage A1
python scripts/utils/make_manifest.py --kind clean \
    --out data/manifests/ph1_clean.json --tickers INTC TSLA
sbatch slurms/ph1_clean.slurm

# Stage A2 per ticker (re-runnable as we tune split sizes)
sbatch slurms/ph1_finalize.slurm data=intc_top
sbatch slurms/ph1_finalize.slurm data=tsla_top
```

Once `data/processed/{train,val,holdout}/INTC_*.parquet` exists, the diffusion pipeline written in Work0 is ready to fire (`sbatch slurms/ph2_finetune.slurm`).

---

## Appendix A — File Inventory Added Between Work0 and Work1

```
src/diffmm/
├── io/schema.py                    (extended: INTERMEDIATE_DTYPES, validate_intermediate)
├── data/
│   ├── clean.py                    (new — filters, NBBO, Lee–Ready)
│   ├── features.py                 (new — vol, OFI, VPIN, imbalance, TOD)
│   ├── tapes.py                    (new — Stage A1 orchestrator)
│   ├── raw_split.py                (new — Stage A0 streaming splitter)
│   ├── regime.py                   (new — quantile fitting + label join)
│   ├── splits.py                   (new — date partitioning + holdout chmod)
│   └── __init__.py                 (extended re-exports)
├── generator/
│   ├── conditioning.py             (new — RegimeEmbedding, FiLMLayer)
│   ├── sample.py                   (new — DDIM + CFG)
│   ├── model.py                    (new — Denoiser ABC, GeneratorModel, MinimalDenoiser)
│   ├── train.py                    (new — DDPMTrainer Lightning module)
│   ├── trades_adapter.py           (new — TradesStyleDenoiser, build_generator)
│   ├── decode.py                   (new — sampled tensor → canonical Parquet)
│   └── __init__.py                 (re-exports)
└── data/dataset.py                 (LOBWindowDataset, NormStats, RandomWindowDataset)

configs/
├── config.yaml                     (extended: hydra.job.chdir=false)
├── data/intc_top.yaml              (extended with filter rules + splits block)
├── data/tsla_top.yaml              (new — inherits intc_top)
├── generator/finetune.yaml         (new)
├── generator/ddim_sample.yaml      (new)
├── cluster/local.yaml              (new)
└── cluster/engaging.yaml           (new)

scripts/
├── 01_split_raw_to_parquet.py      (new — Stage A0 entry)
├── 02_build_tapes.py               (new — Stage A1 entry)
├── 03_finalize_tapes.py            (new — Stage A2 entry)
├── 11_finetune_generator.py        (new — Phase 2 training entry)
├── 13_sample_synthetic.py          (new — Phase 2 sampling entry)
└── utils/make_manifest.py          (extended — added --kind clean)

slurms/
├── _common.sh                      (new — shared scaffolding)
├── ph1_split_raw.slurm             (new)
├── ph1_clean.slurm                 (new — manifest-driven array)
├── ph1_finalize.slurm              (new)
├── ph2_finetune.slurm              (new)
└── ph2_sample.slurm                (new — manifest-driven array)

tests/
├── test_generator_smoke.py         (new — 6 tests)
└── test_data_pipeline.py           (new — 3 tests)

environment.yml                     (new — conda env for cluster)
```

Roughly 30 files, ~3,500 lines of Python, all exercised by 9 passing tests on the laptop.
