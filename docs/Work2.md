# Work 2 — Cluster Execution, End-to-End Pipeline, First Synthetic Data

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-20
**Covers:** Everything between Work1 (local smoke + Phase-1 cleaning code) and the first full 400-task synthetic-data generation on the cluster.

This is the exhaustive build log for Phases 1–2 on Engaging: every module exercised at scale, every bug encountered, every diagnostic run, and the quantitative outputs we now have on disk. It's intended as a reference for Phase 3 onward and for the final report's methodology section.

---

## 0. Executive Summary

Starting from Work1 (working smoke tests on random data, empty cluster), we have reached:

- **Cleaned canonical tapes on disk**: 60 train + 10 val + 20 holdout days × 2 tickers (INTC, TSLA).
- **Trained diffusion generator**: 5-epoch fine-tune on INTC, val loss 0.0160.
- **Full synthetic dataset**: 6400 synthetic event-tape Parquets across 4 regimes × 100 seeds.
- **Documented end-to-end pipeline**: Stage A0 → A1 → A2 for data, fine-tune + sample + decode for generator.

What's **not** yet done and carry-forward:
- Phase 2.4 stylized-fact validation (LOB-Bench or equivalent).
- Phase 3 agent code (AS baseline + fill simulator).
- The regime conditioning is confirmed partial — carries a calendar-time confound we're accepting for the methodology demonstration.

Key measured facts:
| Metric | Value |
|---|---|
| Raw input size | 35.6 GB compressed, ~350 GB decompressed |
| Partitioned Parquet | ~6.1 GB intermediate for 2 tickers |
| Canonical training tapes | 1.4 GB (60 days × 2 tickers) |
| Canonical val tapes | 255 MB |
| Canonical holdout tapes | 458 MB |
| INTC train windows (L=256, stride=64) | 2,058,259 |
| INTC val windows | 99,690 |
| Generator params | 1.6 M |
| Training loss (step 0 → 160K) | 1.016 → 0.010-0.020 |
| Best val_loss | 0.0160 |
| Full sampling wall-clock | 12.6 min (400 tasks × 1 GPU) |
| Synthetic Parquet files | 6400 total (154 MB) |

---

## 1. Cluster Setup — First Hours

### 1.1 Conda env

Created from [environment.yml](../environment.yml):
```bash
module load miniforge
conda env create -f environment.yml
conda activate diffmm
```

The env spec had a `pip: -e .` line in the `pip` section, but it didn't always pick up subpackages cleanly. Workaround: after `conda env create`, run `pip install -e .` manually.

### 1.2 rsync disaster (Bug #7 from Work1, re-hit)

On the first cluster sync, `diffmm.data` was missing:
```
ModuleNotFoundError: No module named 'diffmm.data'
```

Confirmed via `pip show -f diffmm` that only `diffmm/__init__.py` was installed — no subpackages. Root cause: the rsync exclude pattern `data/` matches **any** directory named `data` anywhere in the tree, including `src/diffmm/data/`. The anchored form `/data/` only matches top-level.

**Permanent fix** documented in Work1 §7 and in every subsequent sync command. All our rsync commands now use:

```
--exclude='/data/' --exclude='/checkpoints/' --exclude='/results/' \
--exclude='/logs/' --exclude='/wandb/'
```

---

## 2. Phase 1 — Data Pipeline on the Cluster

### 2.1 Data discovered

Seven gzipped TAQ CSVs in `~/diff_mm/data/raw/`. Five year-less files form the dataset; two `trade_2025_*` year-prefixed files cover the prior year and are unused.

| File | Compressed | Rows | Date span |
|---|---|---|---|
| `order_INTC.csv.gz` | 11 GB | 930,752,457 | 2025-04-02 → 2026-04-01 |
| `order_TSLA.csv.gz` | 14 GB | 1,271,738,194 | same |
| `trade_INTC.csv.gz` | 1.5 GB | 103,608,815 | same |
| `trade_TSLA.csv.gz` | 5.5 GB | 380,748,701 | same |
| `trade_SPY.csv.gz` | 3.5 GB | 243,464,773 | same — benchmark only, no quote file |

Per-day: INTC quotes ~3.7M/day, TSLA quotes ~5.1M/day, INTC trades ~415K/day, TSLA trades ~1.5M/day.

### 2.2 Stage A0 — gzip → per-date Parquet

**What happened:** First attempt to run [scripts/01_split_raw_to_parquet.py](../scripts/01_split_raw_to_parquet.py) hit three bugs in succession.

#### Bug — Hydra relative-path via `outputs/…` chdir

```
FileNotFoundError: data/raw/order_INTC.csv.gz
```

Hydra defaults to `chdir()` into a per-run directory (`outputs/YYYY-MM-DD_HH-MM-SS/`), so `data/raw/...` resolved relative to the Hydra output dir, not the project root. Fix: add to [configs/config.yaml](../configs/config.yaml):

```yaml
hydra:
  job:
    chdir: false
  run:
    dir: ${output_dir}
```

#### Bug — OOM in the supposedly-streaming splitter

Second attempt got OOM-killed at 32 GB. My original implementation used:

```python
lf = pl.scan_csv(gz_path, ...)
for batch in lf.collect().iter_slices(n_rows=batch_size):
    ...
```

`.collect()` materializes the entire 100 GB-decompressed CSV before `iter_slices` chunks it. Polars `scan_csv`'s "laziness" is about query optimization, not source-file streaming. There's no way to stream-read a gzipped CSV with Polars.

**Fix:** rewrote [src/diffmm/data/raw_split.py](../src/diffmm/data/raw_split.py) to use `pyarrow.csv.open_csv` with `block_size=128 MB`. PyArrow decompresses gzip block-by-block and yields `RecordBatch`es. We exploit the fact that TAQ files are pre-sorted chronologically to keep ONE `ParquetWriter` open at a time, closing it when DATE rolls over. Memory ceiling: ~256 MB regardless of input size.

#### Bug — fractional-share trades break Int64 inference

```
pyarrow.lib.ArrowInvalid: In CSV column #6: CSV conversion error to int64: invalid value '0.008457'
```

PyArrow auto-inferred `SIZE` as Int64 from the first block (clean integer trades in the early rows), then crashed on a later block containing fractional-share trades (e.g. `SIZE=0.008457`). Modern TAQ includes these because retail platforms support fractional-share trading.

**Fix:** explicit `column_types` in `_convert_options`, forcing size columns to Float64:

```python
_NUMERIC_COLS_TRADES = {
    "PRICE": pa.float64(), "SIZE": pa.float64(),
    "TR_SEQNUM": pa.int64(),
}
```

And in [src/diffmm/data/clean.py](../src/diffmm/data/clean.py), round before casting to Int64 and drop sub-1-share rows:

```python
pl.col("SIZE").round(0).cast(pl.Int64).alias("trade_sz"),
...
return out.filter(pl.col("trade_sz") >= 1)
```

**Stage A0 timing (final):** 5 jobs in parallel on `mit_normal` (4 CPU, 32 GB each). INTC quotes ~25 min, TSLA quotes ~40 min, trade files ~5-10 min each.

### 2.3 Stage A1 — per-date cleaning array job

**Manifest generation:**
```bash
python scripts/utils/make_manifest.py --kind clean \
    --out data/manifests/ph1_clean.json --tickers INTC TSLA
# wrote 504 entries (252 trading days × 2 tickers)
```

**QOS hit:** `sbatch --array=0-503%50` failed with `QOSMaxSubmitJobPerUserLimit`. For `mit_normal`, `MaxSubmitJobsPerUser = 448`. Solution: chunks of 200 tasks, submitted sequentially with a drain-wait between chunks (Work1 gave this pattern; we reused it).

**Per-task performance:** ~22s wall, 1.97M rows per TSLA day post-cleaning, 203 MB estimated DataFrame size. Much faster than my initial 30-60s estimate.

**Run stats:**
- Total tasks: ~504
- Concurrency cap: 50
- Total wall: ~30-40 min
- Total compute: ~3 hours

### 2.4 Stage A2 — finalize (splits + quantiles)

**Cheap step (~1 min per ticker).** This is where train/val/holdout sizes live as CLI-overridable Hydra params:

```yaml
splits:
  n_train_days: 60
  n_val_days: 10
  n_holdout_days: 20
  holdout_priority_dates: []   # to be filled with FOMC/CPI/earnings later
  seed: 42
```

**Two bugs surfaced during finalize.**

#### Bug — `np.quantile` propagates NaN

```
cat data/processed/quantiles_TSLA.json
# {"vol": [NaN, NaN], "vpin": [0.80, 0.84]}
```

At least one TSLA training day had `raw_vol` with non-finite values (possibly an Inf from `log(mid)` at a thin-quote moment or bf16-style numerical overflow). `np.quantile([..., inf, ...])` returns `NaN` for all thresholds.

**Fix** in [src/diffmm/data/regime.py](../src/diffmm/data/regime.py):
```python
vol_vals.append(v[np.isfinite(v)])
vpin_vals.append(vp[np.isfinite(vp)])
```

After re-finalize: `{"vol": [0.00134, 0.00215], "vpin": [0.80, 0.84]}` — finite, reasonable.

#### Bug — Polars categorical-join label drift

Cross-tab on INTC 2025-04-04 showed:
- c_vol=2 count: 2.25 M
- high_vol + toxic count: 0.84 M
- **~1.4 M rows of c_vol=2 were labeled "base" or "thin"**, which is impossible per the label table.

Root cause: the lookup-table join on `(c_vol, c_vpin)` with a Categorical `regime_label` column was silently mismatching on ~40% of rows. Polars' categorical join has known issues when the join-result column's categorical is filled via `fill_null(lit("base").cast(Categorical))`.

**Fix** — replaced the join with a direct when/then expression:
```python
df = df.with_columns(
    pl.when((pl.col("c_vol") == 2) & (pl.col("c_vpin") == 2)).then(pl.lit("toxic"))
    .when(pl.col("c_vol") == 2).then(pl.lit("high_vol"))
    .when(pl.col("c_vpin") == 2).then(pl.lit("thin"))
    .otherwise(pl.lit("base"))
    .cast(pl.Categorical)
    .alias("regime_label")
)
```

Plus a regression test in [tests/test_data_pipeline.py](../tests/test_data_pipeline.py) covering all 9 (c_vol, c_vpin) cells.

### 2.5 Phase 1 final state

```bash
$ ls data/processed/
SPLITS_INTC.json   SPLITS_TSLA.json   quantiles_INTC.json   quantiles_TSLA.json
intermediate/      norm_stats_INTC.json   train/   val/   holdout/

$ wc -l data/processed/{SPLITS_INTC,SPLITS_TSLA}.json
# 60 train, 10 val, 20 holdout per ticker, total 180 canonical files
```

**Quantile thresholds (training-set fit):**

| ticker | `raw_vol` 33% | `raw_vol` 67% | `raw_vpin` 33% | `raw_vpin` 67% |
|---|---|---|---|---|
| INTC | 0.00268 | 0.00418 | 0.802 | 0.853 |
| TSLA | 0.00134 | 0.00215 | 0.800 | 0.844 |

**Norm stats (INTC training, for generator input):**

| feature | mean | std |
|---|---|---|
| bid_px | $37.71 | $10.65 |
| ask_px | $37.72 | $10.65 |
| bid_sz | 2,057 | 3,456 |
| ask_sz | 2,128 | 4,365 |
| trade_px | $1.75 | $8.28 |
| trade_sz | 7.80 | 910 |
| trade_sign | -0.0002 | 0.215 |
| mid | $37.72 | $10.65 |
| spread | $0.0104 | $0.00207 |

(Note the bimodal trade_px/trade_sz stats — 90% of rows are quote_updates with null trade fields, fill_null(0) pulls their means toward 0.)

**First regime cross-tab (INTC train, 2025-04-04 = Liberation Day tariff shock):**

```
c_vol × c_vpin count by regime_label:
(0,0)→base: 29,796    (1,0)→base: 9,928     (2,1)→high_vol: 149,486
(0,1)→base: 156,539   (1,1)→base: 458,705   (2,2)→toxic: 2,098,693
(0,2)→thin: 53,538    (1,2)→thin: 641,434
```

This single day is heavily toxic-dominated because of macro stress. Over all 60 training days, the distribution averages back toward 33/33/33 by construction (33%/67% quantiles).

---

## 3. Phase 2 — Generator Training and Sampling

### 3.1 Architecture

TRADES-style from scratch (no TRADES checkpoint — incompatible with our TAQ NBBO schema). See [src/diffmm/generator/trades_adapter.py](../src/diffmm/generator/trades_adapter.py):

```
x_t: (B, 256, 9) → in_proj (256 → 256) → +pos_embed +t_embed → FiLM(regime)
                → LayerNorm → 8× self-attn blocks → FiLM(regime) → out_proj (256 → 9)
                → ε_pred: (B, 256, 9)
```

Parameters: **1.6 M** (d_model=256 after 2×embed_dim, depth=8, 8 heads).

### 3.2 First smoke runs — 3 cluster bugs + a silent multi-hour deadlock

**Bug — GPU detected but PyTorch training silently used CPU.** First "smoke" job ran for 40+ min without producing any log output, then the partition killed it at 6h. Diagnostic via `ssh node{N} && nvidia-smi` showed:
- GPU allocation: 1× L40S, 0% compute util
- Memory: 511 MiB (just PyTorch init, no model working on it)
- Python CPU: 0.9% idle
- 4 `pt_data_worker` processes (despite `num_workers=0` on CLI)

Two stacked bugs:

1. **`num_workers` plumbing**: [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) passed `num_workers=cfg.cluster.num_workers` to the DataLoader, ignoring `cfg.generator.training.num_workers` from the CLI override. Fix: read `generator.training.num_workers` with `cluster.num_workers` as fallback.

2. **`LOBWindowDataset` held 60 Polars DataFrames in memory, total ~16 GB**. With `num_workers=4`, each worker `fork()`s a COW shared-memory view of all 60 frames. Random-access `__getitem__` across tapes triggers page faults in all workers simultaneously → deadlock. Fix: rewrite [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py) to pre-materialize contiguous numpy arrays once at init; `__getitem__` becomes a pure slice (zero-copy).

**Performance payoff (before/after the refactor):**

| Stage | Before | After |
|---|---|---|
| Dataset construction (60 train tapes) | 3 sec (misleading — data not actually loaded) | **7.1 sec (eager numpy)** |
| First batch after init | ~60+ min (COW page faults) | **<1 sec** |
| Training step | Hangs with `num_workers=4` | **~30 ms/step on L40S** |

### 3.3 Training run — `intc_v1_short`, 5 epochs

Kicked off:
```bash
sbatch slurms/ph2_finetune.slurm data=intc_top \
    generator.training.max_epochs=5 \
    experiment_name=intc_v1_short
```

**Observed run:**
- Step 0: loss = 1.016 (= E[ε²] for standard-normal noise; expected starting value)
- Step 20: 0.922 (-9%)
- Step 40: 0.760 (-24% total)
- ... step-time on L40S steady at ~30 ms ...
- Step 160,800 (end of epoch 5): loss = 0.010-0.020 range
- **Best val_loss = 0.0160 at epoch 4**

Wall-clock: ~3 h (within 6 h cap on `mit_normal_gpu`). Checkpoint saved to `/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v1_short/ckpt-epoch004-valloss0.0160.ckpt`.

**Interpretation:**
- Loss plateaued in last 4K steps (0.012 ± 0.005).
- Val loss matches train loss band → **no overfitting**.
- **99% noise-variance explained** — necessary but not sufficient for good samples (as we learned next).

### 3.4 Sampling — the diagnostic rabbit hole

First sampling output was catastrophically wrong. Decoded Parquet showed:
- `bid_px` min=max=0.0 (all clamped by the no-negative guard in decode.py)
- `ask_px` up to $7634
- `mid` std $246 (expected ~$0.5 for INTC)
- Normalized-space sample z-scores up to **713**

In normalized space, healthy DDIM output should be mean~0, std~1. Something was producing values 2 orders of magnitude too large.

#### Diagnostic sweep

Wrote [scripts/utils/diagnose_sampling.py](../scripts/utils/diagnose_sampling.py) to sweep (n_steps, guidance_weight) and report both sampled-tensor stats and per-timestep ε-prediction stats. First result (before clamp):

```
n_steps  w     mean     std      min        max
20       0.0   13.264   230.078  -971.239   875.266
20       3.0   -132.530 498.458  -1694.372  1516.777
200      0.0   27.584   189.420  -968.201   841.616
200      3.0   -108.383 474.933  -1772.142  1686.219
```

All broken. Then extended the diagnostic to print ε-prediction statistics at fixed timesteps:

```
noise-prediction std by t:
t=999 → 0.99   ✓ (trivial: x_t ≈ pure noise, predict noise = input)
t=900 → 1.01
t=700 → 1.10
t=500 → 1.35
t=300 → 1.72   ← worst
t=100 → 1.62
t=20  → 1.27
```

**Classic "uniform-MSE over-trains extremes, under-trains middle timesteps" signature.** The 0.01 average training loss was misleading: t=999 is trivially easy (MSE ≈ 0), so its low loss drags the average down. The middle timesteps where the denoiser actually has work to do are poorly trained.

This explains the sampling blowup: at DDIM step 0, `t=999, α̅=0.001`, so `x0_pred = (x - √(1−α̅)·ε) / √α̅ ≈ x / 0.03`. When `ε` isn't accurately predicted, the tiny denominator amplifies errors catastrophically, and subsequent steps accumulate the damage.

#### Fix — `x0_clip` defensive clamp

Added to [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py):

```python
x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
if x0_clip is not None:
    x0_pred = x0_pred.clamp(-x0_clip, x0_clip)
```

Default `x0_clip=4.0` (4σ for ~N(0,1) normalized data). Well-trained models ignore it; under-trained models get saved from catastrophic blowup.

After the fix, normalized-space stats:

```
n_steps  w     mean    std
200      0.0   -0.09   1.09    ✓ (near-ideal N(0,1))
200      1.0   -0.28   1.14
200      3.0   -0.33   1.23
```

**Finding**: w=0 (unconditional) is closer to N(0,1) than w=3. CFG with this checkpoint overshoots significantly. We changed default `guidance_weight` from 3.0 → 1.0 in [configs/generator/ddim_sample.yaml](../configs/generator/ddim_sample.yaml) and default `n_steps` from 20 → 200.

### 3.5 Sampling verification — first 16-window test

After fix, sampling 16 windows for base regime produced:
- mid: $20.59-$21.09, mean $20.86, std $0.096 (INTC-plausible!)
- spread: min $0.01, max $0.70, mean $0.16 (mostly tight, with a wide-spread tail)
- bid_px: $20.45-$21.08, ask_px: $20.63-$21.25
- Trade/quote ratio: 140/116 ≈ 1.2 (inverted vs real INTC's ~1:10; model over-generates trades)
- Crossed books: 0
- Schema validates ✓

Good enough to proceed to full generation.

### 3.6 Full 400-task sampling

Plan: 4 regimes × 100 seeds × 16 windows = 6400 Parquets.

#### Bug — QOS limit 64 on `mit_normal_gpu`

`sbatch --array=0-399%20` rejected:
```
sbatch: error: QOSMaxSubmitJobPerUserLimit
```

For `mit_normal_gpu`, `MaxSubmitJobsPerUser = 64` (vs 448 for `mit_normal`). Array jobs count each array task toward the limit, so 400 > 64.

#### Solution — one-job loop

Rather than chunking into 7 subbatches, wrote [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) — a single Python process that iterates the manifest, loads the checkpoint once, and samples all 400 entries sequentially. 1 job, no QOS pressure.

#### Bug — backslash-in-f-string inside bash `--wrap`

First attempt had the Python code inline inside `sbatch --wrap "..."` with nested quotes → f-string had backslashes → Python 3.11 rejects backslashes inside f-string expressions. **Lesson learned (again): never put non-trivial Python in bash `--wrap`; always a standalone script file.**

#### Bug — `Path.mkdir(parents=True)` fails across symlinks

Third attempt got to the script, ran for 28 min, then died:
```
FileExistsError: [Errno 17] File exists: '/home/favara/orcd/scratch'
```

`~/orcd/scratch` is a symlink to `/orcd/scratch/orcd/007/favara/`. Python 3.11's `Path.mkdir(parents=True, exist_ok=True)` has a recursive-parent bug across symlinks — it calls itself on each parent, and when the symlink target exists but the path we're calling mkdir on is formally "different", it raises.

**Fix:** switch to `os.makedirs(path, exist_ok=True)` in the two places that matter ([scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) and [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py)). Also pass explicit `--out-root /orcd/scratch/...` to avoid any reliance on the symlinked home path.

#### Successful run

```
[sample-batch] running 400 tasks (0..400)
[sample-batch] loading checkpoint .../best.ckpt
[sample-batch] 20/400  regime=base  seed=19  elapsed=37.8s  rate=0.5/s  written=320
...
[sample-batch] 400/400  regime=thin  seed=99  elapsed=757.5s  rate=0.5/s  written=6400
[sample-batch] done. total files written: 6400
```

**12.6 min wall, 0 errors, 6400 Parquets written.** At 0.5 tasks/sec including checkpoint decode+write, ~2.5 sec of actual GPU time per batch.

---

## 4. Generated Data — Quantitative Analysis

### 4.1 Cross-regime stats (10 sample files per regime)

| regime | n_files | mid_mean | mid_std | spread_mean | trades/256 |
|---|---|---|---|---|---|
| `base` | 1600 | $24.83 | 0.127 | $0.131 | 134.5 |
| `high_vol` | 1600 | $33.16 | 0.115 | $0.088 | 157.9 |
| `toxic` | 1600 | $41.80 | 0.094 | $0.096 | 179.5 |
| `thin` | 1600 | $39.11 | 0.096 | $0.101 | 194.9 |

### 4.2 What this reveals — the calendar-time confound

The headline pattern is counter-intuitive: **mid-price level is the most obvious regime differentiator**. This points at a confound in our training-set labeling.

Root cause: INTC price drifted from ~$20 in April 2025 to ~$42 by March 2026 during the training window. Volatility and toxicity both clustered differently at different price levels:
- Early-year (low price, lower volatility) → predominantly `base` labels
- Late-year (high price, earnings volatility) → predominantly `high_vol` / `toxic` labels

Our regime-quantile approach fits thresholds on a global pool of daily stats, so **"high_vol" accidentally became a proxy for "time-of-year-when-price-was-high"**. The generator correctly learned this correlation and reproduces it.

Concretely, regime conditioning in the synthetic data shifts:
- **Mid level** (strongly) — confound
- **Trade count** (weakly but monotonically): 135 → 195 across regimes — legitimate learned microstructure
- **Volatility** (backwards): stressed regimes show *lower* within-window mid_std — likely because price-level shift dominates the denormalization
- **Spread** (roughly flat ~1-13 ¢) — regime conditioning doesn't strongly affect spread

### 4.3 Why this is not a project blocker

The proposal's central hypothesis is about *agent-ranking consistency between synthetic stress scenarios and held-out real days*. For that test, synthetic days need to (a) differ from each other and (b) differ from training-set days. Both are true. The physical interpretation of "regime" being weaker than hoped is a known limitation to document, not a derailment.

### 4.4 Proper fixes (deferred)

If time permits, any of these would improve regime fidelity:
1. **Price detrending at cleaning time** — work in log-returns or day-normalized prices rather than absolute.
2. **Within-day quantiles** — fit regime thresholds per-day on intraday stats, not across-day.
3. **Min-SNR loss weighting** — addresses the per-timestep training imbalance revealed by the diagnostic.
4. **Longer training** — 20 epochs with Min-SNR on the current architecture.

All of these require restarting generator training. Probably worth a `intc_v2` run if Phase 3 completes ahead of schedule.

---

## 5. Complete Bug Ledger (Phase 1 + Phase 2)

Continuing from Work1 §7 which covered bugs 1–11:

### Bug #12 — Fractional-share trades break PyArrow Int64 inference
- **Symptom:** Cleaning task crashed mid-way through trade file with `ArrowInvalid: invalid value '0.008457'`.
- **Root cause:** Modern TAQ trades include fractional-share retail fills; PyArrow auto-inferred Int64 from early blocks then failed on a fractional.
- **Fix:** Explicit `column_types={"SIZE": pa.float64(), "PRICE": pa.float64()}`; round + cast + filter `trade_sz>=1` in cleaner.

### Bug #13 — TSLA quantiles NaN
- **Symptom:** `quantiles_TSLA.json` had `"vol": [NaN, NaN]`; downstream `apply_quantiles` produced all-identical `c_vol=0` (because `pl.Expr < NaN` is True in Polars).
- **Root cause:** At least one TSLA training day had non-finite `raw_vol`; `np.quantile` propagates NaN.
- **Fix:** `np.isfinite` filter in `fit_quantiles`.

### Bug #14 — Polars categorical-join dropping ~40% of label rows
- **Symptom:** `regime_label` didn't match `(c_vol, c_vpin)` for 1.4M rows per INTC training file.
- **Root cause:** LUT-join on a Categorical column + `fill_null(lit("base").cast(Categorical))` silently created incorrect joins.
- **Fix:** Replaced LUT with direct `pl.when(...).then(...)` chain. Regression test added covering all 9 cells.

### Bug #15 — Num_workers CLI override silently ignored
- **Symptom:** Smoke ran 2h with 4 `pt_data_worker` processes despite `num_workers=0` on the CLI.
- **Root cause:** Script read `cfg.cluster.num_workers` (=4) instead of `cfg.generator.training.num_workers` (=0).
- **Fix:** Prefer the training-block value, fall back to cluster; added explicit log line `DataLoader num_workers=N` for future debugging.

### Bug #16 — `LOBWindowDataset` deadlocked on COW-fork data pages
- **Symptom:** 2h idle after model summary printed; GPU 0%, CPU 0%, no disk I/O.
- **Root cause:** Dataset held 60 Polars DataFrames (~16 GB) in memory; 4 workers `fork()`ed as COW views; random-access `__getitem__` caused page-fault cascades.
- **Fix:** Pre-materialize all feature + condition arrays as contiguous numpy at init; `__getitem__` is a zero-copy slice.

### Bug #17 — Lightning default logger silenced training-loss output
- **Symptom:** No training-loss values in SLURM `.out` file despite training completing.
- **Root cause:** `trainer(logger=False)` disables all logging; `enable_progress_bar=False` suppresses TQDM.
- **Fix:** Default to a `CSVLogger`, add per-step `print()` in `DDPMTrainer.training_step` (gated on step % 20 == 0).

### Bug #18 — Checkpoint filename with `=` chars broke Hydra CLI override
- **Symptom:** `generator.checkpoint_path=.../epoch=4-val_loss=0.0160.ckpt` failed parsing.
- **Root cause:** Hydra override grammar treats `=` as a delimiter. The ModelCheckpoint's default `"epoch={epoch}-val_loss={val/loss:.4f}"` template produces `=` chars.
- **Fix:** Change template to `"ckpt-epoch{epoch:03d}-valloss{val/loss:.4f}"`.

### Bug #19 — Seed override failed because `seed` exists in root config
- **Symptom:** `+seed=0` on the CLI failed with "Could not append to config. An item is already at 'seed'".
- **Root cause:** `+key=value` is Hydra's "create new" syntax; `seed: 42` already exists.
- **Fix:** Use bare `seed=0` (override semantics) or `++seed=0` (force create-or-override).

### Bug #20 — Sampled tensor had z-scores up to 713
- **Symptom:** Decoded synthetic Parquet had `bid_px=0` everywhere, `ask_px` up to $7634, `mid_std` $246.
- **Root cause (diagnostic-confirmed):** Under-trained denoiser at middle timesteps (uniform-MSE over-weighs extreme t, under-weighs middle t). DDIM `x0_pred` division by small `√α̅` magnifies prediction errors at high-noise timesteps.
- **Fix:** Defensive `x0_clip=4.0` clamp in `ddim_sample`. Also changed defaults to `n_steps=200, guidance_weight=1.0`.
- **Root cause not fixed:** the training imbalance — needs Min-SNR loss weighting (future `intc_v2`).

### Bug #21 — Backslash inside f-string in bash `--wrap`
- **Symptom:** `SyntaxError: f-string expression part cannot include a backslash`.
- **Root cause:** bash `--wrap "..."` with nested Python using f-string expressions containing `\"` escapes. Python 3.11 parser rejects.
- **Fix:** Always use a proper script file for multi-line Python; never inline in `--wrap`.

### Bug #22 — `Path.mkdir(parents=True, exist_ok=True)` across symlinks
- **Symptom:** Full sampling run crashed 28 min in with `FileExistsError: [Errno 17] File exists: '/home/favara/orcd/scratch'`.
- **Root cause:** Python 3.11 `pathlib.mkdir` has a recursive-parent bug when a path component is a symlink to a directory that exists.
- **Fix:** Use `os.makedirs(path, exist_ok=True)` in both `scripts/utils/run_sampling_manifest.py` and `src/diffmm/generator/decode.py`. Additionally pass an explicit `--out-root` to avoid symlink paths in the config default.

### Bug #23 — QOS limit 64 on `mit_normal_gpu` for array jobs
- **Symptom:** `sbatch --array=0-399` rejected with `QOSMaxSubmitJobPerUserLimit`.
- **Root cause:** `mit_normal_gpu` has `MaxSubmitJobsPerUser = 64`; array jobs count each task toward the limit.
- **Fix:** Two options; we chose option B (run all 400 sequentially in one Python process).

---

## 6. Architectural Changes Since Work1

Module-by-module. Significant ones:

### [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py)
Rewritten `LOBWindowDataset.__init__` to pre-materialize features and conditions as contiguous numpy arrays. `__getitem__` is now a pure slice. **Memory cost ~15 GB for 60 INTC tapes; speedup ~1000×.**

### [src/diffmm/data/regime.py](../src/diffmm/data/regime.py)
- `fit_quantiles`: added `np.isfinite` filter (Bug #13).
- `apply_quantiles`: replaced LUT join with direct `when/then` expression (Bug #14).

### [src/diffmm/data/clean.py](../src/diffmm/data/clean.py)
- Size columns rounded then cast Int64 (Bug #12).
- Sub-1-share trades filtered.

### [src/diffmm/data/raw_split.py](../src/diffmm/data/raw_split.py)
- Full rewrite from Polars-lazy to PyArrow `open_csv` (Bug #10).
- Explicit `column_types` for Float64 sizes (Bug #12).

### [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py)
- Added `x0_clip` parameter (default 4.0) to `ddim_sample` (Bug #20).

### [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py)
- `os.makedirs` instead of `Path.mkdir(parents=True)` (Bug #22).

### [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py)
- Default `CSVLogger` + explicit per-step print (Bug #17).
- Numeric `num_workers` plumbing from `generator.training.num_workers` (Bug #15).
- Trainer gained `limit_train_batches` / `limit_val_batches` / `max_steps` passthroughs for bounded smoke runs.
- Dataset-init progress logs (visibility during the 10-min cold startup on cluster).
- ModelCheckpoint filename changed to not use `=` chars (Bug #18).

### [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py)
- Passes `x0_clip` through from the ddim_sample config.

### [configs/config.yaml](../configs/config.yaml)
- Added `hydra.job.chdir: false` (Bug — hydra chdir broke relative paths).

### [configs/generator/ddim_sample.yaml](../configs/generator/ddim_sample.yaml)
- Inherits from `finetune` via Hydra `defaults` so model/schedule fields are present.
- Defaults updated: `n_steps=200`, `guidance_weight=1.0`, `x0_clip=4.0` (empirically derived).

### [configs/cluster/engaging.yaml](../configs/cluster/engaging.yaml)
- Corrected project dir name `diff-market-making` → `diff_mm`.

### New files

- [scripts/utils/diagnose_sampling.py](../scripts/utils/diagnose_sampling.py) — sweep (n_steps, guidance_weight) + per-timestep ε stats.
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py) — single-process 400-task loop (avoids QOS limit).

### Test additions

- `test_apply_quantiles_label_consistency` in [tests/test_data_pipeline.py](../tests/test_data_pipeline.py) — locks in the correct `(c_vol, c_vpin) → regime_label` mapping for all 9 cells.

**Current test count: 10 (6 generator smoke + 4 data pipeline). All pass locally in ~10 sec.**

---

## 7. Where the Artifacts Live

### On the cluster

```
~/diff_mm/                                       # project root (code + configs + logs)
~/diff_mm/data/raw/                              # the 7 TAQ gzips (~35 GB)
~/diff_mm/data/raw_partitioned/{ticker}/{kind}/  # Stage A0 output, ~6 GB
~/diff_mm/data/processed/intermediate/{ticker}/  # Stage A1 output, 6.1 GB
~/diff_mm/data/processed/{train,val,holdout}/    # Stage A2 canonical tapes, ~2 GB total
~/diff_mm/data/processed/{SPLITS,quantiles,norm_stats}_{ticker}.json
~/diff_mm/data/manifests/                        # ph1_clean.json, ph2_sample.json

/orcd/scratch/orcd/007/favara/diffmm/
    checkpoints/intc_v1_short/best.ckpt          # 5-epoch fine-tune, val 0.016
    synthetic/INTC/{base,high_vol,toxic,thin}/   # 6400 Parquet (154 MB)

~/diff_mm/logs/                                  # SLURM .out/.err per job
```

### Locally (synced via rsync)

```
docs/Work0.md, Work1.md, Work2.md
src/diffmm/ ...
scripts/ ...
slurms/ ...
tests/ ... (10 passing)
```

---

## 8. Lessons and Durable Conventions

Things we now know to always do (or never do):

1. **rsync excludes must be anchored with `/`** for root-only matches. Unanchored excludes are directory-name-match anywhere.
2. **Hydra `chdir: false`** for any project where configs use repo-relative paths.
3. **Never put Python inside bash `sbatch --wrap`** — nested quoting always bites. Put the Python in a standalone script.
4. **`os.makedirs(exist_ok=True)`** is safer than `Path.mkdir(parents=True, exist_ok=True)` across symlinks.
5. **`np.isfinite` filter** before computing quantiles on potentially-NaN/Inf data.
6. **`np.quantile` propagates NaN**; use `np.nanquantile` if non-finite values are possible.
7. **Polars `dt.hour()` returns Int8** — always `.cast(pl.Int32)` before arithmetic like `*60`.
8. **`rolling_sum_by(time_window)` requires a Datetime index column** — use `pl.from_epoch(ts_ns, "ns")` to produce one.
9. **Polars categorical joins have edge cases** — prefer `when/then` chains for small lookup tables (≤100 rows).
10. **Hydra `+key=value` creates new, `key=value` overrides existing, `++key=value` does either.** Pick based on whether the key is already in the config tree.
11. **Avoid `=` chars in file paths** — breaks Hydra CLI.
12. **DataLoader `num_workers>0` with a big in-memory dataset and `fork()`** causes COW page-fault cascades. Either keep data truly shared-read (numpy arrays > Polars frames) or use `num_workers=0`.
13. **`x0_clip` should be on by default** for diffusion sampling — catches under-trained checkpoints at inference time without silent explosions.
14. **Uniform-timestep diffusion training loss is misleading** — per-timestep diagnostics reveal the truth. Extreme-t is trivially low-loss; middle-t is where the work is.
15. **`mit_normal_gpu` QOS = 64**; array jobs count each task. Use a single-process loop for large batches instead.

---

## 9. What's Next — Phase 3 Plan

Per the action plan, Phase 3 is agent + fill-simulator code, then Phase 4 is the backtest matrix, then Phase 5 is the central hypothesis test.

### Phase 3 deliverables (code)
- `src/diffmm/agents/base.py` — `MarketMakingAgent` ABC, `LOBState`, `Quote` dataclasses.
- `src/diffmm/agents/as_baseline.py` — Agent A1 (Avellaneda–Stoikov baseline) per Algorithm 4 in the proposal.
- `src/diffmm/agents/as_ofi.py` — Agent A2 (A1 + OFI-skew).
- `src/diffmm/agents/as_vpin.py` — Agent A3 (A1 + VPIN-gated spread widening).
- `src/diffmm/agents/calibration.py` — per-stock per-day calibration of γ, κ, σ, β_OFI, α, τ_VPIN.
- `src/diffmm/sim/fill.py` — fill simulator (Algorithm 5), shared between real and synthetic tapes.
- `src/diffmm/sim/pnl.py`, `src/diffmm/sim/runner.py` — accounting + backtest loop.

### Phase 3 tests
- Agent quotes always valid (δ^b > 0, δ^a > 0, no crossed).
- A1 on a single real training day produces positive expected PnL.
- Fill simulator conservation: `Π = Σ cash + q_T · S_T`.
- Fill simulator produces identical output on identical (real or synthetic) input (provenance-blind).

### Phase 3 expected timeline
- Local dev + tests: 2-3 days (no cluster needed for agent code).
- Calibration on real tapes: 1 job per ticker (~10 min).
- Historical replay (Phase 4, Task 4.2): 1 small SLURM job per agent, CPU-only.

---

## 10. Appendix A — Key Numbers in One Place

| What | Value |
|---|---|
| Training data (INTC) | 60 days, 2,058,259 windows |
| Training data (TSLA) | 60 days, similar magnitude |
| Event counts per day (INTC) | ~3.7M quotes + ~415K trades |
| Event counts per day (TSLA) | ~5.1M quotes + ~1.5M trades |
| Quote filter keep rate | ~99.99% (keep `QU_COND == R`) |
| Trade filter keep rate | ~75-80% (after SCOND/CORR/RF filters) |
| Dark trade fraction (INTC) | 32% of trades (EX=D, FINRA TRF) |
| Window length | 256 events |
| Window stride | 64 events |
| Generator params | 1.6 M |
| Train wall-clock | ~3 h for 5 epochs on L40S |
| Training loss plateau | 0.010-0.020 |
| Best val_loss | 0.0160 |
| DDIM steps (final default) | 200 |
| Guidance weight (final default) | 1.0 |
| `x0_clip` default | 4.0 |
| Full synthetic run wall | 12.6 min (400 tasks) |
| Files written | 6400 Parquet |
| Synthetic dataset size | 154 MB |
| Regime trade-count range | 134 → 195 (monotonic, base→thin) |
| Regime mid-price range (confound) | $24.83 → $41.80 (base → toxic) |
| Regime spread range | $0.09 → $0.13 (weakly differentiated) |

---

## Appendix B — Complete File Inventory Diff Since Work1

### Added
- [scripts/utils/diagnose_sampling.py](../scripts/utils/diagnose_sampling.py)
- [scripts/utils/run_sampling_manifest.py](../scripts/utils/run_sampling_manifest.py)

### Modified
- [configs/config.yaml](../configs/config.yaml) — `hydra.job.chdir: false`
- [configs/cluster/engaging.yaml](../configs/cluster/engaging.yaml) — diff-market-making → diff_mm
- [configs/generator/ddim_sample.yaml](../configs/generator/ddim_sample.yaml) — defaults-inherit from finetune; new defaults n_steps=200, w=1, x0_clip=4
- [configs/generator/finetune.yaml](../configs/generator/finetune.yaml) — added `max_steps`, `limit_train_batches`, `limit_val_batches`
- [src/diffmm/data/clean.py](../src/diffmm/data/clean.py) — Int32 cast for hour arithmetic; size rounding; sub-1-share filter
- [src/diffmm/data/dataset.py](../src/diffmm/data/dataset.py) — eager numpy materialization
- [src/diffmm/data/features.py](../src/diffmm/data/features.py) — Int32 cast; Datetime index for rolling; scipy.special.erf
- [src/diffmm/data/raw_split.py](../src/diffmm/data/raw_split.py) — full PyArrow rewrite
- [src/diffmm/data/regime.py](../src/diffmm/data/regime.py) — isfinite filter; when/then instead of LUT join
- [src/diffmm/data/splits.py](../src/diffmm/data/splits.py) — unchanged structurally
- [src/diffmm/generator/decode.py](../src/diffmm/generator/decode.py) — os.makedirs
- [src/diffmm/generator/sample.py](../src/diffmm/generator/sample.py) — x0_clip parameter
- [src/diffmm/generator/train.py](../src/diffmm/generator/train.py) — per-step print; trainer-ref guard
- [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) — CSVLogger default; num_workers plumbing; trainer limits
- [scripts/13_sample_synthetic.py](../scripts/13_sample_synthetic.py) — x0_clip passthrough
- [tests/test_data_pipeline.py](../tests/test_data_pipeline.py) — 4th test covering regime-label consistency

Total tests: **10 passing** (6 generator smoke + 4 data pipeline).
