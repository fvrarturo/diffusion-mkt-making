# Work 0 — Project Architecture

**Project:** Stress-Testing High-Frequency Market Makers with Diffusion-Generated Counterfactual LOBs
**Course:** 15.458, Spring 2026
**Document date:** 2026-04-18
**Authors:** Arturo Favara, Nick Bernardini, Maxime Vallot

This document specifies the architecture for the project. It is the contract between the four phases (data, generator, agents, evaluation) and the cluster execution model. Every later "Work N" document should refer back to this one — module boundaries, file formats, and naming conventions defined here are load-bearing.

---

## 1. Design Principles (the five non-negotiables)

These are the rules everything else follows from. If a later decision violates one, that decision is wrong.

1. **One canonical event-tape schema.** Real and synthetic tapes must have identical column types, units, and ordering. The fill simulator, agents, and metrics never need to know whether a tape is real or generated. This is what makes the central hypothesis test honest.

2. **Holdout firewall.** Holdout files live in a directory that is `chmod 444` after split creation, and a `holdout_guard` module raises if any code outside [scripts/40_open_holdout.py](scripts/40_open_holdout.py) tries to read them. The holdout is opened exactly once, in Phase 5.

3. **Reproducibility envelope.** Every artifact (cleaned tape, model checkpoint, backtest CSV, figure) is tagged with `(git_sha, config_hash, seed)` in its filename or sidecar JSON. No exceptions. This is how we honor the proposal's honesty principle.

4. **Compute / interpretation separation.** Heavy jobs produce CSVs and Parquet; figures and conclusions are computed from those CSVs in lightweight scripts (or notebooks for exploration). You never re-run a 6-hour generator job to tweak a y-axis label.

5. **Local mirrors cluster.** The same code runs locally (smoke tests, plotting, paper) and on Engaging (training, sampling, full backtests). The only differences are config files and SLURM wrappers. This kills the "works on my Mac" failure mode.

---

## 2. Repository Layout

```
diff-market-making/
├── README.md
├── pyproject.toml             # package metadata + pinned deps
├── environment.yml            # conda env spec for cluster
├── Makefile                   # local convenience targets
├── .gitignore                 # data/, checkpoints/, wandb/, results/raw/
│
├── _info/                     # plans, papers, cluster docs (already exists)
├── docs/
│   ├── Work0.md               # ← this file
│   └── Work{N}.md             # subsequent design / decision docs
│
├── configs/                   # Hydra YAML configs (composable)
│   ├── config.yaml            # root: defaults list
│   ├── data/
│   │   ├── intc_top.yaml
│   │   ├── intc_depth.yaml
│   │   └── tsla_top.yaml
│   ├── generator/
│   │   ├── trades_pretrained.yaml
│   │   ├── finetune.yaml
│   │   └── ddim_sample.yaml
│   ├── agent/
│   │   ├── a1_as.yaml
│   │   ├── a2_ofi.yaml
│   │   └── a3_vpin.yaml
│   ├── experiment/
│   │   ├── ph2_validation.yaml
│   │   ├── ph4_evaluation.yaml
│   │   └── ph5_hypothesis.yaml
│   └── cluster/
│       ├── local.yaml
│       └── engaging.yaml      # paths, partitions, account
│
├── src/diffmm/                # main package (installable: `pip install -e .`)
│   ├── __init__.py
│   ├── io/
│   │   ├── schema.py          # canonical event-tape Pandera schema
│   │   ├── parquet.py         # read/write helpers (one place)
│   │   ├── manifest.py        # MD5 manifests, artifact sidecars
│   │   └── holdout_guard.py   # firewall (raises on illegal access)
│   ├── data/
│   │   ├── acquisition/
│   │   │   ├── wrds_taq.py
│   │   │   └── lobster.py
│   │   ├── clean.py           # filtering, Lee–Ready, BVC
│   │   ├── reconstruct_lob.py # LOBSTER LOB reconstruction (Algorithm 1)
│   │   ├── tapes.py           # build canonical event tapes
│   │   ├── features.py        # σ, OFI, VPIN, imbalance
│   │   ├── regime.py          # quantile discretization → labels
│   │   ├── splits.py          # train / val / holdout
│   │   └── dataset.py         # PyTorch Dataset + DataLoader
│   ├── generator/
│   │   ├── trades_adapter.py  # canonical schema ↔ TRADES tensor format
│   │   ├── conditioning.py    # FiLM head, regime → embedding
│   │   ├── model.py           # wraps DeepMarket model
│   │   ├── train.py           # fine-tuning loop (PyTorch Lightning)
│   │   ├── sample.py          # DDIM + classifier-free guidance
│   │   └── decode.py          # synthetic tensor → canonical Parquet
│   ├── agents/
│   │   ├── base.py            # MarketMakingAgent ABC, LOBState, Quote
│   │   ├── as_baseline.py     # A1
│   │   ├── as_ofi.py          # A2
│   │   ├── as_vpin.py         # A3
│   │   └── calibration.py     # γ, κ, σ, β_OFI, α, τ_VPIN
│   ├── sim/
│   │   ├── fill.py            # fill simulator (single source of truth)
│   │   ├── pnl.py             # accounting
│   │   └── runner.py          # backtest loop (real or synthetic input)
│   ├── eval/
│   │   ├── metrics.py         # Sharpe, CVaR, inventory variance
│   │   ├── stylized.py        # Wasserstein-1, ACFs, conditional moments
│   │   ├── lob_bench.py       # 12-score validation suite (vendor or impl)
│   │   ├── ranking.py         # π_hist, π_diff, π_truth
│   │   └── bootstrap.py       # rank-correlation hypothesis test
│   ├── viz/
│   │   ├── style.py           # matplotlib style
│   │   ├── validation.py      # Fig 4 (a–d)
│   │   ├── pnl_plots.py       # Fig 5, 6
│   │   └── hypothesis.py      # Fig 7
│   └── utils/
│       ├── seeding.py
│       ├── logging.py
│       ├── git.py             # current SHA, dirty check
│       └── wandb_helpers.py
│
├── scripts/                   # thin CLI wrappers around src/diffmm
│   ├── 01_acquire_data.py
│   ├── 02_build_tapes.py
│   ├── 03_compute_features.py
│   ├── 04_make_splits.py
│   ├── 10_load_trades.py
│   ├── 11_finetune_generator.py
│   ├── 12_validate_generator.py
│   ├── 13_sample_synthetic.py
│   ├── 20_calibrate_agents.py
│   ├── 21_replay_real.py
│   ├── 22_replay_synthetic.py
│   ├── 30_compute_metrics.py
│   ├── 40_open_holdout.py     # ★ the one-time holdout opener
│   ├── 41_hypothesis_test.py
│   ├── 50_make_figures.py
│   └── utils/
│       ├── make_manifest.py
│       └── verify_reproducibility.py
│
├── slurms/                    # SLURM job scripts (cluster only)
│   ├── _common.sh             # shared env activation
│   ├── ph1_clean.slurm        # array over (ticker, date)
│   ├── ph1_features.slurm
│   ├── ph2_finetune.slurm     # single GPU
│   ├── ph2_sample.slurm       # array over (regime × seed)
│   ├── ph2_validate.slurm
│   ├── ph3_calibrate.slurm
│   ├── ph4_replay_real.slurm  # array over (agent × day)
│   ├── ph4_replay_synth.slurm # array over (agent × regime × seed)
│   └── ph5_holdout.slurm
│
├── data/                      # gitignored
│   ├── raw/                   # WRDS / LOBSTER downloads + MANIFEST.md5
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   ├── holdout/           # chmod 444 after creation
│   │   └── SPLITS.json
│   └── synthetic/
│       └── {ticker}/{regime}/day_{seed}.parquet
│
├── checkpoints/               # gitignored, symlinked to scratch on cluster
│   ├── trades_pretrained/
│   └── trades_finetuned_{sha}/
│
├── results/                   # CSVs committed; raw arrays gitignored
│   ├── validation/
│   ├── backtests/
│   │   ├── real/
│   │   └── synthetic/
│   ├── metrics/
│   └── hypothesis/
│
├── reports/                   # final LaTeX report
│   ├── final_report.tex
│   └── figures/
│
├── notebooks/                 # exploratory only, numbered
├── third_party/               # vendored or submoduled (DeepMarket, lob_bench)
└── tests/                     # pytest: schema invariance, firewall, etc.
```

---

## 3. The Canonical Event-Tape Schema (the central contract)

Every Parquet file in `data/processed/` and `data/synthetic/` conforms to this schema. The schema is defined once in [src/diffmm/io/schema.py](src/diffmm/io/schema.py) using [Pandera](https://pandera.readthedocs.io/) and validated on read and write.

| Column | Type | Units | Notes |
|---|---|---|---|
| `event_idx` | int64 | — | Monotonically increasing within a (ticker, date) |
| `ts_ns` | int64 | nanoseconds since epoch (UTC) | Hardware-clock timestamp |
| `ticker` | category | — | One value per file |
| `event_type` | category | — | `quote_update`, `trade`, `cancel` (LOBSTER only) |
| `bid_px` | float64 | dollars | Best bid after the event |
| `ask_px` | float64 | dollars | Best ask after the event |
| `bid_sz` | int64 | shares | Top-of-book bid size |
| `ask_sz` | int64 | shares | Top-of-book ask size |
| `trade_px` | float64 (nullable) | dollars | Set only when `event_type == trade` |
| `trade_sz` | int64 (nullable) | shares | " |
| `trade_sign` | int8 (nullable) | {−1, +1} | Lee–Ready or BVC |
| `mid` | float64 | dollars | Derived: (bid_px + ask_px) / 2 |
| `spread` | float64 | dollars | Derived: ask_px − bid_px |
| `c_vol` | int8 | {0, 1, 2} | Regime: σ bucket |
| `c_vpin` | int8 | {0, 1, 2} | Regime: VPIN bucket |
| `c_imb` | int8 | {0, 1, 2} | Regime: top-of-book imbalance bucket |
| `c_tod` | int8 | {0, 1, 2} | Regime: morning/midday/afternoon |
| `regime_label` | category | — | Derived: `base` / `high_vol` / `toxic` / `thin` |

**Optional depth extension** (LOBSTER mode): columns `bid_px_{1..K}`, `bid_sz_{1..K}`, `ask_px_{1..K}`, `ask_sz_{1..K}` for `K ∈ {5, 10}`. The schema declares them as optional; agents and metrics that need depth assert their presence.

**Why this matters.** If synthetic Parquet looks identical to real Parquet at the byte level (modulo content), the fill simulator and agents are blind to provenance. That blindness is what makes the central rank-correlation test informative.

---

## 4. Module Responsibilities

This section maps each `src/diffmm/` module to specific tasks in the action plan.

### 4.1 `io/` — Persistence and firewall

- **`schema.py`** — Pandera schema, validation hooks. Action plan: §1.2, §1.3.
- **`parquet.py`** — `read_tape(path) → pl.DataFrame` and `write_tape(df, path)`. Calls schema validation on both ends. Single place to change format if needed.
- **`manifest.py`** — Computes MD5 of raw downloads (§1.1) and writes sidecar JSONs `{file}.meta.json` containing `{git_sha, config_hash, seed, schema_version, created_at}` for every produced artifact.
- **`holdout_guard.py`** — A context manager `with holdout_unlocked(reason="phase 5"):` that decorates the one allowed access path. Anywhere else, attempting `read_tape("data/processed/holdout/...")` raises `HoldoutLeakageError`.

### 4.2 `data/` — Pipeline (Phase 1)

- **`acquisition/`** — Pull from WRDS or LOBSTER, write to `data/raw/` with manifest. Action plan: §1.1.
- **`clean.py`** — Continuous-session filter (09:45–15:45 ET), crossed-quote removal, condition-flag filtering, Lee–Ready and BVC trade signing (both implemented; toggle in config). §1.2.
- **`reconstruct_lob.py`** — LOBSTER message-tape → top-of-book state machine (and optional depth). §1.2.
- **`tapes.py`** — Orchestrator: takes raw → cleaned → canonical-schema Parquet, one file per (ticker, date). §1.2.
- **`features.py`** — Realized vol, multi-level OFI, VPIN, top-of-book imbalance. §1.3.
- **`regime.py`** — Compute quantile thresholds **on the training set only**, freeze in `configs/data/{ticker}_quantiles.json`, apply elsewhere. Map (c_vol, c_vpin, c_imb) → human-readable label. §1.3.
- **`splits.py`** — Partition by date according to `SPLITS.json`; chmod the holdout. §1.4.
- **`dataset.py`** — `LOBWindowDataset` yielding `(window: (L, d), condition: (4,))` tensors. Normalization stats computed on train, frozen in JSON, applied on val/holdout. §1.4.

### 4.3 `generator/` — Diffusion model (Phase 2)

- **`trades_adapter.py`** — Bidirectional converter between our canonical Parquet schema and TRADES' expected tensor layout. **This is the single highest-risk module** (action plan §2.1 warning box). Unit tests assert round-trip identity.
- **`conditioning.py`** — FiLM head: `(c_vol, c_vpin, c_imb, c_tod)` → embedding → (γ, β) per layer. §2.2.
- **`model.py`** — Thin wrapper: `Generator.load(checkpoint_path) → loads pretrained TRADES + our conditioning head`. Exposes `forward(x, t, c)` and `predict_noise(...)`.
- **`train.py`** — PyTorch Lightning module. Fine-tunes conditioning head (frozen backbone or low-LR). CFG dropout (`p=0.1`). Logs to W&B every step; runs LOB-Bench eval every 5 epochs. §2.2.
- **`sample.py`** — DDIM with classifier-free guidance (Algorithm 3 from proposal). Takes `(c, w, S, seed) → tensor sequence`. §2.3.
- **`decode.py`** — Tensor sequence → canonical Parquet event tape. Applies inverse-normalization, enforces no crossed books, assigns synthetic timestamps and `event_idx`. Critical: the output must pass `schema.validate()` before being written.

### 4.4 `agents/` — Market makers (Phase 3)

- **`base.py`** — `MarketMakingAgent` ABC with `on_event(LOBState) → Quote` and `reset(AgentParams)`. `LOBState` and `Quote` are frozen dataclasses. Action plan §3.1.
- **`as_baseline.py`** — A1, Algorithm 4 from proposal. §3.2.
- **`as_ofi.py`** — A2: A1 + OFI skew. §3.3.
- **`as_vpin.py`** — A3: A1 + VPIN-gated spread widening. §3.4.
- **`calibration.py`** — Per-stock per-day calibration of (γ, κ, σ) on training data; grid-search of (β_OFI, α, τ_VPIN) on validation. Outputs `configs/agent/calibrated/{ticker}_{date}.json`. §3.2–§3.4.

### 4.5 `sim/` — Backtest engine

- **`fill.py`** — The fill simulator (Algorithm 5). **This is the second-highest-risk module** because it is the shared denominator for real and synthetic comparisons. Implements queue-position estimation, top-of-book fill logic, stale-quote removal. Pluggable fill model (`conservative` / `optimistic` / `pessimistic`) via config. §3.5.
- **`pnl.py`** — Accounting: cash + inventory mark-to-market with terminal liquidation at 15:45.
- **`runner.py`** — `run_backtest(agent, tape_path, fill_config, agent_config) → BacktestResult` with intraday PnL, inventory series, fill log, terminal PnL. Identical entry point for real and synthetic. §4.2, §4.3.

### 4.6 `eval/` — Metrics and hypothesis test

- **`metrics.py`** — Per-day metrics (Sharpe, CVaR_5%, inventory variance, fill rate, adverse-selection cost) from `BacktestResult`s. §4.4.
- **`stylized.py`** — Wasserstein-1 distances on marginals, ACFs of |r|, OFI, trade signs; conditional moment computations E[r|I], Var(r|c). §2.4.
- **`lob_bench.py`** — The 12-score LOB-Bench suite. Either vendored from the paper or pip-installed if available.
- **`ranking.py`** — Builds π_hist, π_diff, π_truth as ordered tuples of agent names ranked by Sharpe (or worst-quartile PnL for π_diff). §4.4, §5.2.
- **`bootstrap.py`** — Spearman ρ + 1000-bootstrap one-sided test on whether ρ(π_diff, π_truth) > ρ(π_hist, π_truth). §5.2.

### 4.7 `viz/` — Plots

Each function takes a CSV path or DataFrame, returns a Matplotlib figure, and writes both PDF (for the report) and PNG (for previews) under `reports/figures/`. No analysis logic in `viz/` — only rendering.

---

## 5. Configuration: Hydra Compositions

We use Hydra so that an experiment is uniquely identified by a single composed config tree. Every script accepts `--config-name=...` and `+overrides=value` style arguments.

A typical experiment invocation:

```bash
python scripts/22_replay_synthetic.py \
    experiment=ph4_evaluation \
    data=intc_top \
    agent=a3_vpin \
    cluster=engaging \
    +seed=42
```

The composed config is hashed and saved alongside outputs. `configs/experiment/ph4_evaluation.yaml` selects defaults across all groups; specific overrides happen on the CLI (and are recorded).

**Why Hydra and not argparse:** the composability across `data × agent × generator × experiment` axes is exactly the matrix structure we need for Phase 4 (12 cells × 100 seeds).

---

## 6. Reproducibility Envelope

Every artifact (Parquet, checkpoint, CSV, figure) is accompanied by a sidecar `*.meta.json`:

```json
{
  "git_sha": "8a3b1c2...",
  "git_dirty": false,
  "config_hash": "sha256:...",
  "schema_version": "v0.1",
  "seed": 42,
  "created_at": "2026-04-18T20:42:11Z",
  "creator_script": "scripts/22_replay_synthetic.py",
  "input_artifacts": ["data/synthetic/INTC/toxic/day_42.parquet"]
}
```

A single utility, [src/diffmm/io/manifest.py](src/diffmm/io/manifest.py), writes these. A reproducibility check script ([scripts/utils/verify_reproducibility.py](scripts/utils/verify_reproducibility.py)) re-runs a chosen seed and asserts byte-identity for deterministic outputs (cleaning, calibration) and statistical-equivalence for stochastic outputs (sampling).

The `git_dirty` field is critical: if true, **no run should be considered final**. The training and sampling scripts refuse to start with a dirty tree unless `--allow-dirty` is passed (and they record it).

---

## 7. Local ↔ Cluster Workflow

The same code runs in both environments. Differences are entirely in `configs/cluster/`.

### Local (Mac)

- Development of all modules.
- Smoke tests: 1-day cleaning, 10-window training, 1 synthetic hour, 1-day backtest.
- All plotting and report writing.
- `pip install -e .` on Mac venv.

### Cluster (Engaging, eofe7.mit.edu / orcd-login*)

- Heavy data cleaning (60+ days × 2 tickers).
- Generator fine-tuning (single GPU, partition `mit_normal_gpu` or `sched_mit_kburdge_r8`).
- Synthetic sampling (400+ days, GPU array job).
- Full backtest matrix (CPU array job, partition `mit_normal`).
- conda env `diffmm` recreated from `environment.yml`.

### Sync protocol (mirrors the DLO project)

```bash
# Mac → cluster (code only, no data/checkpoints)
rsync -avz --exclude='.*' --exclude='_info/' --exclude='venv/' \
    --exclude='__pycache__/' --exclude='data/' --exclude='checkpoints/' \
    --exclude='results/' --exclude='wandb/' \
    /Users/arturofavara/Desktop/diff-market-making/ \
    favara@orcd-login.mit.edu:~/diff-market-making/

# Cluster → Mac (results and figures only)
rsync -avz \
    favara@eofe7.mit.edu:~/diff-market-making/results/ \
    favara@eofe7.mit.edu:~/diff-market-making/reports/figures/ \
    /Users/arturofavara/Desktop/diff-market-making/
```

Data stays on the cluster after acquisition (`data/raw/` lives only in `~/diff-market-making/data/raw/` on Engaging, which is in `/orcd/pool` or `/orcd/scratch`). Checkpoints likewise live in scratch and are symlinked into the project tree.

---

## 8. SLURM Strategy

The action plan's compute is dominated by three patterns. Each gets a SLURM template under `slurms/`.

### 8.1 Embarrassingly parallel data jobs (CPU array)

`slurms/ph1_clean.slurm`: one array task per (ticker, date). 60–80 tasks, ~10 min each, partition `mit_normal`.

```
#SBATCH --array=0-119%30
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
```

A pre-generated manifest `data/manifests/ph1_clean.json` maps `SLURM_ARRAY_TASK_ID → (ticker, date)`. Same pattern as the example NUTS job.

### 8.2 GPU training (single node)

`slurms/ph2_finetune.slurm`: one task, one or more GPUs, partition `sched_mit_kburdge_r8` (preferred for 14-day limit) or `mit_normal_gpu` (6-hour cap; multiple jobs with checkpoint-resume).

```
#SBATCH --partition=sched_mit_kburdge_r8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

W&B logs streaming live; checkpoints saved to `~/orcd/scratch/diffmm/checkpoints/` every 5 epochs.

### 8.3 GPU sampling (array)

`slurms/ph2_sample.slurm`: array over (regime × seed). 4 regimes × 100 seeds = 400 tasks; each generates one synthetic day in ~5 min on a GPU.

```
#SBATCH --array=0-399%20
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
```

### 8.4 CPU backtest (array)

`slurms/ph4_replay_synth.slurm`: array over (agent × regime × seed). 3 × 4 × 100 = 1200 tasks, each ≪1 min. Concurrency capped at 200.

### 8.5 Common scaffolding

`slurms/_common.sh` is sourced by every job and handles:
- `cd "$SLURM_SUBMIT_DIR"`
- conda activation (`source activate diffmm`)
- `set -euo pipefail`
- `export PYTHONUNBUFFERED=1`
- log header (host, job ID, git SHA, python version)

---

## 9. Phase → Architecture Mapping

A quick lookup from the action plan to where the work actually lives.

| Phase | Action plan task | Module(s) | Script | SLURM |
|---|---|---|---|---|
| 0 | 0.1–0.3 setup | (repo + env + W&B) | — | — |
| 1 | 1.1 acquire | `data/acquisition/` | 01 | (interactive) |
| 1 | 1.2 clean & tapes | `data/clean.py`, `data/tapes.py`, `data/reconstruct_lob.py` | 02 | ph1_clean |
| 1 | 1.3 regime features | `data/features.py`, `data/regime.py` | 03 | ph1_features |
| 1 | 1.4 splits + loader | `data/splits.py`, `data/dataset.py` | 04 | (local) |
| 2 | 2.1 load TRADES | `generator/trades_adapter.py`, `generator/model.py` | 10 | (interactive smoke) |
| 2 | 2.2 fine-tune | `generator/conditioning.py`, `generator/train.py` | 11 | ph2_finetune |
| 2 | 2.3 DDIM | `generator/sample.py` | 13 (smoke) | ph2_sample |
| 2 | 2.4 validate | `eval/stylized.py`, `eval/lob_bench.py`, `viz/validation.py` | 12 | ph2_validate |
| 3 | 3.1–3.4 agents | `agents/` | 20 (calibrate) | ph3_calibrate |
| 3 | 3.5 fill sim | `sim/fill.py`, `sim/runner.py` | (used by 21, 22) | — |
| 4 | 4.1 sample synth | `generator/sample.py`, `generator/decode.py` | 13 | ph2_sample |
| 4 | 4.2 hist replay | `sim/runner.py` | 21 | ph4_replay_real |
| 4 | 4.3 stress replay | `sim/runner.py` | 22 | ph4_replay_synth |
| 4 | 4.4 metrics | `eval/metrics.py`, `eval/ranking.py` | 30 | (local) |
| 5 | 5.1 holdout | `io/holdout_guard.py`, `sim/runner.py` | 40 | ph5_holdout |
| 5 | 5.2 hypothesis | `eval/bootstrap.py` | 41 | (local) |
| 6 | 6.1 figures | `viz/` | 50 | (local) |
| 6 | 6.2 report | (LaTeX in `reports/`) | — | — |
| 6 | 6.3 cleanup | `scripts/utils/` | — | — |

---

## 10. The Two Highest-Risk Modules (where to spend extra care)

Out of all modules, two are load-bearing for project credibility and have the most ways to silently fail.

### 10.1 `generator/trades_adapter.py`

The TRADES checkpoint expects a specific tensor layout. If our adapter silently mis-orders columns or mis-scales features, the model will produce plausible-looking but invalid outputs. Mitigations:

1. **Round-trip property test**: `decode(encode(real_window)) == real_window` to within float32 epsilon.
2. **Distribution test on real data passed through**: feed a real window through the adapter and back; confirm the resulting Parquet still passes schema and stylized-fact checks.
3. **Explicit, documented mapping table** in the module docstring listing every TRADES tensor index and its canonical-schema column.

### 10.2 `sim/fill.py`

Any asymmetry in the fill simulator between real and synthetic execution invalidates the central comparison. Mitigations:

1. **One implementation, one config**: the fill simulator never branches on `tape.is_synthetic`. The only difference between real and synthetic backtests is the input Parquet path.
2. **Conservation tests**: PnL = sum(cash deltas from fills) + terminal_inventory × terminal_mid. Asserted in unit tests on every backtest result.
3. **Bracket reporting**: report results under all three fill assumptions (conservative, optimistic, pessimistic) so that the stress comparison is robust to fill-model misspecification.

---

## 11. Testing Strategy

We don't need full coverage. We do need three classes of tests:

1. **Schema invariance** — every produced Parquet validates against `schema.py`. Wired into `parquet.write_tape()` so failures are caught at write time.
2. **Holdout firewall** — a test that attempts to load a holdout file from any non-allowed script and asserts that `HoldoutLeakageError` is raised.
3. **Determinism** — given a fixed seed and config, `clean → tape → calibrate → backtest` produces byte-identical output across two runs on the same machine.

Smoke integration tests (1 day of data, 1 epoch of training, 1 synthetic day, 1 backtest) run on every PR locally; they are not run on the cluster.

---

## 12. What This Document Does Not Specify (intentional gaps)

These are deferred to later "Work N" docs once we know more:

- **Concrete depth representation** (top-of-K vs full LOB Image). Decided after Phase 1.1 confirms data availability.
- **TRADES checkpoint vs scratch decision** (action plan §2.1 warning). Decided after `10_load_trades.py` smoke test.
- **Fine-tuning hyperparameter sweep grid** (action plan §2.2 table). Decided after the first 5-epoch run reveals loss behavior.
- **N (synthetic days per regime)**: the plan says 100; if compute is tight we drop to 50 (action plan §4.1 warning). Decided after the first end-to-end sampling run produces a wall-clock estimate.

Each of those deferred decisions gets its own short Work doc when made, so we have a paper trail of architectural changes.

---

## Appendix A — One-line summary per source document

- **proposal.tex** — The methodological claim: synthetic stress regimes can produce a strategy ranking that predicts held-out real ranking better than historical replay.
- **action_plan.tex** — Seven-phase decomposition with success/kill criteria, dependencies, and a 13-week schedule.
- **reading_plan.tex** — 20 papers in five blocks; Tier 1 (≈23 hours) is required reading before the corresponding phase begins.
- **cluster/DLO_CLUSTER_SETUP.md** and **ORCD_COMMANDS.md** — The proven workflow from the prior project: archive on Mac, scp to Engaging, recreate venv, sbatch, rsync results back. Adopted here verbatim.
