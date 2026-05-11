# 2026-05-11 — Post-Training Pipeline & Report Build-Out

**Status:** TSLA pipeline ~done; SPY chained training in flight; figure suite refactored end-to-end; agent handoff written.

This entry captures the engineering decisions and bug ledger from a single dense session (2026-05-10 evening → 2026-05-11 morning) that transitioned the project from "all 20 generators trained on INTC" to "cross-asset pipeline running, report figures landing".

Cross-references:
- [docs/Work10_update.md](../docs/Work10_update.md) — running status ledger
- [docs/Work10_Prompt.md](../docs/Work10_Prompt.md) — handoff for the report-figure agent
- [docs/Work10_Report_Instructions.md](../docs/Work10_Report_Instructions.md) — original 25-figure brief

---

## 1. The arc of the session

1. **Cross-comparison build-out (INTC)** — pulled all latest cluster artifacts, merged deep_diagnostic + ranking_mechanism CSVs into the canonical inputs for `scripts/80_report_figures.py`, and ran the figure suite end-to-end. Result: 25 figures + 4 tables in `results/report_figures/`.

2. **TRADES-style additions** — added three missing-from-suite figures the user requested:
   - `scripts/86_intraday_lob_volume.py` — TRADES Fig 4 (level-1 queue size, real has intraday U-shape, synth uniform across event-position)
   - `scripts/87_return_distribution_full.py` — TRADES Fig 3 (log-y density + Q-Q plot)
   - upgraded `fig_4_1_pca_summary` in script 80 to consume `pca_projections.csv` for the proper 4-panel scatter+hull view (auto-detects, falls back to bar chart when missing)

3. **Massively-parallel cluster orchestrator** — wrote `scripts/utils/run_all_parallel.sh` that fans out 17 independent jobs to `mit_preemptable` simultaneously (per-regime G1 × 8, PCA re-run, real stylized facts, microstructure, regime t-SNE, intraday volume, return distribution, LSTM-size sweep × 3). All 17 ran concurrently across 12+ nodes; total wall time ~30-45 min vs ~5h serial.

4. **Per-regime G1 visualization** — `scripts/88_per_regime_g1.py` — produces a numerical heatmap with cross-sectional per-row gradient (vs the original binary pass/fail). Cell text = synth value, color = how far from real reference relative to other (model, regime) cells in the same row.

5. **TSLA post-training pipeline** — `scripts/utils/run_tsla_post_training.sh` orchestrates sample → validate → cross-eval after the cross-ticker training run completed.

6. **SPY pipeline disaster recovery** — esampi's SPY trainings completed 0 epochs in 6h (no checkpoints saved at all). Diagnosed: SPY data ~10× bigger than TSLA per day, so an SPY epoch takes ~24h and Lightning's default `every_n_epochs=5` checkpoint cadence never fired. Fix: patched `scripts/11_finetune_generator.py` to optionally add a second `ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, save_last=True)` callback; trigger via Hydra override. Then wrote `scripts/utils/submit_spy_chain.sh` that submits 4×6h chained chunks per model with `--dependency=afterany` so the chain auto-resumes from the latest available checkpoint. SPY now mid-flight.

7. **Login node infrastructure** — discovered login008 has stale NFS mount of `/orcd/scratch`; switched all subsequent ssh + rsync to `orcd-login001.mit.edu` directly (load-balancer endpoint also routes there reliably).

8. **Plotting agent handoff** — wrote [docs/Work10_Prompt.md](../docs/Work10_Prompt.md) so a parallel agent can take over the figure work while this thread continues with cross-comparison runs.

---

## 2. Bug ledger (chronological)

Numbered to match [docs/Work10_update.md §6](../docs/Work10_update.md) and [Work10_Prompt.md §5](../docs/Work10_Prompt.md).

### L1 — zsh strict globbing in rsync paths
**Symptom:** `zsh: no matches found: favara@orcd-login.mit.edu:diff_mm/results/_archive/v2/validation_*/`
**Cause:** zsh globs wildcards locally before invoking rsync; remote path with `*` looks like a local glob with no matches; zsh aborts with "no matches found" (vs bash's silent pass-through with `nullglob` off).
**Fix:** Quote the remote spec: `"user@host:path/{a,b}/dir_*"` — quotes prevent zsh from globbing. Or per-iteration `for` loop with quoted single args. Or `setopt -o nonomatch` (zsh-only) to match bash behavior.
**Also relevant:** brace expansion `{v2,v5,v9}` is also done locally by zsh.

### L2 — Cross-ticker `wait_for_jobs` returned early (cosmetic but caused downstream skip-through)
**Symptom:** Orchestrator's `[hh:mm:ss] EXEC: ...` log line got captured by `$(submit ...)` together with the JID, so the variable held both — `wait_for_jobs` then iterated tokens like `[hh:mm:ss]`, `EXEC:`, `sbatch`, etc. as if they were JIDs. squeue silently failed on each garbage token; running count stayed 0; loop exited immediately.
**Cause:** `log` helper writes to stdout, `submit()` also writes the JID to stdout via `echo "$jid"`, and `$(submit ...)` captures both.
**Fix:** Send all `log` output to stderr (`>&2`) inside helper functions that the caller captures via `$()`. Adopted the `submit_one()` pattern in `scripts/utils/run_all_parallel.sh`.

### L3 / L4 — Stale NFS mount on login008
**Symptom:** `ls /orcd/scratch/orcd/007/favara/diffmm/data` returned "No such file or directory" on login008 but the same path returned 234G of files on login001. Python `pathlib.Path('data').is_dir()` returned False (broken symlink), causing `mkdir(exist_ok=True)` to raise `FileExistsError`.
**Cause:** login008 has a stale or missing mount of the scratch filesystem. login001/005/007 are healthy.
**Fix:** Switch all ssh + rsync to `orcd-login001.mit.edu` directly. The round-robin endpoint `orcd-login.mit.edu` may also route to a working node. Always verify with `ls $HOME/orcd/scratch/` before launching anything.

### L5 — Lightning checkpoints under `<root>/default/` not `<root>/`
**Symptom:** `best_ckpt` couldn't find the trained TSLA checkpoints despite the .out logs explicitly naming the directory.
**Cause:** Lightning's CSVLogger creates a subdirectory under `cluster.checkpoint_root` named after the run (default = "default" when `experiment_name=default`).
**Fix:** Extended the `best_ckpt` glob to look in three locations:
```bash
{
    ls -1 "$d"/ckpt-epoch*-valloss*.ckpt 2>/dev/null
    ls -1 "$d"/default/ckpt-epoch*-valloss*.ckpt 2>/dev/null
    ls -1 "$d"/lightning_logs/version_*/checkpoints/ckpt-epoch*-valloss*.ckpt 2>/dev/null
} | awk -F'valloss' '{print $2, $0}' | sort -n | head -1 | awk '{print $2}'
```

### L6 — SPY trainings: 0 epochs in 6h, 0 checkpoints saved
**Symptom:** SPY v2 + v5 trainings ran 6h cleanly, hit TIMEOUT, and saved zero checkpoint files.
**Cause:** Default `cfg.generator.checkpointing.every_n_epochs=5` and Lightning's `ModelCheckpoint` only fires `on_validation_end` (per-epoch). SPY data is ~10× bigger than TSLA per day (~30M events/day vs ~3M); an SPY epoch ≈ 24h on one GPU. After 6h training never reached epoch 1, never triggered validation, never saved.
**Diagnostic signal:** `p_drop=1.00` throughout the .out log. The CFG dropout curriculum starts ramping at epoch 2. SPY at 343K steps still showed `p_drop=1.00` ≡ still in epoch 0. TSLA at the same wall-time was at `p_drop=0.20` (curriculum complete, past epoch 5).
**Fix:** Patched `scripts/11_finetune_generator.py` to add an optional second `ModelCheckpoint`:
```python
every_n_steps = cfg.generator.checkpointing.get("every_n_train_steps", 0)
if every_n_steps and int(every_n_steps) > 0:
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="ckpt-step{step:08d}",
            every_n_train_steps=int(every_n_steps),
            save_top_k=-1, save_last=True,
            auto_insert_metric_name=False,
        )
    )
```
Trigger via Hydra: `+generator.checkpointing.every_n_train_steps=10000`. Disabled by default — TSLA/INTC behavior unchanged.

### L7 — `${#CKPT[@]:-0}` invalid bash syntax
**Symptom:** `bash: ${#CKPT[@]:-0}: bad substitution`
**Cause:** The `:-0` default-value substitution form doesn't combine with the `#` length-of-array form.
**Fix:** Just use `${#CKPT[@]}` — `${#empty_assoc_array[@]}` returns 0 even under `set -u`. The `:-` was over-defensive.

### L8 — G1 CSV `value` column loaded as string
**Symptom:** `TypeError: 'sub' not supported for dtype 'str' with dtype 'float64'` in heatmap arithmetic.
**Cause:** Some G1 checks emit free-text values like `"real=0.21, synth=1.0"` instead of pure floats; pandas/arrow auto-typed the entire column as string.
**Fix:** `long_val["value"] = pd.to_numeric(long_val["value"], errors="coerce")` in the loader. Strings → NaN; floats stay numeric.

### L9 — Script 50's `_sample_synth` couldn't filter by regime
**Symptom:** P0.D's per-regime G1 evaluation passed the regime subdir as `--synth-dir`, but `_sample_synth` then globbed `<synth_dir>/{base,high_vol,toxic,thin}/*.parquet` and got `<synth_dir>/base/base/...` which doesn't exist.
**Fix:** Added `--regime` CLI flag that filters the inner glob to a single regime, with a fallback that treats `synth_dir` itself as the leaf if no `<synth_dir>/<regime>/` subdir exists.

### L10 — Midprice traces empty for v9-v11
**Symptom:** Midprice trace PDFs for copula models (v9, v9_b, v10_cascade, v11) showed only the real line — no synth lines at all.
**Cause:** The synth parquets for these models only contain the canonical 8-feature schema (`mid_return` etc.) and lack a reconstructed `mid` column. The plotting code only knew how to use `mid`.
**Fix:** Added a `_tape_returns` helper that prefers `mid` when present but falls back to `cumprod(1 + mid_return)` when it's absent. For genuinely-degenerate models (v11 with mid_return=0 everywhere) this produces a flat line at y=1.0, which is the correct visualization.

### L11 — z-space analysis skipped `mid_return` for v8/v9
**Symptom:** Fig 6.1 (z-space calibration) showed bid_sz/ask_sz stats but not mid_return — the most important feature for the copula story.
**Cause:** The deep diagnostic's z-space analysis filters out features in `copula_meta["discrete_features"]`. mid_return is in that list (point mass at 0 makes it formally "discrete").
**Fix:** Added `force_continuous=("mid_return",)` parameter to `z_space_analysis()`. mid_return now always gets z-space stats computed even when flagged discrete.

### L12 — PCA projections never persisted per-tape
**Symptom:** `fig_4_1_pca_panels` wanted per-tape projections to draw the 4-panel TRADES Fig 2 view, but only `pca_coverage.csv` (summary stats) existed.
**Fix:** Added a per-tape projection write at the end of `compute_pca_coverage()` in `scripts/62_trades_metrics.py`:
```python
proj_rows = [{"model": "real", "tape_idx": i, "pc1": float(rx), "pc2": float(ry)}
             for i, (rx, ry) in enumerate(real_pc)]
for label, vecs in synth_vecs_by_model.items():
    z = (vecs - mean) / std
    pc = pca.transform(z)
    proj_rows.extend({"model": label, "tape_idx": i, "pc1": float(px), "pc2": float(py)}
                     for i, (px, py) in enumerate(pc))
pd.DataFrame(proj_rows).to_csv(f"{out_dir}/pca_projections.csv", index=False)
```
Then `fig_4_1` auto-detects the file and switches to panel view.

### L13 — `DataFrame.applymap` removed in pandas 2.x
**Symptom:** `AttributeError: 'DataFrame' object has no attribute 'applymap'`.
**Fix:** Use `.map()` (DataFrame.map is the replacement in pandas 2.x).

### L14 — `iterrows()` returns tuples, not Series
**Symptom:** `for r in df.iterrows(): r.name` → `AttributeError: 'tuple' object has no attribute 'name'`.
**Fix:** `iterrows()` yields `(index, row_series)` tuples. Iterate `df.index` directly when you only need indices, or unpack the tuple.

### L15 / L16 — rsync tilde mishaps
**Symptom 1:** `rsync ... user@host:~/diff_mm/` → `mkdir /home/user/~/diff_mm failed`
**Symptom 2:** `rsync ... user@host:~diff_mm/` → silent partial transfer (only metadata, no files)
**Cause 1:** The `~/` doesn't get expanded by remote shell when it's part of an rsync arg in some quoting scenarios.
**Cause 2:** `~diff_mm` is bash tilde-USER expansion (looking for a user named "diff_mm").
**Fix:** Just use `diff_mm/` (no tilde, no leading slash) — rsync interprets paths without leading `/` as relative to the remote home directory.

### L17 — G1 CSV column case
**Symptom:** Loader's heuristic for `["pass", "passed", "result"]` returned None; all values came through as NaN.
**Cause:** Script 50's `G1_pass_fail_summary.csv` uses capitalized column names (`Check`, `Pass`, `Value`, `Section`, `Criterion`); my heuristic was lowercase-only.
**Fix:** `_find()` helper that lowercases column names for matching:
```python
def _find(df, candidates):
    lower_to_orig = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return None
```

### L18 — `Conditional moments (|z|<2)` row blank in heatmap
**Symptom:** This row showed empty across all (model, regime) cells.
**Cause:** The validator emits the literal string `"majority"` for this check — never a numeric value.
**Fix:** Drop the row in figure scripts: `df = df[~df["check"].isin({"Conditional moments (|z|<2)"})]`.

---

## 3. New scripts

| Script | Purpose | Lines |
|---|---|---|
| `scripts/80_report_figures.py` | Master figure suite (25 figures + 4 tables) | ~1500 |
| `scripts/81_real_stylized_facts.py` | §2.5 real-INTC 6-panel | ~210 |
| `scripts/82_training_curves.py` | Lightning metrics.csv → train/val loss curves | ~140 |
| `scripts/83_microstructure_diagnostics.py` | spread + transitions + intraday volume | ~280 |
| `scripts/84_regime_embedding_tsne.py` | Regime embedding 2D projection | ~150 |
| `scripts/85_per_timestep_denoising.py` | Per-feature MSE across noise levels | ~210 |
| `scripts/86_intraday_lob_volume.py` | TRADES Fig 4 — level-1 queue + envelope | ~250 |
| `scripts/87_return_distribution_full.py` | Full log-y density + QQ plot | ~190 |
| `scripts/88_per_regime_g1.py` | Per-regime G1 numerical heatmap + bars | ~310 |
| `scripts/utils/run_all_parallel.sh` | 17-job mass fan-out orchestrator | ~250 |
| `scripts/utils/run_p0_post_overnight.sh` (rewritten) | P0.B/C/D with sbatch wrappers | ~230 |
| `scripts/utils/run_tsla_post_training.sh` | TSLA full pipeline | ~340 |
| `scripts/utils/run_spy_post_training.sh` | SPY equivalent | ~340 |
| `scripts/utils/run_spy_chained_training.sh` | SPY chained-resume training + auto-handoff | ~250 |
| `scripts/utils/submit_spy_chain.sh` | Submit-and-exit version of above | ~70 |
| `scripts/utils/run_p2_gpu_sweeps.sh` | Guidance + per-timestep + LSTM sweeps | ~150 |

---

## 4. Architectural notes (for future agents)

### 4.1 The `mit_preemptable` partition pattern

Discovery from earlier in the project (P0 recovery): `mit_preemptable` accepts hundreds of concurrent jobs across the cluster's full node pool with no priority queue. For sub-2-min jobs, preemption risk is essentially zero (preemption is only triggered by higher-priority jobs needing the resource, which has to happen mid-run; 2 min is too short).

Adopted as the default partition for ALL CPU analysis jobs. `mit_normal` and `mit_normal_gpu` are reserved for things that need GPU (training) or that can't tolerate preemption (we don't have any).

The pattern:
```bash
sbatch --partition=mit_preemptable \
    --cpus-per-task=4 --mem=64G --time=01:00:00 \
    --wrap="..."
```

50+ such jobs land on 50+ different nodes within seconds.

### 4.2 Cross-sectional gradient heatmaps

Established style for G1-like comparisons across many models:
1. **Per-cell value: actual numeric synth value** (compactly formatted: `2.2e-06`, `3.92e+03`, `0.143`)
2. **Per-row color: min-max normalized error vs real reference**
3. **Per-row sort: by mean error rank** (best-overall checks at top)
4. **REAL column on right:** separated by heavy black bar; shows reference value
5. **Color map:** green-to-red gradient (`#1a7d3a → #d73027` via beige midpoint)
6. **No binary pass/fail markers** — the user explicitly asked to drop them as visual noise

Applied to `fig_2_1_g1_heatmap` (20 models × 16 checks) and `fig_p0d_g1_per_regime` (12 (model, regime) cells × 15 checks).

### 4.3 Two-account workflow

- **favara** — INTC + TSLA training/sampling/eval. Scratch: `/orcd/scratch/orcd/007/favara/`
- **esampi** — SPY training (separate GPU quota, lets us run TSLA + SPY trainings simultaneously). Scratch: `/orcd/scratch/orcd/014/esampi/`

For the rsync convention, push code to BOTH accounts when relevant, pull from BOTH when consolidating results.

### 4.4 Chained-resume training pattern

For models where 1 epoch > 1 job's time budget (like SPY):
1. Submit N sequential SLURM jobs, each `--time=06:00:00`, each `--dependency=afterany:<prev>`.
2. Job 1 starts fresh.
3. Jobs 2..N use a `--wrap` shell script that auto-discovers the latest checkpoint inside the SLURM job (so discovery happens at runtime, not submit time):
   ```bash
   LATEST=$(ls -t $CKPT_DIR/last.ckpt $CKPT_DIR/ckpt-step*.ckpt $CKPT_DIR/ckpt-epoch*.ckpt 2>/dev/null | head -1)
   [[ -z "$LATEST" ]] && { echo "FATAL: no ckpt to resume" >&2; exit 1; }
   python train.py ... +resume_from=$LATEST
   ```
4. Lightning's `+resume_from=<path>` restores optimizer state + scheduler state + epoch counter + global_step (verified in script 11's resume logic).
5. Combined with `every_n_train_steps=10000`, even a job that hits TIMEOUT mid-epoch has saved progress.

---

## 5. What's done vs what's running vs what's pending

See [docs/Work10_update.md §1-3](../docs/Work10_update.md) for the live ledger. This doc captures the architectural state at handoff; the update doc captures the live state.

---

## 6. Open questions for the next session

1. **v9's missing G1 rows in per-regime output** — Trade-sign lag-1 ACF, RV signature shape, ACF power-law β are blank for v9 across all 4 regimes. Likely a v9-specific free-text emit format. Diagnose: `head -20 results/_archive/v9/validation_base/G1_pass_fail_summary.csv` vs v2's. Recover with a parser regex addition or document as known data gap.

2. **Cross-asset hero figure** — once SPY pipeline completes (~24h after launch), update `fig_1_1_three_axis_scatter` and `fig_10_1_rank_bump` to include 6 cross-ticker points (3 tickers × 2 architectures). Tests whether the three-axis tension generalizes.

3. **Training curves** — pull Lightning metrics.csv files for v2/v5/v9 from cluster, then `scripts/82_training_curves.py` produces the convergence figure for §3.3 + Appendix B.

4. **Report prose** — skeleton is in [docs/final_report.tex](../docs/final_report.tex) with section comments naming every figure. Recommended fill order: §5 (results) → §5.7 (copula) → §6 (agents) → §7 (cross-asset) → §2 (data) → §3 (methods) → §1 (intro) → executive summary.

5. **(Maintenance window)** — MGHPCC infrastructure work May 11-16 may slow cluster scheduling. Already affecting some queue times. Use `mit_preemptable` to dodge priority pressure for short jobs.
