# Work10 Prompt — Handoff for the Report-Figure Agent

You are the **plotting agent** for the final 15.458 report on diffusion-based LOB synthesis. Another agent thread (the "cross-comparison agent") is running cluster jobs and producing CSVs in parallel; your job is to **own the report figures + analysis tables** that feed [docs/final_report.tex](final_report.tex).

This document is your complete handoff: project context, directory layout, current state, conventions, bug ledger, and what to focus on next.

---

## 1. Project in one paragraph

We trained 20 diffusion-model variants on INTC limit order book (LOB) data, evaluating them on three independent axes: **distributional realism** (G1 — 16 stylized-fact pass/fail checks), **temporal pattern transferability** (predictive score — train LSTM on synth, test on real, MAE relative to a market-replay baseline), and **multivariate coverage** (PCA convex-hull overlap with real). The headline finding: **no architecture optimizes all three axes simultaneously** — the three are systematically anti-correlated across our 20 checkpoints. As a secondary application, we tested whether agent-ranking metrics (`ρ_diff`, Spearman correlation between synth and real holdout agent rankings) track distributional quality; they don't — the metric is decoupled from all three quality axes for a specific structural reason (A1/A2 agents are economically indistinguishable). We're now extending to TSLA + SPY to test whether the three-axis tension generalizes across microstructure regimes.

The full project narrative + structural skeleton lives in [docs/final_report.tex](final_report.tex). Read its section comments first — they tell you exactly what each figure is supposed to show.

---

## 2. Directory layout you need to know

```
~/Desktop/diff-market-making/
├── docs/
│   ├── final_report.tex                ← report skeleton; section comments name every figure
│   ├── Work10_Report_Instructions.md   ← original brief (figure-by-figure spec, 25 figures + 4 tables)
│   ├── Work10_update.md                ← running status ledger (read this for current state)
│   ├── Work10_Prompt.md                ← this file (your handoff)
│   ├── Work6_phaseG.md                 ← latest phase doc (post-Phase F, current state of generators)
│   └── Work[0-9]*.md                   ← older phase docs; useful background
├── scripts/
│   ├── 80_report_figures.py            ← MAIN figure suite — 25 figures + 4 tables, one driver
│   ├── 81_real_stylized_facts.py       ← cluster-bound, real-INTC standalone panel for §2.5
│   ├── 82_training_curves.py           ← local, reads Lightning metrics.csv (needs logs pulled)
│   ├── 83_microstructure_diagnostics.py← cluster, spread + transitions + intraday volume
│   ├── 84_regime_embedding_tsne.py     ← cluster, t-SNE of ckpt regime embeddings
│   ├── 85_per_timestep_denoising.py    ← cluster GPU, per-feature denoising MSE per t
│   ├── 86_intraday_lob_volume.py       ← cluster, TRADES Fig 4 style (level-1 queue size)
│   ├── 87_return_distribution_full.py  ← cluster, log-y density + QQ plot
│   ├── 88_per_regime_g1.py             ← local, per-regime G1 numerical heatmap
│   ├── 50–63_*.py                      ← validation / diagnostic scripts (already run)
│   └── utils/
│       ├── run_all_parallel.sh         ← mass fan-out orchestrator (hits mit_preemptable hard)
│       ├── run_p0_post_overnight.sh    ← P0.B/C/D orchestrator (sbatch-wrapped)
│       ├── run_tsla_post_training.sh   ← TSLA sample → validate → cross-eval pipeline
│       ├── run_spy_post_training.sh    ← SPY equivalent
│       ├── run_spy_chained_training.sh ← SPY chained-resume training (4×6h chunks)
│       └── submit_spy_chain.sh         ← submit-and-exit version of the above
├── results/
│   ├── _archive/<model>/               ← per-model archive: validation/, hypothesis/, etc.
│   │   ├── multi_axis_comparison.csv   ← THE central comparison table (22 rows × 18 cols)
│   │   ├── G1_comparison_all_models.csv← 16 checks × 20 models, pass/value pairs
│   │   └── <model>/validation_<regime>/G1_pass_fail_summary.csv  ← per-regime G1 (3×4 = 12)
│   ├── trades_metrics_all/             ← script 62 outputs (PCA coverage, midprice traces, vol-vol)
│   │   ├── pca_projections.csv         ← per-tape PC1/PC2 (use for fig_4_1 4-panel scatter)
│   │   ├── pca_coverage.csv
│   │   └── midprice_traces_<model>.pdf ← script 62 also writes these
│   ├── predictive_score_all/predictive_score.csv
│   ├── deep_diagnostic/                ← per_feature_stats, temporal_acf, z_space_stats, etc.
│   ├── ranking_mechanism/              ← agent_decomposition, conditional_returns, a2_minus_a1
│   ├── trades_style/                   ← scripts 86 + 87 outputs
│   ├── microstructure_diagnostics/     ← script 83 outputs
│   └── report_figures/                 ← ALL final figures land here (PDF + PNG, 300 DPI)
├── data/processed/                     ← canonical schema parquets (NOT on this Mac — cluster only)
└── notebooks/
    └── 2026-MM-DD_*.md                 ← lab notes; latest dated entry has live decisions
```

**Critical: the `data/` symlink is cluster-only.** Locally we only have CSVs/JSONs in `results/`. Any script that needs to load parquets MUST run on the cluster. See §6 for how to ask the user to run those.

---

## 3. Current state at handoff (2026-05-11)

### 3.1 Done
- [final_report.tex](final_report.tex) — full skeleton, no body prose. Section comments name every figure. Title is "Diffusion Models for LOB Generation: Three-Axis Tension Between Distributional Realism, Temporal Transferability, and Multivariate Coverage".
- [scripts/80_report_figures.py](../scripts/80_report_figures.py) — **runs clean locally**, produces 25 figures + 4 tables. Last verified: all PDFs landed without errors.
- [scripts/88_per_regime_g1.py](../scripts/88_per_regime_g1.py) — produces per-regime G1 numerical heatmap (cross-sectional gradient) + pass-count bars.
- All P0 cluster batches drained (deep_diagnostic for v5/v9/v9_b, conditional_returns × 20 models, per-regime G1 for v2/v5/v9 × 4 regimes).
- TSLA v2 + v5 trained, sampled, and validated. Cross-eval (TRADES + predictive) is running or done.

### 3.2 Running on cluster
- **SPY chained training** (esampi account): 4×6h chunks per model, both v2 and v5, with `every_n_train_steps=10000` for safety-net checkpoints. Submitted via [submit_spy_chain.sh](../scripts/utils/submit_spy_chain.sh). Expected wall time: ~24h end-to-end.
- **Cross-comparison agent thread**: the user has another agent running the actual cross-ticker comparison. That thread will produce TSLA + SPY analogs of the figures.

### 3.3 Pending (what you'd land next)
| Priority | Item | What's needed | Where it feeds |
|---|---|---|---|
| **P1** | Fig 4.1 PCA panels upgrade — verify auto-upgrade triggered when `pca_projections.csv` is local | Verify `results/trades_metrics_all/pca_projections.csv` exists locally; re-run script 80 | §4.4 of report |
| **P1** | Investigate v9 missing rows in per-regime G1 (script 88) — Trade-sign ACF + RV signature + ACF power-law are blank for v9 | `head -20 results/_archive/v9/validation_base/G1_pass_fail_summary.csv` and compare against v2's format. Likely a free-text emit. Either patch the parser or document as known data gap. | §3.2.2 |
| **P1** | Cross-asset hero figure update (when SPY pipeline completes) | Add TSLA + SPY v2/v5 points to `fig_1_1_three_axis_scatter` and `fig_10_1_rank_bump` | §7 |
| **P2** | Training curves figure — needs Lightning logs pulled from cluster | `rsync` the metrics.csv files; run `scripts/82_training_curves.py` | §3.3, Appendix B |
| **P2** | Real-INTC standalone panel (script 81) — has run on cluster but verify pulled | `rsync results/report_figures/fig_a_real_stylized_facts.{pdf,png}` | §2.5 |
| **P3** | Color/scaling consistency audit across the 25+ figures | manually open each PDF; verify MODEL_COLORS dict applied uniformly | report polish |

### 3.4 Recently merged-in data (you should re-run script 80 to pick up)
- `results/deep_diagnostic_v5_v9/` (P0.B output) → has been merged into `results/deep_diagnostic/per_feature_stats.csv` etc.
- `results/ranking_mechanism_full/` (P0.C output) → merged into `results/ranking_mechanism/*.csv`
- The merge commands are in [Work10_update.md](Work10_update.md) §5.3.

---

## 4. Conventions for this codebase

### 4.1 Figure style
- **All figures: PDF + PNG at 300 DPI**, write to `results/report_figures/`.
- **Consistent `MODEL_COLORS` dict** across every figure — defined at top of [scripts/80_report_figures.py](../scripts/80_report_figures.py). Group palette: blue=scale-accurate, red=scale-exploded (noclip), green=copula, orange=special (cascade/remapped/v11), black=real.
- **Per-model marker shapes** within group so labels stay distinguishable in dense scatters.
- **Seaborn `whitegrid` context, `paper` style.** Font sizes: title=14, axis_labels=12, tick_labels=10, annotations=9.
- **No emojis** in figures or code unless specifically asked.
- **Heatmaps use cross-sectional gradient per row** (greenest = closest to real, reddest = farthest). Per-row min-max normalization so checks at 1e-6 scale and 10³ scale are both legible. The user explicitly chose this representation — don't replace with binary pass/fail.

### 4.2 Output naming
- `fig_<section>_<index>_<short_name>.{pdf,png}` for figures (e.g. `fig_2_1_g1_numerical_heatmap.pdf`)
- `tab_<letter>_<short_name>.{tex,csv}` for tables (e.g. `tab_A_decision_matrix.tex`)
- Never overwrite without re-running the producing script (so we can diff in git)

### 4.3 Loaders + parsers
- **Always coerce CSV value columns with `pd.to_numeric(errors="coerce")`** before arithmetic. Some validators emit free-text like `"real=0.21, synth=1.0"` instead of pure floats — this trips up arithmetic operators silently otherwise.
- **Case-insensitive column matching** for G1 CSVs (script 50 uses capitalized `Check`, `Pass`, `Value`; older outputs use lowercase). Use the `_find()` helper pattern in script 88.
- **Filter `df["check"] != "TOTAL_PASS"`** before plotting — that aggregate row breaks heatmaps.
- **Filter `df["check"] != "Conditional moments (|z|<2)"`** — emits string `"majority"`, no numeric value.

### 4.4 Skip-if-exists guards
Every cluster orchestrator + every figure script should have `[[ -f <output> ]] && skip` style guards. Re-running shouldn't redo work that's already done. The user re-runs orchestrators frequently when adding new data; idempotency is essential.

---

## 5. Bug ledger from the last session (don't re-trip these)

| # | Bug | Symptom | Fix |
|---|---|---|---|
| L1 | zsh strict globbing in rsync paths | `zsh: no matches found: <remote-path>` | Quote remote spec: `"user@host:path/{a,b}/dir_*"` or per-iteration loop |
| L2 | Cross-ticker `wait_for_jobs` returned early | Orchestrator skipped through all stages in 1s | `log "EXEC: ..."` writes to stdout which got captured by `$(submit ...)` along with the JID. Fix: log to stderr in helper functions. |
| L3 | `pathlib.Path.mkdir` failed on broken symlink | `FileExistsError on 'data'` | `data/` symlink target was missing. Was actually a stale NFS mount on login008 — switched to login001. |
| L4 | login008 has stale NFS mount of `/orcd/scratch` | `ls scratch_path` returns "no such file" but works on login001 | Always use `ssh user@orcd-login001.mit.edu` not the round-robin endpoint |
| L5 | Lightning checkpoints saved under `<root>/default/`, not `<root>/` | `best_ckpt` couldn't find them | Extended glob: `ls $d/{,default/,lightning_logs/version_*/checkpoints/}ckpt-epoch*.ckpt` |
| L6 | SPY trainings completed 0 epochs in 6h | Zero checkpoints saved (default save is `every_n_epochs=5`) | Patched [scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py) to optionally add a second `ModelCheckpoint(every_n_train_steps=N)` callback. Trigger via `+generator.checkpointing.every_n_train_steps=10000` Hydra override. |
| L7 | `${#CKPT[@]:-0}` is invalid bash syntax | `bash: bad substitution` | `${#CKPT[@]}` is already safe under `set -u`; drop the `:-0` form |
| L8 | G1 CSV `value` column loaded as string | `TypeError: 'sub' not supported for dtype 'str'` in heatmap arithmetic | `pd.to_numeric(..., errors="coerce")` in the loader |
| L9 | Script 50's `_sample_synth` couldn't filter to one regime | `FileNotFoundError: no synthetic tapes under .../base/{base,high_vol,toxic,thin}/` | Added `--regime` flag + flexible glob |
| L10 | Midprice traces empty for v9-v11 | Synth tapes lack `mid` column (canonical 8-feature schema only) | Fall back to reconstructing path from `cumprod(1+mid_return)` |
| L11 | z-space analysis skipped `mid_return` for v8/v9 | "Discrete features" filter excluded it | Added `force_continuous=("mid_return",)` parameter |
| L12 | PCA projections never persisted per-tape | Fig 4.1 stuck on bar-chart fallback | Added `pca_projections.csv` write at end of `compute_pca_coverage()` in script 62 |
| L13 | DataFrame.applymap removed in pandas 2.x | `AttributeError: no attribute 'applymap'` | Use `.map()` (DataFrame.map is the replacement) |
| L14 | iterrows returns (index, row) tuples not Series | `AttributeError: 'tuple' object has no attribute 'name'` | Iterate over `df.index` directly when you need indices |
| L15 | rsync path with literal `~` in remote spec | `mkdir /home/user/~/diff_mm failed` | Use `diff_mm/` (no tilde) — relative to remote home automatically |
| L16 | rsync `~diff_mm` ≠ `~/diff_mm` | tilde-USER expansion looks for nonexistent user "diff_mm" | Same as L15 — drop the tilde |

---

## 6. How to work with the user

### 6.1 You CANNOT run cluster commands.
The Bash tool here only executes locally. Cluster operations (sbatch, ssh, squeue, training jobs, parquet-loading scripts) MUST be handed to the user as **exact copy-paste commands**. Always include:
- The cluster account context (favara vs esampi — different scratch mounts!)
- Which login node they should use (always `orcd-login001` after the L4 incident)
- The exact rsync commands to push/pull
- A quick verification command they can run to confirm success

Example handoff format:
```bash
# 1. Push patched script
cd ~/Desktop/diff-market-making && rsync -avzR \
    scripts/X.py \
    favara@orcd-login.mit.edu:diff_mm/
# orcd-login001 doesn't work to rsync files!!
# 2. Run on cluster
ssh favara@orcd-login001.mit.edu
cd diff_mm
./scripts/utils/run_X.sh 2>&1 | tee logs/X_$(date +%Y%m%d_%H%M%S).log

# 3. Verify
ls -lh results/.../expected_output.csv
```

### 6.2 You CAN run locally.
- Lint scripts: `python -c "import ast; ast.parse(open('scripts/XX.py').read())"`
- Test figure scripts on local CSVs: `~/Desktop/diff-market-making/venv/bin/python scripts/80_report_figures.py`
- Read CSVs / inspect data structures
- Edit any file in the repo

### 6.3 The `mit_preemptable` partition is your friend
For cluster CPU jobs: `--partition=mit_preemptable` runs across 50+ nodes simultaneously with no priority queue. Used for all parallel orchestrators ([run_all_parallel.sh](../scripts/utils/run_all_parallel.sh)). Light preemption risk for jobs <2 min; safe for all our analysis tasks.

For GPU training: `mit_normal_gpu` (6h cap, normal priority queue).

### 6.4 Two accounts, two scratch paths
- **favara** — INTC + TSLA work. Scratch: `/orcd/scratch/orcd/007/favara/`
- **esampi** — SPY work (separate GPU quota). Scratch: `/orcd/scratch/orcd/014/esampi/`

When pulling results, you may need separate rsyncs from each account.

### 6.5 Communication norms the user expects
- **Terse responses.** No "I've successfully ..." preamble.
- **Don't re-explain what they just told you.** Just act.
- **Show the code change before justifying it.**
- **Parallelize tool calls** when independent — multiple files to read = single message with N Read calls.
- **No emojis** in files, code, or response text unless explicitly requested. (Exception: text indicators in CLI output like ✓/✗/★ in figures are fine — but the user removed the ★ marker from heatmaps so don't add it back.)

---

## 7. Known unknowns + things to investigate

1. **v9 free-text format in per-regime CSVs** — Trade-sign lag-1 ACF, RV signature, ACF power-law β are blank for v9 in [scripts/88_per_regime_g1.py](../scripts/88_per_regime_g1.py) output. Diagnose: `head -20 results/_archive/v9/validation_base/G1_pass_fail_summary.csv` then patch parsers if recoverable.

2. **TSLA cross-eval completion** — the cross-comparison agent thread is running this. Once `results/_archive/tsla_v{2,5}/validation/G1_pass_fail_summary.csv` exists locally and `results/predictive_score_tsla/predictive_score.csv` exists, you can build the cross-asset version of `fig_1_1` (3 tickers × 2 architectures = 6 points instead of 20).

3. **SPY pipeline ETA** — chained training launched ~02:00 UTC on 2026-05-11; finishes ~02:00 UTC on 2026-05-12 best case. SPY post-training runs after that.

4. **The `Conditional moments (|z|<2)` row** — currently dropped from heatmaps. If the validator output format is changed to emit a number, re-include it.

---

## 8. Final report sections you own (the figure side)

From [final_report.tex](final_report.tex):

- §2 Data — Fig 2.1 Spearman correlation heatmap, Fig 2.5 real-INTC stylized facts
- §3 Methods — Fig 3.1 forward/reverse diffusion diagram (TikZ), Algorithm 4.1 (DDPM training), Fig 4.2 ε vs v target magnitudes, Fig 4.3 copula pipeline, Fig 4.4 three-axis triangle
- §4 Results — Fig 4.1 hero (three-axis scatter), Fig 4.2 pairwise anti-correlation, Fig 4.3 G1 numerical heatmap, Fig 4.4 PCA panels, Fig 4.5 ablation impact, Fig 4.6 x0_clip effect
- §4.7 Copula case study — Fig 4.7.1 z-space, Fig 4.7.2 zero-inflation CDF, Fig 4.7.3 v2 vs v9 grouped bars
- §5 Agents — Fig 5.1 PnL overlap, Fig 5.2 ρ_diff vs everything, Fig 5.3 A2-A1 deltas
- §6 Cross-asset — Fig 6.1 cross-ticker ternary
- Appendices — full G1 matrix, all 20 checkpoint configs, etc.

Section comments in the .tex file are explicit about what each figure should communicate. Always read the comment block above a `% FIGURE X.Y:` line before adding/modifying a figure — it tells you what story the figure is in service of.

---

## 9. Your first action

1. Read [docs/Work10_update.md](Work10_update.md) — most recent status (will be more current than this prompt for live state).
2. Read the latest notebook entry in `notebooks/` — captures bugs solved + decisions made.
3. Open [final_report.tex](final_report.tex) and skim the section comments to internalize the report's structure.
4. Verify the figure suite still runs end-to-end:
   ```bash
   /Users/arturofavara/Desktop/diff-market-making/venv/bin/python scripts/80_report_figures.py
   ```
5. Open `results/report_figures/` and visually scan the PDFs — look for obvious render bugs (overlapping text, empty cells, wrong axes).
6. Then ask the user what specific figure or analysis they want next. Don't speculate — they have a working priority order.

Welcome to the project.
