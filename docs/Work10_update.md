# Work 10 — Report Build Status & Cluster Action Plan

**Last updated:** 2026-05-11 (post-overnight: P0 done, parallel batch done, TSLA pipeline ~done, SPY chained training in flight, figure suite refactored)
**Reference:** [Work10_Report_Instructions.md](Work10_Report_Instructions.md) (the original brief)
**Outputs:** [final_report.tex](final_report.tex), 9 scripts (`scripts/8?_*.py`), 6 cluster orchestrators (`scripts/utils/run_*.sh`), [results/report_figures/](../results/report_figures), [Work10_Prompt.md](Work10_Prompt.md) (plotting-agent handoff).

This is the running ledger of (a) what's produced, (b) what's running, (c) what still needs to be sent. Update in place as items move.

---

## 1. What's done locally (as of 2026-05-11)

### 1.1 Document
[final_report.tex](final_report.tex) — full skeleton. Title: "Diffusion Models for Limit Order Book Generation: The Three-Axis Tension Between Distributional Realism, Temporal Transferability, and Multivariate Coverage". §6 Copula Story merged into §5 as §5.7. §2.3 Zero-Inflation Pathology removed (now appears in §5.7.1 where it's load-bearing). §5 has a preamble paragraph anchoring Table 5.1 + Fig 5.1.

### 1.2 Figure suite (9 scripts, all lint-clean)

| Script | Purpose | Run-where | Status |
|---|---|---|---|
| [80_report_figures.py](../scripts/80_report_figures.py) | 25 figures + 4 tables, one-shot driver | local | ✓ runs clean, **55 outputs in `report_figures/`** |
| [81_real_stylized_facts.py](../scripts/81_real_stylized_facts.py) | §2.5 standalone real-INTC 6-panel | cluster | ✓ written + ran on cluster |
| [82_training_curves.py](../scripts/82_training_curves.py) | Lightning metrics.csv plotter for v2/v5/v9 | local (after Lightning logs pulled) | ✓ written, not yet run |
| [83_microstructure_diagnostics.py](../scripts/83_microstructure_diagnostics.py) | spread-conditional + event transitions + intraday volume | cluster | ✓ ran on cluster |
| [84_regime_embedding_tsne.py](../scripts/84_regime_embedding_tsne.py) | regime-embedding 2D projection from .ckpt files | cluster | ✓ ran on cluster |
| [85_per_timestep_denoising.py](../scripts/85_per_timestep_denoising.py) | per-feature MSE across noise levels for v2/v5/v9 | cluster GPU | ✓ written, not yet run |
| [86_intraday_lob_volume.py](../scripts/86_intraday_lob_volume.py) | TRADES Fig 4 — bid/ask mean + 5-95 envelope | cluster | ✓ ran on cluster |
| [87_return_distribution_full.py](../scripts/87_return_distribution_full.py) | TRADES Fig 3 — log-y density + QQ | cluster | ✓ ran on cluster |
| [88_per_regime_g1.py](../scripts/88_per_regime_g1.py) | per-regime G1 numerical heatmap + pass-count bars | local | ✓ ran locally |

### 1.3 In-place script patches (don't re-trip these)
- **[scripts/62_trades_metrics.py](../scripts/62_trades_metrics.py)**:
  - added per-tape PCA-projection write (`pca_projections.csv`) — enables proper Fig 4.1 4-panel scatter+hull
  - midprice_trace_plot now produces 2-panel layout (real+synth overlay + synth-only zoom), with real clipped to synth-window length and falling back to `cumprod(1 + mid_return)` when `mid` column absent (recovers v9-v11 traces)
  - `target_events=10000` per synth line via concatenated tape returns
- **[scripts/60_deep_diagnostic.py](../scripts/60_deep_diagnostic.py)**: `z_space_analysis()` takes `force_continuous=("mid_return",)` so v8/v9 z-space stats include `mid_return` even though it's flagged discrete
- **[scripts/63_predictive_score.py](../scripts/63_predictive_score.py)**: added `--hidden-dim` flag for LSTM-size robustness sweep (P2.G4)
- **[scripts/50_validate_generator.py](../scripts/50_validate_generator.py)**: added `--regime` filter so per-regime G1 evaluation can pass the parent dir + filter (was the P0.D bug from first cluster run)
- **[scripts/11_finetune_generator.py](../scripts/11_finetune_generator.py)**: optional second `ModelCheckpoint(every_n_train_steps=N, save_top_k=-1, save_last=True)` callback. Enabled via `+generator.checkpointing.every_n_train_steps=10000` Hydra override. Disabled by default (TSLA/INTC behavior unchanged). Critical for chained 6h jobs on big-data tickers like SPY where a full epoch doesn't fit in one job's time budget.
- **[scripts/80_report_figures.py](../scripts/80_report_figures.py)**:
  - `load_g1_long()` now: case-insensitive column matching, `pd.to_numeric(errors="coerce")` on value column, drops `TOTAL_PASS` rows, attaches `real_value` from `G1_REAL_VALUES` map
  - `fig_2_1_g1_heatmap` rewrote as cross-sectional gradient over per-row error vs real (greenest = closest to real among 20). Annotation = compact synth value. REAL column on right. No ★ marker (kept clean).
  - `fig_4_1_pca_summary` auto-upgrades to 4-panel scatter+hull when `pca_projections.csv` exists locally, falls back to bar chart with explicit "[bar fallback]" note
- **[scripts/88_per_regime_g1.py](../scripts/88_per_regime_g1.py)**: same numerical-heatmap design as fig_2_1, with per-row gradient + REAL column. Drops `Conditional moments (|z|<2)` (validator emits string `"majority"` not a number).

### 1.4 Cluster orchestrators
| Script | Purpose | Status |
|---|---|---|
| [run_cross_ticker_v2_v5.sh](../scripts/utils/run_cross_ticker_v2_v5.sh) | full pipeline TSLA + SPY | ran for TSLA; SPY had a checkpoint-save bug that's been fixed |
| [run_p0_post_overnight.sh](../scripts/utils/run_p0_post_overnight.sh) | P0.B (deep_diag v5/v9/v9_b) + P0.C (cond_returns × 20) + P0.D (per-regime G1) | ✓ all done; sbatch-wrapped (login-OOM fix) + `submit_and_wait` helper |
| [run_all_parallel.sh](../scripts/utils/run_all_parallel.sh) | 17-job mass fan-out to mit_preemptable | ✓ all done (per-regime G1 + PCA projections + microstructure + tsne + intraday + return-dist + LSTM sweep) |
| [run_tsla_post_training.sh](../scripts/utils/run_tsla_post_training.sh) | TSLA sample → validate → cross-eval | ✓ launched; sampling done (6208 v2 parquets), v5 needed manual resubmit |
| [run_spy_post_training.sh](../scripts/utils/run_spy_post_training.sh) | SPY equivalent | ready, blocked on training completion |
| [run_spy_chained_training.sh](../scripts/utils/run_spy_chained_training.sh) | full SPY chained-resume training + auto-handoff | written; superseded by submit_spy_chain.sh |
| [submit_spy_chain.sh](../scripts/utils/submit_spy_chain.sh) | submit-and-exit chained 4×6h SPY training | ✓ launched |
| [run_p2_gpu_sweeps.sh](../scripts/utils/run_p2_gpu_sweeps.sh) | guidance sweep + per-timestep + regime t-SNE + LSTM-size sweep | written, optional |

---

## 2. What's currently running on the cluster

### 2.1 favara — TSLA pipeline (post-training)
- v2 sampling complete: 6208 parquets, mid_return std=1.06e-05, frac_zero=66.4%
- v5 sampling: launched manually after first attempt didn't submit (cause: silent stderr suppression). Should drain ~15 min on preemptable.
- Validation + TRADES + predictive: re-run [run_tsla_post_training.sh](../scripts/utils/run_tsla_post_training.sh) once v5 sampling drains. The cross-comparison agent thread is handling this.

### 2.2 esampi — SPY chained training (started ~02:00 UTC 2026-05-11)
- 4 chunks × 2 models × 6h on `mit_normal_gpu`
- Submitted via [submit_spy_chain.sh](../scripts/utils/submit_spy_chain.sh)
- Each chunk resumes from `latest = ls -t .../{last.ckpt,ckpt-step*,ckpt-epoch*} | head -1`
- Step-based saves enabled via `+generator.checkpointing.every_n_train_steps=10000`
- Verified via grep: patched script `scripts/11_finetune_generator.py` is on esampi, and chunk-1 logs show config dump with `every_n_train_steps: 10000`
- ETA: ~24h end-to-end if v2 + v5 schedule simultaneously

### 2.3 Already pulled to Mac (2026-05-10 → 2026-05-11)
- ✓ `results/deep_diagnostic_v5_v9/` (P0.B output, 7 CSVs)
- ✓ `results/ranking_mechanism_full/` (P0.C, 4 CSVs)
- ✓ `results/_archive/{v2,v5,v9}/validation_{base,high_vol,toxic,thin}/` (P0.D, 12 dirs)
- ✓ `results/trades_metrics_all/pca_projections.csv` (P1.A — script 62 patched output)
- ✓ All 6 figures from the P1/P2 cluster batch (intraday_volume, return_distribution, regime_tsne, real_stylized_facts, microstructure trio)

### 2.4 Pending pull
- TSLA validation outputs once v5 sampling completes
- TSLA cross-eval (TRADES + predictive)
- SPY checkpoints (~24h from now)
- SPY post-training pipeline outputs (~25h from now)

---

## 3. Cross-asset (TSLA + SPY) — separate workstream

| Ticker | Run | User | Status |
|---|---|---|---|
| TSLA v2 (eps+film) | finished training, sampling done (6208 parquets) | favara | ✓ trained, sampled, awaiting validation |
| TSLA v5 (v+adaln_zero) | finished training, manual resample in flight | favara | ✓ trained, sampling in flight |
| SPY v2 (eps+film) | chunk 1/4 of chained training | esampi | RUNNING (~6h chunks, ~24h total) |
| SPY v5 (v+adaln_zero) | chunk 1/4 of chained training | esampi | RUNNING |

These feed §6 of the report (cross-asset validation). Once both ticker pipelines complete, a single 3-ticker version of `fig_1_1` hero scatter + `fig_10_1` rank-bump will show whether the three-axis tension generalizes.

---

## 4. Cluster action plan — what's left

### P0 — blocking (DONE)
- ✓ P0.A — TSLA training (favara) + SPY training (esampi)
- ✓ P0.B — deep_diag v5/v9/v9_b
- ✓ P0.C — conditional_returns × 20 models
- ✓ P0.D — per-regime G1 v2/v5/v9 × 4 regimes (12 outputs)

### P1 — high value (DONE except for one re-render)
- ✓ P1.A — pca_projections.csv pulled; Fig 4.1 auto-upgrades on next local re-run
- ✓ P1.B — z-space mid_return on v8/v9 (in P0.B output)
- ✓ P1.D — real stylized facts panel (script 81 ran on cluster; PDF in report_figures/)
- ⏳ P1.C — Lightning training curves: pull metrics.csv from `~/orcd/scratch/diffmm/checkpoints/intc_v{2,5,9}/lightning_logs/version_*/metrics.csv` then run script 82

### P2 — nice-to-have
- ✓ P2.G3 — regime-embedding t-SNE (script 84 ran)
- ✓ P2 — microstructure trio (script 83 ran)
- ✓ P2 — intraday LOB volume (script 86 ran)
- ✓ P2 — return distribution density+QQ (script 87 ran)
- ✓ P2.G4 — LSTM-size sweep (3 jobs ran on preemptable; outputs in `results/predictive_lstm_h{32,64,128}/`)
- ⏳ P2.G1 — guidance-weight sweep on v2 (~2h GPU; deferred)
- ⏳ P2.G2 — per-timestep denoising (~30min × 3 GPUs; deferred)

### P3 — out of scope
Autoregressive baseline, mixture-of-experts, Wasserstein loss, longer holdouts. Flagged as future work in §8.4.

---

## 5. Sync plan

### 5.1 Push code to cluster (after any local edit)
```bash
cd ~/Desktop/diff-market-making && rsync -avzR \
    scripts/{50_validate_generator,60_deep_diagnostic,62_trades_metrics,63_predictive_score,11_finetune_generator}.py \
    scripts/{81_real_stylized_facts,82_training_curves,83_microstructure_diagnostics,84_regime_embedding_tsne,85_per_timestep_denoising,86_intraday_lob_volume,87_return_distribution_full,88_per_regime_g1}.py \
    scripts/utils/{run_p0_post_overnight,run_all_parallel,run_tsla_post_training,run_spy_post_training,run_spy_chained_training,submit_spy_chain,run_p2_gpu_sweeps}.sh \
    favara@orcd-login001.mit.edu:diff_mm/
```
**Notes:**
- Use `orcd-login001.mit.edu` directly, not the round-robin endpoint (login008 has stale NFS mount as of 2026-05-10)
- For esampi (SPY work): swap `favara@…` → `esampi@…` in destination

### 5.2 Pull results back from cluster
```bash
# favara — TSLA + INTC artifacts
rsync -avz favara@orcd-login001.mit.edu:diff_mm/results/_archive/ \
    ~/Desktop/diff-market-making/results/_archive/
rsync -avz favara@orcd-login001.mit.edu:diff_mm/results/{trades_metrics_tsla,predictive_score_tsla,microstructure_diagnostics,trades_style,deep_diagnostic_v5_v9,ranking_mechanism_full}/ \
    ~/Desktop/diff-market-making/results/

# esampi — SPY artifacts (after chained training drains)
rsync -avz esampi@orcd-login001.mit.edu:diff_mm/results/{_archive/spy_v2,_archive/spy_v5,trades_metrics_spy,predictive_score_spy}/ \
    ~/Desktop/diff-market-making/results/
```
**Beware zsh strict globbing.** `{a,b}` expands locally; `*` expands locally. Quote the remote spec or use a per-iteration loop:
```bash
for m in v2 v5 v9; do
    rsync -avz "favara@orcd-login001.mit.edu:diff_mm/results/_archive/$m/validation_*" \
        ~/Desktop/diff-market-making/results/_archive/$m/
done
```

### 5.3 Merge new CSVs into the canonical files
After pulling P0.B/C results, merge into the figure-script's input CSVs:
```bash
python -c "
import pandas as pd
for name in ['per_feature_stats','correlation_comparison','temporal_acf',
            'window_level_stats','z_space_stats','regime_conditional','ofi_regression']:
    a = pd.read_csv(f'results/deep_diagnostic/{name}.csv')
    b = pd.read_csv(f'results/deep_diagnostic_v5_v9/{name}.csv')
    out = pd.concat([a, b], ignore_index=True).drop_duplicates()
    out.to_csv(f'results/deep_diagnostic/{name}.csv', index=False)
"

python -c "
import pandas as pd
for name in ['conditional_returns','a2_minus_a1_deltas','agent_decomposition']:
    a = pd.read_csv(f'results/ranking_mechanism/{name}.csv')
    b = pd.read_csv(f'results/ranking_mechanism_full/{name}.csv')
    out = pd.concat([a, b], ignore_index=True).drop_duplicates(
        subset=['model'] + [c for c in ['agent','regime','feature'] if c in a.columns],
        keep='last')
    out.to_csv(f'results/ranking_mechanism/{name}.csv', index=False)
"
```

### 5.4 Re-build figure suite
```bash
/Users/arturofavara/Desktop/diff-market-making/venv/bin/python scripts/80_report_figures.py
/Users/arturofavara/Desktop/diff-market-making/venv/bin/python scripts/88_per_regime_g1.py
```
Idempotent. Picks up whatever new CSVs have landed.

---

## 6. Bug ledger from this session (read this before re-doing any setup)

| # | Bug | Symptom | Fix |
|---|---|---|---|
| L1 | zsh strict globbing in rsync paths | `zsh: no matches found: <remote-path>` | Quote remote spec or per-iteration loop |
| L2 | Cross-ticker `wait_for_jobs` returned early | Orchestrator skipped through all stages in 1s | `log` writes stdout which got captured by `$(submit ...)` along with the JID — log to stderr instead |
| L3 | `pathlib.Path.mkdir` failed on broken symlink | `FileExistsError on 'data'` | Stale NFS mount on login008; switched to login001 |
| L4 | login008 has stale NFS mount of `/orcd/scratch` | `ls scratch_path` returns "no such file" but works on login001 | Always `ssh user@orcd-login001.mit.edu`, not round-robin |
| L5 | Lightning ckpts saved under `<root>/default/` not `<root>/` | `best_ckpt` couldn't find them | Extended glob: `ls $d/{,default/,lightning_logs/version_*/checkpoints/}ckpt-epoch*.ckpt` |
| L6 | SPY trainings completed 0 epochs in 6h → 0 checkpoints | Default save is `every_n_epochs=5`; SPY epoch ≈ 24h | Patched script 11 with `every_n_train_steps=N` callback; trigger via Hydra override |
| L7 | `${#CKPT[@]:-0}` invalid bash syntax | `bash: bad substitution` | `${#CKPT[@]}` is already safe under `set -u` |
| L8 | G1 CSV `value` column loaded as string | `TypeError: 'sub' not supported for dtype 'str'` in heatmap arithmetic | `pd.to_numeric(..., errors="coerce")` in loader |
| L9 | Script 50's `_sample_synth` couldn't filter to one regime | `FileNotFoundError: no synthetic tapes under .../base/{base,high_vol,toxic,thin}/` | Added `--regime` flag + flexible glob |
| L10 | Midprice traces empty for v9-v11 | Synth tapes lack `mid` column | Fall back to `cumprod(1 + mid_return)` when `mid` absent |
| L11 | z-space analysis skipped `mid_return` for v8/v9 | "Discrete features" filter excluded it | `force_continuous=("mid_return",)` parameter |
| L12 | PCA projections never persisted per-tape | Fig 4.1 stuck on bar fallback | Added `pca_projections.csv` write at end of `compute_pca_coverage()` |
| L13 | DataFrame.applymap removed in pandas 2.x | `AttributeError: no attribute 'applymap'` | Use `.map()` on DataFrame |
| L14 | iterrows returns (index, row) tuples | `AttributeError: 'tuple' object has no attribute 'name'` | Iterate `df.index` directly |
| L15 | rsync path with literal `~` | `mkdir /home/user/~/diff_mm failed` | Use `diff_mm/` (no tilde) — relative to remote home |
| L16 | rsync `~diff_mm` ≠ `~/diff_mm` | tilde-USER expansion looks for nonexistent user "diff_mm" | Same as L15 — drop the tilde |
| L17 | G1 CSV columns are `Check`, `Pass`, `Value` (capitalized) | Loader's lowercase-only matching missed them | Case-insensitive column matching via `_find()` helper |
| L18 | `Conditional moments (|z|<2)` row blank in heatmap | Validator emits string `"majority"` not numeric | Drop the row in figure scripts |

---

## 7. Tomorrow-morning checklist (in priority order)

1. **Check overnight runs**:
   ```bash
   ssh favara@orcd-login001.mit.edu "squeue -u favara"
   ssh esampi@orcd-login001.mit.edu "squeue -u esampi"
   ```
   - favara: TSLA v5 sampling should be done; cross-eval next
   - esampi: SPY chained training mid-flight (~12-18h in)

2. **TSLA pipeline completion** — re-run [run_tsla_post_training.sh](../scripts/utils/run_tsla_post_training.sh) on favara@login001:
   ```bash
   tmux new -s tsla_post
   ./scripts/utils/run_tsla_post_training.sh 2>&1 | tee logs/tsla_post_$(date +%Y%m%d_%H%M%S).log
   ```
   Skip-guards mean it only does what's missing.

3. **Pull TSLA results** when done:
   ```bash
   rsync -avz "favara@orcd-login001.mit.edu:diff_mm/results/_archive/tsla_*" \
       ~/Desktop/diff-market-making/results/_archive/
   rsync -avz favara@orcd-login001.mit.edu:diff_mm/results/{trades_metrics_tsla,predictive_score_tsla}/ \
       ~/Desktop/diff-market-making/results/
   ```

4. **Re-build figure suite** with the new TSLA data:
   ```bash
   /Users/arturofavara/Desktop/diff-market-making/venv/bin/python scripts/80_report_figures.py
   ```

5. **Investigate v9 missing rows** in `fig_p0d_g1_per_regime.pdf` — Trade-sign ACF, RV signature, ACF power-law β are blank for v9. Diff `head -20 results/_archive/v9/validation_base/G1_pass_fail_summary.csv` against v2's format. Either patch parser regex or document as known data gap.

6. **Pull Lightning logs for training curves figure**:
   ```bash
   for run in intc_v2 intc_v5 intc_v9; do
     rsync -avz "favara@orcd-login001.mit.edu:orcd/scratch/diffmm/checkpoints/$run/lightning_logs/" \
         ~/Desktop/diff-market-making/checkpoints_lightning/$run/lightning_logs/
   done
   python scripts/82_training_curves.py
   ```

7. **Start filling prose** in [final_report.tex](final_report.tex). Recommended order: §5 → §5.7 → §6 → §7 → §2 → §3 → §1 → executive summary.

8. **Wait for SPY** (~24h end-to-end from start time). Once complete, run [run_spy_post_training.sh](../scripts/utils/run_spy_post_training.sh) on esampi (path-mounted login001), then pull results and add SPY points to cross-asset figures.

---

## 8. Closing observations

The infrastructure is now in a steady state for the final report sprint:
- **All evaluation pipelines exist as one-shot scripts** — no manual orchestration of subordinate steps.
- **All CSVs flow into one figure suite** that auto-picks-up new data on re-run. Idempotent.
- **The cluster orchestrators submit to `mit_preemptable`** for short jobs (50+ node fan-out, no priority queue) and `mit_normal_gpu` for the 6h training chunks.
- **Two-account workflow** (favara INTC+TSLA, esampi SPY) is validated end-to-end.
- **The skeleton report knows where every figure lives** — section comments name the source CSV/script for each one.

Remaining work is largely "let SPY cook overnight + write prose". Most analysis decisions are now reproducible by re-running the relevant orchestrator + the figure suite locally.

---

## 9. Writing plan for [final_report.tex](final_report.tex)

### 9.1 Philosophy
- **Results first, narrative scaffolding last.** §5 (Results) → §5.7 (Copula case study) → §6 (Agents) → §7 (Cross-asset) → §8 (Discussion) → §3 (Methods) → §2 (Data) → §1 (Intro) → Exec Summary.
- **One subsection per task.** Keeps the depth high; each task gets a real research+writing turn rather than skimming the surface.
- **Always read the source code, the CSVs, and the rendered figure** before writing the section that references them. Numbers in prose must come from a verifiable source on disk, not memory.
- **Cite as we go.** Build the bibliography incrementally; never write a claim like "Karras et al. observed…" without putting Karras into `references.bib` in the same task.
- **Human verification between tasks.** After each task, the user reviews the prose + commits if good. Only then do we move on.

### 9.2 Per-task template
Each task runs the same protocol. I'll always declare these explicitly at the start of each task:

1. **Block.** Which §X.Y of `final_report.tex` we're filling.
2. **Goal.** Word count target + what the section needs to land (1-line thesis).
3. **Read first** (research-before-write checklist):
   - **Code:** specific files + line numbers (e.g. `src/diffmm/data/clean.py:144` for the trade filter).
   - **Data:** specific CSVs/JSONs + which columns or rows.
   - **Figures:** specific PDFs in `results/report_figures/` to open and visually verify before captioning them.
4. **External sources to cite.** Papers/textbooks needed; assume nothing is in `references.bib` until I add it.
5. **Questions for user** — ambiguities I'll need answered before writing. Examples: "should this section include the Phase A falsification?", "is your strongest claim 'tension is structural' or 'tension is INTC-specific'?", "want me to lead with the v9 z-collapse story or the v2_remapped story?".
6. **Inputs I might ask the user to provide visually** — e.g. "open `fig_4_1_pca_panels.pdf` and tell me whether the v2 hull crosses into the v9 region or stays disjoint" — for figure-driven prose where I can't see the image directly.
7. **Output.** The actual TeX edits I'll make + draft caption text + any new bibliography entries.
8. **Verification.** What I'll show the user before declaring the task done (typical: the diff, the rendered PDF if compilable locally, bib additions).

### 9.3 Task list
**T1–T16 = main body text** (in writing order, not document order). After T16 we're at "ready to compile end-to-end + polish pass":

| # | Block | Section title | Target | What feeds it |
|---|---|---|---|---|
| **T1** | §5 preamble | Anchor table 5.1 + fig 5.1, define the three axes upfront | 200 words | `multi_axis_comparison.csv`, `fig_1_1_three_axis_scatter.pdf` |
| **T2** | §5.1 | Three Representative Generators (v2/v5/v9) | 300 words | `multi_axis_comparison.csv`, `fig_10_1_rank_bump.pdf`, `tab_1_3_multi_axis.csv` |
| **T3** | §5.2 | Distributional Realism (G1) Results | 400 words | `G1_comparison_all_models.csv`, `fig_2_1_g1_numerical_heatmap.pdf`, `fig_2_3_return_tail.pdf`, `fig_2_4_trade_fraction.pdf` |
| **T4** | §5.3 | Temporal Transferability Results | 350 words | `predictive_score.csv`, `fig_3_1_predictive_score.pdf`, `fig_3_3_predictive_vs_scale.pdf` |
| **T5** | §5.4 | Multivariate Coverage Results | 250 words | `pca_coverage.csv`, `pca_projections.csv`, `fig_4_1_pca_panels.pdf`, `fig_4_2_pca_vs_g1.pdf` |
| **T6** | §5.5 | Anti-Correlation Across Architectures | 300 words | `tab_C_metric_correlations.csv`, `fig_1_2_pairwise_anticorrelation.pdf` |
| **T7** | §5.6 | Architectural Ablation | 400 words | `tab_A_decision_matrix.csv`, `fig_5_1_ablation_heatmap.pdf`, `fig_5_2_x0_clip_effect.pdf` |
| **T8** | §5.7.1 | Copula Story — Zero-Inflation Pathology | 250 words | `deep_diagnostic/per_feature_stats.csv` (frac_zero), `fig_6_2_zero_inflation.pdf` |
| **T9** | §5.7.2-3 | Copula Promise + Failure (v8 collapse) | 300 words | `z_space_stats.csv` (mid_return rows), `fig_6_1_z_space.pdf`, copula-transform code |
| **T10** | §5.7.4 | Copula Fix (v9 dequantization) | 250 words | `deep_diagnostic` for v9, `fig_6_3_v2_vs_v9.pdf`, dequantization code |
| **T11** | §5.7.5 | Spearman ≠ Economic Information (v2_remapped story) | 250 words | `multi_axis_comparison.csv` for v2_remapped row, `fig_3_3_predictive_vs_scale.pdf` |
| **T12** | §6 (Agents) | All four §6 subsections in one task — they're tightly coupled | 600 words total | `ranking_mechanism/*.csv`, `fig_8_1_agent_pnl_overlap.pdf`, `fig_8_2_rdiff_vs_everything.pdf`, `fig_8_3_a2_minus_a1.pdf` |
| **T13** | §7 (Cross-asset) | Cross-asset findings — INTC vs TSLA (vs SPY when ready) | 400 words | `tab_6_1_cross_asset.csv`, `fig_6_1_cross_asset_ternary.pdf`, `fig_6_2_cross_asset_axis_bars.pdf` |
| **T14** | §8.1-2 | Discussion — structural vs practical implications | 500 words | All of the above + Cont 2001 stylized facts paper |
| **T15** | §8.3-4 | Limitations + Future Work | 300 words | Phase A falsification appendix + the marginals-vs-joints framing |
| **T16** | §3 Methods | All of §3 (DDPM + 5 design axes + 20 checkpoints + eval framework) — 4 sub-tasks if too long | 800-1000 words | `src/diffmm/generator/*` source, Ho 2020, Salimans-Ho 2022, Karras 2022, Peebles-Xie 2023, Sklar 1959, Theis 2016, Hang 2023 |
| **T17** | §2 Data | LOB mechanics + features + splits + stylized facts | 700 words | `src/diffmm/data/*` source, TAQ documentation, Cont 2001, `fig_a_real_stylized_facts.pdf` |
| **T18** | §1 Intro | Backtesting problem + why diffusion + contributions | 600 words | TRADES paper (Berti 2025), GAN baselines (Li 2020, Coletta 2021/22), Ho 2020 |
| **T19** | Executive Summary | 250 words, non-technical | 250 words | Whatever's most-quoted in §1, §5 hero, §7 main finding |
| **T20** | references.bib + AI Disclosure | Compile final .bib from all citations accumulated; verify all `\cite{}` keys resolve | — | All previous tasks |

### 9.4 Verification protocol per task
After each task, I'll:
1. Show the diff (the actual TeX text added).
2. List any new entries added to `references.bib`.
3. (When I can compile locally) attach the rendered page or just confirm pdflatex builds without warnings on the relevant section.
4. Tell the user explicitly: "Verify, then say 'next' to advance to T<n+1>."

The user can:
- Approve verbatim ("next")
- Ask for revisions ("rewrite the X paragraph more cautiously / less academic / shorter / etc.")
- Re-direct ("skip T<n>, do T<m> first because <reason>")

### 9.5 Status tracking
Track the running state as `T<n>: status` in this doc. As tasks complete, mark them done here. Example:
```
T1 ✓ done — committed in <sha>
T2 ⏳ in flight
T3-T20 ☐ pending
```
This keeps the writing turn the same way the figure pipeline does — one source of truth.

### 9.6 Things I will NOT do without explicit go-ahead
- Add appendices beyond what's in the skeleton (the user can request them).
- Re-run any cluster jobs (the cross-comparison + plotting agents own that).
- Change the report's structural numbering (§5 ↔ §5.7 etc. is settled).
- Polish for IEEE house style until T20 (early polish wastes effort if structure changes).
- Generate or commit large appendix tables without showing them first.

---

**Ready to start at T1 on user approval.**
