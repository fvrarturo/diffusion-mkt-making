# Work10 — Report F: Appendix Fluidity Pass

**Goal.** Transform the appendix from a collection of disconnected
exhibits and empty stubs into a fluid, readable companion document.
ReportE shrank the body by ~6.8 pages, often by deleting useful
explanatory prose. ReportE-1 deleted some genuinely valuable
content. The appendix is the natural home for that depth. This
pass:

1. **Populates the six empty stubs** that currently contain only
   `% comments`. They look unprofessional in their current state
   and the cross-references from the body now resolve to literally
   nothing.
2. **Reintroduces selected body content** that was cut during
   ReportE — specifically the mechanism explanations and
   per-checkpoint detail that the compressed body now lacks.
3. **Adds orientation prose** to populated-but-spartan sections so
   each appendix section reads as a self-contained mini-essay
   rather than a figure block with a one-sentence caption.
4. **Removes any stub** that has no real content to support it,
   rather than leaving a labelled empty section in the published
   document.

**Working method.** Sixteen chunks (F1--F16), structured like
Work10_ReportE. Every chunk begins with a Read-first step that
grounds in current appendix state and the cut-content provenance.
Each chunk targets one appendix section (or the final audit).

---

## 0. Preflight: current appendix state

After ReportE, the appendix has fifteen `\section{}` blocks. Their
current content status:

| Appendix label | Lines | Figures | Tables | State |
|---|---:|---:|---:|---|
| `app:g1_matrix` | 65 | 4 | 0 | **Well-developed** |
| `app:metric_correlations` | 33 | 2 | 0 | Underdeveloped |
| `app:pca_coverage_full` | 55 | 0 | 1 | Well-developed |
| `app:ablation_full` | 30 | 1 | 0 | Underdeveloped |
| `app:cross_asset_full` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:checkpoints` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:deep_diagnostic` | 54 | 2 | 0 | Well-developed |
| `app:copula_impl` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:copula_mechanism` | 164 | 1 | 1 | **Fully fleshed-out** |
| `app:agents_impl` | 65 | 0 | 1 | Well-developed (populated by E2) |
| `app:lstm_details` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:traces` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:vol_vol` | 17 | 1 | 0 | Underdeveloped |
| `app:phaseA` | 0 | 0 | 0 | **EMPTY STUB** |
| `app:reproducer` | 14 | 0 | 0 | Underdeveloped (SPY only) |

Six empty stubs, four underdeveloped, five well-developed. The six
stubs are the highest-priority targets — they currently resolve to
section headers with no body content, which is worse than not
existing at all because the body cross-references them.

---

## 1. Decision rules

### 1.1 When to reintroduce body content (from ReportE cuts)

**Bring back** if any of the following hold:
- The cut prose contained a technical mechanism that the curious
  reader needs but that the compressed body now omits entirely
  (e.g., the sigma-vs-pred non-monotonicity breakdown,
  per-checkpoint PCA hull-area numbers).
- The cut prose was a structural definition (TRADES protocol,
  dequantization algorithm) that the body forwards to the
  appendix but the appendix doesn't yet supply.
- The cut prose was specific numerical detail that adds rigor
  (per-checkpoint configurations, calibrated parameter values).

**Don't bring back** if any of the following hold:
- The cut prose was redundant restatement of the body's
  consistency-check framing or three-axis-tension narrative.
- The cut prose was a self-referential forward-pointer (e.g.,
  "§7.3 returns to this").
- The body already covers the content with adequate compactness
  and the appendix would only duplicate.

### 1.2 When to drop an empty stub instead of populating

Drop if:
- There is no genuine technical content to write that isn't
  already in the body or another appendix.
- The body has no `\ref{}` to the label (orphan stub).

Populate if:
- Real source material exists in `results/`, `scripts/`, or
  `notebooks/`.
- One or more body refs point at the label (so dropping would
  break a body reference).

### 1.3 Style for new appendix prose

- **Opening sentence orients the reader.** What does this section
  contain, and why is it in the appendix? E.g., "Appendix
  C extracts numerical detail underlying the §4.4 PCA coverage
  result."
- **Sub-paragraphs use `\paragraph{...}` headers** to break long
  exhibits into navigable pieces. Match the style already used in
  `app:copula_mechanism`.
- **Captions describe what's shown**, prose describes how to read
  it. Don't restate the caption.
- **Cross-references to body sections are sparse.** The appendix
  is the deeper-dive; over-pointing back to the body makes it
  feel parasitic.
- **Sentences ≤ 25 words**, paragraphs ≤ 8 sentences. Same prose
  style as the body.

---

## 2. Cut-from-body content catalog (ReportE traceability)

Inventory of what was cut during ReportE that is candidate material
for appendix reintroduction. Each entry: source location in the
pre-ReportE body, what was cut, and recommended disposition.

| # | Source | What was cut | Disposition |
|---|---|---|---|
| 1 | §4.3 opening (E11/finale) | TRADES protocol restatement (LSTM, train-synth/test-real, $1.0\times$ replay meaning) | → `app:lstm_details` (F11) |
| 2 | §4.3 closing (finale) | "What the LSTM is picking up" sign-sequence mechanism paragraph | → `app:lstm_details` (F11) |
| 3 | §4.3 middle (E11) | Sigma-vs-pred three-cluster breakdown (scale-matched 10--36×, noclip 4--14×, copula 28--210×) | → `app:metric_correlations` (F2) |
| 4 | §4.4 (finale) | Full PCA reference-points walkthrough (v8 85.2%, v10\_cascade 84.9%, hull-area numbers, off-center projection mechanism, v8\_b/v11 degeneracy) | → `app:pca_coverage_full` (F3) |
| 5 | §4.6 ablation walks (E13) | Original full prose of three walks (v2→v5 two-toggle decomposition, v2→v9 copula+dequant decomposition, noclip mechanism explanation) | → `app:ablation_full` (F4) |
| 6 | §4.1 closing (finale) | "Three rankings do not agree" Pareto summary paragraph | **DROP** — figure caption carries it |
| 7 | §4.2 spread-conditional (E4 + finale) | Joint-structure description "spread widens AND book thickens AND prices move" | → `app:deep_diagnostic` intro (F7) — partially there |
| 8 | §5.4 (finale) | "What this is not" framing recap | **DROP** — restated §5 opening |
| 9 | §6.1 (finale) | Expanded cross-asset rationale (INTC pathology hypothesis, ticker-selection criteria, v9 deferral reason) | → `app:cross_asset_full` (F5) |
| 10 | §6.2 (finale) | Three observations in full (bolded-header paragraphs with mechanism per finding) | → `app:cross_asset_full` (F5) |
| 11 | §7.2 (E16) | Five practitioner-paragraph blocks (replaced by table) | **DROP** — table caption covers patterns |
| 12 | §7.3+§7.4 (E16) | Expanded limitations and future-work pedagogical detail | **DROP** — numbered list is sufficient |

---

## 3. Chunk-by-chunk plan

Sixteen chunks. Ordering favors the empty stubs first (highest
value per chunk), then reintroduction-of-cut-content chunks, then
the polish/audit pass.

---

### Chunk F1 — `app:cross_asset_full`: populate empty stub

**Priority.** P0 (empty stub; body refs from §6.2 table caption
and from a comment in §6.4).

**Read first.**
1. Current `app:cross_asset_full` section in
   [docs/final_report.tex](final_report.tex) — confirm still empty
   except for the `%` comment about populating when SPY completes.
2. Body §6.1, §6.2, §6.4 — what's been kept and compressed.
3. ReportE-final cuts to §6.1 (cross-asset rationale) and §6.2
   (three observations). These are the natural reintroduction
   targets.
4. `results/_archive/multi_axis_comparison.csv` — does it have
   per-ticker rows?
5. `tab:cross_asset_tsla` block in body §6.2 — extends naturally.

**Scope.** Populate with:
- Brief intro: what the cross-asset evidence base looks like at
  submission time (INTC complete, TSLA complete, SPY in flight).
- Full combined table extending `tab:cross_asset_tsla`. Add a
  real-data reference row per ticker; mark SPY rows as pending if
  the CSV supports it.
- Reintroduce the §6.1 ticker-selection rationale (small-tick TSLA
  vs index-ETF SPY chosen because they bracket the INTC
  large-tick / zero-inflated regime), and the v9-deferral reason
  (per-ticker dequantization recalibration is dataset-specific and
  was out of scope).
- Reintroduce the §6.2 three observations in their original
  fuller form — bolded-header paragraphs that the body now only
  enumerates inline. These describe:
  (i) the TSLA Pareto-dominance finding;
  (ii) TSLA v2's $4.86\times$ best-non-broken predictive score;
  (iii) TSLA v5's PCA collapse to $18.9\%$ as same-mechanism as
  INTC noclip family.
- Forward-pointer to `app:reproducer` for SPY chained-training
  detail.

**Verification.**
- Section grows from $0$ to ~80--100 lines.
- New table label `tab:cross_asset_full` defined.
- Body refs to `app:cross_asset_full` (one from §6.2 caption, one
  comment in §6.4) still resolve.
- No new orphan refs.

**Effort.** ~30 min (some prose recovery from git, some new).

---

### Chunk F2 — `app:checkpoints`: populate empty stub

**Priority.** P0 (empty stub; referenced from §3.3 main body in
the protagonist sweep description, also from §3.2 EDM
preconditioning subsection).

**Read first.**
1. `app:checkpoints` current state (empty stub).
2. Body §3.3 "The 20 Trained Checkpoints" — what's referenced as
   "full configuration matrix is given in
   Appendix~\ref{app:checkpoints}".
3. `results/_archive/multi_axis_comparison.csv` for the per-model
   axis values.
4. `notebooks/2026-05-08_phase-F-experiments.md` and
   `docs/Work6_phaseG.md` for the per-checkpoint training
   configurations.

**Scope.** Populate with:
- One-paragraph intro: this appendix reports the full $20$-model
  ablation matrix, organized along the three logical arms from
  §3.3 (scale-accurate, scale-exploded, copula).
- A table with one row per checkpoint, columns:
  prediction-target, conditioning, EDM yes/no,
  $\sigma_\text{data}$, data-transform (z-score / copula /
  copula+dequant / post-hoc remap), Min-SNR $\gamma$, $x_0$ clip
  yes/no, G1 pass count, predictive ratio, PCA coverage, trade
  fraction. Twenty rows.
- Per-arm summary paragraphs (~3 sentences each) interpreting
  what the arm explored. The scale-accurate arm tests the
  $\eps$-vs-$v$ choice with various EDM settings; the
  scale-exploded arm drops the safety bounds to probe the
  predictive corner; the copula arm tests the transform axis.
- Concluding paragraph naming which checkpoints became
  protagonists (v2/v5/v9) and why.

**Verification.**
- New table `tab:checkpoint_configs` defined.
- Body §3.3 ref to `app:checkpoints` resolves to substantive
  content.

**Effort.** ~40 min (need to compile the per-model configs from
multiple sources).

---

### Chunk F3 — `app:copula_impl`: populate empty stub

**Priority.** P0 (empty stub; referenced from §3.2.4 main body
copula-transform subsection).

**Read first.**
1. `app:copula_impl` current state (empty stub).
2. Body §3.2.4 "Data Transform" — the high-level summary.
3. Body §4.7.1 "The Fix (v9): Dequantization" — already has the
   algorithmic detail. The appendix can either repeat the algorithm
   with implementation parameters, or supplement with the
   per-feature treatment table.
4. `scripts/utils/run_v9_post_training.sh` and the relevant python
   scripts for the dequantization threshold and ε computation.

**Scope.** Populate with:
- One-paragraph intro: this appendix contains the implementation
  detail for the copula+dequantization transform that v9 uses; the
  high-level mechanism is in §3.2.4 and the failure mode is in
  §4.7.1 + `app:copula_mechanism`.
- Per-feature dequantization treatment table:
  - column 1: feature name (\texttt{mid\_return},
    \texttt{trade\_sz}, etc.)
  - column 2: point-mass fraction (e.g., $88.85\%$ for
    \texttt{mid\_return})
  - column 3: dequantization treatment (e.g., uniform on
    $[-\delta, 0]$ with $\delta$ value)
  - column 4: notes (e.g., "skip — no dominant point mass")
- Algorithm box (\texttt{algorithmic} environment) for the
  dequantization step.
- Note on inference-time noise injection: the same $\delta$ is
  applied to synthetic $z$ before inverse transform so the
  dequantized zeros decode back to small returns.

**Verification.**
- Section grows from $0$ to ~40--60 lines.
- Per-feature treatment table label resolves.
- Body §3.2.4 ref still resolves.

**Effort.** ~25 min.

---

### Chunk F4 — `app:lstm_details`: populate empty stub

**Priority.** P0 (empty stub; referenced from §3.4.2 main body
predictive-score definition, and natural home for the cut
TRADES-protocol restatement).

**Read first.**
1. `app:lstm_details` current state (empty stub).
2. Body §3.4.2 "Temporal Pattern Transferability (Predictive
   Score)" — what's defined briefly.
3. `scripts/63_predictive_score.py` for the LSTM architecture and
   training detail.
4. ReportE-finale cut: the §4.3 opening "TRADES protocol follows
   ... train an LSTM on synthetic windows..." restatement.
5. ReportE-finale cut: the §4.3 closing "What the LSTM is picking
   up" sign-sequence mechanism paragraph.
6. `results/predictive_score_all/predictive_score.csv` for the
   per-checkpoint scores.

**Scope.** Populate with:
- One-paragraph intro framing the predictive-score evaluation as
  the TRADES-protocol implementation choice in this study.
- **LSTM architecture paragraph.** Two-layer LSTM, hidden dim
  (recover from training script), trained to predict next
  mid-return from a 20-event history.
- **Training protocol paragraph.** Reintroduce the cut §4.3
  protocol restatement: train on synthetic, test on real
  held-out window, report ratio of test MAE to
  train-on-real/test-on-real baseline.
- **Mechanism: what the LSTM measures.** Reintroduce the cut
  "what the LSTM is picking up" paragraph — the LSTM reads
  per-event value sequences, not marginal shape; this is the
  load-bearing observation for §4.7.2.
- **Robustness paragraph** (optional, if relevant data exists in
  `results/predictive_lstm_h{32,64,128}/`): LSTM-size sensitivity
  sweep results, confirming the qualitative ranking is stable.

**Verification.**
- Section grows from $0$ to ~50--70 lines.
- Body §3.4.2 ref resolves.
- The cut §4.3 sign-sequence mechanism is no longer lost.

**Effort.** ~30 min.

---

### Chunk F5 — `app:phaseA`: populate empty stub

**Priority.** P0 (empty stub; referenced from §3.2.5 Min-SNR
subsection as the falsification log).

**Read first.**
1. `app:phaseA` current state (empty stub).
2. Body §3.2.5 "Loss Weighting: Min-SNR-$\gamma$" — references the
   falsification log.
3. `notebooks/2026-05-06_phase-E-copula-launch.md` and earlier
   phase docs (e.g., `docs/Work6_phaseB.md`) for the Phase A
   misadventure.
4. Confirm that the $\gamma=1$ hypothesis was tested and
   discarded, with a directional finding.

**Scope.** Populate with:
- Intro: this appendix is a brief "what we tried and discarded"
  log for the Min-SNR-$\gamma$ choice. We test $\gamma=1$ early
  on the hypothesis that capping the loss weight more aggressively
  would help convergence on heavy-tailed data; the test
  falsified that direction and Phase B onward uses $\gamma=5$
  (Hang et al.'s default).
- Per-experiment paragraph (~2 sentences each):
  - Setup: $\gamma=1$ vs $\gamma=5$ on the v1 baseline.
  - Result: $\gamma=1$ degrades G1 by N checks, predictive by M×.
  - Diagnosis: $\gamma=1$ caps the low-noise weight too
    aggressively; the model under-fits the bulk-shape signal.
- Lesson: the published default $\gamma=5$ wasn't arbitrary;
  hyperparameter mathematical structure was directionally
  informative.

**Verification.**
- Section grows from $0$ to ~20--30 lines.
- Body §3.2.5 ref resolves.

**Effort.** ~20 min.

---

### Chunk F6 — `app:traces`: populate OR drop

**Priority.** P1 (empty stub; unsure if any body refs).

**Read first.**
1. `app:traces` current state (empty stub).
2. `grep -n "app:traces" docs/final_report.tex` — count body refs.
3. `results/trades_metrics_all/midprice_traces_*.pdf` files — do
   these exist for all twenty models?
4. ReportE figure cuts — was a midprice-trace exhibit cut from the
   body?

**Scope.** Two paths depending on what the Read step finds:

**Path A (populate):** If midprice trace figures exist and there
is at least one body ref:
- Intro paragraph explaining what midprice traces show (per-model
  $2{,}560$-event normalized price walks).
- One composite figure (or a small grid) showing the
  representative trace per model — for v9/v9\_b/v11 note the
  caveat that synth tapes only contain `mid_return` so the trace
  is reconstructed via `cumprod(1 + mid_return)`.
- Brief per-architectural-family commentary (~2 sentences each).

**Path B (drop):** If no body ref exists and no canonical trace
exhibit was prepared:
- Remove the `\section{Additional Mid-Price Traces}` and
  `\label{app:traces}` block entirely.
- Update any body refs to the dropped label (likely none, since
  no body grep hit it).

**Verification.**
- Either the section is meaningfully populated, OR it is removed
  and no orphan refs result.

**Effort.** ~20 min (mostly the read-step decision).

---

### Chunk F7 — `app:metric_correlations`: expand with cut content

**Priority.** P1 (underdeveloped; natural home for the cut
sigma-vs-pred non-monotonicity prose).

**Read first.**
1. Current `app:metric_correlations` content (33 lines, 2 figures
   — `fig:pairwise_anticorrelation_app`,
   `fig:pred_vs_scale_app`).
2. ReportE-E11 cut: the §4.3 paragraph "The relationship between
   synth-to-real return-scale ratio and predictive score is
   \emph{non-monotone}: ..." with the three-cluster breakdown
   (scale-matched cluster spans $10$--$36\times$; noclip family
   at $\sigma_\text{synth} \approx 40\sigma_\text{real}$ sits at
   $4$--$14\times$; copula family at
   $\sigma_\text{synth} \approx 0.6\sigma_\text{real}$ at
   $28$--$210\times$).

**Scope.**
- Keep the existing intro paragraph and both figures.
- Add a new `\paragraph{Sigma-vs-predictive non-monotonicity.}`
  section after `fig:pred_vs_scale_app`, reintroducing the cut
  three-cluster breakdown. The figure already plots the
  non-monotone relationship visually; the prose now interprets
  it cluster-by-cluster.
- Add a closing paragraph linking the two figures: the
  rank-correlation matrix (in body Table~\ref{tab:correlation_subset})
  and the pairwise scatter give the same story from different
  angles; the sigma-vs-predictive view exposes a third axis
  (output magnitude) that the pairwise view collapses.

**Verification.**
- Section grows from 33 to ~50--60 lines.
- The cut sigma-vs-pred mechanism is no longer lost.

**Effort.** ~20 min.

---

### Chunk F8 — `app:pca_coverage_full`: reintroduce cut interpretation

**Priority.** P1 (already has the table; missing the cut
interpretation prose).

**Read first.**
1. Current `app:pca_coverage_full` content (55 lines, full table).
2. ReportE-finale cut: the §4.4 paragraph naming v8 ($85.2\%$),
   v10\_cascade ($84.9\%$), with hull-area numbers
   ($25.2$~PC1-PC2-area-unit reference), v8\_b/v11 degeneracy
   note, and the mechanism explanation (mode-collapsed marginals
   pushed projection off-center).

**Scope.**
- Keep the existing intro and table.
- After the table, add a substantive interpretation section
  reintroducing the cut content as three `\paragraph{...}`
  blocks:
  - `\paragraph{Coverage-dominant families.}` — v8 and
    v10\_cascade win on coverage because their hulls are
    $2$--$2.6\times$ real's, but their marginals are collapsed
    (kurtosis $\sim 2$); coverage is not realism.
  - `\paragraph{The protagonist trio in context.}` — v2/v5/v9 sit
    at $46$--$71\%$ with hull areas $0.5$--$1.2\times$ real,
    inside the "realistic coverage" band.
  - `\paragraph{Noclip-family zero coverage.}` — the three noclip
    checkpoints produce the largest hulls in the study
    ($200$--$500\times$ real's area) yet land at $0\%$ coverage
    because their projections fall entirely outside real's
    region. v8\_b and v11 are excluded because PCA is undefined
    when synth std is zero on at least one feature.

**Verification.**
- Section grows from 55 to ~80--90 lines.
- The cut §4.4 interpretation is recoverable from the appendix
  for any reader following the §4.4 forward-pointer.

**Effort.** ~20 min.

---

### Chunk F9 — `app:ablation_full`: reintroduce cut walks

**Priority.** P1 (currently has the heatmap + one interpretation
paragraph; missing the original detailed walk prose).

**Read first.**
1. Current `app:ablation_full` content (30 lines, 1 figure).
2. ReportE-E13 cut: the three original ablation walks (v2→v5,
   v2→v9, noclip) at ~16/19/16 lines each. The compressed body
   versions now run 3 sentences each.

**Scope.**
- Keep the existing intro and heatmap.
- Reintroduce the three original walks verbatim as
  `\paragraph{v2 $\to$ v5 (full walk).}`,
  `\paragraph{v2 $\to$ v9 (full walk).}`,
  `\paragraph{Noclip family (full walk).}`. The body cites the
  compressed versions; the appendix supplies the per-toggle
  decomposition.
- Add a one-sentence cross-link at the end pointing back to
  Table~\ref{tab:ablation_walks} as the numerical source.

**Verification.**
- Section grows from 30 to ~80--100 lines.
- Body §4.6 compressed walks still readable on their own;
  appendix supplies depth on demand.

**Effort.** ~25 min.

---

### Chunk F10 — `app:deep_diagnostic`: tighten and orient

**Priority.** P2 (already populated with two figures; just needs
better orientation prose).

**Read first.**
1. Current `app:deep_diagnostic` content (54 lines, 2 figures).
2. Confirm the two paragraphs (`Spread-conditional dynamics.`,
   `Intraday structure.`) read as a coherent section.

**Scope.**
- Add a 2-sentence section-opening paragraph BEFORE the two
  `\paragraph{...}` blocks: this appendix collects microstructure
  diagnostics that are not in the body but that diagnose specific
  ways the protagonists fail to reproduce real INTC. Outputs come
  from `scripts/60_deep_diagnostic.py`.
- Verify both `\paragraph{...}` headers are crisp.
- No content reintroduction needed — body cuts already landed
  here.

**Verification.**
- Section grows slightly (~5 lines).
- Reads as a coherent appendix opening rather than two
  disconnected paragraphs.

**Effort.** ~10 min.

---

### Chunk F11 — `app:vol_vol`: add orientation paragraph

**Priority.** P2 (one figure, no prose intro).

**Read first.**
1. Current `app:vol_vol` content (17 lines: one figure with a
   verbose caption, no prose intro).
2. Body §4.2 vol-vol-correlation paragraph — what's said in body.

**Scope.**
- Add a 2--3 sentence intro paragraph BEFORE the figure block.
  Content: vol-vol correlation is Cont (2001) stylized fact (v).
  Real INTC is unimodal at $\approx +0.34$ per-tape Pearson; the
  body summarizes; this appendix shows the per-architecture
  distribution. Note v9's exclusion (high \texttt{frac\_zero}).
- Optionally trim the over-verbose figure caption now that an
  intro paragraph carries the orientation.

**Verification.**
- Section grows from 17 to ~25--30 lines.
- Figure caption tightened.

**Effort.** ~10 min.

---

### Chunk F12 — `app:g1_matrix`: polish

**Priority.** P3 (already well-developed; minor polish only).

**Read first.**
1. Current `app:g1_matrix` content (65 lines, 4 figures).

**Scope.** Light polish:
- Tighten the captions on `fig:trade_fraction_app` and
  `fig:return_tail_app` (currently verbose).
- Add one transitional sentence between the two heatmaps
  (`fig:g1_heatmap_full` and `fig:g1_heatmap_per_regime`) and the
  two complementary-view figures, so the section reads as a
  unified four-exhibit sequence rather than a "main two + extras"
  split.
- Optional: add a closing paragraph linking the G1 numerical
  view to the predictive-score axis (the v9 trade-fraction
  realist is the same generator that pays the predictive cost,
  cross-link to `app:lstm_details` after F4 lands).

**Verification.**
- Section stays around 65--75 lines.
- Reads as a coherent G1-evidence appendix.

**Effort.** ~10 min.

---

### Chunk F13 — `app:copula_mechanism`: minor polish

**Priority.** P3 (fully fleshed-out at 164 lines).

**Read first.**
1. Current `app:copula_mechanism` content.
2. Verify all `\paragraph{...}` headers, the four-step argument,
   table, and figure are in good order.

**Scope.** Minimal — this section already does what F-pass is
trying to achieve for other sections:
- Verify cross-link to new `app:copula_impl` (F3) is added once
  that section lands.
- Audit the "§5.7" references that survived ReportD — the section
  is now §4.7; these should be `\S\ref{sec:copula_story}` not
  literal "§5.7". Currently there are at least two ("the §5.7
  story" appears in the `fig:z_space_collapse` caption).
- One sentence cross-linking to `app:copula_impl` for the
  algorithm: "Appendix~\ref{app:copula_impl} reports the
  per-feature dequantization treatment table and pseudocode."

**Verification.**
- All §5.7-literal references replaced with `\S\ref{sec:copula_story}`.
- New cross-link to `app:copula_impl` exists.

**Effort.** ~10 min.

---

### Chunk F14 — `app:agents_impl`: parameter values + expand

**Priority.** P2 (already populated by E2; can add calibrated
parameter values).

**Read first.**
1. Current `app:agents_impl` content (65 lines, 1 table).
2. `scripts/utils/run_*.sh` or agent calibration outputs for the
   actual $\gamma$, $\kappa$, $T$ values used per ticker.
3. ReportE-finale cuts: anything from §5 that could land here?
   (§5.1 was already moved here by E2; §5.4 cuts were redundant
   restatements — don't bring back.)

**Scope.**
- After the existing AS-family + four-agent bullets, add a small
  parameter-value table (`tab:agent_params`): rows = INTC/TSLA,
  columns = $\gamma$, $\kappa$, $T$, calibration date / sample
  used. Values come from agent calibration outputs.
- Optionally: a small per-generator agent-PnL breakdown table
  (rows = generators, columns = A0/A1/A2/A3 terminal PnL) that
  expands the §5.2 / §5.3 reference. This would mirror the
  `tab:agent_truth` style but across generators.
- No body-prose reintroduction; the §5 cuts were redundant.

**Verification.**
- Section grows by ~20--30 lines (parameter table) up to
  ~60 lines (if per-generator PnL table added).
- New table label(s) defined and resolve.

**Effort.** ~25 min.

---

### Chunk F15 — `app:reproducer`: expand beyond SPY

**Priority.** P2 (currently SPY-only).

**Read first.**
1. Current `app:reproducer` content (14 lines: SPY chained
   training only).
2. `scripts/utils/*.sh` for the canonical reproducer commands:
   training launch, sampling, validation, cross-eval, predictive,
   TRADES metrics.
3. Body refs to `app:reproducer` (currently three: §6.2 TSLA
   intro, `fig:cross_asset_bars` caption, §6.4 SPY mention).

**Scope.** Add canonical reproducer sections beyond SPY:
- `\paragraph{Single-model training.}` — the v2 training command
  (Hydra config + sbatch invocation). Mention how to swap to v5
  / v9 via config overrides.
- `\paragraph{Sampling.}` — `scripts/utils/run_*.sh` invocation
  for sampling $400$ tapes per architecture.
- `\paragraph{Evaluation pipeline.}` — sample → B/C/D validation
  → TRADES metrics → predictive score, with the orchestrator
  script (`run_p0_post_overnight.sh` and equivalents).
- `\paragraph{Cross-asset.}` — TSLA full-pipeline orchestrator.
- The existing SPY paragraph stays last (the most recent and
  most fragile of the reproducers).

**Verification.**
- Section grows from 14 to ~60--80 lines.
- Body refs to `app:reproducer` all resolve to substantive
  content (not just SPY).

**Effort.** ~25 min.

---

### Chunk F16 — Final audit + appendix narrative arc check

**Priority.** P0 (sweeps the whole pass).

**Read first.**
1. Re-extract `\section{}`, `\label{}`, and `\ref{}` from the
   full doc.
2. Read each appendix section's opening paragraph; assess flow.

**Scope.**
- **Cross-ref audit.** Confirm all body `\ref{app:...}` resolve.
  Confirm all body `\ref{fig:..._app}` and `\ref{tab:...}` that
  now live in appendix resolve.
- **Drop orphan stubs.** Any appendix section that is still empty
  after F1--F15 (or that F6 chose to drop) is removed cleanly,
  with body refs updated.
- **Narrative arc.** Each appendix section's first sentence
  should orient the reader. Verify each one does. The implicit
  arc:
  - §A (g1_matrix), §B (metric_correlations), §C
    (pca_coverage_full), §D (ablation_full) = the four axes of
    the three-axis tension in numerical detail.
  - §E (cross_asset_full), §F (checkpoints) = the population
    detail.
  - §G (deep_diagnostic), §H (copula_impl), §I (copula_mechanism)
    = the microstructure + copula deep-dives.
  - §J (agents_impl) = agent backtesting machinery.
  - §K (lstm_details), §L (vol_vol), §M (traces if kept) =
    evaluation methodology details.
  - §N (phaseA), §O (reproducer) = reproducibility log + commands.
- **Page-count check.** Confirm the appendix grew by an
  appropriate amount (target: ~3--4 pages added, $\le 0.5$ pages
  added to body via spillover).

**Verification.**
- Zero orphan refs.
- Zero empty `\section{}` blocks.
- Appendix reads top-to-bottom as a coherent companion document.

**Effort.** ~30 min.

---

## 4. Appendix narrative arc (post-F)

The appendix should ideally be reorderable into a logical narrative.
After F, the order would be:

| Order | Label | Theme |
|---|---|---|
| §A | `app:g1_matrix` | Distributional realism, full evidence |
| §B | `app:metric_correlations` | Architecture-space anti-correlation |
| §C | `app:pca_coverage_full` | Multivariate coverage, per-checkpoint |
| §D | `app:ablation_full` | Per-toggle effects, full walks |
| §E | `app:cross_asset_full` | Cross-ticker findings, full detail |
| §F | `app:checkpoints` | The $20$ trained variants |
| §G | `app:deep_diagnostic` | Microstructure diagnostics |
| §H | `app:copula_impl` | Copula transform implementation |
| §I | `app:copula_mechanism` | Copula failure mechanism (deep-dive) |
| §J | `app:agents_impl` | Agent-based stress testing detail |
| §K | `app:lstm_details` | Predictive-score evaluation detail |
| §L | `app:vol_vol` | Volume-volatility correlation distribution |
| §M | `app:traces` | (if kept) per-model midprice traces |
| §N | `app:phaseA` | Min-SNR-$\gamma$ falsification log |
| §O | `app:reproducer` | Reproducer commands |

Current order in the tex file is close to this — minor reordering
may be worthwhile in the F16 audit step, but is not load-bearing.

---

## 5. Effort + ordering

Sixteen chunks. Suggested execution order (by tier, then by
dependency):

| Chunk | Tier | Effort | Dependency |
|---|---|---|---|
| F1 — `app:cross_asset_full` populate | P0 | 30 min | — |
| F2 — `app:checkpoints` populate | P0 | 40 min | — |
| F3 — `app:copula_impl` populate | P0 | 25 min | — |
| F4 — `app:lstm_details` populate | P0 | 30 min | — |
| F5 — `app:phaseA` populate | P0 | 20 min | — |
| F6 — `app:traces` populate or drop | P1 | 20 min | — |
| F7 — `app:metric_correlations` expand | P1 | 20 min | — |
| F8 — `app:pca_coverage_full` reintroduce | P1 | 20 min | — |
| F9 — `app:ablation_full` reintroduce | P1 | 25 min | — |
| F10 — `app:deep_diagnostic` orient | P2 | 10 min | — |
| F11 — `app:vol_vol` orient | P2 | 10 min | — |
| F12 — `app:g1_matrix` polish | P3 | 10 min | — |
| F13 — `app:copula_mechanism` polish | P3 | 10 min | F3 |
| F14 — `app:agents_impl` parameter table | P2 | 25 min | — |
| F15 — `app:reproducer` expand | P2 | 25 min | — |
| F16 — Final audit | P0 | 30 min | all prior |

**Total estimated effort:** ~6 hours of focused conversation.

**Suggested execution grouping:**

1. **Empty-stub population first** (highest visual + cross-ref
   value): F1, F2, F3, F4, F5, F6.
2. **Reintroduction passes** (recovers cut depth): F7, F8, F9.
3. **Polish passes** (orientation + parameter values): F10, F11,
   F12, F13, F14, F15.
4. **Audit:** F16.

---

## 6. What this pass will NOT do

- **Will not re-grow the body.** All reintroduction lands in the
  appendix. Body stays at its post-ReportE size.
- **Will not reverse any ReportE compression.** Body-side cuts
  remain. The appendix is the deeper-dive companion, not a
  rollback.
- **Will not add new cross-references back from appendix to body
  beyond minimal "see §X" pointers.** The body should feel
  self-contained.
- **Will not introduce new bibliography entries** unless a
  specific chunk requires citing a paper not currently cited
  (none currently anticipated).
- **Will not rewrite already-fluent sections** (`app:copula_mechanism`
  is the model; not touching it except for F13 minor polish).

---

## 7. Decision points the user may want to flag before execution

Two judgment calls worth surfacing before chunks execute:

**Q1: How aggressively should F6 (`app:traces`) be dropped vs.
populated?** The cleanest answer depends on whether a midprice-trace
exhibit was intended in the original 25-figure plan
(`docs/Work10_Report_Instructions.md`). If yes, populate; if not,
drop. The Read-step in F6 will answer this.

**Q2: How much per-checkpoint detail belongs in F2
(`app:checkpoints`)?** A full $20 \times 8$ configuration table is
the standard answer, but the user may want a leaner version that
groups by architectural arm and only names key axis settings
(e.g., $\eps$ vs $v$, FiLM vs AdaLN, clip vs noclip) without all
six toggles per row. Recommend the full table for completeness,
but flag the choice.

---

## 8. Verification protocol per chunk

After each chunk:

1. `wc -l docs/final_report.tex` — confirm growth is in the
   appendix range expected for that chunk.
2. `grep -c "\\\\section{" docs/final_report.tex` — section count
   stays constant (no new top-level sections unless F6 drops
   `app:traces`).
3. `grep -c "label{app:" docs/final_report.tex` — confirm
   appendix-label count (drops by 1 if F6 takes the drop path).
4. Read the chunk's affected section top-to-bottom; confirm the
   intro paragraph orients, the body content reads as prose, and
   the section closes with a transitional sentence or natural
   end-point.
5. Cross-ref audit: `grep -oE '\\\\ref\{(fig|tab|sec|app):[^}]+\}' docs/final_report.tex | sort -u`
   diffed against the label set; confirm zero broken refs.
6. User reviews diff and approves before moving on.

---

**Ready to execute.** Recommended first chunk: **F5
(`app:phaseA`)** — smallest of the P0 stubs and lowest-risk for
validating the F-pass methodology. Alternative first chunk: **F1
(`app:cross_asset_full`)** — highest body-cross-ref impact, since
the §6.2 figure caption and §6.4 comment both point at it.
