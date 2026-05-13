# Work10 — Report E: Final Compression Pass (~12 pages → ~25.5 final)

**Goal.** Bring the body of `docs/final_report.tex` down by ~12 pages
through three concentric tiers of cuts: (Tier 1) move five supporting
figures + the agent-design enumeration + the SPY subsection to the
appendix; (Tier 2) compress every remaining body-figure caption to
"what's shown + one-sentence takeaway"; (Tier 3) tighten prose in
seven specific subsections. The target body figure count drops from
~14 to ~9 (plus one body table that stays).

**Decision rule.** Anything load-bearing for the three-axis argument
or the v9-microstructure case stays. Anything that is supporting
evidence (per-regime gradient, conditional moments, joint coverage of
a tail) moves to appendix. The body is for the headline figures; the
appendix is for the reader who wants to verify them.

**Working method.** Seventeen chunks (E1–E17), structured similarly
to `Work10_ReportD.md` and `Work10_ReportC.md`. **Every chunk begins
with a "Read first" step** that grounds the work in the current tex
state (line ranges drift between chunks, prose has been compressed by
prior chunks). The plotting agent's project context is still
marginal, so each chunk reads the affected section's current state
before editing — this is the load-bearing methodological difference
from ReportD.

**State at preflight (2026-05-12 morning).**
- `docs/final_report.tex` is 2435 lines.
- Body has 13 `\begin{figure}` blocks + 3 body tables (`tab:multi_axis`,
  `tab:correlation_subset`, `tab:ablation_walks`, `tab:rho_diff_spread`,
  `tab:cross_asset_tsla` — actually 5 body tables).
- Appendix already populated by ReportC + ReportD: `app:g1_matrix`,
  `app:metric_correlations`, `app:pca_coverage_full`, `app:ablation_full`,
  `app:cross_asset_full`, `app:checkpoints`, `app:deep_diagnostic`,
  `app:copula_impl`, `app:copula_mechanism` (has `tab:v8_z_collapse` +
  `fig:z_space_collapse`), `app:agents_impl` (has `tab:agent_truth`),
  `app:lstm_details`, `app:traces`, `app:vol_vol`, `app:phaseA`,
  `app:reproducer`.
- `\graphicspath` already includes `{../final_figs_tables/}` and
  `{../results/report_figures/}` — no path changes needed.
- ReportD's D4 (zero-inflation/promise/failure → app:copula_mechanism)
  and D5 (non-responsiveness disclaimer compression) have been applied.
  §5.1 Agent Design has **not** been compressed — still has the
  4-bullet itemize at lines 1503–1521 plus the fill-simulator paragraph.

---

## 1. Scope summary by tier

### Tier 1 — Concentrated moves to appendix (~7.5 pages)

| Chunk | What moves | What stays in body | Saves |
|---|---|---|---|
| E1 | §4.7 preamble bridge mechanism residue (89% zero-mass mechanism, v8/v8_b kurtosis numbers) | Two-sentence bridge per user spec | ~0.3p |
| E2 | §5.1 entire (AS equations + 4 agent bullets + fill sim) + §5.2 (non-responsiveness para already compressed but still its own subsection) | One paragraph at §5 preamble | ~1.5p |
| E3 | `fig:g1_heatmap` (per-regime G1 numerical heatmap) + the three findings paragraphs that read it | Three sentences anchored on v2/v5/v9 base-regime kurtosis | ~1.0p |
| E4 | `fig:spread_cond` (spread-conditional 3-panel) + surrounding paragraph | One sentence | ~0.5p |
| E5 | `fig:vol_vol_kde` (vol-vol KDE) + surrounding paragraph | Append "(Appendix, Figure X)" to existing in-prose vol-vol sentence | ~0.5p |
| E6 | `fig:intraday_lob_volume` + intraday paragraph | Two sentences in §7.3 Limitations | ~0.5p |
| E7 | §6.3 SPY subsection (chained-training narrative) | One sentence in §6.4 | ~0.5p |
| E8 | `fig:ablation_heatmap` (visual companion; the TABLE `tab:ablation_walks` stays) | One sentence at end of §4.6 | ~0.5p |

### Tier 2 — Caption compression (~2.0 pages)

| Chunk | Scope |
|---|---|
| E9 | Compress all remaining body-figure captions to "what's shown + one-sentence takeaway." Interpretation in prose. Apply to all ~9 body figures uniformly. |

### Tier 3 — Distributed prose compression (~2.5 pages)

| Chunk | Location | Current → target | Saves |
|---|---|---|---|
| E10 | §4.0 preamble | Two paragraphs recapping three axes + "pattern that emerges" → one sentence + forward-ref to §§4.1–4.7 | ~0.3p |
| E11 | §4.3 (Predictive Transferability) — the v9 vs v3_e9_noclip extremes column | Full paragraph → two sentences naming the two extremes | ~0.5p |
| E12 | §4.5 (Anti-Correlation) — two "deserves emphasis" paragraphs + "honest summary" | Three paragraphs → one paragraph reading the two bold table entries; drop the summary | ~0.5p |
| E13 | §4.6 (ablation walks) — three half-page walks | Three walks → three 3-sentence blocks: direction + takeaway + table-pointer | ~0.5p |
| E14 | §2.4 Stylized Facts six-item enumeration | Six items at ~1 line each + intro para → one sentence enumeration + appendix forward-ref | ~0.3p |
| E15 | §6.4 Cross-Asset Findings — three headed paragraphs | Three paragraphs → one paragraph | ~0.3p |
| E16 | §7 Discussion subsection merges | §7.1+§7.2 → one subsection (mechanism para + use-case table); §7.3+§7.4 → one subsection (numbered list) | ~0.5p |

### Final audit

| Chunk | Scope |
|---|---|
| E17 | Cross-ref audit, bib audit, body figure count check, page count check. Verify final body figure count = 9 and final page count is in the ~25.5-page band. |

---

## 2. Chunk-by-chunk plan

### Chunk E1 — §4.7 copula bridge tightening

**Read first.**
1. `docs/final_report.tex` lines 1254–1296 (current §4.7 preamble +
   bridge paragraph; the bridge currently runs ~18 lines).
2. `docs/final_report.tex` lines 2190–2366 (`app:copula_mechanism` to
   confirm the four-step argument, the v8/v8_b/v9 z-space histograms,
   and `tab:v8_z_collapse` are already there).

**Scope.** ReportD's D4 chunk moved the zero-inflation/promise/failure
prose into `app:copula_mechanism`, but it left an ~18-line bridge
paragraph in §4.7's preamble (lines 1278–1295). That bridge still
recapitulates the mechanism in too much detail. Compress to the
user-specified two-sentence bridge.

**Target replacement** (replaces lines 1278–1295):

> "v8's copula collapsed because INTC's $\sim 89\%$ zero-return mass
> creates a CDF discontinuity that pins the diffusion model to a
> degenerate $z$-band; Appendix~\ref{app:copula_mechanism} details
> the four-step mechanism and shows the v8 / v8\_b / v9 z-space
> histograms. The headline metrics: v8 G1 $= 8/15$ with excess
> kurtosis $2.6$ vs.\ real's predicted $400$--$800$, and v8\_b
> collapses to zero-variance output."

**Edits.**
1. Delete current lines 1278–1295 (the ~18-line bridge paragraph).
2. Replace with the two-sentence bridge above.
3. Verify the §4.7 preamble (lines 1267–1276) still flows into
   §4.7.1 The Fix cleanly.

**Verification.**
- Body §4.7 preamble word count drops by ~250 words.
- `\ref{app:copula_mechanism}` resolves (target label exists at line 2191).
- `grep -c "v8\\\\_b" docs/final_report.tex` — both instances still present.

---

### Chunk E2 — §5.1 + §5.2 → one-paragraph preamble

**Read first.**
1. `docs/final_report.tex` lines 1480–1552 (the entire current §5
   preamble + §5.1 Agent Design + §5.2 The Setup is Practical).
2. `docs/final_report.tex` lines 2367–2414 (`app:agents_impl` — verify
   `tab:agent_truth` is at line 2412 and the AS-family parameter
   details have somewhere to land if they're not already in
   `app:agents_impl`).
3. Quick read: does `app:agents_impl` already contain the
   AS-reservation-price equation + fill-simulator description? If
   not, those need to be added to the appendix as part of this chunk.

**Scope.** Collapse §5.1 (Agent Design — AS-family equation + four
agent bullets + fill simulator paragraph; lines 1493–1531) **and**
§5.2 (Setup is Practical — non-responsiveness disclaimer; lines
1533–1552) into a single paragraph at the §5 preamble. Drop both
subsection headers. Move AS reservation price equation + agent
bullet details to `app:agents_impl` if not already there.

**Target replacement.** Replace the current §5.1 + §5.2 block
(lines 1493–1552) with one paragraph inserted right after the §5
preamble (current §5 preamble ends at line 1491):

> "We deployed four Avellaneda-Stoikov agents
> (A0 constant-spread, A1 inventory-skewed baseline, A2 with
> order-flow-imbalance, A3 with VPIN-gated widening; full
> specification in Appendix~\ref{app:agents_impl}) against each
> generator in a non-impact fill simulator. Because historical
> replay is non-responsive (\S\ref{sec:backtest_problem}),
> recorded markets do not react to the agent under test, and every
> PnL number we report --- on every generator, including the
> historical-replay baseline --- inherits the same blind spot. We
> therefore treat agent rankings as a consistency check across
> heterogeneous synthetic environments, not as a pseudo-truth: the
> answerable question is which patterns survive the spread."

**Edits.**
1. Delete lines 1493–1552 (the §5.1 header + agent bullets + fill
   simulator + §5.2 header + non-responsiveness paragraph).
2. Insert the one-paragraph replacement immediately after the §5
   preamble.
3. **If** `app:agents_impl` does not already have the
   AS-reservation-price equation + four-agent bullet specifications +
   fill-simulator description (verify in Read step), copy them into
   the appendix before deleting from body.
4. Audit refs: `\ref{sec:agent_design}` and `\ref{sec:ranking_mechanism}`
   are removed labels. Search `grep -n "sec:agent_design\\|sec:ranking_mechanism" docs/final_report.tex` and rewrite any refs to point at `\ref{app:agents_impl}` or `\ref{sec:agents}` as appropriate.

**Verification.**
- §5 body word count drops by ~600 words.
- `\ref{app:agents_impl}` resolves.
- Body subsection count under §5: 4 → 2 (only `sec:rdiff_decoupled`
  and `sec:agent_implications` remain).
- `app:agents_impl` contains agent specifications + table.

---

### Chunk E3 — `fig:g1_heatmap` (per-regime) → appendix

**Read first.**
1. `docs/final_report.tex` lines 758–832 (current §4.2 leading prose +
   `fig:g1_heatmap` block at 807–832).
2. Identify the three findings paragraphs that interpret the per-regime
   heatmap (likely right after line 832; read 833–880 to find them).
3. `docs/final_report.tex` lines 2012–2061 (`app:g1_matrix` — confirms
   `fig:g1_heatmap_full` is the 20-model version already in appendix;
   the per-regime version `fig_p0d_g1_per_regime.pdf` will be a sibling
   exhibit).

**Scope.** Move the per-regime G1 heatmap (currently in §4.2 body,
the v2/v5/v9 × 4-regime slice) to `app:g1_matrix`. Replace with the
user-specified three-sentence summary.

**Target replacement** (replaces the figure block + the three findings
paragraphs that read it):

> "Reading the per-regime G1 matrix
> (Appendix~\ref{app:g1_matrix}, Figure~\ref{fig:g1_heatmap_per_regime}):
> v2's base-regime kurtosis overshoots to $3919$ (other regimes
> $\le 300$), v5's base-regime kurtosis overshoots to $1696$, and v9
> stays in the $57$--$202$ band across all regimes --- the most
> regime-consistent of the three protagonists."

**Edits.**
1. Delete the `fig:g1_heatmap` figure block (lines 807–832 + check
   adjacent prose for interpretation paragraphs).
2. Delete the three findings paragraphs that interpret the heatmap.
3. Insert the three-sentence replacement at the corresponding §4.2
   location.
4. Insert a new `\begin{figure}` block in `app:g1_matrix` after
   `fig:g1_heatmap_full` (lines 2024–2032), labelled
   `fig:g1_heatmap_per_regime`, with `fig_p0d_g1_per_regime.pdf` and
   the original caption from §4.2.

**Verification.**
- §4.2 body figure count drops by 1.
- `app:g1_matrix` figure count grows by 1.
- `\ref{fig:g1_heatmap}` callsites in body: rewrite to
  `\ref{fig:g1_heatmap_per_regime}` (the appendix version).
- `grep -n "fig:g1_heatmap\\b" docs/final_report.tex` — should resolve
  cleanly.

---

### Chunk E4 — `fig:spread_cond` → appendix

**Read first.**
1. `docs/final_report.tex` lines 854–880 (the `fig:spread_cond` block
   + the surrounding interpretation paragraph).

**Scope.** Move the spread-conditional 3-panel figure to a new
appendix slot (`app:microstructure_supplementary` or thread into an
existing stub — `app:deep_diagnostic` is the closest fit). Replace
body figure + paragraph with one sentence per user spec.

**Target replacement.**

> "All three protagonists widen the spread $50$--$87\%$ of the time
> (vs.\ real $4.18\%$); when their spread widens, the book is
> \emph{thinner} and returns are smaller than real's --- the joint
> structure ``spread widens AND book thickens AND prices move'' is
> not reproduced (Appendix~\ref{app:deep_diagnostic},
> Figure~\ref{fig:spread_cond})."

**Edits.**
1. Delete the figure block (lines 854–872) + the interpretation
   paragraph.
2. Insert the one-sentence replacement.
3. Move the figure block into `app:deep_diagnostic` (currently empty
   at line 2183). Preserve `\label{fig:spread_cond}`.

**Verification.**
- Body figure count drops by 1.
- `app:deep_diagnostic` populated.
- `\ref{fig:spread_cond}` resolves.

---

### Chunk E5 — `fig:vol_vol_kde` → appendix

**Read first.**
1. `docs/final_report.tex` lines 915–944 (the `fig:vol_vol_kde` block
   + surrounding paragraph).
2. `app:vol_vol` at line 2424 (already an empty stub — the natural
   destination).

**Scope.** Move the vol-vol KDE figure to `app:vol_vol`. Append an
appendix forward-reference to the in-body sentence that already
describes vol-vol correlation.

**Target replacement.** Locate the in-body sentence that mentions
vol-vol correlation in the §4.2 / G1 prose (likely near where
`fig:vol_vol_kde` is currently called out). Append:

> "... (Appendix~\ref{app:vol_vol}, Figure~\ref{fig:vol_vol_kde})."

**Edits.**
1. Delete the figure block (lines 915–930) + surrounding paragraph.
2. Insert one-sentence appendix forward-reference in remaining prose.
3. Move the figure block into `app:vol_vol`. Preserve label.

**Verification.**
- Body figure count drops by 1.
- `app:vol_vol` populated.
- `\ref{fig:vol_vol_kde}` resolves.

---

### Chunk E6 — `fig:intraday_lob_volume` → appendix, mention in §7.3

**Read first.**
1. `docs/final_report.tex` lines 945–962 (the `fig:intraday_lob_volume`
   block + surrounding paragraph).
2. `docs/final_report.tex` §7.3 Limitations (starts at line 1943) —
   find the right place to insert the two-sentence summary.

**Scope.** Move the intraday LOB volume figure to the appendix.
Replace body figure + paragraph with two sentences in §7.3
Limitations.

**Target replacement** (in §7.3 Limitations, as a new caveat or
appended to an existing caveat):

> "\emph{No intraday structure in synth output.} Our generators are
> trained on regime-conditioned event windows that are not anchored
> to time-of-day, so the synthetic output does not reproduce the
> canonical intraday U-shape that real INTC exhibits in queue size
> and trade volume; see Appendix~\ref{app:intraday},
> Figure~\ref{fig:intraday_lob_volume}."

**Edits.**
1. Delete the figure block (lines 945–962) + the surrounding §4.2
   paragraph introducing it.
2. Insert the two-sentence caveat in §7.3.
3. Create a new appendix `\section{Intraday Structure}
   \label{app:intraday}` (or thread into `app:deep_diagnostic` if E4
   put content there). Move figure block. Preserve label.

**Verification.**
- Body figure count drops by 1.
- New appendix section (or expanded existing one) populated.
- `\ref{fig:intraday_lob_volume}` resolves.
- §7.3 word count grows by ~2 sentences.

---

### Chunk E7 — §6.3 SPY subsection → one sentence in §6.4

**Read first.**
1. `docs/final_report.tex` lines 1788–1853 (current §6.3 SPY subsection
   + start of §6.4 Cross-Asset Findings).
2. SPY pipeline state: check `docs/Work10_update.md` for the live
   SPY-training status (was in-flight as of 2026-05-11; status may
   have changed by 2026-05-12).

**Scope.** Compress §6.3 SPY (currently ~25 lines of chained-training
narrative) to one sentence at the opening of §6.4.

**Target replacement** (insert at the opening of §6.4):

> "SPY training (chained $4 \times 6$h jobs on the full dataset;
> implementation details in Appendix~\ref{app:reproducer}) was still
> in flight at submission; we report a four-cell rather than
> six-cell cross-asset test."

**Edits.**
1. Delete the entire §6.3 subsection (lines 1788–1810 — confirm exact
   range when reading).
2. Insert one-sentence replacement at §6.4 opening.
3. If the SPY chained-training narrative includes any reproducible
   technical detail (e.g.\ Hydra override
   `every_n_train_steps=10000`), move that detail to
   `app:reproducer` (line 2433) so it survives the cut.

**Verification.**
- §6 body subsection count: 4 → 3.
- `app:reproducer` populated if technical detail moved.
- §6.4 opening flows from the SPY mention into the existing
  cross-asset findings.

---

### Chunk E8 — `fig:ablation_heatmap` → appendix (table stays)

**Read first.**
1. `docs/final_report.tex` lines 1116–1253 (current §4.6 ablation
   subsection — confirm `tab:ablation_walks` at line 1164 stays and
   `fig:ablation_heatmap` at line 1251 is the one to move).
2. `app:ablation_full` at line 2163 (currently an empty stub — natural
   destination).

**Scope.** Move the visual heatmap to `app:ablation_full`. The
6-row table extract `tab:ablation_walks` stays in body. Replace
figure with one sentence at end of §4.6.

**Target replacement** (at end of §4.6, after `tab:ablation_walks`):

> "The full $6 \times 6$ heatmap visualization is in
> Appendix~\ref{app:ablation_full}, Figure~\ref{fig:ablation_heatmap}."

**Edits.**
1. Delete the figure block (lines 1235–1251).
2. Insert one-sentence forward-ref.
3. Move the figure block into `app:ablation_full`. Preserve label.

**Verification.**
- Body figure count drops by 1.
- `app:ablation_full` populated.
- `\ref{fig:ablation_heatmap}` resolves.

---

### Chunk E9 — Tier 2: caption compression pass across body figures

**Read first.**
1. After E1–E8 are applied, re-run `grep -n "\\\\begin{figure}"
   docs/final_report.tex` to enumerate the surviving body figures.
   Expected list (~9 figures):
   - `fig:real_stylized_facts` (§2.4)
   - `fig:regime_tsne` (§3.2.2)
   - `fig:hero_three_axis` (§4.0)
   - `fig:rank_bump` (§4.1)
   - `fig:event_tape` (§4.1)
   - `fig:return_dist` (§4.2)
   - `fig:trade_sign_acf` (§4.2)
   - `fig:event_transitions` (§4.7.1)
   - `fig:book_resilience` (§4.7)
   - `fig:cross_asset_bars` (§6.2)
2. Read each figure's current caption end-to-end.

**Scope.** Compress every body-figure caption to "what's shown +
one-sentence takeaway." All interpretation moves to the prose that
introduces the figure (or stays where it already is if prose carries
it). No caption should exceed 4 lines.

**Target caption examples** (from user spec):

- **`fig:event_tape` (15 lines → 4):**
  > "500 consecutive events per source. Four columns (real, v2, v5,
  > v9) $\times$ four rows (mid-price, queue depth, spread, trade
  > markers). Trade rates annotated top-right: real $4.0\%$, v2
  > $70.8\%$, v5 $54.0\%$, v9 $8.8\%$. v9 alone reproduces real's
  > staircase + sparse-trade structure."
- **`fig:book_resilience` (12 lines → 3):**
  > "Queue size on the depleted side around trade events, normalized
  > to pre-trade median. Real drops to $\sim 0.88$ at $t = 0$; v9
  > tracks real; v2 / v5 stay flat (trade-saturated environment
  > averages the signal away)."
- **`fig:hero_three_axis` (10 lines → 4):**
  > "Twenty INTC checkpoints in the (G1, $1/\text{predictive}$, PCA)
  > space. Point size = PCA coverage. Color = architectural family.
  > No point is simultaneously rightmost, topmost, and largest."

Apply uniformly: every caption gets one shown-content sentence + at
most one takeaway sentence. Interpretation that previously lived in
the caption either (a) gets moved to the adjacent prose paragraph
introducing the figure, or (b) gets dropped if the prose already
covers it.

**Edits.**
1. For each of the ~9 body figures, read the current caption, draft
   the compressed version, apply the edit.
2. For each caption that previously carried interpretation, check the
   adjacent prose; if interpretation is missing from prose, add a
   1-sentence prose mention before deleting from caption.

**Verification.**
- `grep -c "\\\\caption" docs/final_report.tex` constant (count
  unchanged; only content shrinks).
- For each body figure, count caption lines: should be $\le 4$ each.
- Body word count drops by ~600–800 words.

---

### Chunk E10 — §4.0 preamble compression

**Read first.**
1. `docs/final_report.tex` lines 588–634 (the current §4 / `sec:results`
   preamble plus the lead-in to `tab/tab_1_3_multi_axis`).

**Scope.** §4.0 currently has (i) a paragraph recapping the three axes
defined in §3.4, and (ii) a paragraph on "the pattern that emerges."
Both are redundant with §3.4 + §4.1's hero figure caption.

**Target replacement.** Replace the two paragraphs with one sentence:

> "Table~\ref{tab:multi_axis} and Figure~\ref{fig:hero_three_axis}
> show all twenty checkpoints on the three axes defined in
> \S\ref{sec:eval_framework}. No architecture wins all three.
> \S\S\ref{sec:res_g1}--\ref{sec:copula_story} unpack why."

**Edits.**
1. Replace lines ~590–632 (verify exact range at read time) with the
   one-sentence preamble.

**Verification.**
- §4.0 word count drops by ~250 words.
- `\ref{tab:multi_axis}`, `\ref{fig:hero_three_axis}`,
  `\ref{sec:eval_framework}`, `\ref{sec:res_g1}`,
  `\ref{sec:copula_story}` all resolve.

---

### Chunk E11 — §4.3 v9-vs-v3_e9_noclip compression

**Read first.**
1. `docs/final_report.tex` lines 965–1025 (`sec:res_predictive`).
2. Locate the column-or-paragraph block explaining the v9 vs
   v3_e9_noclip extremes pair.

**Scope.** Compress the v9-vs-v3_e9_noclip explanation paragraph to
the two-sentence summary per user spec.

**Target replacement.**

> "v9 (best marginals, $106\times$ replay) and v3\_e9\_noclip (worst
> marginals, $4\times$) are the extremes: the model with $-0.43$
> excess kurtosis and $43\times$ real's std produces the best
> predictive score in the study. Scale matching is neither
> necessary nor sufficient for transferability."

**Edits.**
1. Locate the existing paragraph (lines TBD — read at chunk time).
2. Replace with the two-sentence version.

**Verification.**
- §4.3 word count drops by ~250 words.
- The v9 and v3_e9_noclip numerical claims still appear exactly once
  in the body.

---

### Chunk E12 — §4.5 Anti-Correlation prose compression

**Read first.**
1. `docs/final_report.tex` lines 1059–1115 (`sec:res_anticorrelation`).
2. Identify the two `\paragraph{...}`-style "deserves emphasis" blocks
   and the closing "honest summary" paragraph.

**Scope.** Three paragraphs → one paragraph reading the two bold
table entries. Drop the closing "honest summary" (already covered by
§4.0 preamble after E10).

**Target replacement.** One paragraph reading the two emphasis entries:

> "Two entries deserve emphasis. The predictive-vs-PCA correlation of
> $+0.57$ is sign-counterintuitive --- a generator that covers more
> of real's PC1--PC2 region should produce \emph{better} downstream
> training data, not worse. The mechanism is family membership: the
> copula family has wide PCA hulls AND catastrophic predictive
> scores; the noclip family has $0\%$ coverage AND the best
> predictive scores; the protagonist trio sits in between. The
> trade-fraction-vs-predictive $-0.51$ is the same mechanism on the
> trade-fraction axis. The structural reading appears in
> \S\ref{sec:copula_lesson}." Then drop the "honest summary."

**Edits.**
1. Delete the three current paragraphs.
2. Insert the one-paragraph replacement.

**Verification.**
- §4.5 word count drops by ~250–300 words.
- `\ref{sec:copula_lesson}` resolves.
- `tab:correlation_subset` still in §4.5.

---

### Chunk E13 — §4.6 ablation walks compression

**Read first.**
1. `docs/final_report.tex` lines 1116–1253 (`sec:res_ablation`).
2. Locate the three "walks": v2 → v5, v2 → v9, noclip walk.

**Scope.** Keep all three walks but compress each to 3 sentences:
direction + takeaway + table-pointer. The table
`tab:ablation_walks` carries the numbers.

**Target template per walk.**
> "**v2 $\to$ v5 (FiLM $\to$ AdaLN-Zero):** axis movement. One-sentence
> takeaway anchored on the largest single change in the row of
> Table~\ref{tab:ablation_walks}. One-sentence pointer to the
> mechanism subsection where this matters."

**Edits.**
1. Identify each walk's prose block.
2. Compress each to 3 sentences using the template.
3. Preserve the `\paragraph{...}` headers if present.

**Verification.**
- §4.6 word count drops by ~250 words.
- All three walks still present + table still in body.

---

### Chunk E14 — §2.4 Stylized Facts enumeration compression

**Read first.**
1. `docs/final_report.tex` lines 275–338 (the `sec:stylized_facts`
   subsection + the lead-in to `fig:real_stylized_facts`).

**Scope.** Six-item enumeration of Cont (2001) stylized facts → one
sentence enumeration + appendix forward-ref for the per-fact INTC
numbers. The dual-window footnote about 89% vs 98.78% zero-mass moves
to a clearer reference in `app:deep_diagnostic` or `app:copula_mechanism`.

**Target replacement.**

> "We evaluate against six Cont (2001) stylized facts --- heavy
> tails (kurtosis $\approx 755$ on INTC), volatility clustering,
> absence of return autocorrelation, trade-sign persistence
> (lag-1 ACF $\approx 0.67$), volume-volatility correlation, and
> the leverage effect --- operationalized into sixteen G1 checks
> (Appendix~\ref{app:g1_matrix} for the per-check formal
> definitions; Figure~\ref{fig:real_stylized_facts} shows the
> empirical realization on real INTC)."

**Edits.**
1. Replace the six-item enumeration with the one-sentence version.
2. Move dual-window footnote to appendix.
3. Keep `fig:real_stylized_facts` and its 1-paragraph introduction.

**Verification.**
- §2.4 word count drops by ~150 words.
- `\ref{app:g1_matrix}` and `\ref{fig:real_stylized_facts}` resolve.

---

### Chunk E15 — §6.4 Cross-Asset Findings paragraph compression

**Read first.**
1. `docs/final_report.tex` lines 1811–1853 (`sec:cross_findings`).

**Scope.** Three headed paragraphs → one paragraph per user spec.

**Target replacement.**

> "On TSLA, v2 dominates v5 on all three axes --- no Pareto
> trade-off. The three-axis tension is at least partially
> INTC-specific. However, TSLA v5's PCA still collapses to $18.9\%$,
> confirming that the $v$-prediction mechanism operates on smoother
> data too --- it just doesn't produce a coverage niche on TSLA."

**Edits.**
1. Delete the three current paragraphs.
2. Insert the one-paragraph replacement.

**Verification.**
- §6.4 word count drops by ~150 words.

---

### Chunk E16 — §7 Discussion merge subsections

**Read first.**
1. `docs/final_report.tex` lines 1853–2011 (entire §7 Discussion).
2. Identify the four subsections (`sec:disc_structural`,
   `sec:disc_practical`, `sec:limitations`, `sec:future_work`).

**Scope.** Two structural merges:
- §7.1 (Three-Axis Tension as Structural Result) + §7.2 (Practical
  Implications) → one subsection. §7.1 re-derives the mechanism;
  §7.2 currently has five paragraphs; the merge replaces those five
  paragraphs with a single mechanism paragraph + a use-case TABLE.
- §7.3 (Limitations) + §7.4 (Future Work) → one subsection as a
  numbered list. §7.3 has six items; §7.4 has four items.

**Target.**
1. New §7.1 (or renamed): "Structural Result and Practical
   Implications" — mechanism paragraph + use-case table
   (`tab:disc_use_cases` if needed). Caveat about E6's intraday
   addition (which lands in §7.3 → now part of merged §7.2).
2. New §7.2 (or renamed): "Limitations and Future Work" — numbered
   list combining the ten items, with one-line entries.

**Edits.**
1. Merge §7.1 + §7.2 prose, drop redundancies, introduce table if
   useful.
2. Merge §7.3 + §7.4 as numbered list.
3. Update any cross-refs to the four removed labels (`sec:disc_structural`,
   `sec:disc_practical`, `sec:limitations`, `sec:future_work`).

**Verification.**
- §7 subsection count: 4 → 2.
- §7 word count drops by ~250 words.
- All cross-refs to the four removed labels rewritten.

---

### Chunk E17 — Final audit + cross-ref + page count

**Read first.**
1. Re-run the body figure / table enumeration: `grep -n "\\\\begin{figure}\\\\|\\\\begin{table}\\\\|\\\\label{(fig|tab|sec|app):" docs/final_report.tex`.
2. Run `wc -w docs/final_report.tex` for word count.
3. If user has LaTeX locally, render to PDF and confirm page count is
   in the ~25.5-page band; otherwise rely on word count proxy
   (target body ~13,000–14,000 words after all cuts).

**Scope.** Cross-ref audit + bib audit + final figure/page count
check.

**Edits.**
1. Extract all `\ref{...}` and `\label{...}` from the doc; diff.
   Confirm zero orphans.
2. Extract all `\cite{...}` keys; confirm each is in
   `docs/references.bib`. Any new citations added by Tier 3 chunks
   (probably none) → audit.
3. Confirm final body figure count = 9 (or 10 if `fig:event_transitions`
   counts separately from `fig:book_resilience`).
4. Confirm appendix grew by 3–4 new figure / section additions:
   - `fig:g1_heatmap_per_regime` in `app:g1_matrix` (E3)
   - `fig:spread_cond` in `app:deep_diagnostic` (E4)
   - `fig:vol_vol_kde` in `app:vol_vol` (E5)
   - `fig:intraday_lob_volume` in `app:intraday` or `app:deep_diagnostic` (E6)
   - `fig:ablation_heatmap` in `app:ablation_full` (E8)

**Verification.**
- Zero orphan `\ref{}`.
- Zero unresolved `\cite{}`.
- Body figure count == 9 (or 10).
- Word count proxy in target band.
- LaTeX compiles end-to-end (if user runs the build).

---

## 3. Sectional impact summary (post-E)

| § | After Work10_ReportD | After Work10_ReportE |
|---|---|---|
| §2.4 (Stylized Facts) | six-item enumeration + intro para + figure | one-sentence enumeration + figure |
| §4.0 (preamble) | two paragraphs recapping axes | one sentence |
| §4.2 (Distributional Realism) | 5 body figures (return dist, g1_heatmap, spread_cond, trade_sign_acf, vol_vol_kde, intraday_lob_volume) | 2 body figures (return dist, trade_sign_acf) |
| §4.3 (Predictive Transferability) | v9 vs v3_e9_noclip column | two-sentence summary |
| §4.5 (Anti-Correlation) | table + 2 emphasis paras + honest summary | table + 1 emphasis para |
| §4.6 (Ablation) | table + 3 long walks + heatmap figure | table + 3 short walks + appendix forward-ref |
| §4.7 (Copula Story) | preamble + 18-line bridge + 2 subsubsections | preamble + 2-sentence bridge + 2 subsubsections |
| §5.1 (Agent Design) | full subsubsection + 4 bullets + fill sim | **merged into §5 preamble paragraph** |
| §5.2 (Setup is Practical) | non-responsiveness disclaimer | **merged into §5 preamble paragraph** |
| §5 preamble | terse | one paragraph absorbing §5.1+§5.2 |
| §6.3 (SPY) | full subsection on chained training | **one sentence in §6.4** |
| §6.4 (Cross-Asset Findings) | three headed paragraphs | one paragraph |
| §7.1+§7.2 (Discussion: structural + practical) | two separate subsections, 5 paras in §7.2 | merged subsection with mechanism para + use-case table |
| §7.3+§7.4 (Limitations + Future Work) | two separate subsections | merged subsection as numbered list |
| `app:g1_matrix` | `fig:g1_heatmap_full` | + `fig:g1_heatmap_per_regime` |
| `app:deep_diagnostic` | empty stub | + `fig:spread_cond` |
| `app:vol_vol` | empty stub | + `fig:vol_vol_kde` |
| `app:intraday` (NEW) or `app:deep_diagnostic` | n/a | + `fig:intraday_lob_volume` |
| `app:ablation_full` | empty stub | + `fig:ablation_heatmap` |
| `app:agents_impl` | populated by D5 | + AS-reservation-price equation + 4-agent bullet specifications + fill-simulator description (if not already there from D5) |
| `app:reproducer` | empty stub | + SPY chained-training reproducible details (E7) |

---

## 4. Effort + ordering

| Chunk | Effort | Dependency |
|---|---|---|
| E1 — §4.7 bridge tightening | 15 min | none |
| E2 — §5.1+§5.2 → preamble | 35 min | none (heaviest single move) |
| E3 — `fig:g1_heatmap` → appendix | 25 min | none |
| E4 — `fig:spread_cond` → appendix | 15 min | none |
| E5 — `fig:vol_vol_kde` → appendix | 15 min | none |
| E6 — `fig:intraday_lob_volume` → appendix + §7.3 mention | 20 min | none |
| E7 — §6.3 SPY → one sentence | 15 min | none |
| E8 — `fig:ablation_heatmap` → appendix | 15 min | none |
| E9 — Tier 2 caption compression | 45 min | E1–E8 (only touches surviving body figures) |
| E10 — §4.0 preamble | 10 min | none |
| E11 — §4.3 extremes compression | 15 min | none |
| E12 — §4.5 anti-correlation prose | 15 min | none |
| E13 — §4.6 ablation walks | 25 min | none |
| E14 — §2.4 stylized facts | 15 min | none |
| E15 — §6.4 cross-asset paragraph | 15 min | E7 |
| E16 — §7 merges | 35 min | E6 (intraday caveat lands in merged §7.2) |
| E17 — final audit | 30 min | all prior |

**Total estimated effort:** ~5.5 hours of focused conversation, in
chunks of 10–45 minutes each.

**Suggested execution grouping** (each group runs sequentially within
itself; between groups the user reviews):

1. **Tier 1 first** (highest page savings, mechanically simple
   figure-moves; review between each so we catch unintended ref
   breaks): E1, E2, E3, E4, E5, E6, E7, E8.
2. **Tier 2 next** (depends on Tier 1 being done so we only touch
   surviving body figures): E9.
3. **Tier 3 prose** (independent of Tier 1/2, can be done in any
   order within group): E10, E11, E12, E13, E14, E15, E16.
4. **Final audit:** E17.

---

## 5. The "Read first" pattern — why this plan differs from ReportD

ReportD's chunks assumed the agent already knew the current section
state. ReportE's chunks **must not** assume that — line ranges drift
between chunks as prose compresses, and the plotting agent's project
context is still building. Every chunk starts with a concrete Read
step (specific line range, specific appendix label) so the agent
re-grounds before editing. The pattern:

1. Read the affected section's current state.
2. (If the chunk moves content to an appendix) Read the destination
   appendix's current state to confirm the slot is empty or
   appendable.
3. (Optional) Read one CSV or one results file referenced by the
   compressed prose to confirm the numerical claims (e.g.\ "v2
   base-regime kurtosis = 3919") match the underlying data.
4. Edit.
5. Verify (grep / wc / cross-ref).

Step 3 is the safety net for the "memory drifts but the prose makes
claims" failure mode — every numerical statement in the compressed
replacement prose can be re-validated against the CSV before going in.

---

## 6. What I will not touch

- §1, §2.1–§2.3 (already lean; no figure changes).
- §3.1 DDPM, §3.3 Checkpoints (already minimal).
- §3.2.2 figure (`fig:regime_tsne`) — stays in body.
- §3.4 evaluation framework (already short).
- §4.1 protagonist figures (`fig:hero_three_axis`, `fig:rank_bump`,
  `fig:event_tape`) — hero / load-bearing.
- §4.4 PCA Coverage (already compressed in D2).
- §4.7.1 The Fix and §4.7.2 The Deeper Lesson (the punchlines that
  justify §4.7's existence). Their figures `fig:event_transitions`
  and `fig:book_resilience` stay in body.
- §5.3 (Different Generators), §5.4 (Stylized Facts We Can Trust) —
  no compression target after E2 collapses §5.1+§5.2.
- §6.1 (Why Cross-Asset), §6.2 (TSLA + `fig:cross_asset_bars`) — stay.
- Abstract, AI Disclosure block.
- `references.bib` (no new citations needed).
- Any `\label{}` that already exists (preserve; only move the figure
  / table content, keep the label so refs keep working).

---

## 7. Verification protocol per chunk

After each chunk:

1. `wc -w docs/final_report.tex` — confirm body word count dropped by
   the expected magnitude.
2. `grep -c "\\\\begin{figure}" docs/final_report.tex` — total figure
   count moves between body and appendix; net count should be
   constant (every body removal is paired with an appendix
   insertion).
3. Extract `\ref{...}` and `\label{...}`; diff to confirm zero
   orphans.
4. Read the affected section's prose top-to-bottom; confirm every
   figure / table mention resolves to something that exists.
5. User reviews diff and approves before moving on.

---

**Ready to execute.** First suggested chunk: **E1 (§4.7 bridge
tightening)** — small, isolated, lets us validate the Read-first
methodology before tackling the heavier E2 / E3 chunks. Alternative
opener: **E10 (§4.0 preamble)** — even smaller, also good for
methodology validation. Let me know which to start with, or whether
to take a batch.
