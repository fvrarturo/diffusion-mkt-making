# Work10 — Report C: Exhaustive Tex Rewire Plan

**Goal.** Bring the body of `docs/final_report.tex` into alignment with
the canonical 17-figure suite in `final_figs_tables/` (per
`Work10_newfigureorga.md`). Strip every figure + table block from the
body, then rebuild section by section with the new figures in place
and the prose adjusted to match what the figures actually show.

**Decision rule for narrative direction.** Where the new figures imply
a different reading than the current prose, the figures win. The user
explicitly prefers the figure-driven narrative — **the v9 microstructure
case strengthens** (book resilience + event transitions = positive
evidence v9 captures dynamics no aggregate metric measures), the
**v8/v8_b distinction sharpens** (v8_b is the perfect single-point
collapse, v8 is partial-recovery), the **synth-fails-on-intraday-structure
finding moves up to §5.2** (no longer buried in Limitations), and
**all main-body comparisons reduce to v2/v5/v9** (drop "all 20" framing
in §5.5; full-population content moves to appendix).

**Working method.** Each chunk below is a single edit-and-verify pass.
Each chunk has: scope, current state, target state, the specific
edits, and a verification step. We will execute one chunk at a time
and the user reviews between chunks.

---

## 0. Preflight: audit current state

Current `docs/final_report.tex` body has these figure / table blocks
(line numbers from current state):

| Line | Block | Currently shows | Disposition |
|---|---|---|---|
| 373–394 | `fig:real_stylized_facts` | §2.4 — fig_a_real_stylized_facts | **KEEP** (already canonical) |
| 680 | `\input{tab_1_3_multi_axis}` | §5.0 hero table | **KEEP** |
| 681–698 | `fig:hero_three_axis` | §5.0 hero scatter — fig_1_1 | **KEEP** |
| 747–760 | `fig:rank_bump` | §5.1 — fig_10_1 | **KEEP** (re-rendered) |
| 785–800 | `fig:g1_heatmap` | §5.2 — fig_2_1_g1_numerical_heatmap (20-model) | **REPLACE** with `fig_p0d_g1_per_regime` |
| 825–837 | `fig:trade_fraction` | §5.2 — fig_2_4 (20-model bar) | **REMOVE** (→ App.) |
| 852–861 | `fig:return_tail` | §5.2 — fig_2_3 (3-threshold tail) | **REMOVE** (→ App.) |
| 878–894 | (placeholder using fig_2_3_return_tail) | §5.3 duplicate | **REMOVE** outright |
| 910–927 | `fig:pred_vs_scale` | §5.3 — fig_3_3_predictive_vs_scale | **REMOVE** (→ App., user-flagged "could-be-table") |
| 1105–1118 | `fig:pairwise_anticorrelation` | §5.5 — fig_1_2 | **REMOVE** (→ App., user-flagged) |
| 2020–2037 | `fig:intraday_lob_volume` | §8.3 — fig_intraday_lob_volume | **MOVE** to §5.2 |

Currently no figure blocks in §3 (Methods), §5.4 (PCA — has tab inline
only), §5.6 (Ablation — table inline), §5.7 (Copula — tables inline),
§6 (Agents — tables inline), §7 (Cross-Asset — table inline). All of
these are getting figures added.

**`\graphicspath` is already set** to `{{../final_figs_tables/}{../results/report_figures/}{../results/}}`,
so any `\includegraphics{<basename>.pdf}` resolves to the canonical
dir first. No path changes needed.

---

## 1. Chunk-by-chunk plan

Sixteen chunks total. Each chunk should fit in 1-3 Edit operations
plus a verification grep. Estimated effort: 30 minutes of conversation
per chunk, longer for §5.2 and §5.7 which carry the most content
change.

### Chunk 1 — The Wipe

**Scope.** Remove all `\begin{figure}...\end{figure}` blocks from the
body except `fig:real_stylized_facts` (§2.4) and `fig:hero_three_axis`
(§5.0) which keep their canonical placement. Also remove
`\input{...tab_1_3_multi_axis}` from §5.0 — it stays, just not yet.

**Wait, alternative:** instead of wiping then rebuilding (which leaves
the doc in a broken intermediate state), we walk through section by
section. Each chunk does its own remove-and-replace atomically.

**Decision: skip Chunk 1 (no global wipe).** Each chunk handles its
own remove + insert. Cleaner intermediate states, easier to verify,
no risk of accidentally losing a figure caption we wanted to recycle.

### Chunk 2 — §3.2.2 Conditioning: add `fig_p2_4_regime_tsne`

**Scope.** §3.2.2 currently has prose only (no figure). Add the
regime-embedding t-SNE figure at the end of the subsection,
immediately after the FiLM modulation-collapse sentence.

**Current state.** §3.2.2 ends at line 481 with the sentence "All
twenty checkpoints reported in \S\ref{sec:results} use AdaLN-Zero."

**Target state.** Same prose, plus a `\begin{figure}` block + 1-2
sentence prose adjustment to introduce the figure.

**Edits.**
1. Insert before the closing of §3.2.2:
   - One sentence: "Figure~\ref{fig:regime_tsne} shows the four
     regime embeddings each protagonist learned, projected to 2D
     by t-SNE. The four regimes are spatially separated in all
     three panels — i.e., AdaLN-Zero is using the conditioning
     bandwidth to differentiate regimes rather than collapsing
     them to a single embedding."
2. Insert `\begin{figure}` block with:
   - `\includegraphics[width=\linewidth]{fig_p2_4_regime_tsne.pdf}`
   - Caption acknowledging the 4-point limitation.
   - `\label{fig:regime_tsne}`

**Verification.** `grep "fig:regime_tsne" docs/final_report.tex` returns 2 lines (figure label + body ref).

---

### Chunk 3 — §5.0 + §5.1: Hero + Rank-Bump + Event-Tape Walkthrough

**Scope.** §5.0 hero (existing, keep) + §5.1 (existing rank-bump,
keep) + add `fig_event_tape_walkthrough_500` after the rank-bump.

**Current state.** §5.0 hero figure + table are wired. §5.1 has
fig_10_1_rank_bump wired. Both correct.

**Target state.** Same, plus a new figure (event-tape walkthrough)
inserted into §5.1 after the rank-bump prose, with a 1-paragraph
introduction.

**Edits.**
1. After the rank-bump figure block in §5.1, insert one paragraph:
   - "Before unpacking the per-axis numbers, Figure~\ref{fig:event_tape}
     shows what 500 consecutive events look like for each protagonist
     against a representative real INTC tape. Three event-level
     properties are immediately legible without any aggregate
     statistics: real's mid-price is a staircase (long flat
     stretches punctuated by sudden jumps), real's spread is pinned
     at exactly one tick, and real's trades are sparse (4% of
     events). v2 and v5 produce continuously-drifting mid-prices,
     chaotic spreads, and trades on 54-71% of events; v9 alone
     reproduces the staircase + sparse-trade structure."
2. Insert `\begin{figure}` with `fig_event_tape_walkthrough_500.pdf`
   and a multi-line caption walking the reader through the four
   rows (mid-price, queue depth, spread, trade markers).

**Verification.** `\ref{fig:event_tape}` resolves. Figure renders at
~3-page mark.

---

### Chunk 4 — §5.2 (a): swap G1 heatmap, drop trade-fraction + return-tail figures

**Scope.** §5.2 currently leads with a 20-model G1 heatmap +
trade-fraction bar + return-tail figure. Replace heatmap with the
v2/v5/v9-only per-regime version; drop trade-fraction + return-tail
figures (move references to App.).

**Current state (lines 762–861):** three figures, prose narrates
"all 20" framing.

**Target state.** One figure (per-regime G1 heatmap, v2/v5/v9 ×
4 regimes + real). Prose re-anchored on protagonists.

**Edits.**
1. **Remove** the existing `fig:g1_heatmap` block (lines 785–800)
   and replace with a new block citing `fig_p0d_g1_per_regime.pdf`
   under the same `\label{fig:g1_heatmap}` (preserves cross-refs).
   Caption disambiguates the v9 "n/a" cells (validator-level NaN)
   and points to App. A for the full 20-model matrix.
2. **Remove** the `fig:trade_fraction` block (lines 825–837).
   Replace the in-prose reference to it with: "the trade-fraction
   row in Figure~\ref{fig:g1_heatmap} (real $\approx 0.048$; v2
   $\approx 0.55$; v5 $\approx 0.55$; v9 $\approx 0.05$)" — keeps
   the finding visible in prose without a separate figure.
3. **Remove** the `fig:return_tail` block (lines 852–861). Replace
   the in-prose reference with a brief mention that
   Figure~\ref{fig:return_dist} (added in chunk 4b) carries the
   tail-shape comparison.
4. **Rewrite the §5.2 prose** to drop "all 20 checkpoints" framing
   and use v2/v5/v9 as the canonical comparison units. Keep the
   finding about the noclip family in one sentence (since it's the
   most informative third-corner reference) but the rest is
   protagonist-only.

**Verification.** Three `\begin{figure}` blocks reduce to one in §5.2.
`fig:g1_heatmap`, `fig:trade_fraction`, `fig:return_tail` references
that previously pointed at separate figures now either point at
`fig:g1_heatmap` (the kept one) or are deleted. No broken `\ref{}`.

---

### Chunk 5 — §5.2 (b): add lead figure (return distribution) + intraday LOB volume

**Scope.** §5.2 needs the new lead figure (`fig_return_distribution_full`,
TRADES Fig 3 style) and `fig_intraday_lob_volume` (moved up from §8.3).

**Edits.**
1. Insert `fig_return_distribution_full` as the **first** figure in
   §5.2, immediately after the opening paragraph that defines the
   G1 axis. Caption notes:
   - Two panels: A density (log-y), B Q-Q on `|return|>0`.
   - Real kurt 755; v2 125 (closest to mode-covering bulk);
     v5 907 (heaviest tails of the protagonist trio); v9 136 (mode
     OK but tails light).
   - Q-Q conditional on non-zero returns, since the 98.78% zero-mass
     would dominate any unconditional Q-Q.
   - `\label{fig:return_dist}`
2. Insert `fig_intraday_lob_volume` near the end of §5.2 (after the
   per-regime G1 heatmap), introduced by one paragraph:
   - "One stylized fact our generators do not reproduce, regardless
     of architecture, is intraday structure. Figure~\ref{fig:intraday_lob_volume}
     shows the level-1 queue size of real INTC across the trading
     session — the canonical U-shape (queues build up toward the
     close) — alongside v2/v5/v9 outputs that are essentially flat.
     The mechanical reason is that our diffusion model is trained on
     regime-conditioned event windows that are not anchored to
     time-of-day; a synthetic tape carries no information about
     where in the session it sits. We return to the practical
     consequence in \S\ref{sec:limitations}."
3. Caption keeps `\label{fig:intraday_lob_volume}` (matches existing
   ref from §8.3).

**Verification.** §5.2 now has 4 figures total (return_dist,
g1_heatmap, intraday_lob_volume + chunks 6/7 figures). `\ref{fig:return_dist}`
and `\ref{fig:intraday_lob_volume}` resolve.

---

### Chunk 6 — §5.2 (c): add spread-conditional + trade-sign ACF + vol-vol KDE

**Scope.** Three more supporting exhibits in §5.2: `fig_p2_1_spread_conditional`,
`fig_7_3_trade_sign_acf`, `fig_7_1_vol_vol_kde`.

**Edits.**
1. **Spread-conditional figure**: insert with `\label{fig:spread_cond}`
   and a 1-paragraph introduction:
   - "When real INTC spreads widen, the book becomes thicker AND
     prices move more — a joint structural property the synth
     generators do not reproduce. Figure~\ref{fig:spread_cond} shows
     three panels: P(spread > 1 tick) (real 4.18%, synth 50–87%),
     mean bid_sz conditional on wide spread (real 3.76e3, synth
     800–2200), mean |return| conditional on wide spread (real
     1.55e-05, synth 1.5–3.0e-07). Synth widens the spread far too
     often AND fails to populate the wide-spread regime with
     realistic depth or volatility."
2. **Trade-sign ACF**: insert with `\label{fig:trade_sign_acf}` and
   one paragraph:
   - "Order-flow persistence is a Cont 2001 stylized fact that our
     generators undershoot uniformly. Figure~\ref{fig:trade_sign_acf}
     plots trade-sign autocorrelation across lags {1, 5, 10, 50, 100}.
     Real (dashed black) decays slowly with lag-1 ≈ 0.41 in our
     held-out window; v2 starts at 0.22, v9 at 0.24, v5 at 0.06.
     v9's faster decay matches the broader §5.7 pattern that copula
     transforms scramble per-event temporal structure even when they
     get marginals right."
3. **Vol-vol KDE**: insert with `\label{fig:vol_vol_kde}` and one
   paragraph:
   - "Per-tape volume-volatility correlation (Cont 2001 fact v) is
     a per-day positive number ≈ +0.34 on real INTC.
     Figure~\ref{fig:vol_vol_kde} shows the distribution of this
     statistic across 20 real days vs 400 synth tapes per
     architecture. Real is unimodal and tightly centered; v2 and v5
     emit much wider distributions that span -1.5 to +1.5 (i.e.\
     individual synth tapes can show anti-correlated volume and
     volatility, which real never does). v9 is currently excluded
     from this figure — its frac\_zero $\approx 0.99$ means most
     tapes have zero trades, making per-tape Pearson undefined."

**Verification.** §5.2 closes with 6 figures total. All `\label{}` ↔
`\ref{}` resolve. No leftover references to dropped figures.

---

### Chunk 7 — §5.2 (d): final prose pass + remove §5.3 duplicate

**Scope.** Re-read §5.2 end-to-end with the 6 figures in place. Cut
sentences that referenced removed figures by name. Tighten the
narrative around v2/v5/v9.

**Edits.**
1. Walk §5.2 prose top-to-bottom; remove every "all twenty
   checkpoints" or "the bottom of the table is the noclip family"
   reference that doesn't anchor to a figure still in the body.
2. Move any genuinely-load-bearing 20-model finding (e.g., kurtosis
   spectrum, scale-exploded family) to §5.6 (Ablation) or to a
   forward-pointer for App. A.
3. **Remove the §5.3 duplicate** of `fig:return_tail` (lines 878–894
   in the current file). The §5.3 caption text describes a
   predictive-score bar chart, but the figure is a duplicate of
   §5.2's tail-probability plot. Either drop entirely or replace
   with a forward-pointer to App. (likely drop).

**Verification.** Read §5.2 prose top-to-bottom — every figure
reference resolves to a figure in the body or an explicit App.
forward-pointer. Word count for §5.2 should grow modestly (more
figures = more captions = slightly more figure-centric prose).

---

### Chunk 8 — §5.3 Predictive Transferability: drop figure entirely

**Scope.** §5.3 currently has `fig_3_3_predictive_vs_scale` (user
flagged as "could-be-table"). Drop the figure outright. Prose stays
but loses its anchor figure.

**Edits.**
1. **Remove** the `fig:pred_vs_scale` block (lines 910–927).
2. Re-read §5.3 prose; re-anchor any "as Figure X shows" or "in
   Figure X" sentences. The §5.3 narrative (predictive corner
   dominated by noclip; copula corner with 100×+ replay) is fully
   carriable in prose.
3. Replace any figure-reference sentences with the inline equivalent:
   "noclip family at $\sigma_\text{synth}/\sigma_\text{real} \approx
   40$ produces predictive scores $4$–$14\times$, scale-accurate at
   $\approx 1.0$ produces $10$–$36\times$, copula at $\approx 0.6$
   produces $28$–$210\times$".

**Verification.** No `\begin{figure}` in §5.3. `fig:pred_vs_scale`
not referenced anywhere. All §5.3 prose still parses without
forward references to figures we haven't loaded.

---

### Chunk 9 — §5.4 PCA Coverage: keep the table, no figure

**Scope.** §5.4 already carries `tab:pca_coverage_top` inline. No
figure was wired (per your earlier "this can be just a table"
decision). Confirm no changes needed.

**Edits.**
1. **Verify-only.** Read §5.4 prose and confirm the table is
   self-contained. If any "Figure X" references remain, replace
   with table references.

**Verification.** `grep "fig:" docs/final_report.tex` between §5.4
boundaries returns nothing.

---

### Chunk 10 — §5.5 Anti-Correlation: drop pairwise figure, keep table

**Scope.** §5.5 has both `tab:correlation_subset` (kept) and
`fig:pairwise_anticorrelation` (drop, user-flagged).

**Edits.**
1. **Remove** the `fig:pairwise_anticorrelation` block (lines
   1105–1118).
2. Re-read §5.5 prose; replace "Figure~\ref{fig:pairwise_anticorrelation}"
   references with "the matrix in Table~\ref{tab:correlation_subset}".

**Verification.** No `\begin{figure}` in §5.5. `fig:pairwise_anticorrelation`
not referenced.

---

### Chunk 11 — §5.6 Architectural Ablation: add `fig_5_1_ablation_heatmap`

**Scope.** §5.6 currently has `tab:ablation_walks` (the 6-row table
extract). Add the full 6-toggle × 6-metric heatmap as the visual
companion.

**Edits.**
1. Insert `fig_5_1_ablation_heatmap.pdf` block at the end of §5.6,
   after the table:
   - `\label{fig:ablation_heatmap}`
   - Caption: "Visual companion to Table~\ref{tab:ablation_walks}.
     The table extracts six rows (one per toggle); the figure shows
     the entire 6×6 matrix at once. Reading by column: ΔPred is
     dominated by the copula and noclip rows (both make predictive
     dramatically worse — copula by overshoot, noclip by undershoot
     in different directions); ΔPCA is dominated by the noclip row
     (-40%, the destructive move); Δkurt is dominated by FiLM →
     AdaLN-Zero (+1184, the strongest single conditioning effect)."
2. One sentence at the end of §5.6 prose forwarding to the figure:
   "Figure~\ref{fig:ablation_heatmap} renders the same data as the
   full per-cell heatmap and lets the reader spot the cells the
   table doesn't extract."

**Verification.** §5.6 has the table + one figure. `\ref{fig:ablation_heatmap}`
resolves.

---

### Chunk 12 — §5.7 (a): copula story preamble + §5.7.1 zero-inflation pathology

**Scope.** §5.7 narrative restructure (no figures yet at this point —
those go in chunks 13/14). Confirm the §5.7.1 prose still works
with the protagonist v2/v5/v9 framing.

**Edits.**
1. Re-read §5.7 preamble (line 1241+) and §5.7.1 (line 1265+).
   These two are largely already correct (v8 / v9 / v2_remapped
   protagonist framing). Confirm no changes needed; if the
   "88.85% zero-mass" framing collides with the current
   `fig_a_real_stylized_facts` panel A annotation (which says
   98.78% — different val window, footnoted in §2.4), make sure
   the §5.7.1 prose either uses the same convention as §2.4's
   footnote or restates the dual-window story locally.

**Verification.** §5.7.1 prose reads cleanly with the §2.4 footnote
in mind.

---

### Chunk 13 — §5.7 (b): §5.7.3 + §5.7.4 add `fig_z_space_collapse` + body-prose update

**Scope.** §5.7.3 (The Failure) and §5.7.4 (The Fix) are the heaviest
prose-update chunks. The new figure shows v8/v8_b/v9 — the body's
current "v8 collapses, v9 recovers" oversimplifies, per
`Work10_newfigureorga.md` §3.14a.

**Edits.**
1. **Insert** `fig_z_space_collapse.pdf` at the boundary between
   §5.7.3 and §5.7.4 (after §5.7.3's collapse description, before
   §5.7.4's recovery description). Spans the page width.
   `\label{fig:z_space_collapse}`. Caption explicitly:
   - Three panels: v8 (partial), v8_b (perfect collapse), v9 (mode-shifted + tail spread).
   - Trimodal structure (central spike + thin edge masses) is real
     data structure, not a viz artifact.
   - v8_b is the canonical "single-point degenerate" collapse;
     v9's "fix" relative to v8_b is the visible tail mass at ±2.6.
2. **Update §5.7.3 prose** to acknowledge:
   - v8_b is the canonical perfect-collapse exemplar (collapses to
     a single z value, std=0).
   - v8 has been re-sampled and is now a partial-recovery version
     (mode at ≈−2.55, std≈1.09, kurt≈1) — the body's earlier
     "catastrophic collapse with kurt=309.7" was based on an older
     sample no longer reproducible.
   - The mechanism explanation (zero-mass plateau + ECDF discontinuity
     → diffusion model concentrates output at a single z) still
     holds; v8_b makes it visual.
3. **Update §5.7.4 prose** to:
   - Reframe v9's contribution as **mode shift + tail spread**, not
     just "recovers spread".
   - Note that v9 and v8_b share the same mode location (z=0); v9's
     improvement over v8_b is the small but nonzero mass at the
     ±2.6 tail edges.
   - Note that v9's per-feature G1 numbers (kurtosis, trade fraction,
     etc.) recover because the mode shift positions the spike where
     the inverse-CDF needs it for the marginal to come out right.
4. **Update §5.7.4's `tab:v8_z_collapse`** if the per-feature numbers
   have shifted since the table was written. (May need a quick
   cluster check; for now, leave the table as-is and add a footnote
   noting the reproducibility caveat.)

**Verification.** Read §5.7.3 + §5.7.4 end-to-end with the new figure
caption in mind. Numbers in prose should be consistent with the
figure caption + the new `z_space_collapse_stats.csv`.

---

### Chunk 14 — §5.7 (c): §5.7.4 closing + add `fig_p2_2_event_transitions` + `fig_book_resilience`

**Scope.** Add the two microstructure-dynamic figures at the end of
§5.7.4 (or §5.7.5). These are the strongest evidence that v9
captures dynamics no aggregate metric measures.

**Edits.**
1. Insert `fig_p2_2_event_transitions.pdf` near the end of §5.7.4
   with `\label{fig:event_transitions}`. New paragraph:
   - "Trade fraction is a single moment; the underlying event
     composition is more structural. Figure~\ref{fig:event_transitions}
     shows the event-type transition matrix (quote→quote,
     quote→trade, trade→quote, trade→trade) per protagonist. Real:
     ~95% quote→quote, tiny mass on the other three. v9: visually
     indistinguishable from real — ~94% quote→quote. v2 and v5:
     ~40% trade→trade and ~40% trade→quote, ~20% quote→quote — the
     event sequence is dominated by trades that don't exist in the
     real market. v9 is the only protagonist that reproduces real's
     event-process structure."
2. Insert `fig_book_resilience.pdf` at the end of §5.7 (after
   §5.7.5) with `\label{fig:book_resilience}`. New paragraph:
   - "An even sharper test of whether v9 captures market dynamics
     comes from the event-study around trade events. When a buy
     trade hits the ask in a real market, the depleted side
     (ask\_sz) drops sharply, then gradually replenishes as new
     limit orders arrive (LOB resilience, \cite{...}).
     Figure~\ref{fig:book_resilience} normalizes each trade's
     depleted-side queue to its pre-trade-window median and
     averages across thousands of trades per source. Real shows the
     canonical drop from 1.0 to ~0.88 at $t=0$ followed by partial
     recovery; v9 tracks real almost identically (drops to ~0.90,
     plateaus around 0.90); v2 and v5 stay flat at $\approx 1.0$
     across the entire window, because their trade-saturated event
     environment (55-70% trades) means the per-trade dip averages
     away. v9 reproduces a structural microstructure dynamic that
     no marginal-distribution or correlation metric in this report
     measures."
3. May need to add the LOB-resilience reference to `references.bib`
   (Bouchaud et al. 2009 or Large 2007). Cleanest: cite Bouchaud,
   Mezard, Potters 2002 "Statistical properties of stock order
   books".

**Verification.** §5.7.4 now closes with the event-transitions
finding; §5.7 closes with the book-resilience finding. Both
strengthen the v9-as-microstructure-realist case. `\ref{}` resolves.
New bib entry resolves.

---

### Chunk 15 — §6 Agents: keep prose, no figure changes

**Scope.** §6 currently has `tab:agent_truth` and `tab:rho_diff_spread`
inline, no figures. Per the user's existing framing ("§6 is a
practical implementation… real itself is biased…"), no new figures
are forced in §6.

**Optional**: forward-reference `fig:book_resilience` from §5.7 in §6
(book resilience is what market makers depend on, so the §5.7 figure
is mechanically relevant to §6's agent backtesting). One sentence
addition at the start of §6.4 ("Some Stylized Facts We Can Trust")
would do.

**Edits.**
1. Optional one-line forward-reference in §6.4 to `fig:book_resilience`
   if the user prefers cross-linking. Otherwise no change.

**Verification.** §6 unchanged structurally.

---

### Chunk 16 — §7 Cross-Asset: add `fig_6_2_cross_asset_axis_bars`

**Scope.** §7.2 currently has `tab:cross_asset_tsla` inline. Add the
3-panel cross-asset bars figure right after the table.

**Edits.**
1. Insert `fig_6_2_cross_asset_axis_bars.pdf` block immediately
   after `tab:cross_asset_tsla` in §7.2 with
   `\label{fig:cross_asset_bars}`. Caption:
   - 3 panels (G1, predictive, PCA) × 3 tickers (INTC, TSLA, SPY)
     × 2 architectures (v2, v5).
   - SPY column empty until pipeline drains; figure will be
     re-rendered post-SPY.
   - The TSLA finding (v2 dominates v5 on every axis) is visually
     immediate; the headline is that the INTC anti-correlation
     story partially dissolves on TSLA.
2. One paragraph after the figure:
   - "Figure~\ref{fig:cross_asset_bars} visualizes the table as a
     three-axis comparison. The TSLA bars confirm what
     Table~\ref{tab:cross_asset_tsla} reports numerically: v2
     dominates v5 outright on TSLA (the §7.4 'INTC-specific'
     reading). The SPY bars are pending the chained training
     completion."

**Verification.** §7.2 has table + figure. `\ref{fig:cross_asset_bars}`
resolves. SPY-pending caveat in caption.

---

### Chunk 17 — §8.3 Limitations: shrink caveat (vi), drop the figure

**Scope.** §8.3 currently has `fig:intraday_lob_volume` as caveat (vi).
That figure has moved to §5.2 (chunk 5). Reduce §8.3(vi) to a
one-sentence back-reference.

**Edits.**
1. **Remove** the `fig:intraday_lob_volume` block (lines 2020–2037).
2. Replace caveat (vi) prose with one sentence + back-ref:
   - "(vi) \emph{No intraday structure in synth output.} Our
     generators are trained on regime-conditioned event windows
     that are not anchored to time-of-day, so the synthetic output
     does not reproduce the intraday U-shape that real INTC
     exhibits in volume and queue size — see
     Figure~\ref{fig:intraday_lob_volume} (\S\ref{sec:res_g1}) for
     the visual."

**Verification.** §8.3 no longer contains a `\begin{figure}`. The
back-ref resolves to the §5.2 placement.

---

### Chunk 18 — Appendix wiring + cross-ref audit

**Scope.** Move the figures we removed from the body into the
appropriate appendices (currently empty stubs). Audit every
`\ref{fig:...}` and `\ref{tab:...}` to confirm resolution.

**Edits.**
1. **App. A `app:g1_matrix`**: insert `fig_2_1_g1_numerical_heatmap.pdf`
   block. Caption: "Full 20-checkpoint G1 numerical matrix; the
   §5.2 main-body figure (Figure~\ref{fig:g1_heatmap}) is the
   v2/v5/v9 × 4 regime slice."
2. **App.** (new section if needed): `fig_2_4_trade_fraction.pdf`
   and `fig_2_3_return_tail.pdf` as compact appendix exhibits.
3. **App.** (or skip): `fig_3_3_predictive_vs_scale.pdf` and
   `fig_1_2_pairwise_anticorrelation.pdf` — both user-flagged as
   table-replaceable; can either go to App. or be deleted entirely.
   Recommend keep in App. for completeness.
4. **Cross-ref audit**: run
   `grep -oE '\\\\ref\{(fig|tab|sec):[^}]+\}' docs/final_report.tex | sort -u`
   and verify every reference has a matching `\label{}` somewhere
   in the doc. Fix or remove any orphans.
5. **Bib audit**: run cite-key extraction and confirm any new
   citations from chunks 13/14 (Bouchaud-Mezard-Potters for LOB
   resilience, etc.) are in `references.bib`.

**Verification.** Zero broken refs. Zero unused figures in
`final_figs_tables/`. Bib clean.

---

## 2. Implications for the narrative

The user's "the discussions are going to be more towards the
direction i was intending" maps to four specific narrative shifts
that the figure choices imply. Worth flagging upfront so each
chunk's prose updates support them:

### 2.1 The v9 case strengthens

The original report had v9 as "wins coverage but loses predictive"
— ambivalent. With the new figures, v9 is also:
- The only protagonist that reproduces real's event-type transition
  matrix (`fig_p2_2_event_transitions`).
- The only protagonist that reproduces book resilience around trade
  events (`fig_book_resilience`).
- The only protagonist that reproduces real's mid-price staircase +
  sparse-trade event-level pattern (`fig_event_tape_walkthrough`).

These are three pieces of independent positive evidence that v9 captures
microstructure dynamics no aggregate metric in §5.0–§5.6 measures.
**The prose in §5.7.4 and §6 should reflect this** — v9 isn't just
"different fit" but "captures structural dynamics the others don't".

### 2.2 The v8 vs v8_b distinction sharpens

The figure (`fig_z_space_collapse`) shows v8_b is the canonical
single-point collapse (perfect degeneration to one z value), while
v8 is now a partial-recovery version. The §5.7.3 prose should:
- Use v8_b as the "what catastrophic copula collapse looks like"
  reference.
- Use v8 as the partial-recovery middle case.
- Acknowledge the original deep_diagnostic CSV numbers were from
  an older sample.

### 2.3 The intraday-structure failure moves to body

Currently buried in §8.3 (Limitations). The figure
(`fig_intraday_lob_volume`) is one of the strongest TRADES-style
real-vs-synth comparisons in the suite — it deserves §5.2 placement.
This is a structural finding about all three protagonists, not just
a caveat.

### 2.4 The "all 20 models" framing retreats to appendix

§5.5 (anti-correlation) and §5.2 (G1 heatmap) currently lean on
"across the 20 checkpoints we trained". The new figure suite uses
v2/v5/v9 throughout. The 20-model exhibits move to App. A. The
hero scatter (§5.0) is the only main-body exception, because its
whole point is "across the design space, no architecture wins all
three".

---

## 3. Estimated effort + ordering

Chunks ordered by risk + dependency:

| Chunk | Effort | Dependency |
|---|---|---|
| 2 — §3.2.2 t-SNE add | 10 min | none |
| 3 — §5.0/§5.1 + tape walkthrough | 15 min | none |
| 4 — §5.2 (a) swap heatmap, drop 2 figures | 25 min | none |
| 5 — §5.2 (b) add return-dist + intraday | 20 min | chunk 4 |
| 6 — §5.2 (c) add 3 supporting figures | 30 min | chunks 4-5 |
| 7 — §5.2 (d) prose pass + drop §5.3 dup | 25 min | chunks 4-6 |
| 8 — §5.3 drop figure | 15 min | chunk 7 |
| 9 — §5.4 verify | 5 min | none |
| 10 — §5.5 drop figure | 10 min | none |
| 11 — §5.6 add ablation heatmap | 15 min | none |
| 12 — §5.7 preamble + §5.7.1 verify | 10 min | none |
| 13 — §5.7.3 + §5.7.4 z-space figure + prose | 40 min | none (heavy prose update) |
| 14 — §5.7 add transitions + resilience | 30 min | chunk 13 |
| 15 — §6 agents (optional 1-line) | 5 min | chunk 14 |
| 16 — §7.2 add cross-asset bars | 15 min | none |
| 17 — §8.3 shrink caveat (vi) | 10 min | chunk 5 |
| 18 — Appendix + cross-ref audit | 30 min | all prior |

**Total estimated effort:** ~5 hours of focused conversation, in
chunks of 5–40 minutes each.

**Suggested ordering** (groups chunks with no inter-dependency for
parallelization, but we do them sequentially):

1. Chunks 2, 3 (additions to §3, §5.0, §5.1 — no removals, low risk)
2. Chunks 9, 10, 11 (small §5.4–§5.6 changes — independent)
3. Chunks 4 → 5 → 6 → 7 → 8 (§5.2 + §5.3, must be sequential)
4. Chunks 12 → 13 → 14 (§5.7 sequential)
5. Chunk 15 (optional §6)
6. Chunk 16 (§7)
7. Chunk 17 (§8.3 — depends on chunk 5 having moved the figure)
8. Chunk 18 (final audit)

---

## 4. What I will not do without explicit go-ahead

- Add bib entries beyond Bouchaud-Mezard-Potters in chunk 14
  (everything else is already in `references.bib`).
- Re-run any cluster scripts (we treat the current
  `final_figs_tables/` as the source of truth).
- Touch §1, §2.1–§2.3 (no figure changes there; prose stays as-is).
- Touch §3.1 / §3.3 / §3.4 (no figure changes; only §3.2.2 gets the
  t-SNE).
- Touch §8.1, §8.2, §8.4 (no figure changes; prose stays as-is).
- Touch the abstract or the AI Disclosure block (already final).
- Re-rename any `\label{}` (preserve existing labels; replace figure
  contents while keeping the label so all `\ref{}` keep working).

---

## 5. Verification protocol per chunk

After each chunk:

1. `grep -c "\\\\begin{figure}" docs/final_report.tex` — confirm the
   expected count (should grow by N if N figures inserted, shrink if
   removed).
2. `grep -oE "\\\\ref\{(fig|tab):[^}]+\}" docs/final_report.tex | sort -u`
   compared against `\\\\label\{...\}` extraction — confirm zero
   orphan references.
3. Read the affected section's prose top-to-bottom and confirm every
   figure mention resolves to a figure in the body (or an explicit
   App. forward-reference).
4. User reviews the chunk's diff and approves before moving on.

---

**Ready to execute.** Each chunk is small enough to do in one
conversation turn; we go through them in order, and the user
approves between chunks.
