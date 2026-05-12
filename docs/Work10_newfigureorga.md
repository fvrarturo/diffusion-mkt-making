# Work10 — New Figure Organisation for the Main Paper

**Decision rule.** Main body figures are TRADES-style real-vs-synth
exhibits with structural visual content (densities, ACFs, intraday
patterns, transition matrices, event-level dynamics). Anything reducible
to a small table (ordered bar charts, pairwise scatters, correlation
pairs) goes to the appendix. All main-body comparisons are restricted
to the three protagonists **v2 / v5 / v9**.

**Scope.** **Seventeen** figures in the main body (was 14 in the
original plan; +2 HFT-microstructure figures from `Work10_+2figs.md`
— `fig_event_tape_walkthrough` and `fig_book_resilience` — and +1
copula-mechanism figure — `fig_z_space_collapse` — restored after
being dropped during the reorg). Everything else moves to the
appendix or is dropped.

**Canonical figure source.** All main-body figures and tables now live
in `final_figs_tables/`. The report's `\graphicspath` is configured as
`{{../final_figs_tables/}{../results/report_figures/}{../results/}}`,
so any `\includegraphics{<basename>.pdf}` resolves to
`final_figs_tables/` first. This is the directory to keep in sync as
scripts re-render — `cp results/report_figures/X.pdf final_figs_tables/`
is the staging step.

This document is a wiring brief — it names each figure, its target
section, what it shows, what surrounding prose needs to move, and any
blocker that must be resolved before the figure can land.

---

## 1. Section-by-section layout (target state)

All paths below refer to the canonical `final_figs_tables/` location.

| § | Figure | Status | Notes |
|---|---|---|---|
| §2.4 | `fig_a_real_stylized_facts` | ✅ wired | none |
| §2.4 or §5.1 | `fig_p2_3_intraday_volume` | ⚠ pending decision | overlaps panel F of stylized-facts; lean drop unless we add synth overlay |
| §3.2.2 | `fig_p2_4_regime_tsne` | 🟡 ready, not wired | clean as-is |
| §5.0 | `fig_1_1_three_axis_scatter` | ✅ wired (hero) | none |
| §5.1 | `fig_10_1_rank_bump` | ✅ wired + 🟡 re-rendered | v2/v5/v9 highlighted (lw=3.2, alpha=1.0); rest faded (alpha=0.18) |
| §5.1 (NEW) | **`fig_event_tape_walkthrough_500`** | 🟡 ready, not wired | NEW figure: 4 cols × 4 rows event-level tape comparison; v9 visually matches real's staircase + sparse-trade pattern |
| §5.2 | `fig_return_distribution_full` | 🟡 ready, not wired | re-rendered with balanced 4-regime sample; kurt now matches body (real=755, v2=125, v5=907, v9=136); Q-Q now conditional on \|return\|>0 |
| §5.2 | `fig_p2_1_spread_conditional` | 🟡 ready, not wired | already v2/v5/v9 |
| §5.2 | `fig_intraday_lob_volume` | ✅ wired in §8.3 → 🟢 to move to §5.2 | repurpose: better in §5 than buried in Limitations |
| §5.2 | `fig_7_3_trade_sign_acf` | 🟡 ready, not wired | now v2/v5/v9; deduped per (model, lag); real dashed |
| §5.2 or §5.7 | `fig_7_1_vol_vol_kde` | 🟡 ready, not wired | now v2/v5 (v9 missing from underlying CSV — script 62 needs v9 re-run; can ship without v9 with caption note) |
| §5.2 (or App) | `fig_p0d_g1_per_regime` | 🟡 ready, not wired | v9 NaN cells now show "n/a" with gray bg (validator-level NaN, not a parser bug) |
| §5.6 | `fig_5_1_ablation_heatmap` | 🟡 ready, not wired | clean as-is |
| §5.7.3/§5.7.4 (NEW) | **`fig_z_space_collapse`** | 🟡 ready, not wired | NEW figure: 3-panel histograms (v8 / v8_b / v9) of mid_return z-values with N(0,1) overlay. **v8_b is the perfect-collapse exemplar (single z value at z=0); v9 adds tail spread to the same mode location; v8 is the partially-recovered middle ground.** Body prose update needed when wired (current prose oversimplifies the v8/v8_b/v9 distinction — see §3.14a card) |
| §5.7 | `fig_p2_2_event_transitions` | 🟡 ready, not wired | already v2/v5/v9; v9 visually identical to real |
| §5.7 (NEW) | **`fig_book_resilience`** | 🟡 ready, not wired | NEW figure: queue replenishment around trade events; **v9 reproduces real's drop-and-recover pattern, v2/v5 don't** — first piece of evidence that v9 captures a microstructure dynamic invisible to all aggregate metrics |
| §7.2 | `fig_6_2_cross_asset_axis_bars` | 🟡 ready, not wired | SPY column empty until pipeline drains |

That is **17 figures** across sections §2.4, §3.2.2, §5.0–§5.7, and §7.2.

**Status legend:** ✅ in body now / 🟡 in `final_figs_tables/`, ready to wire / 🟢 needs prose rewrite / ⚠ pending decision

---

## 2. What leaves the main body

The current `final_report.tex` cites 10 figures (7 unique). After the
reorg, three of them move to the appendix and two are dropped outright.

### 2.1 Dropped (per user's "could-be-a-table" rule)

| Figure | Currently at | Why drop |
|---|---|---|
| `fig_3_3_predictive_vs_scale` | §5.3 | Sorted scatter; same info as a 3-row table |
| `fig_1_2_pairwise_anticorrelation` | §5.5 | Three pairwise scatters; same info as `tab:correlation_subset` already in body |

### 2.2 Moved to appendix

| Figure | Currently at | New home | Reason |
|---|---|---|---|
| `fig_2_1_g1_numerical_heatmap` | §5.2 | App. A `app:g1_matrix` | 20-column heatmap — main body uses v2/v5/v9 only via `fig_p0d_g1_per_regime` |
| `fig_2_4_trade_fraction` | §5.2 | App. | 20-model sorted bar |
| `fig_2_3_return_tail` | §5.2 + duplicate at §5.3 | App. | Tail probability 3-threshold panel; main body uses `fig_return_distribution_full` |

### 2.3 Already in body, retained

- `fig_a_real_stylized_facts` (§2.4)
- `fig_1_1_three_axis_scatter` (§5.0 hero)
- `fig_10_1_rank_bump` (§5.1)
- `fig_intraday_lob_volume` (§8.3 — to move up to §5.2 per §1 above)

---

## 3. Per-figure deep-dive cards

Each card states: what's in the figure, what story it serves, what
prose changes around it, and any remaining blocker. Items marked
**[CLEARED]** had blockers in the original plan that have since been
resolved.

### 3.1 `fig_a_real_stylized_facts` — §2.4 Stylized Facts of Financial Returns
- **Content:** six panels of real-INTC stats from 10 val days (~25.5M
  events). A: log-y return density with kurt = 755 + frac_zero = 98.78%
  annotated. B: |return| ACF with slow decay (vol clustering). C: return
  ACF near zero. D: trade_sign ACF with lag-1 = 0.668 (order-flow
  persistence). E: spread distribution with 95.82% at one tick. F:
  intraday volume 15-min bucketed (reverse-J shape).
- **Status:** already wired; dual-window footnote remains.
- **Prose around it:** unchanged.

### 3.2 `fig_p2_3_intraday_volume` — §2.4 (overlaps panel F) or skip
- **Content:** real-only intraday volume distribution, 15-min buckets
  09:45–15:45 ET. Reverse-J shape.
- **Recommendation:** **drop** in favor of panel F of
  `fig_a_real_stylized_facts` unless we add v2/v5/v9 overlay lines
  (which would require a script 83 patch + cluster re-run). The
  intraday-LOB-volume figure (§3.8 below) carries the synth-comparison
  story already.

### 3.3 `fig_p2_4_regime_tsne` — §3.2.2 Conditioning: FiLM vs AdaLN-Zero
- **Content:** t-SNE projection of the 4 regime embeddings (base,
  high_vol, toxic, thin) per model, for v2, v5, v9. Three side-by-side
  panels with 4 dots each, well-separated.
- **Story it serves:** "AdaLN-Zero conditioning differentiates
  regimes" — supports §3.2.2's FiLM-collapse vs AdaLN-Zero claim. The
  4-point structure is visually thin but is the only direct evidence
  for the conditioning-bandwidth claim.
- **Recommendation:** keep. Place at end of §3.2.2 immediately after
  the FiLM modulation-collapse sentence. Caption discloses the 4-point
  limitation.

### 3.4 `fig_1_1_three_axis_scatter` — §5.0 hero
- **Content:** 20-checkpoint scatter on (G1 fraction, predictive-score
  inverse, PCA coverage point size, family color).
- **Status:** already wired.
- **Caveat:** has all 20 models, not just v2/v5/v9. This is the one
  exception to the v2/v5/v9 rule because the *whole point* of the hero
  is "across the design space, no architecture wins all three" —
  protagonist-only would defeat the message.

### 3.5 `fig_10_1_rank_bump` — §5.1 Three Representative Generators
- **Content:** rank-bump chart, three columns (G1, predictive, PCA),
  20 line traces with v2 / v5 / v9 highlighted.
- **Status:** already wired + **[CLEARED]** re-rendered.
  - v2/v5/v9 bold (lw=3.2, alpha=1.0) with distinctive colors
    (blue/green/purple from `PROTAGONIST_COLOR`).
  - Other 17 lines faded to alpha=0.18, lw=0.7.
  - Annotations only on the three protagonists.
  - Crossing-lines pattern across G1 → Predictive → PCA cleanly
    visible.

### 3.6 `fig_event_tape_walkthrough_500` — §5.1 (NEW figure, no prior plan slot)
- **Content:** 500-event slice from a representative INTC tape per
  source, displayed as 4 cols (real, v2, v5, v9) × 4 rows
  (mid-price step function in bps, top-of-book queue depth,
  spread, trade markers). Per-column trade rate annotated on the
  trade-markers row (real 4.0%, v2 70.8%, v5 54.0%, v9 8.8%).
- **What the reader sees without statistics:**
  - **Real**: mid-price flat for ~350 events then a single sharp drop
    to −2.5 bps (canonical staircase). Queue depth smooth ~5–15K.
    Spread pinned at exactly $0.01 across all 500 events. Trades
    sparse (4%), clustered.
  - **v2**: mid-price drifts continuously (no flat stretches), V-shape
    from 0 to −0.4 bps. Queue depth similar to real but more variable.
    Spread chaotic, oscillating constantly. Trades on 70% of events.
  - **v5**: similar to v2 on most rows but mid-price has a
    real-like step at event ~250. Queue depth drops sharply mid-tape
    (likely a tape-concat seam).
  - **v9**: mid-price has the flat-then-jump structure (sparse
    moves), spread occasionally widens, trades rare (8.8% — closest
    to real's 4%). Queue depth small (~1K) but stable.
- **Story it serves:** the "what does the data actually look like"
  figure that no aggregate metric exposes. Shows the reader at the
  event level that v2/v5 generate a fundamentally different event
  process than real (continuous drift + constant trading), while v9
  gets the event structure right (rare trades, price pinning,
  occasional spread widening).
- **Recommendation:** §5.1 immediately after `fig_10_1_rank_bump`.
  Pairs naturally with the protagonist introduction and gives the
  reader an event-level intuition before any per-axis prose.
- **Generation:** `scripts/89_event_tape_walkthrough.py` with
  `--events 500 --regime base`. Multi-tape concat handles v9's
  shorter native tape length.

### 3.7 `fig_return_distribution_full` — §5.2 Distributional Realism (lead figure)
- **Content:** TRADES Fig 3 style. Panel A: log-y density real vs
  v2/v5/v9. Panel B: Q-Q plot conditional on `|return| > 0`.
- **Status: [CLEARED]** — re-rendered on cluster with two patches:
  - **Sample size:** raised `--max-n-points` from 2M → 30M, and
    switched synth loading to balanced per-regime (mimicking script
    50's `_sample_synth`). Real now reads kurt=755 ✓, v2=125 (body
    122 ✓), v5=907, v9=136 (body 129 ✓).
  - **Q-Q:** restricted to non-zero return subset. Previously the
    98.78% zero-mass made every quantile ∈ [0.01, 0.99] flat-line at
    zero; now the curves show actual tail-shape comparisons.
- **Story it serves:** v2 too narrow, v5 heavier tails, v9 has the
  matching zero-mass spike + intermediate tails.
- **Recommendation:** §5.2 lead figure, immediately after the
  paragraph that introduces the G1 axis.

### 3.8 `fig_p2_1_spread_conditional` — §5.2 supporting exhibit
- **Content:** three-panel bar chart of spread-conditional dynamics
  (events with spread > 1 tick) for real / v2 / v5 / v9. Real
  P(spread>1 tick) = 4.18%; synth all 50–87%. Mean bid_sz | wide on
  real = 3.76e3, on synth ~800–2200. Mean |return| | wide on real =
  1.55e-05, on synth ~1.5–3.0e-07.
- **Story it serves:** when a real spread widens, the book gets
  thicker AND prices move more. Synth widens spread far too often
  AND fails to populate the wide-spread regime with realistic depth
  or volatility.
- **Format honesty:** bar-chart figure, borderline by the
  table-or-figure rule. Kept because the three-axis joint structure
  ("spread widens AND depth↑ AND |return|↑") is legible at a glance
  in a way a table wouldn't be.

### 3.9 `fig_intraday_lob_volume` — move from §8.3 → §5.2 (or §2.4)
- **Content:** four-panel level-1 LOB queue size by minute-of-session,
  real vs v2 / v5 / v9. Real has the canonical intraday U-shape;
  v2/v5/v9 are essentially flat profiles.
- **Story it serves:** the limitation made visible. Synth windows are
  not time-anchored, so no intraday structure.
- **Current placement:** §8.3 Limitations (just wired).
- **Recommendation:** move to §5.2 as a structural finding rather
  than burying in Limitations, with a back-pointer from §8.3.
- **Required prose changes:**
  - Add 1 paragraph at end of §5.2 introducing the figure.
  - Truncate the §8.3(vi) limitation to a short sentence + back-ref.

### 3.10 `fig_7_3_trade_sign_acf` — §5.2 supporting exhibit
- **Content:** trade-sign ACF (real lag-1 = 0.413 from
  temporal_acf.csv; the `REAL_REF["trade_sign_acf_lag1"] = 0.675`
  dotted line in the figure is from the script-81 source — same
  dual-window discrepancy footnoted in §2.4) vs v2 / v5 / v9 across
  lags 1, 5, 10, 50, 100.
- **Status: [CLEARED]** — re-rendered with v2/v5/v9 (was v2/v7_b/v9)
  and dedupe per (model, lag) so the apparent "jumps" between
  duplicate measurements at the same lag are gone. Real (dashed
  black) is now monotonically decaying. v2 (blue) starts near 0.22 at
  lag-1; v5 (green) drops to 0.06; v9 (purple) starts at 0.24 then
  drops fastest.
- **Reading:** synth all under-shoot real's trade-sign persistence;
  v5 the worst, v9 closest at lag-1 then drops fastest.
- **Cosmetic note:** the "real lag-1 = 0.675" dotted line conflicts
  with the actual real curve at 0.413. Either drop the dotted line
  or annotate why two real-data references disagree.

### 3.11 `fig_7_1_vol_vol_kde` — §5.2 or §5.7
- **Content:** TRADES Fig 4 style. Per-tape Pearson(volume, |return|)
  KDE. Real (n=20 days, centered around mean 0.341, tight unimodal)
  vs v2 + v5 (n=400 each, much wider distributions spanning −1.5 to
  +1.5).
- **Status: [PARTIALLY CLEARED]** — re-rendered with v2/v5 in
  protagonist colors (was v2/v3p5/v7_b). **v9 still missing from
  the underlying `vol_volatility_correlation.csv`** because both
  cluster and local versions of the CSV lack a v9 row; needs script
  62 re-run with v9 in the model list. Likely v9's per-tape volume
  Pearson is undefined for many tapes (frac_zero ~99% means most
  tapes have 0 trades → undefined Pearson).
- **Recommendation:** ship §5.2 with v2/v5 vs real and note in caption
  that v9 is excluded; OR run script 62 on cluster to produce v9
  values and re-render.

### 3.12 `fig_p0d_g1_per_regime` — §5.2 (replacing 20-model G1 heatmap)
- **Content:** per-regime G1 heatmap. 15 rows × 13 columns: v2/v5/v9 ×
  (base, high_vol, toxic, thin) + real. Cross-sectional gradient per
  row.
- **Status: [CLEARED]** — re-rendered with v9 NaN cells now showing
  "n/a" text on a light-gray background (was silently blank). The
  underlying NaN values are validator-level (`synth=nan` literally
  in the upstream CSV), not a parser bug — three checks (RV
  signature shape, Trade-sign lag-1 ACF sign, ACF(|r|) power-law β)
  are undefined for v9 because v9's distinctive output (low trade
  fraction, scrambled per-event values) makes the upstream
  computations fail.
- **Visible findings:**
  - v2 base regime massively overshoots kurtosis (3919 vs real 755).
  - v5 base regime overshoots (1696). High_vol/toxic close to real.
  - v9 across all regimes closer to real on kurtosis.
- **Recommendation:** replace `fig_2_1_g1_numerical_heatmap` in §5.2
  with this; move the 20-model version to App. A `app:g1_matrix`.
  Caption discloses the v9 n/a rows + points to App. A.

### 3.13 `fig_5_1_ablation_heatmap` — §5.6 Architectural Ablation
- **Content:** 6 toggles × 6 metrics heatmap (ΔG1, ΔPred×, ΔPCA%,
  Δkurt, Δtrade frac, Δvol-vol corr). Diverging red/blue colormap
  with per-column normalisation. "FiLM→AdaLN +1184 kurt" jumps out
  as the strongest single move; "−x0_clip −40 PCA%" jumps out as
  the destructive move.
- **Story it serves:** visual companion to `tab:ablation_walks`.
- **Recommendation:** add at end of §5.6 *after* the table, with a
  caption pointing out two-three visually-obvious cells.
- **Status:** clean as-is.

### 3.14 `fig_p2_2_event_transitions` — §5.7 (new supporting exhibit)
- **Content:** event-type transition matrix per model, stacked-bar
  format. Real: ~95% quote→quote, tiny trade→anything. v9: visually
  indistinguishable from real (~94% quote→quote). v2 and v5: ~40%
  trade→trade and trade→quote, ~20% quote→quote.
- **Story it serves:** v9 is the *only* protagonist that reproduces
  real's event composition. Strong support for §5.7.4's "the fix
  recovers" claim.
- **Recommendation:** add at the end of §5.7.4 (after the v9 G1
  recovery prose).
- **Required prose changes:** §5.7.4 currently mentions trade-fraction
  recovery (0.009→0.049) but not event-transition composition; need
  a new paragraph stating the joint structural finding.

### 3.14a `fig_z_space_collapse` — §5.7.3/§5.7.4 (RESTORED figure, was dropped in reorg)
- **Status:** ✅ **shipped to `final_figs_tables/`**, ready to wire.
  Three-panel layout (v8, v8_b, v9). Generation:
  `scripts/92_z_space_collapse.py --models v8 v8_b v9 --feature mid_return --max-files-synth 1000`.
- **Content:** three-panel histogram comparison. Each panel shows
  $z = \Phi^{-1}(\hat{F}(\texttt{mid\_return}))$ for one model on
  log-y, with the $\mathcal{N}(0,1)$ reference density overlaid as
  a dashed black curve. The empirical CDF $\hat{F}$ is the same
  one used by the actual copula transform (script 60's
  `forward_copula`), so these are the z-values the diffusion model
  was trained against.
- **What the figure actually shows (different from the body's
  current prose — see §5.7 prose update note below):**
  - **v8 panel** (olive): mode-spike at z≈−2.55 + secondary mass
    spread to ±3. mean=−2.00, std=1.09, kurt=0.8. v8 is a
    partially-recovered version, not the catastrophic collapse the
    deep_diagnostic CSV originally recorded — v8 has been re-sampled
    since.
  - **v8_b panel** (dark olive): perfectly degenerate single-point
    distribution — every z-value lands at z=0. mean=0.00, std=0.00,
    kurt=undefined. The canonical "collapse to a single point"
    exemplar. Note: v8_b also collapses to z=0, not z=−2.55 as the
    older deep_diagnostic CSV claimed; v8_b has also been
    re-sampled.
  - **v9 panel** (purple): mode at z=0 (same as v8_b) + visible
    tail masses at ±2.7. mean=0.00, std=0.28, kurt=101. v9's
    contribution over v8_b is the tail spread, not the mode
    location.
- **Important visual feature:** the synth z-distributions are
  fundamentally **trimodal** — central spike + thin edge masses
  with a gap between ±2 and ±2.5. The gap is real data structure,
  not a viz artifact: the empirical CDF of `mid_return` has a giant
  zero-mass plateau plus thin tails, so synth z-values land either
  at a single plateau bucket or at the far-tail edges, never in
  between. That gap is the failure mode the figure exposes.
- **Runtime warnings:** running script 92 emits `Precision loss
  occurred in moment calculation` on v8_b (because all values are
  identical, std=0 makes higher moments undefined). Expected,
  non-blocking. Could be silenced with an `np.errstate` guard if
  the noise becomes annoying.
- **Story it serves:** the load-bearing visual for §5.7.3 (The
  Failure) and §5.7.4 (The Fix). v8_b anchors §5.7.3's collapse
  description (degenerate single-point z); v9 anchors §5.7.4's
  recovery description (mode at z=0 + tail spread). v8 sits in
  between as evidence that even partial improvement still leaves
  the trimodal pattern intact.
- **Recommendation for placement:** add at the boundary between
  §5.7.3 and §5.7.4, spanning the page. Caption explicitly
  describes the trimodal pattern + flags the v8 vs v8_b distinction.
- **⚠ Body-prose update needed when §5.7 is rewired:** the body's
  current §5.7.3/§5.7.4 says "v8 collapses to z=-2.55, v9
  recovers". The figure tells a more nuanced story:
  - v8 is partially-recovered (mode at -2.55, but spread).
  - v8_b is the perfect collapse exemplar — collapses to z=0
    (not -2.55).
  - v9's "fix" relative to v8_b is tail spread, not mode shift —
    both v8_b and v9 have mode at z=0; v9 just additionally emits
    small mass at ±2.6.
  - The deep_diagnostic CSV numbers cited in the body
    (z_mean=-2.55 for v8, z_std=0.0 for v8_b) come from older
    sampling that's no longer reproducible without rebuilding the
    archive. Either re-run script 60 on current samples and update
    the body to match, or keep the older numbers in the body with
    a footnote acknowledging the figure shows current-sample
    statistics.
- **Defer this prose update to whenever §5.7 gets its main rewire
  pass** — not blocking the figure ship.

### 3.15 `fig_book_resilience` — §5.7 (NEW figure, no prior plan slot)
- **Content:** event-study of book queue dynamics around trade events.
  X-axis: event offset relative to a trade event, −30 to +30. Y-axis:
  queue size on the depleted side (ask for buy trades, bid for sell
  trades), normalized to the **median of the pre-trade window**
  `[t-30, t-1]` per trade. Curves are **median across trades**;
  bands are **IQR**. n_trades: real 5,000 / v2 4,960 / v5 4,725 /
  v9 3,898.
- **What the figure shows (substantive finding):**
  - **Real**: depleted side drops sharply at t=0 from 1.0 to ~0.88,
    then plateaus at 0.88–0.90 through +30 (recovery timescale on
    INTC > 30 events).
  - **v9**: tracks real almost identically — drops to ~0.90 at t=0,
    plateaus around 0.90–0.92. **v9 reproduces the resilience
    signature.**
  - **v2 / v5**: stay flat at ~1.0 across the entire window. No dip
    visible. Their trade-saturated environment (55–70% of events
    are trades) means the "pre-trade window" is contaminated with
    other trades, so the per-trade dip averages away.
- **Story it serves:** the strongest single piece of evidence that
  v9 captures a microstructure dynamic invisible to every aggregate
  metric. v9 doesn't just match marginals (G1) and event composition
  (`fig_p2_2_event_transitions`) — it also matches an event-study
  dynamic property of the book that no metric in the G1 suite
  measures.
- **Recommendation:** add at end of §5.7 (after `fig_p2_2_event_transitions`)
  as the final §5.7 exhibit. Alternatively §6 (agents) — book
  resilience is what market-makers depend on, so this directly
  supports the §6 framing about agent-relevant microstructure
  properties.
- **Generation:** `scripts/91_book_resilience.py` with `--window 30
  --max-trades 5000`. Robust aggregation (per-trade pre-window
  median normalization + median+IQR across trades) was needed to
  expose the dip; the original mean+std version had outlier-driven
  bands that washed out the signal.

### 3.16 `fig_6_2_cross_asset_axis_bars` — §7 Cross-Asset
- **Content:** 3-panel (G1, Predictive, PCA) × 3-ticker (INTC, TSLA,
  SPY) × 2-architecture (v2, v5). TSLA v2 dominates TSLA v5 on every
  axis; TSLA v5 PCA collapse to 18.9%. SPY column empty (pipeline
  in flight).
- **Recommendation:** add at top of §7.2 immediately after
  `tab:cross_asset_tsla`. Caption notes pending SPY column.
- **Blocker:** SPY column empty until pipeline drains. Acceptable to
  ship as-is and re-render post-SPY.

---

## 4. Required script re-runs

Most blockers from the original plan are resolved. Open items:

| # | Script | What | Status |
|---|---|---|---|
| 1 | `scripts/87_return_distribution_full.py` | balanced 4-regime sample + Q-Q on \|return\|>0 | ✅ DONE (cluster) |
| 2 | `scripts/80_report_figures.py` (`fig_7_1_vol_vol_kde`) | v2/v5/v9 + protagonist palette | ✅ DONE locally; ⚠ v9 missing from upstream CSV (needs script 62 re-run with v9) |
| 3 | `scripts/80_report_figures.py` (`fig_7_3_trade_sign_acf`) | v2/v5/v9 + dedupe per (model, lag) | ✅ DONE |
| 4 | `scripts/88_per_regime_g1.py` | gray + "n/a" annotation for v9 NaN cells | ✅ DONE (display fix; upstream NaN remains) |
| 5 | `scripts/80_report_figures.py` (`fig_10_1_rank_bump`) | highlight v2/v5/v9, fade rest | ✅ DONE |
| 6 | `scripts/89_event_tape_walkthrough.py` | NEW script, multi-tape concat | ✅ DONE (cluster) |
| 7 | `scripts/91_book_resilience.py` | NEW script, robust median+IQR | ✅ DONE (cluster) |
| 8 | `scripts/80_report_figures.py` (`fig_5_1_ablation_heatmap`) | clean re-render | ✅ DONE |
| 9 | `scripts/92_z_space_collapse.py` | NEW script, v8/v8_b/v9 z-histograms with N(0,1) overlay | ✅ DONE (cluster) — 3-panel figure shipped to `final_figs_tables/`; body prose update deferred to §5.7 rewire |
| 10 | `scripts/90_cross_asset_comparison.py` | rerun after SPY artifacts land | ⏳ pending SPY |
| 11 | `scripts/62_trades_metrics.py` (optional) | re-run with v9 to populate vol_volatility_correlation.csv | ⏳ optional, would unblock v9 in `fig_7_1` |

---

## 5. Appendix overflow

| Figure | App. section | Why |
|---|---|---|
| `fig_2_1_g1_numerical_heatmap` (20 models) | App. A `app:g1_matrix` | Full population reference; main body uses v2/v5/v9 only |
| `fig_2_4_trade_fraction` | App. | 20-model sorted bar |
| `fig_2_3_return_tail` | App. | Tail probability 3-threshold panel |
| `fig_3_3_predictive_vs_scale` | App. or skip | Sorted scatter, table-replaceable |
| `fig_1_2_pairwise_anticorrelation` | App. or skip | 3-panel scatter, redundant with `tab:correlation_subset` |
| `B1_return_distribution.pdf` per model (`results_cluster/_archive/v*/validation/`) | App. C | Per-model TRADES-style density panels, all 20 models |
| `C1_acf_abs_return.pdf` per model | App. C | Per-model vol-clustering panels |
| `D1_conditional_return_imbalance.pdf` per model | App. E | Per-model OFI→return signal |
| `midprice_traces_v*.pdf` (all 20) | App. F `app:traces` | TRADES Fig 8 style mid-price overlays |
| `fig_p2_4_regime_tsne` per-regime expansions (future) | App. | Future work: full per-day embedding projection |
| `figure7.pdf` per model | App. (existing `app:agents_impl`) | Bootstrap hypothesis test |

The full body of cluster work in `results_cluster/_archive/<model>/`
becomes the per-model appendix material; the main paper picks the
v2/v5/v9 slices.

---

## 6. Sectional impact summary (what prose needs to change)

| § | Current state | Reorg effect on prose |
|---|---|---|
| §2.4 | one figure (`fig_a_real_stylized_facts`) + footnote | unchanged |
| §3.2.2 | prose-only re. FiLM modulation collapse | **add 1 sentence + figure** (`fig_p2_4_regime_tsne`) |
| §5.0 | hero scatter + table | unchanged |
| §5.1 | rank-bump figure (re-rendered) | **add `fig_event_tape_walkthrough_500`** as event-level intuition exhibit |
| §5.2 | 3 figures (G1 heatmap 20-model + trade fraction + return tail) | **major rewrite:** swap G1 figure to per-regime v2/v5/v9; add return-distribution-full, spread-conditional, intraday LOB volume, trade-sign ACF, vol-vol KDE figures. **Replace 3 figures → ~6 figures.** Prose narrative re-anchors on v2/v5/v9 instead of all 20 |
| §5.3 | broken (duplicate placeholder) | **drop the duplicate**; possibly drop the §5.3 figure entirely since `fig_3_3_predictive_vs_scale` is going to appendix |
| §5.5 | pairwise figure + correlation table | **drop figure**, keep table |
| §5.6 | ablation table | **add `fig_5_1_ablation_heatmap`** |
| §5.7 | prose + tables | **add `fig_z_space_collapse`** at §5.7.3/§5.7.4 boundary (the visual proof of the failure + fix), **`fig_p2_2_event_transitions`** at §5.7.4 + **`fig_book_resilience`** at end of §5.7 (or §6) |
| §7.2 | TSLA table | **add `fig_6_2_cross_asset_axis_bars`** |
| §8.3 | `fig_intraday_lob_volume` as caveat (vi) | **move figure to §5.2**, keep one-sentence (vi) with back-ref |

§5.2 takes the heaviest rewrite — the body currently runs three figures
across G1 in a "look at all 20" framing, and the new layout runs ~six
figures all in the "v2 vs v5 vs v9 vs real" framing. Prose needs to
track this shift. §5.7 also grows by two figures (event transitions +
book resilience), both pieces of evidence that v9 captures
microstructure dynamics no aggregate metric exposes.

---

## 7. Open questions

1. Should `fig_p2_3_intraday_volume` (real-only intraday volume) be
   dropped entirely given panel F of `fig_a_real_stylized_facts`
   covers the same content? Lean drop unless we add v2/v5/v9 overlay.
2. The §5.3 figure: drop entirely (prose-only) or wire
   `fig_3_1_predictive_score`?
3. `fig_intraday_lob_volume` location: §2.4 (real-data exhibit) or
   §5.2 (failure mode of all three protagonists) or both with a
   small/large split? Lean §5.2.
4. `fig_book_resilience` location: §5.7 (final exhibit, supports v9
   case study) or §6 (agent backtesting microstructure assumption)?
   Lean §5.7 because it's the climax of the v9-reproduces-microstructure
   thread that runs through `fig_p2_2_event_transitions`.
5. `fig_event_tape_walkthrough_500` location: §5.1 (immediately after
   `fig_10_1_rank_bump` as protagonist intro) or §2.1 (data-mechanics
   intuition before any modelling)? Lean §5.1 because the figure is
   a comparison, not a data-only exhibit.
6. `fig_7_1_vol_vol_kde` v9 missing: ship as-is with v2/v5 + caption
   note, or run script 62 with v9 (optional cluster re-run, would
   complete the protagonist trio)?
7. After SPY artifacts land: re-render `fig_1_1_three_axis_scatter`
   with TSLA + SPY v2/v5 points overlaid? Currently INTC-only.
