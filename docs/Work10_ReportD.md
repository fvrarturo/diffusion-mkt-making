# Work10 — Report D: Body Compression / Lean Pass

**Goal.** Move ~5 pages of mechanism / table / repeated-prose content
out of the main body into the appendix, leaving the main paper with
the punch lines and the visual evidence. Per the user's six explicit
proposals.

**Decision rule.** Anything the reader needs to follow the central
arc (three-axis tension, v9-microstructure findings, cross-asset
qualifier, practical-implications) stays in the body. Anything that
is mechanism / full-population reference / restatement of what a
table already says moves to appendix or gets compressed to one
sentence.

**Working method.** Six chunks plus one appendix-wiring chunk, same
as Work10_ReportC. Each chunk is a single edit-and-verify pass with
a clear before/after state and a concrete pass count for the
verification step. We execute one chunk at a time; the user reviews
between chunks.

---

## 0. Preflight: where we are

Body figure count after Work10_ReportC: **15** in body, **6** in
appendix (App. A: 3, App. C: 2, body refs to App. C heatmap: 1).

Body word count target: drop ~5 pages = ~2,500 words. Roughly:

| Source | Pages today | Pages target | Cut |
|---|---|---|---|
| §3.2.2--§3.2.5 (4 subsubsections) | ~2.5 | ~1.5 | 1.0 |
| §5.4 (PCA Coverage) | ~1.0 | ~0.25 | 0.75 |
| §5.5 (Anti-Correlation) | ~1.0 | ~0.5 | 0.5 |
| §5.7.1--§5.7.3 (Copula mechanism) | ~1.5 | ~0.25 | 1.25 |
| §6.2 (Ranking Mechanism setup) | ~1.0 | ~0.25 | 0.75 |
| §6.3+§6.4 (RDIFF + facts) | ~1.5 | ~1.0 | 0.5 |
| **Total** | | | **~4.75 pages** |

Five new appendix sections needed (or content threaded into existing
stubs):

| App. label | Source | Status |
|---|---|---|
| `app:copula_mechanism` (NEW) | §5.7.1--§5.7.3 prose + tab + fig | new section |
| `app:pca_coverage_full` (NEW) or thread into App. C | `tab:pca_coverage_top` | new section |
| `app:agents_impl` (existing stub, currently empty) | `tab:agent_truth` + non-responsiveness prose | populate stub |
| App. enumeration: `tab:v8_z_collapse` lands in `app:copula_mechanism` together | | |

---

## 1. Chunk-by-chunk plan

Six chunks (D1--D6) plus a final appendix-wiring chunk (D7).

### Chunk D1 — §3.2 Methods compression

**Scope.** Compress §3.2.2 (Conditioning), §3.2.3 (EDM), §3.2.4
(Copula Transform), §3.2.5 (Min-SNR). Keep intuition, drop the
math-heavy parts that the body's later figures don't need.

**Specific cuts.**

- **§3.2.2 (Conditioning):** keep prose roughly as-is; the FiLM
  modulation-collapse story is load-bearing for the regime t-SNE
  figure we just added. Tighten only by removing the
  "four modulation parameters per block" / "six modulation
  parameters per block" parameter-counting if it doesn't earn its
  airtime. **Modest cut, ~30 words.**
- **§3.2.3 (EDM):** drop the
  $(c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}, c_{\text{noise}})$
  enumeration. Replace with: "EDM~\cite{karras2022elucidating}
  reframes diffusion training around a continuous noise scale
  $\sigma$ and wraps the denoiser with $\sigma$-dependent input,
  output, skip, and noise scalings chosen so the network's
  effective input/output magnitudes are order-unity at every
  $\sigma$ — the network's operating range stays comfortable
  everywhere. The single free hyperparameter is $\sigma_\text{data}$;
  we swept $\{0.5, 1.0, 1.5\}$ across v6 / v7 / v7\_b
  (Appendix~\ref{app:checkpoints})." **Cut ~80 words.**
- **§3.2.4 (Copula Transform):** drop the Sklar's theorem prose
  (§5.7 explains the copula in context anyway). Replace with: "The
  copula construction~\cite{sklar1959fonctions} per-feature
  Gaussianizes the data via the empirical CDF and probit, leaving
  the diffusion model to learn only the dependence structure; at
  sampling time the inverse map restores the marginal shape. For
  zero-inflated features (e.g.\ INTC's mid\_return) the empirical
  CDF has a step discontinuity that breaks the diffusion target;
  dequantization~\cite{theis2016note} fixes it by smearing the
  point mass over a small interval. v9 uses the
  copula+dequantization pipeline; the failure mode of the naive
  version (v8) is the case study of \S\ref{sec:copula_story}."
  **Cut ~120 words.**
- **§3.2.5 (Min-SNR):** compress to two sentences. "The
  Min-SNR-$\gamma$ scheme~\cite{hang2023efficient} caps the
  per-step loss weight to $\min(\text{SNR}(t), \gamma) /
  \text{SNR}(t)$; the published default is $\gamma=5$. We swept
  $\gamma\in\{1, 5\}$, found $\gamma=1$ over-corrects (Appendix
  \ref{app:phaseA} for the falsification log), and use $\gamma=5$
  in all twenty checkpoints." **Cut ~80 words.**

**Total:** ~310 words, roughly ~1 page.

**Verification.** Word count in §3.2 should drop by ~300; all four
subsubsections still parse end-to-end without forward-references to
removed math.

---

### Chunk D2 — §5.4 PCA Coverage compression

**Scope.** Replace `tab:pca_coverage_top` (lines 565-616 in current
file, ~50 lines) with a one-paragraph headline-numbers
restatement. Move the full table to appendix.

**Specific edits.**

- **Body:** §5.4 prose stays but loses the table. Replace the
  "Three regimes" enumeration that follows the table with one
  paragraph: "PCA coverage spans the full
  $0$--$85\%$ range across our twenty checkpoints (full table:
  Appendix~\ref{app:pca_coverage_full}). Three reference points
  carry the §5.4 story: (i) the copula-family generators dominate
  coverage --- v8 at $85.2\%$, v10\_cascade at $84.9\%$ --- by
  emitting hulls $2$--$2.6\times$ larger than real's, but their
  marginals inside that hull have collapsed (kurtosis $\sim 2$
  vs.\ real $755$); (ii) the v2/v5/v9 protagonist trio sits in
  the realistic-coverage band ($46$--$71\%$, hull areas
  $0.5$--$1.2\times$ real); (iii) the noclip family lands at
  $0\%$ coverage \emph{despite} producing the largest synth hulls
  in the study --- their projections are flung so far past real's
  region that the spatial overlap vanishes."
- **Appendix:** new section `\section{Full PCA Coverage Table}
  \label{app:pca_coverage_full}` containing the full
  20-checkpoint table verbatim. Slot in between
  `app:metric_correlations` and `app:ablation_full` (alphabetical
  isn't important; data flow is).

**Verification.** §5.4 word count drops; `tab:pca_coverage_top`
ref's still resolve (label preserved in the moved table).

---

### Chunk D3 — §5.5 Anti-Correlation compression

**Scope.** Keep `tab:correlation_subset` (the 6-row Spearman
extract). Compress the surrounding prose.

**Specific edits.**

- Cut both `\paragraph{Predictive $\leftrightarrow$ PCA: $+0.57$.}`
  and `\paragraph{Trade fraction $\leftrightarrow$ predictive:
  $-0.51$.}` paragraphs (~25 lines combined). Replace with one
  paragraph: "Two entries deserve emphasis. The
  predictive-vs-PCA correlation of $+0.57$ is sign-counterintuitive
  --- a generator that covers more of real's PC1--PC2 region
  should produce \emph{better} downstream training data, not
  worse. The mechanism is family membership: the copula family
  has wide PCA hulls AND catastrophic predictive scores; the
  noclip family has $0\%$ coverage AND the best predictive
  scores; the protagonist trio sits in between, exactly aligning
  with the $+0.57$ correlation. The trade-fraction-vs-predictive
  $-0.51$ is the same mechanism on the trade-fraction axis: the
  generators that emit realistic per-event composition (low trade
  fraction) are the same generators with the worst predictive
  scores (the copula family + v2\_remapped). The structural
  reading appears in \S\ref{sec:copula_lesson}."
- **Cut the "honest summary" closing paragraph** (lines 564-580 in
  current file). It restates §5.0's preamble and is redundant.
  Replace with a single sentence linking directly to §5.6:
  "Section~\ref{sec:res_ablation} disentangles which architectural
  toggle moves which axis."

**Verification.** §5.5 word count drops by ~250-300. Two paragraph
markers gone, one new paragraph + transition sentence.

---

### Chunk D4 — §5.7.1--§5.7.3 → Appendix

**Scope.** The biggest single cut. Move §5.7.1 (Zero-Inflation
Pathology), §5.7.2 (The Promise), §5.7.3 (The Failure: v8 z-Space
Collapse) to a new appendix. Compress to one bridge paragraph in
§5.7. Move `tab:v8_z_collapse` and `fig:z_space_collapse` to the
appendix too.

**Specific edits.**

- **In §5.7 preamble (between current §5.7 opening and new §5.7.4):**
  insert one bridge paragraph that absorbs the three subsubsections'
  punch lines:

  > "v8 was a naive copula attempt --- raw empirical CDF on each
  > feature, no dequantization --- and it failed. The mechanism is
  > simple: INTC's $\sim 89\%$ zero-return mass creates a step
  > discontinuity in the empirical CDF, which collapses an entire
  > $z$-band to a single value at training time and produces a
  > degenerate $z$-distribution at sampling time
  > (Appendix~\ref{app:copula_mechanism} details the four-step
  > argument and shows the v8 vs v8\_b vs v9 z-space histograms;
  > the headline is that v8's emitted $z$ for \texttt{mid\_return}
  > sat at $-2.55 \pm 0.30$ instead of the target $\mathcal{N}(0,1)$,
  > and v8\_b collapsed to a single point). v8 paid for the
  > failure across the board: G1 $= 8/15$ (worst non-degenerate
  > generator), excess kurtosis $= 2.6$ vs.\ real $755$, trade
  > fraction $= 0.009$ vs.\ real $0.048$. \S\ref{sec:copula_fix}
  > below shows how dequantization recovered most of the loss
  > without restoring predictive transferability ---
  > \S\ref{sec:copula_lesson} pins down why."
- **Drop:** §5.7.1 (current lines 1452-1494, ~42 lines), §5.7.2
  (current lines 1496-1514, ~18 lines), §5.7.3 (current lines
  1516-1567 minus the new figure block we added, ~52 lines).
- **Move:** `tab:v8_z_collapse` block into `app:copula_mechanism`.
  `fig:z_space_collapse` block also moves into
  `app:copula_mechanism`.
- **Renumber labels in body:** the §5.7.4 and §5.7.5 subsubsections
  become the only §5.7 children; their `\label{}`s
  (`sec:copula_fix`, `sec:copula_lesson`) stay so existing refs
  still resolve. The labels we lose are `sec:copula_zero_inflation`,
  `sec:copula_promise`, `sec:copula_failure`. Audit any `\ref{}` to
  these and rewrite to point at `app:copula_mechanism`.
- **New appendix:** `\section{Copula Failure Mechanism}
  \label{app:copula_mechanism}` containing the three
  subsubsections' content + table + figure. Slot after
  `app:copula_impl` (which is the existing implementation stub --
  the new mechanism appendix is a more substantive companion).

**Verification.** §5.7 body subsubsection count: 5 → 2 (only
`sec:copula_fix` and `sec:copula_lesson` remain). New appendix
`app:copula_mechanism` defined and referenced. All `\ref{}` to
removed labels rewritten.

---

### Chunk D5 — §6.2 Ranking Mechanism compression

**Scope.** Compress §6.2 to a two-sentence consistency-check
disclaimer. Move `tab:agent_truth` to existing `app:agents_impl`.

**Specific edits.**

- **Body:** replace §6.2's full prose (current lines 1856-1922,
  ~65 lines + the table) with two paragraphs:

  > "We deployed all four agents against each generator and against
  > a historical-replay baseline using the held-out validation
  > tapes. Per agent, per tape, we record terminal PnL, fill count,
  > fill rate, inventory variance, and max drawdown
  > (Appendix~\ref{app:agents_impl} for the per-agent statistics
  > on the real-data baseline).
  >
  > The natural temptation is to treat the historical-replay
  > ranking as ground truth. We resist it, for the same
  > non-responsiveness reason flagged in \S\ref{sec:backtest_problem}:
  > recorded markets do not react to the agent under test, so
  > every PnL number we report --- on every generator, including
  > the historical baseline --- is a non-impact measurement.
  > Generators evaluated in the same simulator inherit the same
  > blind spot. We treat agent rankings as a consistency check
  > across heterogeneous synthetic environments, not as a
  > pseudo-truth: the question we can answer is which patterns
  > survive the spread."
- **Move `tab:agent_truth`** into `app:agents_impl`. Update the
  one body cross-reference (in §6.4 / `sec:agent_implications`)
  to point at the appendix.

**Verification.** §6.2 word count drops by ~600. `tab:agent_truth`
defined exactly once (in appendix). All refs to it resolve.

---

### Chunk D6 — §6.3 + §6.4 compression

**Scope.** Cut §6.3's redundant-with-table opening paragraph.
Compress §6.4's three paragraph-bullets to a tight enumeration.

**Specific edits.**

- **§6.3:** keep `tab:rho_diff_spread` and the "headline observation"
  framing, but cut the paragraph that begins "The headline
  observation is that $\rdiff$ does not track..." (current lines
  ~1971-1982). Replace with a single sentence woven into the
  preceding text: "Whatever $\rdiff$ is measuring, it is not what
  \S\S\ref{sec:res_g1}--\ref{sec:res_pca} measured: v9 has the
  best marginal-distribution match in the study and is
  anti-aligned with truth at $\rdiff = -0.80$; v8 has the worst
  marginals and matches truth at $+0.80$." Cut ~10 lines.
- **§6.4:** keep the four `\paragraph{...}` blocks (A0 wins,
  A1$\approx$A2, A3 fragile, "What this is not") but compress
  each to ~3 sentences instead of the current 6-7 sentence
  paragraphs. The pattern: punch line + one-sentence empirical
  evidence + one-sentence pointer. Cut ~150 words across the
  three.
- **Update §6.4's opening to reference `fig:book_resilience`**
  (already done in chunk 15 of Work10_ReportC --- verify it's
  still there).

**Verification.** §6.3+§6.4 word count drops by ~300. All
paragraph headers preserved (each retains its `\paragraph{...}`).

---

### Chunk D7 — Appendix wiring + final audit

**Scope.** Build the new appendix sections from the moved content.
Final cross-ref + bib audit.

**Specific edits.**

1. **`\section{Copula Failure Mechanism}` (`app:copula_mechanism`)**
   inserted between existing `app:copula_impl` and `app:agents_impl`.
   Body:
   - Original §5.7.1 prose (the four-step CDF-discontinuity
     argument).
   - Original §5.7.2 prose (Sklar's theorem promise + expected
     kurtosis range).
   - Original §5.7.3 prose (v8 z-space collapse description).
   - `tab:v8_z_collapse` block (label preserved).
   - `fig:z_space_collapse` block (label preserved).
2. **`\section{Full PCA Coverage Table}` (`app:pca_coverage_full`)**
   inserted between `app:metric_correlations` and `app:ablation_full`.
   Body:
   - One-paragraph framing.
   - `tab:pca_coverage_top` block (label preserved).
3. **`app:agents_impl`** (existing empty stub) gets:
   - One-paragraph framing.
   - `tab:agent_truth` block (label preserved).
   - The current AS-family parameter / fill-simulator details
     (already promised in the body's §6.1 forward-pointer).
4. **Cross-ref audit:** run the same `diff` between `\ref{}` and
   `\label{}` extractions; confirm zero orphans. Pay special
   attention to refs to the removed §5.7 subsubsections:
   - `\ref{sec:copula_zero_inflation}` → rewrite to
     `\ref{app:copula_mechanism}`
   - `\ref{sec:copula_promise}` → rewrite to
     `\ref{app:copula_mechanism}`
   - `\ref{sec:copula_failure}` → rewrite to
     `\ref{app:copula_mechanism}`
5. **Bib audit:** no new citations needed (everything we move was
   already cited).

**Verification.** Final figure block count: still 21 (figures
themselves don't move count-wise; they migrate from body to App).
Final appendix count grows by 2 new sections + 1 populated stub.
Zero orphan refs.

---

## 2. Sectional impact summary (post-D)

| § | After Work10_ReportC | After Work10_ReportD |
|---|---|---|
| §3.2.2 (Conditioning) | full prose + figure | tightened prose + figure |
| §3.2.3 (EDM) | full prose | one paragraph |
| §3.2.4 (Copula transform) | full prose | one paragraph + forward-ref |
| §3.2.5 (Min-SNR) | full prose | two sentences |
| §5.4 (PCA Coverage) | prose + table | prose only + App. ref |
| §5.5 (Anti-correlation) | table + 2 emphasis paras + summary para | table + 1 compact para + transition |
| §5.7 preamble | prose | prose + bridge para absorbing §5.7.1-3 |
| §5.7.1 (Zero-Inflation) | full subsubsection | **moved to App. `app:copula_mechanism`** |
| §5.7.2 (Promise) | full subsubsection | **moved to App.** |
| §5.7.3 (Failure) | full subsubsection + table + figure | **moved to App.** |
| §5.7.4 (Fix) | full subsubsection + figure | unchanged |
| §5.7.5 (Lesson) | full subsubsection + figure | unchanged |
| §6.2 (Ranking Mechanism) | full prose + table | 2-paragraph disclaimer + App. ref |
| §6.3 (Different Rankings) | full prose + table | tightened prose + table |
| §6.4 (Stylized Facts) | 4 long paragraph-bullets | 4 compact paragraph-bullets |
| App. `app:copula_mechanism` | (didn't exist) | NEW — full mechanism walkthrough |
| App. `app:pca_coverage_full` | (didn't exist) | NEW — full PCA table |
| App. `app:agents_impl` | empty stub | populated — table + AS details |

---

## 3. Effort + ordering

Seven chunks. Suggested order:

| Chunk | Effort | Dependency |
|---|---|---|
| D1 — §3.2 compression | 25 min | none |
| D2 — §5.4 PCA compression + move table | 20 min | needs D7 (App. created) — or do inline |
| D3 — §5.5 prose compression | 15 min | none |
| D4 — §5.7.1-3 move to App. | 35 min | needs D7 (App. created) — or do inline |
| D5 — §6.2 compression + move table | 25 min | needs D7 (App. created) — or do inline |
| D6 — §6.3 + §6.4 compression | 20 min | none |
| D7 — appendix wiring + final audit | 30 min | depends on D2/D4/D5 |

**Resolution of the D2/D4/D5 ↔ D7 dependency:** simplest is to
**create the new appendix sections inline as part of D2 / D4 / D5**
(each chunk creates its target appendix section in the same edit as
the body cut). Then D7 is a final audit + cross-ref pass with no
content additions. This avoids the "broken intermediate state"
problem and lets D2/D4/D5 be self-contained.

**Suggested execution order** (groups for parallel review):

1. D1 + D3 (small Methods + Anti-correlation cuts; independent)
2. D2 + D5 (PCA + Ranking — both move table to App.; independent)
3. D4 (the heaviest cut; §5.7.1-3 → App.)
4. D6 (§6.3+§6.4 compression)
5. D7 (final audit)

**Total estimated effort:** ~3 hours of focused conversation, in
chunks of 15--35 minutes each.

---

## 4. What I will not touch

- §1, §2 (already lean)
- §3.1 DDPM (already minimal)
- §3.3 + §3.4 (Checkpoints, Eval Framework — already short)
- §5.0 + §5.1 (hero + protagonists — keep both figures)
- §5.2 (just rewired in chunks 4-5; no further cuts)
- §5.3 (already cut to prose-only in chunk 8)
- §5.6 (table + heatmap; appropriate length)
- §5.7.4 + §5.7.5 (the punchlines that justify §5.7's existence)
- §7 (cross-asset; already correctly sized)
- §8 (Discussion; already correctly sized)
- Abstract / AI Disclosure / Appendix stubs other than the three
  populated above.
- `references.bib` (no new citations needed)

---

## 5. Verification protocol per chunk

After each chunk:

1. `wc -w docs/final_report.tex` — confirm body word count
   dropped by the expected amount.
2. `grep -c "\\\\begin{figure}" docs/final_report.tex` — figures
   that moved should keep total count constant.
3. `diff <(refs) <(labels)` — confirm zero orphan references.
4. Read the affected section's prose top-to-bottom; confirm every
   forward-pointer to the appendix resolves to a label that
   actually exists.
5. User reviews diff and approves before moving on.

---

**Ready to execute.** Suggested first chunk: **D1 (§3.2 compression)**
or **D3 (§5.5 compression)** — both small, independent, no
appendix wiring. Let me know which to start with, or to take them
together as a single batch.
