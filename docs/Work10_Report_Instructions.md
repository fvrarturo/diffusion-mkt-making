# Report Drafting + Additional Analyses Prompt

You have two deliverables:
1. **`report_proposal.tex`** — A complete LaTeX document skeleton with section-by-section prose outlines, figure/table placeholders, and narrative guidance. NOT a finished report — a detailed structural blueprint that a human can fill in. Every section has: (a) what goes here, (b) which figures/tables, (c) the narrative arc, (d) approximate length.
2. **`Work8_additions.md`** — A prioritized list of additional code/analyses to run on existing INTC models before writing the final report. Each item has: what, why, how (script sketch), estimated runtime, and which report section it feeds.

---

## DELIVERABLE 1: `report_proposal.tex`

### Framing — Read This Carefully

The report is NOT "we built agents and tested them." The report IS:

**"We systematically developed and evaluated diffusion-based generators of synthetic limit order book data, pursuing statistical realism across multiple evaluation dimensions. We discovered that no single architecture optimizes distributional realism, temporal pattern transferability, and multivariate coverage simultaneously — a structural tradeoff inherent to MSE-based diffusion training on heavy-tailed multivariate financial data. As a secondary application, we tested whether these synthetic environments can stress-test market-making agents, revealing that agent-ranking metrics are fundamentally decoupled from distributional quality."**

The narrative is a JOURNEY of understanding, not a sequence of experiments. Present it as:
1. We wanted to generate realistic LOB data (motivation: backtesting is flawed)
2. Here's what LOB data looks like and why it's hard (data section — deep, pedagogical)
3. Here's how diffusion models work and the specific choices we made (architecture section — algorithmic, with diagrams)
4. Here's what happened when we systematically varied the architecture (results — the three-axis tension)
5. Here's what this means for anyone trying to generate synthetic financial data (discussion)
6. Oh and also, we tried using it for agent stress-testing, and learned something unexpected about evaluation metrics (the agent section — shorter, punchline-oriented)

The tone should be: technically rigorous, pedagogically clear, narratively driven. Every architectural choice gets an INTUITION paragraph before the math. Every result gets a "what this means" paragraph after the numbers. The reader should never see a table without understanding why they should care about each column.

### Course Requirements (15.458 Project F)

The report must contain: Title, Executive Summary (~250 words, non-technical), Project Definition, Scope, Data, Methods, Results, Discussion, References, AI Disclosure, Appendix. Length: 15-25 pages excluding appendix. Must be reproducible.

### LaTeX Setup

Use a clean academic template. Suggested:
```latex
\documentclass[11pt, letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx, subcaption}
\usepackage{booktabs, multirow}
\usepackage{hyperref, cleveref}
\usepackage{algorithm2e}
\usepackage{xcolor}
\usepackage{tcolorbox}
```

---

### Section-by-Section Structure

#### 0. TITLE + AUTHORS

**Title**: "Diffusion Models for Limit Order Book Generation: A Systematic Study of the Distributional Realism–Temporal Transferability Tradeoff"

Alternative: "Generating Realistic Limit Order Books with Diffusion Models: Architecture, Evaluation, and the Three-Axis Tension"

Authors: Arturo Favara, Nick Bernardini, Maxime Vallot. Course: 15.458 Financial Data Science and Computing, Spring 2026, MIT Sloan.

#### 1. EXECUTIVE SUMMARY (~250 words)

Write for a non-technical senior manager. Structure:
- One sentence: what we did (built diffusion models to generate synthetic market data)
- One sentence: why it matters (testing trading strategies on historical data is incomplete)
- Two sentences: how we did it (20 model variants, 3 evaluation frameworks, 3 stocks)
- Two sentences: what we found (the three-axis tension; no single architecture wins everything)
- One sentence: practical implication (practitioners must choose which quality dimension matters for their use case)

No jargon. No acronyms. No math.

#### 2. INTRODUCTION AND MOTIVATION (~2 pages)

**Narrative arc**: Start with the PROBLEM (backtesting on historical replay doesn't account for counterfactual market reactions), establish why GENERATIVE models are the right tool, narrow to DIFFUSION models specifically, then state the CONTRIBUTION.

**2.1 The Backtesting Problem**
- Historical replay is non-responsive: the market doesn't react to the agent's actions
- This makes backtesting systematically optimistic for strategies with market impact
- Quote the TRADES paper (Berti et al. 2025): "backtesting is inherently non-responsive since there is no way to measure the market impact"
- The alternative: generate SYNTHETIC market scenarios that are realistic but novel

**2.2 Why Diffusion Models**
- Brief history: GANs were tried first (Li et al. 2020, Coletta et al. 2021-2022) but suffer mode collapse
- Diffusion models (Ho et al. 2020): stable training, good mode coverage, flexible conditioning
- TRADES (Berti et al. 2025): first diffusion-based LOB generator, demonstrated on INTC and TSLA
- Our starting point: fine-tune a TRADES checkpoint on our own data

**2.3 Contribution**
State clearly:
1. We systematically evaluate 20 diffusion architectures across 5 design dimensions
2. We identify a three-way tension between distributional realism, temporal transferability, and multivariate coverage
3. We show this tension is structural to MSE-based diffusion training on heavy-tailed data
4. We demonstrate that agent-ranking metrics are decoupled from all three quality dimensions
5. We validate cross-asset (INTC, TSLA, SPY) to test generalizability

**Figures**: None in introduction. Maybe a teaser of Figure 10.1 (the bump chart) as a visual abstract.

#### 3. DATA (~3 pages — this should be DEEP and pedagogical)

**Narrative arc**: Walk the reader through what LOB data looks like, why it's unusual (zero-inflation, heavy tails, mixed discrete-continuous), and how we prepared it. A reader who has never seen TAQ data should understand it after this section.

**3.1 Limit Order Book Mechanics**
- What is a LOB? Orders, levels, bid/ask, spread, midprice
- Event types: quote updates vs trades
- INTC as a large-tick stock: spread pinned at 1 tick, 89% of events don't change the midprice
- Why INTC is interesting AND pathological for generative models (the zero-inflation)

```
FIGURE 3.1: "Anatomy of a Limit Order Book"
- Diagram showing bid/ask sides, levels, spread, midprice
- Annotate with the feature names used in the tensor (bid_dist, ask_dist, etc.)
- Could be a TikZ diagram or a clean matplotlib figure
```

**3.2 Data Source and Preparation**
- NYSE TAQ via WRDS: what it contains (quotes + trades at nanosecond resolution)
- LOBSTER reconstruction from TAQ: what our pipeline does
- Feature engineering: the 8-feature tensor (enumerate all 8, explain each)
  - delta_t, event_type (discrete — skipped by copula)
  - bid_dist, ask_dist (distance from mid to best bid/ask)
  - bid_sz, ask_sz (queue depth at best level)
  - trade_sz, trade_sign (trade features — zero for non-trade events)
  - mid_return (log return of midprice — the pathological feature)
- Normalization: z-score (mean/std from training set)
- Windowing: 256-event non-overlapping windows
- Stitching: how we chain 10 windows into 2,560-event pseudo-days

**3.3 The Zero-Inflation Problem**
- INTC mid_return: 89% exactly zero (the midprice doesn't change between most consecutive events)
- This creates a mixed distribution: point mass at 0 + continuous heavy-tailed component
- Conditional on nonzero: kurtosis ~56, std ~6.8e-5, skew ~-0.17
- Why this matters for generative models: naive normalization maps 89% of training data to a single point
- TSLA contrast: smaller tick, less zero-inflation, more continuous returns
- SPY contrast: ETF, index-arbitrage-driven, deep liquidity

```
FIGURE 3.2: "The Zero-Inflation Problem"
- Three panels:
  A: Histogram of INTC mid_return (show the massive spike at 0)
  B: Histogram of TSLA mid_return (less spike, more continuous)
  C: Conditional distribution of INTC mid_return | mid_return ≠ 0 (heavy-tailed)
- This figure motivates the entire copula/dequantization story later
```

```
FIGURE 3.3: "Feature Correlation Structure"
- Heatmap of Spearman correlation matrix for the 8 features (from real data)
- Annotate the key correlations: bid_sz-ask_sz (positive), bid_dist-ask_dist (negative, mechanical)
- Highlight OFI-return: Spearman = 0.071 (barely visible — this becomes important later)
```

**3.4 Train/Validation/Holdout Split**
- 60 training days, 10 validation days, 20 holdout days
- Regime labeling: base, high_vol, toxic, thin (explain each regime)
- Why regime conditioning matters: we want the generator to produce stress scenarios, not just average days

```
TABLE 3.1: Data Split Summary
| Split | Days | Events | Period |
```

**3.5 Stylized Facts of Financial Returns**
- Brief summary of Cont (2001) stylized facts that we test against
- Heavy tails (kurtosis >> 3)
- Volatility clustering (ACF of |returns| decays slowly)
- Absence of return autocorrelation (ACF of returns ≈ 0)
- Trade-sign persistence (positive lag-1 ACF of trade signs)
- Volume-volatility correlation
- Leverage effect
- These become our G1 validation criteria

#### 4. METHODS — DIFFUSION MODEL ARCHITECTURE (~4 pages)

**Narrative arc**: Start with DDPM basics (accessible to someone who knows ML but not diffusion), then introduce the architectural CHOICES as a design space with clear intuition for each, then explain our evaluation framework.

**4.1 Denoising Diffusion Probabilistic Models**
- Forward process: gradually add Gaussian noise to data
- Reverse process: learn to denoise
- The ε-prediction formulation (Ho et al. 2020)
- Training loss: MSE on predicted noise
- Keep this BRIEF — 1 page max, with references for depth

```
FIGURE 4.1: "The Diffusion Process"
- Diagram showing forward (noise addition) and reverse (denoising) processes
- Show a LOB event tensor being noised and denoised
- Annotate with the key variables: x_0, x_t, ε, ε_θ
```

```
ALGORITHM 4.1: "DDPM Training" (pseudocode)
- Sample x_0 from dataset
- Sample t ~ Uniform(1, T)
- Sample ε ~ N(0, I)
- Compute x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
- Compute loss = ||ε - ε_θ(x_t, t, c)||²
- Gradient step
```

**4.2 The Design Space**
Introduce the five architectural axes as a TABLE, then explain each:

```
TABLE 4.1: Architectural Design Space
| Axis | Options | What it controls |
| Prediction target | ε, v, x₀ | What the network learns to predict |
| Conditioning | FiLM, AdaLN-Zero | How regime information enters the network |
| Preconditioning | None, EDM | Input/output scaling per noise level |
| Data transform | z-score, copula, copula+dequant | How features are normalized |
| Loss weighting | Min-SNR γ={1,5}, uniform | Which noise levels get more training weight |
```

**4.2.1 Prediction Target: ε vs v**
- ε-prediction (Ho et al. 2020): predict the noise added to the data
  - INTUITION: "the network learns what was the noise that was added — it's trying to separate signal from noise"
  - Pro: mode-covering (produces the conditional mean, preserving bulk correlations)
  - Con: can't commit to extreme values (mode-covering compresses tails)
- v-prediction (Salimans & Ho 2022): predict a combination of signal and noise
  - INTUITION: "the network predicts a blend of what the data looks like and what the noise looks like — this gives it equal information at every noise level"
  - Pro: uniform loss variance across noise levels, better tail calibration
  - Con: signal-correlated target makes per-feature optimization independent, breaking cross-feature correlations

```
FIGURE 4.2: "ε vs v Prediction Targets"
- Two panels showing what the network target looks like at different noise levels t
- For ε: target is always N(0,1) regardless of t
- For v: target magnitude varies with t, capturing signal structure at low noise
- Annotate with the tradeoff: "mode-covering vs tail-committing"
```

**4.2.2 Conditioning: FiLM vs AdaLN-Zero**
- FiLM (Perez et al. 2018): affine modulation of hidden activations
  - γ and β learned from regime embedding
  - INTUITION: "scaling and shifting the activations based on what regime we're in"
- AdaLN-Zero (Peebles & Xie 2023): adaptive layer normalization with zero initialization
  - INTUITION: "the regime information modulates the normalization statistics, starting from no effect and learning how much to adjust"
- The FiLM collapse finding: Work5's F.2 diagnostic showed FiLM modulation was nearly identical across regimes, despite distinct embeddings (F.1b)

**4.2.3 Preconditioning: EDM**
- Karras et al. 2022: learned input/output scaling that normalizes the network's operating range
- σ_data parameter: must match the data's actual standard deviation
- INTUITION: "instead of making the network handle wildly different input magnitudes at different noise levels, we normalize everything so the network always operates in a comfortable range"
- The σ_data sensitivity finding: v6 (σ=0.5) vs v7 (σ=1.0) vs v7_b (σ=1.5)

**4.2.4 Data Transform: z-score vs Copula**
- z-score: subtract mean, divide by std. Standard. Preserves relative structure.
- Copula (Sklar 1959): transform each feature's marginal to Gaussian, train on the transformed data, invert at sampling
  - INTUITION: "we make every feature look Gaussian so the diffusion model only needs to learn the DEPENDENCE structure — heavy tails come from the inverse transform at the end"
  - The zero-inflation pathology: 89% point mass → CDF discontinuity → z-space collapse
  - Dequantization fix (Theis et al. 2016): spread point masses before CDF computation

```
FIGURE 4.3: "Copula Transform Pipeline"
- Flow diagram: x_original → empirical CDF → Φ⁻¹ → z_gaussian → diffusion training → sampling → Φ → inverse CDF → x_generated
- Show what happens at the zero-inflation point: massive CDF jump → z-collapse
- Show the dequantization fix: spread the jump → smooth CDF → distributed z
```

```
ALGORITHM 4.2: "Copula Transform with Dequantization" (pseudocode)
```

**4.2.5 Loss Weighting: Min-SNR-γ**
- Brief: weights the loss by min(SNR(t), γ)/SNR(t)
- γ=5: upweights mid-noise, downweights extremes
- γ=1: pushes weight toward high-noise steps
- INTUITION: "different noise levels are differently informative — this knob controls where the model pays most attention during training"

**4.3 The 20 Checkpoints**
Summary table of all trained models and their configurations.

```
TABLE 4.2: All 20 Checkpoints
| Model | Pred | Cond | EDM | Transform | γ | Clip | Notes |
| v2    | ε    | FiLM | No  | zscore    | 5 | 4    | Baseline |
| ...   | ...  | ...  | ... | ...       | . | .    | ...      |
```

**4.4 Evaluation Framework**
Explain the THREE evaluation axes:

**4.4.1 Distributional Realism (G1)**
- 16 stylized-fact checks from Cont (2001) + microstructure-specific tests
- Pass/fail per check, aggregate count
- This measures: "does the synthetic data look like real market data?"

**4.4.2 Temporal Pattern Transferability (Predictive Score)**
- Train LSTM on synth, test on real, measure MAE (following TRADES)
- This measures: "can you learn useful patterns from the synthetic data?"

**4.4.3 Multivariate Coverage (PCA)**
- Per-tape feature vector → PCA → convex hull overlap with real
- This measures: "does the synthetic data cover the full range of market states?"

**4.4.4 Agent-Based Ranking (ρ_diff) — and Why It Fails**
- Briefly introduce the Avellaneda-Stoikov agents
- Explain ρ_diff: Spearman correlation between synth and holdout agent rankings
- Flag immediately: "we will show in §7 that this metric is decoupled from all three quality dimensions"

```
FIGURE 4.4: "Three Evaluation Axes"
- Diagram showing the three axes as a triangle
- Annotate each vertex with what it measures and example metrics
```

#### 5. RESULTS — THE THREE-AXIS TENSION (~5 pages, the core of the paper)

**Narrative arc**: Present the three primary models for the main story (v2, v5, v9), show the tension, then use the full 20-checkpoint ablation to explain WHY.

**5.1 Three Representative Generators**

Pick v2, v5, v9 as the three protagonists:
- v2 (ε-pred, FiLM, z-score): the "safe baseline" — good temporal patterns, weak tails
- v5 (v-pred, AdaLN, z-score): the "heavy-tail champion" — kurtosis 1256, but breaks correlations
- v9 (ε-pred, FiLM, copula+dequant): the "distributional realist" — perfect trade fraction, best volatility clustering, but poor temporal transferability

```
TABLE 5.1: Three Representative Generators — Headline Metrics
(Extract the relevant rows from the multi-axis comparison table)
```

```
FIGURE 5.1: Hero Figure — Three-Axis Evaluation Space
(Figure 1.1 from the report_figures prompt — ternary plot or 2D with size)
```

**5.2 Distributional Realism Results**
- G1 matrix (heatmap) — which stylized facts does each model pass?
- Return distribution comparison (log-log CDF) — show tail behavior
- Trade fraction comparison — the most dramatic difference across models
- Per-feature violin plots for the three protagonists

```
FIGURE 5.2: G1 Pass/Fail Heatmap (Figure 2.1)
FIGURE 5.3: Return Tail Comparison (Figure 2.3)
FIGURE 5.4: Trade Fraction Bar Chart (Figure 2.4)
TABLE 5.2: Full G1 Results for v2, v5, v9 (with real reference)
```

**5.3 Temporal Transferability Results**
- Predictive score table (all models)
- The surprise: scale-exploded models (v3_e9_noclip at 4×) beat scale-matched models (v9 at 106×)
- Why: temporal PATTERNS transfer regardless of scale; the LSTM learns directional relationships
- ACF comparison showing temporal structure preservation

```
FIGURE 5.5: Predictive Score Bar Chart (Figure 3.1)
FIGURE 5.6: ACF Grid — Real vs v2 vs v9 (Figure 7.2)
TABLE 5.3: Predictive Score — All Models
```

**5.4 Multivariate Coverage Results**
- PCA scatter with convex hulls for key models
- v8 (copula, no dequant) has 85% coverage despite 8/15 G1 — coverage ≠ realism
- v2_noclip has 0% coverage despite 4× predictive score

```
FIGURE 5.7: PCA Coverage (Figure 4.1)
```

**5.5 The Anti-Correlation**
- Scatter matrix showing pairwise anti-correlation between axes
- Spearman ρ between evaluation metrics across 20 checkpoints
- This is NOT cherry-picked — it's systematic across every architectural axis

```
FIGURE 5.8: Pairwise Anti-Correlation Scatter (Figure 1.2)
TABLE 5.4: Metric Correlation Matrix (Table C from report_figures)
```

**5.6 Architectural Ablation: What Drives What**
- The ablation heatmap: each architectural choice → each metric
- Key findings (one paragraph each):
  - ε→v: improves kurtosis, degrades predictive score
  - Removing x0_clip: dramatically improves predictive, destroys G1
  - Adding copula: improves PCA coverage + trade fraction, degrades predictive
  - FiLM→AdaLN: negligible effect on most metrics, some interaction with pred target

```
FIGURE 5.9: Ablation Impact Heatmap (Figure 5.1)
FIGURE 5.10: The x0_clip Effect (Figure 5.2)
```

#### 6. THE COPULA STORY (~2 pages — a self-contained case study)

**Narrative arc**: Present the copula intervention as a case study in "fixing one thing breaks another." This is the most pedagogically rich section.

**6.1 The Promise**
- Copula theory: separate marginals from dependence structure
- Prediction: heavy tails from inverse CDF, correlations preserved by Spearman invariance
- Expected outcome: kurtosis 400-800, ρ_diff ≥ +0.40

**6.2 The Failure (v8)**
- z-space collapse: mid_return z-mean = -2.55, z-std = 0.30
- Root cause: CDF discontinuity at the 89% point mass
- Kurtosis: 2.6 (not 400-800)

**6.3 The Fix (v9 — Dequantization)**
- Standard technique from normalizing flows (Theis et al. 2016)
- Spread the point mass uniformly before CDF computation
- Result: frac_zero recovered (0.99 vs real 0.89), kurtosis 129, trade fraction 4.9%

**6.4 The Deeper Lesson**
- v9 has excellent distributional realism but poor temporal transferability
- The copula round-trip (Gaussianize → train → sample → inverse) adds enough noise to scramble the temporal micro-patterns the predictive LSTM relies on
- Spearman invariance is mathematically correct but economically insufficient: agents and LSTMs operate on absolute thresholds, not ranks
- v2_remapped (post-hoc quantile mapping of v2) proved this: perfect Spearman preservation, 265× predictive score, ρ_diff flipped from +0.40 to -0.40

```
FIGURE 6.1: z-Space Before/After Dequantization (Figure 6.1)
FIGURE 6.2: The Zero-Inflation CDF Problem (Figure 6.2)
FIGURE 6.3: v2 vs v9 Metric Comparison (Figure 6.3)
```

#### 7. AGENT-BASED STRESS TESTING (~2 pages — secondary, punchline-oriented)

**Narrative arc**: "We also tested whether generators can stress-test agents. The answer is nuanced: the ranking metric (ρ_diff) is decoupled from distributional quality, for a specific and interesting reason."

**7.1 Agent Design**
- Brief Avellaneda-Stoikov framework (1 paragraph + equation)
- The 4 agents: A0 (constant), A1 (AS baseline), A2 (AS+OFI), A3 (AS+VPIN)
- Fill simulator: how agent orders get matched against the synthetic book

**7.2 The Ranking Mechanism**
- A1 and A2 are economically indistinguishable: effect size < 0.3 on every metric
- The "correct" A2>A1 ordering is driven by fill-rate differences, not return prediction
- OFI→return R² = 0.0001 — the signal A2 uses is noise-level

**7.3 ρ_diff Is Decoupled From Quality**
- v8 (worst marginals): ρ_diff = +0.80
- v9 (best marginals): ρ_diff = -0.80
- v2_remapped (perfect Spearman preservation): ρ_diff = -0.40
- Show scatter: ρ_diff vs G1, ρ_diff vs Predictive — no correlation

**7.4 Implications**
- Agent-ranking evaluation requires agents with LARGE, STABLE performance gaps
- Our 4 agents are too similar to produce a meaningful ranking signal
- The evaluation framework (ρ_diff) is measuring fill-timing noise, not generator quality
- This is a FINDING, not a failure — it tells practitioners what NOT to do with synthetic stress tests

```
FIGURE 7.1: Agent PnL Overlap (Figure 8.1)
FIGURE 7.2: ρ_diff vs Everything (Figure 8.2)
TABLE 7.1: Agent Decomposition — What Separates A1 from A2
```

#### 8. CROSS-ASSET VALIDATION (~1-2 pages — placeholder for TSLA/SPY results)

**8.1 Why Cross-Asset**
- Does the three-axis tension persist on TSLA (small-tick) and SPY (ETF)?
- If yes: the tension is structural to diffusion-based LOB generation
- If no: the tension is INTC-specific (zero-inflation artifact)

**8.2 TSLA Results**
```
TABLE 8.1: v2 vs v5 on TSLA — Three-Axis Comparison
(Placeholder — fill when TSLA runs complete)
```

**8.3 SPY Results**
```
TABLE 8.2: v2 vs v5 on SPY — Three-Axis Comparison
(Placeholder — fill when SPY runs complete)
```

**8.4 Cross-Asset Findings**
- Does v2 still beat v5 on predictive score? Does v5 still beat on kurtosis?
- Does trade fraction differ across tickers?
- Plot: the three-axis scatter with INTC/TSLA/SPY points for v2 and v5

#### 9. DISCUSSION (~1.5 pages)

**9.1 The Three-Axis Tension as a Structural Result**
- Why does this happen? MSE is per-element, per-timestep → can't simultaneously optimize marginal shape AND cross-feature/temporal correlations
- This is not specific to LOB data — it should arise in any heavy-tailed multivariate diffusion setting
- Connection to broader diffusion model literature: mode-covering vs mode-seeking behavior

**9.2 Practical Implications**
- For practitioners choosing a generator: match the evaluation axis to your use case
  - If you need to train downstream ML models: prioritize temporal transferability (→ use noclip / scale-exploded)
  - If you need distributional realism for regulatory stress tests: prioritize G1 (→ use v2 or v5)
  - If you need diverse scenario coverage: prioritize PCA (→ use copula)
- No one-size-fits-all answer

**9.3 Limitations**
- Single backbone architecture (TRADES transformer) — would the tension persist with U-Net, SSM, etc.?
- INTC-dominant (zero-inflation is pathological) — TSLA/SPY results may qualify
- 20 holdout days insufficient for stable agent rankings
- Predictive score depends on LSTM architecture choice

**9.4 Future Work**
- Autoregressive diffusion (event-by-event generation) might avoid the marginals-vs-joints tension
- Mixture of experts: ε-prediction for bulk + v-prediction head for tails
- Direct distributional loss (MMD, Wasserstein) instead of MSE
- Larger evaluation: more agents with diverse strategies, longer holdout periods

#### 10. REFERENCES

Include at minimum: Ho et al. 2020 (DDPM), Song et al. 2020 (DDIM), Salimans & Ho 2022 (v-prediction), Karras et al. 2022 (EDM), Peebles & Xie 2023 (DiT/AdaLN), Perez et al. 2018 (FiLM), Berti et al. 2025 (TRADES), Cont 2001 (stylized facts), Cont-Kukanov-Stoikov 2014 (OFI), Avellaneda-Stoikov 2008, Sklar 1959 (copula), Theis et al. 2016 (dequantization), Hang et al. 2023 (Min-SNR), Nichol & Dhariwal 2021 (improved DDPM).

#### 11. AI DISCLOSURE

"We used Claude (Anthropic) extensively as a research collaborator throughout this project: for architectural design, experiment planning, diagnostic analysis, code generation, and report drafting. All final decisions, code execution, experimental results, and interpretations are our own. Claude's recommendations were frequently wrong (e.g., the copula kurtosis prediction) and we report negative results honestly."

#### APPENDIX

**A. Full G1 Value Matrix** (the 16×20 table with actual values, not just pass/fail)
**B. All 20 Checkpoint Configurations** (detailed hyperparameters)
**C. Per-Feature Deep Diagnostic** (the full Section A from 60_deep_diagnostic.py)
**D. Copula Transform Implementation Details** (the dequantization algorithm, ε computation)
**E. Agent Implementation Details** (Avellaneda-Stoikov parameters, fill simulator)
**F. Predictive Score Training Details** (LSTM architecture, training curves)
**G. Additional Mid-Price Traces** (all models, not just the 4 in the main text)
**H. Volume-Volatility Correlation Distributions** (the KDE plots)
**I. Reproducer Commands** (exact cluster commands for each checkpoint)

---

## DELIVERABLE 2: `Work8_additions.md`

This file identifies analyses/code that would strengthen the report but haven't been run yet. Each item specifies what, why, which report section it feeds, implementation sketch, and estimated runtime.

### Priority 1 — Must-have before writing (run today)

**Addition 1: Full deep diagnostic on v5 (the v-prediction protagonist)**

We have deep diagnostic data for v2, v7_b, v8, v8_b. We're missing v5 — one of the three main story models. Need per-feature stats, cross-feature correlations, temporal ACF, window-level stats.

- Feeds: §5.2, §5.3, Table 5.2, Figure 5.6
- Script: `python scripts/60_deep_diagnostic.py --real-dir data/processed/val --synth-dirs .../synthetic_v5_stitched/INTC --labels v5 --out results/deep_diagnostic_v5`
- Runtime: ~5 min

**Addition 2: Conditional-on-trade return stats for all 20 models**

The conditional_returns.csv from ranking_mechanism only has v2/v8/v9/v9_b. Need it for all models to show the frac_zero and conditional kurtosis columns in the full comparison table.

- Feeds: §3.3, Table 5.1, the copula story
- Script: extend `61_ranking_mechanism.py` or write a simple loop
- Runtime: ~10 min

**Addition 3: Per-regime validation for v2, v5, v9**

We have regime-conditional stats from the deep diagnostic but not regime-specific G1. Do base/high_vol/toxic/thin produce different pass rates? This tells us whether regime conditioning is working.

- Feeds: §5.2, §4.2.2 (conditioning effectiveness)
- Script: run the G1 validation separately on each regime's parquets
- Runtime: ~20 min (4 regimes × 3 models × ~2 min each)

**Addition 4: Real data stylized fact figures (standalone)**

The report needs standalone figures showing real INTC's stylized facts (not comparisons — just real data). These go in §3.5 to establish what the generator should reproduce.

- Return distribution histogram (the heavy tails)
- ACF of |returns| (the slow decay)
- ACF of returns (the near-zero / slightly negative lag-1)
- Trade-sign ACF (the positive persistence)
- Spread distribution (the 95% one-tick concentration)
- Volume intraday pattern (the U-shape)

- Feeds: §3.5
- Script: simple matplotlib script loading real val data
- Runtime: ~3 min

### Priority 2 — Highly valuable (run this week)

**Addition 5: Guidance weight sweep on v2**

How does classifier-free guidance weight w affect the three axes? Sweep w ∈ {0, 0.5, 1.0, 2.0, 4.0, 8.0} on v2's checkpoint. Sample 50 tapes per w, compute G1 subset + predictive score + PCA.

- Feeds: §4.2 (conditioning), §9.4 (future work), Appendix
- Script: modify sampling to loop over w, then run validation on each
- Runtime: ~2h (6 w values × ~20 min sampling+validation each)

**Addition 6: Per-timestep denoising quality for v2, v5, v9**

Take real validation windows, noise to t ∈ {10, 50, 100, 200, 500, 800}, one-step denoise, compare x̂₀ to x₀. Shows WHERE in the denoising process each architecture succeeds/fails.

- Feeds: §4.2.1 (ε vs v intuition), §5.6 (ablation), Discussion
- Script: load checkpoint, forward-pass real data at each t, compute per-feature reconstruction MSE
- Runtime: ~30 min (needs GPU for forward passes)

**Addition 7: Regime embedding visualization**

t-SNE or UMAP of the 4 regime embeddings for each model. Shows whether the embeddings are distinct and how they cluster relative to each other.

- Feeds: §4.2.2 (conditioning)
- Script: load checkpoint, extract embedding weights, project with t-SNE
- Runtime: ~5 min per model

**Addition 8: Training curve comparison**

Plot training loss and validation loss curves for v2, v5, v9. Show when each model converges, whether they overfit, how patience/early-stopping interacts.

- Feeds: §4.3, Appendix B
- Script: read the train.csv logs for each checkpoint
- Runtime: ~2 min (just plotting)

### Priority 3 — Nice-to-have (if time permits)

**Addition 9: Synthetic day walkthrough**

Pick one synthetic tape from v2 and one from v9. Walk through the first 100 events showing the raw features, the decoded LOB state, and the agent's decisions at each step. This is a pedagogical exhibit showing what the synthetic data actually looks like at the event level.

- Feeds: §3.1 (data understanding), §7.1 (agent design)
- Script: load one parquet, decode, simulate agent, print annotated event log
- Runtime: ~5 min

**Addition 10: Robustness of predictive score to LSTM size**

The predictive score depends on the LSTM architecture. Run it with hidden_dim ∈ {32, 64, 128} to check that the model RANKING is stable even if the absolute MAE changes.

- Feeds: §9.3 (limitations)
- Script: modify `63_predictive_score.py` to loop over hidden_dim
- Runtime: ~1h (3 architectures × 20 models)

**Addition 11: Correlation matrix comparison figure**

For v2, v5, v9: show the 8×8 Pearson correlation matrix side-by-side with real. Highlight the cells where synth deviates most from real (red border). This directly visualizes "which cross-feature relationships break."

- Feeds: §5.5, §6.4 (copula story)
- Script: read from correlation_comparison.csv, produce heatmap grid
- Runtime: ~3 min

**Addition 12: Spread dynamics deep dive**

INTC's spread is pinned at 1 tick 95% of the time. When it widens, what do the models produce? Conditional analysis: compute mean(bid_sz), mean(ask_sz), mean(|mid_return|) conditioned on spread > 1 tick. Are the models realistic in the non-modal spread regime?

- Feeds: §5.2 (distributional realism), §3.1 (INTC microstructure)
- Script: filter parquets for spread > 0.01, compute conditional stats
- Runtime: ~5 min

**Addition 13: Event-type transition matrix**

In real data, what fraction of trade events are followed by another trade vs a quote update? Compare to synthetic. This captures a temporal property the ACF doesn't: the SEQUENCING of event types.

- Feeds: §5.3 (temporal structure), §3.2 (feature engineering)
- Script: compute 2×2 transition matrix (trade→trade, trade→quote, quote→trade, quote→quote) for real and each model
- Runtime: ~3 min

**Addition 14: Intraday volume pattern**

Real markets have a U-shaped volume pattern (high at open/close, low at midday). Do our synthetic tapes reproduce this? Plot average trade_sz per time-of-day bucket.

- Feeds: §3.5 (stylized facts), §5.2
- Note: synthetic tapes may not have realistic timestamps — if ts_ns is just uniformly spaced, this metric doesn't apply. Check first.
- Runtime: ~3 min

Each addition above should note in its output which report section and figure/table number it feeds, so the report-writing agent can pull it in directly.