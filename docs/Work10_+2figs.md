**Addition 1: Event-level tape walkthrough (the "what does this data actually look like" figure)**

This is the figure that doesn't exist anywhere in your current suite and should. Every metric in the paper is an aggregate — kurtosis, ACF, trade fraction. No figure shows the reader what 500 consecutive LOB events actually look like at the event level. For anyone outside HFT, the data is completely abstract.

The figure: 4 columns (real, v2, v5, v9) × 4 rows (mid_price, queue depth bid+ask, spread, trade markers). X-axis is event index 0–500. Each panel is a step-function time series at event resolution.

What the reader instantly sees without any statistics:

- **Real column**: mid_price is a staircase — flat for long stretches (the 89% zeros), then sudden jumps. Trades are rare sparse dots. Spread is pinned at one tick with occasional brief widenings. Queue sizes drift smoothly with slow mean-reversion.

- **v2 column**: mid_price drifts continuously (no flat stretches — the frac_zero=0 artifact made visible). Trades are everywhere (55% of events). Spread is also pinned but the continuous drift is visually wrong. Queue sizes look similar to real (v2's strength).

- **v5 column**:  Similar to v5

- **v9 column**: mid_price has the flat-then-jump structure (frac_zero=0.99). Trades are rare (matching real). But the jumps may look different — more regular, less clustered. Queue sizes may be less smooth.

This figure does more pedagogical work than any bar chart: it shows the reader that v2/v5 generate a fundamentally different EVENT PROCESS than real (continuous drift + constant trading), while v9 gets the event structure right (rare trades, price pinning) but may sacrifice the temporal micro-patterns. The three-axis tension becomes visible at the event level.

Implementation: load one real val tape, one v2 base tape, one v9 base tape. Plot 200 events from the middle of each (avoid edge effects from stitching). Four shared-y-range rows, three columns, tight layout. ~40 lines of matplotlib. No cluster computation needed.

---

**Addition 2: Book resilience around trade events (the "does the synthetic book behave like a real book" figure)**

This is a canonical microstructure analysis (Bouchaud et al. 2009, Large 2007, the LOB resiliency literature). When a trade executes, it consumes some queue depth. In a real market, the book replenishes — new limit orders arrive to refill the depleted queue. This "resilience" is a defining characteristic of healthy markets and a key property that market-making agents depend on.

The figure: event-study design. Define t=0 as each trade event. For all trades in a dataset, compute the average bid_sz, ask_sz, and spread at events t-30 through t+30 around the trade. Plot three curves (real, v2, v5, v9) on the same axes, with t=0 marked.

What the reader should see:

- **Real**: bid_sz (or ask_sz, depending on trade side) drops sharply at t=0, then gradually recovers over the next 10-20 events. Spread may widen at t=0 then compress. This is the queue-replenishment / resilience pattern.

- **v2/v5**: since 55% of events are trades, the "pre-trade" and "post-trade" windows are contaminated with other trades. The dip-and-recovery pattern will be smeared or absent. The queue dynamics around a v2 "trade" look nothing like real because the inter-trade spacing is wrong.

- **v9**: since trade fraction matches real (5%), the inter-trade spacing is correct. The question is whether the dip-and-recovery pattern appears. If it does, v9 captures a dynamic property of the book that no aggregate metric tests for. If it doesn't, that's a specific failure mode worth reporting.

This is powerful for three reasons. First, it tests a DYNAMIC property — not a marginal or a correlation, but the book's response to an event. No metric in the current G1 suite captures this. Second, it directly relates to market-making agent behavior: agents post limit orders that sit in the queue, and queue replenishment determines whether those orders get "protected" by new arrivals. Third, it's visually striking — the dip-and-recovery is one of those patterns that, once you see it, you immediately understand what's happening.

Implementation: for each dataset (real, v2, v5, v9), identify all events where trade_sz > 0. For each trade, extract bid_sz and ask_sz at t-30...t+30. Average across all trades. Plot with confidence bands (±1 std across trades). ~60 lines. Need to handle the trade-side asymmetry (buy trades deplete ask, sell trades deplete bid) — simplest approach is to align by trade_sign: for buy trades plot ask_sz (which should drop), for sell trades plot bid_sz.

This is the figure that a market microstructure professor sees and says "they actually understand the data."