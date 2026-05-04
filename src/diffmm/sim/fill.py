"""Top-of-book fill simulator (proposal Algorithm 5).

Single piece of code that runs identically on real and synthetic event tapes
— that's the symmetry the central hypothesis test depends on. Any
approximation made here (queue position, fill rule) is applied uniformly to
both inputs, so directionally fair across the comparison.

Per-event cycle:
  1. Agent observes current LOBState and posts (bid_quote, ask_quote).
  2. Stale-quote check: if mid moved through our quote, cancel and skip the
     event for fill purposes.
  3. Trade events check: if a trade hit at-or-through our quote price, and
     we have queue priority, fill us at our quote price.
  4. Step the simulator forward by one event.

Conservative queue model: we estimate Q_ahead = total displayed size at our
quote price level at the moment we post. We need (incoming_volume - Q_ahead)
of one-sided trade flow to fill us. Optimistic and pessimistic variants are
trivial to add for sensitivity analysis (action plan §3.5 deepening).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import polars as pl

from ..agents.base import LOBState, MarketMakingAgent, Quote
from .pnl import Position


# ─── data structures ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FillConfig:
    """Knobs that change behavior of the fill model."""
    queue_position_mode: str = "conservative"   # "conservative" | "optimistic" | "pessimistic"
    requote_on_mid_move: bool = True
    inventory_limit: int = 5000                 # hard cap; agent's soft cap is in AgentParams
    tick_size_usd: float = 0.01


@dataclass
class _PostedQuote:
    """Internal: a quote currently resting on the book."""
    side: str               # "bid" | "ask"
    price: float
    size_remaining: int     # decremented as we get partial fills
    queue_ahead: int        # estimated Q_ahead at posting time


@dataclass
class FillEvent:
    """One fill recorded by the simulator."""
    ts_ns: int
    side: str               # "bid" | "ask"
    price: float
    qty: int
    mid_at_fill: float


@dataclass
class BacktestResult:
    """Output of one backtest run on one tape."""
    agent_name: str
    n_events: int
    n_fills: int
    terminal_pnl: float
    intraday_pnl: list[float] = field(default_factory=list)        # mark-to-market at each event
    inventory_series: list[int] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)

    def fill_rate(self, n_quotes_posted: int) -> float:
        if n_quotes_posted == 0:
            return 0.0
        return self.n_fills / n_quotes_posted


# ─── simulator ─────────────────────────────────────────────────────────

class FillSimulator:
    """Runs an agent over a single event-tape DataFrame."""

    def __init__(self, config: FillConfig | None = None):
        self.cfg = config or FillConfig()

    def run(
        self,
        agent: MarketMakingAgent,
        tape: pl.DataFrame,
        *,
        session_end_ts_ns: int | None = None,
    ) -> BacktestResult:
        """Replay `tape` event-by-event, sending each event to `agent` and
        executing fills against subsequent trade events.

        `tape` must conform to the canonical schema (already validated upstream).
        """
        position = Position()
        # Two posted quotes (one per side) at any time.
        active_bid: _PostedQuote | None = None
        active_ask: _PostedQuote | None = None
        n_quotes_posted = 0

        result = BacktestResult(agent_name=agent.name, n_events=tape.height, n_fills=0, terminal_pnl=0.0)

        # Determine session end for time-remaining computation.
        ts_arr = tape["ts_ns"].to_numpy()
        if session_end_ts_ns is None:
            session_end_ts_ns = int(ts_arr[-1])

        # Pre-extract columns for speed (Polars row-iteration is slow).
        bid_px_arr = tape["bid_px"].to_numpy()
        ask_px_arr = tape["ask_px"].to_numpy()
        bid_sz_arr = tape["bid_sz"].to_numpy()
        ask_sz_arr = tape["ask_sz"].to_numpy()
        mid_arr = tape["mid"].to_numpy()
        spread_arr = tape["spread"].to_numpy()
        event_type_arr = tape["event_type"].to_numpy()
        trade_px_arr = tape["trade_px"].to_numpy()
        trade_sz_arr = tape["trade_sz"].to_numpy()
        trade_sign_arr = tape["trade_sign"].to_numpy()
        is_lit_arr = tape["is_lit"].to_numpy()
        regime_label_arr = tape["regime_label"].to_numpy()
        c_vol_arr = tape["c_vol"].to_numpy()
        c_vpin_arr = tape["c_vpin"].to_numpy()
        c_imb_arr = tape["c_imb"].to_numpy()

        for i in range(tape.height):
            ts = int(ts_arr[i])
            mid = float(mid_arr[i])
            spread = float(spread_arr[i])
            time_remaining = max(0.0, (session_end_ts_ns - ts) * 1e-9)

            # 1. Try to fill any active quotes against this event (if it's a trade).
            if str(event_type_arr[i]) == "trade":
                trade_px = float(trade_px_arr[i]) if trade_px_arr[i] is not None else None
                trade_sz = int(trade_sz_arr[i]) if trade_sz_arr[i] is not None else 0
                trade_sign = int(trade_sign_arr[i]) if trade_sign_arr[i] is not None else 0
                if trade_px is not None and trade_sz > 0:
                    # Sell-initiated trade (sign = -1) hits the bid side; if its
                    # price ≤ our resting bid_px, our bid may fill.
                    if active_bid is not None and trade_sign == -1 and trade_px <= active_bid.price:
                        qty = self._consume(active_bid, trade_sz)
                        if qty > 0:
                            position.buy(qty, active_bid.price)
                            result.fills.append(FillEvent(ts, "bid", active_bid.price, qty, mid))
                            result.n_fills += 1
                            if active_bid.size_remaining <= 0:
                                active_bid = None
                    # Buy-initiated trade (sign = +1) hits the ask side; if its
                    # price ≥ our resting ask_px, our ask may fill.
                    if active_ask is not None and trade_sign == +1 and trade_px >= active_ask.price:
                        qty = self._consume(active_ask, trade_sz)
                        if qty > 0:
                            position.sell(qty, active_ask.price)
                            result.fills.append(FillEvent(ts, "ask", active_ask.price, qty, mid))
                            result.n_fills += 1
                            if active_ask.size_remaining <= 0:
                                active_ask = None

            # 2. Stale-quote removal: cancel a quote only if the market has
            # moved past it on the OPPOSITE side (i.e. our bid is at-or-above
            # the new best ask, or our ask is at-or-below the new best bid).
            # Quoting inside the spread is the normal MM behavior and must NOT
            # trigger cancellation.
            if self.cfg.requote_on_mid_move:
                cur_bid_px = float(bid_px_arr[i])
                cur_ask_px = float(ask_px_arr[i])
                if active_bid is not None and active_bid.price >= cur_ask_px:
                    active_bid = None
                if active_ask is not None and active_ask.price <= cur_bid_px:
                    active_ask = None

            # 3. Hard inventory cap.
            if position.inventory >= self.cfg.inventory_limit:
                active_bid = None      # stop buying
            if position.inventory <= -self.cfg.inventory_limit:
                active_ask = None      # stop selling

            # 4. Build the LOBState the agent sees and ask for new quotes.
            state = LOBState(
                ts_ns=ts,
                bid_px=float(bid_px_arr[i]),
                ask_px=float(ask_px_arr[i]),
                bid_sz=int(bid_sz_arr[i]),
                ask_sz=int(ask_sz_arr[i]),
                mid=mid,
                spread=spread,
                inventory=position.inventory,
                cash=position.cash,
                time_remaining=time_remaining,
                regime_label=str(regime_label_arr[i]),
                c_vol=int(c_vol_arr[i]),
                c_vpin=int(c_vpin_arr[i]),
                c_imb=int(c_imb_arr[i]),
            )
            quote = agent.on_event(state)

            # 5. Activate the agent's new quotes (replacing any stale ones).
            if quote.bid_px is not None and quote.bid_sz > 0:
                # Q_ahead estimate: book size at our price (or 0 if we're inside the spread).
                queue_ahead = self._queue_ahead("bid", quote.bid_px, state)
                active_bid = _PostedQuote("bid", quote.bid_px, quote.bid_sz, queue_ahead)
                n_quotes_posted += 1
            elif quote.bid_px is None:
                active_bid = None
            if quote.ask_px is not None and quote.ask_sz > 0:
                queue_ahead = self._queue_ahead("ask", quote.ask_px, state)
                active_ask = _PostedQuote("ask", quote.ask_px, quote.ask_sz, queue_ahead)
                n_quotes_posted += 1
            elif quote.ask_px is None:
                active_ask = None

            # 6. Bookkeeping.
            result.intraday_pnl.append(position.mark_to_market(mid))
            result.inventory_series.append(position.inventory)

        # 7. Terminal liquidation at the last observed mid.
        result.terminal_pnl = position.liquidate(float(mid_arr[-1]))
        return result

    def _consume(self, posted: _PostedQuote, incoming_volume: int) -> int:
        """Apply one trade event to a posted quote. Returns shares filled."""
        if self.cfg.queue_position_mode == "optimistic":
            consumed_by_queue = 0
        elif self.cfg.queue_position_mode == "pessimistic":
            return 0           # only crossing fills, never queue-jumping
        else:  # conservative
            consumed_by_queue = min(posted.queue_ahead, incoming_volume)
        remaining_volume = max(0, incoming_volume - consumed_by_queue)
        posted.queue_ahead = max(0, posted.queue_ahead - incoming_volume)
        fill_qty = min(remaining_volume, posted.size_remaining)
        posted.size_remaining -= fill_qty
        return fill_qty

    def _queue_ahead(self, side: str, price: float, state: LOBState) -> int:
        """Conservative estimate of shares ahead in the queue at posting time.

        Three regimes per side:
          - Improving the market (better price than current best) → 0
            (we're alone at a new top-of-book level).
          - Joining at the current best (within half-tick) → existing book size.
          - Quoting behind the market (worse than current best) → existing book size,
            conservative assumption that the book has at least that depth.
        """
        if self.cfg.queue_position_mode == "optimistic":
            return 0
        half_tick = 0.5 * self.cfg.tick_size_usd
        if side == "bid":
            # Bid: higher price = better.
            if price > state.bid_px + half_tick:
                return 0                            # improving — alone at new best
            if abs(price - state.bid_px) <= half_tick:
                return int(state.bid_sz)            # joining current best
            return int(state.bid_sz)                # behind market — conservative
        else:                                        # ask
            # Ask: lower price = better.
            if price < state.ask_px - half_tick:
                return 0                            # improving — alone at new best
            if abs(price - state.ask_px) <= half_tick:
                return int(state.ask_sz)            # joining current best
            return int(state.ask_sz)                # behind market — conservative
