"""Agent A0 — Constant-Spread (Zero-Intelligence Market Maker).

A naive baseline that always posts a 1-tick-wide quote around mid:

    bid = mid − tick_size
    ask = mid + tick_size

No inventory adjustment, no OFI lean, no VPIN gate. The proposal calls this
the "naive constant-spread" stretch-goal agent (action plan §B). For the
central rank-correlation test, A0 increases the agent count from 3 to 4,
which substantially tightens the bootstrap CIs on Spearman ρ (the ranking
space grows from 6 to 24 distinct orderings).
"""
from __future__ import annotations

from .base import AgentParams, LOBState, MarketMakingAgent, Quote


class ConstantSpreadAgent(MarketMakingAgent):
    name = "A0_const"

    def __init__(self, params: AgentParams | None = None, tick_size_usd: float = 0.01):
        super().__init__(params)
        self.tick = tick_size_usd

    def on_event(self, state: LOBState) -> Quote:
        bid_px = self._round_to_tick(state.mid - self.tick)
        ask_px = self._round_to_tick(state.mid + self.tick)
        # Inventory limit drops the inner side, same convention as A1.
        bid_sz, ask_sz = self.params.base_size, self.params.base_size
        if state.inventory >= self.params.inventory_limit:
            bid_sz = 0
        elif state.inventory <= -self.params.inventory_limit:
            ask_sz = 0
        # Final no-cross safety
        if ask_px <= bid_px:
            ask_px = bid_px + self.tick
        return Quote(
            bid_px=bid_px if bid_sz > 0 else None,
            ask_px=ask_px if ask_sz > 0 else None,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
        )

    def _round_to_tick(self, px: float) -> float:
        return round(px / self.tick) * self.tick
