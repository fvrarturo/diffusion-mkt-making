"""Agent A1 — Avellaneda–Stoikov baseline (proposal Algorithm 4).

Reservation price tilts against inventory; spread depends on volatility,
risk-aversion, and arrival-rate decay κ:

    r_t  = S_t  −  q_t · γ · σ² · (T − t)
    Δ_t  = γ · σ² · (T − t)  +  (2 / γ) · ln(1 + γ / κ)

    bid_quote = r_t − Δ_t / 2
    ask_quote = r_t + Δ_t / 2

(γ, κ, σ) come from per-stock per-day calibration on the training window.
For real execution we round quotes to the tick grid.
"""
from __future__ import annotations

import math

from .base import AgentParams, LOBState, MarketMakingAgent, Quote


class AvellanedaStoikovAgent(MarketMakingAgent):
    name = "A1_AS"

    def __init__(self, params: AgentParams | None = None, tick_size_usd: float = 0.01):
        super().__init__(params)
        self.tick = tick_size_usd

    def on_event(self, state: LOBState) -> Quote:
        γ = self.params.risk_aversion
        κ = max(self.params.arrival_decay, 1e-6)
        σ2 = self.params.volatility ** 2
        T_minus_t = max(state.time_remaining, 0.0)

        # Reservation price (inventory-skewed mid)
        r = state.mid - state.inventory * γ * σ2 * T_minus_t
        # Optimal half-spread
        half_spread = 0.5 * (γ * σ2 * T_minus_t + (2.0 / γ) * math.log(1.0 + γ / κ))
        # Floor at half a tick so we always quote SOMETHING
        half_spread = max(half_spread, 0.5 * self.tick)

        bid_px = self._round_to_tick(r - half_spread)
        ask_px = self._round_to_tick(r + half_spread)

        # Inventory limits: drop the side that would worsen our position.
        bid_sz, ask_sz = self.params.base_size, self.params.base_size
        if state.inventory >= self.params.inventory_limit:
            bid_sz = 0           # already too long → stop buying
        elif state.inventory <= -self.params.inventory_limit:
            ask_sz = 0           # already too short → stop selling

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
