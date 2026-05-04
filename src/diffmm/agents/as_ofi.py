"""Agent A2 — Avellaneda–Stoikov + short-horizon OFI skew.

A1's reservation price is `r_t = S_t − q_t γ σ² (T − t)`. A2 adds an
order-flow-imbalance term that nudges r_t in the direction of recent
top-of-book pressure:

    r_t ← r_t + β_OFI · ofi_signal · σ

where ofi_signal ∈ [−1, +1] is the centered top-of-book imbalance bucket
(c_imb − 1, where c_imb ∈ {0, 1, 2}). Using the bucket rather than a raw
running OFI keeps A2 stateless and reuses the regime feature already
computed at clean time. β_OFI is calibrated on the validation set
(grid-search over {0.1, 0.5, 1.0, 2.0}).

Quoted spread is unchanged from A1; only the midpoint shifts.
"""
from __future__ import annotations

import math

from .as_baseline import AvellanedaStoikovAgent
from .base import LOBState, Quote


class ASPlusOFIAgent(AvellanedaStoikovAgent):
    name = "A2_AS_OFI"

    def on_event(self, state: LOBState) -> Quote:
        γ = self.params.risk_aversion
        κ = max(self.params.arrival_decay, 1e-6)
        σ = self.params.volatility
        σ2 = σ ** 2
        T_minus_t = max(state.time_remaining, 0.0)
        β = self.params.beta_ofi

        # Inventory-skewed mid + OFI lean
        ofi_signal = float(state.c_imb - 1)        # {-1, 0, +1}
        r = state.mid - state.inventory * γ * σ2 * T_minus_t + β * ofi_signal * σ

        half_spread = 0.5 * (γ * σ2 * T_minus_t + (2.0 / γ) * math.log(1.0 + γ / κ))
        half_spread = max(half_spread, 0.5 * self.tick)

        bid_px = self._round_to_tick(r - half_spread)
        ask_px = self._round_to_tick(r + half_spread)

        bid_sz, ask_sz = self.params.base_size, self.params.base_size
        if state.inventory >= self.params.inventory_limit:
            bid_sz = 0
        elif state.inventory <= -self.params.inventory_limit:
            ask_sz = 0

        if ask_px <= bid_px:
            ask_px = bid_px + self.tick

        return Quote(
            bid_px=bid_px if bid_sz > 0 else None,
            ask_px=ask_px if ask_sz > 0 else None,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
        )
