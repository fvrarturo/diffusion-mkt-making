"""Agent A3 — Avellaneda–Stoikov + VPIN-gated spread widening.

A1's spread is `Δ_t = γσ²(T−t) + (2/γ) log(1 + γ/κ)`. A3 widens it whenever
flow toxicity is high:

    (δ_a + δ_b) ← (δ_a + δ_b) · (1 + α · 1{c_vpin ≥ τ_bucket})

The action plan §3.4 originally calls for a continuous τ_VPIN cutoff at the
80th percentile. Since canonical tapes carry the bucketed `c_vpin` ∈ {0, 1, 2}
fitted at 33%/67% quantiles, we use bucket ≥ 2 as a proxy (top 33% — close
enough for the rank-correlation test; refine later if necessary).

Reservation price is unchanged from A1; only the spread widens defensively.
"""
from __future__ import annotations

import math

from .as_baseline import AvellanedaStoikovAgent
from .base import LOBState, Quote


class ASPlusVPINAgent(AvellanedaStoikovAgent):
    name = "A3_AS_VPIN"

    def on_event(self, state: LOBState) -> Quote:
        γ = self.params.risk_aversion
        κ = max(self.params.arrival_decay, 1e-6)
        σ2 = self.params.volatility ** 2
        T_minus_t = max(state.time_remaining, 0.0)
        α = self.params.alpha_vpin_widening
        τ = self.params.tau_vpin_bucket

        r = state.mid - state.inventory * γ * σ2 * T_minus_t

        base_half_spread = 0.5 * (
            γ * σ2 * T_minus_t + (2.0 / γ) * math.log(1.0 + γ / κ)
        )
        # Toxicity gate: widen when current event lies in the top VPIN bucket.
        toxicity_active = state.c_vpin >= τ
        widening = (1.0 + α) if toxicity_active else 1.0
        half_spread = max(base_half_spread * widening, 0.5 * self.tick)

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
