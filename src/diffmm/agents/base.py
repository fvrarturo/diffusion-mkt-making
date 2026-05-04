"""Common contract for market-making agents.

Every agent — Avellaneda–Stoikov baseline, OFI-skewed, VPIN-gated, future
extensions — speaks the same `LOBState`/`Quote` interface. Agents are pure
functions of state in the sense that `on_event(state) → quote` carries no
side effects beyond the agent's internal calibration parameters.

The fill simulator (sim/fill.py) is the SAME code for real and synthetic
tapes (Work0 §10), so any quoting policy that works on real INTC works on
synthetic INTC. That's the symmetry the central hypothesis test relies on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LOBState:
    """Snapshot the agent sees at one event.

    Mid and spread are derivable from bid_px/ask_px but precomputed for speed
    and to avoid agents introducing their own definition.
    """
    ts_ns: int
    bid_px: float
    ask_px: float
    bid_sz: int
    ask_sz: int
    mid: float
    spread: float
    inventory: int           # current q_t (can be negative if short)
    cash: float              # current X_t in dollars
    time_remaining: float    # T - t in SECONDS until forced liquidation
    # Regime context — agents may use these (A3 uses c_vpin) or ignore them.
    regime_label: str
    c_vol: int               # 0/1/2 from training-fit quantiles
    c_vpin: int              # 0/1/2
    c_imb: int               # 0/1/2 from fixed thresholds


@dataclass(frozen=True)
class Quote:
    """The agent's posted limit orders for the next interval.

    Either side may be None (no quote on that side, e.g. inventory limit hit).
    Sizes are in shares, prices in dollars (absolute, NOT deltas from mid —
    keeps the fill simulator's bookkeeping simple).
    """
    bid_px: float | None
    ask_px: float | None
    bid_sz: int
    ask_sz: int

    def is_valid(self, tick_size_usd: float = 0.01) -> bool:
        """Sanity check: no crossed quotes, non-negative sizes."""
        if self.bid_sz < 0 or self.ask_sz < 0:
            return False
        if self.bid_px is not None and self.bid_px <= 0:
            return False
        if self.ask_px is not None and self.ask_px <= 0:
            return False
        if self.bid_px is not None and self.ask_px is not None:
            if self.ask_px <= self.bid_px:
                return False
        return True


@dataclass(frozen=True)
class AgentParams:
    """Per-stock per-day calibration. Each subclass interprets only the
    fields it needs; unknown fields are ignored.

    Defaults are sane fallbacks if calibration hasn't run yet.
    """
    # Common
    risk_aversion: float = 0.1               # γ
    arrival_decay: float = 50.0              # κ (1/$, exp arrival rate)
    volatility: float = 1.0e-4               # σ per second (event-scale)
    base_size: int = 100                     # shares per side
    inventory_limit: int = 1000              # |q| > this → drop inner quote
    # A2 (OFI-skew)
    beta_ofi: float = 0.0
    # A3 (VPIN gate)
    alpha_vpin_widening: float = 0.0
    tau_vpin_bucket: int = 2                 # widen if c_vpin >= this


class MarketMakingAgent(ABC):
    """Base class. Subclasses implement `on_event` to return quotes.

    The agent keeps internal state (calibration params); the rest — inventory,
    cash, fills — is owned by the fill simulator and passed in via LOBState.
    """
    name: str = "base"

    def __init__(self, params: AgentParams | None = None):
        self.params: AgentParams = params or AgentParams()

    def reset(self, params: AgentParams | None = None) -> None:
        """Re-arm for a new day. Called by the runner before each backtest."""
        if params is not None:
            self.params = params

    @abstractmethod
    def on_event(self, state: LOBState) -> Quote:
        """Compute quotes for the next interval."""
        ...
