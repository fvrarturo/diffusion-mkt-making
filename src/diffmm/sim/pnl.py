"""PnL bookkeeping for the fill simulator.

Single source of truth for how cash and inventory evolve. Keep this small and
boring — every change here invalidates every backtest result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    """Mutable per-day state of the market maker.

    Convention: positive inventory means the MM is long. A buy fill (we lifted
    a market sell on our bid) increases inventory and decreases cash.
    """
    inventory: int = 0
    cash: float = 0.0

    def buy(self, qty: int, price: float) -> None:
        """We bought `qty` shares at `price` (a market seller hit our bid)."""
        if qty <= 0:
            return
        self.inventory += qty
        self.cash -= qty * price

    def sell(self, qty: int, price: float) -> None:
        """We sold `qty` shares at `price` (a market buyer lifted our ask)."""
        if qty <= 0:
            return
        self.inventory -= qty
        self.cash += qty * price

    def mark_to_market(self, mid: float) -> float:
        """Total PnL = cash + inventory · mid. Use the prevailing mid."""
        return self.cash + self.inventory * mid

    def liquidate(self, mid: float) -> float:
        """Force-flatten at `mid`. Returns final PnL.

        Convention: liquidation is at mid (no spread paid), which is generous
        for a small inventory and conservative for a large one. Same handling
        on real and synthetic, so directionally fair across the comparison.
        """
        if self.inventory > 0:
            self.cash += self.inventory * mid
        elif self.inventory < 0:
            self.cash -= (-self.inventory) * mid
        terminal = self.cash
        self.inventory = 0
        return terminal
