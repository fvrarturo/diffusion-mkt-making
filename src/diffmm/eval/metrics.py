"""Per-day backtest metrics.

Inputs are `DayMetrics` rows (terminal_pnl, drawdown, inventory_var, fill_rate)
produced by `sim.runner`. Outputs are aggregated per (agent[, regime, ticker])
summaries used by the ranking step (Phase 4 §4.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary across many backtest days."""
    agent: str
    n_days: int
    mean_pnl: float
    sharpe: float                # √252 · mean(R) / sd(R), R = daily PnL
    cvar_5pct: float             # average of worst 5% of daily PnLs
    worst_quartile_pnl: float    # 25th percentile (used for π_diff)
    mean_inventory_var: float
    mean_fill_rate: float
    extra: dict = field(default_factory=dict)


def aggregate(rows: list[dict]) -> AggregateMetrics:
    """Aggregate a list of per-day metrics dicts (one per backtest day) into
    one AggregateMetrics row.

    Each row must have keys: agent, terminal_pnl, inventory_var, fill_rate.
    """
    if not rows:
        raise ValueError("aggregate: rows is empty")
    agents = {r["agent"] for r in rows}
    assert len(agents) == 1, f"aggregate expects one agent per call, got {agents}"

    pnls = np.asarray([r["terminal_pnl"] for r in rows], dtype=np.float64)
    inv_vars = np.asarray([r["inventory_var"] for r in rows], dtype=np.float64)
    fill_rates = np.asarray([r["fill_rate"] for r in rows], dtype=np.float64)

    mean_pnl = float(pnls.mean())
    std_pnl = float(pnls.std(ddof=1)) if pnls.size > 1 else 0.0
    sharpe = math.sqrt(252.0) * mean_pnl / std_pnl if std_pnl > 0 else 0.0

    cvar_5pct = float(pnls[pnls <= np.quantile(pnls, 0.05)].mean()) if pnls.size > 0 else 0.0
    worst_q = float(np.quantile(pnls, 0.25)) if pnls.size > 0 else 0.0

    return AggregateMetrics(
        agent=next(iter(agents)),
        n_days=len(rows),
        mean_pnl=mean_pnl,
        sharpe=sharpe,
        cvar_5pct=cvar_5pct,
        worst_quartile_pnl=worst_q,
        mean_inventory_var=float(inv_vars.mean()),
        mean_fill_rate=float(fill_rates.mean()),
    )


def metrics_to_dataframe(metrics: list[dict]) -> pl.DataFrame:
    """Convert a list of per-day metric dicts to a Polars DataFrame for CSV output."""
    return pl.DataFrame(metrics)
