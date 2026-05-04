"""Backtest runner — wraps a FillSimulator over a set of tape paths and
aggregates per-day metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agents.base import MarketMakingAgent
from ..io import parquet
from .fill import BacktestResult, FillConfig, FillSimulator


@dataclass
class DayMetrics:
    """Per-day summary used by the ranking step (Phase 4)."""
    tape_path: str
    agent: str
    n_events: int
    n_fills: int
    terminal_pnl: float
    pnl_max_drawdown: float
    inventory_var: float
    fill_rate: float


def run_backtest_on_tape(
    agent: MarketMakingAgent,
    tape_path: str | Path,
    *,
    fill_config: FillConfig | None = None,
) -> tuple[BacktestResult, DayMetrics]:
    """Replay one tape file. Returns (raw result, summary metrics)."""
    df = parquet.read_tape(tape_path)
    sim = FillSimulator(fill_config)
    result = sim.run(agent, df)

    intraday = np.asarray(result.intraday_pnl, dtype=np.float64)
    if intraday.size > 0:
        running_peak = np.maximum.accumulate(intraday)
        drawdown = float(np.min(intraday - running_peak))
    else:
        drawdown = 0.0
    inventory_var = float(np.var(np.asarray(result.inventory_series, dtype=np.float64))) \
        if result.inventory_series else 0.0

    metrics = DayMetrics(
        tape_path=str(tape_path),
        agent=agent.name,
        n_events=result.n_events,
        n_fills=result.n_fills,
        terminal_pnl=result.terminal_pnl,
        pnl_max_drawdown=drawdown,
        inventory_var=inventory_var,
        fill_rate=result.fill_rate(n_quotes_posted=result.n_events),
    )
    return result, metrics


def run_backtest_on_split(
    agent: MarketMakingAgent,
    tape_paths: list[str | Path],
    *,
    fill_config: FillConfig | None = None,
) -> list[DayMetrics]:
    """Run an agent over many tapes (e.g. all val tapes for one ticker)."""
    out: list[DayMetrics] = []
    for p in tape_paths:
        agent.reset(agent.params)         # fresh inventory + cash each day
        _, m = run_backtest_on_tape(agent, p, fill_config=fill_config)
        out.append(m)
    return out
