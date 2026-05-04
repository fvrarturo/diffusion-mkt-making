"""Per-stock per-day calibration for the AS-family agents.

Action-plan §3.2 prescribes:
  σ : realized vol from 5-min returns on the training window
  κ : exponential-decay model on observed fill rates at various spread levels
  γ : default 0.1 (or sweep on validation)

This module implements the v1 calibrator:
  - σ ← std of mid log-returns scaled to 5-min window across training tapes
  - κ ← simple proxy from average inter-trade spacing & spread (placeholder
        until we have a proper fill-rate fit; documented below)
  - γ ← config default
  - β_OFI, α_VPIN ← config defaults; grid-search added in a later iteration

The output is a single AgentParams that an agent can `reset(params)` from
before each backtest. Saved to
data/processed/agent_params_{ticker}_{strategy}.json for reproducibility.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from ..io import parquet
from .base import AgentParams


def _load_concat(tape_paths: list[str | Path], cols: list[str]) -> pl.DataFrame:
    frames = [parquet.read_tape(p).select(cols) for p in tape_paths]
    return pl.concat(frames, how="vertical_relaxed")


def estimate_sigma_per_second(tape_paths: list[str | Path]) -> float:
    """Realized vol of mid log-returns, expressed per-second.

    The canonical tape's `mid_return` is per-event; convert to per-second via
    average inter-event spacing in the tape so AS's σ²·(T − t) term has the
    right time units when T-t is in seconds.
    """
    df = _load_concat(tape_paths, ["ts_ns", "mid_return"])
    rets = df["mid_return"].drop_nulls().to_numpy()
    if rets.size < 2:
        return 1e-4
    var_per_event = float(np.var(rets, ddof=0))
    # Average inter-event spacing in seconds.
    ts = df["ts_ns"].to_numpy()
    if ts.size < 2:
        return float(np.sqrt(var_per_event))
    avg_dt_sec = float(np.median(np.diff(ts))) * 1e-9
    avg_dt_sec = max(avg_dt_sec, 1e-6)
    var_per_sec = var_per_event / avg_dt_sec
    return float(np.sqrt(var_per_sec))


def estimate_kappa(tape_paths: list[str | Path], tick_size_usd: float = 0.01) -> float:
    """Crude κ proxy: arrivals-per-second of opposite-side trades, normalized
    by tick size. Sums trading duration WITHIN each tape (so overnight gaps
    don't deflate the rate when training tapes are scattered across the year).

    The full AS κ comes from fitting exponential decay to fill-rate-vs-spread
    observations; that requires a counterfactual replay over many spread
    levels which we defer to a deepening. The proxy here is monotonic in
    liquidity (more arrivals → larger κ → tighter optimal spread) which is
    the qualitative behavior we want.
    """
    n_trades_total = 0
    span_sec_total = 0.0
    for path in tape_paths:
        df = parquet.read_tape(path).select(["ts_ns", "event_type"])
        ts = df["ts_ns"].to_numpy()
        if ts.size < 2:
            continue
        n_trades_total += int((df["event_type"] == "trade").sum())
        span_sec_total += float(ts[-1] - ts[0]) * 1e-9         # WITHIN-day span only
    if n_trades_total < 10 or span_sec_total <= 0:
        return 50.0
    arrivals_per_sec_per_side = (n_trades_total / span_sec_total) * 0.5
    return float(arrivals_per_sec_per_side / tick_size_usd)


def calibrate_a1(
    tape_paths: list[str | Path],
    *,
    risk_aversion: float = 0.1,
    base_size: int = 100,
    inventory_limit: int = 1000,
    tick_size_usd: float = 0.01,
    kappa_override: float | None = None,
    sigma_override: float | None = None,
) -> AgentParams:
    """Calibrate Agent A1 (AS baseline) on a list of training tapes.

    `kappa_override` / `sigma_override` skip the data-fitting step. Useful
    when the proxy estimator under-/over-shoots and we want to lock a
    sensible value for sensitivity analysis.
    """
    σ = sigma_override if sigma_override is not None else estimate_sigma_per_second(tape_paths)
    κ = kappa_override if kappa_override is not None else estimate_kappa(tape_paths, tick_size_usd=tick_size_usd)
    return AgentParams(
        risk_aversion=risk_aversion,
        arrival_decay=κ,
        volatility=σ,
        base_size=base_size,
        inventory_limit=inventory_limit,
    )


def calibrate_a2(
    tape_paths: list[str | Path],
    *,
    risk_aversion: float = 0.1,
    base_size: int = 100,
    inventory_limit: int = 1000,
    tick_size_usd: float = 0.01,
    beta_ofi: float = 1.0,
    kappa_override: float | None = None,
    sigma_override: float | None = None,
) -> AgentParams:
    base = calibrate_a1(
        tape_paths, risk_aversion=risk_aversion, base_size=base_size,
        inventory_limit=inventory_limit, tick_size_usd=tick_size_usd,
        kappa_override=kappa_override, sigma_override=sigma_override,
    )
    return AgentParams(
        risk_aversion=base.risk_aversion,
        arrival_decay=base.arrival_decay,
        volatility=base.volatility,
        base_size=base.base_size,
        inventory_limit=base.inventory_limit,
        beta_ofi=beta_ofi,
    )


def calibrate_a3(
    tape_paths: list[str | Path],
    *,
    risk_aversion: float = 0.1,
    base_size: int = 100,
    inventory_limit: int = 1000,
    tick_size_usd: float = 0.01,
    alpha_vpin_widening: float = 1.0,
    tau_vpin_bucket: int = 2,
    kappa_override: float | None = None,
    sigma_override: float | None = None,
) -> AgentParams:
    base = calibrate_a1(
        tape_paths, risk_aversion=risk_aversion, base_size=base_size,
        inventory_limit=inventory_limit, tick_size_usd=tick_size_usd,
        kappa_override=kappa_override, sigma_override=sigma_override,
    )
    return AgentParams(
        risk_aversion=base.risk_aversion,
        arrival_decay=base.arrival_decay,
        volatility=base.volatility,
        base_size=base.base_size,
        inventory_limit=base.inventory_limit,
        alpha_vpin_widening=alpha_vpin_widening,
        tau_vpin_bucket=tau_vpin_bucket,
    )


def save_params(params: AgentParams, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(params), indent=2))
    return p


def load_params(path: str | Path) -> AgentParams:
    return AgentParams(**json.loads(Path(path).read_text()))
