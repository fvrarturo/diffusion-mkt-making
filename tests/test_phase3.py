"""Phase 3 smoke tests — agent quoting math + fill simulator + runner.

Synthesizes a tiny canonical-schema event tape and verifies:
  - Quote validity (no crossed, non-negative, bounded)
  - Fill conservation: terminal PnL = Σ cash deltas + final_inventory · last_mid
  - Inventory limit dropping the inner quote
  - A1 (Avellaneda–Stoikov) produces non-degenerate quotes
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from diffmm.agents.as_baseline import AvellanedaStoikovAgent
from diffmm.agents.base import AgentParams, LOBState, Quote
from diffmm.io import schema as schema_mod
from diffmm.sim.fill import FillConfig, FillSimulator
from diffmm.sim.pnl import Position
from diffmm.sim.runner import run_backtest_on_tape


# ─── tape factory ──────────────────────────────────────────────────────

def _make_canonical_tape(
    n_events: int = 100,
    *,
    seed: int = 0,
    base_mid: float = 50.0,
    spread: float = 0.01,
    n_trades: int = 30,
) -> pl.DataFrame:
    """Build a tiny canonical-schema event tape suitable for FillSimulator."""
    rng = np.random.default_rng(seed)
    L = n_events
    log_returns = rng.normal(0.0, 1e-5, size=L)
    log_returns[0] = 0.0
    raw_mid = base_mid * np.exp(np.cumsum(log_returns))
    # Real TAQ prices are tick-aligned (e.g. $XX.XX for 1-cent stocks). Round
    # bid/ask to the grid; mid then sits naturally at half-tick increments.
    tick = 0.01
    bid_px = np.round((raw_mid - spread / 2) / tick) * tick
    ask_px = bid_px + spread     # exact 1-tick spread for the test
    mid = 0.5 * (bid_px + ask_px)
    bid_sz = rng.integers(100, 500, size=L)
    ask_sz = rng.integers(100, 500, size=L)

    is_trade = np.zeros(L, dtype=bool)
    trade_idx = rng.choice(np.arange(1, L), size=min(n_trades, L - 1), replace=False)
    is_trade[trade_idx] = True
    trade_sign = np.where(rng.random(L) > 0.5, 1, -1).astype(np.int8)
    trade_px = np.where(is_trade & (trade_sign == 1), ask_px, np.where(is_trade & (trade_sign == -1), bid_px, np.nan))
    trade_sz = np.where(is_trade, rng.integers(50, 200, size=L), 0)

    rows = {
        "event_idx":    np.arange(L, dtype=np.int64),
        "ts_ns":        1_700_000_000_000_000_000 + np.arange(L, dtype=np.int64) * 1_000_000_000,  # 1 sec apart
        "ticker":       ["INTC"] * L,
        "event_type":   np.where(is_trade, "trade", "quote_update"),
        "bid_px":       bid_px,
        "ask_px":       ask_px,
        "bid_sz":       bid_sz.astype(np.int64),
        "ask_sz":       ask_sz.astype(np.int64),
        "trade_px":     [float(v) if is_trade[i] else None for i, v in enumerate(trade_px)],
        "trade_sz":     [int(v) if is_trade[i] else None for i, v in enumerate(trade_sz)],
        "trade_sign":   [int(v) if is_trade[i] else None for i, v in enumerate(trade_sign)],
        "is_lit":       [True if is_trade[i] else None for i in range(L)],
        "mid":          mid,
        "spread":       np.full(L, spread),
        "bid_dist":     bid_px - mid,
        "ask_dist":     ask_px - mid,
        "trade_dist":   [float(trade_px[i] - mid[i]) if is_trade[i] else None for i in range(L)],
        "mid_return":   [None] + np.diff(np.log(mid)).tolist(),
        "c_vol":        np.ones(L, dtype=np.int8),
        "c_vpin":       np.ones(L, dtype=np.int8),
        "c_imb":        np.ones(L, dtype=np.int8),
        "c_tod":        np.ones(L, dtype=np.int8),
        "regime_label": ["base"] * L,
    }
    df = pl.DataFrame(rows, schema_overrides=schema_mod.CORE_DTYPES)
    return schema_mod.validate(df)


# ─── tests ─────────────────────────────────────────────────────────────

def test_avellaneda_stoikov_produces_valid_quotes():
    agent = AvellanedaStoikovAgent(AgentParams(risk_aversion=0.1, arrival_decay=50.0,
                                                volatility=1e-4, base_size=100))
    state = LOBState(
        ts_ns=0, bid_px=50.0, ask_px=50.02, bid_sz=200, ask_sz=300,
        mid=50.01, spread=0.02, inventory=0, cash=0.0, time_remaining=100.0,
        regime_label="base", c_vol=1, c_vpin=1, c_imb=1,
    )
    q = agent.on_event(state)
    assert q.is_valid()
    assert q.bid_sz == 100 and q.ask_sz == 100
    assert q.bid_px is not None and q.ask_px is not None
    assert q.ask_px - q.bid_px >= 0.01


def test_inventory_limit_drops_inner_quote():
    """At inventory_limit, the bid (or ask) is dropped."""
    agent = AvellanedaStoikovAgent(AgentParams(inventory_limit=10, base_size=5))
    long_state = LOBState(
        ts_ns=0, bid_px=50.0, ask_px=50.02, bid_sz=200, ask_sz=300,
        mid=50.01, spread=0.02, inventory=10, cash=0.0, time_remaining=100.0,
        regime_label="base", c_vol=1, c_vpin=1, c_imb=1,
    )
    q = agent.on_event(long_state)
    assert q.bid_px is None and q.bid_sz == 0
    assert q.ask_px is not None and q.ask_sz > 0


def test_position_conservation():
    """Position bookkeeping: terminal cash matches a hand-computed scenario."""
    pos = Position()
    pos.buy(100, 50.0)        # cash = -5000, inv = +100
    pos.sell(50, 51.0)        # cash = -5000 + 2550 = -2450, inv = +50
    assert pos.inventory == 50
    assert pos.cash == pytest.approx(-2450.0)
    final = pos.liquidate(50.5)
    # +50 shares × $50.5 = +2525 → cash = -2450 + 2525 = 75
    assert final == pytest.approx(75.0)
    assert pos.inventory == 0


def test_fill_simulator_no_quotes_no_pnl():
    """Agent that never quotes → zero PnL, zero fills."""
    class NeverQuoteAgent(AvellanedaStoikovAgent):
        def on_event(self, state):
            return Quote(bid_px=None, ask_px=None, bid_sz=0, ask_sz=0)

    tape = _make_canonical_tape(n_events=50, seed=1)
    agent = NeverQuoteAgent()
    sim = FillSimulator()
    result = sim.run(agent, tape)
    assert result.n_fills == 0
    assert result.terminal_pnl == pytest.approx(0.0)


def test_fill_simulator_runs_end_to_end():
    """A1 on a synthetic tape: produces fills, plausible PnL trace.

    Calibration here is tuned so that A1's optimal-spread formula yields a
    quote at-or-inside the test tape's $0.01 spread; otherwise the agent
    quotes behind the market and (correctly) never fills. Real calibration
    will derive κ and γ from observed market arrivals (Phase 3.2).
    """
    tape = _make_canonical_tape(n_events=200, seed=2, n_trades=80)
    agent = AvellanedaStoikovAgent(AgentParams(
        risk_aversion=0.01, arrival_decay=500.0, volatility=1e-4, base_size=50,
    ))
    result = FillSimulator().run(agent, tape)
    assert result.n_events == 200
    assert len(result.intraday_pnl) == 200
    assert len(result.inventory_series) == 200
    assert result.n_fills > 0, (
        "expected at least one fill — likely A1's calibrated half-spread is "
        "wider than the test tape's market spread; tune κ/γ"
    )


def test_runner_produces_per_day_metrics(tmp_path):
    """run_backtest_on_tape persists Parquet + reports metrics."""
    tape = _make_canonical_tape(n_events=100, seed=3, n_trades=30)
    out = tmp_path / "INTC_test.parquet"
    tape.write_parquet(out)

    agent = AvellanedaStoikovAgent(AgentParams(volatility=1e-4, base_size=10))
    _, metrics = run_backtest_on_tape(agent, out)
    assert metrics.n_events == 100
    assert metrics.agent == "A1_AS"
    assert metrics.tape_path == str(out)
    assert isinstance(metrics.terminal_pnl, float)
    assert isinstance(metrics.inventory_var, float)


def test_a0_constant_spread_quotes_one_tick_around_mid():
    """A0 always quotes ±1 tick around mid, ignores inventory + regime, drops
    the inner side when |inventory| ≥ limit."""
    from diffmm.agents.constant_spread import ConstantSpreadAgent

    agent = ConstantSpreadAgent(AgentParams(base_size=100, inventory_limit=1000),
                                tick_size_usd=0.01)

    state = LOBState(
        ts_ns=0, bid_px=50.0, ask_px=50.02, bid_sz=200, ask_sz=300,
        mid=50.01, spread=0.02, inventory=0, cash=0.0, time_remaining=100.0,
        regime_label="base", c_vol=1, c_vpin=1, c_imb=1,
    )
    q = agent.on_event(state)
    assert q.is_valid()
    assert q.bid_px == 50.00
    assert q.ask_px == 50.02
    assert q.bid_sz == 100 and q.ask_sz == 100

    # Inventory limit drops bid side
    state_long = LOBState(
        ts_ns=0, bid_px=50.0, ask_px=50.02, bid_sz=200, ask_sz=300,
        mid=50.01, spread=0.02, inventory=1000, cash=0.0, time_remaining=100.0,
        regime_label="base", c_vol=1, c_vpin=1, c_imb=1,
    )
    q_long = agent.on_event(state_long)
    assert q_long.bid_px is None and q_long.bid_sz == 0
    assert q_long.ask_px is not None


def test_a2_ofi_skews_quotes_with_imbalance():
    """A2's reservation price shifts in the direction of c_imb pressure."""
    from diffmm.agents.as_ofi import ASPlusOFIAgent

    agent = ASPlusOFIAgent(AgentParams(
        risk_aversion=0.01, arrival_decay=500.0, volatility=1e-3, base_size=10,
        beta_ofi=2.0,
    ))

    def state_for_imb(c_imb: int) -> LOBState:
        return LOBState(
            ts_ns=0, bid_px=50.0, ask_px=50.01, bid_sz=100, ask_sz=100,
            mid=50.005, spread=0.01, inventory=0, cash=0.0, time_remaining=100.0,
            regime_label="base", c_vol=1, c_vpin=1, c_imb=c_imb,
        )

    q_neutral = agent.on_event(state_for_imb(c_imb=1))   # ofi_signal=0
    q_buy_pressure = agent.on_event(state_for_imb(c_imb=2))    # ofi_signal=+1
    q_sell_pressure = agent.on_event(state_for_imb(c_imb=0))   # ofi_signal=-1

    # Skew shifts both bid and ask in the SAME direction (the reservation
    # price tilts; the spread is preserved).
    assert q_buy_pressure.bid_px >= q_neutral.bid_px
    assert q_buy_pressure.ask_px >= q_neutral.ask_px
    assert q_sell_pressure.bid_px <= q_neutral.bid_px
    assert q_sell_pressure.ask_px <= q_neutral.ask_px


def test_a3_vpin_widens_spread_when_toxic():
    """A3's spread widens (by factor 1+α) when c_vpin >= τ_bucket."""
    from diffmm.agents.as_vpin import ASPlusVPINAgent

    # Use a γ/κ regime where the base half-spread is already several ticks
    # wide so that the widening multiplier has a visible post-rounding effect.
    agent = ASPlusVPINAgent(AgentParams(
        risk_aversion=1.0, arrival_decay=10.0, volatility=1e-3, base_size=10,
        alpha_vpin_widening=2.0, tau_vpin_bucket=2,
    ))

    def state_for_vpin(c_vpin: int) -> LOBState:
        return LOBState(
            ts_ns=0, bid_px=50.0, ask_px=50.01, bid_sz=100, ask_sz=100,
            mid=50.005, spread=0.01, inventory=0, cash=0.0, time_remaining=100.0,
            regime_label="base", c_vol=1, c_vpin=c_vpin, c_imb=1,
        )

    q_calm = agent.on_event(state_for_vpin(c_vpin=0))
    q_toxic = agent.on_event(state_for_vpin(c_vpin=2))

    spread_calm = q_calm.ask_px - q_calm.bid_px
    spread_toxic = q_toxic.ask_px - q_toxic.bid_px
    assert spread_toxic > spread_calm, (
        f"toxic spread {spread_toxic} should exceed calm spread {spread_calm}"
    )


def test_calibration_produces_sane_sigma_and_kappa(tmp_path):
    """Calibrator estimates positive σ and κ from a tiny synthetic tape set."""
    from diffmm.agents.calibration import (
        calibrate_a1, estimate_kappa, estimate_sigma_per_second, save_params, load_params,
    )

    paths = []
    for k in range(3):
        df = _make_canonical_tape(n_events=200, seed=10 + k, n_trades=80)
        out = tmp_path / f"INTC_2025-01-{k+1:02d}.parquet"
        df.write_parquet(out)
        paths.append(out)

    σ = estimate_sigma_per_second(paths)
    κ = estimate_kappa(paths)
    assert σ > 0.0 and σ < 1.0, f"unreasonable σ: {σ}"
    assert κ > 0.0, f"unreasonable κ: {κ}"

    params = calibrate_a1(paths, risk_aversion=0.05, base_size=25)
    assert params.volatility == pytest.approx(σ)
    assert params.arrival_decay == pytest.approx(κ)
    assert params.risk_aversion == 0.05
    assert params.base_size == 25

    # Round-trip save/load
    out = tmp_path / "params.json"
    save_params(params, out)
    loaded = load_params(out)
    assert loaded == params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
