"""Phase 4 §4.2 — historical replay on REAL val tapes → π_hist.

Loads each agent's calibrated params, replays every val-set tape, writes a
per-day metrics CSV and the agent ranking by Sharpe.

Usage:
    python scripts/21_replay_real.py data=intc_top
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from diffmm.agents.as_baseline import AvellanedaStoikovAgent
from diffmm.agents.as_ofi import ASPlusOFIAgent
from diffmm.agents.as_vpin import ASPlusVPINAgent
from diffmm.agents.calibration import load_params
from diffmm.agents.constant_spread import ConstantSpreadAgent
from diffmm.eval.ranking import rank_by_sharpe
from diffmm.sim.runner import run_backtest_on_split
from diffmm.utils.logging import get_logger

log = get_logger("ph4_replay_real")

AGENT_REGISTRY = {
    "A0": ConstantSpreadAgent,
    "A1": AvellanedaStoikovAgent,
    "A2": ASPlusOFIAgent,
    "A3": ASPlusVPINAgent,
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    val_dir = Path(cfg.data.processed_dir) / "val"
    val_paths = sorted(val_dir.glob(f"{cfg.data.ticker}_*.parquet"))
    if not val_paths:
        raise FileNotFoundError(f"no val tapes under {val_dir} for {cfg.data.ticker}")
    log.info("running replay on %d val tapes for %s", len(val_paths), cfg.data.ticker)

    params_dir = Path(cfg.cluster.results_root) / "agent_params"
    out_dir = Path(cfg.cluster.results_root) / "backtests" / "real" / cfg.data.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for agent_name, AgentCls in AGENT_REGISTRY.items():
        params_path = params_dir / f"{cfg.data.ticker}_{agent_name}.json"
        if not params_path.exists():
            raise FileNotFoundError(f"missing calibration: {params_path}. Run 20_calibrate_agents.py first.")
        params = load_params(params_path)
        agent = AgentCls(params=params)
        log.info("%s: %s", agent_name, agent.params)
        per_day = run_backtest_on_split(agent, val_paths)
        for m in per_day:
            all_rows.append(asdict(m))

    csv_path = out_dir / "metrics.csv"
    pl.DataFrame(all_rows).write_csv(csv_path)
    log.info("per-day metrics: %s (%d rows)", csv_path, len(all_rows))

    ranking = rank_by_sharpe(all_rows)
    ranking_path = out_dir / "ranking_pi_hist.json"
    ranking_path.write_text(json.dumps({
        "metric": ranking.metric,
        "agents": list(ranking.agents),
        "scores": ranking.scores,
    }, indent=2))
    log.info("π_hist (best→worst): %s", " > ".join(ranking.agents))
    log.info("scores: %s", ranking.scores)
    log.info("wrote %s", ranking_path)


if __name__ == "__main__":
    main()
