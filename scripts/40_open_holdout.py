"""Phase 5 §5.1 — open the holdout once and replay all agents on it.

★ THIS IS THE ONLY AUTHORIZED SCRIPT ALLOWED TO READ HOLDOUT FILES. ★

It uses `diffmm.io.holdout_guard.unlocked(reason=...)` to open the firewall
(see Work0 §3) for the duration of the read. Every other script that
attempts to load holdout files raises HoldoutLeakageError.

Outputs:
  results/backtests/holdout/{ticker}/metrics.csv
  results/backtests/holdout/{ticker}/ranking_pi_truth.json

After this runs, scripts/41_hypothesis_test.py reads those outputs and
runs the bootstrap test against π_hist and π_diff.
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
from diffmm.io import holdout_guard
from diffmm.sim.runner import run_backtest_on_split
from diffmm.utils.logging import get_logger

log = get_logger("ph5_holdout")

AGENT_REGISTRY = {
    "A0": ConstantSpreadAgent,
    "A1": AvellanedaStoikovAgent,
    "A2": ASPlusOFIAgent,
    "A3": ASPlusVPINAgent,
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    holdout_dir = Path(cfg.data.processed_dir) / "holdout"
    params_dir = Path(cfg.cluster.results_root) / "agent_params"
    out_dir = Path(cfg.cluster.results_root) / "backtests" / "holdout" / cfg.data.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confirm we have all agent calibrations BEFORE we open the firewall.
    for agent_name in AGENT_REGISTRY:
        p = params_dir / f"{cfg.data.ticker}_{agent_name}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; run scripts/20_calibrate_agents.py first")

    log.warning("=" * 70)
    log.warning("OPENING THE HOLDOUT FIREWALL — Phase 5 central hypothesis test")
    log.warning("This must be a one-time, no-iteration use.")
    log.warning("=" * 70)

    with holdout_guard.unlocked(reason="phase 5 §5.1 — holdout replay for π_truth"):
        holdout_paths = sorted(holdout_dir.glob(f"{cfg.data.ticker}_*.parquet"))
        if not holdout_paths:
            raise FileNotFoundError(f"no holdout tapes for {cfg.data.ticker} under {holdout_dir}")
        log.info("holdout tapes: %d", len(holdout_paths))

        all_rows: list[dict] = []
        for agent_name, AgentCls in AGENT_REGISTRY.items():
            params = load_params(params_dir / f"{cfg.data.ticker}_{agent_name}.json")
            agent = AgentCls(params=params)
            log.info("running %s on %d holdout tapes", agent_name, len(holdout_paths))
            per_day = run_backtest_on_split(agent, holdout_paths)
            for m in per_day:
                all_rows.append(asdict(m))

    csv_path = out_dir / "metrics.csv"
    pl.DataFrame(all_rows).write_csv(csv_path)
    log.info("per-day metrics: %s (%d rows)", csv_path, len(all_rows))

    pi_truth = rank_by_sharpe(all_rows)
    out_path = out_dir / "ranking_pi_truth.json"
    out_path.write_text(json.dumps({
        "metric": pi_truth.metric,
        "agents": list(pi_truth.agents),
        "scores": pi_truth.scores,
    }, indent=2))
    log.info("π_truth (best→worst by %s): %s", pi_truth.metric, " > ".join(pi_truth.agents))
    log.info("scores: %s", pi_truth.scores)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
