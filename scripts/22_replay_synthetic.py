"""Phase 4 §4.3 — stress replay on SYNTHETIC tapes → π_diff.

For each agent × each regime × every synthetic Parquet, run the same fill
simulator we used on real data. Aggregate per (agent, regime) and globally
to produce π_diff (rank by worst-quartile PnL across all stress regimes).

Usage:
    python scripts/22_replay_synthetic.py data=intc_top \\
        +synthetic_root=/orcd/scratch/orcd/007/favara/diffmm/synthetic
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
from diffmm.eval.ranking import rank_by_mean_pnl, rank_by_worst_quartile
from diffmm.sim.runner import run_backtest_on_split
from diffmm.utils.logging import get_logger

log = get_logger("ph4_replay_synth")

AGENT_REGISTRY = {
    "A0": ConstantSpreadAgent,
    "A1": AvellanedaStoikovAgent,
    "A2": ASPlusOFIAgent,
    "A3": ASPlusVPINAgent,
}
REGIMES = ("base", "high_vol", "toxic", "thin")


def _resolve_synthetic_root(cfg: DictConfig) -> Path:
    if "synthetic_root" in cfg:
        return Path(cfg.synthetic_root)
    return Path(cfg.cluster.synthetic_root)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    synth_root = _resolve_synthetic_root(cfg) / cfg.data.ticker
    if not synth_root.exists():
        raise FileNotFoundError(f"synthetic root {synth_root} does not exist")

    params_dir = Path(cfg.cluster.results_root) / "agent_params"
    out_dir = Path(cfg.cluster.results_root) / "backtests" / "synthetic" / cfg.data.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for agent_name, AgentCls in AGENT_REGISTRY.items():
        params_path = params_dir / f"{cfg.data.ticker}_{agent_name}.json"
        params = load_params(params_path)
        agent = AgentCls(params=params)
        for regime in REGIMES:
            tape_paths = sorted((synth_root / regime).glob("day_*.parquet"))
            if not tape_paths:
                log.warning("no tapes for regime=%s, skipping", regime)
                continue
            log.info("%s × %s: %d tapes", agent_name, regime, len(tape_paths))
            per_day = run_backtest_on_split(agent, tape_paths)
            for m in per_day:
                row = asdict(m)
                row["regime"] = regime
                all_rows.append(row)

    csv_path = out_dir / "metrics.csv"
    pl.DataFrame(all_rows).write_csv(csv_path)
    log.info("per-day metrics: %s (%d rows)", csv_path, len(all_rows))

    # π_diff: try worst-quartile first (the proposal's preferred metric since
    # it's tail-aware). If all worst-quartile scores are identical (degenerate
    # — happens when most windows produce zero fills, e.g. short synthetic
    # windows), fall back to mean PnL so the ranking is still meaningful.
    ranking = rank_by_worst_quartile(all_rows)
    if len(set(round(s, 9) for s in ranking.scores.values())) <= 1:
        log.warning("worst_quartile_pnl is degenerate (all agents tied at %.4g) — "
                    "falling back to mean_pnl", next(iter(ranking.scores.values())))
        ranking = rank_by_mean_pnl(all_rows)
    ranking_path = out_dir / "ranking_pi_diff.json"
    ranking_path.write_text(json.dumps({
        "metric": ranking.metric,
        "agents": list(ranking.agents),
        "scores": ranking.scores,
    }, indent=2))
    log.info("π_diff (best→worst): %s", " > ".join(ranking.agents))
    log.info("scores: %s", ranking.scores)
    log.info("wrote %s", ranking_path)


if __name__ == "__main__":
    main()
