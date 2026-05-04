"""Stage A2 — make splits, fit quantiles on training set, materialize canonical tapes.

Idempotent: re-run any time the split sizes change. Cheap (~5 min for both
tickers) since intermediate Parquets are already on disk.

Outputs per ticker:
  data/processed/SPLITS_{ticker}.json
  data/processed/quantiles_{ticker}.json
  data/processed/{train,val,holdout}/{ticker}_{date}.parquet  (canonical schema)

Usage:
    python scripts/03_finalize_tapes.py data=intc_top
    python scripts/03_finalize_tapes.py data=intc_top \\
        data.splits.n_train_days=80 data.splits.n_val_days=20
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from diffmm.data import regime, splits
from diffmm.io.manifest import write_meta
from diffmm.utils.logging import get_logger
from diffmm.utils.seeding import seed_all

log = get_logger("ph1_finalize")


def _enumerate_intermediate_dates(intermediate_dir: Path, ticker: str) -> list[str]:
    return sorted(p.stem for p in (intermediate_dir / ticker).glob("*.parquet"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_all(cfg.seed)
    ticker = str(cfg.data.ticker)

    inter_dir = Path(cfg.data.intermediate_dir)
    out_dir = Path(cfg.data.processed_dir)

    available = _enumerate_intermediate_dates(inter_dir, ticker)
    if not available:
        raise FileNotFoundError(
            f"no intermediate Parquets found under {inter_dir/ticker}. "
            "Run scripts/02_build_tapes.py first."
        )
    log.info("ticker=%s  intermediate dates available: %d", ticker, len(available))

    sp_cfg = splits.SplitConfig(
        n_train_days=cfg.data.splits.n_train_days,
        n_val_days=cfg.data.splits.n_val_days,
        n_holdout_days=cfg.data.splits.n_holdout_days,
        holdout_priority_dates=list(cfg.data.splits.holdout_priority_dates),
        seed=cfg.data.splits.seed,
    )
    sp = splits.make_splits(available, sp_cfg)
    log.info("splits  train=%d  val=%d  holdout=%d",
             len(sp.train), len(sp.val), len(sp.holdout))
    splits_path = splits.write_splits_json(sp, cfg.data.splits_file)
    write_meta(splits_path, seed=cfg.seed, config=cfg,
               creator_script="scripts/03_finalize_tapes.py")

    train_intermediates = [inter_dir / ticker / f"{d}.parquet" for d in sp.train]
    log.info("fitting quantiles on %d training tapes", len(train_intermediates))
    q = regime.fit_quantiles(train_intermediates)
    q.save(cfg.data.quantiles_file)
    log.info("quantiles  vol=%s  vpin=%s", q.vol, q.vpin)
    write_meta(Path(cfg.data.quantiles_file), seed=cfg.seed, config=cfg,
               creator_script="scripts/03_finalize_tapes.py",
               input_artifacts=[str(p) for p in train_intermediates])

    written = splits.materialize_split(
        ticker=ticker, splits=sp,
        intermediate_dir=inter_dir, output_dir=out_dir, quantiles=q,
    )
    for split_name, paths in written.items():
        log.info("materialized %s: %d files", split_name, len(paths))


if __name__ == "__main__":
    main()
