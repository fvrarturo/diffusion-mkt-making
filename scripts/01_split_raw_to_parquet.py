"""Stage A0 — split a TAQ gzip CSV into per-date Parquet files.

One-shot per gzip. Run twice for each ticker (quotes + trades).

Local example:
    python scripts/01_split_raw_to_parquet.py \\
        +gz=data/raw/order_INTC.csv.gz +kind=quotes +ticker=INTC

Cluster:
    sbatch slurms/ph1_split_raw.slurm \\
        +gz=data/raw/order_INTC.csv.gz +kind=quotes +ticker=INTC
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from diffmm.data.raw_split import split_gzip_by_date
from diffmm.utils.logging import get_logger

log = get_logger("ph1_split_raw")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    gz = Path(cfg.gz) if "gz" in cfg else Path(cfg.data.raw_quotes_gz)
    kind = str(cfg.get("kind", "quotes"))
    ticker = str(cfg.get("ticker", cfg.data.ticker))

    out_dir = Path(cfg.data.raw_partitioned_dir) / ticker / kind
    log.info("splitting %s → %s/", gz, out_dir)

    written = split_gzip_by_date(gz, out_dir, kind=kind)
    log.info("wrote %d per-date parquet files", len(written))
    if written:
        log.info("first: %s   last: %s", written[0].name, written[-1].name)


if __name__ == "__main__":
    main()
