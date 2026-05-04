"""Generate task manifests for SLURM array jobs.

Each manifest is a JSON list whose index maps SLURM_ARRAY_TASK_ID to a
specific (regime, seed) — or whatever else the array job needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig


def make_sample_manifest(regimes: list[str], n_seeds: int) -> list[dict]:
    return [
        {"regime": regime, "seed": seed}
        for regime in regimes
        for seed in range(n_seeds)
    ]


def make_clean_manifest(tickers: list[str], raw_partitioned_dir: Path) -> list[dict]:
    """Enumerate (ticker, date) pairs from already-partitioned raw quotes."""
    entries: list[dict] = []
    for ticker in tickers:
        quote_dir = raw_partitioned_dir / ticker / "quotes"
        if not quote_dir.exists():
            raise FileNotFoundError(
                f"no partitioned quotes for {ticker} at {quote_dir} — "
                "run scripts/01_split_raw_to_parquet.py first"
            )
        for p in sorted(quote_dir.glob("*.parquet")):
            entries.append({"ticker": ticker, "date": p.stem})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sample", "clean"], default="sample")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--regimes", nargs="+",
        default=["base", "high_vol", "toxic", "thin"],
    )
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--tickers", nargs="+", default=["INTC", "TSLA"])
    ap.add_argument(
        "--raw-partitioned-dir",
        default="data/raw_partitioned",
        type=Path,
    )
    args = ap.parse_args()

    if args.kind == "sample":
        manifest = make_sample_manifest(args.regimes, args.n_seeds)
    elif args.kind == "clean":
        manifest = make_clean_manifest(args.tickers, args.raw_partitioned_dir)
    else:
        raise ValueError(args.kind)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} entries to {out}")
    print(f"sbatch with: --array=0-{len(manifest)-1}")


if __name__ == "__main__":
    main()
