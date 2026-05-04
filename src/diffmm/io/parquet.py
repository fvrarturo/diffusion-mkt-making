"""Single source of truth for canonical event-tape Parquet I/O.

Validates the schema on read and write, and routes every read through the
holdout firewall.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema
from .holdout_guard import assert_path_allowed


def write_tape(df: pl.DataFrame, path: str | Path, *, validate: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if validate:
        schema.validate(df)
    df.write_parquet(path, compression="zstd", statistics=True)
    return path


def read_tape(path: str | Path, *, validate: bool = True) -> pl.DataFrame:
    assert_path_allowed(path)
    df = pl.read_parquet(path)
    if validate:
        schema.validate(df)
    return df


def scan_tape(path: str | Path) -> pl.LazyFrame:
    """Lazy read; schema validation deferred to materialization."""
    assert_path_allowed(path)
    return pl.scan_parquet(path)
