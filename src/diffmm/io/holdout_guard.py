"""Holdout firewall.

Holdout files live in `data/processed/holdout/`. They must not be touched
outside of the explicit Phase-5 holdout-opening script
(`scripts/40_open_holdout.py`). The guard wraps any path resolution that
goes near the holdout directory.

Usage:
    from diffmm.io import holdout_guard

    # Allowed (unwraps must be brief and reasoned):
    with holdout_guard.unlocked(reason="phase 5 hypothesis test"):
        df = read_tape(holdout_path)

    # Disallowed — raises HoldoutLeakageError:
    df = read_tape(holdout_path)
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
from typing import Iterator

HOLDOUT_DIR_NAME = "holdout"
_unlocked: bool = False


class HoldoutLeakageError(RuntimeError):
    pass


@contextmanager
def unlocked(reason: str) -> Iterator[None]:
    global _unlocked
    if not reason:
        raise ValueError("must give a reason for unlocking the holdout")
    print(f"[holdout_guard] UNLOCKING holdout — reason: {reason}", flush=True)
    _unlocked = True
    try:
        yield
    finally:
        _unlocked = False
        print("[holdout_guard] holdout re-locked", flush=True)


def assert_path_allowed(path: str | os.PathLike) -> None:
    """Call this inside any tape-opening helper. No-op for non-holdout paths."""
    p = Path(path).resolve()
    if HOLDOUT_DIR_NAME in p.parts and not _unlocked:
        raise HoldoutLeakageError(
            f"attempted to read holdout file {p} without unlocking. "
            "Wrap the read in `with holdout_guard.unlocked(reason=...):`"
        )
