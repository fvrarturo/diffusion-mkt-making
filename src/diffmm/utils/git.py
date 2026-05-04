"""Thin wrappers around git for reproducibility tagging."""
from __future__ import annotations

import subprocess


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def current_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"]) or "unknown"


def current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def is_dirty() -> bool:
    return bool(_run(["git", "status", "--porcelain"]))
