"""Reproducibility sidecars: every artifact gets `{path}.meta.json`.

The sidecar lets us trace every number in the report back to a git SHA, a
config, a seed, and the upstream artifacts that produced it (Work0.md §6).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import schema as _schema
from ..utils import git as _git


def config_hash(cfg: Any) -> str:
    """sha256 of a config (dict, OmegaConf DictConfig, or arbitrary)."""
    if hasattr(cfg, "to_yaml"):
        text = cfg.to_yaml()
    elif hasattr(cfg, "to_container"):
        text = json.dumps(cfg.to_container(resolve=True), sort_keys=True, default=str)
    else:
        text = json.dumps(cfg, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def write_meta(
    artifact_path: str | Path,
    *,
    seed: int | None = None,
    config: Any = None,
    creator_script: str | None = None,
    input_artifacts: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    artifact_path = Path(artifact_path)
    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    meta = {
        "git_sha": _git.current_sha(),
        "git_dirty": _git.is_dirty(),
        "config_hash": config_hash(config) if config is not None else None,
        "schema_version": _schema.SCHEMA_VERSION,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "creator_script": creator_script,
        "input_artifacts": list(input_artifacts) if input_artifacts else [],
    }
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta_path


def read_meta(artifact_path: str | Path) -> dict:
    p = Path(artifact_path)
    return json.loads(p.with_suffix(p.suffix + ".meta.json").read_text())
