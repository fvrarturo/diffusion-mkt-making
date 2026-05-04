"""Minimal structured logger — single format, single sink (stdout)."""
from __future__ import annotations

import logging
import sys


_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(h)
        logger.setLevel(getattr(logging, level))
        logger.propagate = False
    return logger
