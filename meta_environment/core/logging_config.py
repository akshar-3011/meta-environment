"""Centralized logging bootstrap for the project."""

from __future__ import annotations

import logging
from typing import Optional

from .config import get_config


def setup_logging(level: Optional[str] = None) -> None:
    cfg = get_config().logging
    resolved_level = (level or cfg.level).upper()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=cfg.format,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
