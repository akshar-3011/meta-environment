"""Centralized logging bootstrap for the project."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .config import get_config


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line — ideal for production log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: Optional[str] = None) -> None:
    cfg = get_config()
    log_cfg = cfg.logging
    resolved_level = (level or log_cfg.level).upper()

    root = logging.getLogger()
    if root.handlers:
        return  # Already configured — avoid duplicate handlers

    handler = logging.StreamHandler()

    # Use JSON in production, human-readable otherwise
    if cfg.env == "production":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(log_cfg.format))

    root.setLevel(getattr(logging, resolved_level, logging.INFO))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
