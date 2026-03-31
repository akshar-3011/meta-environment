"""Centralized configuration loader (.env + environment variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env parser (no external dependency required)."""
    dotenv_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    server_port: int = 8000
    pipeline_port: int = 8010
    max_concurrent_envs: int = 1


@dataclass(frozen=True)
class InferenceConfig:
    base_url: str = "http://localhost:8000"
    timeout_seconds: float = 10.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class EnvironmentConfig:
    debug: bool = False


@dataclass(frozen=True)
class AppConfig:
    env: str
    logging: LoggingConfig
    api: ApiConfig
    inference: InferenceConfig
    environment: EnvironmentConfig


def load_config() -> AppConfig:
    _load_dotenv()

    return AppConfig(
        env=os.getenv("APP_ENV", "development"),
        logging=LoggingConfig(
            level=os.getenv("APP_LOG_LEVEL", "INFO"),
            format=os.getenv(
                "APP_LOG_FORMAT",
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            ),
        ),
        api=ApiConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            server_port=_get_int("API_SERVER_PORT", 8000),
            pipeline_port=_get_int("API_PIPELINE_PORT", 8010),
            max_concurrent_envs=_get_int("API_MAX_CONCURRENT_ENVS", 1),
        ),
        inference=InferenceConfig(
            base_url=os.getenv("INFERENCE_BASE_URL", "http://localhost:8000"),
            timeout_seconds=_get_float("INFERENCE_TIMEOUT_SECONDS", 10.0),
            retry_attempts=_get_int("INFERENCE_RETRY_ATTEMPTS", 3),
            retry_backoff_seconds=_get_float("INFERENCE_RETRY_BACKOFF_SECONDS", 0.5),
        ),
        environment=EnvironmentConfig(
            debug=_get_bool("ENV_DEBUG", False),
        ),
    )


_CONFIG = load_config()


def get_config() -> AppConfig:
    return _CONFIG
