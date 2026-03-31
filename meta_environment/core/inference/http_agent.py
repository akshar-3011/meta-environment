"""Backward-compatible runner shim over strategy-based inference system."""

from typing import List, Optional, Tuple

try:
    from ..config import get_config
except ImportError:  # pragma: no cover
    from core.config import get_config

from .base import DEFAULT_BASE_URL, RetryConfig
from .strategies import EnhancedInference, StandardInference


_CFG = get_config()


def run_agent(
    actions: Optional[List[Tuple[str, str]]] = None,
    reveal_label: bool = False,
    title: str = "WORKPLACE ENVIRONMENT AGENT",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = _CFG.inference.timeout_seconds,
    max_attempts: int = _CFG.inference.retry_attempts,
):
    """Legacy entrypoint preserved for existing scripts.

    Internally dispatches to `StandardInference` or `EnhancedInference`.
    """
    retry = RetryConfig(max_attempts=max_attempts)
    strategy = (
        EnhancedInference(base_url=base_url, timeout=timeout, retry=retry)
        if reveal_label
        else StandardInference(base_url=base_url, timeout=timeout, retry=retry)
    )

    if title and title != strategy.title:
        class _CustomTitleStrategy(type(strategy)):  # pragma: no cover
            @property
            def title(self) -> str:  # type: ignore[override]
                return title

        strategy = _CustomTitleStrategy(base_url=base_url, timeout=timeout, retry=retry)

    return strategy.run_episode(actions=actions)
