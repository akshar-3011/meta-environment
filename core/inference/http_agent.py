"""Backward-compatible runner shim over strategy-based inference system."""

from typing import List, Optional, Tuple

from .base import DEFAULT_BASE_URL, RetryConfig
from .strategies import EnhancedInference, StandardInference


def run_agent(
    actions: Optional[List[Tuple[str, str]]] = None,
    reveal_label: bool = False,
    title: str = "WORKPLACE ENVIRONMENT AGENT",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
    max_attempts: int = 3,
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
