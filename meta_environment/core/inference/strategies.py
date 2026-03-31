"""Concrete inference strategies built on top of BaseInference."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import BaseInference


class StandardInference(BaseInference):
    """Default straightforward inference strategy."""

    @property
    def title(self) -> str:
        return "WORKPLACE ENVIRONMENT AGENT"

    def build_actions(self, observation: Dict[str, Any]) -> List[Tuple[str, str]]:
        return [
            ("classify", "complaint"),
            (
                "reply",
                "We sincerely apologize for the issue you experienced. We understand your frustration and "
                "will resolve this immediately. Our team will contact you within 24 hours with a solution.",
            ),
            ("escalate", "yes"),
        ]


class EnhancedInference(StandardInference):
    """Verbose strategy that reveals metadata labels for diagnostics."""

    @property
    def title(self) -> str:
        return "ENHANCED WORKPLACE ENVIRONMENT AGENT"

    @property
    def reveal_label(self) -> bool:
        return True


class AsyncInference(StandardInference):
    """Async strategy for concurrent batch processing.

    Uses `asyncio.to_thread` over the sync HTTP implementation so we keep
    dependencies minimal while still enabling parallel episode execution.
    """

    @property
    def title(self) -> str:
        return "ASYNC WORKPLACE ENVIRONMENT AGENT"

    async def run_episode_async(
        self,
        actions: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self.run_episode, actions)

    async def run_batch_async(
        self,
        batch_actions: Iterable[Optional[Sequence[Tuple[str, str]]]],
    ) -> List[Optional[Dict[str, Any]]]:
        tasks = [self.run_episode_async(actions=actions) for actions in batch_actions]
        return await asyncio.gather(*tasks)
