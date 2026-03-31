"""Async execution utilities for controlled concurrent workloads."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, List, TypeVar


T = TypeVar("T")


async def run_limited(tasks: Iterable[Callable[[], Awaitable[T]]], concurrency: int = 4) -> List[T]:
    """Run awaitable task factories with bounded concurrency."""
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return list(await asyncio.gather(*[_run(factory) for factory in tasks]))
