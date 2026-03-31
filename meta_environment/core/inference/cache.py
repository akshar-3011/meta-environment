"""Caching primitives for inference and pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Generic, Optional, TypeVar


T = TypeVar("T")


def make_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    """Build a deterministic cache key from a namespace and payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class InMemoryTTLCache(Generic[T]):
    """Thread-safe in-memory cache with TTL and bounded size."""

    def __init__(self, *, ttl_seconds: float = 300.0, max_entries: int = 1024):
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._data: Dict[str, _CacheEntry[T]] = {}
        self._lock = Lock()

    def _now(self) -> float:
        return time.time()

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [k for k, v in self._data.items() if v.expires_at <= now]
        for key in expired:
            self._data.pop(key, None)

    def _evict_if_needed(self) -> None:
        if len(self._data) < self.max_entries:
            return
        oldest_key = min(self._data.keys(), key=lambda k: self._data[k].expires_at)
        self._data.pop(oldest_key, None)

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            self._purge_expired()
            entry = self._data.get(key)
            if not entry:
                return None
            return entry.value

    def set(self, key: str, value: T, *, ttl_seconds: Optional[float] = None) -> T:
        ttl = self.ttl_seconds if ttl_seconds is None else max(0.0, float(ttl_seconds))
        with self._lock:
            self._purge_expired()
            self._evict_if_needed()
            self._data[key] = _CacheEntry(value=value, expires_at=self._now() + ttl)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
