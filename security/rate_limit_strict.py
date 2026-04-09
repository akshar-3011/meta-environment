"""Strict per-endpoint rate limiter with sliding window and Retry-After headers.

Provides granular rate limiting per endpoint and per API key, supplementing
the global per-IP rate limiter in middleware.py.

Configuration (via environment):
    RATE_LIMIT_RESET=10       /reset: 10/min per IP
    RATE_LIMIT_STEP=100       /step: 100/min per IP
    RATE_LIMIT_INFER=30       /infer: 30/min per IP
    RATE_LIMIT_METRICS=5      /metrics: 5/min per IP (auth required)
    RATE_LIMIT_GLOBAL=1000    global: 1000/min per API key
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EndpointLimit:
    """Rate limit for a specific endpoint pattern."""
    path_prefix: str
    max_requests: int
    window_seconds: int = 60


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# Default per-endpoint limits
DEFAULT_ENDPOINT_LIMITS = [
    EndpointLimit("/reset", _get_int("RATE_LIMIT_RESET", 10)),
    EndpointLimit("/step", _get_int("RATE_LIMIT_STEP", 100)),
    EndpointLimit("/infer", _get_int("RATE_LIMIT_INFER", 30)),
    EndpointLimit("/metrics", _get_int("RATE_LIMIT_METRICS", 5)),
    EndpointLimit("/experiments", _get_int("RATE_LIMIT_EXPERIMENTS", 30)),
]

DEFAULT_GLOBAL_LIMIT = _get_int("RATE_LIMIT_GLOBAL", 1000)


# ─── Sliding Window Rate Limiter ─────────────────────────────────────────────

class SlidingWindowLimiter:
    """Thread-safe sliding window rate limiter.

    Uses sorted timestamp lists with periodic pruning. Memory-bounded
    by evicting entries for IPs not seen in 2× the window.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_gc = time.monotonic()
        self._gc_interval = max(window_seconds * 2, 120)

    def check(self, key: str) -> Tuple[bool, int, float]:
        """Check if request is allowed.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        if self.max_requests <= 0:
            return True, 0, 0.0

        now = time.monotonic()
        cutoff = now - self.window

        with self._lock:
            # Periodic garbage collection
            if now - self._last_gc > self._gc_interval:
                self._gc(cutoff)
                self._last_gc = now

            hits = self._hits[key]
            # Prune expired
            self._hits[key] = [t for t in hits if t > cutoff]
            hits = self._hits[key]

            remaining = max(0, self.max_requests - len(hits))

            if len(hits) >= self.max_requests:
                # Calculate retry-after: time until oldest hit expires
                oldest = hits[0] if hits else now
                retry_after = max(0.0, (oldest + self.window) - now)
                return False, 0, math.ceil(retry_after)

            hits.append(now)
            return True, remaining - 1, 0.0

    def _gc(self, cutoff: float) -> None:
        """Remove stale entries to bound memory."""
        stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._hits[k]


# ─── Composite Rate Limiter ─────────────────────────────────────────────────

class CompositeRateLimiter:
    """Manages per-endpoint + global rate limiters."""

    def __init__(
        self,
        endpoint_limits: Optional[List[EndpointLimit]] = None,
        global_limit: int = DEFAULT_GLOBAL_LIMIT,
    ):
        self._endpoint_limiters: List[Tuple[str, SlidingWindowLimiter]] = []
        for el in (endpoint_limits or DEFAULT_ENDPOINT_LIMITS):
            self._endpoint_limiters.append(
                (el.path_prefix, SlidingWindowLimiter(el.max_requests, el.window_seconds))
            )
        self._global_limiter = SlidingWindowLimiter(global_limit, 60)

    def check(
        self,
        path: str,
        client_ip: str,
        api_key: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, str]]:
        """Check all applicable rate limits.

        Returns:
            (allowed, response_headers)
        """
        headers: Dict[str, str] = {}

        # 1. Check per-endpoint limit (keyed by IP)
        for prefix, limiter in self._endpoint_limiters:
            if path.startswith(prefix):
                allowed, remaining, retry_after = limiter.check(client_ip)
                headers["X-RateLimit-Limit"] = str(limiter.max_requests)
                headers["X-RateLimit-Remaining"] = str(remaining)
                if not allowed:
                    headers["Retry-After"] = str(int(retry_after))
                    headers["X-RateLimit-Scope"] = f"endpoint:{prefix}"
                    return False, headers
                break

        # 2. Check global limit (keyed by API key or IP)
        global_key = api_key or client_ip
        allowed, remaining, retry_after = self._global_limiter.check(global_key)
        headers.setdefault("X-RateLimit-Limit", str(self._global_limiter.max_requests))
        headers["X-RateLimit-Global-Remaining"] = str(remaining)
        if not allowed:
            headers["Retry-After"] = str(int(retry_after))
            headers["X-RateLimit-Scope"] = "global"
            return False, headers

        return True, headers


# ─── Middleware ───────────────────────────────────────────────────────────────

class StrictRateLimitMiddleware(BaseHTTPMiddleware):
    """Per-endpoint + global rate limiting with Retry-After headers."""

    # Paths exempt from rate limiting
    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/"}

    def __init__(self, app: FastAPI, limiter: Optional[CompositeRateLimiter] = None):
        super().__init__(app)
        self.limiter = limiter or CompositeRateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip rate limiting for health/docs endpoints
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        api_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or None
        )

        allowed, headers = self.limiter.check(path, client_ip, api_key)

        if not allowed:
            # Log rate limit violation (picked up by audit_logging)
            try:
                from security.audit_logging import log_rate_limit_violation
                log_rate_limit_violation(client_ip, path, api_key, headers)
            except (ImportError, Exception):
                pass

            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(headers.get("Retry-After", 60)),
                    "scope": headers.get("X-RateLimit-Scope", "unknown"),
                },
            )
            for k, v in headers.items():
                response.headers[k] = v
            return response

        response = await call_next(request)

        # Add rate limit headers to successful responses
        for k, v in headers.items():
            response.headers[k] = v

        return response
