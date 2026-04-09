"""Production middleware: API key auth, CORS, rate limiting, Prometheus metrics."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from ..core.config import get_config
except ImportError:  # pragma: no cover
    from core.config import get_config


# ─── Prometheus-style metrics (no external dependency) ───────────────────────

class _Metrics:
    """In-process counters for /metrics scraping."""

    def __init__(self):
        self.request_count: int = 0
        self.error_count: int = 0
        self.latency_sum: float = 0.0
        self.latency_count: int = 0
        self.status_counts: Dict[int, int] = defaultdict(int)
        self.endpoint_counts: Dict[str, int] = defaultdict(int)

    def record(self, path: str, status: int, latency: float):
        self.request_count += 1
        self.latency_sum += latency
        self.latency_count += 1
        self.status_counts[status] += 1
        self.endpoint_counts[path] += 1
        if status >= 400:
            self.error_count += 1

    def to_prometheus(self) -> str:
        lines: List[str] = []
        lines.append(f"# HELP http_requests_total Total HTTP requests")
        lines.append(f"# TYPE http_requests_total counter")
        lines.append(f"http_requests_total {self.request_count}")
        lines.append(f"# HELP http_errors_total Total HTTP errors (4xx+5xx)")
        lines.append(f"# TYPE http_errors_total counter")
        lines.append(f"http_errors_total {self.error_count}")
        lines.append(f"# HELP http_request_duration_seconds_sum Sum of request durations")
        lines.append(f"# TYPE http_request_duration_seconds_sum counter")
        lines.append(f"http_request_duration_seconds_sum {self.latency_sum:.6f}")
        lines.append(f"# HELP http_request_duration_seconds_count Count of timed requests")
        lines.append(f"# TYPE http_request_duration_seconds_count counter")
        lines.append(f"http_request_duration_seconds_count {self.latency_count}")
        for status, count in sorted(self.status_counts.items()):
            lines.append(f'http_responses_total{{status="{status}"}} {count}')
        for path, count in sorted(self.endpoint_counts.items()):
            lines.append(f'http_endpoint_requests_total{{path="{path}"}} {count}')
        return "\n".join(lines) + "\n"


METRICS = _Metrics()


# ─── Rate limiter (sliding window per IP) ────────────────────────────────────

class _RateLimiter:
    """Simple in-memory sliding-window rate limiter (per IP)."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        if self.max_requests <= 0:
            return True  # Disabled
        now = time.monotonic()
        cutoff = now - self.window
        hits = self._hits[client_ip]
        # Prune old entries
        self._hits[client_ip] = [t for t in hits if t > cutoff]
        if len(self._hits[client_ip]) >= self.max_requests:
            return False
        self._hits[client_ip].append(now)
        return True


# ─── Middleware classes ───────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid API key (skip health/docs/metrics)."""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics", "/"}

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable):
        if self.api_key and request.url.path not in self.EXEMPT_PATHS:
            provided = (
                request.headers.get("X-API-Key")
                or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            )
            if provided != self.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP rate limiting."""

    def __init__(self, app, limiter: _RateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count, latency, and status codes."""

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.monotonic()
        response: Response = await call_next(request)
        latency = time.monotonic() - start
        METRICS.record(request.url.path, response.status_code, latency)
        return response


# ─── Wiring ──────────────────────────────────────────────────────────────────

def apply_production_middleware(app: FastAPI) -> FastAPI:
    """Apply all production middleware + /metrics endpoint to a FastAPI app.

    Reads from the global config to decide what to enable.
    """
    cfg = get_config()
    sec = cfg.security

    # 1. Metrics (innermost — records everything)
    app.add_middleware(MetricsMiddleware)

    # 2. Rate limiting
    if sec.rate_limit_per_minute > 0:
        limiter = _RateLimiter(max_requests=sec.rate_limit_per_minute)
        app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # 3. CORS
    if sec.cors_origins:
        origins = [o.strip() for o in sec.cors_origins.split(",") if o.strip()]
    else:
        origins = []  # No CORS headers at all when empty

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # 4. API key auth (outermost)
    if sec.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=sec.api_key)

    # 5. /metrics endpoint
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return METRICS.to_prometheus()

    return app
