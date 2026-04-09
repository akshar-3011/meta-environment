"""Production middleware: API key auth, CORS, rate limiting, Prometheus metrics, request-ID tracing."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from ..core.config import get_config
except ImportError:  # pragma: no cover
    from core.config import get_config

try:
    from ..api.metrics import (
        REQUEST_COUNT, REQUEST_LATENCY, register_metrics_endpoint,
    )
except ImportError:  # pragma: no cover
    from api.metrics import (
        REQUEST_COUNT, REQUEST_LATENCY, register_metrics_endpoint,
    )


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

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate X-Request-ID if missing, inject into response + request state.

    Downstream handlers can access via ``request.state.request_id``.
    The trace context (trace_id, span_id) is also attached when OTel is active.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Use provided request ID or generate one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Attach trace context if available
        try:
            from api.tracing import get_current_trace_context
            trace_ctx = get_current_trace_context()
            request.state.trace_id = trace_ctx.get("trace_id", "")
            request.state.span_id = trace_ctx.get("span_id", "")
        except (ImportError, Exception):
            request.state.trace_id = ""
            request.state.span_id = ""

        response: Response = await call_next(request)

        # Inject into response headers for correlation
        response.headers["X-Request-ID"] = request_id
        if request.state.trace_id:
            response.headers["X-Trace-ID"] = request.state.trace_id

        return response


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


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count, latency, and status via prometheus_client."""

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.monotonic()
        response: Response = await call_next(request)
        latency = time.monotonic() - start

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        return response


# ─── Wiring ──────────────────────────────────────────────────────────────────

def apply_production_middleware(app: FastAPI) -> FastAPI:
    """Apply all production middleware + /metrics + tracing to a FastAPI app.

    Reads from the global config to decide what to enable.
    """
    cfg = get_config()
    sec = cfg.security

    # 1. Prometheus metrics (innermost — records everything)
    app.add_middleware(PrometheusMiddleware)

    # 2. Request-ID + trace context injection
    app.add_middleware(RequestIDMiddleware)

    # 3. Rate limiting
    if sec.rate_limit_per_minute > 0:
        limiter = _RateLimiter(max_requests=sec.rate_limit_per_minute)
        app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # 4. CORS
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

    # 5. API key auth (outermost)
    if sec.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=sec.api_key)

    # 6. /metrics endpoint (Prometheus exposition format)
    register_metrics_endpoint(app)

    # 7. OpenTelemetry distributed tracing (deferred to avoid circular import)
    try:
        from api.tracing import setup_tracing  # noqa: E402
    except ImportError:
        setup_tracing = None  # type: ignore[assignment]

    if setup_tracing is not None:
        setup_tracing(app)

    return app
