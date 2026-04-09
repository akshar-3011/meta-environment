"""Production middleware: security headers, auth, rate limiting, Prometheus metrics, tracing.

Security features:
  - Content-Security-Policy: default-src 'self'
  - Strict-Transport-Security: max-age=31536000; includeSubDomains
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - X-XSS-Protection: 0 (deprecated, CSP replaces it)
  - Referrer-Policy: strict-origin-when-cross-origin
  - Error sanitization (no stack traces to clients)
  - Request body size limit (1MB default)
  - Per-endpoint + global rate limiting
  - Audit logging for auth and rate limit events
"""

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


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_REQUEST_BODY_SIZE = 1 * 1024 * 1024  # 1MB

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store, no-cache, must-revalidate",
}


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

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers into every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)

        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding MAX_REQUEST_BODY_SIZE."""

    def __init__(self, app, max_size: int = MAX_REQUEST_BODY_SIZE):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            # Audit log
            try:
                from security.audit_logging import log_request_too_large
                client_ip = request.client.host if request.client else "unknown"
                log_request_too_large(client_ip, request.url.path, int(content_length), self.max_size)
            except (ImportError, Exception):
                pass

            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request body too large",
                    "max_size_bytes": self.max_size,
                },
            )
        return await call_next(request)


class ErrorSanitizationMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return generic error (no stack traces)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response: Response = await call_next(request)

            # Sanitize 500-level error bodies in production
            if response.status_code >= 500:
                try:
                    from security.audit_logging import log_error_sanitized
                    client_ip = request.client.host if request.client else "unknown"
                    log_error_sanitized(client_ip, request.url.path, response.status_code)
                except (ImportError, Exception):
                    pass

                cfg = get_config()
                if cfg.env == "production":
                    return JSONResponse(
                        status_code=response.status_code,
                        content={
                            "error": "Internal server error",
                            "request_id": getattr(request.state, "request_id", ""),
                        },
                    )

            return response

        except Exception:
            # Catch truly unhandled exceptions
            try:
                from security.audit_logging import log_error_sanitized
                client_ip = request.client.host if request.client else "unknown"
                log_error_sanitized(client_ip, request.url.path, 500, "unhandled_exception")
            except (ImportError, Exception):
                pass

            request_id = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                },
            )


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
        client_ip = request.client.host if request.client else "unknown"
        request_id = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""

        if self.api_key and request.url.path not in self.EXEMPT_PATHS:
            provided = (
                request.headers.get("X-API-Key")
                or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            )
            if provided != self.api_key:
                # Audit log: auth failure
                try:
                    from security.audit_logging import log_auth_failure
                    log_auth_failure(
                        client_ip, request_id, provided or "",
                        request.url.path, "invalid_api_key",
                    )
                except (ImportError, Exception):
                    pass

                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )

            # Audit log: auth success
            try:
                from security.audit_logging import log_auth_success
                log_auth_success(client_ip, request_id, provided, request.url.path)
            except (ImportError, Exception):
                pass

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
                headers={"Retry-After": "60"},
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

    Middleware execution order (outermost → innermost):
      1. API Key Auth (reject unauthenticated requests first)
      2. CORS (handle preflight)
      3. Strict Rate Limiting (per-endpoint)
      4. Global Rate Limiting (per-IP fallback)
      5. Request Size Limit (reject oversized bodies)
      6. Error Sanitization (catch exceptions, strip stack traces)
      7. Request ID + Trace Context
      8. Security Headers (inject headers into every response)
      9. Prometheus Metrics (innermost — records everything)
    """
    cfg = get_config()
    sec = cfg.security

    # 9. Prometheus metrics (innermost — records everything)
    app.add_middleware(PrometheusMiddleware)

    # 8. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 7. Request-ID + trace context injection
    app.add_middleware(RequestIDMiddleware)

    # 6. Error sanitization
    app.add_middleware(ErrorSanitizationMiddleware)

    # 5. Request body size limit (1MB)
    app.add_middleware(RequestSizeLimitMiddleware, max_size=MAX_REQUEST_BODY_SIZE)

    # 4. Global rate limiting (per-IP fallback)
    if sec.rate_limit_per_minute > 0:
        limiter = _RateLimiter(max_requests=sec.rate_limit_per_minute)
        app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # 3. Strict per-endpoint rate limiting
    try:
        from security.rate_limit_strict import StrictRateLimitMiddleware
        app.add_middleware(StrictRateLimitMiddleware)
    except ImportError:
        pass

    # 2. CORS
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

    # 1. API key auth (outermost)
    if sec.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=sec.api_key)

    # /metrics endpoint (Prometheus exposition format)
    register_metrics_endpoint(app)

    # OpenTelemetry distributed tracing (deferred to avoid circular import)
    try:
        from api.tracing import setup_tracing  # noqa: E402
    except ImportError:
        setup_tracing = None  # type: ignore[assignment]

    if setup_tracing is not None:
        setup_tracing(app)

    return app
