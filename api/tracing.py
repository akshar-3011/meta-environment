"""OpenTelemetry distributed tracing for the workplace environment.

Auto-instruments FastAPI with span creation for every request.
Exports traces to any OTLP-compatible backend (Jaeger, Tempo, Honeycomb, etc.)

Configuration is read from ``core.config.TracingConfig``:
- ``service_name``:   "meta-environment" (override via OTEL_SERVICE_NAME)
- ``otlp_endpoint``:  gRPC/HTTP endpoint (override via OTEL_EXPORTER_OTLP_ENDPOINT)
- ``sample_rate``:    0.1 for production, 1.0 for development (override via OTEL_TRACES_SAMPLE_RATE)
- ``enabled``:        true/false (override via OTEL_TRACING_ENABLED)

When no endpoint is configured, traces are silently discarded (no-op exporter).
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from fastapi import FastAPI

LOGGER = logging.getLogger(__name__)

# Lazy-loaded tracer reference
_TRACER = None


def setup_tracing(app: FastAPI) -> None:
    """Bootstrap OpenTelemetry tracing on the FastAPI app.

    Reads ``TracingConfig`` from the centralized config for service name,
    sample rate, and export endpoint.
    """
    try:
        from core.config import get_config
    except ImportError:
        try:
            from ..core.config import get_config
        except ImportError:
            LOGGER.warning("Cannot import config — tracing disabled")
            return

    cfg = get_config()
    tc = cfg.tracing

    if not tc.enabled:
        LOGGER.info("OpenTelemetry tracing disabled via config")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({
            "service.name": tc.service_name,
            "service.version": "1.0.0",
            "deployment.environment": cfg.env,
        })

        sampler = TraceIdRatioBased(tc.sample_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        LOGGER.info(
            "OpenTelemetry tracing: service=%s sample_rate=%.0f%% env=%s",
            tc.service_name, tc.sample_rate * 100, cfg.env,
        )

        # Only add an exporter if an endpoint is configured
        if tc.otlp_endpoint:
            _add_exporter(provider, tc.otlp_endpoint)
            LOGGER.info("OTLP export → %s", tc.otlp_endpoint)
        else:
            LOGGER.info("No OTLP endpoint configured — traces not exported")

        trace.set_tracer_provider(provider)

        # Cache global tracer reference
        global _TRACER
        _TRACER = trace.get_tracer(tc.service_name)

        # Auto-instrument FastAPI
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,metrics,docs,openapi.json,redoc",
        )

    except ImportError:
        LOGGER.warning("opentelemetry packages not installed — tracing disabled")
    except Exception as exc:
        LOGGER.warning("Failed to initialize tracing: %s", exc)


def _add_exporter(provider, endpoint: str) -> None:
    """Add OTLP exporter (gRPC preferred, HTTP fallback)."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure="localhost" in endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except ImportError:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = HTTPExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            LOGGER.warning("No OTLP exporter available — install opentelemetry-exporter-otlp")


def get_tracer(name: str = "meta-environment"):
    """Get a tracer for manual span creation in business logic.

    Returns the cached tracer if tracing was initialized, or a no-op tracer.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None


def get_current_trace_context() -> dict:
    """Extract current trace_id and span_id for log injection.

    Returns ``{"trace_id": "...", "span_id": "..."}`` or empty dict if
    no active span exists.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
    except (ImportError, Exception):
        pass
    return {}
