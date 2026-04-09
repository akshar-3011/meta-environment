"""OpenTelemetry distributed tracing for the workplace environment.

Auto-instruments FastAPI with span creation for every request.
Exports traces to any OTLP-compatible backend (Jaeger, Tempo, Honeycomb, etc.)
via the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.

When no endpoint is configured, traces are silently discarded (no-op exporter).
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from fastapi import FastAPI

LOGGER = logging.getLogger(__name__)


def setup_tracing(app: FastAPI, service_name: str = "workplace-env") -> None:
    """Bootstrap OpenTelemetry tracing on the FastAPI app.

    Configuration via environment variables (standard OTEL env vars):
        - ``OTEL_EXPORTER_OTLP_ENDPOINT`` — gRPC or HTTP endpoint
          (e.g. ``http://jaeger:4317``, ``https://api.honeycomb.io``)
        - ``OTEL_EXPORTER_OTLP_HEADERS`` — auth headers
          (e.g. ``x-honeycomb-team=<your-key>``)
        - ``OTEL_SERVICE_NAME`` — overrides ``service_name`` param
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("APP_ENV", "development"),
        })

        provider = TracerProvider(resource=resource)

        # Only add an exporter if an endpoint is configured
        if otlp_endpoint:
            _add_exporter(provider, otlp_endpoint)
            LOGGER.info("OpenTelemetry tracing enabled → %s", otlp_endpoint)
        else:
            LOGGER.info("OpenTelemetry tracing initialized (no exporter — traces discarded)")

        trace.set_tracer_provider(provider)

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


def get_tracer(name: str = "workplace-env"):
    """Get a tracer for manual span creation in business logic."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None
