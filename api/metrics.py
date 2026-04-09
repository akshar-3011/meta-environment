"""Prometheus metrics for the workplace environment.

Exports counters, histograms, and gauges that are scraped via ``GET /metrics``.
Uses the official ``prometheus_client`` library for full Prometheus compatibility
(exposition format, HELP/TYPE annotations, label cardinality, etc.).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response


# ─── Counters ────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "env_requests_total",
    "Total HTTP requests to the environment server",
    ["endpoint", "method", "status"],
)

EPISODE_COUNT = Counter(
    "env_episodes_total",
    "Total RL episodes completed (done=true)",
)

STEP_COUNT = Counter(
    "env_steps_total",
    "Total environment steps executed",
    ["action_type"],
)

ERROR_COUNT = Counter(
    "env_errors_total",
    "Total errors (grading failures, invalid actions, etc.)",
    ["error_type"],
)

# ─── Histograms ──────────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "env_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

REWARD_HISTOGRAM = Histogram(
    "env_reward_distribution",
    "Distribution of rewards per step",
    ["step", "difficulty"],
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EPISODE_SCORE = Histogram(
    "env_episode_score",
    "Distribution of total episode scores",
    ["difficulty"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ─── Gauges ──────────────────────────────────────────────────────────────────

ACTIVE_EPISODES = Gauge(
    "env_active_episodes",
    "Number of currently active (in-progress) episodes",
)

SCENARIO_POOL_SIZE = Gauge(
    "env_scenario_pool_size",
    "Total number of scenarios in the pool",
    ["difficulty"],
)

# ─── Step-level instrumentation ──────────────────────────────────────────────

STEP_LATENCY = Histogram(
    "env_step_duration_seconds",
    "Latency of individual environment steps (classify/reply/escalate)",
    ["action_type", "difficulty"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
)

ESCALATION_DECISIONS = Counter(
    "env_escalation_decisions_total",
    "Total escalation decisions by outcome",
    ["decision"],  # "escalated" or "not_escalated"
)

GRADER_LATENCY = Histogram(
    "env_grader_duration_seconds",
    "Latency of grader evaluation per step",
    ["action_type"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
)


# ─── /metrics endpoint ──────────────────────────────────────────────────────

def register_metrics_endpoint(app: FastAPI) -> None:
    """Register ``GET /metrics`` that returns Prometheus exposition format."""

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
