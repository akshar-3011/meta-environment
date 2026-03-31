"""Tests for advanced extensibility features."""

from __future__ import annotations

import asyncio
import time

from workplace_env.core.benchmarking import BenchmarkRunner
from workplace_env.core.graders.interfaces import BaseGrader, EvaluationContext, GraderResult
from workplace_env.core.graders.plugins import GraderPluginRegistry
from workplace_env.core.inference.cache import InMemoryTTLCache, make_cache_key
from workplace_env.core.visualization import ascii_bar_chart, benchmark_report


class _DummyStrategy:
    def build_actions(self, observation):  # noqa: ARG002
        return [
            ("classify", "complaint"),
            ("reply", "We are sorry and will resolve this quickly."),
            ("escalate", "yes"),
        ]


class _CustomGrader(BaseGrader):
    @property
    def name(self) -> str:
        return "custom"

    def grade(self, context: EvaluationContext) -> GraderResult:  # noqa: ARG002
        return GraderResult(score=0.75, explanation="custom plugin")


def test_cache_key_is_deterministic_for_same_payload():
    payload = {"b": 2, "a": 1}

    key1 = make_cache_key("infer", payload)
    key2 = make_cache_key("infer", {"a": 1, "b": 2})

    assert key1 == key2


def test_in_memory_ttl_cache_expires_entries():
    cache = InMemoryTTLCache(ttl_seconds=0.02, max_entries=10)
    cache.set("x", {"value": 1})
    assert cache.get("x") == {"value": 1}

    time.sleep(0.03)
    assert cache.get("x") is None


def test_grader_plugin_registry_register_and_create():
    registry = GraderPluginRegistry()
    registry.register("custom", _CustomGrader)

    instance = registry.create("custom")

    assert isinstance(instance, BaseGrader)
    assert instance.name == "custom"


def test_benchmark_runner_sync_and_async_modes():
    scenarios = [
        {
            "email": "This is frustrating and unresolved",
            "label": "complaint",
            "difficulty": "easy",
            "urgency": "high",
            "sentiment": "negative",
            "complexity": 2,
            "min_reply_length": 30,
        }
    ]
    strategies = {
        "standard": _DummyStrategy(),
        "enhanced": _DummyStrategy(),
    }

    runner = BenchmarkRunner(strategies=strategies, scenarios=scenarios)

    sync_summary = runner.run_sync(iterations=2)
    async_summary = asyncio.run(runner.run_async(iterations=2, concurrency=2))

    assert sync_summary["total_runs"] == 4
    assert async_summary["total_runs"] == 4
    assert set(sync_summary["per_strategy"].keys()) == {"standard", "enhanced"}
    assert set(async_summary["per_strategy"].keys()) == {"standard", "enhanced"}


def test_visualization_outputs_expected_labels():
    chart = ascii_bar_chart({"a": 0.2, "b": 0.8}, width=10)
    assert "a" in chart
    assert "b" in chart

    report = benchmark_report(
        {
            "total_runs": 4,
            "elapsed_seconds": 0.1,
            "per_strategy": {
                "standard": {"avg": 0.8},
                "enhanced": {"avg": 0.7},
            },
            "ranking": [
                {"strategy": "standard", "avg": 0.8},
                {"strategy": "enhanced", "avg": 0.7},
            ],
        }
    )

    assert "BENCHMARK SUMMARY" in report
    assert "standard" in report
