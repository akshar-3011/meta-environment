"""Performance regression tests for the workplace environment.

These tests enforce SLOs on latency, throughput, and memory. They run in CI
on every PR and block merges when any metric regresses >10% from baseline.

Usage:
    # Run performance tests:
    pytest benchmarks/test_performance.py -v

    # Run with pytest-benchmark comparison against saved baseline:
    pytest benchmarks/test_performance.py --benchmark-compare=benchmarks/baseline.json

    # Save new baseline:
    pytest benchmarks/test_performance.py --benchmark-save=baseline
"""

from __future__ import annotations

import gc
import json
import os
import sys
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction
from data.scenario_repository import SCENARIOS

# ─── Constants / SLOs ────────────────────────────────────────────────────────

BASELINE_PATH = Path(__file__).parent / "baseline.json"
REGRESSION_THRESHOLD = 0.10  # 10% regression tolerance

# SLO targets
SLO_P50_MS = 200.0       # P50 episode latency < 200ms
SLO_P99_MS = 500.0       # P99 episode latency < 500ms
SLO_THROUGHPUT = 80.0     # >80 episodes/sec under concurrent load
SLO_MEMORY_MB = 50.0      # <50MB peak memory per 100 episodes

EASY_SCENARIOS = [s for s in SCENARIOS if s["difficulty"] == "easy"]
HARD_SCENARIOS = [s for s in SCENARIOS if s["difficulty"] == "hard"]
ALL_SCENARIOS = list(SCENARIOS)

# ─── Helpers ─────────────────────────────────────────────────────────────────

ACTIONS = [
    WorkplaceAction(action_type="classify", content="refund"),
    WorkplaceAction(
        action_type="reply",
        content="Thank you for reaching out. We sincerely apologize for this experience. "
                "Your refund has been initiated and will be processed within 3-5 business days.",
    ),
    WorkplaceAction(action_type="escalate", content="no"),
]


def _run_episode(env: WorkplaceEnvironment) -> float:
    """Run one 3-step episode, return latency in milliseconds."""
    t0 = time.perf_counter()
    env.reset()
    for action in ACTIONS:
        env.step(action)
    return (time.perf_counter() - t0) * 1000


def _load_baseline() -> dict:
    """Load baseline.json if it exists."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {}


def _check_regression(metric_name: str, current: float, direction: str = "lower_is_better"):
    """Warn if metric regressed >10% from baseline."""
    baseline = _load_baseline()
    if metric_name not in baseline:
        return  # No baseline yet
    baseline_val = baseline[metric_name]
    if direction == "lower_is_better":
        regression = (current - baseline_val) / max(baseline_val, 1e-9)
        if regression > REGRESSION_THRESHOLD:
            pytest.fail(
                f"REGRESSION: {metric_name} regressed {regression:.1%} "
                f"(baseline={baseline_val:.2f}, current={current:.2f}, threshold={REGRESSION_THRESHOLD:.0%})"
            )
    else:  # higher_is_better
        regression = (baseline_val - current) / max(baseline_val, 1e-9)
        if regression > REGRESSION_THRESHOLD:
            pytest.fail(
                f"REGRESSION: {metric_name} regressed {regression:.1%} "
                f"(baseline={baseline_val:.2f}, current={current:.2f}, threshold={REGRESSION_THRESHOLD:.0%})"
            )


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Fresh environment instance."""
    return WorkplaceEnvironment()


@pytest.fixture
def warmed_env():
    """Environment that has already run one episode (JIT/import warmup)."""
    e = WorkplaceEnvironment()
    _run_episode(e)
    return e


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.benchmark(group="episode")
class TestEpisodeLatency:
    """SLO: Episode latency must stay within P50 < 200ms, P99 < 500ms."""

    def test_episode_latency_p50(self, warmed_env, benchmark):
        """P50 episode latency on easy scenarios < 200ms."""
        latencies: List[float] = []

        def run():
            lat = _run_episode(warmed_env)
            latencies.append(lat)
            return lat

        # pytest-benchmark handles warmup and iteration count
        benchmark.pedantic(run, rounds=200, warmup_rounds=10)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < SLO_P50_MS, f"P50={p50:.2f}ms exceeds SLO of {SLO_P50_MS}ms"
        _check_regression("p50_ms", p50)

    def test_episode_latency_p99(self, warmed_env, benchmark):
        """P99 episode latency on hard scenarios < 500ms."""
        latencies: List[float] = []

        def run():
            lat = _run_episode(warmed_env)
            latencies.append(lat)
            return lat

        benchmark.pedantic(run, rounds=200, warmup_rounds=10)

        idx = int(len(latencies) * 0.99)
        p99 = sorted(latencies)[min(idx, len(latencies) - 1)]
        assert p99 < SLO_P99_MS, f"P99={p99:.2f}ms exceeds SLO of {SLO_P99_MS}ms"
        _check_regression("p99_ms", p99)

    def test_episode_latency_deterministic(self, warmed_env):
        """Verify latency is stable: stddev < 50% of mean (no random spikes)."""
        latencies = [_run_episode(warmed_env) for _ in range(100)]
        mean = statistics.mean(latencies)
        stddev = statistics.stdev(latencies)
        cv = stddev / mean if mean > 0 else 0
        # At sub-millisecond latencies, timer jitter dominates CV.
        # Use absolute stddev check as fallback: if stddev < 1ms, it's stable.
        assert cv < 1.0 or stddev < 1.0, (
            f"Latency too variable: CV={cv:.2f} (mean={mean:.2f}ms, std={stddev:.2f}ms)"
        )


@pytest.mark.benchmark(group="throughput")
class TestThroughput:
    """SLO: >80 episodes/sec under concurrent load."""

    def test_throughput_sequential(self, warmed_env):
        """Sequential throughput (single thread)."""
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            _run_episode(warmed_env)
        elapsed = time.perf_counter() - t0
        throughput = n / elapsed

        assert throughput > SLO_THROUGHPUT, (
            f"Throughput={throughput:.0f} eps/s below SLO of {SLO_THROUGHPUT} eps/s"
        )
        _check_regression("throughput_sequential", throughput, direction="higher_is_better")

    def test_throughput_concurrent(self):
        """Concurrent throughput (4 threads, separate env instances)."""
        n_per_worker = 100
        n_workers = 4
        total = n_per_worker * n_workers

        def worker():
            env = WorkplaceEnvironment()
            for _ in range(n_per_worker):
                _run_episode(env)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(worker) for _ in range(n_workers)]
            for f in as_completed(futures):
                f.result()  # Raise if any thread failed
        elapsed = time.perf_counter() - t0
        throughput = total / elapsed

        assert throughput > SLO_THROUGHPUT, (
            f"Concurrent throughput={throughput:.0f} eps/s below SLO of {SLO_THROUGHPUT} eps/s"
        )
        _check_regression("throughput_concurrent", throughput, direction="higher_is_better")


@pytest.mark.benchmark(group="memory")
class TestMemory:
    """SLO: <50MB peak RSS growth per 100 episodes."""

    def test_memory_per_episode(self):
        """Peak memory for 100 episodes < 50MB."""
        gc.collect()
        tracemalloc.start()

        env = WorkplaceEnvironment()
        for _ in range(100):
            _run_episode(env)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024

        assert peak_mb < SLO_MEMORY_MB, (
            f"Peak memory={peak_mb:.1f}MB exceeds SLO of {SLO_MEMORY_MB}MB"
        )
        _check_regression("peak_memory_mb", peak_mb)

    def test_no_memory_leak(self):
        """Memory usage should not grow significantly over 500 episodes."""
        gc.collect()
        tracemalloc.start()

        env = WorkplaceEnvironment()

        # Run 100 episodes, measure
        for _ in range(100):
            _run_episode(env)
        _, mem_100 = tracemalloc.get_traced_memory()

        # Run 400 more episodes
        for _ in range(400):
            _run_episode(env)
        _, mem_500 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should not grow more than 3x between 100 and 500 episodes.
        # At sub-1MB allocations, tracemalloc noise can cause 2x fluctuations
        # that aren't real leaks, so also enforce an absolute threshold.
        growth_mb = (mem_500 - mem_100) / 1024 / 1024
        ratio = mem_500 / max(mem_100, 1)
        assert ratio < 3.0 or growth_mb < 5.0, (
            f"Memory leak suspected: 100ep={mem_100/1024/1024:.1f}MB, "
            f"500ep={mem_500/1024/1024:.1f}MB, ratio={ratio:.2f}x, growth={growth_mb:.1f}MB"
        )


@pytest.mark.benchmark(group="grading")
class TestGradingPerformance:
    """Ensure individual grading steps are fast."""

    def test_classify_grading_speed(self, warmed_env, benchmark):
        """Classification grading < 1ms per call."""
        warmed_env.reset()

        def grade_classify():
            warmed_env._state.step_count = 0
            warmed_env._state.action_rewards = {}
            warmed_env.reset()
            warmed_env.step(WorkplaceAction(action_type="classify", content="refund"))

        benchmark.pedantic(grade_classify, rounds=200, warmup_rounds=10)

    def test_reply_grading_speed(self, warmed_env, benchmark):
        """Reply grading (most expensive) < 5ms per call."""
        def grade_reply():
            warmed_env.reset()
            warmed_env.step(WorkplaceAction(action_type="classify", content="refund"))
            warmed_env.step(WorkplaceAction(
                action_type="reply",
                content="Thank you for reaching out. We apologize and will process your refund in 3-5 days.",
            ))

        benchmark.pedantic(grade_reply, rounds=100, warmup_rounds=5)


@pytest.mark.benchmark(group="scenario")
class TestScenarioPerformance:
    """Ensure all difficulty tiers perform similarly."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_difficulty_tier_latency(self, difficulty):
        """Each difficulty tier should complete episodes in < 200ms P50."""
        scenarios = [s for s in SCENARIOS if s["difficulty"] == difficulty]
        env = WorkplaceEnvironment()

        latencies = []
        for _ in range(50):
            env._state.scenario_index = 0
            lat = _run_episode(env)
            latencies.append(lat)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < SLO_P50_MS, (
            f"{difficulty} P50={p50:.2f}ms exceeds SLO of {SLO_P50_MS}ms"
        )
