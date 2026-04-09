#!/usr/bin/env python3
"""Performance benchmark suite for the workplace environment.

Runs three benchmark modes:
  1. **Direct** — In-process episode execution (no HTTP overhead)
  2. **HTTP**  — Full HTTP round-trip against a running server
  3. **Load**  — Concurrent HTTP episodes to test throughput

Usage:
    # Direct benchmark (no server needed):
    python benchmarks/load_test.py --mode direct

    # HTTP benchmark (start server first):
    python benchmarks/load_test.py --mode http --url http://localhost:8000

    # Full load test (concurrent episodes):
    python benchmarks/load_test.py --mode load --url http://localhost:8000 --concurrency 20

    # All modes:
    python benchmarks/load_test.py --mode all --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    latency_ms: float
    total_reward: float
    success: bool
    steps: int = 3
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    mode: str
    episodes: int
    concurrency: int
    results: List[EpisodeResult] = field(default_factory=list)
    wall_clock_s: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def latencies(self) -> List[float]:
        return sorted(r.latency_ms for r in self.results if r.success)

    @property
    def p50(self) -> float:
        lat = self.latencies
        return lat[len(lat) // 2] if lat else 0.0

    @property
    def p95(self) -> float:
        lat = self.latencies
        idx = int(len(lat) * 0.95)
        return lat[min(idx, len(lat) - 1)] if lat else 0.0

    @property
    def p99(self) -> float:
        lat = self.latencies
        idx = int(len(lat) * 0.99)
        return lat[min(idx, len(lat) - 1)] if lat else 0.0

    @property
    def mean(self) -> float:
        lat = self.latencies
        return statistics.mean(lat) if lat else 0.0

    @property
    def stddev(self) -> float:
        lat = self.latencies
        return statistics.stdev(lat) if len(lat) > 1 else 0.0

    @property
    def throughput(self) -> float:
        return self.episodes / self.wall_clock_s if self.wall_clock_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results) * 100

    def print_report(self):
        print(f"\n{'=' * 60}")
        print(f"  BENCHMARK REPORT — {self.mode.upper()}")
        print(f"{'=' * 60}")
        print(f"  Episodes:       {self.episodes}")
        print(f"  Concurrency:    {self.concurrency}")
        print(f"  Wall clock:     {self.wall_clock_s:.3f}s")
        print(f"  Success rate:   {self.success_rate:.1f}%")
        print(f"  Peak memory:    {self.peak_memory_mb:.1f} MB")
        print()
        print(f"  ┌─── Latency (ms) ───────────────────┐")
        print(f"  │  P50:    {self.p50:>8.2f}                  │")
        print(f"  │  P95:    {self.p95:>8.2f}                  │")
        print(f"  │  P99:    {self.p99:>8.2f}                  │")
        print(f"  │  Mean:   {self.mean:>8.2f} ± {self.stddev:.2f}        │")
        print(f"  └────────────────────────────────────┘")
        print()
        print(f"  Throughput:     {self.throughput:.1f} episodes/sec")

        # Pass/fail against targets
        print()
        _check("P50 < 200ms", self.p50 < 200)
        _check("P99 < 500ms", self.p99 < 500)
        _check(f"Throughput > 100 eps/s", self.throughput > 100)
        _check(f"Memory < 50 MB", self.peak_memory_mb < 50)
        _check(f"Success rate 100%", self.success_rate == 100.0)
        print(f"{'=' * 60}\n")


def _check(label: str, passed: bool):
    icon = "✅" if passed else "❌"
    print(f"  {icon} {label}")


# ─── Direct benchmark (in-process) ──────────────────────────────────────────

def run_direct_benchmark(num_episodes: int = 100) -> BenchmarkReport:
    """Run episodes directly against the environment — no HTTP."""
    from environment.workplace_environment import WorkplaceEnvironment
    from models import WorkplaceAction

    report = BenchmarkReport(mode="direct", episodes=num_episodes, concurrency=1)

    # Warm up
    env = WorkplaceEnvironment()
    env.reset()
    env.step(WorkplaceAction(action_type="classify", content="refund"))
    env.step(WorkplaceAction(action_type="reply", content="Thanks."))
    env.step(WorkplaceAction(action_type="escalate", content="no"))

    gc.collect()
    tracemalloc.start()
    wall_start = time.perf_counter()

    for _ in range(num_episodes):
        t0 = time.perf_counter()
        try:
            env.reset()
            r1 = env.step(WorkplaceAction(action_type="classify", content="refund"))
            r2 = env.step(WorkplaceAction(action_type="reply",
                content="Thank you for reaching out. We sincerely apologize and will resolve this."))
            r3 = env.step(WorkplaceAction(action_type="escalate", content="no"))
            latency = (time.perf_counter() - t0) * 1000
            total_reward = (r1.reward or 0) + (r2.reward or 0) + (r3.reward or 0)
            report.results.append(EpisodeResult(
                latency_ms=latency, total_reward=total_reward, success=True
            ))
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            report.results.append(EpisodeResult(
                latency_ms=latency, total_reward=0, success=False, error=str(exc)
            ))

    report.wall_clock_s = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report.peak_memory_mb = peak / 1024 / 1024

    return report


# ─── HTTP benchmark (sequential) ────────────────────────────────────────────

async def run_http_benchmark(base_url: str, num_episodes: int = 50) -> BenchmarkReport:
    """Run episodes via HTTP — sequential to measure per-episode latency."""
    import httpx

    report = BenchmarkReport(mode="http", episodes=num_episodes, concurrency=1)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Warm up
        await client.post("/reset", json={})
        await client.post("/step", json={"action": {"action_type": "classify", "content": "refund"}})

        tracemalloc.start()
        wall_start = time.perf_counter()

        for _ in range(num_episodes):
            result = await _run_http_episode(client)
            report.results.append(result)

        report.wall_clock_s = time.perf_counter() - wall_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report.peak_memory_mb = peak / 1024 / 1024

    return report


async def _run_http_episode(client) -> EpisodeResult:
    """Run a single 3-step episode via HTTP."""
    t0 = time.perf_counter()
    try:
        await client.post("/reset", json={})
        total_reward = 0.0

        actions = [
            {"action_type": "classify", "content": "refund"},
            {"action_type": "reply", "content": "Thank you. We sincerely apologize and will resolve this quickly."},
            {"action_type": "escalate", "content": "no"},
        ]

        for action in actions:
            resp = await client.post("/step", json={"action": action})
            data = resp.json()
            total_reward += data.get("reward", 0) or 0

        latency = (time.perf_counter() - t0) * 1000
        return EpisodeResult(latency_ms=latency, total_reward=total_reward, success=True)
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return EpisodeResult(latency_ms=latency, total_reward=0, success=False, error=str(exc))


# ─── Load test (concurrent) ─────────────────────────────────────────────────

async def run_load_test(
    base_url: str, num_episodes: int = 200, concurrency: int = 20
) -> BenchmarkReport:
    """Run many episodes concurrently to measure throughput under load."""
    import httpx

    report = BenchmarkReport(mode="load", episodes=num_episodes, concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_episode(client):
        async with semaphore:
            return await _run_http_episode(client)

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        tracemalloc.start()
        wall_start = time.perf_counter()

        tasks = [_bounded_episode(client) for _ in range(num_episodes)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        report.wall_clock_s = time.perf_counter() - wall_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report.peak_memory_mb = peak / 1024 / 1024

        for r in results:
            if isinstance(r, EpisodeResult):
                report.results.append(r)
            else:
                report.results.append(EpisodeResult(
                    latency_ms=0, total_reward=0, success=False, error=str(r)
                ))

    return report


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Workplace Env Performance Benchmark")
    parser.add_argument("--mode", choices=["direct", "http", "load", "all"], default="direct",
                        help="Benchmark mode (default: direct)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL for HTTP/load modes (default: http://localhost:8000)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Concurrent episode count for load mode (default: 20)")
    args = parser.parse_args()

    print(f"\n🏋️ Workplace Env Benchmark — mode={args.mode}, episodes={args.episodes}")
    print(f"{'─' * 60}")

    reports: List[BenchmarkReport] = []

    if args.mode in ("direct", "all"):
        print("\n⏱  Running DIRECT benchmark (in-process)...")
        report = run_direct_benchmark(num_episodes=args.episodes)
        report.print_report()
        reports.append(report)

    if args.mode in ("http", "all"):
        print("\n⏱  Running HTTP benchmark (sequential)...")
        report = asyncio.run(run_http_benchmark(args.url, num_episodes=min(args.episodes, 50)))
        report.print_report()
        reports.append(report)

    if args.mode in ("load", "all"):
        print(f"\n⏱  Running LOAD test ({args.concurrency} concurrent)...")
        report = asyncio.run(run_load_test(args.url, num_episodes=args.episodes, concurrency=args.concurrency))
        report.print_report()
        reports.append(report)

    # Summary
    if reports:
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY")
        print(f"{'=' * 60}")
        for r in reports:
            print(f"  {r.mode:8s}  P50={r.p50:.1f}ms  P99={r.p99:.1f}ms  "
                  f"tput={r.throughput:.0f} eps/s  mem={r.peak_memory_mb:.1f}MB")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
