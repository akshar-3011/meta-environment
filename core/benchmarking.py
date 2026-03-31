"""Benchmarking utilities for strategy/model comparison."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .async_execution import run_limited
from .graders.rule_based import RuleBasedRewardPolicy


Action = Tuple[str, str]


@dataclass(frozen=True)
class BenchmarkRunResult:
    strategy: str
    score: float
    action_count: int
    scenario_label: str


class BenchmarkRunner:
    """Compare inference strategies over scenario datasets."""

    def __init__(
        self,
        *,
        strategies: Mapping[str, Any],
        scenarios: Iterable[Dict[str, Any]],
        policy: RuleBasedRewardPolicy | None = None,
    ):
        self._strategies = dict(strategies)
        self._scenarios = list(scenarios)
        self._policy = policy or RuleBasedRewardPolicy()

    def _make_observation(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "email": scenario.get("email", ""),
            "category_options": ["refund", "complaint", "query"],
            "scenario_difficulty": scenario.get("difficulty", "easy"),
            "urgency": scenario.get("urgency", "medium"),
            "sentiment": scenario.get("sentiment", "neutral"),
            "complexity_score": scenario.get("complexity", 2),
            "scenario_metadata": {
                "min_reply_length": scenario.get("min_reply_length", 30),
            },
        }

    def _evaluate_actions(self, actions: List[Action], scenario: Dict[str, Any]) -> float:
        previous_actions: Dict[str, float] = {}
        total = 0.0
        label = scenario.get("label", "query")

        for idx, (action_type, content) in enumerate(actions, start=1):
            reward, _ = self._policy.calculate_step_reward(
                action_type=action_type,
                content=content,
                actual_category=label,
                step_count=min(idx, 3),
                scenario_difficulty=scenario.get("difficulty", "easy"),
                min_reply_length=scenario.get("min_reply_length", 30),
                previous_actions=previous_actions,
            )
            previous_actions[action_type] = reward
            total += reward
        return total

    def _run_single(self, strategy_name: str, strategy: Any, scenario: Dict[str, Any]) -> BenchmarkRunResult:
        obs = self._make_observation(scenario)
        actions: List[Action] = strategy.build_actions(obs)
        score = self._evaluate_actions(actions, scenario)
        return BenchmarkRunResult(
            strategy=strategy_name,
            score=score,
            action_count=len(actions),
            scenario_label=scenario.get("label", "unknown"),
        )

    def _summarize(self, runs: List[BenchmarkRunResult], elapsed_seconds: float) -> Dict[str, Any]:
        grouped: Dict[str, List[float]] = {}
        for run in runs:
            grouped.setdefault(run.strategy, []).append(run.score)

        per_strategy: Dict[str, Dict[str, float]] = {}
        for strategy, scores in grouped.items():
            per_strategy[strategy] = {
                "avg": float(statistics.mean(scores)),
                "min": float(min(scores)),
                "max": float(max(scores)),
                "std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
                "runs": float(len(scores)),
            }

        ranking = sorted(
            [{"strategy": k, "avg": v["avg"]} for k, v in per_strategy.items()],
            key=lambda x: x["avg"],
            reverse=True,
        )

        return {
            "elapsed_seconds": elapsed_seconds,
            "total_runs": len(runs),
            "per_strategy": per_strategy,
            "ranking": ranking,
        }

    def run_sync(self, *, iterations: int = 1) -> Dict[str, Any]:
        start = time.perf_counter()
        runs: List[BenchmarkRunResult] = []
        for _ in range(max(1, iterations)):
            for scenario in self._scenarios:
                for strategy_name, strategy in self._strategies.items():
                    runs.append(self._run_single(strategy_name, strategy, scenario))
        elapsed = time.perf_counter() - start
        return self._summarize(runs, elapsed)

    async def run_async(self, *, iterations: int = 1, concurrency: int = 4) -> Dict[str, Any]:
        start = time.perf_counter()

        task_factories = []
        for _ in range(max(1, iterations)):
            for scenario in self._scenarios:
                for strategy_name, strategy in self._strategies.items():
                    task_factories.append(
                        lambda sname=strategy_name, strat=strategy, sc=scenario: asyncio.to_thread(
                            self._run_single,
                            sname,
                            strat,
                            sc,
                        )
                    )

        runs = await run_limited(task_factories, concurrency=concurrency)
        elapsed = time.perf_counter() - start
        return self._summarize(list(runs), elapsed)
