"""Command-line interface for inference, grading, and end-to-end pipeline runs."""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

import typer

from .core.benchmarking import BenchmarkRunner
from .core.config import get_config
from .core.exceptions import WorkplaceEnvError
from .core.graders import CATEGORY_OPTIONS, RuleBasedRewardPolicy, create_default_registry
from .core.inference import AsyncInference, EnhancedInference, InMemoryTTLCache, StandardInference, make_cache_key
from .core.visualization import benchmark_report, write_jsonl
from .data import get_default_repository


class InferenceStrategy(str, Enum):
    standard = "standard"
    enhanced = "enhanced"
    async_strategy = "async"


app = typer.Typer(help="Workplace Env CLI")
CFG = get_config()
_INFERENCE_CACHE: InMemoryTTLCache[Dict[str, Any]] = InMemoryTTLCache(
    ttl_seconds=CFG.cache.inference_ttl_seconds,
    max_entries=CFG.cache.max_entries,
)


def _emit_json(payload: Dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=False))


def _fail(message: str, *, code: str = "CLI_ERROR", details: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }
    _emit_json(payload)
    raise typer.Exit(code=1)


def _select_strategy(strategy: InferenceStrategy):
    if strategy == InferenceStrategy.standard:
        return StandardInference()
    if strategy == InferenceStrategy.enhanced:
        return EnhancedInference()
    return AsyncInference()


def _cache_lookup_or_run(
    *,
    namespace: str,
    payload: Dict[str, Any],
    enabled: bool,
    compute: Any,
) -> Dict[str, Any]:
    if not enabled or not CFG.cache.enabled:
        result = compute()
        result.setdefault("meta", {})
        result["meta"]["cache"] = {"enabled": False, "hit": False}
        return result

    key = make_cache_key(namespace, payload)
    cached = _INFERENCE_CACHE.get(key)
    if cached is not None:
        result = dict(cached)
        result.setdefault("meta", {})
        result["meta"]["cache"] = {"enabled": True, "hit": True, "key": key}
        return result

    result = compute()
    _INFERENCE_CACHE.set(key, result)
    result.setdefault("meta", {})
    result["meta"]["cache"] = {"enabled": True, "hit": False, "key": key}
    return result


def _run_inference_impl(
    *,
    email: str,
    strategy: InferenceStrategy,
    scenario_difficulty: str,
    urgency: str,
    sentiment: str,
    complexity_score: int,
    use_cache: bool,
) -> Dict[str, Any]:
    payload = {
        "email": email,
        "strategy": strategy.value,
        "scenario_difficulty": scenario_difficulty,
        "urgency": urgency,
        "sentiment": sentiment,
        "complexity_score": complexity_score,
    }

    def _compute() -> Dict[str, Any]:
        engine = _select_strategy(strategy)
        observation = {
            "email": email,
            "category_options": list(CATEGORY_OPTIONS),
            "scenario_difficulty": scenario_difficulty,
            "urgency": urgency,
            "sentiment": sentiment,
            "complexity_score": complexity_score,
            "scenario_metadata": {"min_reply_length": 30},
        }
        actions = engine.build_actions(observation)
        return {
            "success": True,
            "score": 1.0,
            "breakdown": {
                "strategy": strategy.value,
                "email": email,
                "observation": observation,
                "action_count": len(actions),
                "actions": [{"action_type": a, "content": c} for a, c in actions],
            },
        }

    return _cache_lookup_or_run(
        namespace="infer",
        payload=payload,
        enabled=use_cache,
        compute=_compute,
    )


def _run_grader_impl(
    *,
    action_type: str,
    content: str,
    actual_category: str,
    step_count: int,
    scenario_difficulty: str,
    min_reply_length: int,
    previous_actions: Dict[str, float],
) -> Dict[str, Any]:
    policy = RuleBasedRewardPolicy()
    score, breakdown = policy.calculate_step_reward(
        action_type=action_type,
        content=content,
        actual_category=actual_category,
        step_count=step_count,
        scenario_difficulty=scenario_difficulty,
        min_reply_length=min_reply_length,
        previous_actions=previous_actions,
    )
    return {"success": True, "score": score, "breakdown": breakdown}


def _run_pipeline_impl(
    *,
    email: str,
    actual_category: str,
    strategy: InferenceStrategy,
    scenario_difficulty: str,
    min_reply_length: int,
    use_cache: bool,
    plugin_paths: Sequence[str],
    plugin_weight: float,
) -> Dict[str, Any]:
    payload = {
        "email": email,
        "actual_category": actual_category,
        "strategy": strategy.value,
        "scenario_difficulty": scenario_difficulty,
        "min_reply_length": min_reply_length,
        "plugins": list(plugin_paths),
        "plugin_weight": plugin_weight,
    }

    def _compute() -> Dict[str, Any]:
        registry = create_default_registry()
        plugin_graders = []
        for plugin_path in plugin_paths:
            loaded_name = registry.load_from_path(plugin_path)
            plugin_graders.append((registry.create(loaded_name), plugin_weight))

        extra = None
        if plugin_graders:
            extra = {
                "classify": list(plugin_graders),
                "reply": list(plugin_graders),
                "escalate": list(plugin_graders),
            }

        engine = _select_strategy(strategy)
        policy = RuleBasedRewardPolicy(extra_graders=extra)

        observation = {
            "email": email,
            "category_options": list(CATEGORY_OPTIONS),
            "scenario_difficulty": scenario_difficulty,
            "urgency": "medium",
            "sentiment": "neutral",
            "complexity_score": 2,
            "scenario_metadata": {"min_reply_length": min_reply_length},
        }

        actions = engine.build_actions(observation)
        previous_actions: Dict[str, float] = {}
        cumulative = 0.0
        steps: List[Dict[str, Any]] = []

        for idx, (action_type, content) in enumerate(actions, start=1):
            reward, step_breakdown = policy.calculate_step_reward(
                action_type=action_type,
                content=content,
                actual_category=actual_category,
                step_count=min(idx, 3),
                scenario_difficulty=scenario_difficulty,
                min_reply_length=min_reply_length,
                previous_actions=previous_actions,
            )
            previous_actions[action_type] = reward
            cumulative += reward
            steps.append(
                {
                    "step": idx,
                    "action_type": action_type,
                    "content": content,
                    "score": reward,
                    "breakdown": step_breakdown,
                }
            )

        return {
            "success": True,
            "score": cumulative,
            "breakdown": {
                "strategy": strategy.value,
                "email": email,
                "actual_category": actual_category,
                "steps": steps,
                "action_rewards": previous_actions,
                "total_steps": len(steps),
                "plugins": [g.name for g, _ in plugin_graders],
            },
        }

    return _cache_lookup_or_run(
        namespace="pipeline",
        payload=payload,
        enabled=use_cache,
        compute=_compute,
    )


@app.command("run-inference")
def run_inference(
    email: str = typer.Option(..., help="Customer email text"),
    strategy: InferenceStrategy = typer.Option(InferenceStrategy.standard, help="Inference strategy"),
    scenario_difficulty: str = typer.Option("easy", help="Scenario difficulty: easy/medium/hard"),
    urgency: str = typer.Option("medium", help="Urgency metadata"),
    sentiment: str = typer.Option("neutral", help="Sentiment metadata"),
    complexity_score: int = typer.Option(2, min=1, max=5, help="Complexity score (1-5)"),
    no_cache: bool = typer.Option(False, help="Disable cache for this run"),
) -> None:
    """Generate agent actions for an email."""
    try:
        _emit_json(
            _run_inference_impl(
                email=email,
                strategy=strategy,
                scenario_difficulty=scenario_difficulty,
                urgency=urgency,
                sentiment=sentiment,
                complexity_score=complexity_score,
                use_cache=not no_cache,
            )
        )
    except WorkplaceEnvError as exc:
        _fail(str(exc), code=exc.code, details=exc.details)
    except Exception as exc:
        _fail("Inference command failed", details={"exception": str(exc)})


@app.command("run-grader")
def run_grader(
    action_type: str = typer.Option(..., help="Action type: classify/reply/escalate"),
    content: str = typer.Option(..., help="Action content to evaluate"),
    actual_category: str = typer.Option(..., help="Ground-truth category: refund/complaint/query"),
    step_count: int = typer.Option(1, min=1, max=3, help="Pipeline step number"),
    scenario_difficulty: str = typer.Option("easy", help="Scenario difficulty"),
    min_reply_length: int = typer.Option(30, min=1, help="Minimum expected reply length"),
    previous_actions_json: str = typer.Option("{}", help='JSON map of previous action scores, e.g. {"classify": 0.3}'),
) -> None:
    """Grade a single action with reward breakdown."""
    try:
        previous_actions = json.loads(previous_actions_json)
    except json.JSONDecodeError as exc:
        _fail("Invalid JSON for --previous-actions-json", details={"exception": str(exc)})
        return

    try:
        _emit_json(
            _run_grader_impl(
                action_type=action_type,
                content=content,
                actual_category=actual_category,
                step_count=step_count,
                scenario_difficulty=scenario_difficulty,
                min_reply_length=min_reply_length,
                previous_actions=previous_actions,
            )
        )
    except WorkplaceEnvError as exc:
        _fail(str(exc), code=exc.code, details=exc.details)
    except Exception as exc:
        _fail("Grader command failed", details={"exception": str(exc)})


@app.command("run-pipeline")
def run_pipeline(
    email: str = typer.Option(..., help="Customer email text"),
    actual_category: str = typer.Option(..., help="Ground-truth category: refund/complaint/query"),
    strategy: InferenceStrategy = typer.Option(InferenceStrategy.standard, help="Inference strategy"),
    scenario_difficulty: str = typer.Option("easy", help="Scenario difficulty"),
    min_reply_length: int = typer.Option(30, min=1, help="Minimum expected reply length"),
    no_cache: bool = typer.Option(False, help="Disable cache for this run"),
    plugin: List[str] = typer.Option([], help="Plugin grader path(s) in module:attribute format"),
    plugin_weight: float = typer.Option(0.05, min=0.0, max=1.0, help="Weight for each plugin grader"),
) -> None:
    """Run full inference + grading pipeline."""
    try:
        _emit_json(
            _run_pipeline_impl(
                email=email,
                actual_category=actual_category,
                strategy=strategy,
                scenario_difficulty=scenario_difficulty,
                min_reply_length=min_reply_length,
                use_cache=not no_cache,
                plugin_paths=plugin,
                plugin_weight=plugin_weight,
            )
        )
    except WorkplaceEnvError as exc:
        _fail(str(exc), code=exc.code, details=exc.details)
    except Exception as exc:
        _fail("Pipeline command failed", details={"exception": str(exc)})


@app.command("run-benchmark")
def run_benchmark(
    strategies: str = typer.Option("standard,enhanced,async", help="Comma-separated strategies"),
    iterations: int = typer.Option(CFG.benchmark.default_runs, min=1, help="Benchmark iterations over all scenarios"),
    async_run: bool = typer.Option(False, help="Run benchmark with async concurrency"),
    concurrency: int = typer.Option(CFG.benchmark.default_concurrency, min=1, help="Async concurrency"),
    plugin: List[str] = typer.Option([], help="Plugin grader path(s) in module:attribute format"),
    plugin_weight: float = typer.Option(0.05, min=0.0, max=1.0, help="Weight for each plugin grader"),
    show_chart: bool = typer.Option(True, help="Include ASCII chart in output"),
    log_file: str = typer.Option("", help="Optional path to write JSONL benchmark logs"),
) -> None:
    """Compare strategies/models and emit benchmark metrics + visualization."""
    chosen = [item.strip() for item in strategies.split(",") if item.strip()]
    registry: Dict[str, Any] = {
        "standard": StandardInference(),
        "enhanced": EnhancedInference(),
        "async": AsyncInference(),
    }

    missing = [name for name in chosen if name not in registry]
    if missing:
        _fail("Unknown strategy requested", details={"missing": missing, "available": list(registry.keys())})

    selected = {name: registry[name] for name in chosen}

    plugin_registry = create_default_registry()
    plugin_graders = []
    for plugin_path in plugin:
        loaded_name = plugin_registry.load_from_path(plugin_path)
        plugin_graders.append((plugin_registry.create(loaded_name), plugin_weight))

    extra = None
    if plugin_graders:
        extra = {
            "classify": list(plugin_graders),
            "reply": list(plugin_graders),
            "escalate": list(plugin_graders),
        }

    scenarios = get_default_repository().list_scenarios()
    runner = BenchmarkRunner(
        strategies=selected,
        scenarios=scenarios,
        policy=RuleBasedRewardPolicy(extra_graders=extra),
    )

    try:
        if async_run:
            summary = asyncio.run(runner.run_async(iterations=iterations, concurrency=concurrency))
        else:
            summary = runner.run_sync(iterations=iterations)
    except WorkplaceEnvError as exc:
        _fail(str(exc), code=exc.code, details=exc.details)
        return
    except Exception as exc:
        _fail("Benchmark failed", details={"exception": str(exc)})
        return

    payload: Dict[str, Any] = {"success": True, "benchmark": summary}
    if show_chart:
        payload["chart"] = benchmark_report(summary)

    if log_file:
        records = [
            {
                "strategy": strategy,
                **metrics,
            }
            for strategy, metrics in summary.get("per_strategy", {}).items()
        ]
        write_jsonl(log_file, records)
        payload["log_file"] = log_file

    _emit_json(payload)


def run_inference_entry() -> None:
    """Single-command entrypoint for run-inference."""
    typer.run(run_inference)


def run_grader_entry() -> None:
    """Single-command entrypoint for run-grader."""
    typer.run(run_grader)


def run_pipeline_entry() -> None:
    """Single-command entrypoint for run-pipeline."""
    typer.run(run_pipeline)


def main() -> None:
    app()


if __name__ == "__main__":
    main()