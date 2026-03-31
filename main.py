"""Command-line interface for inference, grading, and end-to-end pipeline runs."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List

import typer

from .core.exceptions import WorkplaceEnvError
from .core.graders import CATEGORY_OPTIONS, RuleBasedRewardPolicy
from .core.inference import AsyncInference, EnhancedInference, StandardInference


class InferenceStrategy(str, Enum):
    standard = "standard"
    enhanced = "enhanced"
    async_strategy = "async"


app = typer.Typer(help="Workplace Env CLI")


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


def _run_inference_impl(
    *,
    email: str,
    strategy: InferenceStrategy,
    scenario_difficulty: str,
    urgency: str,
    sentiment: str,
    complexity_score: int,
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    engine = _select_strategy(strategy)
    policy = RuleBasedRewardPolicy()

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
        },
    }


@app.command("run-inference")
def run_inference(
    email: str = typer.Option(..., help="Customer email text"),
    strategy: InferenceStrategy = typer.Option(InferenceStrategy.standard, help="Inference strategy"),
    scenario_difficulty: str = typer.Option("easy", help="Scenario difficulty: easy/medium/hard"),
    urgency: str = typer.Option("medium", help="Urgency metadata"),
    sentiment: str = typer.Option("neutral", help="Sentiment metadata"),
    complexity_score: int = typer.Option(2, min=1, max=5, help="Complexity score (1-5)"),
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
            )
        )
    except WorkplaceEnvError as exc:
        _fail(str(exc), code=exc.code, details=exc.details)
    except Exception as exc:
        _fail("Pipeline command failed", details={"exception": str(exc)})


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