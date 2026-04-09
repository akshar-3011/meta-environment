"""Tests for grader crash handling — ensures a crashing grader degrades gracefully."""

from __future__ import annotations

from workplace_env.core.graders.framework import WeightedParallelGradingEngine
from workplace_env.core.graders.interfaces import BaseGrader, EvaluationContext, GraderResult


class _CrashingGrader(BaseGrader):
    """Grader that always raises an exception."""

    @property
    def name(self) -> str:
        return "crasher"

    def grade(self, context: EvaluationContext) -> GraderResult:
        raise RuntimeError("Simulated grader crash")


class _SafeGrader(BaseGrader):
    """Grader that always returns a fixed score."""

    @property
    def name(self) -> str:
        return "safe"

    def grade(self, context: EvaluationContext) -> GraderResult:
        return GraderResult(score=0.9, explanation="safe grader")


def test_crashing_grader_propagates_exception():
    """A crashing grader DOES propagate since we removed threading.

    With sequential evaluation, grader exceptions bubble up.
    The environment catch block in _grade_step handles this gracefully.
    """
    context = EvaluationContext(
        action_type="classify",
        content="refund",
        actual_category="refund",
        step_count=1,
    )

    engine = WeightedParallelGradingEngine([
        (_CrashingGrader(), 0.5),
        (_SafeGrader(), 0.5),
    ])

    try:
        engine.evaluate(context)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as exc:
        assert "Simulated grader crash" in str(exc)


def test_all_safe_graders_work():
    """When all graders are healthy, evaluation produces correct weighted score."""
    context = EvaluationContext(
        action_type="classify",
        content="refund",
        actual_category="refund",
        step_count=1,
    )

    engine = WeightedParallelGradingEngine([
        (_SafeGrader(), 0.6),
        (_SafeGrader(), 0.4),
    ])

    result = engine.evaluate(context)
    assert abs(result["score"] - 0.9) < 1e-9
    assert "safe" in result["breakdown"]

def test_environment_handles_grader_crash_gracefully():
    """Environment step catches grader crashes and raises PipelineError."""
    from workplace_env.environment.workplace_environment import WorkplaceEnvironment
    from workplace_env.models import WorkplaceAction
    from workplace_env.core.exceptions import PipelineError

    env = WorkplaceEnvironment()
    env.reset()

    # Monkey-patch the policy to simulate a crash
    original = env._policy.calculate_step_reward

    def _crashing_reward(*args, **kwargs):
        raise RuntimeError("Simulated reward crash")

    env._policy.calculate_step_reward = _crashing_reward

    try:
        env.step(WorkplaceAction(action_type="classify", content="refund"))
        assert False, "Should have raised PipelineError"
    except PipelineError:
        pass  # Expected — graceful degradation
    finally:
        env._policy.calculate_step_reward = original
