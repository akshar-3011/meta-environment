"""Unit tests for grading framework and reward policy."""

from __future__ import annotations

from workplace_env.core.graders.framework import WeightedParallelGradingEngine
from workplace_env.core.graders.interfaces import BaseGrader, EvaluationContext, GraderResult
from workplace_env.core.graders.rule_based import RuleBasedRewardPolicy


class _FixedGrader(BaseGrader):
    def __init__(self, *, name: str, score: float):
        self._name = name
        self._score = score

    @property
    def name(self) -> str:
        return self._name

    def grade(self, context: EvaluationContext) -> GraderResult:  # noqa: ARG002
        return GraderResult(score=self._score, explanation=f"fixed:{self._name}")


def test_weighted_parallel_engine_ignores_negative_weights():
    context = EvaluationContext(action_type="classify", content="refund", actual_category="refund", step_count=1)

    engine = WeightedParallelGradingEngine(
        [
            (_FixedGrader(name="positive", score=0.8), 1.0),
            (_FixedGrader(name="negative", score=0.0), -10.0),
        ]
    )

    result = engine.evaluate(context)

    assert result["score"] == 0.8
    assert result["breakdown"]["negative"]["weight"] == 0.0


def test_rule_policy_classification_rewards_exact_more_than_related():
    policy = RuleBasedRewardPolicy()

    exact_score, _ = policy.grade_classification("complaint", "complaint", scenario_difficulty="easy")
    related_score, _ = policy.grade_classification("refund", "complaint", scenario_difficulty="easy")

    assert exact_score > related_score
    assert exact_score > 0.9


def test_rule_policy_reply_applies_consistency_penalty_without_prior_classification():
    policy = RuleBasedRewardPolicy()
    response = "We sincerely apologize and resolve this quickly with a clear solution."

    with_penalty, breakdown_penalty = policy.calculate_step_reward(
        action_type="reply",
        content=response,
        actual_category="complaint",
        step_count=2,
        previous_actions={},
    )
    without_penalty, breakdown_no_penalty = policy.calculate_step_reward(
        action_type="reply",
        content=response,
        actual_category="complaint",
        step_count=2,
        previous_actions={"classify": 1.0},
    )

    assert breakdown_penalty["consistency_penalty"] == 0.2
    assert breakdown_no_penalty["consistency_penalty"] == 0.0
    assert with_penalty < without_penalty


def test_rule_policy_escalation_early_penalty_for_complaint():
    policy = RuleBasedRewardPolicy()

    early_score, _ = policy.grade_escalation("yes", "complaint", step_count=1)
    normal_score, _ = policy.grade_escalation("yes", "complaint", step_count=2)

    assert early_score < normal_score


def test_rule_policy_unknown_action_returns_zero_with_error_key():
    policy = RuleBasedRewardPolicy()

    reward, breakdown = policy.calculate_step_reward(
        action_type="unknown_action",
        content="whatever",
        actual_category="query",
        step_count=1,
    )

    assert reward == 0.0
    assert "error" in breakdown
