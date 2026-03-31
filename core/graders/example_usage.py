"""Example usage for the modular grading framework."""

from __future__ import annotations

from .framework import (
    AccuracyGrader,
    RuleBasedGrader,
    SemanticSimilarityGrader,
    WeightedParallelGradingEngine,
)
from .interfaces import BaseGrader, EvaluationContext, GraderResult
from .rule_based import RuleBasedRewardPolicy


class BonusPolitenessGrader(BaseGrader):
    """Example plug-in grader that rewards polite language."""

    @property
    def name(self) -> str:
        return "bonus_politeness"

    def grade(self, context: EvaluationContext) -> GraderResult:
        text = (context.content or "").lower()
        polite_terms = ["please", "thank you", "sorry"]
        score = 1.0 if any(term in text for term in polite_terms) else 0.4
        return GraderResult(score=score, explanation="Politeness bonus")


def run_example() -> dict:
    policy = RuleBasedRewardPolicy()

    context = EvaluationContext(
        action_type="reply",
        content="We are sorry for the inconvenience. Please allow us to resolve this quickly.",
        actual_category="complaint",
        step_count=2,
        min_reply_length=30,
        metadata={
            "expected_keywords": ["sorry", "resolve", "understand"],
            "related_labels": ["refund"],
        },
    )

    # Build a custom plug-and-play engine
    engine = WeightedParallelGradingEngine(
        [
            (AccuracyGrader(), 0.15),
            (SemanticSimilarityGrader(), 0.15),
            (RuleBasedGrader(policy._rule_grade_reply), 0.60),  # internal rule callback
            (BonusPolitenessGrader(), 0.10),
        ]
    )

    return engine.evaluate(context)


if __name__ == "__main__":
    print(run_example())
