"""Modular grading framework: concrete graders + weighted sequential aggregation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

from .interfaces import BaseGrader, EvaluationContext, GraderResult, clamp01


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch if ch.isalnum() else " " for ch in (text or "").lower()).split() if t]


def _jaccard(a: str, b: str) -> float:
    a_set = set(_tokenize(a))
    b_set = set(_tokenize(b))
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


class AccuracyGrader(BaseGrader):
    """Exact-match or deterministic correctness checks."""

    @property
    def name(self) -> str:
        return "accuracy"

    def grade(self, context: EvaluationContext) -> GraderResult:
        action = context.action_type
        content = (context.content or "").strip().lower()

        if action == "classify":
            score = 1.0 if content == context.actual_category else 0.0
            return GraderResult(score=score, explanation="Exact classification match")

        if action == "reply":
            min_len = max(1, context.min_reply_length)
            score = min(1.0, len(content) / min_len)
            return GraderResult(
                score=score,
                explanation="Reply length adequacy",
                details={"length": len(content), "min_length": min_len},
            )

        if action == "escalate":
            should_escalate = context.actual_category == "complaint"
            did_escalate = content in {"yes", "true", "urgent", "1", "escalate"}
            score = 1.0 if should_escalate == did_escalate else 0.0
            return GraderResult(score=score, explanation="Escalation decision correctness")

        return GraderResult(score=0.0, explanation=f"Unsupported action type: {action}")


class SemanticSimilarityGrader(BaseGrader):
    """Lightweight semantic overlap grader (token/Jaccard heuristic)."""

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def grade(self, context: EvaluationContext) -> GraderResult:
        action = context.action_type
        content = context.content or ""

        if action == "classify":
            related = context.metadata.get("related_labels", [])
            pred = content.strip().lower()
            if pred == context.actual_category:
                score = 1.0
            elif pred in related:
                score = 0.7
            else:
                score = 0.0
            return GraderResult(score=score, explanation="Label semantic proximity")

        if action == "reply":
            expected_terms = " ".join(context.metadata.get("expected_keywords", []))
            score = _jaccard(content, expected_terms)
            return GraderResult(
                score=score,
                explanation="Reply keyword semantic overlap",
                details={"expected_terms": context.metadata.get("expected_keywords", [])},
            )

        if action == "escalate":
            expected = "yes" if context.actual_category == "complaint" else "no"
            score = _jaccard(content, expected)
            return GraderResult(score=score, explanation="Escalation semantic agreement")

        return GraderResult(score=0.0, explanation=f"Unsupported action type: {action}")


class RuleBasedGrader(BaseGrader):
    """Adapter that wraps existing deterministic rule evaluators."""

    def __init__(self, evaluator: Callable[[EvaluationContext], Tuple[float, str, Dict[str, Any]]]):
        self._evaluator = evaluator

    @property
    def name(self) -> str:
        return "rule_based"

    def grade(self, context: EvaluationContext) -> GraderResult:
        score, explanation, details = self._evaluator(context)
        return GraderResult(score=score, explanation=explanation, details=details)


class WeightedParallelGradingEngine:
    """Runs multiple graders sequentially and combines with weighted aggregation.

    All graders are CPU-bound Python (GIL prevents true parallelism), so
    sequential execution avoids thread creation/synchronization overhead.

    Output contract:
      {
        "score": float,
        "breakdown": {...}
      }
    """

    def __init__(self, graders: Sequence[Tuple[BaseGrader, float]]):
        if not graders:
            raise ValueError("At least one grader must be provided")
        self._graders = list(graders)

    def evaluate(self, context: EvaluationContext) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for grader, weight in self._graders:
            result = grader.grade(context).normalized()
            w = max(0.0, float(weight))
            weighted_score = result.score * w
            weighted_sum += weighted_score
            total_weight += w

            breakdown[grader.name] = {
                "weight": w,
                "score": result.score,
                "weighted_score": weighted_score,
                "explanation": result.explanation,
                "details": result.details,
            }

        final_score = 0.0 if total_weight == 0 else clamp01(weighted_sum / total_weight)
        return {
            "score": final_score,
            "breakdown": breakdown,
        }

