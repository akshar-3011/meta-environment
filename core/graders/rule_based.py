"""Rule-based reward policy backed by modular evaluation framework."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .interfaces import BaseGrader, EvaluationContext, RewardPolicy
from .framework import (
    AccuracyGrader,
    RuleBasedGrader,
    SemanticSimilarityGrader,
    WeightedParallelGradingEngine,
)

CATEGORY_OPTIONS = ["refund", "complaint", "query"]

RELATED_LABELS = {
    "refund": ["complaint"],
    "complaint": ["refund"],
    "query": [],
}

REQUIRED_KEYWORDS = {
    "refund": ["refund", "return", "process", "business days", "apologize", "3-5 business", "processed", "amount"],
    "complaint": ["sorry", "apologize", "understand", "resolve", "immediately", "priority", "team", "contact you"],
    "query": ["happy to help", "please", "contact", "let us know", "information", "details", "answer", "clarify", "provide"],
}

ESCALATION_REQUIRED = {
    "complaint": True,
    "refund": False,
    "query": False,
}

HARSH_PHRASES = ["not my problem", "figure it out", "stop emailing", "nothing we can do"]


class RuleBasedRewardPolicy(RewardPolicy):
    """Deterministic reward policy that composes modular graders in parallel."""

    def __init__(
        self,
        *,
        extra_graders: Optional[Dict[str, Sequence[Tuple[BaseGrader, float]]]] = None,
    ):
        extra = extra_graders or {}
        classify_graders: List[Tuple[BaseGrader, float]] = [
            (AccuracyGrader(), 0.25),
            (SemanticSimilarityGrader(), 0.15),
            (RuleBasedGrader(self._rule_grade_classification), 0.60),
        ]
        classify_graders.extend(list(extra.get("classify", [])))

        reply_graders: List[Tuple[BaseGrader, float]] = [
            (AccuracyGrader(), 0.15),
            (SemanticSimilarityGrader(), 0.15),
            (RuleBasedGrader(self._rule_grade_reply), 0.70),
        ]
        reply_graders.extend(list(extra.get("reply", [])))

        escalation_graders: List[Tuple[BaseGrader, float]] = [
            (AccuracyGrader(), 0.20),
            (SemanticSimilarityGrader(), 0.10),
            (RuleBasedGrader(self._rule_grade_escalation), 0.70),
        ]
        escalation_graders.extend(list(extra.get("escalate", [])))

        self._classification_engine = WeightedParallelGradingEngine(
            classify_graders
        )
        self._reply_engine = WeightedParallelGradingEngine(
            reply_graders
        )
        self._escalation_engine = WeightedParallelGradingEngine(
            escalation_graders
        )

    def _build_context(
        self,
        *,
        action_type: str,
        content: str,
        actual_category: str,
        step_count: int,
        scenario_difficulty: str = "easy",
        min_reply_length: int = 30,
        previous_actions: Optional[Dict[str, float]] = None,
        requires_escalation: Optional[bool] = None,
    ) -> EvaluationContext:
        # Prefer the caller-supplied requires_escalation (from the scenario's own
        # field) over the ESCALATION_REQUIRED policy dict.  Removes the duplicate
        # source-of-truth identified in the architecture analysis.
        escalation_flag = (
            requires_escalation
            if requires_escalation is not None
            else ESCALATION_REQUIRED.get(actual_category, False)
        )
        return EvaluationContext(
            action_type=action_type,
            content=content or "",
            actual_category=actual_category,
            step_count=step_count,
            scenario_difficulty=scenario_difficulty,
            min_reply_length=min_reply_length,
            previous_actions=previous_actions or {},
            metadata={
                "expected_keywords": REQUIRED_KEYWORDS.get(actual_category, []),
                "related_labels": RELATED_LABELS.get(actual_category, []),
                "requires_escalation": escalation_flag,
            },
        )

    def _rule_grade_classification(self, context: EvaluationContext) -> Tuple[float, str, Dict[str, Any]]:
        pred = (context.content or "").lower().strip()

        if pred == context.actual_category:
            return 1.0, f"Correct classification: {context.actual_category}", {"match": "exact"}

        related = RELATED_LABELS.get(context.actual_category, [])
        if pred in related:
            if context.scenario_difficulty == "easy":
                score = 0.3
            elif context.scenario_difficulty == "medium":
                score = 0.4
            else:
                score = 0.2
            return score, f"Partially correct: chose {pred}, should be {context.actual_category}", {"match": "related"}

        return 0.0, f"Wrong classification: {pred} (actual: {context.actual_category})", {"match": "none"}

    def _rule_grade_reply(self, context: EvaluationContext) -> Tuple[float, str, Dict[str, Any]]:
        text = (context.content or "").lower().strip()
        score = 0.0
        components = []
        details: Dict[str, Any] = {}

        if len(text) >= context.min_reply_length:
            score += 0.35
            components.append(f"length_ok({len(text)})")
            details["length_component"] = 0.35
        elif len(text) > 10:
            score += 0.1
            components.append(f"too_short({len(text)}/{context.min_reply_length})")
            details["length_component"] = 0.1

        if len(text) < 500:
            score += 0.15
            components.append("concise")
            details["conciseness_component"] = 0.15
        else:
            score -= 0.1
            components.append("too_verbose")
            details["conciseness_component"] = -0.1

        keywords = REQUIRED_KEYWORDS.get(context.actual_category, [])
        matched = sum(1 for kw in keywords if kw in text)
        if matched > 0:
            keyword_score = min(0.4, 0.15 * matched)
            score += keyword_score
            components.append(f"keywords({matched}/{len(keywords)})")
            details["keyword_component"] = keyword_score
        else:
            score -= 0.2
            components.append("no_keywords")
            details["keyword_component"] = -0.2

        if any(phrase in text for phrase in HARSH_PHRASES):
            score -= 0.15
            components.append("harsh_tone")
            details["tone_penalty"] = -0.15

        solution_words = ["help", "assist", "resolve", "fix", "solution", "refund", "process"]
        if any(word in text for word in solution_words):
            score += 0.1
            components.append("solution_oriented")
            details["solution_component"] = 0.1

        has_greeting = any(g in text for g in ["dear", "hello", "thank you"])
        has_signoff = any(s in text for s in ["regards", "sincerely", "team"])
        if has_greeting and has_signoff:
            score += 0.05
            components.append("professionalism_bonus")
            details["professionalism_bonus"] = 0.05

        score = max(0.0, min(1.0, score))
        return score, f"Reply scoring: {', '.join(components)}", details

    def _rule_grade_escalation(self, context: EvaluationContext) -> Tuple[float, str, Dict[str, Any]]:
        decision = (context.content or "").lower().strip()
        did_escalate = decision in ["yes", "true", "urgent", "1", "escalate"]
        # Prefer the scenario's own requires_escalation field (passed via context
        # metadata) over the hardcoded ESCALATION_REQUIRED fallback dict.
        # This eliminates the duplicate source-of-truth (Bug 3).
        should_escalate = context.metadata.get(
            "requires_escalation",
            ESCALATION_REQUIRED.get(context.actual_category, False),
        )

        score = 0.0
        reason = ""
        details: Dict[str, Any] = {
            "did_escalate": did_escalate,
            "should_escalate": should_escalate,
        }

        if should_escalate and did_escalate:
            score = 1.0
            reason = f"Correctly escalated {context.actual_category}"
        elif not should_escalate and not did_escalate:
            score = 0.9
            reason = f"Correctly handled {context.actual_category} without escalation"
        elif should_escalate and not did_escalate:
            score = 0.1
            reason = f"Should escalate {context.actual_category}, but did not"
        elif not should_escalate and did_escalate:
            score = 0.3
            reason = f"Over-escalated {context.actual_category}"

        if did_escalate and context.step_count < 2:
            score *= 0.7
            reason += " (early escalation penalty)"
            details["timing_penalty"] = 0.3

        if did_escalate and context.step_count == 2:
            score = min(1.0, score + 0.1)
            reason += " (good timing)"
            details["timing_bonus"] = 0.1

        return score, reason, details

    def grade_classification(
        self,
        predicted_category: str,
        actual_category: str,
        scenario_difficulty: str = "easy",
    ) -> Tuple[float, str]:
        context = self._build_context(
            action_type="classify",
            content=predicted_category,
            actual_category=actual_category,
            step_count=1,
            scenario_difficulty=scenario_difficulty,
        )
        evaluation = self._classification_engine.evaluate(context)
        explanation = evaluation["breakdown"]["rule_based"]["explanation"]
        return float(evaluation["score"]), explanation

    def grade_reply(
        self,
        response: str,
        actual_category: str,
        min_length: int = 30,
    ) -> Tuple[float, str]:
        context = self._build_context(
            action_type="reply",
            content=response,
            actual_category=actual_category,
            step_count=2,
            min_reply_length=min_length,
        )
        evaluation = self._reply_engine.evaluate(context)
        explanation = evaluation["breakdown"]["rule_based"]["explanation"]
        return float(evaluation["score"]), explanation

    def grade_escalation(
        self,
        escalation_decision: str,
        actual_category: str,
        step_count: int,
    ) -> Tuple[float, str]:
        context = self._build_context(
            action_type="escalate",
            content=escalation_decision,
            actual_category=actual_category,
            step_count=step_count,
        )
        evaluation = self._escalation_engine.evaluate(context)
        explanation = evaluation["breakdown"]["rule_based"]["explanation"]
        return float(evaluation["score"]), explanation

    def calculate_step_reward(
        self,
        action_type: str,
        content: str,
        actual_category: str,
        step_count: int,
        scenario_difficulty: str = "easy",
        min_reply_length: int = 30,
        previous_actions: Optional[Dict[str, float]] = None,
        requires_escalation: Optional[bool] = None,
    ) -> Tuple[float, Dict]:
        content = content or ""
        previous_actions = previous_actions or {}

        context = self._build_context(
            action_type=action_type,
            content=content,
            actual_category=actual_category,
            step_count=step_count,
            scenario_difficulty=scenario_difficulty,
            min_reply_length=min_reply_length,
            previous_actions=previous_actions,
            requires_escalation=requires_escalation,
        )

        breakdown = {
            "step_count": step_count,
            "action_type": action_type,
            "category": actual_category,
        }

        if action_type == "classify":
            evaluation = self._classification_engine.evaluate(context)
            score = float(evaluation["score"])
            explanation = evaluation["breakdown"]["rule_based"]["explanation"]
            breakdown["raw_score"] = score
            breakdown["explanation"] = explanation
            breakdown["evaluation"] = evaluation
            breakdown["weight"] = 0.4
            reward = score * 0.4

        elif action_type == "reply":
            classification_reward = previous_actions.get("classify", 0.0)
            consistency_penalty = 0.0 if classification_reward > 0.5 else 0.2

            evaluation = self._reply_engine.evaluate(context)
            score = float(evaluation["score"])
            explanation = evaluation["breakdown"]["rule_based"]["explanation"]
            breakdown["raw_score"] = score
            breakdown["explanation"] = explanation
            breakdown["evaluation"] = evaluation
            breakdown["consistency_penalty"] = consistency_penalty
            breakdown["weight"] = 0.35
            reward = (score - consistency_penalty) * 0.35

        elif action_type == "escalate":
            evaluation = self._escalation_engine.evaluate(context)
            score = float(evaluation["score"])
            explanation = evaluation["breakdown"]["rule_based"]["explanation"]
            breakdown["raw_score"] = score
            breakdown["explanation"] = explanation
            breakdown["evaluation"] = evaluation
            breakdown["weight"] = 0.25
            reward = score * 0.25

        else:
            reward = 0.0
            breakdown["error"] = f"Unknown action type: {action_type}"

        reward = max(0.0, min(1.0, reward))
        breakdown["final_reward"] = reward

        return reward, breakdown


_DEFAULT_POLICY = RuleBasedRewardPolicy()


def grade_classification(predicted_category: str, actual_category: str, scenario_difficulty: str = "easy") -> Tuple[float, str]:
    return _DEFAULT_POLICY.grade_classification(predicted_category, actual_category, scenario_difficulty)


def grade_reply(response: str, actual_category: str, min_length: int = 30) -> Tuple[float, str]:
    return _DEFAULT_POLICY.grade_reply(response, actual_category, min_length)


def grade_escalation(escalation_decision: str, actual_category: str, step_count: int) -> Tuple[float, str]:
    return _DEFAULT_POLICY.grade_escalation(escalation_decision, actual_category, step_count)


def calculate_step_reward(
    action_type: str,
    content: str,
    actual_category: str,
    step_count: int,
    scenario_difficulty: str = "easy",
    min_reply_length: int = 30,
    previous_actions: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    return _DEFAULT_POLICY.calculate_step_reward(
        action_type,
        content,
        actual_category,
        step_count,
        scenario_difficulty,
        min_reply_length,
        previous_actions,
    )