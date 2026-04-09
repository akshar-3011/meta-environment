"""Experimental reward policies for A/B testing.

Each policy subclasses ``RuleBasedRewardPolicy`` and overrides only the
step-level weights, keeping the grading logic identical. This ensures
that the only variable under test is the reward *composition*, not the
underlying scoring rubric.

Production baseline (control): 40% classify, 35% reply, 25% escalate

Available variants:
  - EqualWeightPolicy:       33/33/33 — unbiased across skills
  - EscalationFirstPolicy:   25/25/50 — rewards correct escalation heavily
  - ReplyQualityPolicy:      30/50/20 — rewards reply craftsmanship
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..graders.interfaces import RewardPolicy
from ..graders.rule_based import RuleBasedRewardPolicy


# ─── Policy Registry ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyWeights:
    """Reward weights for the 3-step episode.  Must sum to ~1.0."""
    classify: float
    reply: float
    escalate: float

    def as_dict(self) -> Dict[str, float]:
        return {"classify": self.classify, "reply": self.reply, "escalate": self.escalate}


# Pre-defined weight sets
POLICY_WEIGHTS = {
    "control":          PolicyWeights(0.40, 0.35, 0.25),
    "equal":            PolicyWeights(0.333, 0.334, 0.333),
    "escalation_first": PolicyWeights(0.25, 0.25, 0.50),
    "reply_quality":    PolicyWeights(0.30, 0.50, 0.20),
}


class ConfigurableRewardPolicy(RuleBasedRewardPolicy):
    """RewardPolicy with configurable per-step weights.

    Overrides ``calculate_step_reward`` to apply custom weights while
    keeping all grading logic (accuracy, semantic similarity, rule-based)
    identical to the production policy.
    """

    def __init__(self, weights: PolicyWeights, **kwargs):
        super().__init__(**kwargs)
        self._weights = weights

    @property
    def weights(self) -> PolicyWeights:
        return self._weights

    @property
    def policy_name(self) -> str:
        for name, w in POLICY_WEIGHTS.items():
            if w == self._weights:
                return name
        return "custom"

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
        """Apply custom weights to the standard grading pipeline."""
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

        breakdown: Dict[str, Any] = {
            "step_count": step_count,
            "action_type": action_type,
            "category": actual_category,
            "policy": self.policy_name,
            "weights": self._weights.as_dict(),
        }

        if action_type == "classify":
            evaluation = self._classification_engine.evaluate(context)
            score = float(evaluation["score"])
            w = self._weights.classify
            breakdown["raw_score"] = score
            breakdown["weight"] = w
            breakdown["evaluation"] = evaluation
            breakdown["explanation"] = evaluation["breakdown"]["rule_based"]["explanation"]
            reward = score * w

        elif action_type == "reply":
            classification_reward = previous_actions.get("classify", 0.0)
            consistency_penalty = (
                max(0.0, 0.4 * (0.5 - classification_reward))
                if classification_reward < 0.5 else 0.0
            )

            evaluation = self._reply_engine.evaluate(context)
            score = float(evaluation["score"])
            w = self._weights.reply
            breakdown["raw_score"] = score
            breakdown["weight"] = w
            breakdown["consistency_penalty"] = consistency_penalty
            breakdown["evaluation"] = evaluation
            breakdown["explanation"] = evaluation["breakdown"]["rule_based"]["explanation"]
            reward = (score - consistency_penalty) * w

        elif action_type == "escalate":
            evaluation = self._escalation_engine.evaluate(context)
            score = float(evaluation["score"])
            w = self._weights.escalate
            breakdown["raw_score"] = score
            breakdown["weight"] = w
            breakdown["evaluation"] = evaluation
            breakdown["explanation"] = evaluation["breakdown"]["rule_based"]["explanation"]
            reward = score * w

            # Trajectory consistency (same as production)
            classify_r = previous_actions.get("classify", 0.0)
            reply_r = previous_actions.get("reply", 0.0)
            if classify_r >= 0.35 and reply_r >= 0.25:
                reward += 0.05
                breakdown["trajectory_bonus"] = 0.05
            escalated = content.lower().strip() in ["yes", "true", "1", "escalate", "urgent"]
            if escalated and actual_category in ["query", "refund"]:
                reward = max(0.0, reward - 0.03)
                breakdown["trajectory_penalty"] = -0.03

        else:
            reward = 0.0
            breakdown["error"] = f"Unknown action type: {action_type}"

        reward = max(0.0, min(1.0, reward))
        breakdown["final_reward"] = reward
        return reward, breakdown


# ─── Named Policy Factories ─────────────────────────────────────────────────

class EqualWeightPolicy(ConfigurableRewardPolicy):
    """33% classify / 33% reply / 33% escalate — unbiased across skills."""
    def __init__(self, **kwargs):
        super().__init__(POLICY_WEIGHTS["equal"], **kwargs)


class EscalationFirstPolicy(ConfigurableRewardPolicy):
    """25% classify / 25% reply / 50% escalate — rewards escalation heavily."""
    def __init__(self, **kwargs):
        super().__init__(POLICY_WEIGHTS["escalation_first"], **kwargs)


class ReplyQualityPolicy(ConfigurableRewardPolicy):
    """30% classify / 50% reply / 20% escalate — rewards reply craftsmanship."""
    def __init__(self, **kwargs):
        super().__init__(POLICY_WEIGHTS["reply_quality"], **kwargs)


# ─── Policy Resolution ──────────────────────────────────────────────────────

POLICY_CLASSES = {
    "control":          RuleBasedRewardPolicy,
    "equal":            EqualWeightPolicy,
    "escalation_first": EscalationFirstPolicy,
    "reply_quality":    ReplyQualityPolicy,
}


def get_policy(name: str) -> RewardPolicy:
    """Resolve a policy by name. Returns a fresh instance."""
    if name in POLICY_CLASSES:
        return POLICY_CLASSES[name]()
    if name == "custom":
        raise ValueError("Custom policies require explicit PolicyWeights")
    raise ValueError(f"Unknown policy: {name}. Available: {list(POLICY_CLASSES.keys())}")
