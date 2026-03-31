"""Grader and reward-policy interfaces for extensible evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class EvaluationContext:
    """Context provided to graders for a single evaluation."""

    action_type: str
    content: str
    actual_category: str
    step_count: int
    scenario_difficulty: str = "easy"
    min_reply_length: int = 30
    previous_actions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderResult:
    """Normalized output from a single grader."""

    score: float
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "GraderResult":
        return GraderResult(
            score=clamp01(self.score),
            explanation=self.explanation,
            details=dict(self.details),
        )


class BaseGrader(ABC):
    """Plug-and-play contract for any grader implementation."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def grade(self, context: EvaluationContext) -> GraderResult:
        raise NotImplementedError


class RewardPolicy(ABC):
    """Contract for reward policies used by the environment."""

    @abstractmethod
    def grade_classification(
        self,
        predicted_category: str,
        actual_category: str,
        scenario_difficulty: str = "easy",
    ) -> Tuple[float, str]:
        raise NotImplementedError

    @abstractmethod
    def grade_reply(
        self,
        response: str,
        actual_category: str,
        min_length: int = 30,
    ) -> Tuple[float, str]:
        raise NotImplementedError

    @abstractmethod
    def grade_escalation(
        self,
        escalation_decision: str,
        actual_category: str,
        step_count: int,
    ) -> Tuple[float, str]:
        raise NotImplementedError

    @abstractmethod
    def calculate_step_reward(
        self,
        action_type: str,
        content: str,
        actual_category: str,
        step_count: int,
        scenario_difficulty: str = "easy",
        min_reply_length: int = 30,
        previous_actions: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict]:
        raise NotImplementedError
