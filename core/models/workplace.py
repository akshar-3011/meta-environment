"""Core Pydantic/domain models for the workplace environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation


class WorkplaceObservation(Observation):
    """Observation returned after each reset/step."""

    email: str
    category_options: List[str]
    history: List[str]
    scenario_difficulty: Optional[str] = None
    urgency: Optional[str] = None
    sentiment: Optional[str] = None
    complexity_score: Optional[int] = None
    scenario_metadata: Optional[Dict[str, Any]] = None


class WorkplaceAction(Action):
    """Action submitted by an agent for a workflow step."""

    action_type: str
    content: str
    confidence: Optional[float] = None
    explanation: Optional[str] = None


class GradeResult:
    """Structured grading result with normalized score and details."""

    def __init__(
        self,
        score: float,
        explanation: str = "",
        components: Optional[Dict[str, float]] = None,
    ):
        self.score = max(0.0, min(1.0, score))
        self.explanation = explanation
        self.components = components or {}

    def __float__(self):
        return self.score

    def __repr__(self):
        return f"GradeResult(score={self.score:.2f}, explanation='{self.explanation}')"
