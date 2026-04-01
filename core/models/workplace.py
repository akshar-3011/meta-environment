"""Core Pydantic/domain models for the workplace environment."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server.types import Action, Observation


class Scenario(BaseModel):
    """A single customer-support scenario used for one RL episode.

    Validated at dataset load time so malformed entries surface immediately
    as a ValidationError rather than silently producing wrong rewards at
    runtime.  Frozen to prevent accidental mutation after load.
    """

    email: str
    label: Literal["refund", "complaint", "query"]
    difficulty: Literal["easy", "medium", "hard"]
    sentiment: Literal["negative", "neutral", "positive", "mixed"]
    urgency: Literal["low", "medium", "high"]
    complexity: int = Field(ge=1, le=5)
    requires_escalation: bool
    min_reply_length: int = Field(ge=10)

    model_config = ConfigDict(frozen=True)


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