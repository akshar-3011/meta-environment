"""
models.py — Enhanced Workplace Env Pydantic models.

KEY FIX: Both WorkplaceObservation and WorkplaceAction must inherit from
openenv.core's Observation and Action base classes (not plain BaseModel).

Why this matters:
  - OpenEnv's serialize_observation() calls observation.model_dump(exclude={"reward","done","metadata"})
    and then reads observation.reward / observation.done separately.
    This only works if those fields are defined on the base Observation class.
  - deserialize_action() calls action_cls.model_validate(action_data) where
    action_data is the INNER dict (the value of the "action" key in the request body).
    This only works if your action class inherits from Action.

ENHANCEMENTS:
  - Added scenario_difficulty, urgency, sentiment to observation
  - Added confidence scoring capability
  - Added rich metadata for debugging and analysis
"""

from typing import List, Optional, Dict, Any
from openenv.core.env_server.types import Action, Observation


class WorkplaceObservation(Observation):
    """
    Observation returned after every reset() and step().

    Inherits from openenv.core Observation which already defines:
        reward: float | int | bool | None  (default None)
        done:   bool                        (default False)
        metadata: dict                      (default {})

    Do NOT redefine reward/done here — they live on the base class and
    serialize_observation() reads them directly from there.
    """
    email: str
    category_options: List[str]
    history: List[str]
    
    # Scenario metadata (helps agent learn difficulty progression)
    scenario_difficulty: Optional[str] = None  # easy/medium/hard
    urgency: Optional[str] = None              # low/medium/high
    sentiment: Optional[str] = None            # negative/neutral/positive/mixed
    complexity_score: Optional[int] = None     # 1-5
    
    # Rich metadata for analysis
    scenario_metadata: Optional[Dict[str, Any]] = None


class WorkplaceAction(Action):
    """
    Action sent by the agent on every step().

    Inherits from openenv.core Action which already defines:
        metadata: dict  (default {})

    The /step endpoint receives {"action": {"action_type": "...", "content": "..."}}
    and deserializes the INNER dict into this model via model_validate().
    
    ENHANCEMENTS:
      - confidence: Agent's confidence in this action (0.0-1.0)
      - explanation: Optional reasoning (for debugging/learning)
    """
    action_type: str
    content: str
    confidence: Optional[float] = None
    explanation: Optional[str] = None


class GradeResult:
    """
    Rich grading result with score, explanation, and components.
    Returned by grading functions to provide transparency.
    """
    def __init__(
        self,
        score: float,
        explanation: str = "",
        components: Optional[Dict[str, float]] = None,
    ):
        self.score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        self.explanation = explanation
        self.components = components or {}
    
    def __float__(self):
        return self.score
    
    def __repr__(self):
        return f"GradeResult(score={self.score:.2f}, explanation='{self.explanation}')"