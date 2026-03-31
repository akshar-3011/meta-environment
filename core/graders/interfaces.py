"""Grader interfaces for extensible reward policies."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


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
