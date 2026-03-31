"""Scenario repository abstractions for environment data access."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

try:
    from ..data import SCENARIOS  # Backward-compatible source of truth
except ImportError:  # pragma: no cover
    from data import SCENARIOS


class ScenarioRepository(ABC):
    """Abstract source of scenario definitions."""

    @abstractmethod
    def list_scenarios(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class StaticScenarioRepository(ScenarioRepository):
    """In-memory repository backed by static scenario constants."""

    def __init__(self, scenarios: List[Dict[str, Any]]):
        self._scenarios = scenarios

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return list(self._scenarios)


_DEFAULT_REPOSITORY = StaticScenarioRepository(SCENARIOS)


def get_default_repository() -> ScenarioRepository:
    return _DEFAULT_REPOSITORY
