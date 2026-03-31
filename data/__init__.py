"""Data access abstractions and repositories.

This package intentionally re-exports `SCENARIOS` by loading the legacy
top-level `data.py` module to maintain backwards compatibility while enabling
new modular imports.
"""

from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List


def _load_legacy_scenarios() -> List[Dict[str, Any]]:
    legacy_path = Path(__file__).resolve().parent.parent / "data.py"
    spec = spec_from_file_location("_workplace_env_legacy_data", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy scenario data")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.SCENARIOS)


SCENARIOS = _load_legacy_scenarios()


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


__all__ = ["SCENARIOS", "ScenarioRepository", "StaticScenarioRepository", "get_default_repository"]
