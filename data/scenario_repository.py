"""Scenario repository abstractions for environment data access."""

from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..core.models import Scenario
except ImportError:  # pragma: no cover
    from core.models import Scenario


def _load_legacy_scenarios() -> List[Dict[str, Any]]:
    """Load and **validate** every scenario at import time.

    Each raw dict is parsed through the ``Scenario`` Pydantic model so that
    missing/invalid fields surface as ``ValidationError`` immediately rather
    than silently producing wrong rewards at runtime.
    """
    legacy_path = Path(__file__).resolve().parent.parent / "data.py"
    spec = spec_from_file_location("_workplace_env_legacy_data", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy scenario data")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    validated: List[Dict[str, Any]] = []
    for idx, raw in enumerate(module.SCENARIOS):
        try:
            scenario = Scenario(**raw)
            validated.append(scenario.model_dump())
        except Exception as exc:
            raise ValueError(
                f"Scenario {idx} failed validation: {exc}\n  Data: {raw}"
            ) from exc
    return validated


SCENARIOS = _load_legacy_scenarios()


class ScenarioRepository(ABC):
    """Abstract source of scenario definitions."""

    @abstractmethod
    def list_scenarios(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class StaticScenarioRepository(ScenarioRepository):
    """In-memory repository backed by static scenario constants."""

    def __init__(self, scenarios: Optional[List[Dict[str, Any]]] = None):
        self._scenarios = list(SCENARIOS if scenarios is None else scenarios)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return list(self._scenarios)

    def get_all(self) -> List[Dict[str, Any]]:
        """Compatibility alias for callers expecting get_all()."""
        return self.list_scenarios()


_DEFAULT_REPOSITORY = StaticScenarioRepository(SCENARIOS)


def get_default_repository() -> ScenarioRepository:
    return _DEFAULT_REPOSITORY


def get_refund_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "refund"])


def get_complaint_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "complaint"])


def get_query_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "query"])
