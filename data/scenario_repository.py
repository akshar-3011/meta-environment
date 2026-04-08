"""Scenario repository abstractions for environment data access."""

import copy
from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List


def _load_legacy_scenarios() -> List[Dict[str, Any]]:
    """Load scenarios from scenario_data.py (or legacy data.py fallback).

    C5 Fix: Wrapped exec_module in try/except with clear error messages.
    C6 Fix: Prefers scenario_data.py to avoid data.py/data/ name collision.
    N2 Fix: Validates each scenario through the Scenario Pydantic model.
    """
    # C6: prefer scenario_data.py; fall back to data.py for backwards compat
    parent = Path(__file__).resolve().parent.parent
    legacy_path = parent / "scenario_data.py"
    if not legacy_path.exists():
        legacy_path = parent / "data.py"

    if not legacy_path.exists():
        raise RuntimeError(
            f"Scenario data file not found. Looked for:\n"
            f"  {parent / 'scenario_data.py'}\n"
            f"  {parent / 'data.py'}"
        )

    try:
        spec = spec_from_file_location("_workplace_env_legacy_data", legacy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create module spec for {legacy_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load scenario data from {legacy_path}: {exc}"
        ) from exc

    raw_scenarios = getattr(module, "SCENARIOS", None)
    if not raw_scenarios:
        raise RuntimeError(
            f"Scenario data loaded from {legacy_path} but SCENARIOS list is empty or missing"
        )

    # N2: Validate each scenario through the Pydantic model at load time
    # so malformed entries are caught immediately on startup, not at grading time.
    try:
        from core.models.workplace import Scenario
    except ImportError:
        try:
            from ..core.models.workplace import Scenario
        except ImportError:
            # If Scenario model is not available, skip validation
            Scenario = None  # type: ignore

    validated: List[Dict[str, Any]] = []
    for i, s in enumerate(raw_scenarios):
        if Scenario is not None:
            try:
                validated.append(Scenario(**s).model_dump())
            except Exception as exc:
                raise ValueError(
                    f"Scenario at index {i} is invalid: {exc}\n"
                    f"  Data: { {k: v for k, v in s.items() if k != 'email'} }"
                ) from exc
        else:
            validated.append(dict(s))

    return validated


SCENARIOS = _load_legacy_scenarios()


class ScenarioRepository(ABC):
    """Abstract source of scenario definitions."""

    @abstractmethod
    def list_scenarios(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class StaticScenarioRepository(ScenarioRepository):
    """In-memory repository backed by static scenario constants.

    Returns defensive copies to prevent callers from corrupting shared data.
    """

    def __init__(self, scenarios: List[Dict[str, Any]]):
        self._scenarios = scenarios

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._scenarios)


_DEFAULT_REPOSITORY = StaticScenarioRepository(SCENARIOS)


def get_default_repository() -> ScenarioRepository:
    return _DEFAULT_REPOSITORY


def get_refund_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "refund"])


def get_complaint_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "complaint"])


def get_query_repository() -> ScenarioRepository:
    return StaticScenarioRepository([s for s in SCENARIOS if s["label"] == "query"])
