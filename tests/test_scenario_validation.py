"""Tests for Pydantic scenario validation at load time."""

from __future__ import annotations

import pytest

from workplace_env.core.models.workplace import Scenario
from workplace_env.data.scenario_repository import SCENARIOS


def test_all_scenarios_are_validated():
    """Every scenario in SCENARIOS has passed Pydantic validation at load time."""
    assert len(SCENARIOS) >= 30
    for idx, s in enumerate(SCENARIOS):
        # Re-validate each scenario to confirm it still passes
        scenario = Scenario(**s)
        assert scenario.email, f"Scenario {idx} has empty email"
        assert scenario.label in {"refund", "complaint", "query"}
        assert scenario.difficulty in {"easy", "medium", "hard"}


def test_invalid_scenario_missing_label_raises():
    """A scenario missing required 'label' field raises ValueError."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        Scenario(
            email="Test email",
            # label is missing
            difficulty="easy",
            sentiment="neutral",
            urgency="low",
            complexity=1,
            requires_escalation=False,
            min_reply_length=20,
        )


def test_invalid_scenario_bad_difficulty_raises():
    """A scenario with invalid difficulty value raises ValueError."""
    with pytest.raises(Exception):
        Scenario(
            email="Test email",
            label="refund",
            difficulty="impossible",  # Invalid
            sentiment="neutral",
            urgency="low",
            complexity=1,
            requires_escalation=False,
            min_reply_length=20,
        )


def test_invalid_scenario_complexity_out_of_range_raises():
    """A scenario with complexity > 5 raises ValueError."""
    with pytest.raises(Exception):
        Scenario(
            email="Test email",
            label="refund",
            difficulty="easy",
            sentiment="neutral",
            urgency="low",
            complexity=10,  # Out of range (1-5)
            requires_escalation=False,
            min_reply_length=20,
        )


def test_invalid_scenario_min_reply_length_too_low_raises():
    """A scenario with min_reply_length < 10 raises ValueError."""
    with pytest.raises(Exception):
        Scenario(
            email="Test email",
            label="refund",
            difficulty="easy",
            sentiment="neutral",
            urgency="low",
            complexity=1,
            requires_escalation=False,
            min_reply_length=5,  # Below minimum (10)
        )


def test_scenario_is_frozen():
    """Scenario instances are immutable after creation."""
    s = Scenario(
        email="Test",
        label="refund",
        difficulty="easy",
        sentiment="neutral",
        urgency="low",
        complexity=1,
        requires_escalation=False,
        min_reply_length=20,
    )
    with pytest.raises(Exception):  # Pydantic frozen validation error
        s.label = "complaint"


def test_scenario_count_by_difficulty():
    """Verify scenario distribution: >= 10 easy, >= 10 medium, >= 15 hard."""
    counts = {}
    for s in SCENARIOS:
        d = s["difficulty"]
        counts[d] = counts.get(d, 0) + 1

    assert counts.get("easy", 0) >= 10, f"Only {counts.get('easy', 0)} easy scenarios"
    assert counts.get("medium", 0) >= 10, f"Only {counts.get('medium', 0)} medium scenarios"
    assert counts.get("hard", 0) >= 15, f"Only {counts.get('hard', 0)} hard scenarios"
