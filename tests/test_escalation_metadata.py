"""Tests for hard scenarios with requires_escalation=True rewarding correctly."""

from __future__ import annotations

import pytest

from workplace_env.core.graders.rule_based import RuleBasedRewardPolicy
from workplace_env.data.scenario_repository import SCENARIOS


# Collect all hard scenarios that require escalation
_HARD_ESCALATION_SCENARIOS = [
    (idx, s) for idx, s in enumerate(SCENARIOS)
    if s["difficulty"] == "hard" and s.get("requires_escalation", False)
]


@pytest.mark.parametrize(
    "scenario_idx,scenario",
    _HARD_ESCALATION_SCENARIOS,
    ids=[f"H{idx}" for idx, _ in _HARD_ESCALATION_SCENARIOS],
)
def test_requires_escalation_scenarios_reward_correctly(scenario_idx, scenario):
    """Hard scenarios with requires_escalation=True get high reward for correct escalation."""
    policy = RuleBasedRewardPolicy()
    previous_actions = {}

    # Step 1: Classify correctly
    r_classify, bd = policy.calculate_step_reward(
        action_type="classify",
        content=scenario["label"],
        actual_category=scenario["label"],
        step_count=1,
        scenario_difficulty="hard",
        min_reply_length=scenario.get("min_reply_length", 30),
        previous_actions=previous_actions,
    )
    previous_actions["classify"] = r_classify
    assert r_classify > 0.3, f"Classification reward too low: {r_classify}"

    # Step 2: Reply with professional response
    reply = (
        "Thank you for reaching out. We sincerely apologize for this experience. "
        "We understand your frustration and we take this very seriously. "
        "Our dedicated team will process your request and contact you within 24 hours."
    )
    r_reply, _ = policy.calculate_step_reward(
        action_type="reply",
        content=reply,
        actual_category=scenario["label"],
        step_count=2,
        scenario_difficulty="hard",
        min_reply_length=scenario.get("min_reply_length", 30),
        previous_actions=previous_actions,
    )
    previous_actions["reply"] = r_reply
    assert r_reply > 0.1, f"Reply reward too low: {r_reply}"

    # Step 3: Escalate correctly (yes for requires_escalation=True)
    r_escalate, bd = policy.calculate_step_reward(
        action_type="escalate",
        content="yes",
        actual_category=scenario["label"],
        step_count=3,
        scenario_difficulty="hard",
        min_reply_length=scenario.get("min_reply_length", 30),
        previous_actions=previous_actions,
        requires_escalation=True,
    )
    assert r_escalate > 0.15, f"Escalation reward too low: {r_escalate}"

    total = r_classify + r_reply + r_escalate
    assert total > 0.7, f"Total reward {total:.3f} too low for correct episode on H{scenario_idx}"


@pytest.mark.parametrize(
    "scenario_idx,scenario",
    _HARD_ESCALATION_SCENARIOS,
    ids=[f"H{idx}-missed" for idx, _ in _HARD_ESCALATION_SCENARIOS],
)
def test_missed_escalation_penalized(scenario_idx, scenario):
    """Missing required escalation produces a significantly lower reward."""
    policy = RuleBasedRewardPolicy()

    r_correct, _ = policy.calculate_step_reward(
        action_type="escalate",
        content="yes",
        actual_category=scenario["label"],
        step_count=3,
        scenario_difficulty="hard",
        previous_actions={"classify": 0.4, "reply": 0.2},
        requires_escalation=True,
    )

    r_missed, _ = policy.calculate_step_reward(
        action_type="escalate",
        content="no",
        actual_category=scenario["label"],
        step_count=3,
        scenario_difficulty="hard",
        previous_actions={"classify": 0.4, "reply": 0.2},
        requires_escalation=True,
    )

    assert r_correct > r_missed, (
        f"Correct escalation ({r_correct:.3f}) should beat missed ({r_missed:.3f})"
    )
