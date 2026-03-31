"""Regression tests for core models/data invariants and deterministic grading."""

from workplace_env.core.models import GradeResult, WorkplaceAction, WorkplaceObservation
from workplace_env.core.graders import (
    CATEGORY_OPTIONS,
    calculate_step_reward,
    grade_classification,
    grade_reply,
)
from workplace_env.data import get_default_repository

SCENARIOS = get_default_repository().list_scenarios()


def test_imports():
    assert WorkplaceObservation is not None
    assert WorkplaceAction is not None
    assert len(SCENARIOS) > 0


def test_scenario_metadata():
    required_keys = [
        "email",
        "label",
        "difficulty",
        "sentiment",
        "urgency",
        "complexity",
        "requires_escalation",
        "min_reply_length",
    ]

    for scenario in SCENARIOS:
        for key in required_keys:
            assert key in scenario


def test_grading_deterministic():
    for _ in range(3):
        score1, _ = grade_classification("refund", "refund", "easy")
        score2, _ = grade_classification("refund", "refund", "easy")
        assert score1 == score2

    for _ in range(3):
        score1, _ = grade_reply("This is a detailed response with proper keywords", "refund")
        score2, _ = grade_reply("This is a detailed response with proper keywords", "refund")
        assert score1 == score2


def test_reward_weighting():
    scenario = SCENARIOS[0]

    classify_reward, _ = calculate_step_reward(
        action_type="classify",
        content=scenario["label"],
        actual_category=scenario["label"],
        step_count=1,
        scenario_difficulty=scenario["difficulty"],
    )
    assert 0.39 < classify_reward < 0.41

    reply_reward, _ = calculate_step_reward(
        action_type="reply",
        content="A very long detailed response with excellent keywords and thoughtful approach",
        actual_category=scenario["label"],
        step_count=2,
        scenario_difficulty=scenario["difficulty"],
        min_reply_length=scenario["min_reply_length"],
        previous_actions={"classify": 1.0},
    )
    assert 0.0 <= reply_reward <= 0.35


def test_observation_creation():
    obs = WorkplaceObservation(
        email="Test email",
        category_options=CATEGORY_OPTIONS,
        history=["action1"],
        scenario_difficulty="medium",
        urgency="high",
        sentiment="negative",
        complexity_score=3,
        scenario_metadata={"label": "complaint"},
    )

    assert obs.email == "Test email"
    assert obs.scenario_difficulty == "medium"
    assert obs.urgency == "high"
    assert obs.sentiment == "negative"
    assert obs.complexity_score == 3


def test_action_creation():
    action = WorkplaceAction(
        action_type="classify",
        content="refund",
        confidence=0.95,
        explanation="Email contains refund keywords",
    )

    assert action.action_type == "classify"
    assert action.content == "refund"
    assert action.confidence == 0.95


def test_grade_result():
    result = GradeResult(
        score=0.85,
        explanation="Good response",
        components={"length": 0.5, "keywords": 0.35},
    )

    assert float(result) == 0.85
    assert result.components["length"] == 0.5


def test_reward_clamping():
    assert float(GradeResult(score=1.5)) == 1.0
    assert float(GradeResult(score=-0.5)) == 0.0


def test_scenario_cycling():
    indices = []
    for i in range(len(SCENARIOS) * 2):
        indices.append(i % len(SCENARIOS))

    first = indices[: len(SCENARIOS)]
    second = indices[len(SCENARIOS) :]
    assert first == second
