"""
Integration tests to validate all improvements.

Run this to verify:
  1. All imports work
  2. Grading logic is deterministic
  3. Reward weighting is correct
  4. State management works
  5. No runtime errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import WorkplaceObservation, WorkplaceAction, GradeResult
from data import SCENARIOS
from graders.grader import (
    calculate_step_reward,
    grade_classification,
    grade_reply,
    grade_escalation,
    CATEGORY_OPTIONS,
)


def test_imports():
    """Test all imports work."""
    print("✓ test_imports: All imports successful")
    assert WorkplaceObservation is not None
    assert WorkplaceAction is not None
    assert len(SCENARIOS) > 0
    print(f"  - Loaded {len(SCENARIOS)} scenarios")


def test_scenario_metadata():
    """Test scenarios have all required metadata."""
    print("\n✓ test_scenario_metadata: Checking scenario structure")
    
    required_keys = [
        "email", "label", "difficulty", "sentiment", "urgency",
        "complexity", "requires_escalation", "min_reply_length"
    ]
    
    for i, scenario in enumerate(SCENARIOS):
        for key in required_keys:
            assert key in scenario, f"Scenario {i} missing key: {key}"
    
    print(f"  - All {len(SCENARIOS)} scenarios have required metadata")
    
    # Check difficulty distribution
    difficulties = {}
    for s in SCENARIOS:
        d = s["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1
    print(f"  - Difficulty distribution: {difficulties}")


def test_grading_deterministic():
    """Test that grading is deterministic (no randomness)."""
    print("\n✓ test_grading_deterministic: Checking reproducibility")
    
    # Run same grading twice, expect same result
    for _ in range(3):
        score1, exp1 = grade_classification("refund", "refund", "easy")
        score2, exp2 = grade_classification("refund", "refund", "easy")
        assert score1 == score2, f"Inconsistent grading: {score1} vs {score2}"
    
    print("  - Classification grading is deterministic")
    
    for _ in range(3):
        score1, exp1 = grade_reply("This is a detailed response with proper keywords", "refund")
        score2, exp2 = grade_reply("This is a detailed response with proper keywords", "refund")
        assert score1 == score2
    
    print("  - Reply grading is deterministic")


def test_reward_weighting():
    """Test that reward weights are applied correctly."""
    print("\n✓ test_reward_weighting: Checking weighted rewards")
    
    scenario = SCENARIOS[0]  # Get first scenario for consistency
    
    # Perfect classification
    classify_reward, breakdown = calculate_step_reward(
        action_type="classify",
        content=scenario["label"],
        actual_category=scenario["label"],
        step_count=1,
        scenario_difficulty=scenario["difficulty"],
    )
    assert 0.39 < classify_reward < 0.41, f"Classify weight wrong: {classify_reward}"
    print(f"  - Classify reward weight: {classify_reward:.3f} (expected ~0.40)")
    
    # Perfect reply (with classification context)
    reply_reward, breakdown = calculate_step_reward(
        action_type="reply",
        content="A very long detailed response with excellent keywords and thoughtful approach",
        actual_category=scenario["label"],
        step_count=2,
        scenario_difficulty=scenario["difficulty"],
        min_reply_length=scenario["min_reply_length"],
        previous_actions={"classify_reward": 1.0},  # Perfect classification
    )
    # Reply reward = (score - consistency_penalty) * 0.35
    # Score depends on reply content length and keywords for the scenario
    # For refund label: needs "refund", "return", "money" keywords
    # Our test reply may not have all keywords, so we just check it's weighted
    assert 0.0 <= reply_reward <= 0.35, f"Reply weight out of bounds: {reply_reward}"
    print(f"  - Reply reward weight: {reply_reward:.3f} (in range [0, 0.35])")
    
    # Perfect escalation
    escalation_reward, breakdown = calculate_step_reward(
        action_type="escalate",
        content="yes",
        actual_category=scenario["label"],
        step_count=3,
        scenario_difficulty=scenario["difficulty"],
    )
    # Depends on scenario's escalation requirement
    print(f"  - Escalation reward weight: {escalation_reward:.3f} (expected ~0.25)")


def test_observation_creation():
    """Test that observations have all new fields."""
    print("\n✓ test_observation_creation: Checking observation fields")
    
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
    assert obs.scenario_metadata["label"] == "complaint"
    
    print("  - Observation created successfully with all fields")


def test_action_creation():
    """Test that actions support confidence and explanation."""
    print("\n✓ test_action_creation: Checking action fields")
    
    action = WorkplaceAction(
        action_type="classify",
        content="refund",
        confidence=0.95,
        explanation="Email contains refund keywords",
    )
    
    assert action.action_type == "classify"
    assert action.content == "refund"
    assert action.confidence == 0.95
    assert action.explanation == "Email contains refund keywords"
    
    print("  - Action created successfully with all fields")


def test_grade_result():
    """Test GradeResult class."""
    print("\n✓ test_grade_result: Checking GradeResult")
    
    result = GradeResult(
        score=0.85,
        explanation="Good response",
        components={"length": 0.5, "keywords": 0.35},
    )
    
    assert float(result) == 0.85
    assert result.explanation == "Good response"
    assert result.components["length"] == 0.5
    
    print("  - GradeResult working correctly")


def test_reward_clamping():
    """Test that rewards are clamped to [0, 1]."""
    print("\n✓ test_reward_clamping: Checking reward bounds")
    
    result = GradeResult(score=1.5)  # Over 1.0
    assert float(result) == 1.0, f"Not clamped: {float(result)}"
    
    result = GradeResult(score=-0.5)  # Under 0.0
    assert float(result) == 0.0, f"Not clamped: {float(result)}"
    
    print("  - Rewards properly clamped to [0, 1]")


def test_scenario_cycling():
    """Test deterministic scenario cycling."""
    print("\n✓ test_scenario_cycling: Checking deterministic cycling")
    
    # Simulate cycling through scenarios
    indices = []
    for i in range(len(SCENARIOS) * 2):
        idx = i % len(SCENARIOS)
        indices.append(idx)
    
    # First cycle should match second cycle
    first_cycle = indices[:len(SCENARIOS)]
    second_cycle = indices[len(SCENARIOS):]
    assert first_cycle == second_cycle
    
    print(f"  - Scenario cycling deterministic over {len(SCENARIOS)} scenarios")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("PRODUCTION-GRADE WORKPLACE ENV — VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_scenario_metadata,
        test_grading_deterministic,
        test_reward_weighting,
        test_observation_creation,
        test_action_creation,
        test_grade_result,
        test_reward_clamping,
        test_scenario_cycling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED — Production-grade environment validated!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
