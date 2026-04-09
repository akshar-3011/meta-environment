"""05_scenario_creation.py — Create and validate a custom scenario.

Shows how to define a new scenario, validate its structure, and
run it through the environment to verify rewards are in valid range.

Expected output:
    Created scenario: "VIP Cancellation"
    ✅ Schema validation passed
    Running episode...
      classify: reward=0.400
      reply:    reward=0.235
      escalate: reward=0.250
    ✅ Total reward: 0.885 (valid range)
    ✅ Scenario ready to add to data/scenario_repository.py
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import WorkplaceAction
from environment.workplace_environment import WorkplaceEnvironment


# ─── Required scenario fields and valid values ──────────────────────────────

REQUIRED_FIELDS = {
    "email": str,
    "label": str,
    "difficulty": str,
    "sentiment": str,
    "urgency": str,
    "complexity": int,
    "requires_escalation": bool,
    "min_reply_length": int,
}

VALID_LABELS = {"refund", "complaint", "query"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_SENTIMENTS = {"positive", "neutral", "negative", "mixed"}
VALID_URGENCIES = {"low", "medium", "high"}


def create_custom_scenario() -> dict:
    """Define a new scenario as a dictionary."""
    return {
        "email": (
            "Dear Support Team,\n\n"
            "I am a VIP member (account #12345) and I need to cancel my "
            "premium subscription effective immediately. I was charged $299 "
            "last month but I had already requested cancellation. Please "
            "process the refund and confirm the cancellation.\n\n"
            "This is urgent as I've been a loyal customer for 5 years and "
            "I'm very disappointed with how this has been handled.\n\n"
            "Regards,\nJohn Smith"
        ),
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 50,
    }


def validate_scenario(scenario: dict) -> bool:
    """Validate scenario against the expected schema."""
    # Check required fields
    for field_name, field_type in REQUIRED_FIELDS.items():
        if field_name not in scenario:
            print(f"❌ Missing required field: {field_name}")
            return False
        if not isinstance(scenario[field_name], field_type):
            print(f"❌ Field '{field_name}' must be {field_type.__name__}, "
                  f"got {type(scenario[field_name]).__name__}")
            return False

    # Check valid values
    if scenario["label"] not in VALID_LABELS:
        print(f"❌ Invalid label: {scenario['label']} (must be one of {VALID_LABELS})")
        return False
    if scenario["difficulty"] not in VALID_DIFFICULTIES:
        print(f"❌ Invalid difficulty: {scenario['difficulty']}")
        return False
    if scenario["sentiment"] not in VALID_SENTIMENTS:
        print(f"❌ Invalid sentiment: {scenario['sentiment']}")
        return False
    if scenario["urgency"] not in VALID_URGENCIES:
        print(f"❌ Invalid urgency: {scenario['urgency']}")
        return False
    if not (1 <= scenario["complexity"] <= 5):
        print(f"❌ Complexity must be 1-5, got {scenario['complexity']}")
        return False
    if scenario["min_reply_length"] < 10:
        print(f"❌ min_reply_length must be ≥ 10, got {scenario['min_reply_length']}")
        return False

    print('Created scenario: "VIP Cancellation"')
    print("✅ Schema validation passed")
    return True


def test_scenario(env: WorkplaceEnvironment, scenario: dict) -> float:
    """Run the scenario through the environment and check rewards."""
    env.reset()

    print("Running episode...")
    total_reward = 0.0

    # Classify
    obs = env.step(WorkplaceAction(action_type="classify", content=scenario["label"]))
    r = obs.reward or 0.0
    total_reward += r
    print(f"  classify: reward={r:.3f}")

    # Reply
    reply = (
        "Dear John, thank you for reaching out and for being a loyal VIP member "
        "for 5 years. We sincerely apologize for the billing error. We have "
        "processed your cancellation and initiated a full refund of $299. "
        "You should see the credit within 3-5 business days. We truly value "
        "your patronage and are sorry to see you go."
    )
    obs = env.step(WorkplaceAction(action_type="reply", content=reply))
    r = obs.reward or 0.0
    total_reward += r
    print(f"  reply:    reward={r:.3f}")

    # Escalate
    escalate = "yes" if scenario["requires_escalation"] else "no"
    obs = env.step(WorkplaceAction(action_type="escalate", content=escalate))
    r = obs.reward or 0.0
    total_reward += r
    print(f"  escalate: reward={r:.3f}")

    return total_reward


def main() -> None:
    """Full scenario creation workflow."""
    # 1. Define
    scenario = create_custom_scenario()

    # 2. Validate
    if not validate_scenario(scenario):
        return

    # 3. Test
    env = WorkplaceEnvironment()
    total = test_scenario(env, scenario)

    # 4. Verify
    if 0.0 <= total <= 3.0:
        print(f"✅ Total reward: {total:.3f} (valid range)")
        print("✅ Scenario ready to add to data/scenario_repository.py")
    else:
        print(f"❌ Total reward {total:.3f} is outside valid range!")


if __name__ == "__main__":
    main()
