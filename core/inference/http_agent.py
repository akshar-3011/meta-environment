"""Shared HTTP inference runner to avoid script duplication."""

import json
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = "http://localhost:8000"


def _post(path: str, payload: Dict[str, Any], base_url: str) -> Optional[Dict[str, Any]]:
    url = f"{base_url}{path}"
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if not resp.ok:
            print(f"\n❌ HTTP {resp.status_code} on {path}")
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
            resp.raise_for_status()
        return resp.json()
    except Exception as e:  # pragma: no cover
        print(f"\n❌ Request failed: {e}")
        return None


def _step(action_type: str, content: str, base_url: str) -> Optional[Dict[str, Any]]:
    return _post("/step", {"action": {"action_type": action_type, "content": content}}, base_url)


def _print_metadata(obs: Dict[str, Any], reveal_label: bool):
    metadata = obs.get("scenario_metadata", {})

    print(f"\n📧 Email: {obs.get('email')}")
    print("\n📊 Scenario Metadata:")
    print(f"   Difficulty  : {obs.get('scenario_difficulty', 'N/A')}")
    print(f"   Urgency     : {obs.get('urgency', 'N/A')}")
    print(f"   Sentiment   : {obs.get('sentiment', 'N/A')}")
    print(f"   Complexity  : {obs.get('complexity_score', 'N/A')}/5")
    print(f"   Categories  : {obs.get('category_options')}")

    if reveal_label:
        print(f"\n✅ True Label         : {metadata.get('label', 'N/A')}")
        print(f"   Requires Escalation: {metadata.get('requires_escalation', False)}")
    else:
        print("   (True label withheld — agent must classify from email content)")
    print(f"   Min Reply Length   : {metadata.get('min_reply_length', 'N/A')}")


def run_agent(
    actions: Optional[List[Tuple[str, str]]] = None,
    reveal_label: bool = False,
    title: str = "WORKPLACE ENVIRONMENT AGENT",
    base_url: str = BASE_URL,
):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    print("\n[RESET]")
    reset_resp = _post("/reset", {}, base_url)
    if not reset_resp:
        return

    obs = reset_resp.get("observation", {})
    _print_metadata(obs, reveal_label=reveal_label)

    actions = actions or [
        ("classify", "complaint"),
        (
            "reply",
            "We sincerely apologize for the issue you experienced. We understand your frustration and "
            "will resolve this immediately. Our team will contact you within 24 hours with a solution.",
        ),
        ("escalate", "yes"),
    ]

    total_reward = 0.0
    rewards_by_action: Dict[str, float] = {}

    for i, (action_type, content) in enumerate(actions, start=1):
        print(f"\n{'=' * 70}")
        print(f"STEP {i}: {action_type.upper()}")
        print("=" * 70)

        step_resp = _step(action_type, content, base_url)
        if not step_resp:
            return

        reward = float(step_resp.get("reward", 0.0))
        done = bool(step_resp.get("done", False))
        obs = step_resp.get("observation", {})

        total_reward += reward
        rewards_by_action[action_type] = reward

        print(f"\n📤 Action  : {action_type}")
        print(f"   Content : {content[:80]}{'...' if len(content) > 80 else ''}")
        print(f"\n🎯 Reward     : {reward:.3f}")
        print(f"   Cumulative : {total_reward:.3f}")
        print(f"   Done       : {done}")
        print(f"\n📋 History ({len(obs.get('history', []))} steps):")
        for j, h in enumerate(obs.get("history", []), 1):
            print(f"   {j}. {h}")

        if done:
            break

    print(f"\n{'=' * 70}")
    print("EPISODE SUMMARY")
    print("=" * 70)
    print("\n📊 Reward Breakdown:")
    print(f"   Classify : {rewards_by_action.get('classify', 0.0):.3f}  (weight 40%)")
    print(f"   Reply    : {rewards_by_action.get('reply', 0.0):.3f}  (weight 35%)")
    print(f"   Escalate : {rewards_by_action.get('escalate', 0.0):.3f}  (weight 25%)")
    print(f"\n   Total    : {total_reward:.3f}")

    if total_reward >= 0.85:
        print("\n🌟 Excellent — agent handled the scenario very well.")
    elif total_reward >= 0.70:
        print("\n✓ Good — mostly correct decisions.")
    elif total_reward >= 0.50:
        print("\n~ Fair — room for improvement.")
    else:
        print("\n✗ Poor — agent needs better classification and escalation logic.")
