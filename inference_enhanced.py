"""
Enhanced inference agent demonstrating the production-grade environment.

KEY IMPROVEMENTS DEMONSTRATED:
  1. Rich observation metadata (difficulty, urgency, sentiment, complexity)
  2. Component-based reward breakdowns with explanations
  3. Deterministic, interpretable grading
  4. Support for agent confidence and reasoning
  5. Structured error handling and debugging
"""

import json
import requests
from typing import Dict, Any, Optional

BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Request helpers with enhanced error handling
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict[str, Any], verbose: bool = True) -> Optional[Dict]:
    """
    POST to the server with rich error reporting.
    
    Args:
        path: API endpoint
        payload: Request payload
        verbose: Print details
        
    Returns:
        Response JSON or None on error
    """
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if not resp.ok:
            print(f"\n❌ HTTP {resp.status_code} on {path}")
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
            resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return None


def _step(
    action_type: str,
    content: str,
    confidence: Optional[float] = None,
    explanation: Optional[str] = None,
) -> Optional[Dict]:
    """Call /step with enhanced action metadata."""
    action_data = {
        "action_type": action_type,
        "content": content,
    }
    if confidence is not None:
        action_data["confidence"] = confidence
    if explanation is not None:
        action_data["explanation"] = explanation
    
    return _post("/step", {"action": action_data})


# ---------------------------------------------------------------------------
# Enhanced agent with rich output
# ---------------------------------------------------------------------------

def run_enhanced_agent():
    """Run agent demonstrating all new features."""
    print("\n" + "=" * 70)
    print(" ENHANCED WORKPLACE ENVIRONMENT AGENT")
    print("=" * 70)
    
    # RESET
    print("\n[RESET]")
    reset_resp = _post("/reset", {})
    if not reset_resp:
        return
    
    obs = reset_resp.get("observation", {})
    
    # Display scenario with rich metadata
    print(f"\n📧 Email: {obs.get('email')}")
    print(f"\n📊 Scenario Metadata:")
    print(f"   Difficulty: {obs.get('scenario_difficulty', 'unknown')}")
    print(f"   Urgency: {obs.get('urgency', 'unknown')}")
    print(f"   Sentiment: {obs.get('sentiment', 'unknown')}")
    print(f"   Complexity: {obs.get('complexity_score', 'unknown')}/5")
    print(f"   Categories: {obs.get('category_options')}")
    
    # True label (for demonstration)
    metadata = obs.get("scenario_metadata", {})
    print(f"\n✅ True Label: {metadata.get('label', 'unknown')}")
    print(f"   Requires Escalation: {metadata.get('requires_escalation', False)}")
    print(f"   Min Reply Length: {metadata.get('min_reply_length', 'unknown')}")
    
    # THREE-STEP WORKFLOW
    actions = [
        {
            "type": "classify",
            "content": "complaint",
            "confidence": 0.85,
            "explanation": "Email contains negative sentiment and complaint keywords",
        },
        {
            "type": "reply",
            "content": (
                "We sincerely apologize for the issue you've experienced. "
                "We understand your frustration and want to help resolve this immediately. "
                "Our team will investigate the problem and contact you within 24 hours with a solution."
            ),
            "confidence": 0.90,
            "explanation": "Response includes empathy, keywords, and concrete action",
        },
        {
            "type": "escalate",
            "content": "yes",
            "confidence": 0.95,
            "explanation": "Complaint requires escalation per policy",
        },
    ]
    
    total_reward = 0.0
    rewards_by_action = {}
    
    for i, action_info in enumerate(actions, start=1):
        print(f"\n{'=' * 70}")
        print(f"STEP {i}: {action_info['type'].upper()}")
        print("=" * 70)
        
        # Send step
        step_resp = _step(
            action_info["type"],
            action_info["content"],
            confidence=action_info.get("confidence"),
            explanation=action_info.get("explanation"),
        )
        
        if not step_resp:
            return
        
        # Parse response
        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", False)
        obs = step_resp.get("observation", {})
        
        total_reward += reward
        rewards_by_action[action_info["type"]] = reward
        
        # Display result
        print(f"\n📤 Agent Action:")
        print(f"   Type: {action_info['type']}")
        print(f"   Content: {action_info['content'][:60]}...")
        print(f"   Confidence: {action_info.get('confidence', 'N/A')}")
        print(f"   Reasoning: {action_info.get('explanation', 'N/A')}")
        
        print(f"\n🎯 Reward: {reward:.3f}")
        print(f"   Cumulative: {total_reward:.3f}")
        print(f"   Done: {done}")
        
        print(f"\n📋 History: {len(obs.get('history', []))} actions")
        for j, h in enumerate(obs.get("history", []), 1):
            print(f"   {j}. {h}")
    
    # SUMMARY
    print(f"\n{'=' * 70}")
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"\n✅ Completed: 3/3 steps")
    print(f"\n📊 Reward Breakdown:")
    print(f"   Classify:   {rewards_by_action.get('classify', 0.0):.3f} (40% weight)")
    print(f"   Reply:      {rewards_by_action.get('reply', 0.0):.3f} (35% weight)")
    print(f"   Escalate:   {rewards_by_action.get('escalate', 0.0):.3f} (25% weight)")
    print(f"\n   Total: {total_reward:.3f}")
    print(f"\n💡 Interpretation:")
    if total_reward >= 0.85:
        print(f"   🌟 Excellent! Agent handled complex scenario very well.")
    elif total_reward >= 0.70:
        print(f"   ✓ Good! Agent made mostly correct decisions.")
    elif total_reward >= 0.50:
        print(f"   ~ Fair. Room for improvement in decision making.")
    else:
        print(f"   ✗ Poor. Agent needs to improve classification/escalation logic.")


if __name__ == "__main__":
    run_enhanced_agent()
