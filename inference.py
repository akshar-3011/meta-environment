"""
inference.py — Agent script that drives the WorkplaceEnv server.

ROOT CAUSE OF THE ORIGINAL BUG (and the fix applied here):
============================================================
OpenEnv's /step endpoint is defined as:

    class StepRequest(BaseModel):
        action: Dict[str, Any]   # <-- required key
        ...

So the correct payload is ALWAYS:

    POST /step
    Content-Type: application/json
    {
        "action": {
            "action_type": "classify",
            "content": "refund"
        }
    }

The original inference.py used the right shape but had two problems:
  1. It called /step on an episode that was already done (step_count >= 3),
     which caused the environment to misbehave.
  2. It did not check response.ok before calling .json(), causing a
     JSONDecodeError when the server returned a 422 validation error.

This version:
  - Wraps every request in a helper that checks status and prints errors.
  - Stops calling /step once done=True is received.
  - Shows the full observation, reward and done after every step.
"""

import json
import requests

BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict | None = None) -> dict:
    """POST to the server, raise clearly on HTTP errors."""
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, json=payload or {}, headers={"Content-Type": "application/json"})
    if not resp.ok:
        print(f"\n[ERROR] POST {path} → HTTP {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text)
        resp.raise_for_status()
    return resp.json()


def _step(action_type: str, content: str) -> dict:
    """
    Call /step with the correct OpenEnv payload shape:

        {"action": {"action_type": "...", "content": "..."}}

    The outer "action" key is REQUIRED by StepRequest.
    The inner dict is validated against WorkplaceAction.
    """
    return _post("/step", {"action": {"action_type": action_type, "content": content}})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run_agent():
    print("=" * 60)
    print("RESET")
    print("=" * 60)
    reset_data = _post("/reset")
    obs = reset_data["observation"]
    print(f"Email          : {obs['email']}")
    print(f"Categories     : {obs['category_options']}")
    print(f"History        : {obs['history']}")
    print(f"Reward         : {reset_data['reward']}")
    print(f"Done           : {reset_data['done']}")

    # Define the three-step workflow
    actions = [
        ("classify", "refund"),
        ("reply",    "Sorry for the inconvenience. We will process your refund within 3 business days."),
        ("escalate", "urgent"),
    ]

    for i, (action_type, content) in enumerate(actions, start=1):
        print(f"\n{'=' * 60}")
        print(f"STEP {i}: {action_type!r} → {content!r}")
        print("=" * 60)

        data = _step(action_type, content)

        obs = data.get("observation") or {}
        print(f"Email          : {obs.get('email', '')}")
        print(f"History        : {obs.get('history', [])}")
        print(f"Reward         : {data['reward']}")
        print(f"Done           : {data['done']}")

        if data["done"]:
            print("\nEpisode finished.")
            break

    print("\n[DONE] Full workflow completed successfully.")


if __name__ == "__main__":
    run_agent()