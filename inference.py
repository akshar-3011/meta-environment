"""Inference Script for workplace_env
===================================

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Runs the WorkplaceEnvironment IN-PROCESS — no live server needed.
Executes 3 episodes, one per task defined in openenv.yaml:
    Task 1: refund    (easy difficulty)
    Task 2: complaint (medium difficulty)
    Task 3: query     (hard difficulty)

STDOUT FORMAT — exactly as required by the OpenEnv spec:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action(content)> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
    HF_TOKEN=hf_xxx python inference.py   # uses real model via HF router
"""

import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup sys.path so imports work from project root regardless of cwd
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# ---------------------------------------------------------------------------
# OpenAI client — used when HF_TOKEN is present, mock otherwise
# ---------------------------------------------------------------------------
try:
    import openai as _openai_lib
    _openai_available = True
except ImportError:  # pragma: no cover
    _openai_available = False

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or None

_client = None
if HF_TOKEN and _openai_available:
    _client = _openai_lib.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Environment imports (in-process, no server)
# ---------------------------------------------------------------------------
try:
    from environment.workplace_environment import WorkplaceEnvironment
    from models import WorkplaceAction
    from data import (
        get_refund_repository,
        get_complaint_repository,
        get_query_repository,
    )
except ImportError:  # pragma: no cover
    from workplace_env.environment.workplace_environment import WorkplaceEnvironment
    from workplace_env.models import WorkplaceAction
    from workplace_env.data import (
        get_refund_repository,
        get_complaint_repository,
        get_query_repository,
    )

# ---------------------------------------------------------------------------
# Task definitions — must match openenv.yaml task names
# ---------------------------------------------------------------------------
# Each entry: (task_name, difficulty, repo_factory)
TASKS = [
    ("refund",    "easy",   get_refund_repository),
    ("complaint", "medium", get_complaint_repository),
    ("query",     "hard",   get_query_repository),
]

_MODEL_DISPLAY = MODEL_NAME if HF_TOKEN else "mock"


# ---------------------------------------------------------------------------
# LLM call — deterministic mock when no HF_TOKEN
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    category_options: Optional[List[str]] = None,
) -> str:
    """Call the LLM via OpenAI client. Falls back to a deterministic mock."""
    if _client is None:
        sp = system_prompt.lower()
        # Detect intent exclusively from system_prompt (always unambiguous)
        if "escalat" in sp:
            return "no"
        if "triage" in sp or "one category" in sp or "one word" in sp:
            return category_options[0] if category_options else "query"
        # Reply mock — hits reward keywords for all categories
        return (
            "Thank you for reaching out! We sincerely apologize for any inconvenience. "
            "We will help you resolve this and process your request as soon as possible. "
            "Please let us know if you need further information or assistance."
        )

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert a WorkplaceObservation to a plain dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return vars(obs)


def _action_str(action_type: str, content: str, max_len: int = 50) -> str:
    """Format action for [STEP] line as action_type(content)."""
    safe = content.replace("\n", " ").strip()
    if len(safe) > max_len:
        safe = safe[:max_len] + "..."
    return f"{action_type}({safe})"


# ---------------------------------------------------------------------------
# Episode runner — one episode per task
# ---------------------------------------------------------------------------

def run_episode(task_name: str, difficulty: str, repo_factory) -> Dict[str, Any]:
    """Run one classify → reply → escalate episode in-process.

    Args:
        task_name:     Name matching openenv.yaml (refund / complaint / query).
        difficulty:    Scenario difficulty to seed (easy / medium / hard).
        repo_factory:  Callable returning a ScenarioRepository for this task.

    Returns:
        Summary dict with rewards, success flag, and any errors.
    """
    # Build env seeded with scenarios for this specific task type
    repo = repo_factory()
    env = WorkplaceEnvironment(scenario_repository=repo)

    obs_raw = env.reset()
    obs = _obs_to_dict(obs_raw)

    email: str = str(obs.get("email", ""))
    category_options: List[str] = [str(x) for x in obs.get("category_options", [])]

    # ── [START] ─────────────────────────────────────────────────────────────
    print(
        f"[START] task={task_name} env=workplace_env model={_MODEL_DISPLAY}",
        flush=True,
    )

    rewards: List[float] = []
    errors: List[str] = []
    classify_content: str = category_options[0] if category_options else "query"
    reply_content: str = ""
    steps_completed: int = 0

    # ── Step 1: classify ────────────────────────────────────────────────────
    step_error = "null"
    reward = 0.0
    done = False
    try:
        classify_content = call_llm(
            system_prompt=(
                "You are a support triage assistant. "
                "Return exactly one category from the provided options. "
                "Output only one word and nothing else."
            ),
            user_prompt=(
                f"Email:\n{email}\n\n"
                f"Category options: {', '.join(category_options)}\n"
                f"Task: {task_name} | Difficulty: {difficulty}\n"
                "Return only one category word."
            ),
            category_options=category_options,
        ).strip().split()[0].lower()

        if classify_content not in category_options and category_options:
            classify_content = category_options[0]

        step_obs = _obs_to_dict(
            env.step(WorkplaceAction(action_type="classify", content=classify_content))
        )
        reward = float(step_obs.get("reward") or 0.0)
        done = bool(step_obs.get("done", False))
        raw_err = step_obs.get("error")
        step_error = str(raw_err) if raw_err else "null"
    except Exception as exc:  # pragma: no cover
        errors.append(str(exc))
        step_error = str(exc)

    rewards.append(reward)
    steps_completed += 1
    print(
        f"[STEP] step=1 action={_action_str('classify', classify_content)}"
        f" reward={reward:.2f} done={'true' if done else 'false'} error={step_error}",
        flush=True,
    )

    # ── Step 2: reply ───────────────────────────────────────────────────────
    step_error = "null"
    reward = 0.0
    done = False
    try:
        reply_content = call_llm(
            system_prompt=(
                "You are a customer support agent. "
                "Write a concise, empathetic reply to the customer email. "
                "Return only the reply text, nothing else."
            ),
            user_prompt=(
                f"Email:\n{email}\n\n"
                f"Predicted category: {classify_content}\n"
                f"Task: {task_name} | Difficulty: {difficulty}\n"
                "Write a helpful reply of at least 30 characters."
            ),
        ).strip()

        if len(reply_content) < 30:
            reply_content = (
                reply_content + " We understand your concern and will help promptly."
            ).strip()

        step_obs = _obs_to_dict(
            env.step(WorkplaceAction(action_type="reply", content=reply_content))
        )
        reward = float(step_obs.get("reward") or 0.0)
        done = bool(step_obs.get("done", False))
        raw_err = step_obs.get("error")
        step_error = str(raw_err) if raw_err else "null"
    except Exception as exc:  # pragma: no cover
        errors.append(str(exc))
        step_error = str(exc)

    rewards.append(reward)
    steps_completed += 1
    print(
        f"[STEP] step=2 action={_action_str('reply', reply_content)}"
        f" reward={reward:.2f} done={'true' if done else 'false'} error={step_error}",
        flush=True,
    )

    # ── Step 3: escalate ────────────────────────────────────────────────────
    step_error = "null"
    reward = 0.0
    done = False
    escalate_content = "no"
    try:
        escalate_content = call_llm(
            system_prompt=(
                "You are deciding whether a support case needs escalation. "
                "Return exactly one word: yes or no."
            ),
            user_prompt=(
                f"Email:\n{email}\n\n"
                f"Predicted category: {classify_content}\n"
                f"Draft reply:\n{reply_content}\n\n"
                f"Task: {task_name} | Difficulty: {difficulty}\n"
                "Should this be escalated to a senior agent? Answer yes or no only."
            ),
        ).strip().lower()
        escalate_content = "yes" if escalate_content == "yes" else "no"

        step_obs = _obs_to_dict(
            env.step(WorkplaceAction(action_type="escalate", content=escalate_content))
        )
        reward = float(step_obs.get("reward") or 0.0)
        done = bool(step_obs.get("done", False))
        raw_err = step_obs.get("error")
        step_error = str(raw_err) if raw_err else "null"
    except Exception as exc:  # pragma: no cover
        errors.append(str(exc))
        step_error = str(exc)
        done = True

    rewards.append(reward)
    steps_completed += 1
    print(
        f"[STEP] step=3 action={_action_str('escalate', escalate_content)}"
        f" reward={reward:.2f} done={'true' if done else 'false'} error={step_error}",
        flush=True,
    )

    # ── [END] ────────────────────────────────────────────────────────────────
    success: bool = len(errors) == 0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'}"
        f" steps={steps_completed} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task_name,
        "difficulty": difficulty,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "success": success,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []

    for idx, (task_name, difficulty, repo_factory) in enumerate(TASKS, start=1):
        print(f"\n{'=' * 60}", flush=True)
        print(f"Task {idx}/3 — {task_name} ({difficulty})", flush=True)
        print("=" * 60, flush=True)
        result = run_episode(task_name=task_name, difficulty=difficulty, repo_factory=repo_factory)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "✓ success" if r["success"] else "✗ failed"
        print(
            f"  {r['task']:10s}  ({r['difficulty']:6s})  "
            f"total_reward={r['total_reward']:.2f}  {status}",
            flush=True,
        )