"""Inference script for the workplace_env OpenEnv environment.

Runs the WorkplaceEnvironment **in-process** — no live server required.
Executes one episode per task (refund / complaint / query), matching the
task names declared in openenv.yaml.

Mandatory environment variables:
    API_BASE_URL  - LLM endpoint  (default: HuggingFace router)
    MODEL_NAME    - Model identifier
    HF_TOKEN      - HuggingFace / API key  (if absent: deterministic mock)

Usage:
    python inference.py
    HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py

STDOUT FORMAT  (exactly as required by the OpenEnv spec):
    [START] task=<task_name> env=workplace_env model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,r3>
"""

import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration — mandatory variables per submission spec
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Accept both HF_TOKEN (primary) and OPENAI_API_KEY (alias)
HF_TOKEN: Optional[str] = (
    os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or None
)

# ---------------------------------------------------------------------------
# OpenAI client — only instantiated when a token is present
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI as _OpenAI
    _openai_available = True
except ImportError:  # pragma: no cover
    _OpenAI = None  # type: ignore
    _openai_available = False

_client = None
if HF_TOKEN and _openai_available:
    _client = _OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Environment path bootstrap
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Top-level imports so they're resolved once, not inside a loop
try:
    from environment.workplace_environment import WorkplaceEnvironment
    from data import get_default_repository
    from models import WorkplaceAction
    from core.inference.strategies import EmailAwareInference
except ImportError:
    from workplace_env.environment.workplace_environment import WorkplaceEnvironment
    from workplace_env.data import get_default_repository
    from workplace_env.models import WorkplaceAction
    from workplace_env.core.inference.strategies import EmailAwareInference


# ---------------------------------------------------------------------------
# Task → difficulty pinning (shows easy → medium → hard progression)
# ---------------------------------------------------------------------------
TASK_DIFFICULTY: Dict[str, str] = {
    "refund": "easy",
    "complaint": "medium",
    "query": "hard",
}

# Task names must match openenv.yaml exactly
TASKS: List[str] = ["refund", "complaint", "query"]


# ---------------------------------------------------------------------------
# LLM helper — falls back to a deterministic mock when no token is set
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    category_options: Optional[List[str]] = None,
    email: str = "",
) -> str:
    """Call the chat model; fall back to mock when HF_TOKEN is absent."""
    if _client is None:
        agent = EmailAwareInference()
        category = agent._classify_email(email) if email else "query"
        
        # Deterministic mock — checks prompt type from system instruction
        if "escalate" in system_prompt.lower():
            return "yes" if category == "complaint" else "no"
        if "classify" in system_prompt.lower() or "category" in user_prompt.lower():
            return category
        # Return fallback reply
        return agent._REPLIES.get(category, agent._REPLIES["query"])

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Environment factory — filter scenarios to the right label + difficulty
# ---------------------------------------------------------------------------

def _make_env(task_name: str) -> WorkplaceEnvironment:
    """Return a WorkplaceEnvironment pre-seeded with matching scenarios."""
    difficulty = TASK_DIFFICULTY[task_name]
    all_scenarios = get_default_repository().list_scenarios()

    # Prefer label + difficulty match; fall back to label-only if needed
    filtered = [
        s for s in all_scenarios
        if s.get("label") == task_name and s.get("difficulty") == difficulty
    ]
    if not filtered:
        filtered = [s for s in all_scenarios if s.get("label") == task_name]
    if not filtered:
        filtered = all_scenarios  # ultimate fallback

    class _Repo:
        def list_scenarios(self):
            return filtered

    return WorkplaceEnvironment(scenario_repository=_Repo())


def _obs_dict(obs) -> Dict[str, Any]:
    """Convert a Pydantic observation to a plain dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return vars(obs)


# ---------------------------------------------------------------------------
# Episode runner — one task = one classify → reply → escalate episode
# ---------------------------------------------------------------------------

def run_episode(task_name: str) -> Dict[str, Any]:
    """Run a single episode for *task_name* and print spec-compliant lines."""
    env = _make_env(task_name)
    obs = _obs_dict(env.reset())

    email: str = str(obs.get("email", ""))
    category_options: List[str] = [str(x) for x in obs.get("category_options", [])]
    difficulty: str = TASK_DIFFICULTY[task_name]

    # [START] — exactly one per episode, no other stdout before this
    print(
        f"[START] task={task_name} env=workplace_env model={MODEL_NAME}",
        flush=True,
    )

    classify_reward = reply_reward = escalate_reward = 0.0
    errors: List[str] = []
    classify_content = category_options[0] if category_options else "query"
    reply_content = ""

    # ---- Step 1: classify ------------------------------------------------
    try:
        classify_system = (
            "You are a support triage assistant. "
            "Return exactly one category from the provided category_options. "
            "Output only one word and nothing else."
        )
        classify_user = (
            f"Email:\n{email}\n\n"
            f"Category options: {', '.join(category_options)}\n"
            f"Difficulty: {difficulty}\n"
            "Return only the category name — one word."
        )
        raw = call_llm(classify_system, classify_user, category_options=category_options, email=email)
        classify_content = raw.strip().split()[0].lower()
        if classify_content not in category_options and category_options:
            classify_content = category_options[0]

        step_obs = _obs_dict(env.step(WorkplaceAction(action_type="classify", content=classify_content)))
        classify_reward = float(step_obs.get("reward") or 0.0)
        step_error = step_obs.get("error")
        print(
            f"[STEP] step=1 action=classify reward={classify_reward:.2f}"
            f" done=false error={step_error or 'null'}",
            flush=True,
        )
    except Exception as exc:
        errors.append(str(exc))
        print(
            f"[STEP] step=1 action=classify reward=0.00 done=false error={exc}",
            flush=True,
        )

    # ---- Step 2: reply ---------------------------------------------------
    try:
        reply_system = (
            "You are a customer support agent. "
            "Write a concise and empathetic reply to the customer email. "
            "Return only the reply text."
        )
        reply_user = (
            f"Email:\n{email}\n\n"
            f"Predicted category: {classify_content}\n"
            f"Category options: {', '.join(category_options)}\n"
            f"Difficulty: {difficulty}\n"
            "Write a helpful reply of at least 30 characters."
        )
        reply_content = call_llm(reply_system, reply_user, email=email).strip()
        if len(reply_content) < 30:
            reply_content = (
                reply_content + " We understand your concern and will help promptly."
            ).strip()

        step_obs = _obs_dict(env.step(WorkplaceAction(action_type="reply", content=reply_content)))
        reply_reward = float(step_obs.get("reward") or 0.0)
        step_error = step_obs.get("error")
        print(
            f"[STEP] step=2 action=reply reward={reply_reward:.2f}"
            f" done=false error={step_error or 'null'}",
            flush=True,
        )
    except Exception as exc:
        errors.append(str(exc))
        print(
            f"[STEP] step=2 action=reply reward=0.00 done=false error={exc}",
            flush=True,
        )

    # ---- Step 3: escalate ------------------------------------------------
    try:
        escalate_system = (
            "You are deciding whether to escalate a support interaction. "
            "Return exactly one word: yes or no."
        )
        escalate_user = (
            f"Email:\n{email}\n\n"
            f"Predicted category: {classify_content}\n"
            f"Draft reply:\n{reply_content}\n\n"
            f"Difficulty: {difficulty}\n"
            "Should this be escalated? Return yes or no only."
        )
        raw = call_llm(escalate_system, escalate_user, email=email).strip().lower()
        escalate_content = "yes" if raw == "yes" else "no"

        step_obs = _obs_dict(env.step(WorkplaceAction(action_type="escalate", content=escalate_content)))
        escalate_reward = float(step_obs.get("reward") or 0.0)
        done_flag = bool(step_obs.get("done", False))
        step_error = step_obs.get("error")
        print(
            f"[STEP] step=3 action=escalate reward={escalate_reward:.2f}"
            f" done={'true' if done_flag else 'false'} error={step_error or 'null'}",
            flush=True,
        )
    except Exception as exc:
        errors.append(str(exc))
        print(
            f"[STEP] step=3 action=escalate reward=0.00 done=true error={exc}",
            flush=True,
        )

    rewards = [classify_reward, reply_reward, escalate_reward]
    success: bool = sum(rewards) >= 0.5
    print(
        f"[END] success={'true' if success else 'false'} steps=3"
        f" rewards={classify_reward:.2f},{reply_reward:.2f},{escalate_reward:.2f}",
        flush=True,
    )

    return {
        "task": task_name,
        "difficulty": difficulty,
        "classify_reward": classify_reward,
        "reply_reward": reply_reward,
        "escalate_reward": escalate_reward,
        "total_reward": classify_reward + reply_reward + escalate_reward,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []
    for task in TASKS:
        result = run_episode(task)
        results.append(result)
    # Summary goes to stderr so it never pollutes the stdout evaluation stream
    print("\n--- Baseline Summary ---", file=sys.stderr)
    for r in results:
        status = "success" if r["success"] else "FAILED"
        print(
            f"  [{status:7s}] task={r['task']:9s} difficulty={r['difficulty']:6s}"
            f"  total={r['total_reward']:.3f}"
            f"  (classify={r['classify_reward']:.2f}"
            f" reply={r['reply_reward']:.2f}"
            f" escalate={r['escalate_reward']:.2f})",
            file=sys.stderr,
        )