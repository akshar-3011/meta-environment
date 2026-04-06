"""Inference script for workplace_env — runs entirely in-process, no HTTP server needed."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI

from environment.workplace_environment import WorkplaceEnvironment
from core.models.workplace import WorkplaceAction
from data.scenario_repository import (
    get_refund_repository,
    get_complaint_repository,
    get_query_repository,
)

# ---------------------------------------------------------------------------
# Config — exact names required by spec
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

SUCCESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured chat completion model and return plain text output."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, repo) -> float:
    """Run one episode for a given task. Returns total reward."""
    try:
        env = WorkplaceEnvironment(scenario_repository=repo)
        obs = env.reset()

        email = obs.email
        category_options = obs.category_options
        difficulty = obs.scenario_difficulty

        print(
            f"[START] task={task_name} env=workplace_env model={MODEL_NAME}",
            flush=True,
        )

        rewards = [0.0, 0.0, 0.0]

        # ── Step 1: classify ─────────────────────────────────────────────────
        classify_content = call_llm(
            system_prompt=(
                "You are a support triage agent. "
                "Return exactly one word from the options provided. "
                "Output only one word."
            ),
            user_prompt=(
                f"Email: {email}\n"
                f"Options: {', '.join(category_options)}\n"
                "Return one word only."
            ),
        ).strip().split()[0].lower()

        if classify_content not in category_options:
            classify_content = category_options[0] if category_options else "query"

        step_obs = env.step(WorkplaceAction(action_type="classify", content=classify_content))
        rewards[0] = float(step_obs.reward or 0.0)
        done_1 = "true" if step_obs.done else "false"
        print(
            f"[STEP] step=1 action=classify reward={rewards[0]:.2f}"
            f" done={done_1} error=null",
            flush=True,
        )

        # ── Step 2: reply ────────────────────────────────────────────────────
        reply_content = call_llm(
            system_prompt=(
                "You are a customer support agent. "
                "Write an empathetic reply of at least 40 words."
            ),
            user_prompt=(
                f"Email: {email}\n"
                f"Category: {classify_content}\n"
                "Write your reply."
            ),
        ).strip()

        step_obs = env.step(WorkplaceAction(action_type="reply", content=reply_content))
        rewards[1] = float(step_obs.reward or 0.0)
        done_2 = "true" if step_obs.done else "false"
        print(
            f"[STEP] step=2 action=reply reward={rewards[1]:.2f}"
            f" done={done_2} error=null",
            flush=True,
        )

        # ── Step 3: escalate ─────────────────────────────────────────────────
        escalate_content = call_llm(
            system_prompt="You decide escalation. Return exactly: yes or no.",
            user_prompt=(
                f"Email: {email}\n"
                f"Category: {classify_content}\n"
                f"Reply: {reply_content}\n"
                "Escalate? yes or no only."
            ),
        ).strip().lower()
        escalate_content = "yes" if escalate_content == "yes" else "no"

        step_obs = env.step(WorkplaceAction(action_type="escalate", content=escalate_content))
        rewards[2] = float(step_obs.reward or 0.0)
        done_3 = "true" if step_obs.done else "false"
        print(
            f"[STEP] step=3 action=escalate reward={rewards[2]:.2f}"
            f" done={done_3} error=null",
            flush=True,
        )

        # ── [END] ─────────────────────────────────────────────────────────────
        total_reward = sum(rewards)
        success = "true" if total_reward >= SUCCESS_THRESHOLD else "false"
        print(
            f"[END] success={success} steps=3"
            f" rewards={rewards[0]:.2f},{rewards[1]:.2f},{rewards[2]:.2f}",
            flush=True,
        )
        return total_reward

    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.00,0.00,0.00", flush=True)
        return 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks = [
        ("refund", get_refund_repository()),
        ("complaint", get_complaint_repository()),
        ("query", get_query_repository()),
    ]
    for task_name, repo in tasks:
        run_task(task_name, repo)