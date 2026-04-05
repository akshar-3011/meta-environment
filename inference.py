"""Standalone inference runner for the workplace_env HTTP environment."""

import os
from typing import Any, Dict, List

import openai
import requests


# Required env vars with sensible defaults
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def run_episode(env_base_url: str) -> float:
    """Run one classify -> reply -> escalate episode and return total reward."""
    env_base_url = env_base_url.rstrip("/")

    reset_data = _post_json(f"{env_base_url}/reset", {})
    observation: Dict[str, Any] = reset_data.get("observation", {})
    email = str(observation.get("email", ""))
    category_options: List[str] = [str(x) for x in observation.get("category_options", [])]
    difficulty = observation.get("difficulty", "unknown")

    print(f"[START] task=workplace_env env=workplace_env model={MODEL_NAME}", flush=True)

    total_reward = 0.0
    reply_text = ""

    # 1) classify
    classify_system = (
        "You are a support triage assistant. "
        "Return exactly one category from the provided category_options. "
        "Output only one word and nothing else."
    )
    classify_user = (
        f"Email:\n{email}\n\n"
        f"Category options: {', '.join(category_options)}\n"
        f"Difficulty: {difficulty}\n"
        "Return only one category."
    )
    classify_content = call_llm(classify_system, classify_user).strip().split()[0].lower()
    if classify_content not in category_options and category_options:
        classify_content = category_options[0]

    classify_step = _post_json(
        f"{env_base_url}/step",
        {"action": {"action_type": "classify", "content": classify_content}},
    )
    classify_reward = float(classify_step.get("reward", 0.0))
    total_reward += classify_reward
    print(
        f"[STEP] step=1 action=classify reward={classify_reward:.2f} done=false error=null",
        flush=True,
    )

    # 2) reply
    reply_system = (
        "You are a customer support agent. "
        "Write a concise and empathetic response to the customer. "
        "Return only the reply text."
    )
    reply_user = (
        f"Email:\n{email}\n\n"
        f"Predicted category: {classify_content}\n"
        f"Category options: {', '.join(category_options)}\n"
        f"Difficulty: {difficulty}\n"
        "Reply with at least 30 characters."
    )
    reply_content = call_llm(reply_system, reply_user).strip()
    if len(reply_content) < 30:
        reply_content = (reply_content + " We understand your concern and will help promptly.").strip()
    reply_text = reply_content

    reply_step = _post_json(
        f"{env_base_url}/step",
        {"action": {"action_type": "reply", "content": reply_content}},
    )
    reply_reward = float(reply_step.get("reward", 0.0))
    total_reward += reply_reward
    print(
        f"[STEP] step=2 action=reply reward={reply_reward:.2f} done=false error=null",
        flush=True,
    )

    # 3) escalate
    escalate_system = (
        "You are deciding escalation for a support interaction. "
        "Return exactly one word: yes or no."
    )
    escalate_user = (
        f"Email:\n{email}\n\n"
        f"Predicted category: {classify_content}\n"
        f"Draft reply:\n{reply_text}\n\n"
        f"Difficulty: {difficulty}\n"
        "Should this be escalated? Return yes or no only."
    )
    escalate_content = call_llm(escalate_system, escalate_user).strip().lower()
    escalate_content = "yes" if escalate_content == "yes" else "no"

    escalate_step = _post_json(
        f"{env_base_url}/step",
        {"action": {"action_type": "escalate", "content": escalate_content}},
    )
    escalate_reward = float(escalate_step.get("reward", 0.0))
    escalate_done = "true" if bool(escalate_step.get("done", False)) else "false"
    total_reward += escalate_reward
    print(
        f"[STEP] step=3 action=escalate reward={escalate_reward:.2f} done={escalate_done} error=null",
        flush=True,
    )

    print(
        f"[END] success=true steps=3 rewards={classify_reward:.2f},{reply_reward:.2f},{escalate_reward:.2f}",
        flush=True,
    )
    return total_reward


if __name__ == "__main__":
    result = run_episode(env_base_url=os.environ.get("ENV_BASE_URL", "http://localhost:8000"))
    print(result)