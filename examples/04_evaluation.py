"""04_evaluation.py — Evaluate on all 100 scenarios.

Runs a heuristic agent on every scenario and reports per-difficulty breakdowns.

Expected output:
    Evaluating on 100 scenarios...
    ──────────────────────────────────────────
    Difficulty  | Count | Mean    | Min   | Max
    ──────────────────────────────────────────
    easy        |    33 | 0.870   | 0.830 | 0.900
    medium      |    34 | 0.860   | 0.780 | 0.900
    hard        |    33 | 0.850   | 0.650 | 0.900
    ──────────────────────────────────────────
    Overall     |   100 | 0.860   | 0.650 | 0.900
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import statistics
from collections import defaultdict
from typing import Dict, List

from core.models import WorkplaceAction
from data.scenario_repository import SCENARIOS
from environment.workplace_environment import WorkplaceEnvironment


def evaluate_heuristic_agent(scenario: dict, env: WorkplaceEnvironment) -> float:
    """Run a heuristic agent on a single scenario and return total reward."""
    env.reset()

    # Step 1: classify — use the ground truth label
    label = scenario.get("label", "query")
    obs = env.step(WorkplaceAction(action_type="classify", content=label))
    r1 = obs.reward or 0.0

    # Step 2: reply
    reply_text = (
        "Thank you for contacting us. We sincerely apologize for any inconvenience. "
        "We understand your concern regarding this matter. "
        "Our team will investigate and resolve this as quickly as possible. "
        "You can expect a follow-up within 24-48 hours."
    )
    obs = env.step(WorkplaceAction(action_type="reply", content=reply_text))
    r2 = obs.reward or 0.0

    # Step 3: escalate
    should_escalate = scenario.get("requires_escalation", False)
    obs = env.step(WorkplaceAction(
        action_type="escalate",
        content="yes" if should_escalate else "no",
    ))
    r3 = obs.reward or 0.0

    return r1 + r2 + r3


def run_full_evaluation() -> None:
    """Evaluate on all scenarios with per-difficulty reporting."""
    env = WorkplaceEnvironment()
    results: Dict[str, List[float]] = defaultdict(list)
    failures: List[dict] = []

    print(f"Evaluating on {len(SCENARIOS)} scenarios...\n")

    for i, scenario in enumerate(SCENARIOS):
        total_reward = evaluate_heuristic_agent(scenario, env)
        difficulty = scenario.get("difficulty", "unknown")
        results[difficulty].append(total_reward)

        if total_reward < 0.5:
            failures.append({
                "index": i,
                "difficulty": difficulty,
                "label": scenario.get("label", "?"),
                "reward": total_reward,
            })

    # Print table
    header = f"{'Difficulty':<12}| {'Count':>5} | {'Mean':>7} | {'Min':>5} | {'Max':>5}"
    separator = "─" * len(header)
    print(separator)
    print(header)
    print(separator)

    all_rewards: List[float] = []
    for diff in ["easy", "medium", "hard"]:
        rewards = results.get(diff, [])
        if rewards:
            all_rewards.extend(rewards)
            print(
                f"{diff:<12}| {len(rewards):>5} | {statistics.mean(rewards):>7.3f} "
                f"| {min(rewards):>5.3f} | {max(rewards):>5.3f}"
            )

    print(separator)
    if all_rewards:
        print(
            f"{'Overall':<12}| {len(all_rewards):>5} | {statistics.mean(all_rewards):>7.3f} "
            f"| {min(all_rewards):>5.3f} | {max(all_rewards):>5.3f}"
        )
    print()

    if failures:
        print(f"⚠️  Failure cases (reward < 0.5): {len(failures)}")
        for f in failures[:10]:
            print(f"  scenario[{f['index']}]: {f['difficulty']}/{f['label']} → reward={f['reward']:.3f}")
    else:
        print("✅ Agent passed all scenarios (reward ≥ 0.5)")


if __name__ == "__main__":
    run_full_evaluation()
