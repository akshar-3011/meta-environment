#!/usr/bin/env python3
"""Compare trained agent archetypes across all 39 scenarios.

Loads final checkpoints, evaluates each on every scenario (multiple episodes),
and produces a comparison table + CSV export.

Usage:
    python training/compare_agents.py
    python training/compare_agents.py --models-dir models/ --episodes 10
    python training/compare_agents.py --agents conservative aggressive balanced
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO

from environment.gym_wrapper import WorkplaceGymEnv
from data.scenario_repository import SCENARIOS


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_agent(
    model_path: str,
    n_episodes_per_scenario: int = 5,
) -> Dict:
    """Evaluate a single agent on all 39 scenarios.

    Returns per-difficulty stats and per-scenario raw results.
    """
    model = PPO.load(model_path)

    by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    escalation_decisions: Dict[str, Dict[str, int]] = {
        "correct": {"escalated": 0, "not_escalated": 0},
        "incorrect": {"escalated": 0, "not_escalated": 0},
    }
    scenario_results = []
    total_steps = 0

    for scenario_idx in range(len(SCENARIOS)):
        scenario = SCENARIOS[scenario_idx]
        difficulty = scenario["difficulty"]
        ep_rewards = []

        for ep in range(n_episodes_per_scenario):
            env = WorkplaceGymEnv()
            env._env._state.scenario_index = scenario_idx
            obs, _ = env.reset()
            cumulative = 0.0
            ep_info = {}

            for step in range(3):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                cumulative += reward
                total_steps += 1
                ep_info = info

                # Track escalation on step 3
                if info.get("action_type") == "escalate":
                    esc_reward = info.get("step_reward", 0.0)
                    did_escalate = "yes" in info.get("content", "").lower()
                    if esc_reward > 0.2:
                        escalation_decisions["correct"][
                            "escalated" if did_escalate else "not_escalated"
                        ] += 1
                    else:
                        escalation_decisions["incorrect"][
                            "escalated" if did_escalate else "not_escalated"
                        ] += 1

            ep_rewards.append(cumulative)
            by_difficulty[difficulty].append(cumulative)

        scenario_results.append({
            "scenario_idx": scenario_idx,
            "label": scenario["label"],
            "difficulty": difficulty,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
        })

    # Compute accuracy
    total_correct = sum(escalation_decisions["correct"].values())
    total_decisions = total_correct + sum(escalation_decisions["incorrect"].values())
    esc_accuracy = total_correct / max(total_decisions, 1) * 100

    return {
        "easy_avg": float(np.mean(by_difficulty["easy"])) if by_difficulty["easy"] else 0,
        "medium_avg": float(np.mean(by_difficulty["medium"])) if by_difficulty["medium"] else 0,
        "hard_avg": float(np.mean(by_difficulty["hard"])) if by_difficulty["hard"] else 0,
        "total_avg": float(np.mean([r for v in by_difficulty.values() for r in v])),
        "escalation_accuracy": esc_accuracy,
        "escalation_decisions": escalation_decisions,
        "total_steps": total_steps,
        "scenarios": scenario_results,
    }


def compare_agents(
    models_dir: str = "models",
    agents: Optional[List[str]] = None,
    n_episodes: int = 5,
    output_dir: str = "results",
) -> List[Dict]:
    """Compare all trained agents and produce a comparison table."""
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover agents
    if agents:
        agent_dirs = [(models_path / a) for a in agents]
    else:
        agent_dirs = sorted([
            d for d in models_path.iterdir()
            if d.is_dir() and (d / "final_model.zip").exists()
        ])

    if not agent_dirs:
        print("❌ No trained agents found. Run training first:")
        print("   python training/train_all.py --config training/configs/")
        return []

    print(f"\n{'═' * 80}")
    print(f"  MULTI-AGENT COMPARISON")
    print(f"  Agents: {', '.join(d.name for d in agent_dirs)}")
    print(f"  Scenarios: {len(SCENARIOS)} × {n_episodes} episodes each")
    print(f"{'═' * 80}\n")

    results = []
    for agent_dir in agent_dirs:
        model_path = agent_dir / "final_model.zip"
        if not model_path.exists():
            print(f"  ⚠️  {agent_dir.name}: final_model.zip not found, skipping")
            continue

        print(f"  🔍 Evaluating {agent_dir.name}...", end="", flush=True)
        t0 = time.time()
        eval_result = evaluate_agent(str(model_path), n_episodes_per_scenario=n_episodes)
        elapsed = time.time() - t0
        eval_result["agent"] = agent_dir.name
        eval_result["eval_time_s"] = round(elapsed, 1)
        results.append(eval_result)
        print(f" done ({elapsed:.1f}s)")

    if not results:
        return []

    # Print comparison table
    print(f"\n{'═' * 80}")
    print(f"  {'Agent':15s} │ {'Easy Avg':>8s} │ {'Medium Avg':>10s} │ {'Hard Avg':>8s} │ "
          f"{'Esc Acc':>7s} │ {'Total Avg':>9s} │ {'Steps':>7s}")
    print(f"  {'─' * 15}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 8}─┼─"
          f"{'─' * 7}─┼─{'─' * 9}─┼─{'─' * 7}")
    for r in results:
        print(
            f"  {r['agent']:15s} │ {r['easy_avg']:8.3f} │ {r['medium_avg']:10.3f} │ "
            f"{r['hard_avg']:8.3f} │ {r['escalation_accuracy']:6.1f}% │ "
            f"{r['total_avg']:9.3f} │ {r['total_steps']:7,d}"
        )
    print(f"{'═' * 80}")

    # Determine winner
    best = max(results, key=lambda r: r["total_avg"])
    print(f"\n  🏆 Best overall: {best['agent']} (avg reward: {best['total_avg']:.3f})")

    best_esc = max(results, key=lambda r: r["escalation_accuracy"])
    print(f"  🎯 Best escalation: {best_esc['agent']} (accuracy: {best_esc['escalation_accuracy']:.1f}%)")

    best_hard = max(results, key=lambda r: r["hard_avg"])
    print(f"  💪 Best on hard: {best_hard['agent']} (hard avg: {best_hard['hard_avg']:.3f})")

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"comparison_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent", "easy_avg", "medium_avg", "hard_avg",
            "total_avg", "escalation_accuracy", "total_steps", "eval_time_s",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\n  📊 CSV saved to {csv_path}")

    # Save detailed JSON
    json_path = output_path / f"comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  📋 JSON saved to {json_path}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Trained Agents")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Specific agents to compare (default: all in models/)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per scenario per agent (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    compare_agents(
        models_dir=args.models_dir,
        agents=args.agents,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
