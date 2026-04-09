#!/usr/bin/env python3
"""Validate generated scenarios by running them through the environment.

Checks:
  1. Reward is in valid range [0.0, 1.0] for all 3 steps
  2. Escalation logic matches requires_escalation flag
  3. Low reward variance across repeated runs (< 0.2)
  4. All scenarios pass Pydantic validation

Usage:
    python tools/validate_scenarios.py
    python tools/validate_scenarios.py --input data/generated_scenarios.py
    python tools/validate_scenarios.py --runs 10  # More runs for stability check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import Scenario
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction


# ─── Scenario Loader ────────────────────────────────────────────────────────

def load_scenarios_from_file(path: str) -> List[Dict[str, Any]]:
    """Load GENERATED_SCENARIOS from a Python file."""
    spec = importlib.util.spec_from_file_location("_gen_scenarios", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "GENERATED_SCENARIOS", [])


# ─── Validation Checks ──────────────────────────────────────────────────────

CORRECT_CLASSIFY = {"refund": "refund", "complaint": "complaint", "query": "query"}

REPLY_TEMPLATES = [
    "Thank you for reaching out. We sincerely apologize for this experience. "
    "We understand your frustration and take this seriously. "
    "Our dedicated team will process your request within 24 hours.",
]


def run_scenario_episode(
    env: WorkplaceEnvironment,
    scenario: Dict[str, Any],
    scenario_idx: int,
) -> Dict[str, Any]:
    """Run one episode for a scenario and collect metrics."""
    env._state.scenario_index = scenario_idx
    env.reset()

    steps = []

    # Step 1: Classify correctly
    action1 = WorkplaceAction(action_type="classify", content=scenario["label"])
    obs1 = env.step(action1)
    steps.append({"action": "classify", "reward": obs1.reward or 0.0})

    # Step 2: Reply
    action2 = WorkplaceAction(action_type="reply", content=REPLY_TEMPLATES[0])
    obs2 = env.step(action2)
    steps.append({"action": "reply", "reward": obs2.reward or 0.0})

    # Step 3: Escalate (correctly based on requires_escalation)
    should_escalate = scenario.get("requires_escalation", False)
    action3 = WorkplaceAction(
        action_type="escalate",
        content="yes" if should_escalate else "no",
    )
    obs3 = env.step(action3)
    steps.append({"action": "escalate", "reward": obs3.reward or 0.0})

    total_reward = sum(s["reward"] for s in steps)
    return {
        "steps": steps,
        "total_reward": total_reward,
        "done": obs3.done,
    }


def validate_scenario(
    scenario: Dict[str, Any],
    scenario_idx: int,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """Run N episodes and validate reward consistency."""
    # Check 1: Pydantic validation
    try:
        Scenario(**scenario)
        pydantic_valid = True
        pydantic_error = None
    except Exception as exc:
        pydantic_valid = False
        pydantic_error = str(exc)

    if not pydantic_valid:
        return {
            "valid": False,
            "pydantic_valid": False,
            "pydantic_error": pydantic_error,
            "flags": ["PYDANTIC_FAIL"],
        }

    env = WorkplaceEnvironment()
    # Inject the scenario into the environment's scenario list
    original_scenarios = env._scenarios
    env._scenarios = original_scenarios + [scenario]
    injected_idx = len(original_scenarios)

    run_results = []
    flags: List[str] = []

    for run in range(n_runs):
        try:
            result = run_scenario_episode(env, scenario, injected_idx)
            run_results.append(result)
        except Exception as exc:
            flags.append(f"RUNTIME_ERROR_RUN{run}: {exc}")

    # Restore
    env._scenarios = original_scenarios

    if not run_results:
        return {"valid": False, "flags": flags, "pydantic_valid": True}

    # Check 2: Rewards in valid range [0.0, 1.0]
    all_step_rewards = []
    for result in run_results:
        for step in result["steps"]:
            r = step["reward"]
            all_step_rewards.append(r)
            if r < -0.1 or r > 1.1:
                flags.append(f"REWARD_OUT_OF_RANGE: {step['action']}={r:.3f}")

    # Check 3: Reward variance < 0.2 across runs
    total_rewards = [r["total_reward"] for r in run_results]
    variance = float(np.var(total_rewards))
    if variance > 0.2:
        flags.append(f"HIGH_VARIANCE: var={variance:.4f}")

    # Check 4: Escalation logic
    for result in run_results:
        esc_step = result["steps"][2]
        esc_reward = esc_step["reward"]
        requires_esc = scenario.get("requires_escalation", False)

        # If correct escalation decision gives very low reward, logic mismatch
        if requires_esc and esc_reward < 0.05:
            flags.append(f"ESCALATION_MISMATCH: requires_escalation=True but reward={esc_reward:.3f}")
            break

    mean_reward = float(np.mean(total_rewards))
    is_valid = len([f for f in flags if not f.startswith("HIGH_VARIANCE")]) == 0

    return {
        "valid": is_valid,
        "pydantic_valid": True,
        "mean_reward": round(mean_reward, 4),
        "reward_variance": round(variance, 6),
        "reward_range": [round(min(all_step_rewards), 4), round(max(all_step_rewards), 4)],
        "n_runs": len(run_results),
        "flags": flags,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def validate_all(
    input_path: str = "data/generated_scenarios.py",
    n_runs: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """Validate all generated scenarios. Returns (valid, invalid) lists."""
    scenarios = load_scenarios_from_file(input_path)
    print(f"\n🔍 Validating {len(scenarios)} generated scenarios ({n_runs} runs each)...\n")

    valid_scenarios: List[Dict] = []
    invalid_scenarios: List[Dict] = []
    flagged: List[Dict] = []

    for idx, scenario in enumerate(scenarios):
        result = validate_scenario(scenario, idx, n_runs=n_runs)
        entry = {**scenario, "_validation": result}

        if result["valid"]:
            valid_scenarios.append(entry)
            status = "✅"
        else:
            invalid_scenarios.append(entry)
            status = "❌"

        # Progress every 10 scenarios
        if (idx + 1) % 10 == 0 or not result["valid"]:
            diff = scenario["difficulty"]
            label = scenario["label"]
            flags_str = "; ".join(result.get("flags", []))
            if flags_str:
                status += f" [{flags_str}]"
            print(f"  {status} [{idx+1:3d}/{len(scenarios)}] {diff:6s} {label:10s} "
                  f"reward={result.get('mean_reward', 0):.3f} var={result.get('reward_variance', 0):.5f}")
        elif result.get("flags"):
            flagged.append(entry)

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  VALIDATION RESULTS")
    print(f"  Valid:   {len(valid_scenarios):3d} / {len(scenarios)}")
    print(f"  Invalid: {len(invalid_scenarios):3d} / {len(scenarios)}")
    print(f"  Flagged: {len(flagged):3d} (valid but with warnings)")
    print(f"{'═' * 60}")

    if invalid_scenarios:
        print(f"\n  ❌ Invalid scenarios:")
        for s in invalid_scenarios:
            v = s["_validation"]
            print(f"     [{s['difficulty']:6s}] {s['email'][:60]}...")
            for flag in v.get("flags", []):
                print(f"       ⚠️  {flag}")

    return valid_scenarios, invalid_scenarios


def main():
    parser = argparse.ArgumentParser(description="Validate generated scenarios")
    parser.add_argument("--input", type=str, default="data/generated_scenarios.py")
    parser.add_argument("--runs", type=int, default=5, help="Episodes per scenario")
    parser.add_argument("--output", type=str, default=None, help="Save validated JSON")
    args = parser.parse_args()

    valid, invalid = validate_all(args.input, n_runs=args.runs)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "valid_scenarios": [{k: v for k, v in s.items() if k != "_validation"} for s in valid],
                "invalid_scenarios": [{k: v for k, v in s.items() if k != "_validation"} for s in invalid],
            }, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
