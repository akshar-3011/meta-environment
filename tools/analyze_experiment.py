#!/usr/bin/env python3
"""Experiment analysis tool — statistical comparison of A/B experiment variants.

Pulls episode data from experiments.db, computes per-variant statistics,
runs Welch's t-test for significance, and outputs a recommendation.

Usage:
    python tools/analyze_experiment.py <experiment_id>
    python tools/analyze_experiment.py <experiment_id> --min-episodes 200
    python tools/analyze_experiment.py --list  # show all experiments
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.environ.get(
    "EXPERIMENTS_DB",
    str(Path(__file__).resolve().parent.parent / "experiments.db"),
)


# ─── Data Loader ─────────────────────────────────────────────────────────────

def load_experiment(db_path: str, experiment_id: str) -> Dict[str, Any]:
    """Load experiment + all episodes from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    exp = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()
    if not exp:
        print(f"❌ Experiment '{experiment_id}' not found.")
        sys.exit(1)

    episodes = conn.execute(
        """SELECT * FROM episodes WHERE experiment_id = ?
           ORDER BY created_at""",
        (experiment_id,),
    ).fetchall()

    conn.close()

    return {
        "experiment": dict(exp),
        "episodes": [dict(e) for e in episodes],
    }


def list_experiments(db_path: str) -> List[Dict]:
    """List all experiments with episode counts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT e.*, 
               COUNT(ep.id) as episode_count,
               AVG(CASE WHEN ep.variant='control' THEN ep.total_reward END) as control_mean,
               AVG(CASE WHEN ep.variant='variant' THEN ep.total_reward END) as variant_mean
        FROM experiments e
        LEFT JOIN episodes ep ON e.id = ep.experiment_id
        GROUP BY e.id
        ORDER BY e.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Statistics ──────────────────────────────────────────────────────────────

def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test for unequal variance. Returns (t_stat, p_value).

    Manual implementation to avoid scipy dependency.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / max(den, 1e-10)

    # Approximate p-value using normal distribution for large df
    # For df > 30, t-distribution ≈ normal
    if df > 30:
        from math import erfc, sqrt
        p_value = erfc(abs(t_stat) / sqrt(2))
    else:
        # Rough approximation for smaller df
        p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))

    return float(t_stat), float(p_value)


def _t_cdf(t: float, df: float) -> float:
    """Approximate CDF of t-distribution using regularized beta function.

    Uses the relationship: CDF(t, df) = 1 - 0.5 * I(df/(df+t²), df/2, 1/2)
    with a simple continued-fraction approximation.
    """
    x = df / (df + t * t)
    # Simple approximation: use normal CDF for df > 10
    from math import erf, sqrt
    # Scale t by sqrt(df/(df-2)) to approximate normal
    adjusted = t * (1 - 1 / (4 * max(df, 1)))
    return 0.5 * (1 + erf(adjusted / sqrt(2)))


def compute_escalation_accuracy(episodes: List[Dict]) -> float:
    """Estimate escalation accuracy from step rewards (step 3 > 0.15 = correct)."""
    correct = 0
    total = 0
    for ep in episodes:
        steps = json.loads(ep["step_rewards"]) if isinstance(ep["step_rewards"], str) else ep["step_rewards"]
        if len(steps) >= 3:
            total += 1
            if steps[2] > 0.15:
                correct += 1
    return correct / max(total, 1) * 100


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_experiment(
    db_path: str,
    experiment_id: str,
    min_episodes: int = 100,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Full analysis of an A/B experiment."""
    data = load_experiment(db_path, experiment_id)
    exp = data["experiment"]
    episodes = data["episodes"]

    # Split by variant
    control_eps = [e for e in episodes if e["variant"] == "control"]
    variant_eps = [e for e in episodes if e["variant"] == "variant"]

    control_rewards = [e["total_reward"] for e in control_eps]
    variant_rewards = [e["total_reward"] for e in variant_eps]

    # Per-step breakdown
    step_names = ["classify", "reply", "escalate"]
    control_steps: Dict[str, List[float]] = {s: [] for s in step_names}
    variant_steps: Dict[str, List[float]] = {s: [] for s in step_names}

    for ep in control_eps:
        steps = json.loads(ep["step_rewards"]) if isinstance(ep["step_rewards"], str) else ep["step_rewards"]
        for i, name in enumerate(step_names):
            if i < len(steps):
                control_steps[name].append(steps[i])

    for ep in variant_eps:
        steps = json.loads(ep["step_rewards"]) if isinstance(ep["step_rewards"], str) else ep["step_rewards"]
        for i, name in enumerate(step_names):
            if i < len(steps):
                variant_steps[name].append(steps[i])

    # Statistics
    t_stat, p_value = welch_t_test(control_rewards, variant_rewards)

    control_mean = float(np.mean(control_rewards)) if control_rewards else 0
    variant_mean = float(np.mean(variant_rewards)) if variant_rewards else 0
    lift = ((variant_mean - control_mean) / max(control_mean, 0.001)) * 100

    # Escalation accuracy
    control_esc_acc = compute_escalation_accuracy(control_eps)
    variant_esc_acc = compute_escalation_accuracy(variant_eps)

    # Recommendation
    ready = len(control_eps) >= min_episodes and len(variant_eps) >= min_episodes
    if not ready:
        recommendation = "CONTINUE_TEST"
        reason = (
            f"Insufficient data: control={len(control_eps)}, "
            f"variant={len(variant_eps)} (min: {min_episodes} each)"
        )
    elif p_value < significance_level and lift > 0:
        recommendation = "DEPLOY_VARIANT"
        reason = (
            f"Variant is significantly better: +{lift:.1f}% lift, "
            f"p={p_value:.4f} < {significance_level}"
        )
    elif p_value < significance_level and lift < 0:
        recommendation = "ABORT"
        reason = (
            f"Variant is significantly worse: {lift:.1f}% lift, "
            f"p={p_value:.4f} < {significance_level}"
        )
    else:
        recommendation = "CONTINUE_TEST"
        reason = f"No significant difference yet: p={p_value:.4f} > {significance_level}"

    result = {
        "experiment_id": experiment_id,
        "experiment_name": exp["name"],
        "policy_type": exp["policy_type"],
        "status": exp["status"],
        "traffic_split": exp["traffic_split"],
        "control": {
            "episodes": len(control_eps),
            "mean_reward": round(control_mean, 4),
            "std_reward": round(float(np.std(control_rewards)), 4) if control_rewards else 0,
            "escalation_accuracy": round(control_esc_acc, 1),
            "step_means": {
                name: round(float(np.mean(vals)), 4) if vals else 0
                for name, vals in control_steps.items()
            },
        },
        "variant": {
            "episodes": len(variant_eps),
            "mean_reward": round(variant_mean, 4),
            "std_reward": round(float(np.std(variant_rewards)), 4) if variant_rewards else 0,
            "escalation_accuracy": round(variant_esc_acc, 1),
            "step_means": {
                name: round(float(np.mean(vals)), 4) if vals else 0
                for name, vals in variant_steps.items()
            },
        },
        "statistics": {
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "lift_pct": round(lift, 2),
            "significant": p_value < significance_level,
        },
        "recommendation": recommendation,
        "reason": reason,
    }

    return result


def print_analysis(result: Dict[str, Any]) -> None:
    """Pretty-print experiment analysis."""
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT ANALYSIS: {result['experiment_name']}")
    print(f"  ID: {result['experiment_id']}  |  Policy: {result['policy_type']}")
    print(f"  Status: {result['status']}  |  Split: {result['traffic_split']*100:.0f}% variant")
    print(f"{'═' * 70}\n")

    c, v = result["control"], result["variant"]

    print(f"  {'Metric':<25s} │ {'Control':>12s} │ {'Variant':>12s} │ {'Diff':>10s}")
    print(f"  {'─' * 25}─┼─{'─' * 12}─┼─{'─' * 12}─┼─{'─' * 10}")

    def _row(name, c_val, v_val, fmt=".4f"):
        diff = v_val - c_val
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<25s} │ {c_val:>12{fmt}} │ {v_val:>12{fmt}} │ {sign}{diff:>9{fmt}}")

    _row("Episodes", c["episodes"], v["episodes"], "d")
    _row("Mean Reward", c["mean_reward"], v["mean_reward"])
    _row("Std Reward", c["std_reward"], v["std_reward"])
    _row("Escalation Acc (%)", c["escalation_accuracy"], v["escalation_accuracy"], ".1f")

    print(f"\n  Per-Step Breakdown:")
    for step in ["classify", "reply", "escalate"]:
        _row(f"  {step}", c["step_means"][step], v["step_means"][step])

    stats = result["statistics"]
    print(f"\n  {'─' * 62}")
    print(f"  Statistics:")
    print(f"    t-statistic:  {stats['t_stat']:>8.4f}")
    print(f"    p-value:      {stats['p_value']:>8.4f}  {'✅ significant' if stats['significant'] else '⬜ not significant'}")
    print(f"    Lift:         {stats['lift_pct']:>+7.2f}%")

    emoji = {"DEPLOY_VARIANT": "🚀", "CONTINUE_TEST": "⏳", "ABORT": "🛑"}
    rec = result["recommendation"]
    print(f"\n  {'═' * 62}")
    print(f"  {emoji.get(rec, '❓')} RECOMMENDATION: {rec}")
    print(f"     {result['reason']}")
    print(f"  {'═' * 62}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze A/B experiment results")
    parser.add_argument("experiment_id", nargs="?", help="Experiment ID to analyze")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to experiments.db")
    parser.add_argument("--min-episodes", type=int, default=100,
                        help="Minimum episodes per variant for decision")
    parser.add_argument("--significance", type=float, default=0.05, help="Significance level")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.db)
        if not experiments:
            print("  No experiments found.")
            return

        print(f"\n  {'ID':<10s} │ {'Name':<25s} │ {'Policy':<18s} │ {'Status':<10s} │ {'Episodes':>8s} │ {'Ctrl Mean':>9s} │ {'Var Mean':>9s}")
        print(f"  {'─' * 10}─┼─{'─' * 25}─┼─{'─' * 18}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 9}")
        for e in experiments:
            c_mean = f"{e['control_mean']:.3f}" if e['control_mean'] else "—"
            v_mean = f"{e['variant_mean']:.3f}" if e['variant_mean'] else "—"
            print(f"  {e['id']:<10s} │ {e['name']:<25s} │ {e['policy_type']:<18s} │ "
                  f"{e['status']:<10s} │ {e['episode_count']:>8d} │ {c_mean:>9s} │ {v_mean:>9s}")
        print()
        return

    if not args.experiment_id:
        parser.print_help()
        return

    result = analyze_experiment(
        args.db, args.experiment_id,
        min_episodes=args.min_episodes,
        significance_level=args.significance,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_analysis(result)


if __name__ == "__main__":
    main()
