"""Presentation-only demo script.

Reads previously generated artifacts and prints a clean
before/optimizing/after story. No heavy computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.improvement.failure_analyzer import FailureAnalyzer
from core.memory.reward_memory import EpisodeRecord, RewardMemory


BASELINE_PATH = "baseline_memory.json"
IMPROVED_PATH = "improved_memory.json"
STRATEGY_PATH = "final_strategy.json"


def _load_memory(path: str) -> RewardMemory:
    try:
        if not Path(path).exists():
            return RewardMemory()
        return RewardMemory.load(path)
    except Exception:
        return RewardMemory()


def _load_strategy(path: str) -> Dict[str, Any]:
    try:
        if not Path(path).exists():
            return {}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _means(memory: RewardMemory) -> Dict[str, float]:
    records = list(memory.records)
    return {
        "total": _mean([r.total_reward for r in records]),
        "classify": _mean([r.classify_reward for r in records]),
        "reply": _mean([r.reply_reward for r in records]),
        "escalate": _mean([r.escalate_reward for r in records]),
    }


def _truncate(text: str, n: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= n:
        return clean
    return clean[: max(0, n - 3)] + "..."


def _worst_episodes(memory: RewardMemory, k: int = 3) -> List[EpisodeRecord]:
    return sorted(memory.records, key=lambda r: r.total_reward)[: max(0, k)]


def _dominant_mistake(classify_block: Dict[str, Any]) -> str:
    grouped = classify_block.get("grouped_by_prediction", {})
    if not isinstance(grouped, dict) or not grouped:
        return "insufficient data"

    # Keep only numeric-ish counts.
    pairs: List[Tuple[str, int]] = []
    for key, value in grouped.items():
        try:
            pairs.append((str(key), int(value)))
        except Exception:
            continue

    if not pairs:
        return "insufficient data"

    category, _count = max(pairs, key=lambda x: x[1])
    return f"over-predicting '{category}' on low-reward episodes"


def _print_before_panel(memory: RewardMemory) -> None:
    m = _means(memory)
    print("=== BEFORE (BASELINE PERFORMANCE) ===")
    print()
    print(f"Total Episodes: {len(memory.records)}")
    print(f"Mean Total Reward: {m['total']:.2f}")
    print()
    print(f"Classify: {m['classify']:.2f}")
    print(f"Reply: {m['reply']:.2f}")
    print(f"Escalate: {m['escalate']:.2f}")
    print()
    print("Worst 3 episodes:")
    worst = _worst_episodes(memory, 3)
    if not worst:
        print("- No episodes available")
    else:
        for item in worst:
            print(f"- {_truncate(item.email_snippet, 80)} | total={item.total_reward:.2f}")
    print()


def _print_optimizing_panel(baseline: RewardMemory, strategy: Dict[str, Any]) -> None:
    analyzer = FailureAnalyzer()
    analysis = analyzer.analyze(baseline)
    total_episodes = max(1, len(baseline.records))

    classify_block = analysis.get("classify_failures", {}) if isinstance(analysis, dict) else {}
    reply_block = analysis.get("reply_failures", {}) if isinstance(analysis, dict) else {}
    escalate_block = analysis.get("escalate_failures", {}) if isinstance(analysis, dict) else {}

    classify_failures = int(classify_block.get("total_failures", 0) or 0)
    dominant = _dominant_mistake(classify_block if isinstance(classify_block, dict) else {})

    too_short = int(reply_block.get("too_short_count", 0) or 0) if isinstance(reply_block, dict) else 0
    missing_keywords = int(reply_block.get("missing_keyword_count", 0) or 0) if isinstance(reply_block, dict) else 0

    over_esc = int(escalate_block.get("over_escalation_count", 0) or 0) if isinstance(escalate_block, dict) else 0
    under_esc = int(escalate_block.get("under_escalation_count", 0) or 0) if isinstance(escalate_block, dict) else 0

    reasoning = ""
    if isinstance(strategy, dict):
        reasoning = str(strategy.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = "No strategy reasoning available."

    print("=== OPTIMIZING... ===")
    print()
    print("Classify:")
    print(f"- failures: {classify_failures}/{total_episodes} episodes")
    print(f"- dominant mistake: {dominant}")
    print()
    print("Reply:")
    print(f"- too short: {too_short}")
    print(f"- missing keywords: {missing_keywords}")
    print()
    print("Escalate:")
    print(f"- over-escalation: {over_esc}")
    print(f"- under-escalation: {under_esc}")
    print()
    print("Strategy Update:")
    print(f'"{reasoning}"')
    print()


def _status(change: float) -> str:
    return "✅ IMPROVED" if change >= 0 else "❌ REGRESSED"


def _print_after_and_delta(before: RewardMemory, after: RewardMemory) -> None:
    b = _means(before)
    a = _means(after)

    print("=== AFTER (IMPROVED PERFORMANCE) ===")
    print()
    print(f"Total Episodes: {len(after.records)}")
    print(f"Mean Total Reward: {a['total']:.2f}")
    print()
    print(f"Classify: {a['classify']:.2f}")
    print(f"Reply: {a['reply']:.2f}")
    print(f"Escalate: {a['escalate']:.2f}")
    print()

    print("=== DELTA ===")
    print()
    print("Step        Before   After   Change   Status")
    print("-------------------------------------------")

    rows = [
        ("Total", b["total"], a["total"]),
        ("Classify", b["classify"], a["classify"]),
        ("Reply", b["reply"], a["reply"]),
        ("Escalate", b["escalate"], a["escalate"]),
    ]

    for name, bv, av in rows:
        delta = av - bv
        print(f"{name:<10}  {bv:>6.2f}   {av:>5.2f}   {delta:+6.2f}   {_status(delta)}")

    total_before = b["total"]
    total_after = a["total"]
    if total_before > 0:
        improvement_pct = ((total_after - total_before) / total_before) * 100.0
    else:
        improvement_pct = 0.0

    step_names = ["Classify", "Reply", "Escalate"]
    step_before = [b["classify"], b["reply"], b["escalate"]]
    step_after = [a["classify"], a["reply"], a["escalate"]]

    best_idx = 0
    best_gain_pct = -10**9
    for idx, (sb, sa) in enumerate(zip(step_before, step_after)):
        if sb > 0:
            pct = ((sa - sb) / sb) * 100.0
        else:
            pct = 0.0
        if pct > best_gain_pct:
            best_gain_pct = pct
            best_idx = idx

    best_gain_display = best_gain_pct if best_gain_pct > 0 else 0.0

    print()
    print(
        f"Improvement: {improvement_pct:.2f}% overall | "
        f"Best gain: {step_names[best_idx]} {best_gain_display:+.2f}% | "
        "Strategy version: 1"
    )


def main() -> None:
    baseline = _load_memory(BASELINE_PATH)
    improved = _load_memory(IMPROVED_PATH)
    strategy = _load_strategy(STRATEGY_PATH)

    _print_before_panel(baseline)
    _print_optimizing_panel(baseline, strategy)
    _print_after_and_delta(baseline, improved)


if __name__ == "__main__":
    main()
