"""Failure analysis utilities over RewardMemory episode traces.

This module transforms episodic reward data into a structured failure report
for downstream strategy generation systems.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from core.memory.reward_memory import EpisodeRecord, RewardMemory


class FailureAnalyzer:
    """Analyze `RewardMemory` and return structured failure insights."""

    def analyze(self, memory: RewardMemory) -> Dict[str, Any]:
        records = list(memory.records)

        classify_failures = [r for r in records if r.classify_reward < 0.55]
        reply_failures = [r for r in records if r.reply_reward < 0.35]
        escalate_failures = [r for r in records if r.escalate_reward < 0.5]

        return {
            "classify_failures": self._analyze_classify_failures(classify_failures),
            "reply_failures": self._analyze_reply_failures(reply_failures),
            "escalate_failures": self._analyze_escalate_failures(escalate_failures),
            "difficulty_breakdown": self._analyze_difficulty_breakdown(records),
        }

    # ── helpers for building example dicts ──────────────────────────────────

    @staticmethod
    def _safe_breakdown(bd: Any) -> Dict[str, Any]:
        """Return a JSON-safe copy of a breakdown dict."""
        if isinstance(bd, dict):
            return dict(bd)
        return {}

    @staticmethod
    def _correct_category(record: EpisodeRecord) -> str:
        """Extract the correct category from classify_breakdown."""
        bd = record.classify_breakdown or {}
        # The grader stores the ground-truth label in 'category'
        return str(bd.get("category", "unknown"))

    @staticmethod
    def _reply_weakness(record: EpisodeRecord) -> List[str]:
        """Identify the weakest reply subfields from breakdown details."""
        bd = record.reply_breakdown or {}
        details = (
            bd.get("evaluation", {})
            .get("breakdown", {})
            .get("rule_based", {})
            .get("details", {})
        )
        if not details:
            return ["unknown"]

        scored = {
            k: float(v) for k, v in details.items()
            if isinstance(v, (int, float))
        }
        if not scored:
            return ["unknown"]

        # Return subfields sorted ascending (weakest first)
        return [k for k, _ in sorted(scored.items(), key=lambda x: x[1])]

    @staticmethod
    def _escalation_direction(record: EpisodeRecord) -> str:
        """Determine whether failure was over- or under-escalation."""
        details = (
            (record.escalate_breakdown or {})
            .get("evaluation", {})
            .get("breakdown", {})
            .get("rule_based", {})
            .get("details", {})
        )
        did = details.get("did_escalate")
        should = details.get("should_escalate")

        if did is True and should is False:
            return "over_escalation"
        elif did is False and should is True:
            return "under_escalation"
        elif did is not None and should is not None:
            return "correct" if did == should else "mismatch"

        # Fallback heuristic from action string
        action = (record.escalate_action or "").strip().lower()
        if action == "yes":
            return "over_escalation"
        elif action == "no":
            return "under_escalation"
        return "unknown"

    # ── per-step analyzers ────────────────────────────────────────────────

    def _analyze_classify_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        grouped = Counter(record.classify_action for record in failures)

        # Pick 3 records where classify_reward is lowest AND action != correct
        misclassified = [
            r for r in failures
            if r.classify_action != self._correct_category(r)
        ]
        misclassified.sort(key=lambda r: r.classify_reward)
        worst = misclassified[:3]

        # If fewer than 3 misclassified, fill from lowest-reward failures
        if len(worst) < 3:
            seen = {id(r) for r in worst}
            remaining = sorted(failures, key=lambda r: r.classify_reward)
            for r in remaining:
                if id(r) not in seen:
                    worst.append(r)
                    seen.add(id(r))
                if len(worst) >= 3:
                    break

        examples = [
            {
                "email_snippet": r.email_snippet,
                "agent_action": r.classify_action,
                "correct_action": self._correct_category(r),
                "reward_received": round(r.classify_reward, 4),
                "breakdown": self._safe_breakdown(r.classify_breakdown),
            }
            for r in worst
        ]

        return {
            "total_failures": len(failures),
            "grouped_by_prediction": dict(grouped),
            "examples": examples,
        }

    def _analyze_reply_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        too_short_count = 0
        missing_keyword_count = 0
        dominant_counter: Counter[str] = Counter()

        for record in failures:
            bd = record.reply_breakdown or {}
            # Check rule_based details for component scores
            details = (
                bd.get("evaluation", {})
                .get("breakdown", {})
                .get("rule_based", {})
                .get("details", {})
            )
            length_score = self._to_float(details.get("length_component", 0.0))
            keyword_score = self._to_float(details.get("keyword_component", 0.0))

            if length_score < 0.2:
                too_short_count += 1
            if keyword_score < 0.2:
                missing_keyword_count += 1

            dominant_counter[record.classify_action] += 1

        dominant_category = None
        if dominant_counter:
            dominant_category = dominant_counter.most_common(1)[0][0]

        # Pick 3 records with lowest reply_reward
        sorted_failures = sorted(failures, key=lambda r: r.reply_reward)
        worst = sorted_failures[:3]

        examples = [
            {
                "email_snippet": r.email_snippet,
                "agent_action": (r.reply_action or "")[:120],
                "correct_action": "adequate reply meeting length/keyword/solution criteria",
                "reward_received": round(r.reply_reward, 4),
                "weakest_subfields": self._reply_weakness(r),
                "breakdown": self._safe_breakdown(r.reply_breakdown),
            }
            for r in worst
        ]

        return {
            "failure_count": len(failures),
            "too_short_count": too_short_count,
            "missing_keyword_count": missing_keyword_count,
            "dominant_category": dominant_category,
            "examples": examples,
        }

    def _analyze_escalate_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        over_escalation_count = 0
        under_escalation_count = 0

        for record in failures:
            direction = self._escalation_direction(record)
            if direction == "over_escalation":
                over_escalation_count += 1
            elif direction == "under_escalation":
                under_escalation_count += 1

        # Pick 3 records with lowest escalate_reward
        sorted_failures = sorted(failures, key=lambda r: r.escalate_reward)
        worst = sorted_failures[:3]

        examples = [
            {
                "email_snippet": r.email_snippet,
                "agent_action": r.escalate_action,
                "correct_action": (
                    "escalate" if self._escalation_direction(r) == "under_escalation"
                    else "do not escalate" if self._escalation_direction(r) == "over_escalation"
                    else "unknown"
                ),
                "direction": self._escalation_direction(r),
                "reward_received": round(r.escalate_reward, 4),
                "breakdown": self._safe_breakdown(r.escalate_breakdown),
            }
            for r in worst
        ]

        return {
            "over_escalation_count": over_escalation_count,
            "under_escalation_count": under_escalation_count,
            "examples": examples,
        }

    def _analyze_difficulty_breakdown(self, records: List[EpisodeRecord]) -> Dict[str, Any]:
        levels = ("easy", "medium", "hard")
        grouped: Dict[str, List[EpisodeRecord]] = {level: [] for level in levels}

        for record in records:
            difficulty = (record.difficulty or "").strip().lower()
            if difficulty in grouped:
                grouped[difficulty].append(record)

        result: Dict[str, Any] = {}
        for level in levels:
            bucket = grouped[level]
            if not bucket:
                result[level] = {
                    "classify_reward": 0.0,
                    "reply_reward": 0.0,
                    "escalate_reward": 0.0,
                }
                continue

            count = len(bucket)
            result[level] = {
                "classify_reward": sum(r.classify_reward for r in bucket) / count,
                "reply_reward": sum(r.reply_reward for r in bucket) / count,
                "escalate_reward": sum(r.escalate_reward for r in bucket) / count,
            }

        return result

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["FailureAnalyzer"]
