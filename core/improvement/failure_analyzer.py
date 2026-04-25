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

    def _analyze_classify_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        grouped = Counter(record.classify_action for record in failures)
        return {
            "total_failures": len(failures),
            "grouped_by_prediction": dict(grouped),
            "examples": [record.email_snippet for record in failures[:3]],
        }

    def _analyze_reply_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        too_short_count = 0
        missing_keyword_count = 0
        dominant_counter: Counter[str] = Counter()

        for record in failures:
            breakdown = record.reply_breakdown or {}
            length_score = self._to_float(breakdown.get("length_score", 0.0))
            keyword_score = self._to_float(breakdown.get("keyword_score", 0.0))

            if length_score < 0.2:
                too_short_count += 1
            if keyword_score < 0.2:
                missing_keyword_count += 1

            dominant_counter[record.classify_action] += 1

        dominant_category = None
        if dominant_counter:
            dominant_category = dominant_counter.most_common(1)[0][0]

        return {
            "failure_count": len(failures),
            "too_short_count": too_short_count,
            "missing_keyword_count": missing_keyword_count,
            "dominant_category": dominant_category,
        }

    def _analyze_escalate_failures(self, failures: List[EpisodeRecord]) -> Dict[str, Any]:
        over_escalation_count = 0
        under_escalation_count = 0

        for record in failures:
            action = (record.escalate_action or "").strip().lower()
            if action == "yes":
                over_escalation_count += 1
            elif action == "no":
                under_escalation_count += 1

        return {
            "over_escalation_count": over_escalation_count,
            "under_escalation_count": under_escalation_count,
            "examples": [record.email_snippet for record in failures[:3]],
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
