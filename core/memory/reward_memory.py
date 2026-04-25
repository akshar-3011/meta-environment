"""Episode-level reward memory for triage workflow analysis.

This module stores compact per-episode reward traces with per-step details,
provides simple failure slicing by step reward thresholds, summarizes mean
rewards overall and by difficulty, and supports JSON persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List


@dataclass
class EpisodeRecord:
    """Structured reward trace for one 3-step episode."""

    episode_id: int
    scenario_id: str
    difficulty: str
    sentiment: str
    urgency: str
    email_snippet: str

    classify_action: str
    classify_reward: float
    classify_breakdown: Dict[str, Any]

    reply_action: str
    reply_reward: float
    reply_breakdown: Dict[str, Any]

    escalate_action: str
    escalate_reward: float
    escalate_breakdown: Dict[str, Any]

    def __post_init__(self) -> None:
        # Keep snippets bounded to requested shape.
        self.email_snippet = (self.email_snippet or "")[:120]

        # Ensure rewards are numeric floats.
        self.classify_reward = float(self.classify_reward)
        self.reply_reward = float(self.reply_reward)
        self.escalate_reward = float(self.escalate_reward)

    @property
    def total_reward(self) -> float:
        """Return the episode-level total reward across all 3 steps."""
        return self.classify_reward + self.reply_reward + self.escalate_reward

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record into a JSON-safe dictionary."""
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "difficulty": self.difficulty,
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "email_snippet": self.email_snippet,
            "classify_action": self.classify_action,
            "classify_reward": self.classify_reward,
            "classify_breakdown": self.classify_breakdown,
            "reply_action": self.reply_action,
            "reply_reward": self.reply_reward,
            "reply_breakdown": self.reply_breakdown,
            "escalate_action": self.escalate_action,
            "escalate_reward": self.escalate_reward,
            "escalate_breakdown": self.escalate_breakdown,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeRecord":
        """Build a record from a dictionary (e.g., loaded JSON)."""
        return cls(
            episode_id=int(data["episode_id"]),
            scenario_id=str(data["scenario_id"]),
            difficulty=str(data["difficulty"]),
            sentiment=str(data["sentiment"]),
            urgency=str(data["urgency"]),
            email_snippet=str(data.get("email_snippet", "")),
            classify_action=str(data.get("classify_action", "")),
            classify_reward=float(data.get("classify_reward", 0.0)),
            classify_breakdown=dict(data.get("classify_breakdown", {})),
            reply_action=str(data.get("reply_action", "")),
            reply_reward=float(data.get("reply_reward", 0.0)),
            reply_breakdown=dict(data.get("reply_breakdown", {})),
            escalate_action=str(data.get("escalate_action", "")),
            escalate_reward=float(data.get("escalate_reward", 0.0)),
            escalate_breakdown=dict(data.get("escalate_breakdown", {})),
        )


@dataclass
class RewardMemory:
    """In-memory store for episode reward traces with JSON persistence."""

    records: List[EpisodeRecord] = field(default_factory=list)

    def add(self, record: EpisodeRecord) -> None:
        """Add one episode record to memory."""
        self.records.append(record)

    def get_step_failures(self, step_name: str, threshold: float) -> List[EpisodeRecord]:
        """Return records where a specific step reward is below ``threshold``.

        Valid step names: ``classify``, ``reply``, ``escalate``.
        """
        key_map = {
            "classify": "classify_reward",
            "reply": "reply_reward",
            "escalate": "escalate_reward",
        }

        if step_name not in key_map:
            valid = ", ".join(sorted(key_map.keys()))
            raise ValueError(f"Invalid step_name '{step_name}'. Expected one of: {valid}")

        reward_key = key_map[step_name]
        threshold_value = float(threshold)
        return [r for r in self.records if float(getattr(r, reward_key)) < threshold_value]

    def summary(self) -> Dict[str, Any]:
        """Return mean reward overall and per difficulty level.

        Overall mean is computed over episode total rewards
        (classify + reply + escalate), and per-difficulty means use the same
        episode-total basis.
        """
        if not self.records:
            return {
                "count": 0,
                "mean_reward_overall": 0.0,
                "mean_reward_per_difficulty": {},
            }

        totals = [r.total_reward for r in self.records]
        mean_overall = sum(totals) / len(totals)

        grouped: Dict[str, List[float]] = {}
        for record in self.records:
            grouped.setdefault(record.difficulty, []).append(record.total_reward)

        mean_per_difficulty = {
            difficulty: (sum(values) / len(values))
            for difficulty, values in grouped.items()
        }

        return {
            "count": len(self.records),
            "mean_reward_overall": mean_overall,
            "mean_reward_per_difficulty": mean_per_difficulty,
        }

    def save(self, path: str) -> None:
        """Save records to JSON at ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "records": [record.to_dict() for record in self.records],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "RewardMemory":
        """Load memory from JSON file at ``path``."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        records = [EpisodeRecord.from_dict(item) for item in data.get("records", [])]
        return cls(records=records)


__all__ = ["EpisodeRecord", "RewardMemory"]
