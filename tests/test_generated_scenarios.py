"""Tests for the generated scenario pipeline.

Verifies:
  1. All scenarios pass Pydantic Scenario model validation
  2. No semantic duplicates (pairwise Jaccard similarity < 0.85)
  3. Difficulty tier balance (within 10% of target)
  4. Label distribution is reasonably balanced
  5. Escalation flags are logically consistent
  6. All scenarios run through the environment without errors
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Set

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import Scenario
from data.scenario_repository import SCENARIOS
from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase words (3+ chars)."""
    import re
    return set(
        t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(t) > 2
    )


def jaccard_similarity(a: str, b: str) -> float:
    a_set = _tokenize(a)
    b_set = _tokenize(b)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestScenarioValidation:
    """All scenarios must pass Pydantic validation."""

    @pytest.mark.parametrize("idx", range(len(SCENARIOS)))
    def test_scenario_pydantic_valid(self, idx):
        """Each scenario individually validates against the Scenario model."""
        scenario = SCENARIOS[idx]
        validated = Scenario(**scenario)
        assert validated.email
        assert validated.label in ("refund", "complaint", "query")
        assert validated.difficulty in ("easy", "medium", "hard")
        assert validated.sentiment in ("negative", "neutral", "positive", "mixed")
        assert validated.urgency in ("low", "medium", "high")
        assert 1 <= validated.complexity <= 5
        assert isinstance(validated.requires_escalation, bool)
        assert validated.min_reply_length >= 10


class TestSemanticDuplicates:
    """No two scenarios should be semantically identical (Jaccard > 0.85)."""

    def test_no_near_duplicates(self):
        """Pairwise Jaccard similarity between all scenario emails < 0.85."""
        emails = [s["email"] for s in SCENARIOS]
        duplicates = []

        for i in range(len(emails)):
            for j in range(i + 1, len(emails)):
                sim = jaccard_similarity(emails[i], emails[j])
                if sim >= 0.85:
                    duplicates.append((i, j, sim))

        if duplicates:
            msg_parts = [
                f"  [{i}] vs [{j}]: similarity={sim:.3f}\n"
                f"    A: {SCENARIOS[i]['email'][:80]}...\n"
                f"    B: {SCENARIOS[j]['email'][:80]}..."
                for i, j, sim in duplicates[:5]  # Show max 5
            ]
            pytest.fail(
                f"Found {len(duplicates)} near-duplicate pairs (threshold=0.85):\n"
                + "\n".join(msg_parts)
            )


class TestDifficultyBalance:
    """Difficulty tiers should be reasonably balanced."""

    def test_all_difficulties_present(self):
        """All three difficulty levels must be represented."""
        difficulties = {s["difficulty"] for s in SCENARIOS}
        assert difficulties == {"easy", "medium", "hard"}

    def test_minimum_per_difficulty(self):
        """Each difficulty should have at least 5 scenarios."""
        counts = Counter(s["difficulty"] for s in SCENARIOS)
        for diff in ["easy", "medium", "hard"]:
            assert counts[diff] >= 5, f"{diff} has only {counts[diff]} scenarios (min: 5)"

    def test_no_tier_dominates(self):
        """No single tier should have more than 60% of all scenarios."""
        counts = Counter(s["difficulty"] for s in SCENARIOS)
        total = len(SCENARIOS)
        for diff, count in counts.items():
            pct = count / total
            assert pct < 0.60, (
                f"{diff} tier has {count}/{total} = {pct:.0%} of scenarios (max: 60%)"
            )


class TestLabelDistribution:
    """Labels should be distributed across all difficulties."""

    def test_all_labels_present(self):
        labels = {s["label"] for s in SCENARIOS}
        assert labels == {"refund", "complaint", "query"}

    def test_minimum_per_label(self):
        counts = Counter(s["label"] for s in SCENARIOS)
        for label in ["refund", "complaint", "query"]:
            assert counts[label] >= 3, f"{label} has only {counts[label]} scenarios"


class TestEscalationLogic:
    """Escalation flags should be logically consistent."""

    def test_hard_complaints_may_need_escalation(self):
        """At least some hard complaints should require escalation."""
        hard_complaints = [
            s for s in SCENARIOS
            if s["difficulty"] == "hard" and s["label"] == "complaint"
        ]
        if hard_complaints:
            esc = [s for s in hard_complaints if s["requires_escalation"]]
            assert len(esc) > 0, "No hard complaints require escalation"

    def test_easy_low_urgency_rarely_escalate(self):
        """Easy scenarios with low urgency and low complexity should rarely escalate."""
        easy_low = [
            s for s in SCENARIOS
            if s["difficulty"] == "easy"
            and s["urgency"] == "low"
            and s["complexity"] <= 2
        ]
        escalating = [s for s in easy_low if s["requires_escalation"]]
        if easy_low:
            ratio = len(escalating) / len(easy_low)
            assert ratio <= 0.5, (
                f"{len(escalating)}/{len(easy_low)} easy+low scenarios require escalation "
                f"({ratio:.0%} > 50%)"
            )


class TestEnvironmentCompatibility:
    """All scenarios must work with the environment."""

    def test_all_scenarios_complete_episode(self):
        """Every scenario runs a 3-step episode without errors."""
        env = WorkplaceEnvironment()

        for idx, scenario in enumerate(SCENARIOS):
            env._state.scenario_index = idx
            env.reset()

            # Step through a complete episode
            obs1 = env.step(WorkplaceAction(action_type="classify", content=scenario["label"]))
            assert obs1.reward is not None, f"Scenario {idx}: classify reward is None"
            assert 0.0 <= obs1.reward <= 1.0, f"Scenario {idx}: classify reward OOB: {obs1.reward}"

            obs2 = env.step(WorkplaceAction(
                action_type="reply",
                content="Thank you for contacting us. We apologize for the inconvenience.",
            ))
            assert obs2.reward is not None, f"Scenario {idx}: reply reward is None"

            esc = "yes" if scenario.get("requires_escalation") else "no"
            obs3 = env.step(WorkplaceAction(action_type="escalate", content=esc))
            assert obs3.reward is not None, f"Scenario {idx}: escalate reward is None"
            assert obs3.done, f"Scenario {idx}: episode not done after step 3"
