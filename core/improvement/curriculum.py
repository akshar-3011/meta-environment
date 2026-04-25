"""Curriculum-based scenario sampling for the improvement loop.

Upweights scenarios where the agent performs poorly (total_reward < 0.60),
so subsequent generations are evaluated on a harder, failure-focused subset.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from core.memory.reward_memory import RewardMemory
from data.scenario_repository import SCENARIOS


class CurriculumSampler:
    """Weighted scenario sampler that biases toward failure scenarios.

    Initialized with the full 100-scenario corpus from ``data.py``.
    After each generation, call ``update_weights`` to upweight scenarios
    where the agent scored below 0.60 total reward.  Then call ``sample(n)``
    to draw the next generation's evaluation pool.

    Parameters
    ----------
    corpus : list[dict] | None
        Override the default 100-scenario list.  If ``None``, uses the
        validated ``SCENARIOS`` from ``data.scenario_repository``.
    failure_threshold : float
        Episodes with ``total_reward < failure_threshold`` trigger
        weight increases for the corresponding scenario.
    weight_multiplier : float
        Factor applied to a scenario's weight on failure (default 3.0×).
    max_weight : float
        Hard cap on any single scenario's weight (default 10.0).
    seed : int | None
        Optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]] | None = None,
        failure_threshold: float = 0.60,
        weight_multiplier: float = 3.0,
        max_weight: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.corpus: List[Dict[str, Any]] = list(corpus if corpus is not None else SCENARIOS)
        self.failure_threshold = failure_threshold
        self.weight_multiplier = weight_multiplier
        self.max_weight = max_weight

        # Default weight 1.0 for every scenario index.
        self.scenario_weights: Dict[int, float] = {
            i: 1.0 for i in range(len(self.corpus))
        }

        self._rng = random.Random(seed)

        # Pre-compute snippet → index mapping (first 120 chars, as used by EpisodeRecord).
        self._snippet_index: Dict[str, int] = {}
        for idx, scenario in enumerate(self.corpus):
            snippet = str(scenario.get("email", ""))[:120]
            self._snippet_index[snippet] = idx

    def _find_scenario_index(self, email_snippet: str) -> int | None:
        """Match an episode's email_snippet to a corpus scenario index.

        Uses exact match on the first 120 characters — the same truncation
        applied by ``EpisodeRecord.__post_init__``.
        """
        snippet = (email_snippet or "")[:120]
        if snippet in self._snippet_index:
            return self._snippet_index[snippet]

        # Fallback: prefix match for minor whitespace differences.
        for stored_snippet, idx in self._snippet_index.items():
            if stored_snippet.startswith(snippet[:80]) and len(snippet) > 10:
                return idx
        return None

    def update_weights(self, memory: RewardMemory) -> None:
        """Upweight scenarios where the agent scored below the failure threshold.

        For each ``EpisodeRecord`` in *memory* with ``total_reward < 0.60``,
        find the matching scenario in the corpus (by email_snippet), and
        multiply its weight by ``weight_multiplier``, capped at ``max_weight``.
        """
        for record in memory.records:
            if record.total_reward < self.failure_threshold:
                idx = self._find_scenario_index(record.email_snippet)
                if idx is not None:
                    current_weight = self.scenario_weights[idx]
                    new_weight = min(
                        current_weight * self.weight_multiplier,
                        self.max_weight,
                    )
                    self.scenario_weights[idx] = new_weight

    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Return *n* scenarios sampled with replacement using current weights.

        Uses ``random.choices`` with weights derived from ``scenario_weights``.
        """
        weights = [self.scenario_weights[i] for i in range(len(self.corpus))]
        return self._rng.choices(self.corpus, weights=weights, k=n)

    def weight_summary(self) -> Dict[str, Any]:
        """Return a summary of current weight distribution."""
        weights = list(self.scenario_weights.values())
        upweighted = sum(1 for w in weights if w > 1.0)
        return {
            "total_scenarios": len(self.corpus),
            "upweighted_count": upweighted,
            "max_weight": max(weights) if weights else 0.0,
            "mean_weight": (sum(weights) / len(weights)) if weights else 0.0,
        }


__all__ = ["CurriculumSampler"]
