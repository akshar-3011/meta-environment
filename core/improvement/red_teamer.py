"""Regression testing via golden scenario validation.

Detects catastrophic forgetting by evaluating candidate strategies against a
fixed set of 10 representative scenarios spanning easy, medium, and hard
difficulty levels.  If the golden score drops below 90% of the baseline, the
strategy is flagged for retry with a broadened prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.inference.adaptive_agent import AdaptiveAgent
from core.memory.reward_memory import RewardMemory
from data.scenario_repository import SCENARIOS


# 10 indices spread across the 100-scenario corpus:
# Covers easy (0, 5, 12), medium (18, 24, 35), hard (48, 62, 74, 88)
GOLDEN_SCENARIO_INDICES: List[int] = [0, 5, 12, 18, 24, 35, 48, 62, 74, 88]


class RegressionTester:
    """Validates strategies against a fixed golden scenario set.

    Parameters
    ----------
    golden_indices : list[int] | None
        Override the default golden scenario indices.
    regression_threshold : float
        Fraction of baseline score that must be maintained (default 0.90 = 90%).
    """

    def __init__(
        self,
        golden_indices: List[int] | None = None,
        regression_threshold: float = 0.90,
    ) -> None:
        self.golden_indices = golden_indices or list(GOLDEN_SCENARIO_INDICES)
        self.regression_threshold = regression_threshold

        # Build the golden scenario pool from the full corpus.
        self._golden_scenarios: List[Dict[str, Any]] = []
        for idx in self.golden_indices:
            if 0 <= idx < len(SCENARIOS):
                self._golden_scenarios.append(SCENARIOS[idx])

    @property
    def golden_scenarios(self) -> List[Dict[str, Any]]:
        """Return the fixed golden scenario pool."""
        return list(self._golden_scenarios)

    def validate(
        self,
        strategy: Dict[str, Any],
        baseline_score: float,
    ) -> Tuple[bool, float]:
        """Run a candidate strategy on the golden scenarios.

        Parameters
        ----------
        strategy : dict
            The strategy dict to test (passed to AdaptiveAgent).
        baseline_score : float
            The baseline golden score to compare against.

        Returns
        -------
        (passed, score) : tuple[bool, float]
            ``passed`` is True if ``score >= baseline_score * 0.90``.
        """
        # Import here to avoid circular imports at module level.
        from improvement_loop import run_evaluation, _safe_strategy

        safe = _safe_strategy(strategy)
        agent = AdaptiveAgent(safe)

        memory = run_evaluation(
            agent=agent,
            n_episodes=len(self._golden_scenarios),
            strategy_version="golden_regression",
            scenario_pool=self._golden_scenarios,
        )

        # Compute mean total reward across golden scenarios.
        if memory.records:
            score = sum(r.total_reward for r in memory.records) / len(memory.records)
        else:
            score = 0.0

        threshold = baseline_score * self.regression_threshold
        passed = score >= threshold

        return (passed, round(score, 4))


__all__ = ["RegressionTester", "GOLDEN_SCENARIO_INDICES"]
