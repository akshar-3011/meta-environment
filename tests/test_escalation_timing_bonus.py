"""Tests for C2/C9 escalation timing bonus — validates bonus fires at step 3."""

from __future__ import annotations

import pytest

from workplace_env.core.graders.rule_based import RuleBasedRewardPolicy


class TestEscalationTimingBonus:
    """Verify the escalation timing logic after C2 fix."""

    def setup_method(self):
        self.policy = RuleBasedRewardPolicy()

    def test_timing_bonus_at_step_3(self):
        """Correct escalation at step 3 gets timing bonus (+0.1)."""
        score, _ = self.policy.grade_escalation("yes", "complaint", step_count=3)
        # 1.0 (correct) + 0.1 (timing) = 1.1 → clamped to 1.0
        assert score >= 0.9

    def test_early_escalation_penalty_at_step_1(self):
        """Escalation at step 1 gets timing penalty (*0.7)."""
        score_early, _ = self.policy.grade_escalation("yes", "complaint", step_count=1)
        score_normal, _ = self.policy.grade_escalation("yes", "complaint", step_count=3)
        assert score_early < score_normal

    def test_early_escalation_penalty_at_step_2(self):
        """Escalation at step 2 still gets timing penalty (*0.7)."""
        score_step2, _ = self.policy.grade_escalation("yes", "complaint", step_count=2)
        score_step3, _ = self.policy.grade_escalation("yes", "complaint", step_count=3)
        assert score_step2 < score_step3

    def test_no_escalation_no_timing_effects(self):
        """When agent doesn't escalate, timing logic doesn't apply."""
        score_step1, _ = self.policy.grade_escalation("no", "complaint", step_count=1)
        score_step3, _ = self.policy.grade_escalation("no", "complaint", step_count=3)
        # Both should be the same low score (incorrect non-escalation)
        assert score_step1 == score_step3

    def test_trajectory_bonus_fires_in_calculate_step_reward(self):
        """Trajectory consistency bonus adds +0.05 when prior steps are good."""
        good_prior = {"classify": 0.40, "reply": 0.30}

        r_with_bonus, bd_bonus = self.policy.calculate_step_reward(
            action_type="escalate",
            content="yes",
            actual_category="complaint",
            step_count=3,
            previous_actions=good_prior,
        )

        bad_prior = {"classify": 0.10, "reply": 0.05}
        r_without, bd_no = self.policy.calculate_step_reward(
            action_type="escalate",
            content="yes",
            actual_category="complaint",
            step_count=3,
            previous_actions=bad_prior,
        )

        assert "trajectory_bonus" in bd_bonus
        assert r_with_bonus > r_without

    @pytest.mark.parametrize("step", [1, 2])
    def test_all_early_steps_penalized(self, step):
        """All steps < 3 get early escalation penalty."""
        r_early, _ = self.policy.calculate_step_reward(
            action_type="escalate",
            content="yes",
            actual_category="complaint",
            step_count=step,
            previous_actions={"classify": 0.4, "reply": 0.2},
        )
        r_normal, _ = self.policy.calculate_step_reward(
            action_type="escalate",
            content="yes",
            actual_category="complaint",
            step_count=3,
            previous_actions={"classify": 0.4, "reply": 0.2},
        )
        assert r_early < r_normal
