"""Tests for A/B experiment framework.

Covers:
  1. Experimental policies produce valid rewards
  2. Consistent hashing is deterministic
  3. Experiment store CRUD operations
  4. Routing latency < 5ms
  5. Max 2 concurrent experiments enforced
  6. Episode recording and metrics
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rewards.experimental_policies import (
    ConfigurableRewardPolicy,
    EqualWeightPolicy,
    EscalationFirstPolicy,
    ReplyQualityPolicy,
    POLICY_WEIGHTS,
    get_policy,
)
from core.graders.rule_based import RuleBasedRewardPolicy


# ─── Policy Tests ────────────────────────────────────────────────────────────

class TestExperimentalPolicies:

    @pytest.fixture
    def policies(self):
        return {
            "control": RuleBasedRewardPolicy(),
            "equal": EqualWeightPolicy(),
            "escalation_first": EscalationFirstPolicy(),
            "reply_quality": ReplyQualityPolicy(),
        }

    def test_all_policies_produce_valid_rewards(self, policies):
        """All policies return rewards in [0.0, 1.0]."""
        for name, policy in policies.items():
            r, breakdown = policy.calculate_step_reward(
                action_type="classify",
                content="refund",
                actual_category="refund",
                step_count=0,
            )
            assert 0.0 <= r <= 1.0, f"{name} classify reward OOB: {r}"

            r, breakdown = policy.calculate_step_reward(
                action_type="reply",
                content="Thank you for contacting us. We apologize for the inconvenience. We will process your request.",
                actual_category="refund",
                step_count=1,
                previous_actions={"classify": 0.4},
            )
            assert 0.0 <= r <= 1.0, f"{name} reply reward OOB: {r}"

            r, breakdown = policy.calculate_step_reward(
                action_type="escalate",
                content="no",
                actual_category="refund",
                step_count=2,
                previous_actions={"classify": 0.4, "reply": 0.3},
            )
            assert 0.0 <= r <= 1.0, f"{name} escalate reward OOB: {r}"

    def test_weight_differences_affect_rewards(self, policies):
        """Different policies produce different rewards for the same action."""
        rewards = {}
        for name, policy in policies.items():
            r, _ = policy.calculate_step_reward(
                action_type="classify",
                content="refund",
                actual_category="refund",
                step_count=0,
            )
            rewards[name] = r

        # Control uses 0.40 weight, equal uses 0.333
        # Since grading score is the same, only weight differs
        assert rewards["control"] != rewards["equal"], (
            "Control and equal should differ for classify"
        )

    def test_escalation_first_emphasizes_escalation(self, policies):
        """EscalationFirstPolicy gives higher escalation reward weight."""
        correct_esc_args = dict(
            action_type="escalate",
            content="yes",
            actual_category="complaint",
            step_count=2,
            previous_actions={"classify": 0.4, "reply": 0.3},
        )
        control_r, _ = policies["control"].calculate_step_reward(**correct_esc_args)
        esc_first_r, _ = policies["escalation_first"].calculate_step_reward(**correct_esc_args)

        # escalation_first uses 0.50 weight vs control's 0.25
        assert esc_first_r > control_r, (
            f"Escalation-first ({esc_first_r}) should give higher escalation "
            f"reward than control ({control_r})"
        )

    def test_reply_quality_emphasizes_reply(self, policies):
        """ReplyQualityPolicy gives higher reply reward weight."""
        good_reply_args = dict(
            action_type="reply",
            content="Thank you for contacting us. We sincerely apologize for this experience. "
                    "We will process your refund within 3-5 business days.",
            actual_category="refund",
            step_count=1,
            min_reply_length=30,
            previous_actions={"classify": 0.4},
        )
        control_r, _ = policies["control"].calculate_step_reward(**good_reply_args)
        reply_r, _ = policies["reply_quality"].calculate_step_reward(**good_reply_args)

        # reply_quality uses 0.50 weight vs control's 0.35
        assert reply_r > control_r, (
            f"Reply-quality ({reply_r}) should give higher reply "
            f"reward than control ({control_r})"
        )

    def test_get_policy_factory(self):
        """get_policy returns correct types."""
        assert isinstance(get_policy("control"), RuleBasedRewardPolicy)
        assert isinstance(get_policy("equal"), EqualWeightPolicy)
        assert isinstance(get_policy("escalation_first"), EscalationFirstPolicy)
        assert isinstance(get_policy("reply_quality"), ReplyQualityPolicy)

    def test_get_policy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            get_policy("nonexistent")

    def test_policy_weights_sum_to_one(self):
        """All predefined policy weights should sum to approximately 1.0."""
        for name, weights in POLICY_WEIGHTS.items():
            total = weights.classify + weights.reply + weights.escalate
            assert abs(total - 1.0) < 0.01, (
                f"Policy '{name}' weights sum to {total}, expected ~1.0"
            )

    def test_configurable_policy_name(self):
        """Policy name matches registry."""
        p = EqualWeightPolicy()
        assert p.policy_name == "equal"
        p2 = EscalationFirstPolicy()
        assert p2.policy_name == "escalation_first"


# ─── Experiment Store Tests ──────────────────────────────────────────────────

class TestExperimentStore:

    @pytest.fixture
    def store(self, tmp_path):
        from api.experiments import ExperimentStore
        db_path = str(tmp_path / "test_experiments.db")
        return ExperimentStore(db_path=db_path)

    @pytest.fixture
    def sample_experiment(self, store):
        from api.experiments import CreateExperimentRequest
        req = CreateExperimentRequest(
            name="test-equal",
            policy_type="equal",
            traffic_split=0.2,
        )
        return store.create_experiment(req)

    def test_create_experiment(self, sample_experiment):
        assert sample_experiment.name == "test-equal"
        assert sample_experiment.policy_type == "equal"
        assert sample_experiment.traffic_split == 0.2
        assert sample_experiment.status == "active"

    def test_get_experiment(self, store, sample_experiment):
        fetched = store.get_experiment(sample_experiment.id)
        assert fetched.id == sample_experiment.id
        assert fetched.name == "test-equal"

    def test_max_2_concurrent_enforced(self, store):
        from api.experiments import CreateExperimentRequest
        from fastapi import HTTPException

        store.create_experiment(CreateExperimentRequest(
            name="exp-1", policy_type="equal", traffic_split=0.1,
        ))
        store.create_experiment(CreateExperimentRequest(
            name="exp-2", policy_type="escalation_first", traffic_split=0.1,
        ))

        with pytest.raises(HTTPException) as exc_info:
            store.create_experiment(CreateExperimentRequest(
                name="exp-3", policy_type="reply_quality", traffic_split=0.1,
            ))
        assert exc_info.value.status_code == 409

    def test_routing_deterministic(self, store, sample_experiment):
        """Same scenario always routes to same variant."""
        results = [
            store.route_episode("scenario_E1")
            for _ in range(10)
        ]
        # All results should be identical
        variants = {r["variant"] for r in results if r}
        assert len(variants) == 1, "Routing should be deterministic"

    def test_routing_splits_traffic(self, store, sample_experiment):
        """Traffic split roughly matches configuration."""
        variant_count = 0
        total = 200
        for i in range(total):
            result = store.route_episode(f"scenario_{i}")
            if result and result["variant"] == "variant":
                variant_count += 1

        # With 20% split, expect ~40 out of 200 (allow wide tolerance)
        assert 10 < variant_count < 80, (
            f"Expected ~20% variant, got {variant_count}/{total} = "
            f"{variant_count/total*100:.0f}%"
        )

    def test_routing_latency_under_5ms(self, store, sample_experiment):
        """Routing should add <5ms latency, measured over 1000 iterations."""
        # Warm up cache
        store.route_episode("warmup")

        start = time.perf_counter()
        iterations = 1000
        for i in range(iterations):
            store.route_episode(f"scenario_{i}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_call_ms = elapsed_ms / iterations
        assert per_call_ms < 5.0, (
            f"Routing latency {per_call_ms:.2f}ms per call exceeds 5ms target"
        )

    def test_record_episode(self, store, sample_experiment):
        from api.experiments import RecordEpisodeRequest
        ep_id = store.record_episode(RecordEpisodeRequest(
            experiment_id=sample_experiment.id,
            scenario_id="E1",
            variant="variant",
            step_rewards=[0.4, 0.3, 0.25],
            total_reward=0.95,
            policy_type="equal",
        ))
        assert ep_id is not None

        # Verify metrics
        exp = store.get_experiment(sample_experiment.id)
        assert exp.metrics["variant"]["count"] == 1
        assert exp.metrics["variant"]["mean_reward"] == 0.95

    def test_update_status(self, store, sample_experiment):
        from api.experiments import ExperimentStatus
        updated = store.update_status(sample_experiment.id, ExperimentStatus.COMPLETED)
        assert updated.status == "completed"

    def test_scenario_filter(self, store):
        from api.experiments import CreateExperimentRequest
        exp = store.create_experiment(CreateExperimentRequest(
            name="filtered",
            policy_type="equal",
            traffic_split=0.5,
            target_scenarios=["E1", "E2", "E3"],
        ))

        # E1 should route to experiment
        result = store.route_episode("E1")
        assert result is not None
        assert result["experiment_id"] == exp.id

        # E99 should NOT route (not in filter)
        result2 = store.route_episode("E99")
        assert result2 is None
