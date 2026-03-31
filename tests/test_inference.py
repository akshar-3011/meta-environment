"""Unit tests for inference strategies."""

from __future__ import annotations

import asyncio

from workplace_env.core.inference.strategies import AsyncInference, EnhancedInference, StandardInference


def test_standard_inference_build_actions_shape_and_order():
    strategy = StandardInference()

    actions = strategy.build_actions({"email": "Need help"})

    assert len(actions) == 3
    assert actions[0][0] == "classify"
    assert actions[1][0] == "reply"
    assert actions[2][0] == "escalate"
    assert actions[0][1] == "complaint"
    assert "apolog" in actions[1][1].lower()


def test_enhanced_strategy_overrides_title_and_reveal_label():
    strategy = EnhancedInference()

    assert strategy.reveal_label is True
    assert "ENHANCED" in strategy.title


def test_async_inference_run_episode_async_delegates_to_run_episode(monkeypatch):
    strategy = AsyncInference()
    expected = {"total_reward": 0.5}

    def fake_run_episode(actions=None):
        assert actions == [("classify", "query")]
        return expected

    monkeypatch.setattr(strategy, "run_episode", fake_run_episode)

    result = asyncio.run(strategy.run_episode_async(actions=[("classify", "query")]))

    assert result == expected


def test_async_inference_run_batch_async_preserves_order(monkeypatch):
    strategy = AsyncInference()

    def fake_run_episode(actions=None):
        action_count = 0 if actions is None else len(actions)
        return {"action_count": action_count, "actions": actions}

    monkeypatch.setattr(strategy, "run_episode", fake_run_episode)

    batch = [
        [("classify", "complaint")],
        None,
        [("classify", "refund"), ("escalate", "no")],
    ]

    results = asyncio.run(strategy.run_batch_async(batch))

    assert len(results) == 3
    assert results[0]["action_count"] == 1
    assert results[1]["action_count"] == 0
    assert results[2]["action_count"] == 2
