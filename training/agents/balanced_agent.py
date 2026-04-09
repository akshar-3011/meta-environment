"""Balanced agent: Standard PPO with default rewards.

Philosophy: Equal weight to all steps — no reward shaping bias. Serves as
the baseline for comparing archetype-specific strategies.
"""

from __future__ import annotations

from training.agents import AgentConfig

BALANCED_CONFIG = AgentConfig(
    name="balanced",
    penalty_scale=1.0,
    escalation_bonus=0.0,
    escalation_threshold=0.5,
    classify_weight=1.0,
    reply_weight=1.0,
    escalate_weight=1.0,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    total_timesteps=50_000,
)
