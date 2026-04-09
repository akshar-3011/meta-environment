"""Aggressive agent: PPO with high reward for successful escalations.

Philosophy: Maximize escalation capture — better to escalate than miss
a critical issue that causes customer churn. This agent learns to escalate
proactively, even on borderline cases.
"""

from __future__ import annotations

from training.agents import AgentConfig

AGGRESSIVE_CONFIG = AgentConfig(
    name="aggressive",
    penalty_scale=0.5,
    escalation_bonus=0.15,
    escalation_threshold=0.3,
    classify_weight=0.9,
    reply_weight=1.0,
    escalate_weight=1.5,
    learning_rate=5e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=8,
    gamma=0.95,
    ent_coef=0.02,
    clip_range=0.25,
    total_timesteps=50_000,
)
