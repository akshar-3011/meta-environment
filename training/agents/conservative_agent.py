"""Conservative agent: PPO with high penalty for wrong escalations.

Philosophy: Minimize harm — unnecessary escalation wastes senior team time.
This agent learns to be very selective about escalation, preferring to resolve
issues at the frontline level whenever possible.
"""

from __future__ import annotations

from training.agents import AgentConfig

CONSERVATIVE_CONFIG = AgentConfig(
    name="conservative",
    penalty_scale=2.0,
    escalation_bonus=0.0,
    escalation_threshold=0.8,
    classify_weight=1.2,
    reply_weight=1.0,
    escalate_weight=0.8,
    learning_rate=2e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.005,
    clip_range=0.15,
    total_timesteps=50_000,
)
