"""Agent archetype definitions and reward-shaping wrapper.

Each archetype wraps the base WorkplaceGymEnv with a RewardShapingWrapper
that modifies escalation rewards according to the agent's personality.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from environment.gym_wrapper import WorkplaceGymEnv


@dataclass
class AgentConfig:
    """Archetype configuration loaded from YAML."""
    name: str = "balanced"
    penalty_scale: float = 1.0       # Multiplier for wrong-escalation penalty
    escalation_bonus: float = 0.0    # Bonus for correct escalation
    escalation_threshold: float = 0.5  # Reward below this = "wrong" escalation
    classify_weight: float = 1.0     # Multiplier for classify rewards
    reply_weight: float = 1.0        # Multiplier for reply rewards
    escalate_weight: float = 1.0     # Multiplier for escalate rewards
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01
    clip_range: float = 0.2
    total_timesteps: int = 50_000
    net_arch_pi: list = field(default_factory=lambda: [128, 128])
    net_arch_vf: list = field(default_factory=lambda: [128, 128])

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RewardShapingWrapper(gym.Wrapper):
    """Reshape rewards based on agent archetype config.

    - Conservative: Heavy penalty for wrong escalations, moderate classify/reply
    - Aggressive: Bonus for correct escalations, lower penalty for wrong ones
    - Balanced: Pass-through with unit weights
    """

    def __init__(self, env: gym.Env, config: AgentConfig):
        super().__init__(env)
        self.config = config
        self._step_in_episode = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_in_episode = 0
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_in_episode += 1

        action_type = info.get("action_type", "")
        shaped_reward = reward

        if action_type == "classify":
            shaped_reward = reward * self.config.classify_weight
        elif action_type == "reply":
            shaped_reward = reward * self.config.reply_weight
        elif action_type == "escalate":
            shaped_reward = reward * self.config.escalate_weight
            if reward < self.config.escalation_threshold:
                # Wrong escalation → apply penalty scaling
                shaped_reward = reward * self.config.penalty_scale * -1.0
            else:
                # Correct escalation → apply bonus
                shaped_reward = reward + self.config.escalation_bonus

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["agent_type"] = self.config.name

        return obs, shaped_reward, terminated, truncated, info


def make_archetype_env(
    config: AgentConfig,
    difficulty_filter: Optional[str] = None,
    seed: int = 42,
) -> gym.Env:
    """Create a reward-shaped environment for the given archetype."""
    from stable_baselines3.common.monitor import Monitor

    base_env = WorkplaceGymEnv(difficulty_filter=difficulty_filter)
    shaped_env = RewardShapingWrapper(base_env, config)
    monitored_env = Monitor(shaped_env)
    monitored_env.reset(seed=seed)
    return monitored_env
