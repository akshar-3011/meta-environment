"""Gymnasium wrapper for WorkplaceEnvironment — bridges OpenEnv to SB3.

The workplace environment uses text-based actions/observations. This wrapper
maps them to numerical spaces so that RL algorithms like PPO can train on it.

Action space (MultiDiscrete):
    [action_type (3), classify_choice (3), escalate_choice (2), reply_template (5)]
    - Step 1: agent picks classify_choice → maps to category label
    - Step 2: agent picks reply_template → maps to a reply string
    - Step 3: agent picks escalate_choice → maps to "yes"/"no"

Observation space (Box):
    [step_num, difficulty, sentiment, urgency, complexity,
     last_reward, cumulative_reward, classify_score, reply_score,
     email_length, has_exclamation, has_question, has_threat_words]
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.workplace_environment import WorkplaceEnvironment
from models import WorkplaceAction
from data.scenario_repository import SCENARIOS

logging.getLogger("environment.workplace_environment").setLevel(logging.WARNING)

# ─── Constants ───────────────────────────────────────────────────────────────

CATEGORIES = ["refund", "complaint", "query"]
ESCALATION_CHOICES = ["no", "yes"]
DIFFICULTY_MAP = {"easy": 0, "medium": 1, "hard": 2}
SENTIMENT_MAP = {"positive": 0, "neutral": 1, "mixed": 2, "negative": 3}
URGENCY_MAP = {"low": 0, "medium": 1, "high": 2}

REPLY_TEMPLATES = [
    # Template 0: Generic professional
    "Thank you for reaching out. We sincerely apologize for this experience. "
    "We understand your frustration and take this seriously. "
    "Our dedicated team will process your request within 24 hours.",

    # Template 1: Refund-focused
    "We apologize for the inconvenience. Your refund has been initiated and will be processed "
    "within 3-5 business days. The amount will be returned to your original payment method. "
    "Please contact us if you need further assistance.",

    # Template 2: Complaint-focused
    "We are deeply sorry for this completely unacceptable experience. You deserve better. "
    "We take this seriously and have escalated this to our priority team. "
    "Someone will contact you within 24 hours to resolve this immediately.",

    # Template 3: Query-focused
    "We're happy to help with your question. Here is the information you requested. "
    "Please let us know if you need any additional details or clarification. "
    "Our support team is available to assist you further.",

    # Template 4: Empathetic + closing
    "Dear valued customer, we understand your frustration and are deeply sorry. "
    "This is completely unacceptable and we assure you our team is dedicated to resolving this. "
    "We will contact you shortly. Best regards, Support Team.",
]

THREAT_WORDS = {"lawyer", "legal", "sue", "attorney", "court", "bbb", "report"}

# ─── Gymnasium Wrapper ──────────────────────────────────────────────────────


class WorkplaceGymEnv(gym.Env):
    """Gymnasium-compatible wrapper for the OpenEnv WorkplaceEnvironment.

    Args:
        difficulty_filter: Only use scenarios of this difficulty ("easy", "medium", "hard", or None for all).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty_filter: Optional[str] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Filter scenarios if requested
        if difficulty_filter:
            filtered = [s for s in SCENARIOS if s["difficulty"] == difficulty_filter]
            assert filtered, f"No scenarios with difficulty={difficulty_filter}"
            self._scenario_pool = filtered
        else:
            self._scenario_pool = list(SCENARIOS)

        self._env = WorkplaceEnvironment()
        self._current_step = 0
        self._last_obs_raw = None
        self._episode_rewards: Dict[str, float] = {}

        # Action space: [action_type(3), classify_choice(3), escalate_choice(2), reply_template(5)]
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 5])

        # Observation space: 13 numerical features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([3, 2, 3, 2, 5, 1, 3, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        )

    def _encode_observation(self, obs, last_reward: float = 0.0) -> np.ndarray:
        """Convert text observation to numerical feature vector."""
        email = obs.email.lower() if obs.email else ""
        email_len = min(len(email) / 500.0, 1.0)  # Normalize to [0, 1]
        has_excl = 1.0 if "!" in email else 0.0
        has_question = 1.0 if "?" in email else 0.0
        has_threat = 1.0 if any(w in email for w in THREAT_WORDS) else 0.0

        features = np.array([
            self._current_step,                                         # step_num: 0-3
            DIFFICULTY_MAP.get(obs.scenario_difficulty, 0),             # difficulty: 0-2
            SENTIMENT_MAP.get(obs.sentiment, 1),                       # sentiment: 0-3
            URGENCY_MAP.get(obs.urgency, 0),                           # urgency: 0-2
            min(obs.complexity_score, 5) / 5.0,                        # complexity: 0-1
            last_reward,                                                # last_reward: -1 to 1
            min(self._cumulative / 3.0, 1.0),                         # cumulative: 0-1
            self._episode_rewards.get("classify", 0.0),                # classify_score
            self._episode_rewards.get("reply", 0.0),                   # reply_score
            email_len,                                                  # email_length: 0-1
            has_excl,                                                   # has_exclamation
            has_question,                                               # has_question
            has_threat,                                                 # has_threat_words
        ], dtype=np.float32)
        return features

    def _decode_action(self, action: np.ndarray) -> WorkplaceAction:
        """Convert numerical action to WorkplaceAction based on current step."""
        action_type_idx, classify_idx, escalate_idx, reply_idx = action

        if self._current_step == 0:
            # Step 1: Classification
            return WorkplaceAction(
                action_type="classify",
                content=CATEGORIES[classify_idx],
            )
        elif self._current_step == 1:
            # Step 2: Reply
            return WorkplaceAction(
                action_type="reply",
                content=REPLY_TEMPLATES[reply_idx],
            )
        else:
            # Step 3: Escalation
            return WorkplaceAction(
                action_type="escalate",
                content=ESCALATION_CHOICES[escalate_idx],
            )

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._current_step = 0
        self._cumulative = 0.0
        self._episode_rewards = {}

        # Set a random scenario from the pool
        if self.np_random is not None:
            idx = int(self.np_random.integers(0, len(self._scenario_pool)))
        else:
            idx = 0
        self._env._state.scenario_index = idx % len(self._env._scenarios)

        obs_raw = self._env.reset()
        self._last_obs_raw = obs_raw
        obs = self._encode_observation(obs_raw)
        return obs, {"scenario": self._env._state.current}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        workplace_action = self._decode_action(action)
        obs_raw = self._env.step(workplace_action)
        self._last_obs_raw = obs_raw

        reward = obs_raw.reward or 0.0
        self._cumulative += reward
        self._current_step += 1
        self._episode_rewards[workplace_action.action_type] = reward

        done = obs_raw.done or self._current_step >= 3
        obs = self._encode_observation(obs_raw, last_reward=reward)

        info = {
            "action_type": workplace_action.action_type,
            "content": workplace_action.content[:50],
            "step_reward": reward,
            "cumulative_reward": self._cumulative,
            "episode_rewards": dict(self._episode_rewards),
        }

        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == "human" and self._last_obs_raw:
            print(f"  Step {self._current_step}: "
                  f"email={self._last_obs_raw.email[:40]}... "
                  f"reward={self._last_obs_raw.reward}")


# ─── Registration ────────────────────────────────────────────────────────────

def register_envs():
    """Register all variants with Gymnasium."""
    for diff in [None, "easy", "medium", "hard"]:
        suffix = f"-{diff}" if diff else ""
        env_id = f"WorkplaceTriage{suffix.replace('-', '').title()}-v1" if diff else "WorkplaceTriage-v1"

        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point="environment.gym_wrapper:WorkplaceGymEnv",
                kwargs={"difficulty_filter": diff},
                max_episode_steps=3,
            )


register_envs()
