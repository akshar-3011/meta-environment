"""Base abstractions for extensible inference strategies."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    from ..config import get_config
    from ..exceptions import InferenceError
    from ..logging_config import get_logger
except ImportError:  # pragma: no cover
    from core.config import get_config
    from core.exceptions import InferenceError
    from core.logging_config import get_logger


_CFG = get_config()
LOGGER = get_logger(__name__)
DEFAULT_BASE_URL = _CFG.inference.base_url


@dataclass(frozen=True)
class RetryConfig:
    """Retry/backoff configuration for HTTP calls."""

    max_attempts: int = _CFG.inference.retry_attempts
    backoff_seconds: float = _CFG.inference.retry_backoff_seconds


class BaseInference(ABC):
    """Template base class for inference strategies.

    Contract:
      - subclasses provide scenario-specific action plan via `build_actions()`
      - base class provides robust HTTP operations, retries, timeout handling,
        and batch execution.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = _CFG.inference.timeout_seconds,
        retry: Optional[RetryConfig] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.retry = retry or RetryConfig()

    @property
    @abstractmethod
    def title(self) -> str:
        raise NotImplementedError

    @property
    def reveal_label(self) -> bool:
        return False

    @abstractmethod
    def build_actions(self, observation: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Build ordered actions for a single episode."""
        raise NotImplementedError

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST with retry and timeout handling."""
        url = f"{self.base_url}{path}"
        attempts = max(1, self.retry.max_attempts)

        for attempt in range(1, attempts + 1):
            try:
                LOGGER.debug("POST %s attempt %s/%s", path, attempt, attempts)
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=(5.0, self.timeout),  # N7: (connect, read) timeouts
                )
                if response.ok:
                    return response.json()

                LOGGER.warning(
                    "HTTP %s on %s attempt %s/%s",
                    response.status_code,
                    path,
                    attempt,
                    attempts,
                )
                # C9 Fix: replace print() with structured logging
                LOGGER.warning(
                    "HTTP %s on %s attempt %s/%s body=%s",
                    response.status_code, path, attempt, attempts,
                    response.text[:200],
                )
                response.raise_for_status()
            except requests.Timeout:
                LOGGER.warning(
                    "Timeout after %ss on %s attempt %s/%s",
                    self.timeout, path, attempt, attempts,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.exception(
                    "Request failed on %s attempt %s/%s: %s",
                    path, attempt, attempts, exc,
                )

            if attempt < attempts:
                time.sleep(self.retry.backoff_seconds * attempt)

        return None

    def _step(self, action_type: str, content: str) -> Optional[Dict[str, Any]]:
        return self._post("/step", {"action": {"action_type": action_type, "content": content}})

    def _print_metadata(self, observation: Dict[str, Any]):
        metadata = observation.get("scenario_metadata", {})

        print(f"\n📧 Email: {observation.get('email')}")
        print("\n📊 Scenario Metadata:")
        print(f"   Difficulty  : {observation.get('scenario_difficulty', 'N/A')}")
        print(f"   Urgency     : {observation.get('urgency', 'N/A')}")
        print(f"   Sentiment   : {observation.get('sentiment', 'N/A')}")
        print(f"   Complexity  : {observation.get('complexity_score', 'N/A')}/5")
        print(f"   Categories  : {observation.get('category_options')}")

        if self.reveal_label:
            print(f"\n✅ True Label         : {metadata.get('label', 'N/A')}")
            print(f"   Requires Escalation: {metadata.get('requires_escalation', False)}")
        else:
            print("   (True label withheld — agent must classify from email content)")
        print(f"   Min Reply Length   : {metadata.get('min_reply_length', 'N/A')}")

    def _print_episode_summary(self, rewards_by_action: Dict[str, float], total_reward: float):
        print(f"\n{'=' * 70}")
        print("EPISODE SUMMARY")
        print("=" * 70)
        print("\n📊 Reward Breakdown:")
        print(f"   Classify : {rewards_by_action.get('classify', 0.0):.3f}  (weight 40%)")
        print(f"   Reply    : {rewards_by_action.get('reply', 0.0):.3f}  (weight 35%)")
        print(f"   Escalate : {rewards_by_action.get('escalate', 0.0):.3f}  (weight 25%)")
        print(f"\n   Total    : {total_reward:.3f}")

        if total_reward >= 0.85:
            print("\n🌟 Excellent — agent handled the scenario very well.")
        elif total_reward >= 0.70:
            print("\n✓ Good — mostly correct decisions.")
        elif total_reward >= 0.50:
            print("\n~ Fair — room for improvement.")
        else:
            print("\n✗ Poor — agent needs better classification and escalation logic.")

    def run_episode(self, actions: Optional[Sequence[Tuple[str, str]]] = None) -> Optional[Dict[str, Any]]:
        """Execute a single episode using strategy-defined or provided actions."""
        print("\n" + "=" * 70)
        print(f" {self.title}")
        print("=" * 70)

        print("\n[RESET]")
        reset_resp = self._post("/reset", {})
        if not reset_resp:
            raise InferenceError(
                "Failed to reset inference episode",
                details={"base_url": self.base_url},
            )

        observation = reset_resp.get("observation", {})
        self._print_metadata(observation)

        planned_actions = list(actions) if actions is not None else self.build_actions(observation)

        rewards_by_action: Dict[str, float] = {}
        total_reward = 0.0
        done = False

        for i, (action_type, content) in enumerate(planned_actions, start=1):
            print(f"\n{'=' * 70}")
            print(f"STEP {i}: {action_type.upper()}")
            print("=" * 70)

            step_resp = self._step(action_type, content)
            if not step_resp:
                raise InferenceError(
                    "Failed during inference step",
                    details={"action_type": action_type, "step": i},
                )

            reward = float(step_resp.get("reward", 0.0))
            done = bool(step_resp.get("done", False))
            observation = step_resp.get("observation", {})

            total_reward += reward
            rewards_by_action[action_type] = reward

            print(f"\n📤 Action  : {action_type}")
            print(f"   Content : {content[:80]}{'...' if len(content) > 80 else ''}")
            print(f"\n🎯 Reward     : {reward:.3f}")
            print(f"   Cumulative : {total_reward:.3f}")
            print(f"   Done       : {done}")
            print(f"\n📋 History ({len(observation.get('history', []))} steps):")
            for j, entry in enumerate(observation.get("history", []), 1):
                print(f"   {j}. {entry}")

            if done:
                break

        self._print_episode_summary(rewards_by_action, total_reward)

        return {
            "title": self.title,
            "total_reward": total_reward,
            "done": done,
            "rewards_by_action": rewards_by_action,
            "observation": observation,
        }

    def run_batch(self, batch_actions: Iterable[Optional[Sequence[Tuple[str, str]]]]) -> List[Optional[Dict[str, Any]]]:
        """Run multiple episodes sequentially (batch processing)."""
        results: List[Optional[Dict[str, Any]]] = []
        for index, actions in enumerate(batch_actions, start=1):
            print(f"\n🔁 Batch Episode {index}")
            results.append(self.run_episode(actions=actions))
        return results
