"""Environment orchestration with explicit state and dependency injection."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openenv.core import Environment
except ImportError:
    class Environment:
        """Minimal stub when openenv-core is not installed (local dev / tests)."""
        pass

try:
    from ..core.config import get_config
except ImportError:  # pragma: no cover
    from core.config import get_config

try:
    from ..core.exceptions import PipelineError
except ImportError:  # pragma: no cover
    from core.exceptions import PipelineError

try:
    from ..core.graders import CATEGORY_OPTIONS, RewardPolicy, RuleBasedRewardPolicy
except ImportError:  # pragma: no cover
    from core.graders import CATEGORY_OPTIONS, RewardPolicy, RuleBasedRewardPolicy

try:
    from ..core.logging_config import get_logger, setup_logging
except ImportError:  # pragma: no cover
    from core.logging_config import get_logger, setup_logging

try:
    from ..models import WorkplaceAction, WorkplaceObservation
except ImportError:  # pragma: no cover
    from models import WorkplaceAction, WorkplaceObservation

try:
    from ..data import get_default_repository, ScenarioRepository
except ImportError:  # pragma: no cover
    from data import get_default_repository, ScenarioRepository


setup_logging()
CFG = get_config()
LOGGER = get_logger(__name__)
DEBUG = CFG.environment.debug

# Lazy metrics + tracer (loaded on first use to avoid duplicate registry errors)
_HAS_METRICS = None  # tri-state: None=unchecked, True, False
_METRICS = {}
_TRACER = None


def _rebalance_scenarios_by_difficulty(
    scenarios: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Interleave easy/medium/hard scenarios so sequential resets cover all levels."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    leftovers: List[Dict[str, Any]] = []

    for scenario in scenarios:
        difficulty = str(scenario.get("difficulty", "easy")).lower()
        if difficulty in buckets:
            buckets[difficulty].append(scenario)
        else:
            leftovers.append(scenario)

    interleaved: List[Dict[str, Any]] = []
    max_len = max((len(values) for values in buckets.values()), default=0)
    for idx in range(max_len):
        for difficulty in ("easy", "medium", "hard"):
            group = buckets[difficulty]
            if idx < len(group):
                interleaved.append(group[idx])

    interleaved.extend(leftovers)
    return interleaved


def _ensure_metrics():
    """Import prometheus metrics on first use — prevents duplicate registrations."""
    global _HAS_METRICS, _METRICS
    if _HAS_METRICS is not None:
        return _HAS_METRICS
    try:
        from api.metrics import (
            ACTIVE_EPISODES, STEP_LATENCY, STEP_COUNT, ESCALATION_DECISIONS,
            REWARD_HISTOGRAM, EPISODE_COUNT, EPISODE_SCORE, GRADER_LATENCY,
            ERROR_COUNT,
        )
        _METRICS.update(
            ACTIVE_EPISODES=ACTIVE_EPISODES, STEP_LATENCY=STEP_LATENCY,
            STEP_COUNT=STEP_COUNT, ESCALATION_DECISIONS=ESCALATION_DECISIONS,
            REWARD_HISTOGRAM=REWARD_HISTOGRAM, EPISODE_COUNT=EPISODE_COUNT,
            EPISODE_SCORE=EPISODE_SCORE, GRADER_LATENCY=GRADER_LATENCY,
            ERROR_COUNT=ERROR_COUNT,
        )
        _HAS_METRICS = True
    except (ImportError, ValueError):
        _HAS_METRICS = False
    return _HAS_METRICS


def _ensure_tracer():
    """Import tracer on first use."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from api.tracing import get_tracer
        _TRACER = get_tracer()
    except (ImportError, Exception):
        _TRACER = None
    return _TRACER


# _debug_log is now an instance method on WorkplaceEnvironment (see C4 fix).


@dataclass
class EpisodeState:
    """Current environment episode state."""

    scenario_index: int = 0
    history: List[str] = field(default_factory=list)
    step_count: int = 0
    current: Dict[str, Any] = field(default_factory=dict)
    episode_count: int = 0
    action_rewards: Dict[str, float] = field(default_factory=dict)
    cumulative_reward: float = 0.0
    step_details: List[Dict[str, Any]] = field(default_factory=list)


class WorkplaceEnvironment(Environment):
    """Production-oriented OpenEnv environment with modular dependencies.

    Each instance owns its own EpisodeState so concurrent WebSocket sessions
    cannot corrupt each other's state. The original class-level `_state`
    attribute was a singleton shared across all instances — equivalent to the
    global `_SHARED_STATE` dict in the original codebase.
    """

    def __init__(
        self,
        debug: bool = CFG.environment.debug,
        reward_policy: Optional[RewardPolicy] = None,
        scenario_repository: Optional[ScenarioRepository] = None,
    ):
        self._debug = debug
        self._policy = reward_policy or RuleBasedRewardPolicy()
        self._scenario_repo = scenario_repository or get_default_repository()
        source_scenarios = self._scenario_repo.list_scenarios()
        self._scenarios = _rebalance_scenarios_by_difficulty(source_scenarios)

        difficulties = [str(s.get("difficulty", "easy")).lower() for s in self._scenarios]
        easy_count = sum(1 for d in difficulties if d == "easy")
        medium_count = sum(1 for d in difficulties if d == "medium")
        hard_count = sum(1 for d in difficulties if d == "hard")
        if os.environ.get("ENV_DEBUG"):
            print(f"Scenario pool: {easy_count} easy, {medium_count} medium, {hard_count} hard")
            print(f"Scenario difficulties first5={difficulties[:5]} last5={difficulties[-5:]}")

        # Instance-owned state — no sharing between concurrent sessions.
        self._state = EpisodeState()
        if self._scenarios:
            self._state.current = self._scenarios[0]

    def _debug_log(self, msg: str):
        if self._debug:
            LOGGER.debug(msg)

    def _next_scenario(self) -> Dict[str, Any]:
        idx = self._state.scenario_index % len(self._scenarios)
        self._state.scenario_index += 1
        scenario = self._scenarios[idx]
        self._debug_log(f"Loaded scenario {idx}: {scenario['email'][:50]}...")
        return scenario

    def _make_obs(self, reward: Optional[float] = None, done: bool = False) -> WorkplaceObservation:
        scenario = self._state.current
        return WorkplaceObservation(
            email=scenario["email"],
            category_options=CATEGORY_OPTIONS,
            history=list(self._state.history),
            reward=reward,
            done=done,
            scenario_difficulty=scenario.get("difficulty", "easy"),
            urgency=scenario.get("urgency", "low"),
            sentiment=scenario.get("sentiment", "neutral"),
            complexity_score=scenario.get("complexity", 1),
            scenario_metadata={
                "min_reply_length": scenario.get("min_reply_length", 20),
            },
        )

    def _grade_step(self, action: WorkplaceAction, step_count: int) -> float:
        scenario = self._state.current
        reward_value, breakdown = self._policy.calculate_step_reward(
            action_type=action.action_type,
            content=action.content,
            actual_category=scenario["label"],
            step_count=step_count,
            scenario_difficulty=scenario.get("difficulty", "easy"),
            min_reply_length=scenario.get("min_reply_length", 30),
            previous_actions=self._state.action_rewards,
        )

        self._state.step_details.append(breakdown)
        self._debug_log(
            f"Step {step_count} ({action.action_type}): "
            f"reward={reward_value:.3f}, {breakdown.get('explanation', '')}"
        )
        return reward_value

    def reset(self) -> WorkplaceObservation:
        self._debug_log("=" * 60)
        self._debug_log(f"RESET (episode {self._state.episode_count})")
        self._debug_log("=" * 60)

        self._state.history = []
        self._state.step_count = 0
        self._state.action_rewards = {}
        self._state.cumulative_reward = 0.0
        self._state.step_details = []
        self._state.current = self._next_scenario()
        self._state.episode_count += 1

        if _ensure_metrics():
            _METRICS["ACTIVE_EPISODES"].inc()

        return self._make_obs(reward=None, done=False)

    def step(self, action: WorkplaceAction) -> WorkplaceObservation:
        import time as _time

        # Start trace span if available
        _span = None
        tracer = _ensure_tracer()
        if tracer is not None:
            try:
                _span = tracer.start_span(f"env.step.{action.action_type}")
                _span.set_attribute("env.action_type", action.action_type)
                _span.set_attribute("env.step_count", self._state.step_count + 1)
                _span.set_attribute("env.difficulty", self._state.current.get("difficulty", "unknown"))
            except Exception:
                _span = None

        step_start = _time.monotonic()
        try:
            if action.action_type not in ["classify", "reply", "escalate"]:
                self._debug_log(f"Invalid action type: {action.action_type}")
                if _ensure_metrics():
                    _METRICS["ERROR_COUNT"].labels(error_type="invalid_action").inc()
                return self._make_obs(reward=0.0, done=True)

            self._state.step_count += 1
            step_num = self._state.step_count

            content = action.content or ""
            content_preview = content[:50] + ("..." if len(content) > 50 else "")
            action_str = f"{action.action_type}: {content_preview}"
            self._state.history.append(action_str)
            self._debug_log(f"Step {step_num}: {action_str}")

            # Grade with timing
            grade_start = _time.monotonic()
            reward_value = self._grade_step(action, step_num)
            grade_elapsed = _time.monotonic() - grade_start

            self._state.action_rewards[action.action_type] = reward_value
            self._state.cumulative_reward += reward_value

            # Record metrics
            if _ensure_metrics():
                difficulty = self._state.current.get("difficulty", "unknown")
                _METRICS["STEP_COUNT"].labels(action_type=action.action_type).inc()
                _METRICS["GRADER_LATENCY"].labels(action_type=action.action_type).observe(grade_elapsed)
                _METRICS["REWARD_HISTOGRAM"].labels(
                    step=action.action_type, difficulty=difficulty,
                ).observe(reward_value)

                # Track escalation decisions
                if action.action_type == "escalate":
                    did_escalate = (content.strip().lower() in {"yes", "true", "urgent", "1", "escalate"})
                    _METRICS["ESCALATION_DECISIONS"].labels(
                        decision="escalated" if did_escalate else "not_escalated",
                    ).inc()

            done = step_num >= 3
            if done:
                log_data = {
                    "episode": self._state.episode_count,
                    "difficulty": self._state.current.get("difficulty", "unknown"),
                    "total_reward": round(self._state.cumulative_reward, 2),
                    "classify": round(self._state.action_rewards.get("classify", 0.0), 2),
                    "reply": round(self._state.action_rewards.get("reply", 0.0), 2),
                    "escalate": round(self._state.action_rewards.get("escalate", 0.0), 2),
                }
                LOGGER.info(json.dumps(log_data))

                if _ensure_metrics():
                    _METRICS["EPISODE_COUNT"].inc()
                    _METRICS["EPISODE_SCORE"].labels(
                        difficulty=self._state.current.get("difficulty", "unknown"),
                    ).observe(self._state.cumulative_reward)
                    _METRICS["ACTIVE_EPISODES"].dec()

            # Record step latency
            step_elapsed = _time.monotonic() - step_start
            if _ensure_metrics():
                _METRICS["STEP_LATENCY"].labels(
                    action_type=action.action_type,
                    difficulty=self._state.current.get("difficulty", "unknown"),
                ).observe(step_elapsed)

            # Enrich trace span
            if _span is not None:
                try:
                    _span.set_attribute("env.reward", reward_value)
                    _span.set_attribute("env.cumulative_reward", self._state.cumulative_reward)
                    _span.set_attribute("env.done", done)
                    _span.end()
                except Exception:
                    pass

            return self._make_obs(reward=reward_value, done=done)
        except Exception as exc:  # pragma: no cover
            self._debug_log(f"Step error: {exc}")
            self._state.history.append(f"error: {exc}")
            LOGGER.exception("Environment step failed")
            if _span is not None:
                try:
                    _span.set_attribute("error", True)
                    _span.set_attribute("error.message", str(exc))
                    _span.end()
                except Exception:
                    pass
            raise PipelineError(
                "Environment step execution failed",
                details={"exception": str(exc), "step_count": self._state.step_count},
            ) from exc

    @property
    def state(self) -> Dict[str, Any]:
        """Return current episode state for debug / introspection.

        NOTE: scenario label and requires_escalation are intentionally excluded.
        The /state endpoint is publicly accessible and must not expose answer keys.
        """
        return {
            "episode_count": self._state.episode_count,
            "step_count": self._state.step_count,
            "difficulty": self._state.current.get("difficulty", "unknown"),
            "cumulative_reward": self._state.cumulative_reward,
            "action_rewards": dict(self._state.action_rewards),
            "history": list(self._state.history),
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self._state.episode_count,
            "scenario": {
                "label": self._state.current.get("label"),
                "difficulty": self._state.current.get("difficulty"),
                "urgency": self._state.current.get("urgency"),
            },
            "performance": {
                "cumulative_reward": self._state.cumulative_reward,
                "reward_breakdown": dict(self._state.action_rewards),
                "step_details": self._state.step_details,
            },
            "history": self._state.history,
        }