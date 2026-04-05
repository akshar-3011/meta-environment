"""Environment orchestration with explicit state and dependency injection."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openenv.core import Environment
except ImportError:
    from pydantic import BaseModel as Environment

try:
    from ..core.config import get_config
    from ..core.exceptions import PipelineError
    from ..core.graders import CATEGORY_OPTIONS, RewardPolicy, RuleBasedRewardPolicy
    from ..core.logging_config import get_logger, setup_logging
    from ..core.models import WorkplaceAction, WorkplaceObservation
    from ..data import get_default_repository, ScenarioRepository
except ImportError:  # pragma: no cover
    from core.config import get_config
    from core.exceptions import PipelineError
    from core.graders import CATEGORY_OPTIONS, RewardPolicy, RuleBasedRewardPolicy
    from core.logging_config import get_logger, setup_logging
    from core.models import WorkplaceAction, WorkplaceObservation
    from data import get_default_repository, ScenarioRepository


setup_logging()
CFG = get_config()
LOGGER = get_logger(__name__)
DEBUG = CFG.environment.debug


def _debug_log(msg: str):
    if DEBUG:
        LOGGER.debug(msg)


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
        global DEBUG
        DEBUG = debug
        self._policy = reward_policy or RuleBasedRewardPolicy()
        self._scenario_repo = scenario_repository or get_default_repository()
        self._scenarios = self._scenario_repo.list_scenarios()
        # Instance-owned state — no sharing between concurrent sessions.
        self._state = EpisodeState()
        if self._scenarios:
            self._state.current = self._scenarios[0]

    def _next_scenario(self) -> Dict[str, Any]:
        idx = self._state.scenario_index % len(self._scenarios)
        self._state.scenario_index += 1
        scenario = self._scenarios[idx]
        _debug_log(f"Loaded scenario {idx}: {scenario['email'][:50]}...")
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
        _debug_log(
            f"Step {step_count} ({action.action_type}): "
            f"reward={reward_value:.3f}, {breakdown.get('explanation', '')}"
        )
        return reward_value

    def reset(self) -> WorkplaceObservation:
        _debug_log("=" * 60)
        _debug_log(f"RESET (episode {self._state.episode_count})")
        _debug_log("=" * 60)

        self._state.history = []
        self._state.step_count = 0
        self._state.action_rewards = {}
        self._state.cumulative_reward = 0.0
        self._state.step_details = []
        self._state.current = self._next_scenario()
        self._state.episode_count += 1

        return self._make_obs(reward=None, done=False)

    def step(self, action: WorkplaceAction) -> WorkplaceObservation:
        try:
            if action.action_type not in ["classify", "reply", "escalate"]:
                _debug_log(f"Invalid action type: {action.action_type}")
                return self._make_obs(reward=0.0, done=True)

            self._state.step_count += 1
            step_num = self._state.step_count

            content = action.content or ""
            content_preview = content[:50] + ("..." if len(content) > 50 else "")
            action_str = f"{action.action_type}: {content_preview}"
            self._state.history.append(action_str)
            _debug_log(f"Step {step_num}: {action_str}")

            reward_value = self._grade_step(action, step_num)
            self._state.action_rewards[action.action_type] = reward_value
            self._state.cumulative_reward += reward_value

            done = step_num >= 3
            return self._make_obs(reward=reward_value, done=done)
        except Exception as exc:  # pragma: no cover
            _debug_log(f"Step error: {exc}")
            self._state.history.append(f"error: {exc}")
            LOGGER.exception("Environment step failed")
            raise PipelineError(
                "Environment step execution failed",
                details={"exception": str(exc), "step_count": self._state.step_count},
            ) from exc

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