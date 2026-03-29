"""
server/workplace_env_environment.py — Production-Grade WorkplaceEnvironment.

ARCHITECTURE:
  1. Module-level singleton (_SHARED_STATE) maintains state across stateless
     HTTP requests from OpenEnv's HTTP handler.
     
  2. Rich episode tracking with cumulative rewards, difficulty progression.
  
  3. Enhanced grading with component breakdown and penalty system.
  
  4. Deterministic scenario cycling with difficulty-based progression.
  
  5. Debug logging support for transparency and analysis.
"""

from openenv.core import Environment
from typing import Optional, Dict, Any
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import WorkplaceObservation, WorkplaceAction
from graders.grader import (
    calculate_step_reward,
    CATEGORY_OPTIONS,
)
from data import SCENARIOS

# Default env config
DEBUG = False


# ---------------------------------------------------------------------------
# Module-level singleton — persistent state across HTTP requests
# ---------------------------------------------------------------------------
_SHARED_STATE: Dict[str, Any] = {
    "scenario_index": 0,        # Cycle through scenarios deterministically
    "history": [],              # Action history for this episode
    "step_count": 0,            # Steps in current episode
    "current": SCENARIOS[0],    # Current scenario dict
    "episode_count": 0,         # Total episodes seen
    "action_rewards": {},       # Track reward for each action
    "cumulative_reward": 0.0,   # Episode cumulative reward
    "step_details": [],         # Detailed breakdown of each step
}


def _debug_log(msg: str):
    """Print debug message if DEBUG mode enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


class WorkplaceEnvironment(Environment):
    """
    Production-grade OpenEnv environment for customer support workflow.
    
    Workflow (3 steps):
      1. CLASSIFY: Categorize email (refund/complaint/query)
      2. REPLY: Generate appropriate response
      3. ESCALATE: Decide if escalation needed
    
    Features:
      - Difficulty-based scenario progression
      - Weighted composite rewards
      - Penalty system for inconsistencies
      - Rich metadata in observations
      - Deterministic, interpretable grading
    
    State persistence:
      - Uses module-level singleton for HTTP statelessness
      - WebSocket connections work naturally (session-scoped)
    """

    def __init__(self, debug: bool = False):
        global DEBUG
        DEBUG = debug
        self._s = _SHARED_STATE

    # ======================================================================
    # Helpers
    # ======================================================================

    def _next_scenario(self) -> Dict[str, Any]:
        """
        Deterministically cycle through scenarios.
        Preserves difficulty progression for learning agents.
        """
        idx = self._s["scenario_index"] % len(SCENARIOS)
        self._s["scenario_index"] += 1
        scenario = SCENARIOS[idx]
        _debug_log(f"Loaded scenario {idx}: {scenario['email'][:50]}...")
        return scenario

    def _make_obs(
        self,
        reward: Optional[float] = None,
        done: bool = False,
    ) -> WorkplaceObservation:
        """Construct WorkplaceObservation with rich metadata."""
        scenario = self._s["current"]
        
        return WorkplaceObservation(
            email=scenario["email"],
            category_options=CATEGORY_OPTIONS,
            history=list(self._s["history"]),
            reward=reward,
            done=done,
            # Difficulty metadata (helps agent learn)
            scenario_difficulty=scenario.get("difficulty", "easy"),
            urgency=scenario.get("urgency", "low"),
            sentiment=scenario.get("sentiment", "neutral"),
            complexity_score=scenario.get("complexity", 1),
            # Full scenario metadata for analysis
            scenario_metadata={
                "label": scenario["label"],
                "requires_escalation": scenario.get("requires_escalation", False),
                "min_reply_length": scenario.get("min_reply_length", 20),
            },
        )

    def _grade_step(
        self,
        action: WorkplaceAction,
        step_count: int,
    ) -> float:
        """
        Grade a single action step with rich feedback.
        Applies weighted composite reward system.
        """
        scenario = self._s["current"]
        reward_value, breakdown = calculate_step_reward(
            action_type=action.action_type,
            content=action.content,
            actual_category=scenario["label"],
            step_count=step_count,
            scenario_difficulty=scenario.get("difficulty", "easy"),
            min_reply_length=scenario.get("min_reply_length", 30),
            previous_actions=self._s["action_rewards"],
        )
        
        # Store breakdown for analysis
        self._s["step_details"].append(breakdown)
        _debug_log(
            f"Step {step_count} ({action.action_type}): "
            f"reward={reward_value:.3f}, {breakdown.get('explanation', '')}"
        )
        
        return reward_value

    # ======================================================================
    # OpenEnv Interface
    # ======================================================================

    def reset(self) -> WorkplaceObservation:
        """Reset for a new episode."""
        _debug_log("=" * 60)
        _debug_log(f"RESET (episode {self._s['episode_count']})")
        _debug_log("=" * 60)
        
        self._s["history"] = []
        self._s["step_count"] = 0
        self._s["action_rewards"] = {}
        self._s["cumulative_reward"] = 0.0
        self._s["step_details"] = []
        self._s["current"] = self._next_scenario()
        self._s["episode_count"] += 1
        
        return self._make_obs(reward=None, done=False)

    def step(self, action: WorkplaceAction) -> WorkplaceObservation:
        """
        Execute one step in the workflow.
        
        Always returns after step 3 (done=True).
        """
        # Validate action
        if action.action_type not in ["classify", "reply", "escalate"]:
            _debug_log(f"Invalid action type: {action.action_type}")
            return self._make_obs(reward=0.0, done=True)
        
        # Increment step counter
        self._s["step_count"] += 1
        step_num = self._s["step_count"]
        
        # Record action in history
        action_str = f"{action.action_type}: {action.content[:50]}..."
        self._s["history"].append(action_str)
        _debug_log(f"Step {step_num}: {action_str}")
        
        # Grade the action
        reward_value = self._grade_step(action, step_num)
        
        # Store reward for this action type
        self._s["action_rewards"][action.action_type] = reward_value
        
        # Accumulate reward
        self._s["cumulative_reward"] += reward_value
        
        # Episode ends after 3 steps
        done = step_num >= 3
        
        if done:
            _debug_log(
                f"Episode complete. "
                f"Cumulative reward: {self._s['cumulative_reward']:.3f}, "
                f"Step details: {len(self._s['step_details'])} steps"
            )
        
        return self._make_obs(reward=reward_value, done=done)

    def state(self) -> Dict[str, Any]:
        """
        Return current episode state (for analysis/debugging).
        Must return a plain dict (not Pydantic model).
        """
        return {
            "episode_count": self._s["episode_count"],
            "step_count": self._s["step_count"],
            "scenario_label": self._s["current"].get("label", ""),
            "difficulty": self._s["current"].get("difficulty", "unknown"),
            "cumulative_reward": self._s["cumulative_reward"],
            "action_rewards": dict(self._s["action_rewards"]),
            "history": list(self._s["history"]),
        }

    # ======================================================================
    # Optional: Debugging / Analysis
    # ======================================================================

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return detailed summary of completed episode."""
        return {
            "episode_id": self._s["episode_count"],
            "scenario": {
                "label": self._s["current"].get("label"),
                "difficulty": self._s["current"].get("difficulty"),
                "urgency": self._s["current"].get("urgency"),
            },
            "performance": {
                "cumulative_reward": self._s["cumulative_reward"],
                "reward_breakdown": dict(self._s["action_rewards"]),
                "step_details": self._s["step_details"],
            },
            "history": self._s["history"],
        }
    