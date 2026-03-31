# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workplace Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .core.models import WorkplaceAction, WorkplaceObservation


class WorkplaceEnv(
    EnvClient[WorkplaceAction, WorkplaceObservation, State]
):
    """
    Client for the Workplace Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with WorkplaceEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.email)
        ...
        ...     result = client.step(WorkplaceAction(action_type="classify", content="refund"))
        ...     print(result.observation.history)
    """

    def _step_payload(self, action: WorkplaceAction) -> Dict:
        """Convert WorkplaceAction to JSON payload for step message."""
        return {
            "action_type": action.action_type,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WorkplaceObservation]:
        """Parse server response into StepResult[WorkplaceObservation]."""
        obs_data = payload.get("observation", {})
        observation = WorkplaceObservation(
            email=obs_data.get("email", ""),
            category_options=obs_data.get("category_options", []),
            history=obs_data.get("history", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            scenario_difficulty=obs_data.get("scenario_difficulty", ""),
            urgency=obs_data.get("urgency", ""),
            sentiment=obs_data.get("sentiment", ""),
            complexity_score=obs_data.get("complexity_score", 0),
            scenario_metadata=obs_data.get("scenario_metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict) -> State:     
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )