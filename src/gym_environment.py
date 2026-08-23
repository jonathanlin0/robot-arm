from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cartesian_actions import (
    CARTESIAN_ACTION_SIZE,
    CartesianActionAdapter,
    CartesianActionConfig,
)
from environment import (
    DEFAULT_SCENE_PATH,
    CubeStackEnvironment,
    StateSnapshot,
)
from observations import (
    PRIVILEGED_OBSERVATION_SIZE,
    PrivilegedObservationBuilder,
)
from randomization import CubeSpawnConfig
from rewards import StackRewardCalculator, StackRewardConfig
from success import StackSuccessConfig


class CubeStackGymEnvironment(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium interface for privileged-state cube-stacking training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        *,
        seed: int | None = None,
        maximum_episode_steps: int = 400,
        spawn_config: CubeSpawnConfig | None = None,
        success_config: StackSuccessConfig | None = None,
        action_config: CartesianActionConfig | None = None,
        reward_config: StackRewardConfig | None = None,
    ) -> None:
        super().__init__()

        if maximum_episode_steps < 1:
            raise ValueError("maximum_episode_steps must be at least 1.")

        self.simulation = CubeStackEnvironment(
            scene_path=scene_path,
            seed=seed,
            spawn_config=spawn_config,
            success_config=success_config,
        )
        self.action_adapter = CartesianActionAdapter(
            self.simulation,
            action_config,
        )
        self.observation_builder = PrivilegedObservationBuilder(
            self.simulation
        )
        self.reward_config = reward_config or StackRewardConfig()
        self.reward_calculator = StackRewardCalculator(
            self.reward_config,
            self.simulation.success_config,
        )
        self.maximum_episode_steps = maximum_episode_steps
        self.episode_step_count = 0
        self.previous_state: StateSnapshot | None = None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(CARTESIAN_ACTION_SIZE,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(PRIVILEGED_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode and return its initial observation and info."""
        super().reset(seed=seed)

        initial_state = self.simulation.reset(seed=seed)
        self.reward_calculator.reset(initial_state)
        self.episode_step_count = 0
        self.previous_state = initial_state

        observation = self.observation_builder.build(initial_state)
        return observation, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one policy action and return the Gymnasium transition."""
        if self.previous_state is None:
            raise RuntimeError(
                "reset() must be called before the first step()."
            )

        action_result = self.action_adapter.step(action)
        current_state = action_result.state
        failed = self.simulation.is_failure()
        succeeded = self.simulation.is_success() and not failed

        reward_result = self.reward_calculator.calculate(
            previous_state=self.previous_state,
            action=action,
            action_result=action_result,
            succeeded=succeeded,
        )

        self.episode_step_count += 1
        terminated = self.simulation.is_terminated()
        truncated = (
            self.episode_step_count >= self.maximum_episode_steps
            and not terminated
        )

        observation = self.observation_builder.build(current_state)
        self.previous_state = current_state

        info = {
            "is_success": succeeded,
            "is_failure": failed,
            "orange_fell_off_table": bool(
                current_state["orange_fell_off_table"]
            ),
            "blue_fell_off_table": bool(
                current_state["blue_fell_off_table"]
            ),
            "reward_components": reward_result.components,
            "ik_position_converged": (
                action_result.ik_result.position_converged
            ),
            "ik_tool_axis_converged": (
                action_result.ik_result.tool_axis_converged
            ),
            "safe_lift_completed": (
                self.reward_calculator.safe_lift_completed
            ),
            "hover_alignment_completed": (
                self.reward_calculator.hover_alignment_completed
            ),
        }

        return (
            observation,
            reward_result.total,
            terminated,
            truncated,
            info,
        )
