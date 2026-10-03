from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike

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
from waypoint_start import RecoveryStartConfig, prepare_waypoint_start


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
        start_position: ArrayLike | None = None,
        start_position_half_range: ArrayLike = (0.0, 0.0, 0.0),
        start_at_orange_waypoint: bool = False, # TEMP
        recovery_start_probability: float = 0.0,
        recovery_xy_offset_range: tuple[float, float] = (0.03, 0.05),
        recovery_height_offset_range: tuple[float, float] = (0.03, 0.05),
        recovery_closed_gripper_probability: float = 0.5,
    ) -> None:
        super().__init__()
        if seed is not None:
            self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)

        if maximum_episode_steps < 1:
            raise ValueError("maximum_episode_steps must be at least 1.")

        self.reward_config = reward_config or StackRewardConfig()
        action_config = action_config or CartesianActionConfig()
        self.recovery_start_config = RecoveryStartConfig(
            probability=recovery_start_probability,
            xy_offset_range=recovery_xy_offset_range,
            height_offset_range=recovery_height_offset_range,
            closed_gripper_probability=recovery_closed_gripper_probability,
        )
        if start_at_orange_waypoint:
            self.recovery_start_config.validate_waypoint_height(
                self.reward_config.approach_orange_height_offset
            )
        self.simulation = CubeStackEnvironment(
            scene_path=scene_path,
            seed=seed,
            spawn_config=spawn_config,
            success_config=success_config,
            start_position=start_position,
            start_position_half_range=start_position_half_range,
            require_downward=action_config.require_downward,
            target_tool_yaw=action_config.target_tool_yaw,
            tool_yaw_tolerance=action_config.tool_yaw_tolerance,
            orange_waypoint_height_offset=(
                self.reward_config.approach_orange_height_offset
            ),
            orange_waypoint_tolerance=(
                self.reward_config.approach_orange_waypoint_tolerance
            ),
        )
        self.action_adapter = CartesianActionAdapter(
            self.simulation,
            action_config,
        )
        self.observation_builder = PrivilegedObservationBuilder(
            self.simulation
        )
        # self.reward_config = reward_config or StackRewardConfig()
        self.reward_calculator = StackRewardCalculator(
            self.reward_config,
            self.simulation.success_config,
        )
        self.maximum_episode_steps = maximum_episode_steps
        self.start_at_orange_waypoint = start_at_orange_waypoint # TEMP
        self.episode_start_type = "home"
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
        self.action_adapter.reset(initial_state)
        self.reward_calculator.reset(
            initial_state,
            open_gripper_target=self.action_adapter.config.open_gripper_target,
        )
        self.episode_step_count = 0
        self.episode_start_type = "home"
        self.previous_state = initial_state

        # TEMP
        if self.start_at_orange_waypoint:
            initial_state = prepare_waypoint_start(self)

        observation = self.observation_builder.build(initial_state)
        return observation, {
            "episode_start_type": self.episode_start_type,
            "target_gripper_position": (
                self.action_adapter.current_target_gripper_position
            ),
        }

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
            "episode_start_type": self.episode_start_type,
            # The accepted command after workspace clipping and IK
            # backtracking; a failed action preserves the earlier target.
            "target_gripper_position": (
                action_result.target_gripper_position.copy()
            ),
            "is_success": succeeded,
            "is_failure": failed,
            "stack_stable_time": float(self.simulation.stack_stable_time),
            "orange_fell_off_table": bool(
                current_state["orange_fell_off_table"]
            ),
            "blue_fell_off_table": bool(
                current_state["blue_fell_off_table"]
            ),
            "orange_currently_held": bool(
                current_state["orange_currently_held"]
            ),
            "orange_grasp_hold_time": float(
                current_state["orange_grasp_hold_time"]
            ),
            "orange_pregrasp_waypoint_reached": (
                self.reward_calculator.orange_pregrasp_waypoint_reached
            ),
            "reward_components": reward_result.components,
            "ik_position_converged": (
                action_result.ik_result.position_converged
            ),
            "ik_tool_axis_converged": (
                action_result.ik_result.tool_axis_converged
            ),
            "ik_require_downward": self.action_adapter.config.require_downward,
            "ik_tool_axis_error": float(action_result.ik_result.tool_axis_error),
            "ik_tool_yaw_converged": action_result.ik_result.tool_yaw_converged,
            "ik_tool_yaw_error": float(action_result.ik_result.tool_yaw_error),
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
