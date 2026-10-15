"""Prepare waypoint or missed-grasp starts before policy episodes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment import StateSnapshot
    from gym_environment import CubeStackGymEnvironment


MAXIMUM_PREPARATION_STEPS = 200
REQUIRED_SETTLED_STEPS = 5


@dataclass(frozen=True)
class RecoveryStartConfig:
    """Sampling ranges in metres relative to orange's settled center."""

    probability: float = 0.0
    xy_offset_range: tuple[float, float] = (0.03, 0.05)
    height_offset_range: tuple[float, float] = (0.03, 0.05)
    closed_gripper_probability: float = 0.5

    def __post_init__(self) -> None:
        for name in ("probability", "closed_gripper_probability"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"Recovery {name} must be finite and between 0 and 1.")
        for name in ("xy_offset_range", "height_offset_range"):
            values = getattr(self, name)
            try:
                valid = len(values) == 2 and all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and np.isfinite(value)
                    for value in values
                ) and 0.0 < values[0] <= values[1]
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(
                    f"Recovery {name} must contain two ordered, finite, positive values."
                )
            object.__setattr__(self, name, tuple(float(value) for value in values))

    def validate_waypoint_height(self, height: float) -> None:
        if self.probability > 0.0 and self.height_offset_range[1] >= height:
            raise ValueError("Recovery height must be below the orange waypoint height.")


def _move_to_start_pose(
    environment: CubeStackGymEnvironment,
    offset: np.ndarray,
    *,
    gripper_command: float = 1.0,
    cube_anchor: np.ndarray | None = None,
    label: str = "Waypoint",
) -> StateSnapshot:
    """Physically settle at a pose, retaining the adapter's accepted target."""
    adapter = environment.action_adapter
    simulation = environment.simulation
    reward_config = environment.reward_config
    pose_tolerance = min(0.005, reward_config.approach_orange_waypoint_tolerance)
    target_tolerance = min(0.001, reward_config.approach_orange_waypoint_tolerance)
    gripper_target = (
        adapter.config.open_gripper_target
        if gripper_command > 0.0 else adapter.config.closed_gripper_target
    )
    settled_steps = 0
    state = simulation.get_state()

    for _ in range(MAXIMUM_PREPARATION_STEPS):
        anchor = np.asarray(state["orange_position"]) if cube_anchor is None else cube_anchor
        target = anchor + offset
        action = np.array([0.0, 0.0, 0.0, gripper_command], dtype=np.float32)
        action[:3] = np.clip(
            (target - adapter.current_target_gripper_position)
            / adapter.config.maximum_position_delta,
            -1.0,
            1.0,
        )
        state = adapter.step(action).state
        if simulation.is_failure() or state["confirmed_grasp_seen"]:
            raise RuntimeError(
                f"{label} preparation failed: a cube fell off the table "
                "or orange was grasped before the policy episode started."
            )

        # Follow gravity settling for the waypoint. Recovery targets stay
        # fixed so incidental cube contact cannot make preparation chase it.
        anchor = np.asarray(state["orange_position"]) if cube_anchor is None else cube_anchor
        target = anchor + offset
        pose_error = float(np.linalg.norm(state["gripper_position"] - target))
        target_error = float(np.linalg.norm(
            adapter.current_target_gripper_position - target
        ))
        gripper_error = abs(float(state["joint_positions"][-1]) - gripper_target)
        if (
            pose_error <= pose_tolerance
            and target_error <= target_tolerance
            and gripper_error <= 0.01
            and state["orange_touches_table"]
        ):
            settled_steps += 1
        else:
            settled_steps = 0
        if settled_steps >= REQUIRED_SETTLED_STEPS:
            return state

    raise RuntimeError(
        f"{label} preparation did not settle after {MAXIMUM_PREPARATION_STEPS} "
        f"steps: gripper error={pose_error:.6f} m, "
        f"accepted target error={target_error:.6f} m, "
        f"target={target.tolist()}, "
        f"accepted target={adapter.current_target_gripper_position.tolist()}."
    )


def prepare_waypoint_start(environment: CubeStackGymEnvironment) -> StateSnapshot:
    """Choose and prepare one initial situation before recording policy data."""
    config = environment.recovery_start_config
    waypoint_height = environment.reward_config.approach_orange_height_offset
    state = _move_to_start_pose(
        environment, np.array([0.0, 0.0, waypoint_height]),
    )
    start_type = "waypoint"
    # Use Gym's seeded RNG independently of the simulator's cube placement RNG.
    # Probability zero preserves the previous waypoint-only reset path.
    if config.probability > 0.0 and environment.np_random.random() < config.probability:
        rng = environment.np_random
        radius = rng.uniform(*config.xy_offset_range)
        angle = rng.uniform(-np.pi, np.pi)
        height = rng.uniform(*config.height_offset_range)
        closed = rng.random() < config.closed_gripper_probability
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle), waypoint_height])
        cube_anchor = np.asarray(state["orange_position"]).copy()
        # Clear the cube laterally at waypoint height before descending beside
        # it. Close while high, where an empty gripper can settle freely.
        gripper_command = -1.0 if closed else 1.0
        _move_to_start_pose(
            environment, offset,
            cube_anchor=cube_anchor, label="Recovery lateral",
        )
        if closed:
            _move_to_start_pose(
                environment, offset, gripper_command=gripper_command,
                cube_anchor=cube_anchor, label="Recovery closing",
            )
        offset[2] = height
        state = _move_to_start_pose(
            environment, offset, gripper_command=gripper_command,
            cube_anchor=cube_anchor, label="Recovery descent",
        )
        start_type = "recovery_closed" if closed else "recovery_open"

    # Every prepared episode passed through the waypoint. Preparation consumes
    # no policy steps or rewards; history is initialized after this returns.
    environment.reward_calculator.reset(
        state, orange_pregrasp_waypoint_reached=True,
        open_gripper_target=environment.action_adapter.config.open_gripper_target,
    )
    environment.previous_state = state
    environment.episode_step_count = 0
    environment.episode_start_type = start_type
    return state
