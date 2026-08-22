from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from environment import (
    ARM_JOINT_NAMES,
    PHYSICS_STEPS_PER_ACTION,
    CubeStackEnvironment,
    StateSnapshot,
)
from kinematics import (
    ToolAxisIKResult,
    WORLD_DOWN,
    solve_position_and_tool_axis_ik,
)

# (dx, dy, dz, gripper)
CARTESIAN_ACTION_SIZE = 4


@dataclass(frozen=True)
class CartesianActionConfig:
    """Physical interpretation of one normalized policy action."""

    maximum_position_delta: float = 0.01 # multiply output by this since output is restricted to [-1, 1]. so, this will be 0.01 m/action. assuming 20 actions / second leads to 0.2 m/s as max speed
    closed_gripper_target: float = -0.1
    open_gripper_target: float = 0.5
    target_tool_axis: tuple[float, float, float] = WORLD_DOWN

    # these are loose bounds and are temporary placeholders. grabbed from valid cube spawn locations plus a little margin.
    # modify these in future if unreachable by arm
    workspace_lower_bounds: tuple[float, float, float] = (
        0.10,
        -0.25,
        0.02,
    )
    workspace_upper_bounds: tuple[float, float, float] = (
        0.45,
        0.25,
        0.35,
    )

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.maximum_position_delta)
            or self.maximum_position_delta <= 0.0
        ):
            raise ValueError(
                "maximum_position_delta must be finite and greater than "
                "zero."
            )

        gripper_targets = (
            self.closed_gripper_target,
            self.open_gripper_target,
        )
        if not np.all(np.isfinite(gripper_targets)):
            raise ValueError("gripper targets must be finite.")
        if self.closed_gripper_target >= self.open_gripper_target:
            raise ValueError(
                "closed gripper target must be less than open gripper "
                "target."
            )

        # converted to temporary numpy arrs for easy validation
        workspace_lower_bounds_np = np.asarray(
            self.workspace_lower_bounds,
            dtype=float,
        )
        workspace_upper_bounds_np = np.asarray(
            self.workspace_upper_bounds,
            dtype=float,
        )
        if (
            workspace_lower_bounds_np.shape != (3,)
            or workspace_upper_bounds_np.shape != (3,)
        ):
            raise ValueError("workspace bounds must each contain XYZ values.")
        if not np.all(
            np.isfinite(
                np.concatenate(
                    (workspace_lower_bounds_np, workspace_upper_bounds_np)
                )
            )
        ):
            raise ValueError("workspace bounds must be finite.")
        if np.any(workspace_lower_bounds_np >= workspace_upper_bounds_np):
            raise ValueError(
                "workspace lower bounds must be below upper bounds."
            )

        target_tool_axis = np.asarray(self.target_tool_axis, dtype=float)
        if target_tool_axis.shape != (3,):
            raise ValueError("target_tool_axis must contain XYZ values.")
        if (
            not np.all(np.isfinite(target_tool_axis))
            or np.linalg.norm(target_tool_axis) == 0.0
        ):
            raise ValueError(
                "target_tool_axis must be finite and have nonzero length."
            )


@dataclass(frozen=True)
class CartesianActionResult:
    """Simulation result and IK diagnostics for one policy action."""

    state: StateSnapshot
    target_gripper_position: np.ndarray
    ik_result: ToolAxisIKResult


class CartesianActionAdapter:
    """Translate normalized Cartesian actions and execute them.

    The action has four values in ``[-1, 1]``:

    ``[dx, dy, dz, gripper_command]``

    The first three values become a bounded XYZ displacement from the
    current gripper position. The final value maps to an absolute gripper
    position between the configured closed and open targets.
    """

    def __init__(
        self,
        environment: CubeStackEnvironment,
        config: CartesianActionConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or CartesianActionConfig()

    def step(self, action: ArrayLike) -> CartesianActionResult:
        """Apply one normalized Cartesian action to the simulation."""
        normalized_action = np.asarray(action, dtype=float)
        expected_shape = (CARTESIAN_ACTION_SIZE,)

        if normalized_action.shape != expected_shape:
            raise ValueError(
                f"action must have shape {expected_shape}; received "
                f"{normalized_action.shape}."
            )
        if not np.all(np.isfinite(normalized_action)):
            raise ValueError("action must contain only finite values.")

        applied_action = np.clip(normalized_action, -1.0, 1.0)
        current_state = self.environment.get_state()

        position_delta = (
            applied_action[:3] * self.config.maximum_position_delta
        )
        target_gripper_position = np.clip(
            current_state["gripper_position"] + position_delta,
            self.config.workspace_lower_bounds,
            self.config.workspace_upper_bounds,
        )

        # convert gripper range from [-1, 1] to [0, 1]
        gripper_fraction = (applied_action[3] + 1.0) / 2.0
        # convert gripper fraction to real joint action (angle)
        gripper_target = self.config.closed_gripper_target + (
            gripper_fraction
            * (
                self.config.open_gripper_target
                - self.config.closed_gripper_target
            )
        )

        ik_result = solve_position_and_tool_axis_ik(
            model=self.environment.model,
            initial_joint_positions=current_state["joint_positions"][
                : len(ARM_JOINT_NAMES) # essentially to remove the gripper joint position
            ],
            target_position=target_gripper_position,
            target_tool_axis=self.config.target_tool_axis,
            stop_when_position_converged=True, # note: this makes function return when position converged, even if gripper angle didn't
            minimum_iterations=1,
        )

        if ik_result.position_converged:
            joint_targets = np.concatenate(
                (ik_result.joint_positions, [gripper_target])
            )
            next_state = self.environment.step_joint_targets(joint_targets)
        else:
            # Treat one policy action atomically. If its Cartesian target is
            # unreachable, preserve the previous six actuator commands while
            # still advancing the normal amount of simulated time.
            self.environment.step_physics(PHYSICS_STEPS_PER_ACTION)
            next_state = self.environment.get_state()

        return CartesianActionResult(
            state=next_state,
            target_gripper_position=target_gripper_position.copy(),
            ik_result=ik_result,
        )
