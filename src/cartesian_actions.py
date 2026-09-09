from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from environment import (
    ARM_JOINT_NAMES,
    DEFAULT_OPEN_GRIPPER_POSITION,
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
# Try the complete bounded displacement first, then progressively shorter
# displacements along the same Cartesian direction.
IK_BACKTRACKING_SCALES = (1.0, 0.5, 0.25, 0.125)


@dataclass(frozen=True)
class CartesianActionConfig:
    """Physical interpretation of one normalized policy action."""

    # Each policy action spans 50 ms. At 20 actions/second this permits at
    # most 0.05 m/s of commanded motion independently along each axis.
    maximum_position_delta: float = 0.0025
    closed_gripper_target: float = -0.1
    open_gripper_target: float = DEFAULT_OPEN_GRIPPER_POSITION
    close_gripper_command_threshold: float = -0.5
    open_gripper_command_threshold: float = 0.5
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

        gripper_command_thresholds = (
            self.close_gripper_command_threshold,
            self.open_gripper_command_threshold,
        )
        if not np.all(np.isfinite(gripper_command_thresholds)):
            raise ValueError("gripper command thresholds must be finite.")
        if not (
            -1.0 <= self.close_gripper_command_threshold
            < self.open_gripper_command_threshold
            <= 1.0
        ):
            raise ValueError(
                "gripper command thresholds must satisfy "
                "-1 <= close < open <= 1."
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
    """Simulation result and IK diagnostics for one policy action.

    ``target_gripper_position`` is the persistent target committed after the
    action. ``attempted_target_gripper_position`` is the final absolute point
    passed to IK; the two differ only when every backtracking attempt fails.
    """

    state: StateSnapshot
    target_gripper_position: np.ndarray
    ik_result: ToolAxisIKResult
    attempted_target_gripper_position: np.ndarray | None = None


class CartesianActionAdapter:
    """Translate normalized Cartesian actions and execute them.

    The action has four values in ``[-1, 1]``:

    ``[dx, dy, dz, gripper_command]``

    The first three values become a bounded XYZ displacement from the last
    accepted Cartesian target. The final value is a persistent gripper
    command: sufficiently negative closes, sufficiently positive opens, and
    values between the two thresholds retain the previous actuator target.

    Call ``reset()`` whenever the underlying simulation is reset. The Gym
    wrapper does this automatically; direct users must do it themselves.
    """

    def __init__(
        self,
        environment: CubeStackEnvironment,
        config: CartesianActionConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or CartesianActionConfig()
        self._current_target_gripper_position: np.ndarray | None = None
        self._previous_target_gripper_position: np.ndarray | None = None

    @property
    def current_target_gripper_position(self) -> np.ndarray:
        """Return a copy of the persistent Cartesian target."""
        if self._current_target_gripper_position is None:
            raise RuntimeError(
                "CartesianActionAdapter.reset() must be called before "
                "reading its target."
            )
        return self._current_target_gripper_position.copy()

    @property
    def previous_target_gripper_position(self) -> np.ndarray:
        """Return a copy of the target from before the most recent action."""
        if self._previous_target_gripper_position is None:
            raise RuntimeError(
                "CartesianActionAdapter.reset() must be called before "
                "reading its previous target."
            )
        return self._previous_target_gripper_position.copy()

    def reset(self, initial_state: StateSnapshot | None = None) -> None:
        """Initialize both Cartesian targets from the measured gripper pose."""
        state = (
            self.environment.get_state()
            if initial_state is None
            else initial_state
        )
        initial_target = np.asarray(
            state["gripper_position"],
            dtype=float,
        )

        self._current_target_gripper_position = initial_target.copy()
        self._previous_target_gripper_position = initial_target.copy()

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
        if self._current_target_gripper_position is None:
            # Direct users of the adapter may not have a Gym wrapper to call
            # reset(). Initialize lazily from the first measured state.
            self.reset(current_state)

        assert self._current_target_gripper_position is not None
        previous_target_gripper_position = (
            self._current_target_gripper_position.copy()
        )

        position_delta = (
            applied_action[:3] * self.config.maximum_position_delta
        )
        requested_target_gripper_position = np.clip(
            previous_target_gripper_position + position_delta,
            self.config.workspace_lower_bounds,
            self.config.workspace_upper_bounds,
        )
        bounded_position_delta = (
            requested_target_gripper_position
            - previous_target_gripper_position
        )

        gripper_command = applied_action[3]
        if gripper_command <= self.config.close_gripper_command_threshold:
            gripper_target = self.config.closed_gripper_target
        elif gripper_command >= self.config.open_gripper_command_threshold:
            gripper_target = self.config.open_gripper_target
        else:
            gripper_target = float(current_state["gripper_target"])

        ik_result: ToolAxisIKResult | None = None
        target_gripper_position = requested_target_gripper_position
        for position_delta_scale in IK_BACKTRACKING_SCALES:
            target_gripper_position = (
                previous_target_gripper_position
                + position_delta_scale * bounded_position_delta
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

            # Scales are ordered from largest to smallest, so the first
            # converged result applies the largest feasible displacement.
            if ik_result.position_converged:
                break

        assert ik_result is not None

        if ik_result.position_converged:
            joint_targets = np.concatenate(
                (ik_result.joint_positions, [gripper_target])
            )
            next_state = self.environment.step_joint_targets(joint_targets)
            current_target_gripper_position = target_gripper_position
        else:
            # Treat one policy action atomically. If its Cartesian target is
            # unreachable, preserve the previous six actuator commands while
            # still advancing the normal amount of simulated time. Keep the
            # last accepted Cartesian target as well, preventing target
            # windup toward an unreachable point.
            self.environment.step_physics(PHYSICS_STEPS_PER_ACTION)
            next_state = self.environment.get_state()
            current_target_gripper_position = (
                previous_target_gripper_position
            )

        self._previous_target_gripper_position = (
            previous_target_gripper_position.copy()
        )
        self._current_target_gripper_position = (
            current_target_gripper_position.copy()
        )

        return CartesianActionResult(
            state=next_state,
            target_gripper_position=(
                current_target_gripper_position.copy()
            ),
            ik_result=ik_result,
            attempted_target_gripper_position=(
                target_gripper_position.copy()
            ),
        )
