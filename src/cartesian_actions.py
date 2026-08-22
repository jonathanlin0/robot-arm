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
        raise NotImplementedError(
            "Cartesian action conversion has not been implemented yet."
        )
