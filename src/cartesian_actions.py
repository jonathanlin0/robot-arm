from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from environment import CubeStackEnvironment, StateSnapshot
from kinematics import ToolAxisIKResult


CARTESIAN_ACTION_SIZE = 4


@dataclass(frozen=True)
class CartesianActionConfig:
    """Physical interpretation of one normalized policy action."""

    maximum_position_delta: float = 0.01 # multiply output by this since output is restricted to [-1, 1]. so, this will be 0.01 m/action. assuming 20 actions / second leads to 0.2 m/s as max speed
    closed_gripper_target: float = -0.1
    open_gripper_target: float = 0.5

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


@dataclass(frozen=True)
class CartesianActionResult:
    """Simulation result and IK diagnostics for one policy action."""

    state: StateSnapshot
    target_gripper_position: np.ndarray
    ik_result: ToolAxisIKResult


class CartesianActionAdapter:
    """Convert normalized Cartesian policy actions into joint targets.

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
