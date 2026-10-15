"""Proposed and issued commands, distinct from measured robot arrival."""

from dataclasses import dataclass
from schemas.arrays import JointVector, Vec3, PolicyAction
from schemas.state import RobotState, CubeStates

@dataclass
class PlannedCommand:
    """Proposed joint/Cartesian targets, not yet committed as issued commands."""

    joint_targets: JointVector
    cartesian_target: Vec3
    ik_diagnostics: dict[str, object]


@dataclass
class CommandReceipt:
    """Actual joint targets transmitted, without asserting measured arrival."""

    timestamp: float
    joint_targets: JointVector


@dataclass
class StepResult:
    """Inputs, policy action, and command information from one iteration."""

    robot_state: RobotState
    cube_states: CubeStates
    action: PolicyAction
    command: PlannedCommand
    receipt: CommandReceipt
