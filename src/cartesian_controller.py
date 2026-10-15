"""Shared Cartesian action interpretation and transactional command acceptance."""

import mujoco
import numpy as np

from kinematics import IKConvergenceError, forward_kinematics, joint_limits, joint_vector, solve_ik
from schemas import CommandReceipt, PlannedCommand, PolicyAction, RobotState
from schemas.config import CartesianActionConfig


class CartesianController:
    """Convert Cartesian deltas into limited joints for simulation or hardware."""

    def __init__(self, robot_model: mujoco.MjModel, config: CartesianActionConfig) -> None:
        self.robot_model = robot_model
        self.config = config
        self.prev_accepted_target = None
        self.gripper_target = None

    def reset(self, robot_state: RobotState) -> None:
        """Initialize command history from measured state without issuing motion."""
        self.prev_accepted_target = np.asarray(robot_state.gripper_position, dtype=np.float64).copy()
        self.gripper_target = float(robot_state.commanded_joint_targets[-1])

    def compute(self, action: PolicyAction, robot_state: RobotState) -> PlannedCommand:
        """Propose IK targets without changing state until a send is accepted."""
        if self.prev_accepted_target is None or self.gripper_target is None:
            raise RuntimeError("Reset the CartesianController with measured state before computing actions.")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("action must contain four finite [dx, dy, dz, gripper] values.")
        values = np.clip(values, -1.0, 1.0)
        gripper = self.gripper_target
        if values[3] >= self.config.open_gripper_command_threshold:
            gripper = self.config.open_gripper_target
        elif values[3] <= self.config.close_gripper_command_threshold:
            gripper = self.config.closed_gripper_target
        initial = joint_vector(robot_state.joint_positions)
        lower, upper = joint_limits(self.robot_model)
        initial[-1] = np.clip(gripper, lower[-1], upper[-1])
        delta = values[:3] * self.config.maximum_position_delta
        attempts = []
        for scale in self.config.backtracking_scales:
            target = self.prev_accepted_target + scale * delta
            joints, outcome = solve_ik(self.robot_model, initial, target, self.config)
            attempts.append({"scale": float(scale), "target": target.tolist(), **outcome})
            if outcome["converged"]:
                return PlannedCommand(
                    joint_targets=joints, cartesian_target=target.copy(),
                    ik_diagnostics={"attempts": attempts, "accepted_scale": float(scale)},
                )
        raise IKConvergenceError({
            "attempts": attempts, "requested_target": (self.prev_accepted_target + delta).tolist(),
            "position_tolerance": self.config.position_tolerance,
            "tool_axis_tolerance": self.config.tool_axis_tolerance,
            "tool_yaw_tolerance": self.config.tool_yaw_tolerance,
        })

    def update_prev_accepted_xyz(self, command: PlannedCommand, receipt: CommandReceipt) -> None:
        """Commit only successfully issued targets, reconciling backend limiting."""
        actual = joint_vector(receipt.joint_targets)
        if np.array_equal(actual, command.joint_targets):
            target = np.asarray(command.cartesian_target, dtype=np.float64).copy()
        else:
            target, _ = forward_kinematics(self.robot_model, actual)
        self.prev_accepted_target = target
        self.gripper_target = float(actual[-1])
