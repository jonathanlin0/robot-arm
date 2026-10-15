"""Coherent telemetry from the shared simulator or a calibrated SO-101 session."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from time import monotonic

import mujoco
import numpy as np

from environment import CubeStackEnvironment
from hardware_connection import SO101Connection
from kinematics import forward_kinematics, joint_vector
from robot_constants import ROBOT_JOINT_NAMES, TOOL_FRAME_SITE_NAME
from schemas import RobotState


class RobotStateProvider(ABC):
    """Read independent robot state without motion or owning resource lifetimes."""

    @abstractmethod
    def read(self) -> RobotState:
        """Return timestamped joints and rigid gripperframe XYZ in agreed units."""
        raise NotImplementedError


class SimRobotStateProvider(RobotStateProvider):
    def __init__(self, environment: CubeStackEnvironment) -> None:
        self.environment = environment

    def read(self) -> RobotState:
        with self.environment.state_lock:
            data = self.environment.data
            return RobotState(
                timestamp=float(data.time),
                joint_positions=np.array([data.joint(name).qpos[0] for name in ROBOT_JOINT_NAMES]),
                joint_velocities=np.array([data.joint(name).qvel[0] for name in ROBOT_JOINT_NAMES]),
                commanded_joint_targets=np.array([data.actuator(name).ctrl[0] for name in ROBOT_JOINT_NAMES]),
                gripper_position=data.site(TOOL_FRAME_SITE_NAME).xpos.copy(),
            )


class SO101RobotStateProvider(RobotStateProvider):
    """Read a supplied calibrated adapter and compute XYZ with matching geometry."""

    def __init__(
        self, connection: SO101Connection, robot_model: mujoco.MjModel,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.connection = connection
        self.robot_model = robot_model
        self.clock = clock

    def read(self) -> RobotState:
        positions, velocities, targets = self.connection.read_joint_state()
        positions = joint_vector(positions, "joint positions")
        xyz, _ = forward_kinematics(self.robot_model, positions)
        return RobotState(
            timestamp=float(self.clock()), joint_positions=positions,
            joint_velocities=joint_vector(velocities, "joint velocities"),
            commanded_joint_targets=joint_vector(targets), gripper_position=xyz,
        )
