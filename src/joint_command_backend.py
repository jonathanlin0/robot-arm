"""Command destinations; only the simulated backend advances MuJoCo physics."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from time import monotonic

import mujoco
import numpy as np

from environment import CubeStackEnvironment
from hardware_connection import SO101Connection
from kinematics import joint_limits, joint_vector
from robot_constants import ROBOT_JOINT_NAMES
from schemas import CommandReceipt, JointVector


class JointCommandBackend(ABC):
    @abstractmethod
    def send_joint_targets(self, targets: JointVector) -> CommandReceipt:
        """Return actual successfully transmitted targets, without awaiting arrival."""
        raise NotImplementedError

    @abstractmethod
    def advance(self, duration_seconds: float) -> None:
        """Advance simulated time only; physical time is paced by LoopRunner."""
        raise NotImplementedError


class SimJointCommandBackend(JointCommandBackend):
    def __init__(self, environment: CubeStackEnvironment) -> None:
        self.environment = environment
        self._lower, self._upper = joint_limits(environment.model)

    def send_joint_targets(self, targets: JointVector) -> CommandReceipt:
        actual = np.clip(joint_vector(targets), self._lower, self._upper)
        with self.environment.state_lock:
            for name, target in zip(ROBOT_JOINT_NAMES, actual, strict=True):
                self.environment.data.actuator(name).ctrl[0] = target
            return CommandReceipt(timestamp=float(self.environment.data.time), joint_targets=actual.copy())

    def advance(self, duration_seconds: float) -> None:
        if not np.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and nonnegative.")
        # Do not silently round a control interval to another duration. The
        # scene timestep is shared with renderers and remains unchanged.
        timestep = float(self.environment.model.opt.timestep)
        steps = int(round(duration_seconds / timestep))
        if not np.isclose(steps * timestep, duration_seconds, rtol=0, atol=1e-10):
            raise ValueError("Control interval must be an integer multiple of the simulation timestep.")
        with self.environment.state_lock:
            if steps:
                mujoco.mj_step(self.environment.model, self.environment.data, nstep=steps)
                # Refresh derived poses to the final integrated joint state.
                mujoco.mj_forward(self.environment.model, self.environment.data)


class SO101JointCommandBackend(JointCommandBackend):
    """Transmit through the same externally owned adapter as the state reader."""

    def __init__(self, connection: SO101Connection, clock: Callable[[], float] = monotonic) -> None:
        self.connection = connection
        self.clock = clock

    def send_joint_targets(self, targets: JointVector) -> CommandReceipt:
        actual = self.connection.send_joint_targets(joint_vector(targets))
        return CommandReceipt(timestamp=float(self.clock()), joint_targets=joint_vector(actual))

    def advance(self, duration_seconds: float) -> None:
        if not np.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and nonnegative.")
        # Physical time advances naturally. LoopRunner performs the waiting.
