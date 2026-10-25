"""Temporary actuator overrides for demonstrations with observed recovery.

The intended action and planned command remain owned by the caller. Only the
targets sent to the wrapped backend change; its actual receipt lets the
Cartesian controller reconcile accepted targets without hidden target windup.
"""

from __future__ import annotations

import math

import numpy as np

from joint_command_backend import JointCommandBackend
from kinematics import joint_vector
from schemas import CommandReceipt, JointVector, RobotState
from schemas.config import DropDisturbanceConfig


class DropDisturbanceBackend(JointCommandBackend):
    """Freeze the measured arm, then open the gripper, while preserving labels.

    Call ``begin_step`` once per intended action before sending its command,
    then advance one control interval. Eligibility is supplied by the teacher's
    observation of a held orange cube; this layer never reads simulation truth.
    Phase lengths and per-action sampling match the legacy demonstration
    generator, including the possibility of another drop after recovery.
    """

    def __init__(
        self,
        backend: JointCommandBackend,
        config: DropDisturbanceConfig,
        rng: np.random.Generator,
        control_interval: float,
        open_gripper_target: float,
    ) -> None:
        if not math.isfinite(control_interval) or control_interval <= 0:
            raise ValueError("control_interval must be positive and finite.")
        if not math.isfinite(open_gripper_target):
            raise ValueError("open_gripper_target must be finite.")
        self.backend = backend
        self.config = config
        self.rng = rng
        self.control_interval = float(control_interval)
        self.open_gripper_target = float(open_gripper_target)
        self.stop_steps = max(1, math.ceil(config.stop_seconds / control_interval - 1e-9))
        self.release_steps = max(1, math.ceil(config.release_seconds / control_interval - 1e-9))
        self.steps = 0
        self.remaining_steps = 0
        self.events: list[dict[str, int]] = []
        self._held_targets: np.ndarray | None = None
        self._command_sent = False
        self._send_failed = False

    @property
    def active(self) -> bool:
        return self.remaining_steps > 0

    def begin_step(self, robot_state: RobotState, orange_held: bool) -> None:
        """Sample one eligible action, taking independent copies on activation."""
        if (self.active or not orange_held or self.config.probability == 0
                or self.rng.random() >= self.config.probability):
            return
        measured = joint_vector(robot_state.joint_positions, "measured joint positions")
        commanded = joint_vector(robot_state.commanded_joint_targets, "commanded joint targets")
        # The measured gripper can be blocked by the cube: retain the previous
        # closing target, and therefore squeezing force, during the stop phase.
        measured[-1] = commanded[-1]
        self._held_targets = measured
        self.remaining_steps = self.stop_steps + self.release_steps
        self.events.append({
            "start_step": self.steps,
            "stop_steps": self.stop_steps,
            "release_steps": self.release_steps,
        })

    def send_joint_targets(self, targets: JointVector) -> CommandReceipt:
        """Send intended or overridden targets and return the backend's receipt."""
        self._command_sent = False
        self._send_failed = True
        # Validate intended commands even during an override, so disturbances
        # never conceal a malformed controller result. Copy to preserve input.
        actual = joint_vector(targets)
        if self.active:
            assert self._held_targets is not None
            actual = self._held_targets.copy()
            if self.remaining_steps <= self.release_steps:
                actual[-1] = self.open_gripper_target
        receipt = self.backend.send_joint_targets(actual)
        self._command_sent = True
        self._send_failed = False
        return receipt

    def advance(self, duration_seconds: float) -> None:
        """Advance physics once; count the action only after successful advance."""
        if self._send_failed or (self.active and not self._command_sent):
            raise RuntimeError("Cannot advance a disturbance without a successfully sent command.")
        self.backend.advance(duration_seconds)
        if self.active:
            self.remaining_steps -= 1
        self.steps += 1
        self._command_sent = False
