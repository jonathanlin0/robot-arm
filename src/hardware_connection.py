"""The calibrated connection boundary supplied by a physical robot session.

Raw servo units and joint zero/sign calibration are deliberately owned by the
adapter, not inferred here. A future hardware setup supplies an implementation
backed by its robot library. State provider and command backend share that one
already-open adapter and never close it.
"""

from typing import Protocol

from schemas import JointVector


class SO101Connection(Protocol):
    """All vectors follow ROBOT_JOINT_NAMES and use radians/radians per second."""

    def read_joint_state(self) -> tuple[JointVector, JointVector, JointVector]:
        """Return measured positions, velocities, and last issued joint targets."""
        ...

    def send_joint_targets(self, targets: JointVector) -> JointVector:
        """Transmit targets; return actual limited targets or raise on failure."""
        ...
