"""Timestamped sensor and perception records, in metres and radians."""

from dataclasses import dataclass
from typing import Any, TypeAlias
import numpy as np
from numpy.typing import NDArray
from schemas.arrays import JointVector, Vec3

@dataclass
class RobotState:
    """Timestamped telemetry, including the rigid gripper reference's XYZ.

    gripper_position is the measured/FK position of TOOL_FRAME_SITE_NAME
    (gripperframe), exactly as in the legacy observation. No additional offset
    is applied. Joint vectors follow ROBOT_JOINT_NAMES in radians/radians per
    second; the gripper joint uses the same angular convention.
    """

    timestamp: float
    joint_positions: JointVector
    joint_velocities: JointVector
    commanded_joint_targets: JointVector
    gripper_position: Vec3


@dataclass
class CameraFrame:
    """One 300x300 RGB image, identifier, and source sample timestamp.

    Incoming images already have this resolution; resizing/cropping is deferred.
    Retain the sample timestamp when the same buffered frame is read again.
    Keep uint8 values in [0, 255] here; CubeStateProvider converts model inputs
    to float32 and divides by 255.0 without modifying this raw image.
    """

    camera_id: str
    timestamp: float
    rgb: NDArray[np.uint8]  # Shape CAMERA_IMAGE_SHAPE; no resize/crop here.


CameraFrames: TypeAlias = dict[str, CameraFrame]
"""Frames keyed by camera identifier, normally 'wrist' and 'external'."""


@dataclass
class CubeState:
    """Estimated cube pose and linear velocity relative to the robot base.

    timestamp is the combined observation's current session time. Latest
    images/joint readings may have different sample times; no synchronization
    is implied by this estimate timestamp.
    """

    timestamp: float
    position: Vec3
    orientation: NDArray[np.floating[Any]]  # Quaternion (w, x, y, z).
    linear_velocity: Vec3


CubeStates: TypeAlias = dict[str, CubeState]
"""Cube estimates keyed by 'orange' and 'blue'."""


