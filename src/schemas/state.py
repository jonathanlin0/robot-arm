"""Timestamped sensor and perception records, in metres and radians."""

from dataclasses import dataclass
from typing import Any, TypeAlias
import numpy as np
from numpy.typing import NDArray
from schemas.arrays import CAMERA_IMAGE_SHAPES, JointVector, Vec3

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
    """One RGB image, identifier, and source sample timestamp.

    Incoming wrist images are 128x128; external images are 300x300. Images
    already have the camera's resolution; resizing/cropping is deferred.
    Retain the sample timestamp when the same buffered frame is read again.
    Keep uint8 values in [0, 255] here; CubeStateProvider converts model inputs
    to float32 and divides by 255.0 without modifying this raw image.
    """

    camera_id: str
    timestamp: float
    rgb: NDArray[np.uint8]  # Shape CAMERA_IMAGE_SHAPES[camera_id]; no resize/crop here.

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IMAGE_SHAPES:
            raise ValueError(f"Unknown camera ID {self.camera_id!r}.")
        expected = CAMERA_IMAGE_SHAPES[self.camera_id]
        if self.rgb.shape != expected:
            raise ValueError(
                f"Camera {self.camera_id!r} requires RGB image shape {expected}; "
                f"got {self.rgb.shape}."
            )


CameraFrames: TypeAlias = dict[str, CameraFrame]
"""Frames keyed by camera identifier, normally 'wrist' and 'external'."""


@dataclass
class CubeState:
    """Cube position, orientation representation, and velocity in the robot base.

    timestamp is the combined observation's current session time. Latest
    images/joint readings may have different sample times; no synchronization
    is implied by this estimate timestamp. Raw simulator observations retain a
    quaternion; learned vision supplies only the normalized yaw pair.
    """

    timestamp: float
    position: Vec3
    orientation: NDArray[np.floating[Any]] | None  # Raw quaternion (w, x, y, z), when available.
    linear_velocity: Vec3
    yaw: NDArray[np.floating[Any]] | None = None  # Already normalized with epsilon; never normalize again.


CubeStates: TypeAlias = dict[str, CubeState]
"""Cube estimates keyed by 'orange' and 'blue'."""
