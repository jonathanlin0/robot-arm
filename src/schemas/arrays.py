"""Numerical layouts shared by training and deployed inference."""

from typing import Any, TypeAlias
import numpy as np
from numpy.typing import NDArray

# Preserve the legacy observation order, excluding cube angular velocities and
# jaw-contact flags. XYZ vectors use (x, y, z); quaternions use (w, x, y, z).
# Each six-value joint group follows ROBOT_JOINT_NAMES, including the gripper.
OBSERVATION_FIELD_LAYOUT = (
    ("joint_positions", 6),
    ("joint_velocities", 6),
    ("commanded_joint_targets", 6),
    ("gripper_position", 3),
    ("orange_position", 3),
    ("orange_orientation", 4),
    ("orange_linear_velocity", 3),
    ("blue_position", 3),
    ("blue_orientation", 4),
    ("blue_linear_velocity", 3),
)
OBSERVATION_FIELDS = tuple(name for name, _ in OBSERVATION_FIELD_LAYOUT)
OBSERVATION_SIZE = sum(size for _, size in OBSERVATION_FIELD_LAYOUT)  # 41.
# Raw observations retain quaternions for reusable saved demonstrations. The
# policy receives symmetry-invariant yaw features after shared preprocessing.
POLICY_OBSERVATION_FIELD_LAYOUT = tuple(
    (name.replace("_orientation", "_yaw"), 2) if name.endswith("_orientation") else (name, size)
    for name, size in OBSERVATION_FIELD_LAYOUT
)
POLICY_OBSERVATION_FIELDS = tuple(name for name, _ in POLICY_OBSERVATION_FIELD_LAYOUT)
POLICY_OBSERVATION_SIZE = sum(size for _, size in POLICY_OBSERVATION_FIELD_LAYOUT)  # 37.
POLICY_ACTION_SIZE = 4
# [processed observation, previous policy action, prev_accepted_target XYZ].
HISTORY_TOKEN_SIZE = POLICY_OBSERVATION_SIZE + POLICY_ACTION_SIZE + 3  # 44.
MODEL_INPUT_DTYPE = np.float32
CAMERA_IMAGE_SHAPES = {  # Height, width, RGB channels for each view.
    "wrist": (128, 128, 3),
    "external": (300, 300, 3),
}


Vec3: TypeAlias = NDArray[np.floating[Any]]
"""Floating-point XYZ array with shape (3,), using meters or meters/second."""

JointVector: TypeAlias = NDArray[np.floating[Any]]
"""Shape (6,) in ROBOT_JOINT_NAMES order; radians or radians/second."""

PolicyAction: TypeAlias = NDArray[np.float32]
"""Normalized [dx, dy, dz, gripper] action with shape (4,)."""
