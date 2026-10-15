"""Shared robot order, reference frame, and legacy controller defaults.

All distances use meters. Joint positions, commanded targets, and orientation
tolerances use radians; joint velocities use radians/second.
"""

from math import radians


ROBOT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = ROBOT_JOINT_NAMES[:-1]

# The legacy observation/controller reads this rigid site's XYZ directly.
# There is no additional offset and it is not the aperture-dependent jaw midpoint.
TOOL_FRAME_SITE_NAME = "gripperframe"
DEFAULT_GRIPPER_POSITION = 1.0  # Initial measured joint and actuator target; open.

# Per-axis displacement per normalized action: 2.5 mm per 50 ms control step.
MAXIMUM_POSITION_DELTA = 0.0025
DEFAULT_POSITION_TOLERANCE = 0.001
# Preserve the current legacy setting, including its deliberately loose tilt limit.
DEFAULT_TOOL_AXIS_TOLERANCE = radians(100.0)
DEFAULT_TOOL_YAW_TOLERANCE = radians(1.0)
WORLD_DOWN = (0.0, 0.0, -1.0)
IK_BACKTRACKING_SCALES = (1.0, 0.5, 0.25, 0.125)
