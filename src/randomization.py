from dataclasses import dataclass

import mujoco
import numpy as np


ORANGE_CUBE_JOINT = "orange_cube_joint"
BLUE_CUBE_JOINT = "blue_cube_joint"


@dataclass(frozen=True)
class CubeSpawnConfig:
    """Valid tabletop region and spacing for randomized cube placements."""

    x_range: tuple[float, float] = (0.22, 0.38)
    y_range: tuple[float, float] = (-0.16, 0.16)
    cube_center_z: float = 0.02
    minimum_center_distance: float = 0.08
    maximum_attempts: int = 1_000

    def __post_init__(self) -> None:
        if self.x_range[0] >= self.x_range[1]:
            raise ValueError("x_range must have increasing bounds.")
        if self.y_range[0] >= self.y_range[1]:
            raise ValueError("y_range must have increasing bounds.")
        if self.minimum_center_distance < 0:
            raise ValueError("minimum_center_distance cannot be negative.")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least 1.")


def sample_cube_positions(
    rng: np.random.Generator,
    config: CubeSpawnConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample non-overlapping XY positions for the orange and blue cubes."""
    for _ in range(config.maximum_attempts):
        orange_xy = np.array(
            [
                rng.uniform(*config.x_range),
                rng.uniform(*config.y_range),
            ]
        )
        blue_xy = np.array(
            [
                rng.uniform(*config.x_range),
                rng.uniform(*config.y_range),
            ]
        )

        center_distance = np.linalg.norm(orange_xy - blue_xy)
        if center_distance >= config.minimum_center_distance:
            return orange_xy, blue_xy

    raise RuntimeError(
        "Could not sample valid cube positions. Check the spawn ranges and "
        "minimum separation."
    )


def set_cube_pose(
    data: mujoco.MjData,
    joint_name: str,
    xy_position: np.ndarray,
    center_z: float,
) -> None:
    """Set one cube's free-joint pose and clear its six velocities."""
    joint = data.joint(joint_name)

    if joint.qpos.shape != (7,) or joint.qvel.shape != (6,):
        raise ValueError(f"{joint_name!r} must be a free joint.")

    # Free-joint qpos: x, y, z, quaternion_w, quaternion_x,
    # quaternion_y, quaternion_z. This quaternion means no rotation.
    joint.qpos[:] = [*xy_position, center_z, 1.0, 0.0, 0.0, 0.0]
    joint.qvel.fill(0.0)


def randomize_cube_placements(
    data: mujoco.MjData,
    rng: np.random.Generator,
    config: CubeSpawnConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample and apply valid cube poses to an existing MuJoCo state."""
    orange_xy, blue_xy = sample_cube_positions(rng, config)

    set_cube_pose(data, ORANGE_CUBE_JOINT, orange_xy, config.cube_center_z)
    set_cube_pose(data, BLUE_CUBE_JOINT, blue_xy, config.cube_center_z)

    return orange_xy, blue_xy
