import mujoco
import numpy as np


FIXED_JAW_TIP_GEOM_NAMES = (
    "fixed_jaw_sph_tip1",
    "fixed_jaw_sph_tip2",
    "fixed_jaw_sph_tip3",
)
MOVING_JAW_TIP_GEOM_NAMES = (
    "moving_jaw_sph_tip1",
    "moving_jaw_sph_tip2",
    "moving_jaw_sph_tip3",
)


def _jaw_tip_center(
    data: mujoco.MjData,
    geom_names: tuple[str, ...],
) -> np.ndarray:
    """Return one jaw's mean tip-marker position in world coordinates."""
    return np.mean(
        [data.geom(geom_name).xpos for geom_name in geom_names],
        axis=0,
    )


def jaw_tip_midpoint(data: mujoco.MjData) -> np.ndarray:
    """Return the live world-space midpoint between the two jaw tips."""
    fixed_jaw_center = _jaw_tip_center(
        data,
        FIXED_JAW_TIP_GEOM_NAMES,
    )
    moving_jaw_center = _jaw_tip_center(
        data,
        MOVING_JAW_TIP_GEOM_NAMES,
    )
    return (fixed_jaw_center + moving_jaw_center) / 2.0


def _jaw_tip_center_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_names: tuple[str, ...],
) -> np.ndarray:
    """Return the position Jacobian of one jaw's mean tip position."""
    jaw_jacobian = np.zeros((3, model.nv))
    geom_jacobian = np.zeros((3, model.nv))

    for geom_name in geom_names:
        geom_jacobian.fill(0.0)
        mujoco.mj_jacGeom(
            model,
            data,
            geom_jacobian,
            None,
            model.geom(geom_name).id,
        )
        jaw_jacobian += geom_jacobian

    return jaw_jacobian / len(geom_names)


def jaw_tip_midpoint_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """Return the position Jacobian of the live midpoint between jaw tips.

    The midpoint is the average of the fixed- and moving-jaw tip centers, so
    its Jacobian is the same average of their position Jacobians.
    """
    fixed_jaw_jacobian = _jaw_tip_center_position_jacobian(
        model,
        data,
        FIXED_JAW_TIP_GEOM_NAMES,
    )
    moving_jaw_jacobian = _jaw_tip_center_position_jacobian(
        model,
        data,
        MOVING_JAW_TIP_GEOM_NAMES,
    )
    return (fixed_jaw_jacobian + moving_jaw_jacobian) / 2.0
