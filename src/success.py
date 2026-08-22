from dataclasses import dataclass
import math

import mujoco
import numpy as np

from randomization import BLUE_CUBE_JOINT, ORANGE_CUBE_JOINT


ORANGE_CUBE_BODY = "orange_cube"
BLUE_CUBE_BODY = "blue_cube"
ORANGE_CUBE_GEOM = "orange_cube_geom"
BLUE_CUBE_GEOM = "blue_cube_geom"
TABLE_GEOM = "table"
GRIPPER_BODY = "gripper"

# These are the collision primitives on the two surfaces that actually
# pinch an object. The more proximal box geoms and camera geoms are excluded
# so bumping the gripper housing does not count as a grasp.
# grabbed from models/so101/so101.xml
FIXED_JAW_PAD_GEOM_NAMES = (
    "fixed_jaw_sph_tip1",
    "fixed_jaw_sph_tip2",
    "fixed_jaw_sph_tip3",
    "fixed_jaw_box3",
    "fixed_jaw_box4",
    "fixed_jaw_box5",
    "fixed_jaw_box6",
    "fixed_jaw_box7",
)
MOVING_JAW_PAD_GEOM_NAMES = (
    "moving_jaw_sph_tip1",
    "moving_jaw_sph_tip2",
    "moving_jaw_sph_tip3",
    "moving_jaw_box2",
    "moving_jaw_box3",
)


@dataclass(frozen=True)
class StackSuccessConfig:
    """Tolerances for recognizing a released, stable cube stack."""

    # max allowed distance for x or y offset of center of cubes
    max_horizontal_center_offset: float = 0.01
    expected_vertical_center_distance: float = 0.04
    vertical_center_tolerance: float = 0.005
    max_linear_speed: float = 0.01 # m/s
    max_angular_speed: float = 0.05 # rad/s
    required_stable_time: float = 0.5 # seconds
    floating_point_numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        nonnegative_values = [
            self.max_horizontal_center_offset,
            self.vertical_center_tolerance,
            self.max_linear_speed,
            self.max_angular_speed,
            self.floating_point_numerical_tolerance,
        ]
        assert all(
            math.isfinite(value) and value >= 0
            for value in nonnegative_values
        )

        positive_values = [
            self.expected_vertical_center_distance,
            self.required_stable_time,
        ]
        assert all(
            math.isfinite(value) and value > 0
            for value in positive_values
        )


def _geoms_are_in_contact(
    data: mujoco.MjData,
    first_geom_id: int,
    second_geom_id: int,
) -> bool:
    for contact in data.contact:
        contact_geom_ids = {int(contact.geom1), int(contact.geom2)}
        if contact_geom_ids == {first_geom_id, second_geom_id}:
            return True

    return False


def orange_gripper_pad_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[bool, bool]:
    """Return whether orange touches the fixed and moving gripping pads."""
    orange_geom_id = model.geom(ORANGE_CUBE_GEOM).id
    fixed_pad_geom_ids = {
        model.geom(name).id for name in FIXED_JAW_PAD_GEOM_NAMES
    }
    moving_pad_geom_ids = {
        model.geom(name).id for name in MOVING_JAW_PAD_GEOM_NAMES
    }
    touches_fixed_pad = False
    touches_moving_pad = False

    # can also solve this by using _geoms_are_in_contact and then looping through to see if orange is touching any of fixed_pad_geom_ids and then any of moving_pad_geom_ids
    # but just looping through the collisions is more efficient since there's so many possible collisions to check

    for contact in data.contact:
        first_geom_id = int(contact.geom1)
        second_geom_id = int(contact.geom2)

        if orange_geom_id not in (first_geom_id, second_geom_id):
            continue

        other_geom_id = (
            second_geom_id
            if first_geom_id == orange_geom_id
            else first_geom_id
        )
        # geom id is -1 when that side of contact is a flexible object rather
        # than a regular geom
        if other_geom_id < 0:
            continue

        if other_geom_id in fixed_pad_geom_ids:
            touches_fixed_pad = True
        elif other_geom_id in moving_pad_geom_ids:
            touches_moving_pad = True

        if touches_fixed_pad and touches_moving_pad:
            break

    return touches_fixed_pad, touches_moving_pad


def orange_touches_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> bool:
    """Return whether the orange cube currently contacts the table."""
    return _geoms_are_in_contact(
        data,
        model.geom(ORANGE_CUBE_GEOM).id,
        model.geom(TABLE_GEOM).id,
    )


def _orange_touches_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    orange_geom_id: int,
) -> bool:
    gripper_body_id = model.body(GRIPPER_BODY).id

    for contact in data.contact:
        first_geom_id = int(contact.geom1)
        second_geom_id = int(contact.geom2)

        if orange_geom_id not in (first_geom_id, second_geom_id):
            continue

        other_geom_id = (
            second_geom_id
            if first_geom_id == orange_geom_id
            else first_geom_id
        )

        if other_geom_id < 0:
            continue

        other_body_id = int(model.geom_bodyid[other_geom_id])
        while other_body_id != 0:
            if other_body_id == gripper_body_id:
                return True
            other_body_id = int(model.body_parentid[other_body_id])

    return False


def stack_conditions_met(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: StackSuccessConfig,
) -> bool:
    """Check the instantaneous physical conditions for a valid stack."""
    orange_position = data.body(ORANGE_CUBE_BODY).xpos
    blue_position = data.body(BLUE_CUBE_BODY).xpos
    orange_velocity = data.joint(ORANGE_CUBE_JOINT).qvel
    blue_velocity = data.joint(BLUE_CUBE_JOINT).qvel

    values = (
        orange_position,
        blue_position,
        orange_velocity,
        blue_velocity,
    )
    for value in values:
        assert np.all(np.isfinite(value))

    # check for xy offset
    horizontal_error = np.abs(orange_position[:2] - blue_position[:2])
    if np.any(
        horizontal_error
        > config.max_horizontal_center_offset
        + config.floating_point_numerical_tolerance
    ):
        return False

    # check vertical relationship
    vertical_center_distance = orange_position[2] - blue_position[2]
    # check orange above blue
    if vertical_center_distance <= 0:
        return False
    # check centers are 0.04 m apart
    if (
        abs(
            vertical_center_distance
            - config.expected_vertical_center_distance
        )
        > config.vertical_center_tolerance
        + config.floating_point_numerical_tolerance
    ):
        return False

    for velocity in (orange_velocity, blue_velocity):
        linear_speed = np.linalg.norm(velocity[:3])
        angular_speed = np.linalg.norm(velocity[3:])
        if linear_speed > config.max_linear_speed:
            return False
        if angular_speed > config.max_angular_speed:
            return False

    # check that the two blocks are touching
    orange_geom_id = model.geom(ORANGE_CUBE_GEOM).id
    blue_geom_id = model.geom(BLUE_CUBE_GEOM).id
    if not _geoms_are_in_contact(
        data,
        orange_geom_id,
        blue_geom_id,
    ):
        return False

    # check that the gripper isn't touching orange anymore
    if _orange_touches_gripper(model, data, orange_geom_id):
        return False

    return True
