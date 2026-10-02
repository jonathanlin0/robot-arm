import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

import environment as environment_module
from environment import MINIMUM_HOLD_TIME, CubeStackEnvironment
from success import StackSuccessConfig


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")
BLUE_CENTER = np.array([0.30, 0.0, 0.02])


@pytest.fixture
def environment() -> CubeStackEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    environment = CubeStackEnvironment(scene_path=SCENE_PATH)
    environment.reset(seed=0)
    return environment


def arrange_stack(
    environment: CubeStackEnvironment,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    vertical_center_distance: float = 0.04,
) -> None:
    blue_joint = environment.data.joint("blue_cube_joint")
    orange_joint = environment.data.joint("orange_cube_joint")

    blue_joint.qpos[:] = [*BLUE_CENTER, 1.0, 0.0, 0.0, 0.0]
    orange_joint.qpos[:] = [
        BLUE_CENTER[0] + x_offset,
        BLUE_CENTER[1] + y_offset,
        BLUE_CENTER[2] + vertical_center_distance,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    blue_joint.qvel.fill(0.0)
    orange_joint.qvel.fill(0.0)
    mujoco.mj_forward(environment.model, environment.data)


def record_prior_confirmed_grasp(
    environment: CubeStackEnvironment,
) -> None:
    # These tests isolate stack-stability tracking from grasp detection.
    environment._confirmed_grasp_seen = True


def geoms_are_in_contact(
    environment: CubeStackEnvironment,
    first_geom_name: str,
    second_geom_name: str,
) -> bool:
    first_geom_id = environment.model.geom(first_geom_name).id
    second_geom_id = environment.model.geom(second_geom_name).id

    for contact in environment.data.contact:
        if {int(contact.geom1), int(contact.geom2)} == {
            first_geom_id,
            second_geom_id,
        }:
            return True

    return False


def test_aligned_released_stack_meets_instantaneous_conditions(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)

    assert environment.stack_conditions_met() is True
    assert environment.is_success() is False
    assert environment._stack_success is False


@pytest.mark.parametrize(
    ("x_offset", "y_offset"),
    [
        (0.01, 0.0),
        (-0.01, 0.0),
        (0.0, 0.01),
        (0.0, -0.01),
        (0.01, 0.01),
    ],
)
def test_horizontal_center_offset_boundary_is_inclusive(
    environment: CubeStackEnvironment,
    x_offset: float,
    y_offset: float,
) -> None:
    arrange_stack(
        environment,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    assert environment.stack_conditions_met() is True


@pytest.mark.parametrize(
    ("x_offset", "y_offset"),
    [
        (0.0101, 0.0),
        (-0.0101, 0.0),
        (0.0, 0.0101),
        (0.0, -0.0101),
    ],
)
def test_excessive_horizontal_center_offset_fails(
    environment: CubeStackEnvironment,
    x_offset: float,
    y_offset: float,
) -> None:
    arrange_stack(
        environment,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    assert environment.stack_conditions_met() is False


def test_wrong_vertical_separation_fails_despite_cube_contact(
    environment: CubeStackEnvironment,
) -> None:
    invalid_distance = (
        environment.success_config.expected_vertical_center_distance
        - environment.success_config.vertical_center_tolerance
        - 0.001
    )
    arrange_stack(
        environment,
        vertical_center_distance=invalid_distance,
    )

    assert geoms_are_in_contact(
        environment,
        "orange_cube_geom",
        "blue_cube_geom",
    )
    assert environment.stack_conditions_met() is False


def test_orange_cube_below_blue_cube_fails(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment, vertical_center_distance=-0.04)
    environment.data.joint("blue_cube_joint").qpos[2] = 0.10
    # Add one micrometer of overlap so contact generation is not sensitive
    # to floating-point rounding at exactly 4 cm separation.
    environment.data.joint("orange_cube_joint").qpos[2] = 0.060001
    mujoco.mj_forward(environment.model, environment.data)

    assert geoms_are_in_contact(
        environment,
        "orange_cube_geom",
        "blue_cube_geom",
    )
    assert environment.stack_conditions_met() is False


def test_floating_orange_cube_fails_without_cube_contact(
    environment: CubeStackEnvironment,
) -> None:
    floating_distance = (
        environment.success_config.expected_vertical_center_distance
        + 0.8 * environment.success_config.vertical_center_tolerance
    )
    arrange_stack(
        environment,
        vertical_center_distance=floating_distance,
    )

    assert not geoms_are_in_contact(
        environment,
        "orange_cube_geom",
        "blue_cube_geom",
    )
    assert environment.stack_conditions_met() is False


@pytest.mark.parametrize(
    ("gripper_geom_name", "orange_is_first_geom"),
    [
        ("fixed_jaw_box1", True),
        ("fixed_jaw_box1", False),
        ("moving_jaw_box1", True),
        ("moving_jaw_box1", False),
    ],
)
def test_orange_cube_touching_gripper_fails(
    environment: CubeStackEnvironment,
    gripper_geom_name: str,
    orange_is_first_geom: bool,
) -> None:
    arrange_stack(environment)
    orange_geom_id = environment.model.geom("orange_cube_geom").id
    gripper_geom_id = environment.model.geom(gripper_geom_name).id

    # Preserve the real orange-blue contacts and replace one existing contact
    # with a deterministic orange-gripper contact.
    if orange_is_first_geom:
        environment.data.contact[0].geom1 = orange_geom_id
        environment.data.contact[0].geom2 = gripper_geom_id
    else:
        environment.data.contact[0].geom1 = gripper_geom_id
        environment.data.contact[0].geom2 = orange_geom_id

    assert geoms_are_in_contact(
        environment,
        "orange_cube_geom",
        "blue_cube_geom",
    )
    assert environment.stack_conditions_met() is False


@pytest.mark.parametrize(
    ("joint_name", "velocity_index", "speed_config_name"),
    [
        ("orange_cube_joint", 0, "max_linear_speed"),
        ("blue_cube_joint", 0, "max_linear_speed"),
        ("orange_cube_joint", 3, "max_angular_speed"),
        ("blue_cube_joint", 3, "max_angular_speed"),
    ],
)
def test_moving_cube_fails(
    environment: CubeStackEnvironment,
    joint_name: str,
    velocity_index: int,
    speed_config_name: str,
) -> None:
    arrange_stack(environment)
    maximum_speed = getattr(environment.success_config, speed_config_name)
    environment.data.joint(joint_name).qvel[velocity_index] = (
        maximum_speed + 0.001
    )
    mujoco.mj_forward(environment.model, environment.data)

    assert environment.stack_conditions_met() is False


def test_combined_velocity_norm_above_limit_fails(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    component_speed = 0.8 * environment.success_config.max_linear_speed
    environment.data.joint("orange_cube_joint").qvel[:2] = component_speed
    mujoco.mj_forward(environment.model, environment.data)

    assert component_speed < environment.success_config.max_linear_speed
    assert environment.stack_conditions_met() is False


def test_velocity_at_limit_is_accepted(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    environment.data.joint("orange_cube_joint").qvel[0] = (
        environment.success_config.max_linear_speed
    )
    mujoco.mj_forward(environment.model, environment.data)

    assert environment.stack_conditions_met() is True


@pytest.mark.parametrize("required_stable_time", [0.5, 0.25, 0.517])
def test_stack_must_remain_valid_for_required_time(
    environment: CubeStackEnvironment,
    required_stable_time: float,
) -> None:
    environment.success_config = StackSuccessConfig(
        required_stable_time=required_stable_time,
    )
    arrange_stack(environment)
    record_prior_confirmed_grasp(environment)
    required_step_count = math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    )

    environment.step_physics(required_step_count - 1)

    assert environment._stack_success is False
    assert environment.is_success() is False
    assert environment.is_terminated() is False
    assert environment.stack_stable_time < (
        environment.success_config.required_stable_time
    )

    environment.step_physics(1)

    assert environment._stack_success is True
    assert environment.is_success() is True
    assert environment.is_terminated() is True
    assert environment.stack_stable_time == pytest.approx(
        environment.success_config.required_stable_time
    )


def test_invalid_step_restarts_stability_window(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    record_prior_confirmed_grasp(environment)
    required_step_count = math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count // 2)

    assert environment.stack_stable_time > 0.0
    assert environment._stack_success is False

    arrange_stack(environment, x_offset=0.0101)
    environment.step_physics(1)

    assert environment.stack_stable_time == 0.0
    assert environment._stack_success is False

    arrange_stack(environment)
    environment.step_physics(required_step_count - 1)
    assert environment._stack_success is False

    environment.step_physics(1)
    assert environment._stack_success is True
    assert environment.is_success() is True


def test_reset_clears_success_tracking(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    record_prior_confirmed_grasp(environment)
    required_step_count = math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count)

    assert environment._stack_success is True
    assert environment.is_success() is True

    environment.reset(seed=1)

    assert environment.is_success() is False
    assert environment._stack_success is False
    assert environment.stack_stable_time == 0.0
    assert environment.confirmed_grasp_seen is False


def test_success_remains_latched_until_reset(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    record_prior_confirmed_grasp(environment)
    required_step_count = math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count)
    assert environment._stack_success is True

    arrange_stack(environment, x_offset=0.02)
    environment.step_physics(1)

    assert environment.stack_conditions_met() is False
    assert environment._stack_success is True
    assert environment.is_success() is True


@pytest.mark.parametrize("cube_name", ["orange", "blue"])
def test_off_table_failure_overrides_latched_stack_success(
    environment: CubeStackEnvironment,
    cube_name: str,
) -> None:
    arrange_stack(environment)
    record_prior_confirmed_grasp(environment)
    environment.step_physics(math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    ))
    assert environment.is_success() is True

    cube_joint = environment.data.joint(f"{cube_name}_cube_joint")
    cube_joint.qpos[0] = 1.0
    cube_joint.qpos[2] = 0.0
    mujoco.mj_forward(environment.model, environment.data)
    environment.step_physics(1)

    assert environment.is_failure() is True
    assert environment.is_success() is False
    assert environment.is_terminated() is True
    assert environment.stack_stable_time == 0.0


def test_stack_stability_does_not_accumulate_before_grasp(
    environment: CubeStackEnvironment,
) -> None:
    arrange_stack(environment)
    required_step_count = math.ceil(
        environment.success_config.required_stable_time
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count)

    assert environment.confirmed_grasp_seen is False
    assert environment.stack_stable_time == 0.0
    assert environment.is_success() is False
    assert environment._stack_success is False


def test_environment_records_grasp_without_terminating_episode(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        lambda model, data: (True, True),
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: False,
    )

    environment.step_physics(1)
    state = environment.get_state()

    assert environment.confirmed_grasp_seen is True
    assert state["confirmed_grasp_seen"] is True
    assert environment.get_hold_time() == pytest.approx(
        environment.model.opt.timestep
    )
    assert state["orange_grasp_hold_time"] == pytest.approx(
        environment.model.opt.timestep
    )
    assert environment.is_success() is False
    assert environment.is_terminated() is False


def test_two_second_pickup_hold_does_not_succeed_without_stack(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        lambda model, data: (True, True),
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: False,
    )
    required_step_count = math.ceil(
        MINIMUM_HOLD_TIME
        / environment.model.opt.timestep
    )

    environment.step_physics(required_step_count - 1)

    assert environment.get_hold_time() == pytest.approx(
        MINIMUM_HOLD_TIME
        - environment.model.opt.timestep
    )
    assert environment.is_success() is False

    environment.step_physics(1)

    assert environment.get_hold_time() == pytest.approx(MINIMUM_HOLD_TIME)
    assert environment.confirmed_grasp_seen is True
    assert environment.stack_stable_time == 0.0
    assert environment.is_success() is False
    assert environment.is_terminated() is False


@pytest.mark.parametrize(
    ("fixed_contact", "moving_contact", "table_contact"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_invalid_hold_resets_continuous_hold_timer(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    fixed_contact: bool,
    moving_contact: bool,
    table_contact: bool,
) -> None:
    contact_state = {
        "fixed": True,
        "moving": True,
        "table": False,
    }
    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        lambda model, data: (
            contact_state["fixed"],
            contact_state["moving"],
        ),
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: contact_state["table"],
    )
    environment.step_physics(100)
    assert environment.get_hold_time() == pytest.approx(0.5)

    contact_state.update(
        fixed=fixed_contact,
        moving=moving_contact,
        table=table_contact,
    )
    environment.step_physics(1)

    assert environment.get_hold_time() == 0.0
    assert environment.is_success() is False


def test_regrasp_restarts_hold_diagnostic_without_triggering_success(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contacts = {"valid": True}
    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        lambda model, data: (contacts["valid"], contacts["valid"]),
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: False,
    )
    required_step_count = math.ceil(
        MINIMUM_HOLD_TIME
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count - 1)
    contacts["valid"] = False
    environment.step_physics(1)
    contacts["valid"] = True

    environment.step_physics(required_step_count - 1)
    assert environment.get_hold_time() == pytest.approx(
        MINIMUM_HOLD_TIME - environment.model.opt.timestep
    )
    assert environment.is_success() is False

    environment.step_physics(1)
    assert environment.get_hold_time() == pytest.approx(MINIMUM_HOLD_TIME)
    assert environment.is_success() is False


def test_transient_contact_loss_inside_one_action_resets_hold_timer(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contact_check_count = 0

    def contacts_with_one_transient_loss(model, data) -> tuple[bool, bool]:
        nonlocal contact_check_count
        contact_check_count += 1
        valid = contact_check_count != 5
        return valid, valid

    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        contacts_with_one_transient_loss,
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: False,
    )

    environment.step_physics(10)

    assert environment.get_hold_time() == pytest.approx(0.025)
    assert environment.is_success() is False


def test_hold_diagnostic_reflects_current_hold_and_reset_clears_start_time(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contacts = {"valid": True}
    monkeypatch.setattr(
        environment_module,
        "orange_gripper_pad_contacts",
        lambda model, data: (contacts["valid"], contacts["valid"]),
    )
    monkeypatch.setattr(
        environment_module,
        "orange_touches_table",
        lambda model, data: False,
    )
    required_step_count = math.ceil(
        MINIMUM_HOLD_TIME
        / environment.model.opt.timestep
    )
    environment.step_physics(required_step_count)
    assert environment.get_hold_time() == pytest.approx(MINIMUM_HOLD_TIME)
    assert environment.is_success() is False

    contacts["valid"] = False
    environment.step_physics(1)
    assert environment.get_hold_time() == 0.0
    assert environment.is_success() is False

    environment.reset(seed=1)
    assert environment.get_hold_time() == 0.0
    assert environment.confirmed_grasp_seen is False
    assert environment.is_success() is False


@pytest.mark.parametrize(
    ("height_offset", "tolerance"),
    [(0.08, 0.01), (0.05, 0.004)],
)
@pytest.mark.parametrize(
    "distance_in_tolerances",
    [
        (0.0, 0.0, 0.0),
        (0.9, 0.0, 0.0),
        (-1.1, 0.0, 0.0),
        (0.0, 0.0, 1.1),
        (0.7, 0.7, 0.0),
        (0.8, 0.8, 0.0),
    ],
)
def test_waypoint_proximity_alone_does_not_succeed(
    height_offset: float,
    tolerance: float,
    distance_in_tolerances: tuple[float, float, float],
) -> None:
    environment = CubeStackEnvironment(
        scene_path=SCENE_PATH,
        orange_waypoint_height_offset=height_offset,
        orange_waypoint_tolerance=tolerance,
    )
    environment.reset(seed=0)
    gripper_position = environment.data.site("gripperframe").xpos.copy()
    environment.data.joint("orange_cube_joint").qpos[:3] = (
        gripper_position
        - np.array([0.0, 0.0, height_offset])
        + tolerance * np.asarray(distance_in_tolerances)
    )
    mujoco.mj_forward(environment.model, environment.data)

    assert environment.get_hold_time() == 0.0
    assert environment.is_success() is False
    assert environment.is_terminated() is False

    environment.reset(seed=1)
    assert environment.is_success() is False
    assert environment.is_terminated() is False


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value"),
    [
        ("max_horizontal_center_offset", -0.001),
        ("vertical_center_tolerance", -0.001),
        ("max_linear_speed", np.inf),
        ("max_angular_speed", np.nan),
        ("floating_point_numerical_tolerance", -1e-12),
        ("expected_vertical_center_distance", 0.0),
        ("required_stable_time", 0.0),
    ],
)
def test_invalid_success_config_raises(
    parameter_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(AssertionError):
        StackSuccessConfig(**{parameter_name: invalid_value})
