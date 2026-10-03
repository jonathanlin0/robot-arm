from pathlib import Path

import mujoco
import numpy as np
import pytest

from cartesian_actions import CartesianActionResult
from environment import (
    MINIMUM_HOLD_TIME,
    CubeStackEnvironment,
    StateSnapshot,
)
from kinematics import ToolAxisIKResult
from rewards import StackRewardCalculator, StackRewardConfig
from success import (
    FIXED_JAW_PAD_GEOM_NAMES,
    MOVING_JAW_PAD_GEOM_NAMES,
    StackSuccessConfig,
    orange_gripper_pad_contacts,
)


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")
COMPONENT_NAMES = {
    "approach_orange_progress",
    "approach_orange_waypoint",
    "grasp_candidate",
    "grasp",
    "hold_orange_duration",
    "lift_orange_height",
    "move_toward_hover_progress",
    "lower_toward_stack_progress",
    "stack_alignment_progress",
    "successful_stack",
    "dropped_cube",
    "ik_failure",
    "action_magnitude",
    "gripper_state_change",
    "unproductive_close",
}


@pytest.fixture
def environment() -> CubeStackEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    environment = CubeStackEnvironment(scene_path=SCENE_PATH)
    environment.reset(seed=1)
    return environment


def make_state(
    *,
    simulation_time: float = 0.0,
    gripper_position: tuple[float, float, float] = (0.1, 0.0, 0.1),
    gripper_target: float = 1.0,
    orange_position: tuple[float, float, float] = (0.2, 0.0, 0.02),
    blue_position: tuple[float, float, float] = (0.3, 0.0, 0.02),
    fixed_jaw_contact: bool = False,
    moving_jaw_contact: bool = False,
    table_contact: bool = True,
    orange_grasp_hold_time: float = 0.0,
    confirmed_grasp_seen: bool | None = None,
    orange_fell_off_table: bool = False,
    blue_fell_off_table: bool = False,
) -> StateSnapshot:
    if confirmed_grasp_seen is None:
        confirmed_grasp_seen = (
            fixed_jaw_contact
            and moving_jaw_contact
            and not table_contact
        )

    return {
        "time": simulation_time,
        "gripper_position": np.array(gripper_position, dtype=float),
        "gripper_target": gripper_target,
        "orange_position": np.array(orange_position, dtype=float),
        "blue_position": np.array(blue_position, dtype=float),
        "orange_touches_fixed_jaw": fixed_jaw_contact,
        "orange_touches_moving_jaw": moving_jaw_contact,
        "orange_touches_table": table_contact,
        "orange_currently_held": (
            fixed_jaw_contact
            and moving_jaw_contact
            and not table_contact
        ),
        "orange_grasp_hold_time": orange_grasp_hold_time,
        "confirmed_grasp_seen": confirmed_grasp_seen,
        "orange_fell_off_table": orange_fell_off_table,
        "blue_fell_off_table": blue_fell_off_table,
    }


def make_action_result(
    state: StateSnapshot,
    *,
    position_converged: bool = True,
    tool_axis_converged: bool = True,
) -> CartesianActionResult:
    return CartesianActionResult(
        state=state,
        target_gripper_position=np.asarray(
            state["gripper_position"],
            dtype=float,
        ).copy(),
        ik_result=ToolAxisIKResult(
            joint_positions=np.zeros(5),
            position_converged=position_converged,
            tool_axis_converged=tool_axis_converged,
            position_error=0.0 if position_converged else 1.0,
            tool_axis_error=0.0 if tool_axis_converged else 1.0,
            iterations=1,
        ),
    )


def zero_reward_config(**overrides: float) -> StackRewardConfig:
    values = {
        "approach_orange_progress_weight": 0.0,
        "approach_orange_height_offset": 0.05,
        "approach_orange_waypoint_tolerance": 0.005,
        "approach_orange_waypoint_reward": 0.0,
        "grasp_candidate_reward": 0.0,
        "grasp_reward": 0.0,
        "hold_orange_duration_weight": 0.0,
        "lift_orange_height_weight": 0.0,
        "vertical_lift_margin": 0.03,
        "move_toward_hover_progress_weight": 0.0,
        "lower_toward_stack_progress_weight": 0.0,
        "stack_alignment_progress_weight": 0.0,
        "successful_stack_reward": 0.0,
        "dropped_cube_penalty": 0.0,
        "ik_failure_penalty": 0.0,
        "action_magnitude_penalty_weight": 0.0,
        "gripper_state_change_penalty": 0.0,
        "unproductive_close_penalty": 0.0,
    }
    values.update(overrides)
    return StackRewardConfig(**values)


def test_default_reward_config_is_pickup_only() -> None:
    config = StackRewardConfig()

    assert config.approach_orange_height_offset == pytest.approx(0.08)
    assert config.approach_orange_waypoint_tolerance == pytest.approx(0.01)
    assert config.approach_orange_capture_radius == pytest.approx(0.015)
    assert config.approach_orange_waypoint_reward == pytest.approx(1.0)
    assert config.hold_orange_duration_weight > 0.0
    assert config.lift_orange_height_weight > 0.0
    assert config.move_toward_hover_progress_weight == 0.0
    assert config.lower_toward_stack_progress_weight == 0.0
    assert config.stack_alignment_progress_weight == 0.0


def calculate_transition(
    calculator: StackRewardCalculator,
    previous_state: StateSnapshot,
    current_state: StateSnapshot,
    *,
    action: np.ndarray | None = None,
    succeeded: bool = False,
    position_converged: bool = True,
    tool_axis_converged: bool = True,
):
    if action is None:
        action = np.zeros(4)

    return calculator.calculate(
        previous_state,
        action,
        make_action_result(
            current_state,
            position_converged=position_converged,
            tool_axis_converged=tool_axis_converged,
        ),
        succeeded,
    )


def replace_contact(
    environment: CubeStackEnvironment,
    contact_index: int,
    first_geom_name: str,
    second_geom_name: str,
) -> None:
    contact = environment.data.contact[contact_index]
    contact.geom1 = environment.model.geom(first_geom_name).id
    contact.geom2 = environment.model.geom(second_geom_name).id


def test_orange_gripper_pad_contacts_detects_both_jaws_in_either_order(
    environment: CubeStackEnvironment,
) -> None:
    replace_contact(
        environment,
        0,
        "orange_cube_geom",
        FIXED_JAW_PAD_GEOM_NAMES[0],
    )
    replace_contact(
        environment,
        1,
        MOVING_JAW_PAD_GEOM_NAMES[0],
        "orange_cube_geom",
    )

    assert orange_gripper_pad_contacts(
        environment.model,
        environment.data,
    ) == (True, True)


@pytest.mark.parametrize(
    ("first_pad", "second_pad", "expected"),
    [
        (
            FIXED_JAW_PAD_GEOM_NAMES[0],
            FIXED_JAW_PAD_GEOM_NAMES[1],
            (True, False),
        ),
        (
            MOVING_JAW_PAD_GEOM_NAMES[0],
            MOVING_JAW_PAD_GEOM_NAMES[1],
            (False, True),
        ),
        ("fixed_jaw_box1", "moving_jaw_box1", (False, False)),
        ("camera_box1", "camera_box2", (False, False)),
    ],
)
def test_orange_gripper_pad_contacts_rejects_non_bilateral_contacts(
    environment: CubeStackEnvironment,
    first_pad: str,
    second_pad: str,
    expected: tuple[bool, bool],
) -> None:
    replace_contact(
        environment,
        0,
        "orange_cube_geom",
        first_pad,
    )
    replace_contact(
        environment,
        1,
        "orange_cube_geom",
        second_pad,
    )

    assert orange_gripper_pad_contacts(
        environment.model,
        environment.data,
    ) == expected


def test_all_configured_jaw_pad_geoms_exist(
    environment: CubeStackEnvironment,
) -> None:
    for geom_name in (
        *FIXED_JAW_PAD_GEOM_NAMES,
        *MOVING_JAW_PAD_GEOM_NAMES,
    ):
        assert environment.model.geom(geom_name).id >= 0


def test_environment_snapshot_reports_initial_table_contact(
    environment: CubeStackEnvironment,
) -> None:
    state = environment.get_state()

    assert state["orange_touches_fixed_jaw"] is False
    assert state["orange_touches_moving_jaw"] is False
    assert state["orange_touches_table"] is True


def test_calculator_must_be_reset_before_calculate() -> None:
    state = make_state()
    calculator = StackRewardCalculator()

    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        calculate_transition(calculator, state, state)


@pytest.mark.parametrize(
    ("current_gripper_z", "expected_reward"),
    [(0.07, 0.03939375), (-0.03, -0.01464375)],
)
def test_approach_reward_targets_position_above_orange(
    current_gripper_z: float,
    expected_reward: float,
) -> None:
    previous_state = make_state(
        gripper_position=(0.0, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=(0.0, 0.0, current_gripper_z),
        orange_position=(0.0, 0.0, 0.02),
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=2.0,
            approach_orange_height_offset=0.10,
        )
    )
    calculator.reset(previous_state)

    result = calculate_transition(
        calculator,
        previous_state,
        current_state,
    )

    assert result.components["approach_orange_progress"] == pytest.approx(
        expected_reward
    )


def test_approach_reward_height_offset_is_configurable() -> None:
    previous_state = make_state(
        gripper_position=(0.0, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=(0.0, 0.0, 0.07),
        orange_position=(0.0, 0.0, 0.02),
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=2.0,
            approach_orange_height_offset=0.05,
        )
    )
    calculator.reset(previous_state)

    result = calculate_transition(
        calculator,
        previous_state,
        current_state,
    )

    assert result.components["approach_orange_progress"] == pytest.approx(
        0.08720625
    )


def test_reaching_pregrasp_waypoint_switches_next_target_to_cube_center(
) -> None:
    initial_state = make_state(
        gripper_position=(0.0, 0.0, 0.13),
        orange_position=(0.0, 0.0, 0.02),
    )
    waypoint_state = make_state(
        gripper_position=(0.0, 0.0, 0.124),
        orange_position=(0.0, 0.0, 0.02),
    )
    descending_state = make_state(
        gripper_position=(0.0, 0.0, 0.1215),
        orange_position=(0.0, 0.0, 0.02),
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=2.0,
            approach_orange_height_offset=0.10,
            approach_orange_waypoint_tolerance=0.005,
            approach_orange_waypoint_reward=1.25,
        )
    )
    calculator.reset(initial_state)

    waypoint_result = calculate_transition(
        calculator,
        initial_state,
        waypoint_state,
    )

    assert waypoint_result.components[
        "approach_orange_progress"
    ] == pytest.approx(
        2.0 * 30.0 * (0.296**5 - 0.29**5)
    )
    assert waypoint_result.components[
        "approach_orange_waypoint"
    ] == pytest.approx(1.25)
    assert calculator.orange_pregrasp_waypoint_reached is True

    descent_result = calculate_transition(
        calculator,
        waypoint_state,
        descending_state,
    )

    # Descending is progress only if the latched target is now orange's
    # center; it would be negative progress toward the overhead waypoint.
    assert descent_result.components[
        "approach_orange_progress"
    ] == pytest.approx(
        2.0 * 30.0 * (0.1985**5 - 0.196**5)
    )
    assert descent_result.components["approach_orange_waypoint"] == 0.0


def test_pregrasp_waypoint_reward_is_once_per_episode() -> None:
    initial_state = make_state(
        gripper_position=(0.0, 0.0, 0.14),
        orange_position=(0.0, 0.0, 0.02),
    )
    waypoint_state = make_state(
        gripper_position=(0.0, 0.0, 0.105),
        orange_position=(0.0, 0.0, 0.02),
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_height_offset=0.08,
            approach_orange_waypoint_tolerance=0.01,
            approach_orange_waypoint_reward=2.5,
        )
    )
    calculator.reset(initial_state)

    first_entry = calculate_transition(
        calculator,
        initial_state,
        waypoint_state,
    )
    after_entry = calculate_transition(
        calculator,
        waypoint_state,
        waypoint_state,
    )

    assert first_entry.components["approach_orange_waypoint"] == 2.5
    assert after_entry.components["approach_orange_waypoint"] == 0.0

    calculator.reset(initial_state)
    after_reset = calculate_transition(
        calculator,
        initial_state,
        waypoint_state,
    )
    assert after_reset.components["approach_orange_waypoint"] == 2.5


def reach_pregrasp_waypoint(calculator: StackRewardCalculator) -> None:
    orange_position = (0.0, 0.0, 0.02)
    waypoint_z = orange_position[2] + calculator.config.approach_orange_height_offset
    initial_state = make_state(
        gripper_position=(0.0, 0.0, waypoint_z + 0.02),
        orange_position=orange_position,
    )
    waypoint_state = make_state(
        gripper_position=(0.0, 0.0, waypoint_z),
        orange_position=orange_position,
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, waypoint_state)
    assert calculator.orange_pregrasp_waypoint_reached is True


@pytest.mark.parametrize(
    ("previous_x", "current_x"),
    [
        (0.020, 0.010),  # Entering the radius earns no approach reward.
        (0.010, 0.005),  # Moving closer within the radius.
        (0.005, 0.010),  # Moving away within the radius.
        (0.020, 0.015),  # The 15 mm boundary is included.
        (0.010, 0.000),  # Exact alignment has no extra reward.
    ],
)
def test_post_waypoint_approach_reward_is_zero_within_capture_radius(
    previous_x: float,
    current_x: float,
) -> None:
    calculator = StackRewardCalculator(
        zero_reward_config(approach_orange_progress_weight=2.0)
    )
    reach_pregrasp_waypoint(calculator)
    previous_state = make_state(
        gripper_position=(previous_x, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=(current_x, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    # The gate must use the rigid gripperframe, even if the jaw midpoint
    # lies outside the capture radius.
    current_state["jaw_midpoint"] = np.array([0.1, 0.0, 0.02])

    result = calculate_transition(calculator, previous_state, current_state)

    assert result.components["approach_orange_progress"] == 0.0


@pytest.mark.parametrize(
    ("previous_position", "current_position", "expected_sign"),
    [
        ((0.025, 0.0, 0.02), (0.020, 0.0, 0.02), 1),
        ((0.010, 0.0, 0.02), (0.020, 0.0, 0.02), -1),
        # Each axis error is below 15 mm, but the 3D distance exceeds it.
        ((0.025, 0.0, 0.02), (0.010, 0.010, 0.03), 1),
    ],
)
def test_post_waypoint_approach_reward_outside_radius_is_unchanged(
    previous_position: tuple[float, float, float],
    current_position: tuple[float, float, float],
    expected_sign: int,
) -> None:
    previous_state = make_state(
        gripper_position=previous_position,
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=current_position,
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state["jaw_midpoint"] = current_state["orange_position"].copy()
    rewards = []
    for radius in (0.0, 0.015):
        calculator = StackRewardCalculator(
            zero_reward_config(
                approach_orange_progress_weight=2.0,
                approach_orange_capture_radius=radius,
            )
        )
        reach_pregrasp_waypoint(calculator)
        result = calculate_transition(calculator, previous_state, current_state)
        rewards.append(result.components["approach_orange_progress"])

    assert rewards[1] == pytest.approx(rewards[0])
    assert expected_sign * rewards[1] > 0.0


def test_capture_radius_does_not_suppress_pre_waypoint_approach_reward() -> None:
    calculator = StackRewardCalculator(
        zero_reward_config(approach_orange_progress_weight=2.0)
    )
    previous_state = make_state(
        gripper_position=(0.0, 0.0, 0.030),
        orange_position=(0.0, 0.0, 0.020),
    )
    current_state = make_state(
        gripper_position=(0.0, 0.0, 0.032),
        orange_position=(0.0, 0.0, 0.020),
    )
    calculator.reset(previous_state)

    result = calculate_transition(calculator, previous_state, current_state)

    assert calculator.orange_pregrasp_waypoint_reached is False
    assert result.components["approach_orange_progress"] > 0.0


def test_capture_radius_is_configurable() -> None:
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=2.0,
            approach_orange_capture_radius=0.025,
        )
    )
    reach_pregrasp_waypoint(calculator)
    previous_state = make_state(
        gripper_position=(0.030, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=(0.020, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )

    result = calculate_transition(calculator, previous_state, current_state)

    assert result.components["approach_orange_progress"] == 0.0


def test_capture_radius_preserves_grasp_hold_and_lift_rewards() -> None:
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=2.0,
            grasp_candidate_reward=1.0,
            grasp_reward=5.0,
            hold_orange_duration_weight=2.0,
            lift_orange_height_weight=3.0,
        )
    )
    reach_pregrasp_waypoint(calculator)
    previous_state = make_state(
        gripper_position=(0.010, 0.0, 0.02),
        orange_position=(0.0, 0.0, 0.02),
    )
    current_state = make_state(
        gripper_position=(0.010, 0.0, 0.03),
        orange_position=(0.0, 0.0, 0.03),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
        orange_grasp_hold_time=0.05,
    )

    result = calculate_transition(calculator, previous_state, current_state)

    assert result.components["approach_orange_progress"] == 0.0
    assert result.components["grasp_candidate"] == 1.0
    assert result.components["grasp"] == 5.0
    assert result.components["hold_orange_duration"] == pytest.approx(0.1)
    assert result.components["lift_orange_height"] == pytest.approx(0.03)
    assert result.total == pytest.approx(6.13)


def test_approach_reward_values_same_progress_more_near_target() -> None:
    orange_position = (0.0, 0.0, 0.02)
    config = zero_reward_config(
        approach_orange_progress_weight=1.0,
        approach_orange_height_offset=0.0,
    )

    far_previous = make_state(
        gripper_position=(0.101, 0.0, 0.02),
        orange_position=orange_position,
    )
    far_current = make_state(
        gripper_position=(0.100, 0.0, 0.02),
        orange_position=orange_position,
    )
    far_calculator = StackRewardCalculator(config)
    far_calculator.reset(far_previous)
    far_result = calculate_transition(
        far_calculator,
        far_previous,
        far_current,
    )

    near_previous = make_state(
        gripper_position=(0.051, 0.0, 0.02),
        orange_position=orange_position,
    )
    near_current = make_state(
        gripper_position=(0.050, 0.0, 0.02),
        orange_position=orange_position,
    )
    near_calculator = StackRewardCalculator(config)
    near_calculator.reset(near_previous)
    near_result = calculate_transition(
        near_calculator,
        near_previous,
        near_current,
    )

    far_reward = far_result.components["approach_orange_progress"]
    near_reward = near_result.components["approach_orange_progress"]
    assert far_reward > 0.0
    assert near_reward > far_reward


def test_approach_reward_is_zero_beyond_thirty_centimeters() -> None:
    orange_position = (0.0, 0.0, 0.02)
    previous_state = make_state(
        gripper_position=(0.40, 0.0, 0.02),
        orange_position=orange_position,
    )
    current_state = make_state(
        gripper_position=(0.35, 0.0, 0.02),
        orange_position=orange_position,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=1.0,
            approach_orange_height_offset=0.0,
        )
    )
    calculator.reset(previous_state)

    result = calculate_transition(
        calculator,
        previous_state,
        current_state,
    )

    assert result.components["approach_orange_progress"] == 0.0


def test_pregrasp_waypoint_latch_is_permanent_until_reset() -> None:
    initial_state = make_state(
        gripper_position=(0.0, 0.0, 0.13),
        orange_position=(0.0, 0.0, 0.02),
    )
    outside_tolerance_state = make_state(
        gripper_position=(0.0, 0.0, 0.1221),
        orange_position=(0.0, 0.0, 0.02),
    )
    inside_tolerance_state = make_state(
        gripper_position=(0.0, 0.0, 0.1219),
        orange_position=(0.0, 0.0, 0.02),
    )
    moved_away_state = make_state(
        gripper_position=(0.0, 0.0, 0.14),
        orange_position=(0.0, 0.0, 0.02),
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_progress_weight=1.0,
            approach_orange_height_offset=0.10,
            approach_orange_waypoint_tolerance=0.002,
        )
    )
    calculator.reset(initial_state)

    calculate_transition(
        calculator,
        initial_state,
        outside_tolerance_state,
    )
    assert calculator.orange_pregrasp_waypoint_reached is False

    calculate_transition(
        calculator,
        outside_tolerance_state,
        inside_tolerance_state,
    )
    assert calculator.orange_pregrasp_waypoint_reached is True

    calculate_transition(
        calculator,
        inside_tolerance_state,
        moved_away_state,
    )
    assert calculator.orange_pregrasp_waypoint_reached is True

    calculator.reset(initial_state)
    assert calculator.orange_pregrasp_waypoint_reached is False


def test_approach_reward_ignores_aperture_induced_jaw_midpoint_motion(
    environment: CubeStackEnvironment,
) -> None:
    previous_state = environment.reset(seed=31)
    calculator = StackRewardCalculator(
        zero_reward_config(approach_orange_progress_weight=1.0)
    )
    calculator.reset(previous_state)

    environment.data.joint("gripper").qpos[0] = -0.1
    mujoco.mj_forward(environment.model, environment.data)
    current_state = environment.get_state()

    result = calculator.calculate(
        previous_state,
        np.zeros(4),
        make_action_result(current_state),
        succeeded=False,
    )
    np.testing.assert_allclose(
        current_state["gripper_position"],
        previous_state["gripper_position"],
    )
    assert not np.allclose(
        current_state["jaw_midpoint"],
        previous_state["jaw_midpoint"],
    )
    assert result.components[
        "approach_orange_progress"
    ] == pytest.approx(0.0, abs=1e-12)


def test_grasp_requires_both_jaws_and_loss_of_table_support() -> None:
    initial_state = make_state()
    calculator = StackRewardCalculator(
        zero_reward_config(grasp_reward=5.0)
    )

    for current_state in (
        make_state(fixed_jaw_contact=True, table_contact=False),
        make_state(
            fixed_jaw_contact=True,
            moving_jaw_contact=True,
            table_contact=True,
        ),
    ):
        calculator.reset(initial_state)
        result = calculate_transition(
            calculator,
            initial_state,
            current_state,
        )
        assert result.components["grasp"] == 0.0

    confirmed_grasp_state = make_state(
        orange_position=(0.2, 0.0, 0.04),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator.reset(initial_state)
    result = calculate_transition(
        calculator,
        initial_state,
        confirmed_grasp_state,
    )

    assert result.components["grasp"] == 5.0


def test_bilateral_contact_reward_is_small_and_once_per_episode() -> None:
    initial_state = make_state()
    bilateral_table_state = make_state(
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(grasp_candidate_reward=1.0, grasp_reward=5.0)
    )
    calculator.reset(initial_state)

    first_result = calculate_transition(
        calculator,
        initial_state,
        bilateral_table_state,
    )
    repeated_result = calculate_transition(
        calculator,
        bilateral_table_state,
        bilateral_table_state,
    )

    assert first_result.components["grasp_candidate"] == 1.0
    assert first_result.components["grasp"] == 0.0
    assert repeated_result.components["grasp_candidate"] == 0.0


def test_grasp_reward_is_once_per_episode_and_reset_restores_it() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.06),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(grasp_reward=5.0)
    )
    calculator.reset(initial_state)

    first_result = calculate_transition(
        calculator,
        initial_state,
        grasped_state,
    )
    repeated_result = calculate_transition(
        calculator,
        grasped_state,
        grasped_state,
    )

    assert first_result.components["grasp"] == 5.0
    assert repeated_result.components["grasp"] == 0.0

    calculator.reset(initial_state)
    after_reset_result = calculate_transition(
        calculator,
        initial_state,
        grasped_state,
    )
    assert after_reset_result.components["grasp"] == 5.0


def test_approach_reward_stops_after_confirmed_grasp() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.06),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    closer_state = make_state(
        gripper_position=(0.15, 0.0, 0.06),
        orange_position=(0.2, 0.0, 0.06),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(approach_orange_progress_weight=1.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, grasped_state)

    result = calculate_transition(
        calculator,
        grasped_state,
        closer_state,
    )

    assert result.components["approach_orange_progress"] == 0.0


def test_approach_reward_stops_immediately_on_bilateral_contact() -> None:
    initial_state = make_state(
        gripper_position=(0.0, 0.0, 0.20),
        orange_position=(0.0, 0.0, 0.02),
    )
    bilateral_contact_state = make_state(
        gripper_position=(0.0, 0.0, 0.10),
        orange_position=(0.0, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(approach_orange_progress_weight=1.0)
    )
    calculator.reset(initial_state)

    result = calculate_transition(
        calculator,
        initial_state,
        bilateral_contact_state,
    )

    assert result.components["approach_orange_progress"] == 0.0


def test_pickup_rewards_increase_with_hold_duration_and_height() -> None:
    initial_state = make_state(orange_position=(0.2, 0.0, 0.02))
    low_short_hold_state = make_state(
        orange_position=(0.2, 0.0, 0.06),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
        orange_grasp_hold_time=0.5,
    )
    high_long_hold_state = make_state(
        orange_position=(0.2, 0.0, 0.14),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
        orange_grasp_hold_time=1.5,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            hold_orange_duration_weight=2.0,
            lift_orange_height_weight=5.0,
        )
    )
    calculator.reset(initial_state)

    low_short_result = calculate_transition(
        calculator,
        initial_state,
        low_short_hold_state,
    )
    high_long_result = calculate_transition(
        calculator,
        low_short_hold_state,
        high_long_hold_state,
    )

    assert low_short_result.components[
        "hold_orange_duration"
    ] == pytest.approx(1.0)
    assert high_long_result.components[
        "hold_orange_duration"
    ] == pytest.approx(3.0)
    assert low_short_result.components["lift_orange_height"] == pytest.approx(
        0.2
    )
    assert high_long_result.components[
        "lift_orange_height"
    ] == pytest.approx(
        0.6
    )


def test_hold_duration_reward_is_capped_at_success_duration() -> None:
    initial_state = make_state()
    held_state = make_state(
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
        orange_grasp_hold_time=10.0,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(hold_orange_duration_weight=2.0)
    )
    calculator.reset(initial_state)

    result = calculate_transition(calculator, initial_state, held_state)

    assert result.components["hold_orange_duration"] == pytest.approx(
        2.0 * MINIMUM_HOLD_TIME
    )


@pytest.mark.parametrize(
    "held_state",
    [
        make_state(
            fixed_jaw_contact=True,
            moving_jaw_contact=True,
            table_contact=True,
            orange_grasp_hold_time=1.0,
        ),
        make_state(
            fixed_jaw_contact=True,
            moving_jaw_contact=False,
            table_contact=False,
            orange_grasp_hold_time=1.0,
        ),
    ],
)
def test_pickup_rewards_require_a_current_valid_hold(
    held_state: StateSnapshot,
) -> None:
    initial_state = make_state()
    calculator = StackRewardCalculator(
        zero_reward_config(
            hold_orange_duration_weight=2.0,
            lift_orange_height_weight=5.0,
        )
    )
    calculator.reset(initial_state)

    result = calculate_transition(calculator, initial_state, held_state)

    assert result.components["hold_orange_duration"] == 0.0
    assert result.components["lift_orange_height"] == 0.0


def test_horizontal_movement_below_safe_height_gets_no_transport_reward(
) -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    low_grasped_state = make_state(
        orange_position=(0.0, 0.0, 0.06),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    moved_horizontally_state = make_state(
        orange_position=(0.1, 0.0, 0.06),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            move_toward_hover_progress_weight=2.0,
            lower_toward_stack_progress_weight=2.0,
            stack_alignment_progress_weight=5.0,
        )
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, low_grasped_state)

    result = calculate_transition(
        calculator,
        low_grasped_state,
        moved_horizontally_state,
    )

    assert calculator.safe_lift_completed is False
    assert result.components["move_toward_hover_progress"] == 0.0
    assert result.components["lower_toward_stack_progress"] == 0.0
    assert result.components["stack_alignment_progress"] == 0.0


def test_reaching_safe_height_enables_alignment_on_next_transition() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_height_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    closer_state = make_state(
        orange_position=(0.1, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(stack_alignment_progress_weight=5.0)
    )
    calculator.reset(initial_state)

    reaching_height_result = calculate_transition(
        calculator,
        initial_state,
        safe_height_state,
    )
    assert reaching_height_result.components["stack_alignment_progress"] == 0.0
    assert calculator.safe_lift_completed is True

    alignment_result = calculate_transition(
        calculator,
        safe_height_state,
        closer_state,
    )

    assert alignment_result.components[
        "stack_alignment_progress"
    ] == pytest.approx(0.5)


def test_moving_toward_hover_target_gives_positive_reward() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_height_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    closer_state = make_state(
        orange_position=(0.1, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(move_toward_hover_progress_weight=2.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_height_state)

    result = calculate_transition(
        calculator,
        safe_height_state,
        closer_state,
    )

    assert result.components["move_toward_hover_progress"] == pytest.approx(
        0.2
    )


def test_hover_reward_uses_success_config_for_target_height() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.2, 0.0, 0.02),
    )
    safe_height_state = make_state(
        orange_position=(0.0, 0.0, 0.11),
        blue_position=(0.2, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    closer_state = make_state(
        orange_position=(0.1, 0.0, 0.11),
        blue_position=(0.2, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    success_config = StackSuccessConfig(
        expected_vertical_center_distance=0.06
    )
    calculator = StackRewardCalculator(
        zero_reward_config(move_toward_hover_progress_weight=2.0),
        success_config,
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_height_state)

    result = calculate_transition(
        calculator,
        safe_height_state,
        closer_state,
    )

    assert result.components["move_toward_hover_progress"] == pytest.approx(
        0.2
    )


def test_alignment_reward_uses_largest_axis_error_after_safe_lift() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.2, 0.1, 0.02),
    )
    safe_height_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.2, 0.1, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    closer_state = make_state(
        orange_position=(0.1, 0.05, 0.09),
        blue_position=(0.2, 0.1, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(stack_alignment_progress_weight=5.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_height_state)

    result = calculate_transition(
        calculator,
        safe_height_state,
        closer_state,
    )

    assert result.components["stack_alignment_progress"] == pytest.approx(
        0.5
    )


def test_descending_before_alignment_gets_no_placement_reward() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_unaligned_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    descended_state = make_state(
        orange_position=(0.0, 0.0, 0.08),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            lift_orange_height_weight=5.0,
            lower_toward_stack_progress_weight=2.0,
        )
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_unaligned_state)

    result = calculate_transition(
        calculator,
        safe_unaligned_state,
        descended_state,
    )

    assert calculator.hover_alignment_completed is False
    assert result.components["lower_toward_stack_progress"] == 0.0
    assert result.components["lift_orange_height"] == pytest.approx(0.3)


def test_descending_after_hover_alignment_gives_placement_reward() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_unaligned_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    aligned_hover_state = make_state(
        orange_position=(0.3, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    descended_state = make_state(
        orange_position=(0.3, 0.0, 0.08),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            lift_orange_height_weight=5.0,
            lower_toward_stack_progress_weight=2.0,
        )
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_unaligned_state)
    calculate_transition(
        calculator,
        safe_unaligned_state,
        aligned_hover_state,
    )
    assert calculator.hover_alignment_completed is True

    result = calculate_transition(
        calculator,
        aligned_hover_state,
        descended_state,
    )

    assert result.components["lower_toward_stack_progress"] == pytest.approx(
        0.02
    )
    assert result.components["lift_orange_height"] == pytest.approx(0.3)


def test_moving_away_from_hover_target_gives_negative_reward() -> None:
    initial_state = make_state(
        orange_position=(0.1, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_unaligned_state = make_state(
        orange_position=(0.1, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    farther_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(move_toward_hover_progress_weight=2.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_unaligned_state)

    result = calculate_transition(
        calculator,
        safe_unaligned_state,
        farther_state,
    )

    assert result.components["move_toward_hover_progress"] == pytest.approx(
        -0.2
    )


def test_losing_alignment_gives_negative_reward() -> None:
    initial_state = make_state(
        orange_position=(0.0, 0.0, 0.02),
        blue_position=(0.3, 0.0, 0.02),
    )
    safe_unaligned_state = make_state(
        orange_position=(0.0, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    aligned_hover_state = make_state(
        orange_position=(0.3, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    misaligned_state = make_state(
        orange_position=(0.25, 0.0, 0.09),
        blue_position=(0.3, 0.0, 0.02),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(stack_alignment_progress_weight=5.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, safe_unaligned_state)
    calculate_transition(
        calculator,
        safe_unaligned_state,
        aligned_hover_state,
    )

    result = calculate_transition(
        calculator,
        aligned_hover_state,
        misaligned_state,
    )

    assert result.components["stack_alignment_progress"] == pytest.approx(
        -0.25
    )


def test_drop_is_penalized_when_grasped_cube_lands_on_table() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.08),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    released_midair_state = make_state(
        orange_position=(0.2, 0.0, 0.07),
        table_contact=False,
        confirmed_grasp_seen=True,
    )
    landed_state = make_state(
        orange_position=(0.2, 0.0, 0.02),
        table_contact=True,
        confirmed_grasp_seen=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(dropped_cube_penalty=-10.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, grasped_state)

    release_result = calculate_transition(
        calculator,
        grasped_state,
        released_midair_state,
    )
    landed_result = calculate_transition(
        calculator,
        released_midair_state,
        landed_state,
    )
    repeated_result = calculate_transition(
        calculator,
        landed_state,
        landed_state,
    )

    assert release_result.components["dropped_cube"] == 0.0
    assert landed_result.components["dropped_cube"] == -10.0
    assert repeated_result.components["dropped_cube"] == 0.0


@pytest.mark.parametrize(
    ("orange_fell_off_table", "blue_fell_off_table"),
    [(True, False), (False, True)],
)
def test_off_table_drop_is_penalized_without_prior_grasp(
    orange_fell_off_table: bool,
    blue_fell_off_table: bool,
) -> None:
    initial_state = make_state()
    off_table_state = make_state(
        orange_position=(
            0.2,
            0.0,
            0.0 if orange_fell_off_table else 0.02,
        ),
        blue_position=(
            0.3,
            0.0,
            0.0 if blue_fell_off_table else 0.02,
        ),
        table_contact=not orange_fell_off_table,
        confirmed_grasp_seen=False,
        orange_fell_off_table=orange_fell_off_table,
        blue_fell_off_table=blue_fell_off_table,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(dropped_cube_penalty=-10.0)
    )
    calculator.reset(initial_state)

    result = calculate_transition(
        calculator,
        initial_state,
        off_table_state,
    )

    assert calculator.confirmed_grasp_seen is False
    assert result.components["dropped_cube"] == -10.0


@pytest.mark.parametrize(
    ("orange_fell_off_table", "blue_fell_off_table"),
    [(True, False), (False, True)],
)
def test_off_table_failure_takes_precedence_over_latched_success(
    orange_fell_off_table: bool,
    blue_fell_off_table: bool,
) -> None:
    initial_state = make_state(
        table_contact=False,
        confirmed_grasp_seen=True,
    )
    off_table_state = make_state(
        orange_position=(
            0.2,
            0.0,
            0.0 if orange_fell_off_table else 0.02,
        ),
        blue_position=(
            0.3,
            0.0,
            0.0 if blue_fell_off_table else 0.02,
        ),
        table_contact=False,
        confirmed_grasp_seen=True,
        orange_fell_off_table=orange_fell_off_table,
        blue_fell_off_table=blue_fell_off_table,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            successful_stack_reward=100.0,
            dropped_cube_penalty=-10.0,
        )
    )
    calculator.reset(initial_state)

    result = calculate_transition(
        calculator,
        initial_state,
        off_table_state,
        succeeded=True,
    )

    assert result.components["dropped_cube"] == -10.0
    assert result.components["successful_stack"] == 0.0


def test_pickup_height_reward_stops_after_release_without_clawback() -> None:
    initial_state = make_state(orange_position=(0.2, 0.0, 0.02))
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.08),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    released_state = make_state(
        orange_position=(0.2, 0.0, 0.07),
        table_contact=False,
        confirmed_grasp_seen=True,
    )
    landed_state = make_state(
        orange_position=(0.2, 0.0, 0.02),
        table_contact=True,
        confirmed_grasp_seen=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(lift_orange_height_weight=5.0)
    )
    calculator.reset(initial_state)

    lift_result = calculate_transition(
        calculator,
        initial_state,
        grasped_state,
    )
    release_result = calculate_transition(
        calculator,
        grasped_state,
        released_state,
    )
    fall_result = calculate_transition(
        calculator,
        released_state,
        landed_state,
    )

    assert lift_result.components["lift_orange_height"] == pytest.approx(
        0.3
    )
    assert release_result.components["lift_orange_height"] == 0.0
    assert fall_result.components["lift_orange_height"] == 0.0


def test_confirmed_regrasp_rearms_drop_penalty() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.08),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    landed_state = make_state(
        orange_position=(0.2, 0.0, 0.02),
        table_contact=True,
        confirmed_grasp_seen=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(dropped_cube_penalty=-10.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, grasped_state)

    first_drop = calculate_transition(
        calculator,
        grasped_state,
        landed_state,
    )
    calculate_transition(calculator, landed_state, grasped_state)
    second_drop = calculate_transition(
        calculator,
        grasped_state,
        landed_state,
    )

    assert first_drop.components["dropped_cube"] == -10.0
    assert second_drop.components["dropped_cube"] == -10.0


def test_successful_stack_does_not_receive_drop_penalty() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.08),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    released_state = make_state(
        orange_position=(0.3, 0.0, 0.06),
        table_contact=False,
        confirmed_grasp_seen=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(
            dropped_cube_penalty=-10.0,
            successful_stack_reward=100.0,
        )
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, grasped_state)

    result = calculate_transition(
        calculator,
        grasped_state,
        released_state,
        succeeded=True,
    )

    assert result.components["dropped_cube"] == 0.0
    assert result.components["successful_stack"] == 100.0


def test_only_position_ik_failure_is_penalized() -> None:
    state = make_state()
    calculator = StackRewardCalculator(
        zero_reward_config(ik_failure_penalty=-1.5)
    )
    calculator.reset(state)

    tool_axis_failure = calculate_transition(
        calculator,
        state,
        state,
        tool_axis_converged=False,
    )
    position_failure = calculate_transition(
        calculator,
        state,
        state,
        position_converged=False,
    )

    assert tool_axis_failure.components["ik_failure"] == 0.0
    assert position_failure.components["ik_failure"] == -1.5


def test_action_penalty_clips_deltas_and_excludes_absolute_gripper() -> None:
    state = make_state()
    calculator = StackRewardCalculator(
        zero_reward_config(action_magnitude_penalty_weight=-0.12)
    )
    calculator.reset(state)

    movement_result = calculate_transition(
        calculator,
        state,
        state,
        action=np.array([2.0, -2.0, 0.5, 1.0]),
    )
    gripper_only_result = calculate_transition(
        calculator,
        state,
        state,
        action=np.array([0.0, 0.0, 0.0, -1.0]),
    )

    assert movement_result.components["action_magnitude"] == pytest.approx(
        -0.09
    )
    assert gripper_only_result.components["action_magnitude"] == 0.0


@pytest.mark.parametrize(
    ("previous_target", "current_target"),
    [
        (1.0, -0.1),
        (-0.1, 1.0),
    ],
)
def test_changing_gripper_state_is_penalized(
    previous_target: float,
    current_target: float,
) -> None:
    previous_state = make_state(gripper_target=previous_target)
    current_state = make_state(gripper_target=current_target)
    calculator = StackRewardCalculator(
        zero_reward_config(gripper_state_change_penalty=-0.01)
    )
    calculator.reset(previous_state)

    result = calculate_transition(
        calculator,
        previous_state,
        current_state,
    )

    assert result.components["gripper_state_change"] == -0.01
    assert result.total == -0.01


def test_retaining_gripper_state_is_not_penalized() -> None:
    state = make_state(gripper_target=-0.1)
    calculator = StackRewardCalculator(
        zero_reward_config(gripper_state_change_penalty=-0.01)
    )
    calculator.reset(state)

    result = calculate_transition(calculator, state, state)

    assert result.components["gripper_state_change"] == 0.0
    assert result.total == 0.0


def test_closed_gripper_is_penalized_every_step_before_waypoint() -> None:
    open_state = make_state(simulation_time=0.0, gripper_target=1.0)
    closing_state = make_state(
        simulation_time=0.05,
        gripper_target=-0.1,
    )
    retained_closed_state = make_state(
        simulation_time=0.10,
        gripper_target=-0.1,
    )
    later_closed_state = make_state(
        simulation_time=0.15,
        gripper_target=-0.1,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(open_state)

    closing_result = calculate_transition(
        calculator,
        open_state,
        closing_state,
    )
    retained_result = calculate_transition(
        calculator,
        closing_state,
        retained_closed_state,
    )
    later_result = calculate_transition(
        calculator,
        retained_closed_state,
        later_closed_state,
    )

    assert closing_result.components["unproductive_close"] == -0.05
    assert retained_result.components["unproductive_close"] == -0.05
    assert later_result.components["unproductive_close"] == -0.05


def test_opening_gripper_stops_pre_waypoint_close_penalty() -> None:
    open_state = make_state(simulation_time=0.0, gripper_target=1.0)
    closing_state = make_state(simulation_time=0.05, gripper_target=-0.1)
    reopened_state = make_state(simulation_time=0.10, gripper_target=1.0)
    retained_open_state = make_state(
        simulation_time=0.15,
        gripper_target=1.0,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(open_state)

    closing_result = calculate_transition(
        calculator,
        open_state,
        closing_state,
    )
    reopened_result = calculate_transition(
        calculator,
        closing_state,
        reopened_state,
    )
    retained_open_result = calculate_transition(
        calculator,
        reopened_state,
        retained_open_state,
    )

    assert closing_result.components["unproductive_close"] == -0.05
    assert reopened_result.components["unproductive_close"] == 0.0
    assert retained_open_result.components["unproductive_close"] == 0.0


def test_closed_start_uses_configured_open_target_for_gripper_shaping() -> None:
    closed_state = make_state(gripper_target=-0.1)
    reopened_state = make_state(gripper_target=1.0)
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(closed_state, open_gripper_target=1.0)

    retained_result = calculate_transition(calculator, closed_state, closed_state)
    reopened_result = calculate_transition(calculator, closed_state, reopened_state)

    assert retained_result.components["unproductive_close"] == -0.05
    assert reopened_result.components["unproductive_close"] == 0.0


def test_prepared_closed_start_does_not_earn_waypoint_or_gripper_switch_rewards() -> None:
    closed_state = make_state(gripper_target=-0.1)
    calculator = StackRewardCalculator(
        zero_reward_config(
            approach_orange_waypoint_reward=2.5,
            gripper_state_change_penalty=-0.01,
            unproductive_close_penalty=-0.05,
        )
    )
    calculator.reset(
        closed_state,
        orange_pregrasp_waypoint_reached=True,
        open_gripper_target=1.0,
    )

    result = calculate_transition(calculator, closed_state, closed_state)

    assert calculator.orange_pregrasp_waypoint_reached
    assert result.total == 0.0
    assert result.components["approach_orange_waypoint"] == 0.0
    assert result.components["gripper_state_change"] == 0.0
    assert result.components["unproductive_close"] == 0.0


def test_bilateral_contact_exempts_closed_gripper_before_waypoint() -> None:
    open_state = make_state(gripper_target=1.0)
    clamped_state = make_state(
        gripper_target=-0.1,
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(open_state)

    result = calculate_transition(
        calculator,
        open_state,
        clamped_state,
    )

    assert calculator.orange_pregrasp_waypoint_reached is False
    assert result.components["unproductive_close"] == 0.0
    assert result.total == 0.0


def test_closed_gripper_is_not_penalized_when_entering_waypoint() -> None:
    initial_state = make_state(
        gripper_position=(0.2, 0.0, 0.10),
        gripper_target=1.0,
    )
    waypoint_state = make_state(
        gripper_position=(0.2, 0.0, 0.07),
        gripper_target=-0.1,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(initial_state)

    result = calculate_transition(
        calculator,
        initial_state,
        waypoint_state,
    )

    assert calculator.orange_pregrasp_waypoint_reached is True
    assert result.components["unproductive_close"] == 0.0
    assert result.total == 0.0


def test_closed_gripper_is_not_penalized_after_leaving_reached_waypoint(
) -> None:
    initial_state = make_state(
        gripper_position=(0.2, 0.0, 0.10),
        gripper_target=1.0,
    )
    waypoint_state = make_state(
        gripper_position=(0.2, 0.0, 0.07),
        gripper_target=-0.1,
    )
    moved_away_state = make_state(
        gripper_position=(0.1, 0.0, 0.15),
        gripper_target=-0.1,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(initial_state)
    calculate_transition(
        calculator,
        initial_state,
        waypoint_state,
    )

    result = calculate_transition(
        calculator,
        waypoint_state,
        moved_away_state,
    )

    assert calculator.orange_pregrasp_waypoint_reached is True
    assert result.components["unproductive_close"] == 0.0
    assert result.total == 0.0


def test_reset_reenables_pre_waypoint_close_penalty() -> None:
    initial_state = make_state(
        gripper_position=(0.2, 0.0, 0.10),
        gripper_target=1.0,
    )
    waypoint_state = make_state(
        gripper_position=(0.2, 0.0, 0.07),
        gripper_target=1.0,
    )
    reset_state = make_state(gripper_target=1.0)
    closed_state = make_state(gripper_target=-0.1)
    calculator = StackRewardCalculator(
        zero_reward_config(unproductive_close_penalty=-0.05)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, waypoint_state)
    assert calculator.orange_pregrasp_waypoint_reached is True

    calculator.reset(reset_state)

    result = calculate_transition(
        calculator,
        reset_state,
        closed_state,
    )

    assert calculator.orange_pregrasp_waypoint_reached is False
    assert result.components["unproductive_close"] == -0.05
    assert result.total == -0.05


def test_result_always_contains_named_components_and_correct_total() -> None:
    initial_state = make_state()
    grasped_state = make_state(
        orange_position=(0.2, 0.0, 0.08),
        fixed_jaw_contact=True,
        moving_jaw_contact=True,
        table_contact=False,
    )
    stacked_state = make_state(
        orange_position=(0.3, 0.0, 0.06),
        blue_position=(0.3, 0.0, 0.02),
        table_contact=False,
        confirmed_grasp_seen=True,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(successful_stack_reward=100.0)
    )
    calculator.reset(initial_state)
    calculate_transition(calculator, initial_state, grasped_state)

    result = calculate_transition(
        calculator,
        grasped_state,
        stacked_state,
        succeeded=True,
    )

    assert set(result.components) == COMPONENT_NAMES
    assert result.total == pytest.approx(sum(result.components.values()))
    assert result.total == 100.0


def test_physical_stack_without_prior_grasp_is_not_task_success() -> None:
    initial_state = make_state()
    stacked_state = make_state(
        orange_position=(0.3, 0.0, 0.06),
        blue_position=(0.3, 0.0, 0.02),
        table_contact=False,
    )
    calculator = StackRewardCalculator(
        zero_reward_config(successful_stack_reward=100.0)
    )
    calculator.reset(initial_state)

    result = calculate_transition(
        calculator,
        initial_state,
        stacked_state,
        succeeded=True,
    )

    assert calculator.task_succeeded(True) is False
    assert result.components["successful_stack"] == 0.0


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value", "expected_message"),
    [
        ("approach_orange_progress_weight", -1.0, "nonnegative"),
        ("approach_orange_height_offset", -0.01, "nonnegative"),
        ("approach_orange_waypoint_tolerance", -0.01, "nonnegative"),
        ("approach_orange_capture_radius", -0.01, "nonnegative"),
        ("approach_orange_capture_radius", np.inf, "finite"),
        ("approach_orange_capture_radius", np.nan, "finite"),
        ("approach_orange_waypoint_reward", -0.01, "nonnegative"),
        ("grasp_reward", np.inf, "finite"),
        ("hold_orange_duration_weight", -0.01, "nonnegative"),
        ("lift_orange_height_weight", np.inf, "finite"),
        ("vertical_lift_margin", -0.01, "nonnegative"),
        ("dropped_cube_penalty", 1.0, "nonpositive"),
        ("ik_failure_penalty", np.nan, "finite"),
        ("gripper_state_change_penalty", np.inf, "finite"),
        ("unproductive_close_penalty", np.nan, "finite"),
        ("unproductive_close_penalty", 0.005, "nonpositive"),
    ],
)
def test_invalid_reward_config_raises(
    parameter_name: str,
    invalid_value: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        StackRewardConfig(**{parameter_name: invalid_value})


def test_gripper_state_change_shaping_may_be_positive() -> None:
    config = StackRewardConfig(
        gripper_state_change_penalty=0.005,
    )

    assert config.gripper_state_change_penalty == pytest.approx(0.005)


@pytest.mark.parametrize("invalid_hold_time", [np.nan, np.inf, -0.01])
def test_calculate_rejects_invalid_grasp_hold_time(
    invalid_hold_time: float,
) -> None:
    initial_state = make_state()
    invalid_state = make_state(
        orange_grasp_hold_time=invalid_hold_time,
    )
    calculator = StackRewardCalculator()
    calculator.reset(initial_state)

    with pytest.raises(ValueError, match="grasp hold time"):
        calculate_transition(calculator, initial_state, invalid_state)


@pytest.mark.parametrize(
    "invalid_action",
    [
        np.zeros(3),
        np.zeros(5),
        np.array([0.0, np.nan, 0.0, 0.0]),
        np.array([0.0, np.inf, 0.0, 0.0]),
    ],
)
def test_calculate_rejects_invalid_actions(
    invalid_action: np.ndarray,
) -> None:
    state = make_state()
    calculator = StackRewardCalculator()
    calculator.reset(state)

    with pytest.raises(ValueError):
        calculate_transition(
            calculator,
            state,
            state,
            action=invalid_action,
        )
