from pathlib import Path

import mujoco
import numpy as np
import pytest

import environment as environment_module
from environment import (
    DEFAULT_JOINT_POSITIONS,
    DEFAULT_START_POSITION,
    OFF_TABLE_HEIGHT_TOLERANCE,
    ROBOT_JOINT_NAMES,
    CubeStackEnvironment,
    StateSnapshot,
)


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
JOINT_COUNT = len(ROBOT_JOINT_NAMES)
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")


@pytest.fixture
def environment() -> CubeStackEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    return CubeStackEnvironment(scene_path=SCENE_PATH)


def cube_layout(state: StateSnapshot) -> np.ndarray:
    return np.concatenate(
        [state["orange_position"], state["blue_position"]]
    )


def safe_joint_target_bounds(
    environment: CubeStackEnvironment,
) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = []
    upper_bounds = []

    for name in ROBOT_JOINT_NAMES:
        actuator_range = environment.model.actuator(name).ctrlrange
        joint_range = environment.model.joint(name).range
        lower_bounds.append(max(actuator_range[0], joint_range[0]))
        upper_bounds.append(min(actuator_range[1], joint_range[1]))

    return np.array(lower_bounds), np.array(upper_bounds)


def test_reset_reseeding_repeats_layout(
    environment: CubeStackEnvironment,
) -> None:
    first_layout = cube_layout(environment.reset(seed=42))
    environment.reset()
    repeated_layout = cube_layout(environment.reset(seed=42))

    np.testing.assert_array_equal(first_layout, repeated_layout)


def test_seeded_environments_produce_the_same_sequence() -> None:
    first_environment = CubeStackEnvironment(scene_path=SCENE_PATH, seed=7)
    second_environment = CubeStackEnvironment(scene_path=SCENE_PATH, seed=7)

    for _ in range(10):
        first_layout = cube_layout(first_environment.reset())
        second_layout = cube_layout(second_environment.reset())
        np.testing.assert_array_equal(first_layout, second_layout)


def test_reset_placements_are_valid(
    environment: CubeStackEnvironment,
) -> None:
    config = environment.spawn_config

    for _ in range(100):
        state = environment.reset()
        orange_position = state["orange_position"]
        blue_position = state["blue_position"]

        for position in (orange_position, blue_position):
            assert config.x_range[0] <= position[0] <= config.x_range[1]
            assert config.y_range[0] <= position[1] <= config.y_range[1]
            assert position[2] == pytest.approx(config.cube_center_z)

        center_distance = np.linalg.norm(
            orange_position[:2] - blue_position[:2]
        )
        assert center_distance >= config.minimum_center_distance


def test_reset_clears_dynamics_and_controls(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=11)
    environment.data.qvel[:] = 1.0
    environment.data.ctrl[:] = 0.5
    environment.step_physics(5)

    assert environment.data.time > 0.0
    environment._stack_stable_time = 0.5
    environment._stack_success = True
    environment._confirmed_grasp_seen = True
    environment._orange_lifted_at_time = 0.0
    environment._orange_fell_off_table = True
    environment._blue_fell_off_table = True

    state = environment.reset()

    assert state["time"] == 0.0
    np.testing.assert_array_equal(environment.data.qvel, 0.0)
    np.testing.assert_array_equal(state["controls"], state["joint_positions"])
    assert state["gripper_target"] == pytest.approx(
        DEFAULT_JOINT_POSITIONS[-1]
    )
    np.testing.assert_array_equal(state["orange_velocity"], 0.0)
    np.testing.assert_array_equal(state["blue_velocity"], 0.0)
    assert environment.stack_stable_time == 0.0
    assert not environment.is_success()
    assert not environment.is_failure()
    assert not state["confirmed_grasp_seen"]
    assert not state["orange_fell_off_table"]
    assert not state["blue_fell_off_table"]
    assert state["orange_grasp_hold_time"] == 0.0


def test_reset_places_gripper_at_default_start_with_matching_controls(
    environment: CubeStackEnvironment,
) -> None:
    state = environment.reset(seed=12)

    assert np.linalg.norm(state["gripper_position"] - DEFAULT_START_POSITION) <= 1e-6
    assert state["joint_positions"][-1] == DEFAULT_JOINT_POSITIONS[-1]
    np.testing.assert_array_equal(
        state["controls"],
        state["joint_positions"],
    )
    lower_bounds, upper_bounds = safe_joint_target_bounds(environment)
    assert np.all(state["joint_positions"] >= lower_bounds)
    assert np.all(state["joint_positions"] <= upper_bounds)


def test_start_position_hyperparameter_changes_the_reset_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_position = (0.39, 0.015, 0.24)
    monkeypatch.setattr(environment_module, "DEFAULT_START_POSITION", configured_position)
    # This high pose tests configurable XYZ in legacy position-only mode.
    simulation = CubeStackEnvironment(scene_path=SCENE_PATH, require_downward=False)

    state = simulation.reset(seed=12)

    assert np.linalg.norm(state["gripper_position"] - configured_position) <= 1e-6
    np.testing.assert_array_equal(state["controls"], state["joint_positions"])
    assert state["time"] == 0.0


def test_unreachable_start_position_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment_module, "DEFAULT_START_POSITION", (10.0, 0.0, 0.25))

    with pytest.raises(ValueError, match="(?i)start"):
        CubeStackEnvironment(scene_path=SCENE_PATH, require_downward=False)


def test_state_exposes_named_gripper_actuator_target(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=13)
    environment.data.actuator("gripper").ctrl[0] = 0.17

    state = environment.get_state()

    assert state["gripper_target"] == pytest.approx(0.17)


def test_state_exposes_rigid_gripperframe_and_live_jaw_midpoint(
    environment: CubeStackEnvironment,
) -> None:
    state = environment.reset(seed=14)
    fixed_jaw_center = np.mean(
        [
            environment.data.geom(
                f"fixed_jaw_sph_tip{tip_index}"
            ).xpos
            for tip_index in range(1, 4)
        ],
        axis=0,
    )
    moving_jaw_center = np.mean(
        [
            environment.data.geom(
                f"moving_jaw_sph_tip{tip_index}"
            ).xpos
            for tip_index in range(1, 4)
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        state["gripper_position"],
        environment.data.site("gripperframe").xpos,
    )
    np.testing.assert_allclose(
        state["jaw_midpoint"],
        (fixed_jaw_center + moving_jaw_center) / 2.0,
    )


def test_gripperframe_is_rigid_while_jaw_midpoint_moves_with_gripper_joint(
    environment: CubeStackEnvironment,
) -> None:
    open_state = environment.reset(seed=15)

    environment.data.joint("gripper").qpos[0] = -0.1
    mujoco.mj_forward(environment.model, environment.data)
    closed_state = environment.get_state()

    np.testing.assert_allclose(
        closed_state["gripper_position"],
        open_state["gripper_position"],
    )
    assert not np.allclose(
        closed_state["jaw_midpoint"],
        open_state["jaw_midpoint"],
    )


def test_reset_updates_derived_body_poses(
    environment: CubeStackEnvironment,
) -> None:
    """
    check that the snapshot's world-space body poses match.

    mainly testing the mj_forward call in environment.rest()
    """
    state = environment.reset(seed=21)

    orange_qpos = environment.data.joint("orange_cube_joint").qpos
    blue_qpos = environment.data.joint("blue_cube_joint").qpos

    np.testing.assert_allclose(state["orange_position"], orange_qpos[:3])
    np.testing.assert_allclose(state["orange_orientation"], orange_qpos[3:])
    np.testing.assert_allclose(state["blue_position"], blue_qpos[:3])
    np.testing.assert_allclose(state["blue_orientation"], blue_qpos[3:])


def test_snapshot_arrays_are_independent_copies(
    environment: CubeStackEnvironment,
) -> None:
    """
    Make sure 
    """
    snapshot = environment.reset(seed=99)
    saved_snapshot = {
        name: value.copy()
        for name, value in snapshot.items()
        if isinstance(value, np.ndarray)
    }

    environment.reset()

    # check that env reset don't modify old snapshots
    for name, saved_value in saved_snapshot.items():
        np.testing.assert_array_equal(snapshot[name], saved_value)

    state_before_mutation = environment.get_state()

    for value in snapshot.values():
        if isinstance(value, np.ndarray):
            value[:] += 1.0

    # check that modifying generated snapshot doesn't change
    # other snapshots or new snapshots
    state_after_mutation = environment.get_state()
    for name, before_value in state_before_mutation.items():
        if isinstance(before_value, np.ndarray):
            np.testing.assert_array_equal(
                state_after_mutation[name],
                before_value,
            )


@pytest.mark.parametrize(
    "invalid_targets",
    [
        np.zeros(JOINT_COUNT - 1),
        np.zeros(JOINT_COUNT + 1),
        np.zeros((JOINT_COUNT, 1)),
    ],
)
def test_step_joint_targets_rejects_wrong_shape(
    environment: CubeStackEnvironment,
    invalid_targets: np.ndarray,
) -> None:
    environment.reset(seed=1)
    controls_before = environment.data.ctrl.copy()

    with pytest.raises(
        ValueError,
        match=rf"shape \({JOINT_COUNT},\)",
    ):
        environment.step_joint_targets(invalid_targets)

    np.testing.assert_array_equal(environment.data.ctrl, controls_before)
    assert environment.data.time == 0.0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_step_joint_targets_rejects_non_finite_values(
    environment: CubeStackEnvironment,
    invalid_value: float,
) -> None:
    environment.reset(seed=1)
    controls_before = environment.data.ctrl.copy()
    targets = np.zeros(JOINT_COUNT)
    targets[2] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        environment.step_joint_targets(targets)

    np.testing.assert_array_equal(environment.data.ctrl, controls_before)
    assert environment.data.time == 0.0


def test_step_joint_targets_sets_controls_and_advances_time(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=2)
    targets = np.array([0.20, -0.25, 0.30, -0.20, 0.40, 0.50])

    state = environment.step_joint_targets(targets)

    for action_index, actuator_name in enumerate(ROBOT_JOINT_NAMES):
        actuator_id = environment.model.actuator(actuator_name).id
        control_index = environment.model.actuator_ctrladr[actuator_id]
        assert environment.data.ctrl[control_index] == pytest.approx(
            targets[action_index]
        )

    np.testing.assert_allclose(state["controls"], targets)
    assert state["gripper_target"] == pytest.approx(targets[-1])
    assert state["time"] == pytest.approx(
        10 * environment.model.opt.timestep
    )


@pytest.mark.parametrize(
    ("requested_value", "bound_index"),
    [(100.0, 1), (-100.0, 0)],
)
def test_step_joint_targets_clips_to_safe_limits(
    environment: CubeStackEnvironment,
    requested_value: float,
    bound_index: int,
) -> None:
    environment.reset(seed=3)
    lower_bounds, upper_bounds = safe_joint_target_bounds(environment)
    expected_targets = (lower_bounds, upper_bounds)[bound_index]

    state = environment.step_joint_targets(
        np.full(JOINT_COUNT, requested_value)
    )

    np.testing.assert_allclose(state["controls"], expected_targets)


def test_step_joint_targets_does_not_teleport_joints(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=4)
    initial_positions = initial_state["joint_positions"].copy()
    targets = initial_positions.copy()
    targets[0] = 0.75

    positions_before_each_step = []
    original_mj_step = mujoco.mj_step

    def record_then_step(
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        positions_before_each_step.append(
            np.array(
                [data.joint(name).qpos[0] for name in ROBOT_JOINT_NAMES]
            )
        )
        original_mj_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", record_then_step)

    state = environment.step_joint_targets(targets)

    assert len(positions_before_each_step) == 10
    np.testing.assert_array_equal(
        positions_before_each_step[0],
        initial_positions,
    )
    assert state["joint_positions"][0] > initial_positions[0]
    assert state["joint_positions"][0] < targets[0]


@pytest.mark.parametrize("cube_name", ["orange", "blue"])
def test_off_table_failure_is_recorded_without_prior_grasp(
    environment: CubeStackEnvironment,
    cube_name: str,
) -> None:
    initial_state = environment.reset(seed=5)
    cube_joint = environment.data.joint(f"{cube_name}_cube_joint")
    cube_joint.qpos[0] = 1.0
    cube_joint.qpos[2] = (
        initial_state[f"{cube_name}_position"][2]
        - OFF_TABLE_HEIGHT_TOLERANCE
        - 0.001
    )
    cube_joint.qvel.fill(0.0)
    mujoco.mj_forward(environment.model, environment.data)

    environment.step_physics(1)
    failed_state = environment.get_state()

    assert environment.confirmed_grasp_seen is False
    assert getattr(environment, f"{cube_name}_fell_off_table") is True
    assert environment.is_failure() is True
    assert environment.is_terminated() is True
    assert failed_state[f"{cube_name}_fell_off_table"] is True


def test_off_table_failure_threshold_is_strict(
    environment: CubeStackEnvironment,
) -> None:
    initial_state = environment.reset(seed=6)
    orange_joint = environment.data.joint("orange_cube_joint")
    threshold_height = (
        initial_state["orange_position"][2]
        - OFF_TABLE_HEIGHT_TOLERANCE
    )
    orange_joint.qpos[2] = threshold_height
    mujoco.mj_forward(environment.model, environment.data)

    environment._update_off_table_failure()

    assert environment.is_failure() is False

    orange_joint.qpos[2] = threshold_height - 1e-6
    mujoco.mj_forward(environment.model, environment.data)
    environment._update_off_table_failure()

    assert environment.is_failure() is True


def test_off_table_failure_remains_latched_until_reset(
    environment: CubeStackEnvironment,
) -> None:
    initial_state = environment.reset(seed=7)
    orange_joint = environment.data.joint("orange_cube_joint")
    orange_joint.qpos[2] = (
        initial_state["orange_position"][2]
        - OFF_TABLE_HEIGHT_TOLERANCE
        - 0.001
    )
    mujoco.mj_forward(environment.model, environment.data)
    environment._update_off_table_failure()
    assert environment.is_failure() is True

    orange_joint.qpos[2] = initial_state["orange_position"][2]
    mujoco.mj_forward(environment.model, environment.data)
    environment._update_off_table_failure()

    assert environment.is_failure() is True

    reset_state = environment.reset(seed=8)

    assert environment.orange_fell_off_table is False
    assert environment.blue_fell_off_table is False
    assert environment.is_failure() is False
    assert environment.is_terminated() is False
    assert reset_state["orange_fell_off_table"] is False
    assert reset_state["blue_fell_off_table"] is False
