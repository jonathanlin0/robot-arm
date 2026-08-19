from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment import (
    DEFAULT_JOINT_POSITIONS,
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

    state = environment.reset()

    assert state["time"] == 0.0
    np.testing.assert_array_equal(environment.data.qvel, 0.0)
    np.testing.assert_array_equal(state["controls"], DEFAULT_JOINT_POSITIONS)
    np.testing.assert_array_equal(state["orange_velocity"], 0.0)
    np.testing.assert_array_equal(state["blue_velocity"], 0.0)


def test_reset_uses_default_joint_positions(
    environment: CubeStackEnvironment,
) -> None:
    expected_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

    state = environment.reset(seed=12)

    np.testing.assert_array_equal(
        DEFAULT_JOINT_POSITIONS,
        expected_positions,
    )
    np.testing.assert_array_equal(
        state["joint_positions"],
        expected_positions,
    )
    np.testing.assert_array_equal(
        state["controls"],
        expected_positions,
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
