from pathlib import Path

import numpy as np
import pytest

from environment import CubeStackEnvironment, StateSnapshot


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
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
    np.testing.assert_array_equal(state["controls"], 0.0)
    np.testing.assert_array_equal(state["orange_velocity"], 0.0)
    np.testing.assert_array_equal(state["blue_velocity"], 0.0)


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
