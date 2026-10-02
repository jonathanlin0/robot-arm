from pathlib import Path

import numpy as np
import pytest

from environment import CubeStackEnvironment, StateSnapshot
from observations import (
    PRIVILEGED_OBSERVATION_SIZE,
    PrivilegedObservationBuilder,
)


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


def state_with_distinct_values(
    environment: CubeStackEnvironment,
) -> StateSnapshot:
    state = environment.reset(seed=1)
    state["time"] = 999.0
    state["joint_positions"] = np.arange(0.0, 6.0)
    state["joint_velocities"] = np.arange(10.0, 16.0)
    state["controls"] = np.arange(20.0, 26.0)
    state["gripper_position"] = np.arange(30.0, 33.0)
    state["orange_position"] = np.arange(40.0, 43.0)
    state["orange_orientation"] = np.arange(50.0, 54.0)
    state["orange_velocity"] = np.arange(60.0, 66.0)
    state["blue_position"] = np.arange(70.0, 73.0)
    state["blue_orientation"] = np.arange(80.0, 84.0)
    state["blue_velocity"] = np.arange(90.0, 96.0)
    state["orange_touches_fixed_jaw"] = True
    state["orange_touches_moving_jaw"] = False
    return state


def test_build_uses_documented_field_order(
    environment: CubeStackEnvironment,
) -> None:
    state = state_with_distinct_values(environment)
    builder = PrivilegedObservationBuilder(environment)

    observation = builder.build(state)

    expected = np.concatenate(
        (
            np.arange(0.0, 6.0),
            np.arange(10.0, 16.0),
            np.arange(20.0, 26.0),
            np.arange(30.0, 33.0),
            np.arange(40.0, 43.0),
            np.arange(50.0, 54.0),
            np.arange(60.0, 66.0),
            np.arange(70.0, 73.0),
            np.arange(80.0, 84.0),
            np.arange(90.0, 96.0),
            np.array([1.0, 0.0]),
        )
    ).astype(np.float32)

    np.testing.assert_array_equal(observation, expected)
    assert PRIVILEGED_OBSERVATION_SIZE == 49
    assert observation.shape == (PRIVILEGED_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32


def test_build_accepts_real_environment_snapshot(
    environment: CubeStackEnvironment,
) -> None:
    state = environment.reset(seed=2)

    observation = PrivilegedObservationBuilder(environment).build(state)

    assert observation.shape == (PRIVILEGED_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))


def test_build_returns_an_independent_array(
    environment: CubeStackEnvironment,
) -> None:
    state = state_with_distinct_values(environment)
    builder = PrivilegedObservationBuilder(environment)
    observation = builder.build(state)
    saved_observation = observation.copy()

    for value in state.values():
        if isinstance(value, np.ndarray):
            value[:] += 1.0

    np.testing.assert_array_equal(observation, saved_observation)

    state_after_mutation = {
        name: value.copy()
        for name, value in state.items()
        if isinstance(value, np.ndarray)
    }
    observation[:] = -1.0
    for name, expected_value in state_after_mutation.items():
        np.testing.assert_array_equal(state[name], expected_value)


@pytest.mark.parametrize(
    ("fixed_contact", "moving_contact", "expected"),
    [
        (False, False, [0.0, 0.0]),
        (True, False, [1.0, 0.0]),
        (False, True, [0.0, 1.0]),
        (True, True, [1.0, 1.0]),
    ],
)
def test_build_encodes_orange_jaw_contacts(
    environment: CubeStackEnvironment,
    fixed_contact: bool,
    moving_contact: bool,
    expected: list[float],
) -> None:
    state = state_with_distinct_values(environment)
    state["orange_touches_fixed_jaw"] = fixed_contact
    state["orange_touches_moving_jaw"] = moving_contact

    observation = PrivilegedObservationBuilder(environment).build(state)

    np.testing.assert_array_equal(
        observation[-2:],
        np.asarray(expected, dtype=np.float32),
    )


def test_build_rejects_wrong_observation_size(
    environment: CubeStackEnvironment,
) -> None:
    state = state_with_distinct_values(environment)
    state["controls"] = np.zeros(5)

    with pytest.raises(ValueError, match=r"shape \(49,\)"):
        PrivilegedObservationBuilder(environment).build(state)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_build_rejects_non_finite_values(
    environment: CubeStackEnvironment,
    invalid_value: float,
) -> None:
    state = state_with_distinct_values(environment)
    state["orange_velocity"][2] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        PrivilegedObservationBuilder(environment).build(state)
