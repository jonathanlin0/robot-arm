from pathlib import Path

import numpy as np
import pytest

import environment as environment_module
from action_observation_history import ActionObservationHistoryWrapper
from cartesian_actions import CartesianActionConfig
from environment import (
    DEFAULT_OPEN_GRIPPER_POSITION,
    DEFAULT_START_POSITION,
    CubeStackEnvironment,
    StateSnapshot,
)
from gym_environment import CubeStackGymEnvironment
from kinematics import IKResult


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
HALF_RANGE = (0.04, 0.04, 0.02)
UUID_SEED = (1 << 120) + 123456789
POSITION_TOLERANCE = 1e-6


def cube_layout(state: StateSnapshot) -> np.ndarray:
    return np.concatenate((state["orange_position"], state["blue_position"]))


def test_random_starts_vary_within_bounds() -> None:
    fixed = CubeStackEnvironment(scene_path=SCENE_PATH, seed=UUID_SEED)
    first = CubeStackEnvironment(
        scene_path=SCENE_PATH, seed=UUID_SEED, start_position_half_range=HALF_RANGE,
    )
    second = CubeStackEnvironment(
        scene_path=SCENE_PATH, seed=UUID_SEED, start_position_half_range=HALF_RANGE,
    )
    positions = []

    for _ in range(6):
        fixed_state = fixed.reset()
        first_state = first.reset()
        second_state = second.reset()
        position = first_state["gripper_position"]
        positions.append(position)

        np.testing.assert_array_equal(cube_layout(second_state), cube_layout(first_state))
        np.testing.assert_array_equal(second_state["joint_positions"], first_state["joint_positions"])
        assert np.all(np.abs(position - first.start_position)
                      <= first.start_position_half_range + POSITION_TOLERANCE)
        assert np.linalg.norm(fixed_state["gripper_position"] - DEFAULT_START_POSITION) <= POSITION_TOLERANCE

    assert np.all(np.ptp(positions, axis=0) > 0.001)


def test_reset_reseeds_environment_rng_with_full_uuid_seed() -> None:
    simulation = CubeStackEnvironment(
        scene_path=SCENE_PATH, start_position_half_range=HALF_RANGE,
    )
    first = simulation.reset(seed=UUID_SEED)
    second = simulation.reset()
    simulation.reset()
    repeated_first = simulation.reset(seed=UUID_SEED)
    repeated_second = simulation.reset()

    for expected, actual in ((first, repeated_first), (second, repeated_second)):
        np.testing.assert_array_equal(actual["joint_positions"], expected["joint_positions"])
        np.testing.assert_array_equal(actual["gripper_position"], expected["gripper_position"])
        np.testing.assert_array_equal(cube_layout(actual), cube_layout(expected))

    truncated_seed_state = simulation.reset(seed=UUID_SEED % (1 << 64))
    assert not np.allclose(truncated_seed_state["gripper_position"], first["gripper_position"])


def test_random_reset_restores_fresh_dynamics_and_matching_open_gripper_controls() -> None:
    simulation = CubeStackEnvironment(
        scene_path=SCENE_PATH, start_position_half_range=HALF_RANGE,
    )
    first_state = simulation.reset(seed=15)
    simulation.data.qvel[:] = 0.1
    simulation.data.ctrl[:] = 0.5
    simulation.step_physics(5)
    assert simulation.data.time > 0.0
    simulation._confirmed_grasp_seen = True
    simulation._orange_lifted_at_time = 0.0
    simulation._stack_success = True
    simulation._stack_stable_time = 0.5
    simulation._orange_fell_off_table = True

    state = simulation.reset()

    assert not np.allclose(state["gripper_position"], first_state["gripper_position"])
    assert state["time"] == 0.0
    np.testing.assert_array_equal(simulation.data.qvel, 0.0)
    np.testing.assert_array_equal(state["controls"], state["joint_positions"])
    assert state["joint_positions"][-1] == DEFAULT_OPEN_GRIPPER_POSITION
    assert state["gripper_target"] == DEFAULT_OPEN_GRIPPER_POSITION
    assert not state["confirmed_grasp_seen"]
    assert state["orange_grasp_hold_time"] == 0.0
    assert simulation.stack_stable_time == 0.0
    assert not simulation.is_success()
    assert not simulation.is_failure()


def test_custom_start_center_and_zero_half_range_axis_are_respected() -> None:
    center = np.array([0.39, 0.015, 0.24])
    half_range = np.array([0.01, 0.0, 0.01])
    simulation = CubeStackEnvironment(
        scene_path=SCENE_PATH, start_position=center,
        start_position_half_range=half_range,
        require_downward=False,  # This high pose exercises position-only reset.
    )

    np.testing.assert_array_equal(simulation.start_position, center)
    np.testing.assert_array_equal(simulation.start_position_half_range, half_range)
    for seed in range(3):
        position = simulation.reset(seed=seed)["gripper_position"]
        assert np.all(np.abs(position - center) <= half_range + POSITION_TOLERANCE)
        assert position[1] == pytest.approx(center[1], abs=POSITION_TOLERANCE)


def test_sampled_ik_failure_is_reported_instead_of_using_fixed_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = CubeStackEnvironment(
        scene_path=SCENE_PATH, start_position_half_range=HALF_RANGE,
        require_downward=False,
    )
    failed_solution = IKResult(
        joint_positions=np.zeros(5), converged=False, position_error=0.1, iterations=100,
    )
    monkeypatch.setattr(environment_module, "solve_position_ik", lambda *args, **kwargs: failed_solution)

    with pytest.raises(RuntimeError, match="(?i)start"):
        simulation.reset(seed=10)


def test_gym_and_history_reset_use_the_sampled_pose_as_the_initial_target() -> None:
    environment = CubeStackGymEnvironment(
        scene_path=SCENE_PATH, start_position=(0.39, 0.0, 0.25),
        start_position_half_range=HALF_RANGE,
        action_config=CartesianActionConfig(require_downward=False),
    )
    wrapper = ActionObservationHistoryWrapper(environment, history_length=4)
    try:
        initial, _ = wrapper.reset(seed=UUID_SEED)
        wrapper.step(np.array([0.2, -0.1, 0.0, -1.0], dtype=np.float32))
        observation, info = wrapper.reset(seed=UUID_SEED + 1)
        state = environment.previous_state
        assert state is not None
        target = state["gripper_position"]
        expected_state = environment.observation_builder.build(state)
        expected_token = np.concatenate((expected_state, np.zeros(4), target)).astype(np.float32)

        assert not np.allclose(initial["tokens"][0, -3:], target)
        assert environment.episode_step_count == 0
        assert state["time"] == 0.0
        np.testing.assert_array_equal(info["target_gripper_position"], target)
        np.testing.assert_array_equal(environment.action_adapter.current_target_gripper_position, target)
        np.testing.assert_array_equal(environment.action_adapter.previous_target_gripper_position, target)
        np.testing.assert_array_equal(observation["tokens"][0], expected_token)
        np.testing.assert_array_equal(observation["tokens"][1:], 0.0)
        np.testing.assert_array_equal(observation["valid"], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(observation["episode_start"], [1.0, 0.0, 0.0, 0.0])
    finally:
        wrapper.close()
