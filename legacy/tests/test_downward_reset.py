"""A reset must satisfy the same required claw orientation as policy actions."""

from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

import environment as environment_module
from action_observation_history import ActionObservationHistoryWrapper
from cartesian_actions import CartesianActionConfig, IKConvergenceError
from environment import (
    DEFAULT_JOINT_POSITIONS,
    DEFAULT_OPEN_GRIPPER_POSITION,
    DEFAULT_START_POSITION,
    CubeStackEnvironment,
)
from gym_environment import CubeStackGymEnvironment
from kinematics import (
    DEFAULT_TOOL_AXIS_TOLERANCE,
    DEFAULT_TOOL_YAW_TOLERANCE,
    solve_position_ik,
)
from robot_constants import ARM_JOINT_NAMES


HOME = np.array([0.20, 0.0, 0.05])
HALF_RANGE = np.array([0.04, 0.04, 0.0])


def assert_downward_pose(simulation: CubeStackEnvironment) -> None:
    """Measure the realized MuJoCo orientation, independent of IK flags."""
    rotation = simulation.data.site("gripperframe").xmat.reshape(3, 3)
    downward_error = np.arccos(np.clip(-rotation[2, 0], -1.0, 1.0))
    jaw_plane_error = np.arcsin(np.clip(abs(rotation[1, 2]), 0.0, 1.0))
    assert downward_error <= DEFAULT_TOOL_AXIS_TOLERANCE + 1e-10
    assert jaw_plane_error <= DEFAULT_TOOL_YAW_TOLERANCE + 1e-10
    for name in ARM_JOINT_NAMES:
        joint = simulation.model.joint(name)
        actuator = simulation.model.actuator(name)
        value = simulation.data.joint(name).qpos[0]
        assert max(joint.range[0], actuator.ctrlrange[0]) <= value
        assert value <= min(joint.range[1], actuator.ctrlrange[1])


@pytest.mark.parametrize("xy", [(0.0, 0.0), *product((-0.04, 0.04), repeat=2)])
def test_default_home_and_all_randomized_xy_corners_face_down(xy) -> None:
    target = HOME + np.array([*xy, 0.0])
    simulation = CubeStackEnvironment(start_position=target)
    state = simulation.reset(seed=3)

    np.testing.assert_array_equal(DEFAULT_START_POSITION, HOME)
    np.testing.assert_allclose(state["gripper_position"], target, atol=1e-6, rtol=0.0)
    assert_downward_pose(simulation)


def test_randomized_downward_resets_are_fresh_and_open() -> None:
    simulation = CubeStackEnvironment(start_position_half_range=HALF_RANGE)
    positions = []
    for seed in range(12):
        # A prior episode must not contribute time, motion, or success progress.
        simulation.data.time = 7.0
        simulation.data.qvel[:] = 0.2
        simulation.data.ctrl[:] = 0.5
        simulation._confirmed_grasp_seen = True
        simulation._orange_lifted_at_time = 1.0
        simulation._stack_success = True
        simulation._stack_stable_time = 0.5
        simulation._orange_fell_off_table = True
        simulation._blue_fell_off_table = True

        state = simulation.reset(seed=seed)
        position = state["gripper_position"]
        positions.append(position)
        assert np.all(np.abs(position - HOME) <= HALF_RANGE + 1e-6)
        assert position[2] == pytest.approx(HOME[2], abs=1e-6)
        assert_downward_pose(simulation)
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
    assert np.all(np.ptp(positions, axis=0)[:2] > 0.02)


def test_history_initial_target_matches_the_downward_reset_pose() -> None:
    environment = CubeStackGymEnvironment(start_position_half_range=HALF_RANGE)
    wrapper = ActionObservationHistoryWrapper(environment, history_length=4)
    try:
        wrapper.reset(seed=1)
        observation, info = wrapper.reset(seed=2)
        state = environment.previous_state
        assert state is not None
        target = state["gripper_position"]
        expected_token = np.concatenate((
            environment.observation_builder.build(state), np.zeros(4), target,
        )).astype(np.float32)

        assert_downward_pose(environment.simulation)
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


def test_disabled_requirement_preserves_position_only_reset(monkeypatch) -> None:
    def unexpected_orientation_solve(*args, **kwargs):
        pytest.fail("Disabling the requirement still called orientation IK.")

    monkeypatch.setattr(environment_module, "solve_position_and_tool_axis_ik", unexpected_orientation_solve)
    target = np.array([0.40, 0.0, 0.25])
    environment = CubeStackGymEnvironment(
        start_position=target,
        action_config=CartesianActionConfig(require_downward=False),
    )
    try:
        environment.reset(seed=1)
        expected = solve_position_ik(
            environment.simulation.model, DEFAULT_JOINT_POSITIONS[:-1], target,
            tolerance=1e-6,
        )
        assert expected.converged
        np.testing.assert_allclose(environment.previous_state["joint_positions"][:-1],
                                   expected.joint_positions, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(environment.previous_state["gripper_position"],
                                   target, atol=1e-6, rtol=0.0)
    finally:
        environment.close()


@pytest.mark.parametrize("yaw", [None, 0.0, 0.3])
def test_gym_passes_the_same_orientation_settings_to_reset_and_actions(monkeypatch, yaw) -> None:
    solve = environment_module.solve_position_and_tool_axis_ik
    calls = []

    def observe_solve(*args, **kwargs):
        calls.append(kwargs.copy())
        # The test checks configuration plumbing, not arbitrary yaw feasibility.
        return solve(*args, **{**kwargs, "target_tool_yaw": 0.0})

    monkeypatch.setattr(environment_module, "solve_position_and_tool_axis_ik", observe_solve)
    config = CartesianActionConfig(require_downward=True, target_tool_yaw=yaw)
    environment = CubeStackGymEnvironment(action_config=config, start_position_half_range=HALF_RANGE)
    try:
        environment.reset(seed=2)
        assert len(calls) == 2  # Cached center at construction, then sampled reset.
        assert all(call["require_downward"] is True for call in calls)
        assert all(call["target_tool_yaw"] == yaw for call in calls)
        assert environment.action_adapter.config is config
    finally:
        environment.close()


@pytest.mark.parametrize("failed_flag", [
    "position_converged", "tool_axis_converged", "tool_yaw_converged",
])
def test_failed_downward_reset_is_fatal_and_leaves_the_live_episode_untouched(
    monkeypatch, failed_flag,
) -> None:
    simulation = CubeStackEnvironment(start_position_half_range=HALF_RANGE)
    simulation.reset(seed=2)
    simulation.data.time = 4.5
    simulation.data.qvel[:] = 0.1
    simulation._confirmed_grasp_seen = True
    simulation._stack_stable_time = 0.4
    before = [simulation.data.qpos.copy(), simulation.data.qvel.copy(), simulation.data.ctrl.copy()]
    solve = environment_module.solve_position_and_tool_axis_ik

    def reject(*args, **kwargs):
        result = solve(*args, **kwargs)
        return replace(result, **{failed_flag: False})

    monkeypatch.setattr(environment_module, "solve_position_and_tool_axis_ik", reject)
    with pytest.raises(IKConvergenceError) as caught:
        simulation.reset(seed=3)

    assert isinstance(caught.value, RuntimeError)
    json.dumps(caught.value.diagnostics, allow_nan=False)
    assert caught.value.diagnostics["require_downward"] is True
    for actual, expected in zip((simulation.data.qpos, simulation.data.qvel, simulation.data.ctrl), before):
        np.testing.assert_array_equal(actual, expected)
    assert simulation.data.time == 4.5
    assert simulation.confirmed_grasp_seen
    assert simulation.stack_stable_time == 0.4


def test_unreachable_downward_home_raises_the_shared_convergence_exception() -> None:
    # Position-only IK reaches the old home, but its height prevents world-down.
    with pytest.raises(IKConvergenceError):
        CubeStackEnvironment(start_position=(0.40, 0.0, 0.25))
