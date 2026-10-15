"""Waypoint preparation stays outside policy episodes and preserves control state.

Physical preparation tests use best-effort IK to isolate reset/history behavior
from strict downward-orientation feasibility at the configured home pose.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from action_observation_history import ActionObservationHistoryWrapper
from cartesian_actions import CartesianActionConfig
from environment import DEFAULT_OPEN_GRIPPER_POSITION, DEFAULT_START_POSITION
from gym_environment import CubeStackGymEnvironment
from kinematics import DEFAULT_POSITION_TOLERANCE


def waypoint(environment, state):
    target = np.asarray(state["orange_position"]).copy()
    target[2] += environment.reward_config.approach_orange_height_offset
    return target


def assert_prepared(environment):
    state = environment.previous_state
    assert state is not None
    target = waypoint(environment, state)
    assert np.linalg.norm(state["gripper_position"] - target) <= 0.005
    assert np.linalg.norm(
        environment.action_adapter.current_target_gripper_position - target
    ) <= 0.001
    assert state["gripper_target"] == pytest.approx(
        environment.action_adapter.config.open_gripper_target
    )
    assert state["joint_positions"][-1] == pytest.approx(
        environment.action_adapter.config.open_gripper_target, abs=0.01
    )
    assert not state["orange_currently_held"]
    assert not state["confirmed_grasp_seen"]
    assert state["orange_grasp_hold_time"] == 0.0
    assert not environment.simulation.is_success()
    assert not environment.simulation.is_failure()
    assert environment.reward_calculator.orange_pregrasp_waypoint_reached
    assert environment.episode_step_count == 0


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_reset_physically_prepares_open_gripper_without_awarding_rewards(
    seed, monkeypatch
):
    environment = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
    )
    results = []
    original_step = environment.action_adapter.step

    def record_step(action):
        assert action[-1] == 1.0
        result = original_step(action)
        results.append(result)
        return result

    def unexpected_reward(**kwargs):
        pytest.fail("Preparation must not calculate policy rewards.")

    monkeypatch.setattr(environment.action_adapter, "step", record_step)
    monkeypatch.setattr(environment.reward_calculator, "calculate", unexpected_reward)

    observation, info = environment.reset(seed=seed)

    assert_prepared(environment)
    assert 5 <= len(results) <= 200
    assert environment.previous_state["time"] > 0.0
    assert observation.shape == (49,)
    assert environment.previous_state is results[-1].state
    # The accepted target must survive preparation, including actuator lag.
    np.testing.assert_array_equal(
        info["target_gripper_position"], results[-1].target_gripper_position
    )
    np.testing.assert_array_equal(
        environment.action_adapter.current_target_gripper_position,
        results[-1].target_gripper_position,
    )


def test_default_reset_restores_configured_home_and_unreached_waypoint():
    environment = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
    )

    observation, info = environment.reset(seed=0)

    state = environment.previous_state
    assert np.linalg.norm(state["gripper_position"] - DEFAULT_START_POSITION) <= DEFAULT_POSITION_TOLERANCE
    np.testing.assert_array_equal(
        info["target_gripper_position"], state["gripper_position"]
    )
    assert state["joint_positions"][-1] == DEFAULT_OPEN_GRIPPER_POSITION
    assert state["gripper_target"] == DEFAULT_OPEN_GRIPPER_POSITION
    np.testing.assert_array_equal(state["joint_velocities"], 0.0)
    assert state["time"] == 0.0
    assert environment.episode_step_count == 0
    assert info["episode_start_type"] == "home"
    assert observation.shape == (49,)
    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached

    environment.step(np.array([0.2, 0.1, -1.0, -1.0], dtype=np.float32))
    repeated, repeated_info = environment.reset(seed=0)
    np.testing.assert_array_equal(repeated, observation)
    np.testing.assert_array_equal(
        repeated_info["target_gripper_position"], info["target_gripper_position"]
    )
    assert environment.previous_state["time"] == 0.0
    assert environment.episode_step_count == 0
    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached


def test_prepared_resets_are_repeatable_and_clear_previous_episode_history():
    base = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
    )
    environment = ActionObservationHistoryWrapper(base, history_length=64)
    first, first_info = environment.reset(seed=3)
    environment.step(np.array([0.2, 0.1, -1.0, 1.0], dtype=np.float32))
    environment.reset(seed=11)

    repeated, info = environment.reset(seed=3)

    assert_prepared(base)
    for key in first:
        np.testing.assert_array_equal(repeated[key], first[key])
    np.testing.assert_array_equal(repeated["valid"], [1.0] + [0.0] * 63)
    np.testing.assert_array_equal(repeated["episode_start"], repeated["valid"])
    np.testing.assert_array_equal(repeated["tokens"][1:], 0.0)
    np.testing.assert_array_equal(repeated["tokens"][0, -7:-3], 0.0)
    np.testing.assert_array_equal(
        repeated["tokens"][0, -3:],
        np.asarray(info["target_gripper_position"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        info["target_gripper_position"], first_info["target_gripper_position"]
    )


def test_first_descent_earns_approach_progress_without_a_waypoint_bonus():
    environment = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
    )
    environment.reset(seed=0)

    _, _, terminated, truncated, info = environment.step(
        np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    )

    assert info["reward_components"]["approach_orange_progress"] > 0.0
    assert info["reward_components"]["approach_orange_waypoint"] == 0.0
    assert info["orange_pregrasp_waypoint_reached"] is True
    assert environment.episode_step_count == 1
    assert not terminated
    assert not truncated


def test_preparation_does_not_consume_the_policy_step_budget():
    environment = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True, maximum_episode_steps=3
    )
    environment.reset(seed=0)

    for step in range(1, 4):
        _, _, terminated, truncated, _ = environment.step(np.zeros(4))
        assert environment.episode_step_count == step
        assert not terminated
        assert truncated is (step == 3)


def test_vector_autoresets_prepare_each_environment_and_restart_its_history():
    def make_environment():
        base = CubeStackGymEnvironment(
            action_config=CartesianActionConfig(require_downward=False),
            start_at_orange_waypoint=True, maximum_episode_steps=1
        )
        history = ActionObservationHistoryWrapper(base, history_length=4)
        return Monitor(
            history, info_keywords=("orange_pregrasp_waypoint_reached",)
        )

    vector = DummyVecEnv([make_environment, make_environment])
    try:
        vector.seed(0)
        vector.reset()
        observation, _, done, infos = vector.step(np.zeros((2, 4)))

        np.testing.assert_array_equal(done, [True, True])
        for index, info in enumerate(infos):
            assert_prepared(vector.envs[index].unwrapped)
            assert info["episode"]["l"] == 1
            assert info["episode"]["orange_pregrasp_waypoint_reached"]
            np.testing.assert_array_equal(
                observation["valid"][index], [1, 0, 0, 0]
            )
            np.testing.assert_array_equal(
                info["terminal_observation"]["valid"], [1, 1, 0, 0]
            )
    finally:
        vector.close()


def test_stalled_controller_fails_after_bounded_preparation(monkeypatch):
    environment = CubeStackGymEnvironment(start_at_orange_waypoint=True)
    step_count = 0

    def stalled_step(action):
        nonlocal step_count
        step_count += 1
        return SimpleNamespace(state=environment.simulation.get_state())

    monkeypatch.setattr(environment.action_adapter, "step", stalled_step)

    with pytest.raises(RuntimeError, match="(?i)waypoint"):
        environment.reset(seed=0)

    assert step_count == 200
    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached


@pytest.mark.parametrize("invalid_event", ["grasp", "failure"])
def test_preparation_rejects_invalid_episodes(monkeypatch, invalid_event):
    environment = CubeStackGymEnvironment(start_at_orange_waypoint=True)
    step_count = 0

    def invalid_step(action):
        nonlocal step_count
        step_count += 1
        if invalid_event == "grasp":
            environment.simulation._confirmed_grasp_seen = True
        else:
            environment.simulation._orange_fell_off_table = True
        return SimpleNamespace(state=environment.simulation.get_state())

    monkeypatch.setattr(environment.action_adapter, "step", invalid_step)

    with pytest.raises(RuntimeError):
        environment.reset(seed=0)

    assert step_count == 1
    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("closed", [False, True])
def test_recovery_reset_prepares_a_settled_unheld_cube_and_gripper(
    seed, closed, monkeypatch
):
    environment = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
        recovery_start_probability=1.0,
        recovery_closed_gripper_probability=float(closed),
    )
    results = []
    original_step = environment.action_adapter.step

    def record_step(action):
        result = original_step(action)
        results.append(result)
        return result

    def unexpected_reward(**kwargs):
        pytest.fail("Recovery preparation must not calculate policy rewards.")

    monkeypatch.setattr(environment.action_adapter, "step", record_step)
    monkeypatch.setattr(environment.reward_calculator, "calculate", unexpected_reward)

    observation, info = environment.reset(seed=seed)

    state = environment.previous_state
    assert state is results[-1].state
    target = environment.action_adapter.current_target_gripper_position
    offset = target - state["orange_position"]
    recovery = environment.recovery_start_config
    assert (
        recovery.xy_offset_range[0] - 0.005
        <= np.linalg.norm(offset[:2])
        <= recovery.xy_offset_range[1] + 0.005
    )
    assert (
        recovery.height_offset_range[0] - 0.005
        <= offset[2]
        <= recovery.height_offset_range[1] + 0.005
    )
    assert np.linalg.norm(state["gripper_position"] - target) <= 0.006
    gripper_target = (
        environment.action_adapter.config.closed_gripper_target
        if closed else environment.action_adapter.config.open_gripper_target
    )
    assert state["gripper_target"] == pytest.approx(gripper_target)
    assert state["joint_positions"][-1] == pytest.approx(gripper_target, abs=0.01)
    assert state["orange_touches_table"]
    assert not state["orange_currently_held"]
    assert not state["confirmed_grasp_seen"]
    assert state["orange_grasp_hold_time"] == 0.0
    assert not environment.simulation.is_failure()
    assert not environment.simulation.is_success()
    assert environment.reward_calculator.orange_pregrasp_waypoint_reached
    assert environment.episode_step_count == 0
    assert observation.shape == (49,)
    assert info["episode_start_type"] == (
        "recovery_closed" if closed else "recovery_open"
    )
    np.testing.assert_array_equal(info["target_gripper_position"], target)
    np.testing.assert_array_equal(target, results[-1].target_gripper_position)
    # Preparation physically visits the normal waypoint before placing the
    # gripper beside the cube, so the waypoint latch describes a real visit.
    assert any(
        np.linalg.norm(result.state["gripper_position"] - waypoint(environment, result.state))
        <= 0.005
        for result in results
    )


@pytest.mark.parametrize("closed", [False, True])
def test_recovery_resets_are_repeatable_with_clean_history_and_full_step_budget(closed):
    base = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
        recovery_start_probability=1.0,
        recovery_closed_gripper_probability=float(closed),
        maximum_episode_steps=3,
    )
    environment = ActionObservationHistoryWrapper(base, history_length=4)
    first, first_info = environment.reset(seed=3)
    environment.step(np.zeros(4))
    environment.reset(seed=11)

    repeated, info = environment.reset(seed=3)

    for key in first:
        np.testing.assert_array_equal(repeated[key], first[key])
    assert info["episode_start_type"] == first_info["episode_start_type"]
    np.testing.assert_array_equal(
        info["target_gripper_position"], first_info["target_gripper_position"]
    )
    np.testing.assert_array_equal(repeated["valid"], [1, 0, 0, 0])
    np.testing.assert_array_equal(repeated["episode_start"], repeated["valid"])
    np.testing.assert_array_equal(repeated["tokens"][1:], 0.0)
    np.testing.assert_array_equal(repeated["tokens"][0, -7:-3], 0.0)
    np.testing.assert_array_equal(
        repeated["tokens"][0, -3:],
        np.asarray(info["target_gripper_position"], dtype=np.float32),
    )
    assert base.episode_step_count == 0

    for step in range(1, 4):
        _, _, terminated, truncated, step_info = environment.step(np.zeros(4))
        assert base.episode_step_count == step
        assert not terminated
        assert truncated is (step == 3)
        assert step_info["episode_start_type"] == info["episode_start_type"]
        assert step_info["reward_components"]["approach_orange_waypoint"] == 0.0


def test_zero_recovery_probability_preserves_waypoint_start_and_rng():
    normal = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
    )
    disabled = CubeStackGymEnvironment(
        action_config=CartesianActionConfig(require_downward=False),
        start_at_orange_waypoint=True,
        recovery_start_probability=0.0,
        recovery_closed_gripper_probability=1.0,
    )

    normal_observation, normal_info = normal.reset(seed=3)
    disabled_observation, info = disabled.reset(seed=3)

    assert_prepared(disabled)
    np.testing.assert_array_equal(disabled_observation, normal_observation)
    np.testing.assert_array_equal(
        info["target_gripper_position"], normal_info["target_gripper_position"]
    )
    assert info["episode_start_type"] == "waypoint"
    assert disabled.np_random.bit_generator.state == np.random.default_rng(3).bit_generator.state


def test_waypoint_master_flag_disables_recovery_preparation_and_rng():
    environment = CubeStackGymEnvironment(
        start_at_orange_waypoint=False,
        recovery_start_probability=1.0,
        recovery_closed_gripper_probability=1.0,
    )

    observation, info = environment.reset(seed=3)

    state = environment.previous_state
    assert np.linalg.norm(state["gripper_position"] - DEFAULT_START_POSITION) <= DEFAULT_POSITION_TOLERANCE
    np.testing.assert_array_equal(info["target_gripper_position"], state["gripper_position"])
    assert state["joint_positions"][-1] == DEFAULT_OPEN_GRIPPER_POSITION
    assert state["gripper_target"] == DEFAULT_OPEN_GRIPPER_POSITION
    assert environment.previous_state["time"] == 0.0
    assert observation.shape == (49,)
    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached
    assert info["episode_start_type"] == "home"
    assert environment.np_random.bit_generator.state == np.random.default_rng(3).bit_generator.state


def test_recovery_vector_autoresets_preserve_labels_and_restart_history_targets():
    def make_environment(closed):
        base = CubeStackGymEnvironment(
            action_config=CartesianActionConfig(require_downward=False),
            start_at_orange_waypoint=True,
            recovery_start_probability=1.0,
            recovery_closed_gripper_probability=float(closed),
            maximum_episode_steps=1,
        )
        return Monitor(
            ActionObservationHistoryWrapper(base, history_length=4),
            info_keywords=("episode_start_type",),
        )

    vector = DummyVecEnv([
        lambda: make_environment(False),
        lambda: make_environment(True),
    ])
    try:
        vector.seed(3)
        vector.reset()

        observation, _, done, infos = vector.step(np.zeros((2, 4)))

        np.testing.assert_array_equal(done, [True, True])
        for index, start_type in enumerate(("recovery_open", "recovery_closed")):
            base = vector.envs[index].unwrapped
            info = infos[index]
            assert info["episode_start_type"] == start_type
            assert info["episode"]["episode_start_type"] == start_type
            assert info["episode"]["l"] == 1
            assert base.episode_start_type == start_type
            assert base.episode_step_count == 0
            assert vector.reset_infos[index]["episode_start_type"] == start_type
            np.testing.assert_array_equal(observation["valid"][index], [1, 0, 0, 0])
            np.testing.assert_array_equal(
                observation["episode_start"][index], [1, 0, 0, 0]
            )
            np.testing.assert_array_equal(observation["tokens"][index, 1:], 0.0)
            np.testing.assert_array_equal(observation["tokens"][index, 0, -7:-3], 0.0)
            np.testing.assert_array_equal(
                observation["tokens"][index, 0, -3:],
                base.action_adapter.current_target_gripper_position.astype(np.float32),
            )
            np.testing.assert_array_equal(
                vector.reset_infos[index]["target_gripper_position"],
                base.action_adapter.current_target_gripper_position,
            )
            terminal = info["terminal_observation"]
            np.testing.assert_array_equal(terminal["valid"], [1, 1, 0, 0])
            np.testing.assert_array_equal(
                terminal["tokens"][1, -3:],
                np.asarray(info["target_gripper_position"], dtype=np.float32),
            )
    finally:
        vector.close()


def test_constructor_seed_reproduces_recovery_starts_without_explicit_reset_seeds():
    environments = [
        CubeStackGymEnvironment(
            action_config=CartesianActionConfig(require_downward=False),
            seed=17,
            start_at_orange_waypoint=True,
            recovery_start_probability=1.0,
        )
        for _ in range(2)
    ]
    observations = []
    try:
        for _ in range(2):
            first_observation, first_info = environments[0].reset()
            second_observation, second_info = environments[1].reset()
            np.testing.assert_array_equal(first_observation, second_observation)
            assert first_info["episode_start_type"] == second_info["episode_start_type"]
            np.testing.assert_array_equal(
                first_info["target_gripper_position"],
                second_info["target_gripper_position"],
            )
            observations.append(first_observation)
        assert not np.array_equal(observations[0], observations[1])
    finally:
        for environment in environments:
            environment.close()
