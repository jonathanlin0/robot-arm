import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import cartesian_actions
from cartesian_actions import (
    CARTESIAN_ACTION_SIZE,
    CartesianActionConfig,
    CartesianActionResult,
)
from environment import DEFAULT_START_POSITION, PHYSICS_STEPS_PER_ACTION, StateSnapshot
from gym_environment import CubeStackGymEnvironment
from kinematics import ToolAxisIKResult
# from rewards import RewardResult
from rewards import RewardResult, StackRewardConfig
from success import StackSuccessConfig


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")


@pytest.fixture
def environment() -> CubeStackGymEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    return CubeStackGymEnvironment(scene_path=SCENE_PATH)


def copy_state(state: StateSnapshot) -> StateSnapshot:
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in state.items()
    }


def make_action_result(
    state: StateSnapshot,
    *,
    position_converged: bool = True,
    tool_axis_converged: bool = False,
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


def install_static_step(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    state: StateSnapshot,
    *,
    physical_success: bool = False,
    failed: bool = False,
) -> None:
    action_result = make_action_result(state)
    reward_result = RewardResult(total=0.0, components={})

    monkeypatch.setattr(
        environment.action_adapter,
        "step",
        lambda action: action_result,
    )
    monkeypatch.setattr(
        environment.reward_calculator,
        "calculate",
        lambda **kwargs: reward_result,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_success",
        lambda: physical_success,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_failure",
        lambda: failed,
    )


def test_reset_returns_initial_observation_and_info(
    environment: CubeStackGymEnvironment,
) -> None:
    observation, info = environment.reset(seed=12)

    assert environment.observation_space.contains(observation)
    assert set(info) == {"target_gripper_position", "episode_start_type"}
    assert info["episode_start_type"] == "home"
    assert environment.previous_state is not None
    assert np.linalg.norm(
        environment.previous_state["gripper_position"] - DEFAULT_START_POSITION
    ) <= 1e-6
    assert environment.previous_state["time"] == 0.0
    assert environment.episode_step_count == 0
    np.testing.assert_array_equal(environment.simulation.data.qvel, 0.0)
    np.testing.assert_array_equal(
        info["target_gripper_position"],
        environment.previous_state["gripper_position"],
    )
    np.testing.assert_array_equal(
        environment.action_adapter.current_target_gripper_position,
        info["target_gripper_position"],
    )
    np.testing.assert_array_equal(
        environment.action_adapter.previous_target_gripper_position,
        info["target_gripper_position"],
    )
    np.testing.assert_array_equal(observation[:6], observation[12:18])
    np.testing.assert_array_equal(
        observation[18:21],
        np.asarray(info["target_gripper_position"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observation,
        environment.observation_builder.build(environment.previous_state),
    )
    assert observation.shape == (49,)


def test_reset_target_info_cannot_mutate_controller_or_initial_state(
    environment: CubeStackGymEnvironment,
) -> None:
    _, info = environment.reset(seed=13)
    expected_target = environment.action_adapter.current_target_gripper_position
    info["target_gripper_position"][:] = 0.0

    np.testing.assert_array_equal(
        environment.action_adapter.current_target_gripper_position,
        expected_target,
    )
    assert environment.previous_state is not None
    np.testing.assert_array_equal(
        environment.previous_state["gripper_position"],
        expected_target,
    )


def test_reset_clears_episode_step_count(
    environment: CubeStackGymEnvironment,
) -> None:
    environment.episode_step_count = 17

    environment.reset(seed=23)

    assert environment.episode_step_count == 0


def test_reset_with_same_seed_repeats_initial_observation(
    environment: CubeStackGymEnvironment,
) -> None:
    first_observation, _ = environment.reset(seed=34)
    environment.reset()
    repeated_observation, _ = environment.reset(seed=34)

    np.testing.assert_array_equal(first_observation, repeated_observation)


def test_reset_reinitializes_reward_tracking(
    environment: CubeStackGymEnvironment,
) -> None:
    environment.reset(seed=45)
    environment.reward_calculator._orange_pregrasp_waypoint_reached = True
    environment.reward_calculator._safe_lift_completed = True
    environment.reward_calculator._hover_alignment_completed = True

    environment.reset(seed=46)

    assert not environment.reward_calculator.orange_pregrasp_waypoint_reached
    assert not environment.reward_calculator.safe_lift_completed
    assert not environment.reward_calculator.hover_alignment_completed


def test_reset_reinitializes_cartesian_action_targets(
    environment: CubeStackGymEnvironment,
) -> None:
    # This reset contract is independent of strict orientation feasibility at home.
    environment.action_adapter.config = replace(
        environment.action_adapter.config, require_downward=False
    )
    environment.reset(seed=47)
    initial_state = environment.previous_state
    assert initial_state is not None
    initial_target = initial_state["gripper_position"].copy()
    np.testing.assert_array_equal(
        environment.action_adapter.current_target_gripper_position,
        initial_target,
    )
    np.testing.assert_array_equal(
        environment.action_adapter.previous_target_gripper_position,
        initial_target,
    )

    environment.step(np.array([-0.5, -0.4, 0.6, 0.0]))
    assert not np.allclose(
        environment.action_adapter.current_target_gripper_position,
        initial_target,
    )

    _, reset_info = environment.reset(seed=48)
    reset_state = environment.previous_state
    assert reset_state is not None
    reset_target = reset_state["gripper_position"]
    np.testing.assert_array_equal(
        environment.action_adapter.current_target_gripper_position,
        reset_target,
    )
    np.testing.assert_array_equal(
        environment.action_adapter.previous_target_gripper_position,
        reset_target,
    )
    np.testing.assert_array_equal(
        reset_info["target_gripper_position"],
        reset_target,
    )


def test_best_effort_environment_passes_stable_baselines_checker(
    environment: CubeStackGymEnvironment,
) -> None:
    environment.action_adapter.config = replace(
        environment.action_adapter.config, require_downward=False
    )
    check_env(environment, warn=True)


def test_best_effort_headless_random_action_rollouts_remain_valid(
    environment: CubeStackGymEnvironment,
) -> None:
    environment.action_adapter.config = replace(
        environment.action_adapter.config, require_downward=False
    )
    episode_count = 5
    environment.maximum_episode_steps = 50
    environment.action_space.seed(91)

    for episode_index in range(episode_count):
        observation, info = environment.reset(seed=100 + episode_index)

        assert environment.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))
        assert set(info) == {"target_gripper_position", "episode_start_type"}
        assert info["episode_start_type"] == "home"
        np.testing.assert_array_equal(
            info["target_gripper_position"],
            environment.action_adapter.current_target_gripper_position,
        )

        for _ in range(environment.maximum_episode_steps):
            action = environment.action_space.sample()
            assert environment.action_space.contains(action)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            assert environment.observation_space.contains(observation)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert not (terminated and truncated)
            for component in info["reward_components"].values():
                assert np.isfinite(component)

            if terminated or truncated:
                break
        else:
            pytest.fail("Random-action episode did not end at its limit.")


def test_step_requires_reset(
    environment: CubeStackGymEnvironment,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"reset\(\).*step\(\)",
    ):
        environment.step(np.zeros(CARTESIAN_ACTION_SIZE))


def test_best_effort_step_returns_real_environment_transition_from_home(
    environment: CubeStackGymEnvironment,
) -> None:
    environment.action_adapter.config = replace(
        environment.action_adapter.config, require_downward=False
    )
    environment.reset(seed=57)
    initial_time = environment.simulation.data.time

    observation, reward, terminated, truncated, info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE, dtype=np.float32)
    )

    assert environment.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert environment.simulation.data.time > initial_time
    assert environment.episode_step_count == 1
    assert environment.previous_state is not None
    assert reward == pytest.approx(sum(info["reward_components"].values()))


def test_step_calculates_reward_from_previous_and_current_states(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=68)
    previous_state = environment.previous_state
    assert previous_state is not None

    current_state = copy_state(previous_state)
    current_state["orange_position"][0] += 0.001
    action_result = make_action_result(current_state)
    reward_result = RewardResult(
        total=1.25,
        components={"test_component": 1.25},
    )
    action = np.array([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    adapter_actions: list[np.ndarray] = []
    reward_arguments: dict[str, object] = {}

    def fake_action_step(requested_action: np.ndarray) -> CartesianActionResult:
        adapter_actions.append(requested_action)
        return action_result

    def fake_calculate(**kwargs) -> RewardResult:
        reward_arguments.update(kwargs)
        return reward_result

    monkeypatch.setattr(
        environment.action_adapter,
        "step",
        fake_action_step,
    )
    monkeypatch.setattr(
        environment.reward_calculator,
        "calculate",
        fake_calculate,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_success",
        lambda: False,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_failure",
        lambda: False,
    )

    observation, reward, terminated, truncated, info = environment.step(
        action
    )

    assert len(adapter_actions) == 1
    assert adapter_actions[0] is action
    assert reward_arguments["previous_state"] is previous_state
    assert reward_arguments["action"] is action
    assert reward_arguments["action_result"] is action_result
    assert reward_arguments["succeeded"] is False
    np.testing.assert_array_equal(
        observation,
        environment.observation_builder.build(current_state),
    )
    assert reward == 1.25
    assert terminated is False
    assert truncated is False
    assert environment.previous_state is current_state
    assert info["reward_components"] == {"test_component": 1.25}
    assert info["ik_position_converged"] is True
    assert info["ik_tool_axis_converged"] is False
    assert info["orange_currently_held"] is False
    assert info["orange_grasp_hold_time"] == 0.0
    assert info["stack_stable_time"] == 0.0
    assert info["orange_pregrasp_waypoint_reached"] is False
    np.testing.assert_array_equal(
        info["target_gripper_position"],
        action_result.target_gripper_position,
    )
    assert not np.shares_memory(
        info["target_gripper_position"],
        action_result.target_gripper_position,
    )


@pytest.mark.parametrize(
    ("successful_attempt", "accepted_scale"),
    [(1, 1.0), (3, 0.25), (None, 0.0)],
    ids=["workspace-clipped", "ik-backtracked", "ik-failed"],
)
def test_best_effort_step_target_info_reports_processed_command(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    successful_attempt: int | None,
    accepted_scale: float,
) -> None:
    environment.reset(seed=70)
    assert environment.previous_state is not None
    initial_state = copy_state(environment.previous_state)
    initial_position = initial_state["gripper_position"].copy()
    lower_bounds = initial_position - np.array([0.003, 0.005, 0.007])
    upper_bounds = initial_position + np.array([0.003, 0.005, 0.007])
    environment.action_adapter.config = CartesianActionConfig(
        require_downward=False,
        maximum_position_delta=0.01,
        workspace_lower_bounds=tuple(lower_bounds),
        workspace_upper_bounds=tuple(upper_bounds),
    )
    attempted_targets: list[np.ndarray] = []

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        # First accept an action without moving the simulated arm, making
        # the stored command distinguishable from the measured position.
        position_converged = len(attempted_targets) == 1 or (
            successful_attempt is not None
            and len(attempted_targets) - 1 == successful_attempt
        )
        return ToolAxisIKResult(
            joint_positions=initial_state["joint_positions"][:5].copy(),
            position_converged=position_converged,
            tool_axis_converged=False,
            position_error=0.0 if position_converged else 1.0,
            tool_axis_error=1.0,
            iterations=1,
        )

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment.simulation,
        "step_joint_targets",
        lambda targets: copy_state(initial_state),
    )

    _, _, _, _, earlier_info = environment.step(
        np.array([0.1, 0.0, 0.0, 0.0])
    )
    earlier_target = initial_position + np.array([0.001, 0.0, 0.0])
    np.testing.assert_allclose(
        earlier_info["target_gripper_position"],
        earlier_target,
    )

    _, _, _, _, info = environment.step(np.array([1.0, 1.0, -1.0, 0.0]))

    bounded_requested_target = np.array(
        [upper_bounds[0], upper_bounds[1], lower_bounds[2]]
    )
    expected_target = earlier_target + accepted_scale * (
        bounded_requested_target - earlier_target
    )
    np.testing.assert_allclose(attempted_targets[1], bounded_requested_target)
    assert len(attempted_targets) == 1 + (successful_attempt or 4)
    assert info["ik_position_converged"] is (successful_attempt is not None)
    np.testing.assert_allclose(info["target_gripper_position"], expected_target)
    np.testing.assert_allclose(
        environment.action_adapter.current_target_gripper_position,
        expected_target,
    )
    np.testing.assert_allclose(
        earlier_info["target_gripper_position"],
        earlier_target,
    )
    assert not np.allclose(info["target_gripper_position"], initial_position)
    if successful_attempt is None:
        assert not np.allclose(
            info["target_gripper_position"],
            attempted_targets[-1],
        )

    saved_target = info["target_gripper_position"].copy()
    environment.reset(seed=71)
    np.testing.assert_array_equal(info["target_gripper_position"], saved_target)


def test_step_info_reports_reaching_orange_pregrasp_waypoint(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=69)
    assert environment.previous_state is not None
    current_state = copy_state(environment.previous_state)
    current_state["gripper_position"] = (
        np.asarray(current_state["orange_position"], dtype=float)
        + np.array(
            [
                0.0,
                0.0,
                environment.reward_config.approach_orange_height_offset,
            ]
        )
    )
    action_result = make_action_result(current_state)
    monkeypatch.setattr(
        environment.action_adapter,
        "step",
        lambda action: action_result,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_success",
        lambda: False,
    )
    monkeypatch.setattr(
        environment.simulation,
        "is_failure",
        lambda: False,
    )

    observation, _, terminated, truncated, info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )

    assert info["orange_pregrasp_waypoint_reached"] is True
    np.testing.assert_array_equal(
        observation, environment.observation_builder.build(current_state)
    )
    assert info["reward_components"]["approach_orange_waypoint"] == (
        environment.reward_config.approach_orange_waypoint_reward
    )
    assert info["is_success"] is False
    assert terminated is False
    assert truncated is False


def test_waypoint_latch_changes_info_without_changing_policy_observation(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, _ = environment.reset(seed=69)
    install_static_step(environment, monkeypatch, environment.previous_state)

    unreached, _, _, _, unreached_info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )
    environment.reward_calculator._orange_pregrasp_waypoint_reached = True
    reached, _, _, _, reached_info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )

    assert unreached_info["orange_pregrasp_waypoint_reached"] is False
    assert reached_info["orange_pregrasp_waypoint_reached"] is True
    np.testing.assert_array_equal(unreached, initial)
    np.testing.assert_array_equal(reached, initial)
    assert reached.shape == (49,)


@pytest.mark.parametrize(
    ("height_offset", "tolerance", "position_error", "expected_waypoint_reached"),
    [
        (0.08, 0.01, 0.008, True),
        (0.08, 0.01, 0.012, False),
        (0.12, 0.02, 0.015, True),
        (0.12, 0.02, 0.025, False),
    ],
)
def test_waypoint_reward_uses_config_without_triggering_success(
    monkeypatch: pytest.MonkeyPatch,
    height_offset: float,
    tolerance: float,
    position_error: float,
    expected_waypoint_reached: bool,
) -> None:
    environment = CubeStackGymEnvironment(
        scene_path=SCENE_PATH,
        maximum_episode_steps=1,
        reward_config=StackRewardConfig(
            approach_orange_height_offset=height_offset,
            approach_orange_waypoint_tolerance=tolerance,
        ),
    )
    environment.reset(seed=69)
    simulation = environment.simulation
    simulation.data.joint("orange_cube_joint").qpos[:3] = (
        simulation.data.site("gripperframe").xpos
        + np.array([position_error, 0.0, -height_offset])
    )
    mujoco.mj_forward(simulation.model, simulation.data)
    action_result = make_action_result(simulation.get_state())
    monkeypatch.setattr(
        environment.action_adapter,
        "step",
        lambda action: action_result,
    )

    _, _, terminated, truncated, info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )

    assert info["is_success"] is False
    assert info["orange_pregrasp_waypoint_reached"] is expected_waypoint_reached
    assert info["reward_components"]["approach_orange_waypoint"] == (
        environment.reward_config.approach_orange_waypoint_reward
        if expected_waypoint_reached else 0.0
    )
    assert terminated is False
    assert truncated is True


@pytest.mark.parametrize(
    ("physical_success", "failed", "expected_success"),
    [
        (True, False, True),
        (False, True, False),
        (True, True, False),
    ],
)
def test_step_terminates_for_success_or_failure(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    physical_success: bool,
    failed: bool,
    expected_success: bool,
) -> None:
    environment.maximum_episode_steps = 1
    environment.reset(seed=79)
    assert environment.previous_state is not None
    current_state = copy_state(environment.previous_state)
    current_state["orange_fell_off_table"] = failed
    install_static_step(
        environment,
        monkeypatch,
        current_state,
        physical_success=physical_success,
        failed=failed,
    )

    _, _, terminated, truncated, info = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )

    assert terminated is True
    assert truncated is False
    assert info["is_success"] is expected_success
    assert info["is_failure"] is failed
    assert info["orange_fell_off_table"] is failed


@pytest.mark.parametrize("required_stable_time", [0.075, 0.5])
def test_step_reports_stability_and_terminates_after_released_stack(
    required_stable_time: float,
) -> None:
    environment = CubeStackGymEnvironment(
        scene_path=SCENE_PATH,
        action_config=CartesianActionConfig(require_downward=False),
        success_config=StackSuccessConfig(
            required_stable_time=required_stable_time,
        ),
    )
    environment.reset(seed=81)
    simulation = environment.simulation
    for cube_name, height in (("blue", 0.02), ("orange", 0.06)):
        cube_joint = simulation.data.joint(f"{cube_name}_cube_joint")
        cube_joint.qpos[:] = [0.30, 0.0, height, 1.0, 0.0, 0.0, 0.0]
        cube_joint.qvel.fill(0.0)
    mujoco.mj_forward(simulation.model, simulation.data)
    # Exercise the real stack detector while isolating the prior pickup phase.
    simulation._confirmed_grasp_seen = True
    action_duration = PHYSICS_STEPS_PER_ACTION * simulation.model.opt.timestep
    required_action_count = math.ceil(required_stable_time / action_duration)
    environment.maximum_episode_steps = required_action_count

    for action_index in range(1, required_action_count + 1):
        _, _, terminated, truncated, info = environment.step(
            np.zeros(CARTESIAN_ACTION_SIZE)
        )
        expected_success = action_index == required_action_count
        assert info["stack_stable_time"] == pytest.approx(
            min(action_index * action_duration, required_stable_time)
        )
        assert info["is_success"] is expected_success
        assert info["is_failure"] is False
        assert info["orange_currently_held"] is False
        assert info["orange_grasp_hold_time"] == 0.0
        assert terminated is expected_success
        assert truncated is False


def test_step_truncates_at_episode_step_limit(
    environment: CubeStackGymEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.maximum_episode_steps = 1
    environment.reset(seed=80)
    assert environment.previous_state is not None
    current_state = copy_state(environment.previous_state)
    install_static_step(environment, monkeypatch, current_state)

    _, _, terminated, truncated, _ = environment.step(
        np.zeros(CARTESIAN_ACTION_SIZE)
    )

    assert environment.episode_step_count == 1
    assert terminated is False
    assert truncated is True
