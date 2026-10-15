from pathlib import Path

import numpy as np
import pytest

import cartesian_actions
from cartesian_actions import (
    CARTESIAN_ACTION_SIZE,
    CartesianActionAdapter,
    CartesianActionConfig,
)
from environment import (
    ARM_JOINT_NAMES,
    PHYSICS_STEPS_PER_ACTION,
    CubeStackEnvironment,
    StateSnapshot,
)
from kinematics import ToolAxisIKResult, WORLD_DOWN


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


def tool_axis_ik_result(
    joint_positions: np.ndarray,
    *,
    position_converged: bool = True,
    tool_axis_converged: bool = True,
) -> ToolAxisIKResult:
    return ToolAxisIKResult(
        joint_positions=np.asarray(joint_positions, dtype=float),
        position_converged=position_converged,
        tool_axis_converged=tool_axis_converged,
        position_error=0.0 if position_converged else 1.0,
        tool_axis_error=0.0 if tool_axis_converged else 1.0,
        iterations=1,
    )


def test_reset_initializes_persistent_targets_and_properties_return_copies(
    environment: CubeStackEnvironment,
) -> None:
    initial_state = environment.reset(seed=15)
    adapter = CartesianActionAdapter(environment)

    adapter.reset(initial_state)

    expected_target = initial_state["gripper_position"]
    current_target = adapter.current_target_gripper_position
    previous_target = adapter.previous_target_gripper_position
    np.testing.assert_array_equal(current_target, expected_target)
    np.testing.assert_array_equal(previous_target, expected_target)

    current_target[:] = np.nan
    previous_target[:] = np.nan
    np.testing.assert_array_equal(
        adapter.current_target_gripper_position,
        expected_target,
    )
    np.testing.assert_array_equal(
        adapter.previous_target_gripper_position,
        expected_target,
    )


def test_step_accumulates_deltas_from_target_despite_measured_lag(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=16)
    adapter = CartesianActionAdapter(environment)
    adapter.reset(initial_state)
    action = np.array([0.4, -0.2, 0.1, 0.0])
    position_delta = (
        action[:3] * adapter.config.maximum_position_delta
    )
    attempted_targets: list[np.ndarray] = []

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        return tool_axis_ik_result(
            initial_state["joint_positions"][: len(ARM_JOINT_NAMES)]
        )

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    # Deliberately leave the measured gripper pose frozen. The commanded
    # Cartesian target must still accumulate independently of actuator lag.
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        lambda targets: environment.get_state(),
    )

    first_result = adapter.step(action)
    second_result = adapter.step(action)
    zero_delta_result = adapter.step(np.zeros(CARTESIAN_ACTION_SIZE))

    initial_target = initial_state["gripper_position"]
    expected_targets = [
        initial_target + position_delta,
        initial_target + 2.0 * position_delta,
        initial_target + 2.0 * position_delta,
    ]
    assert len(attempted_targets) == len(expected_targets)
    for attempted_target, expected_target in zip(
        attempted_targets,
        expected_targets,
        strict=True,
    ):
        np.testing.assert_allclose(attempted_target, expected_target)

    np.testing.assert_allclose(
        first_result.target_gripper_position,
        expected_targets[0],
    )
    np.testing.assert_allclose(
        second_result.target_gripper_position,
        expected_targets[1],
    )
    np.testing.assert_allclose(
        zero_delta_result.target_gripper_position,
        expected_targets[2],
    )
    np.testing.assert_allclose(
        adapter.previous_target_gripper_position,
        expected_targets[1],
    )
    np.testing.assert_allclose(
        adapter.current_target_gripper_position,
        expected_targets[2],
    )


def test_step_clips_and_scales_each_position_axis_independently(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=1)
    adapter = CartesianActionAdapter(environment)
    requested_action = np.array([2.0, -2.0, 0.5, 0.0])
    requested_action_before = requested_action.copy()
    planned_arm_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    captured: dict[str, object] = {}

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        captured.update(kwargs)
        return tool_axis_ik_result(planned_arm_positions)

    def fake_step_joint_targets(
        joint_targets: np.ndarray,
    ) -> StateSnapshot:
        captured["joint_targets"] = np.asarray(joint_targets).copy()
        return environment.get_state()

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        fake_step_joint_targets,
    )

    result = adapter.step(requested_action)

    expected_target = initial_state["gripper_position"] + np.array(
        [0.0025, -0.0025, 0.00125]
    )
    np.testing.assert_allclose(captured["target_position"], expected_target)
    np.testing.assert_allclose(
        result.target_gripper_position,
        expected_target,
    )
    np.testing.assert_array_equal(
        captured["initial_joint_positions"],
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)],
    )
    np.testing.assert_array_equal(
        captured["target_tool_axis"],
        WORLD_DOWN,
    )
    assert captured["stop_when_position_converged"] is False
    assert captured["require_downward"] is True
    assert captured["minimum_iterations"] == 1
    np.testing.assert_allclose(
        captured["joint_targets"],
        np.concatenate((planned_arm_positions, [1.0])),
    )
    np.testing.assert_array_equal(requested_action, requested_action_before)


def test_step_clips_target_to_each_workspace_bound(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_state = environment.reset(seed=2)
    current_position = current_state["gripper_position"]
    lower_bounds = current_position + np.array([-0.004, -0.006, -0.008])
    upper_bounds = current_position + np.array([0.003, 0.005, 0.007])
    adapter = CartesianActionAdapter(
        environment,
        CartesianActionConfig(
            maximum_position_delta=0.01,
            workspace_lower_bounds=tuple(lower_bounds),
            workspace_upper_bounds=tuple(upper_bounds),
        ),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        captured["target_position"] = kwargs["target_position"].copy()
        return tool_axis_ik_result(
            current_state["joint_positions"][: len(ARM_JOINT_NAMES)]
        )

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        lambda targets: environment.get_state(),
    )

    result = adapter.step(np.array([1.0, 1.0, -1.0, 0.0]))

    expected_target = np.array(
        [upper_bounds[0], upper_bounds[1], lower_bounds[2]]
    )
    np.testing.assert_allclose(captured["target_position"], expected_target)
    np.testing.assert_allclose(
        result.target_gripper_position,
        expected_target,
    )


@pytest.mark.parametrize(
    ("gripper_command", "expected_target"),
    [
        (-2.0, -0.1),
        (-1.0, -0.1),
        (-0.5, -0.1),
        (0.5, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_step_switches_gripper_target_at_inclusive_thresholds(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    gripper_command: float,
    expected_target: float,
) -> None:
    state = environment.reset(seed=3)
    adapter = CartesianActionAdapter(environment)
    planned_arm_positions = state["joint_positions"][
        : len(ARM_JOINT_NAMES)
    ]
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: tool_axis_ik_result(planned_arm_positions),
    )

    def fake_step_joint_targets(
        joint_targets: np.ndarray,
    ) -> StateSnapshot:
        captured["joint_targets"] = np.asarray(joint_targets).copy()
        return environment.get_state()

    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        fake_step_joint_targets,
    )

    adapter.step(np.array([0.0, 0.0, 0.0, gripper_command]))

    np.testing.assert_allclose(
        captured["joint_targets"][:-1],
        planned_arm_positions,
    )
    assert captured["joint_targets"][-1] == pytest.approx(expected_target)


@pytest.mark.parametrize("gripper_command", [-0.499, 0.0, 0.499])
def test_step_holds_previous_gripper_target_inside_deadband(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    gripper_command: float,
) -> None:
    initial_state = environment.reset(seed=11)
    previous_gripper_target = 0.17
    previous_targets = initial_state["controls"].copy()
    previous_targets[-1] = previous_gripper_target
    environment.step_joint_targets(previous_targets)
    state = environment.get_state()
    planned_arm_positions = state["joint_positions"][
        : len(ARM_JOINT_NAMES)
    ]
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: tool_axis_ik_result(planned_arm_positions),
    )

    def fake_step_joint_targets(
        joint_targets: np.ndarray,
    ) -> StateSnapshot:
        captured["joint_targets"] = np.asarray(joint_targets).copy()
        return environment.get_state()

    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        fake_step_joint_targets,
    )

    adapter = CartesianActionAdapter(environment)
    adapter.step(np.array([0.0, 0.0, 0.0, gripper_command]))

    assert captured["joint_targets"][-1] == pytest.approx(
        previous_gripper_target
    )


def test_gripper_switches_and_holds_target_across_steps(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=12)
    adapter = CartesianActionAdapter(environment)

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: tool_axis_ik_result(
            kwargs["initial_joint_positions"]
        ),
    )

    close_result = adapter.step(np.array([0.0, 0.0, 0.0, -0.5]))
    assert close_result.state["controls"][-1] == pytest.approx(-0.1)

    hold_closed_result = adapter.step(np.zeros(CARTESIAN_ACTION_SIZE))
    assert hold_closed_result.state["controls"][-1] == pytest.approx(-0.1)

    open_result = adapter.step(np.array([0.0, 0.0, 0.0, 0.5]))
    assert open_result.state["controls"][-1] == pytest.approx(1.0)

    hold_open_result = adapter.step(np.zeros(CARTESIAN_ACTION_SIZE))
    assert hold_open_result.state["controls"][-1] == pytest.approx(1.0)


def test_step_seeds_ik_from_measured_joint_positions(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=4)
    environment.step_joint_targets(
        np.array([0.75, -0.5, 0.5, -0.4, 0.6, 0.5])
    )
    state_before = environment.get_state()
    assert not np.allclose(
        state_before["joint_positions"],
        state_before["controls"],
    )
    captured: dict[str, np.ndarray] = {}

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        captured["initial_joint_positions"] = kwargs[
            "initial_joint_positions"
        ].copy()
        return tool_axis_ik_result(
            state_before["joint_positions"][: len(ARM_JOINT_NAMES)]
        )

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        lambda targets: environment.get_state(),
    )

    CartesianActionAdapter(environment).step(np.zeros(CARTESIAN_ACTION_SIZE))

    np.testing.assert_array_equal(
        captured["initial_joint_positions"],
        state_before["joint_positions"][: len(ARM_JOINT_NAMES)],
    )


def test_best_effort_step_accepts_tool_axis_failure_and_advances_one_interval(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=5)
    planned_arm_positions = np.array([0.05, -0.05, 0.05, -0.05, 0.05])
    ik_result = tool_axis_ik_result(
        planned_arm_positions,
        tool_axis_converged=False,
    )
    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: ik_result,
    )

    result = CartesianActionAdapter(
        environment, CartesianActionConfig(require_downward=False)
    ).step(
        np.array([0.0, 0.0, 0.0, -1.0])
    )

    expected_targets = np.concatenate((planned_arm_positions, [-0.1]))
    expected_time = (
        initial_state["time"]
        + PHYSICS_STEPS_PER_ACTION * environment.model.opt.timestep
    )
    np.testing.assert_allclose(result.state["controls"], expected_targets)
    assert result.state["time"] == pytest.approx(expected_time)
    assert environment.data.time == pytest.approx(expected_time)
    assert result.ik_result is ik_result


def test_step_backtracks_to_largest_position_delta_that_converges(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=13)
    adapter = CartesianActionAdapter(environment)
    action = np.array([1.0, -0.5, 0.25, 0.0])
    full_position_delta = np.array([0.0025, -0.00125, 0.000625])
    attempted_targets: list[np.ndarray] = []
    failed_result = tool_axis_ik_result(
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)],
        position_converged=False,
    )
    successful_joint_positions = np.array(
        [0.1, -0.2, 0.3, -0.4, 0.5]
    )
    successful_result = tool_axis_ik_result(
        successful_joint_positions
    )
    captured_joint_targets: list[np.ndarray] = []

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        if len(attempted_targets) < 3:
            return failed_result
        return successful_result

    def fake_step_joint_targets(
        joint_targets: np.ndarray,
    ) -> StateSnapshot:
        captured_joint_targets.append(np.asarray(joint_targets).copy())
        return environment.get_state()

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        fake_step_joint_targets,
    )

    result = adapter.step(action)

    expected_targets = [
        initial_state["gripper_position"] + scale * full_position_delta
        for scale in (1.0, 0.5, 0.25)
    ]
    assert len(attempted_targets) == len(expected_targets)
    for attempted_target, expected_target in zip(
        attempted_targets,
        expected_targets,
        strict=True,
    ):
        np.testing.assert_allclose(attempted_target, expected_target)

    assert result.ik_result is successful_result
    np.testing.assert_allclose(
        result.target_gripper_position,
        expected_targets[-1],
    )
    assert result.attempted_target_gripper_position is not None
    np.testing.assert_allclose(
        result.attempted_target_gripper_position,
        expected_targets[-1],
    )
    assert len(captured_joint_targets) == 1
    np.testing.assert_allclose(
        captured_joint_targets[0],
        np.concatenate((successful_joint_positions, [1.0])),
    )


def test_backtracked_target_is_the_base_for_the_next_delta(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=17)
    adapter = CartesianActionAdapter(environment)
    adapter.reset(initial_state)
    action = np.array([1.0, 0.0, 0.0, 0.0])
    full_position_delta = np.array(
        [adapter.config.maximum_position_delta, 0.0, 0.0]
    )
    attempted_targets: list[np.ndarray] = []
    failed_result = tool_axis_ik_result(
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)],
        position_converged=False,
    )
    successful_result = tool_axis_ik_result(
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)]
    )

    def fail_full_first_step(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        if len(attempted_targets) == 1:
            return failed_result
        return successful_result

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fail_full_first_step,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        lambda targets: environment.get_state(),
    )

    first_result = adapter.step(action)
    second_result = adapter.step(action)

    initial_target = initial_state["gripper_position"]
    accepted_backtracked_target = initial_target + 0.5 * full_position_delta
    expected_targets = [
        initial_target + full_position_delta,
        accepted_backtracked_target,
        accepted_backtracked_target + full_position_delta,
    ]
    assert len(attempted_targets) == len(expected_targets)
    for attempted_target, expected_target in zip(
        attempted_targets,
        expected_targets,
        strict=True,
    ):
        np.testing.assert_allclose(attempted_target, expected_target)
    np.testing.assert_allclose(
        first_result.target_gripper_position,
        accepted_backtracked_target,
    )
    np.testing.assert_allclose(
        adapter.previous_target_gripper_position,
        accepted_backtracked_target,
    )
    np.testing.assert_allclose(
        second_result.target_gripper_position,
        expected_targets[-1],
    )
    np.testing.assert_allclose(
        adapter.current_target_gripper_position,
        expected_targets[-1],
    )


def test_best_effort_step_holds_existing_controls_and_advances_on_position_failure(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=6)
    previous_targets = np.array([0.2, -0.2, 0.2, -0.2, 0.2, 0.5])
    environment.step_joint_targets(previous_targets)
    state_before = environment.get_state()
    failed_result = tool_axis_ik_result(
        np.full(len(ARM_JOINT_NAMES), 1.0),
        position_converged=False,
        tool_axis_converged=False,
    )
    attempted_targets: list[np.ndarray] = []

    def always_fail_ik(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        return failed_result

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        always_fail_ik,
    )

    result = CartesianActionAdapter(
        environment, CartesianActionConfig(require_downward=False)
    ).step(
        np.array([1.0, 0.0, 0.0, -1.0])
    )

    expected_time = (
        state_before["time"]
        + PHYSICS_STEPS_PER_ACTION * environment.model.opt.timestep
    )
    np.testing.assert_array_equal(
        result.state["controls"],
        state_before["controls"],
    )
    assert result.state["time"] == pytest.approx(expected_time)
    assert result.ik_result is failed_result
    full_position_delta = np.array([0.0025, 0.0, 0.0])
    expected_targets = [
        state_before["gripper_position"] + scale * full_position_delta
        for scale in (1.0, 0.5, 0.25, 0.125)
    ]
    assert len(attempted_targets) == len(expected_targets)
    for attempted_target, expected_target in zip(
        attempted_targets,
        expected_targets,
        strict=True,
    ):
        np.testing.assert_allclose(attempted_target, expected_target)
    np.testing.assert_allclose(
        result.target_gripper_position,
        state_before["gripper_position"],
    )
    assert result.attempted_target_gripper_position is not None
    np.testing.assert_allclose(
        result.attempted_target_gripper_position,
        expected_targets[-1],
    )


def test_best_effort_failed_ik_attempts_do_not_advance_persistent_target(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_state = environment.reset(seed=18)
    adapter = CartesianActionAdapter(
        environment, CartesianActionConfig(require_downward=False)
    )
    adapter.reset(initial_state)
    initial_target = initial_state["gripper_position"].copy()
    failed_result = tool_axis_ik_result(
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)],
        position_converged=False,
        tool_axis_converged=False,
    )
    successful_result = tool_axis_ik_result(
        initial_state["joint_positions"][: len(ARM_JOINT_NAMES)]
    )
    attempted_targets: list[np.ndarray] = []

    def fail_one_action_then_succeed(**kwargs) -> ToolAxisIKResult:
        attempted_targets.append(kwargs["target_position"].copy())
        if len(attempted_targets) <= len(
            cartesian_actions.IK_BACKTRACKING_SCALES
        ):
            return failed_result
        return successful_result

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fail_one_action_then_succeed,
    )

    failed_action_result = adapter.step(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(
        adapter.current_target_gripper_position,
        initial_target,
    )
    np.testing.assert_array_equal(
        adapter.previous_target_gripper_position,
        initial_target,
    )
    assert failed_action_result.ik_result is failed_result
    np.testing.assert_allclose(
        failed_action_result.target_gripper_position,
        initial_target,
    )
    assert failed_action_result.attempted_target_gripper_position is not None
    np.testing.assert_allclose(
        failed_action_result.attempted_target_gripper_position,
        attempted_targets[-1],
    )

    retry_result = adapter.step(np.zeros(CARTESIAN_ACTION_SIZE))

    assert len(attempted_targets) == len(
        cartesian_actions.IK_BACKTRACKING_SCALES
    ) + 1
    np.testing.assert_allclose(attempted_targets[-1], initial_target)
    np.testing.assert_allclose(
        retry_result.target_gripper_position,
        initial_target,
    )
    np.testing.assert_allclose(
        adapter.current_target_gripper_position,
        initial_target,
    )


def test_best_effort_step_integrates_real_ik_and_physics_from_home(
    environment: CubeStackEnvironment,
) -> None:
    initial_state = environment.reset(seed=9)
    action = np.array([-0.5, -0.4, 0.6, 1.0])

    result = CartesianActionAdapter(
        environment, CartesianActionConfig(require_downward=False)
    ).step(action)

    expected_target_position = initial_state["gripper_position"] + np.array(
        [-0.00125, -0.001, 0.0015]
    )
    expected_time = (
        PHYSICS_STEPS_PER_ACTION * environment.model.opt.timestep
    )
    np.testing.assert_allclose(
        result.target_gripper_position,
        expected_target_position,
    )
    assert result.ik_result.position_converged is True
    assert result.ik_result.iterations < 100
    np.testing.assert_allclose(
        result.state["controls"][:-1],
        result.ik_result.joint_positions,
    )
    assert result.state["controls"][-1] == pytest.approx(1.0)
    assert result.state["time"] == pytest.approx(expected_time)


def test_step_forwards_custom_target_tool_axis(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.reset(seed=10)
    custom_axis = (0.0, 1.0, 0.0)
    adapter = CartesianActionAdapter(
        environment,
        CartesianActionConfig(target_tool_axis=custom_axis, require_downward=False),
    )
    captured: dict[str, object] = {}

    def fake_solve(**kwargs) -> ToolAxisIKResult:
        captured.update(kwargs)
        return tool_axis_ik_result(
            environment.get_state()["joint_positions"][
                : len(ARM_JOINT_NAMES)
            ]
        )

    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        fake_solve,
    )
    monkeypatch.setattr(
        environment,
        "step_joint_targets",
        lambda targets: environment.get_state(),
    )

    adapter.step(np.zeros(CARTESIAN_ACTION_SIZE))

    np.testing.assert_array_equal(
        captured["target_tool_axis"],
        custom_axis,
    )


@pytest.mark.parametrize(
    "invalid_action",
    [
        np.zeros(CARTESIAN_ACTION_SIZE - 1),
        np.zeros(CARTESIAN_ACTION_SIZE + 1),
        np.zeros((CARTESIAN_ACTION_SIZE, 1)),
    ],
)
def test_step_rejects_wrong_action_shape_without_mutating_simulation(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    invalid_action: np.ndarray,
) -> None:
    environment.reset(seed=7)
    qpos_before = environment.data.qpos.copy()
    qvel_before = environment.data.qvel.copy()
    controls_before = environment.data.ctrl.copy()
    time_before = float(environment.data.time)
    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: pytest.fail("IK should not be called"),
    )

    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        CartesianActionAdapter(environment).step(invalid_action)

    np.testing.assert_array_equal(environment.data.qpos, qpos_before)
    np.testing.assert_array_equal(environment.data.qvel, qvel_before)
    np.testing.assert_array_equal(environment.data.ctrl, controls_before)
    assert environment.data.time == time_before


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_step_rejects_non_finite_action_without_mutating_simulation(
    environment: CubeStackEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: float,
) -> None:
    environment.reset(seed=8)
    action = np.zeros(CARTESIAN_ACTION_SIZE)
    action[1] = invalid_value
    controls_before = environment.data.ctrl.copy()
    time_before = float(environment.data.time)
    monkeypatch.setattr(
        cartesian_actions,
        "solve_position_and_tool_axis_ik",
        lambda **kwargs: pytest.fail("IK should not be called"),
    )

    with pytest.raises(ValueError, match="finite"):
        CartesianActionAdapter(environment).step(action)

    np.testing.assert_array_equal(environment.data.ctrl, controls_before)
    assert environment.data.time == time_before


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"maximum_position_delta": 0.0}, "delta"),
        (
            {
                "closed_gripper_target": 0.5,
                "open_gripper_target": 0.5,
            },
            "gripper",
        ),
        (
            {"close_gripper_command_threshold": -1.01},
            "threshold",
        ),
        (
            {"open_gripper_command_threshold": 1.01},
            "threshold",
        ),
        (
            {
                "close_gripper_command_threshold": 0.5,
                "open_gripper_command_threshold": 0.5,
            },
            "threshold",
        ),
        (
            {
                "workspace_lower_bounds": (0.2, 0.0, 0.0),
                "workspace_upper_bounds": (0.1, 1.0, 1.0),
            },
            "workspace",
        ),
    ],
)
def test_invalid_action_config_raises(
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CartesianActionConfig(**config_kwargs)
