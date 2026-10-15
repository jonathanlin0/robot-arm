from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment import (
    ARM_JOINT_NAMES,
    DEFAULT_JOINT_POSITIONS,
    CubeStackEnvironment,
)
from kinematics import (
    DEFAULT_TOOL_AXIS_TOLERANCE,
    DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES,
    IKResult,
    ToolAxisIKResult,
    WORLD_DOWN,
    solve_position_and_tool_axis_ik,
    solve_position_ik,
)


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")
POSITION_TOLERANCE = 1e-4
TOP_DOWN_ARM_POSITIONS = np.array(
    [0.0, 0.13, -0.145, np.pi / 2.0 + 0.015, 0.0]
)


@pytest.fixture
def environment() -> CubeStackEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    return CubeStackEnvironment(scene_path=SCENE_PATH)


def forward_gripper_position(
    model: mujoco.MjModel,
    arm_joint_positions: np.ndarray,
    gripper_joint_position: float = DEFAULT_JOINT_POSITIONS[-1],
) -> np.ndarray:
    position, _ = forward_gripper_pose(
        model,
        arm_joint_positions,
        gripper_joint_position,
    )
    return position


def forward_gripper_pose(
    model: mujoco.MjModel,
    arm_joint_positions: np.ndarray,
    gripper_joint_position: float = DEFAULT_JOINT_POSITIONS[-1],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)

    for joint_name, joint_position in zip(
        ARM_JOINT_NAMES,
        arm_joint_positions,
        strict=True,
    ):
        data.joint(joint_name).qpos[0] = joint_position

    data.joint("gripper").qpos[0] = gripper_joint_position
    mujoco.mj_forward(model, data)

    gripper_site = data.site("gripperframe")
    position = gripper_site.xpos.copy()
    approach_axis = gripper_site.xmat.reshape(3, 3)[:, 0].copy()
    return position, approach_axis


def angle_between_axes(
    first_axis: np.ndarray,
    second_axis: np.ndarray,
) -> float:
    return float(
        np.arctan2(
            np.linalg.norm(np.cross(first_axis, second_axis)),
            np.clip(np.dot(first_axis, second_axis), -1.0, 1.0),
        )
    )


def safe_arm_joint_bounds(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = []
    upper_bounds = []

    for joint_name in ARM_JOINT_NAMES:
        joint_range = model.joint(joint_name).range
        actuator_range = model.actuator(joint_name).ctrlrange
        lower_bounds.append(max(joint_range[0], actuator_range[0]))
        upper_bounds.append(min(joint_range[1], actuator_range[1]))

    return np.array(lower_bounds), np.array(upper_bounds)


def test_ik_current_position_converges_without_a_correction(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.array([0.10, -0.20, 0.25, -0.10, 0.05])
    target_position = forward_gripper_position(
        environment.model,
        initial_positions,
    )

    result = solve_position_ik(
        environment.model,
        initial_positions,
        target_position,
    )

    assert isinstance(result, IKResult)
    assert result.converged is True
    assert result.iterations == 0
    assert result.position_error == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(result.joint_positions, initial_positions)


def test_ik_position_is_independent_of_gripper_aperture(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.array([0.10, -0.20, 0.25, -0.10, 0.05])
    closed_position = forward_gripper_position(
        environment.model,
        initial_positions,
        -0.1,
    )
    open_position = forward_gripper_position(
        environment.model,
        initial_positions,
        1.0,
    )

    np.testing.assert_allclose(closed_position, open_position)

    result = solve_position_ik(
        environment.model,
        initial_positions,
        open_position,
    )

    assert result.converged is True
    assert result.iterations == 0
    assert result.position_error == pytest.approx(0.0, abs=1e-12)


def test_ik_converges_to_a_reachable_position(
    environment: CubeStackEnvironment,
) -> None:
    target_generating_positions = np.array(
        [0.10, -0.08, 0.10, -0.05, 0.05]
    )
    target_position = forward_gripper_position(
        environment.model,
        target_generating_positions,
    )

    result = solve_position_ik(
        environment.model,
        np.zeros(len(ARM_JOINT_NAMES)),
        target_position,
        tolerance=POSITION_TOLERANCE,
    )
    achieved_position = forward_gripper_position(
        environment.model,
        result.joint_positions,
    )
    measured_error = np.linalg.norm(target_position - achieved_position)

    assert result.converged is True
    assert result.iterations > 0
    assert result.joint_positions.shape == (len(ARM_JOINT_NAMES),)
    assert measured_error <= POSITION_TOLERANCE
    assert result.position_error == pytest.approx(measured_error)


def test_ik_unreachable_position_reports_failure_with_safe_result(
    environment: CubeStackEnvironment,
) -> None:
    target_position = np.array([10.0, 10.0, 10.0])
    maximum_iterations = 20

    result = solve_position_ik(
        environment.model,
        np.zeros(len(ARM_JOINT_NAMES)),
        target_position,
        tolerance=POSITION_TOLERANCE,
        max_iterations=maximum_iterations,
    )
    achieved_position = forward_gripper_position(
        environment.model,
        result.joint_positions,
    )
    measured_error = np.linalg.norm(target_position - achieved_position)
    lower_bounds, upper_bounds = safe_arm_joint_bounds(environment.model)

    assert result.converged is False
    assert result.iterations == maximum_iterations
    assert np.all(np.isfinite(result.joint_positions))
    assert np.all(result.joint_positions >= lower_bounds)
    assert np.all(result.joint_positions <= upper_bounds)
    assert result.position_error == pytest.approx(measured_error)
    assert result.position_error > POSITION_TOLERANCE


def test_ik_does_not_mutate_inputs_or_live_simulation(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=31)
    initial_positions = np.zeros(len(ARM_JOINT_NAMES))
    target_position = (
        environment.data.site("gripperframe").xpos
        + np.array([-0.01, 0.01, 0.01])
    )
    initial_positions_before = initial_positions.copy()
    target_position_before = target_position.copy()
    data_before = {
        "qpos": environment.data.qpos.copy(),
        "qvel": environment.data.qvel.copy(),
        "ctrl": environment.data.ctrl.copy(),
        "time": float(environment.data.time),
        "site_xpos": environment.data.site_xpos.copy(),
    }
    model_before = {
        "qpos0": environment.model.qpos0.copy(),
        "jnt_range": environment.model.jnt_range.copy(),
        "actuator_ctrlrange": environment.model.actuator_ctrlrange.copy(),
    }

    solve_position_ik(
        environment.model,
        initial_positions,
        target_position,
    )

    np.testing.assert_array_equal(initial_positions, initial_positions_before)
    np.testing.assert_array_equal(target_position, target_position_before)
    np.testing.assert_array_equal(environment.data.qpos, data_before["qpos"])
    np.testing.assert_array_equal(environment.data.qvel, data_before["qvel"])
    np.testing.assert_array_equal(environment.data.ctrl, data_before["ctrl"])
    assert environment.data.time == data_before["time"]
    np.testing.assert_array_equal(
        environment.data.site_xpos,
        data_before["site_xpos"],
    )
    np.testing.assert_array_equal(environment.model.qpos0, model_before["qpos0"])
    np.testing.assert_array_equal(
        environment.model.jnt_range,
        model_before["jnt_range"],
    )
    np.testing.assert_array_equal(
        environment.model.actuator_ctrlrange,
        model_before["actuator_ctrlrange"],
    )


@pytest.mark.parametrize(
    ("initial_positions", "target_position"),
    [
        (np.full(len(ARM_JOINT_NAMES), np.nan), np.zeros(3)),
        (np.zeros(len(ARM_JOINT_NAMES)), np.array([0.0, np.inf, 0.0])),
    ],
)
def test_ik_rejects_non_finite_inputs(
    environment: CubeStackEnvironment,
    initial_positions: np.ndarray,
    target_position: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        solve_position_ik(
            environment.model,
            initial_positions,
            target_position,
        )


def test_tool_axis_ik_reaches_position_and_points_gripper_down(
    environment: CubeStackEnvironment,
) -> None:
    target_position, target_axis = forward_gripper_pose(
        environment.model,
        TOP_DOWN_ARM_POSITIONS,
    )
    np.testing.assert_allclose(target_axis, WORLD_DOWN, atol=1e-12)

    result = solve_position_and_tool_axis_ik(
        environment.model,
        np.zeros(len(ARM_JOINT_NAMES)),
        target_position,
        WORLD_DOWN,
        position_tolerance=POSITION_TOLERANCE,
    )
    achieved_position, achieved_axis = forward_gripper_pose(
        environment.model,
        result.joint_positions,
    )
    measured_position_error = np.linalg.norm(
        target_position - achieved_position
    )
    measured_axis_error = angle_between_axes(
        achieved_axis,
        np.array(WORLD_DOWN),
    )

    assert isinstance(result, ToolAxisIKResult)
    assert result.position_converged is True
    assert result.tool_axis_converged is True
    assert measured_position_error <= POSITION_TOLERANCE
    assert measured_axis_error <= DEFAULT_TOOL_AXIS_TOLERANCE
    assert result.position_error == pytest.approx(measured_position_error)
    assert result.tool_axis_error == pytest.approx(measured_axis_error)


def test_tool_axis_ik_accepts_non_unit_axis_and_zero_iteration_solution(
    environment: CubeStackEnvironment,
) -> None:
    target_position, _ = forward_gripper_pose(
        environment.model,
        TOP_DOWN_ARM_POSITIONS,
    )

    result = solve_position_and_tool_axis_ik(
        environment.model,
        TOP_DOWN_ARM_POSITIONS,
        target_position,
        np.array(WORLD_DOWN) * 4.0,
    )

    assert result.position_converged is True
    assert result.tool_axis_converged is True
    assert result.iterations == 0
    np.testing.assert_array_equal(
        result.joint_positions,
        TOP_DOWN_ARM_POSITIONS,
    )


def test_tool_axis_ik_honors_an_arbitrary_target_axis(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.array([0.2, -0.3, 0.25, -0.1, 0.4])
    target_position, target_axis = forward_gripper_pose(
        environment.model,
        initial_positions,
    )

    result = solve_position_and_tool_axis_ik(
        environment.model,
        initial_positions,
        target_position,
        target_axis,
        require_downward=False,
    )

    assert result.position_converged is True
    assert result.tool_axis_converged is True
    assert result.iterations == 0
    np.testing.assert_array_equal(
        result.joint_positions,
        initial_positions,
    )


def test_tool_axis_ik_keeps_position_primary_when_axis_is_not_reached(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.zeros(len(ARM_JOINT_NAMES))
    target_position, _ = forward_gripper_pose(
        environment.model,
        initial_positions,
    )

    result = solve_position_and_tool_axis_ik(
        environment.model,
        initial_positions,
        target_position,
        WORLD_DOWN,
        require_downward=False,
    )
    achieved_position, achieved_axis = forward_gripper_pose(
        environment.model,
        result.joint_positions,
    )
    measured_position_error = np.linalg.norm(
        target_position - achieved_position
    )
    measured_axis_error = angle_between_axes(
        achieved_axis,
        np.array(WORLD_DOWN),
    )

    assert result.position_converged is True
    assert result.tool_axis_converged is False
    assert measured_position_error <= 1e-3
    assert measured_axis_error > DEFAULT_TOOL_AXIS_TOLERANCE
    assert result.position_error == pytest.approx(measured_position_error)
    assert result.tool_axis_error == pytest.approx(measured_axis_error)


def test_tool_axis_ik_can_stop_at_first_position_solution(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.zeros(len(ARM_JOINT_NAMES))
    target_position, _ = forward_gripper_pose(
        environment.model,
        initial_positions,
    )

    result = solve_position_and_tool_axis_ik(
        environment.model,
        initial_positions,
        target_position,
        WORLD_DOWN,
        stop_when_position_converged=True,
        require_downward=False,
    )

    assert result.position_converged is True
    assert result.tool_axis_converged is False
    assert result.iterations == 0
    np.testing.assert_array_equal(
        result.joint_positions,
        initial_positions,
    )


def test_tool_axis_ik_can_require_one_local_orientation_correction(
    environment: CubeStackEnvironment,
) -> None:
    initial_positions = np.zeros(len(ARM_JOINT_NAMES))
    target_position, _ = forward_gripper_pose(
        environment.model,
        initial_positions,
    )

    result = solve_position_and_tool_axis_ik(
        environment.model,
        initial_positions,
        target_position,
        WORLD_DOWN,
        stop_when_position_converged=True,
        minimum_iterations=1,
        require_downward=False,
    )

    assert result.position_converged is True
    assert result.tool_axis_converged is False
    assert result.iterations == 1
    assert np.linalg.norm(result.joint_positions - initial_positions) > 0.0


def test_tool_axis_ik_unreachable_position_returns_safe_result(
    environment: CubeStackEnvironment,
) -> None:
    result = solve_position_and_tool_axis_ik(
        environment.model,
        np.zeros(len(ARM_JOINT_NAMES)),
        np.array([10.0, 10.0, 10.0]),
        WORLD_DOWN,
        max_iterations=20,
    )
    lower_bounds, upper_bounds = safe_arm_joint_bounds(environment.model)

    assert result.position_converged is False
    assert np.all(np.isfinite(result.joint_positions))
    assert np.all(result.joint_positions >= lower_bounds)
    assert np.all(result.joint_positions <= upper_bounds)
    assert np.isfinite(result.position_error)
    assert np.isfinite(result.tool_axis_error)


def test_tool_axis_ik_does_not_mutate_inputs_or_live_simulation(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=31)
    initial_positions = np.zeros(len(ARM_JOINT_NAMES))
    target_position = np.array([0.25, 0.0, 0.075])
    target_axis = np.array(WORLD_DOWN)
    initial_positions_before = initial_positions.copy()
    target_position_before = target_position.copy()
    target_axis_before = target_axis.copy()
    qpos_before = environment.data.qpos.copy()
    qvel_before = environment.data.qvel.copy()
    controls_before = environment.data.ctrl.copy()
    time_before = float(environment.data.time)

    solve_position_and_tool_axis_ik(
        environment.model,
        initial_positions,
        target_position,
        target_axis,
    )

    np.testing.assert_array_equal(initial_positions, initial_positions_before)
    np.testing.assert_array_equal(target_position, target_position_before)
    np.testing.assert_array_equal(target_axis, target_axis_before)
    np.testing.assert_array_equal(environment.data.qpos, qpos_before)
    np.testing.assert_array_equal(environment.data.qvel, qvel_before)
    np.testing.assert_array_equal(environment.data.ctrl, controls_before)
    assert environment.data.time == time_before


@pytest.mark.parametrize(
    "target_axis",
    [
        np.zeros(3),
        np.array([0.0, np.nan, -1.0]),
    ],
)
def test_tool_axis_ik_rejects_invalid_target_axis(
    environment: CubeStackEnvironment,
    target_axis: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        solve_position_and_tool_axis_ik(
            environment.model,
            np.zeros(len(ARM_JOINT_NAMES)),
            np.array([0.25, 0.0, 0.075]),
            target_axis,
        )


def test_strict_downward_overrides_position_early_stopping(
    environment: CubeStackEnvironment,
) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[3] -= np.deg2rad(DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES + 3.6)
    position, initial_axis = forward_gripper_pose(environment.model, initial)
    assert angle_between_axes(initial_axis, np.array(WORLD_DOWN)) > DEFAULT_TOOL_AXIS_TOLERANCE

    result = solve_position_and_tool_axis_ik(
        environment.model, initial, position, stop_when_position_converged=True,
    )
    measured_position, measured_axis = forward_gripper_pose(environment.model, result.joint_positions)

    assert result.position_converged and result.tool_axis_converged
    assert result.iterations > 0
    assert result.total_iterations == result.iterations
    assert np.linalg.norm(position - measured_position) <= 1e-3
    assert angle_between_axes(measured_axis, np.array(WORLD_DOWN)) <= DEFAULT_TOOL_AXIS_TOLERANCE


def test_strict_failure_returns_diagnostics_without_claiming_infeasibility() -> None:
    # Explicitly use the old, high position-only home; the default reset now
    # starts with a valid downward orientation.
    environment = CubeStackEnvironment(
        start_position=(0.40, 0.0, 0.25), require_downward=False,
    )
    state = environment.reset(seed=1)
    result = solve_position_and_tool_axis_ik(
        environment.model, state["joint_positions"][:-1], state["gripper_position"],
        target_tool_yaw=0.0, stop_when_position_converged=True, max_iterations=30,
    )

    assert result.position_converged and result.tool_yaw_converged
    assert not result.tool_axis_converged
    assert result.tool_axis_error > DEFAULT_TOOL_AXIS_TOLERANCE
    assert result.total_iterations == 30
    assert result.iterations <= result.total_iterations
    np.testing.assert_array_equal(environment.get_state()["joint_positions"], state["joint_positions"])


@pytest.mark.parametrize("offset_degrees", [-0.1, 0.1])
def test_strict_downward_uses_configured_angular_tolerance(
    environment: CubeStackEnvironment, offset_degrees: float,
) -> None:
    assert DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES == 10.0
    assert DEFAULT_TOOL_AXIS_TOLERANCE == pytest.approx(
        np.deg2rad(DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES)
    )
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[3] -= np.deg2rad(DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES + offset_degrees)
    position, _ = forward_gripper_pose(environment.model, initial)
    result = solve_position_and_tool_axis_ik(
        environment.model, initial, position, stop_when_position_converged=True,
        max_iterations=1, tool_axis_gain=1e-12,
    )

    assert result.position_converged
    assert result.tool_axis_converged is (offset_degrees < 0)
    assert result.total_iterations == (0 if offset_degrees < 0 else 1)


def test_strict_fallback_prefers_downward_progress_once_yaw_is_within_tolerance(
    environment: CubeStackEnvironment,
) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[3] -= np.deg2rad(8.6)
    # Counter the gripper model's fixed mounting rotation, starting at exactly
    # zero yaw; a tiny yaw drift must not erase useful downward progress.
    initial[4] = 2 * np.arctan2(0.0172091, 0.706897)
    position, initial_axis = forward_gripper_pose(environment.model, initial)
    initial_error = angle_between_axes(initial_axis, np.array(WORLD_DOWN))
    # Keep this numeric regression outside the angular tolerance even if the
    # project's default is relaxed.
    tool_axis_tolerance = np.deg2rad(5.0)

    result = solve_position_and_tool_axis_ik(
        environment.model, initial, position, target_tool_yaw=0.0, max_iterations=2,
        tool_axis_tolerance=tool_axis_tolerance,
    )

    assert result.position_converged and result.tool_yaw_converged
    assert not result.tool_axis_converged
    assert result.tool_axis_error < initial_error - 0.002
    assert result.total_iterations == 2

    legacy_result = solve_position_and_tool_axis_ik(
        environment.model, initial, position, target_tool_yaw=0.0,
        max_iterations=2, require_downward=False,
        tool_axis_tolerance=tool_axis_tolerance,
    )
    assert legacy_result.iterations == 0
    assert legacy_result.total_iterations == 2


@pytest.mark.parametrize("target_axis", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
def test_strict_downward_rejects_conflicting_target_axis(
    environment: CubeStackEnvironment, target_axis: tuple[float, float, float],
) -> None:
    position, _ = forward_gripper_pose(environment.model, TOP_DOWN_ARM_POSITIONS)
    with pytest.raises(ValueError, match="require_downward=True"):
        solve_position_and_tool_axis_ik(
            environment.model, TOP_DOWN_ARM_POSITIONS, position, target_axis,
        )


@pytest.mark.parametrize("invalid_mode", [None, 0, 1, "true"])
def test_downward_requirement_must_be_boolean(
    environment: CubeStackEnvironment, invalid_mode: object,
) -> None:
    position, _ = forward_gripper_pose(environment.model, TOP_DOWN_ARM_POSITIONS)
    with pytest.raises(ValueError, match="require_downward"):
        solve_position_and_tool_axis_ik(
            environment.model, TOP_DOWN_ARM_POSITIONS, position, require_downward=invalid_mode,
        )
