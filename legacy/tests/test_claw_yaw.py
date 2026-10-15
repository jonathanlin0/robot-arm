"""World-frame claw heading and shared Cartesian-controller regressions."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

import cartesian_actions
from cartesian_actions import (
    IK_BACKTRACKING_SCALES,
    CartesianActionAdapter,
    CartesianActionConfig,
)
from environment import PHYSICS_STEPS_PER_ACTION, CubeStackEnvironment
from kinematics import (
    DEFAULT_POSITION_TOLERANCE,
    DEFAULT_TOOL_AXIS_TOLERANCE,
    DEFAULT_TOOL_YAW_TOLERANCE,
    ToolAxisIKResult,
    solve_position_and_tool_axis_ik,
)
from robot_constants import ARM_JOINT_NAMES


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
TOP_DOWN_ARM_POSITIONS = np.array([0.0, 0.13, -0.145, np.pi / 2 + 0.015, 0.0])


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def measured_pose(
    model: mujoco.MjModel, joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINT_NAMES, joint_positions, strict=True):
        data.joint(name).qpos[0] = value
    mujoco.mj_forward(model, data)
    site = data.site("gripperframe")
    rotation = site.xmat.reshape(3, 3)
    closing_axis = rotation[:, 2]
    yaw = float(np.arctan2(closing_axis[1], closing_axis[0]))
    return site.xpos.copy(), rotation[:, 0].copy(), yaw


def angular_error(first: float, second: float) -> float:
    return float(abs(np.arctan2(np.sin(first - second), np.cos(first - second))))


@pytest.mark.parametrize("shoulder_pan", [-0.4, 0.4])
def test_sideways_claw_reaches_world_forward_heading_and_position(
    model: mujoco.MjModel, shoulder_pan: float,
) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[0] = shoulder_pan
    target_position, _, original_yaw = measured_pose(model, initial)
    assert abs(target_position[1]) > 0.07
    assert abs(original_yaw) > np.deg2rad(15)

    result = solve_position_and_tool_axis_ik(
        model, initial, target_position, target_tool_yaw=0.0,
    )
    position, approach_axis, yaw = measured_pose(model, result.joint_positions)

    assert result.position_converged
    assert result.tool_yaw_converged
    assert result.tool_axis_converged
    assert np.linalg.norm(position - target_position) <= DEFAULT_POSITION_TOLERANCE
    assert angular_error(yaw, 0.0) <= DEFAULT_TOOL_YAW_TOLERANCE
    assert np.arccos(np.clip(-approach_axis[2], -1.0, 1.0)) <= DEFAULT_TOOL_AXIS_TOLERANCE
    assert result.tool_yaw_error == pytest.approx(angular_error(yaw, 0.0), abs=1e-8)
    np.testing.assert_array_equal(initial, [shoulder_pan, *TOP_DOWN_ARM_POSITIONS[1:]])


def test_online_position_early_exit_still_corrects_wrong_yaw(
    model: mujoco.MjModel,
) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[0] = 0.4
    target_position, _, _ = measured_pose(model, initial)

    result = solve_position_and_tool_axis_ik(
        model, initial, target_position,
        target_tool_yaw=0.0, stop_when_position_converged=True,
    )

    assert result.iterations > 0
    assert result.position_converged and result.tool_yaw_converged
    _, _, yaw = measured_pose(model, result.joint_positions)
    assert angular_error(yaw, 0.0) <= DEFAULT_TOOL_YAW_TOLERANCE


def test_none_keeps_legacy_unconstrained_wrist_and_early_exit(
    model: mujoco.MjModel,
) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[0] = -0.4
    target_position, _, original_yaw = measured_pose(model, initial)

    result = solve_position_and_tool_axis_ik(
        model, initial, target_position,
        target_tool_yaw=None, stop_when_position_converged=True,
        require_downward=False,
    )

    assert result.iterations == 0
    assert result.position_converged and result.tool_yaw_converged
    assert result.tool_yaw_error == 0.0
    np.testing.assert_array_equal(result.joint_positions, initial)
    assert abs(original_yaw) > DEFAULT_TOOL_YAW_TOLERANCE


def test_wrist_flex_limit_does_not_block_reachable_yaw_correction(
    model: mujoco.MjModel,
) -> None:
    wrist_limit = min(
        model.joint("wrist_flex").range[1],
        model.actuator("wrist_flex").ctrlrange[1],
    )
    initial = np.array([0.4, 0.5, -0.8, wrist_limit, 0.0])
    target_position, _, _ = measured_pose(model, initial)

    # This independent feasible posture establishes that the bound need not
    # prevent holding XYZ while fixing yaw; downward pitch is best effort.
    feasible = np.array([0.41913274, 0.49787007, -0.79589394, 1.65004549, 0.45891657])
    feasible_position, _, feasible_yaw = measured_pose(model, feasible)
    assert np.linalg.norm(feasible_position - target_position) < 1e-6
    assert abs(feasible_yaw) < 1e-6

    result = solve_position_and_tool_axis_ik(
        model, initial, target_position,
        target_tool_yaw=0.0, stop_when_position_converged=True,
        require_downward=False,
    )

    assert result.position_converged and result.tool_yaw_converged
    position, _, yaw = measured_pose(model, result.joint_positions)
    assert np.linalg.norm(position - target_position) <= DEFAULT_POSITION_TOLERANCE
    assert angular_error(yaw, 0.0) <= DEFAULT_TOOL_YAW_TOLERANCE


def test_heading_error_wraps_across_minus_pi_and_pi(model: mujoco.MjModel) -> None:
    initial = TOP_DOWN_ARM_POSITIONS.copy()
    initial[0] = 0.8
    initial[-1] = -2.28
    target_position, _, original_yaw = measured_pose(model, initial)
    assert original_yaw < -3.0
    target_yaw = original_yaw + 2 * np.pi - 0.035

    result = solve_position_and_tool_axis_ik(
        model, initial, target_position, target_tool_yaw=target_yaw,
    )

    assert result.position_converged and result.tool_yaw_converged
    _, _, yaw = measured_pose(model, result.joint_positions)
    assert angular_error(yaw, target_yaw) <= DEFAULT_TOOL_YAW_TOLERANCE
    assert np.linalg.norm(result.joint_positions - initial) < 0.2


@pytest.mark.parametrize("yaw", [np.nan, np.inf, -np.inf, [0.0], [0.0, 1.0]])
def test_yaw_requires_a_finite_scalar(model: mujoco.MjModel, yaw: object) -> None:
    position, _, _ = measured_pose(model, TOP_DOWN_ARM_POSITIONS)
    with pytest.raises(ValueError, match="target_tool_yaw"):
        solve_position_and_tool_axis_ik(
            model, TOP_DOWN_ARM_POSITIONS, position, target_tool_yaw=yaw,
        )
    with pytest.raises(ValueError, match="target_tool_yaw"):
        CartesianActionConfig(target_tool_yaw=yaw)


@pytest.mark.parametrize("tolerance", [0.0, -0.1, np.nan, np.inf])
def test_yaw_tolerance_must_be_positive_and_finite(
    model: mujoco.MjModel, tolerance: float,
) -> None:
    position, _, _ = measured_pose(model, TOP_DOWN_ARM_POSITIONS)
    with pytest.raises(ValueError, match="tool_yaw_tolerance"):
        solve_position_and_tool_axis_ik(
            model, TOP_DOWN_ARM_POSITIONS, position,
            target_tool_yaw=0.0, tool_yaw_tolerance=tolerance,
        )


def test_controller_config_defaults_to_forward_and_accepts_scalar_or_none() -> None:
    assert CartesianActionConfig().target_tool_yaw == 0.0
    assert CartesianActionConfig(target_tool_yaw=np.float64(0.2)).target_tool_yaw == 0.2
    assert CartesianActionConfig(target_tool_yaw=None).target_tool_yaw is None


def test_yaw_failure_preserves_accepted_target_and_gripper_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = CubeStackEnvironment(scene_path=SCENE_PATH)
    initial_state = environment.reset(seed=15)
    adapter = CartesianActionAdapter(environment, CartesianActionConfig(require_downward=False))
    adapter.reset(initial_state)
    accepted_target = adapter.current_target_gripper_position
    initial_controls = environment.data.ctrl.copy()
    attempts = []
    physics_steps = []

    def rejected_yaw(**kwargs) -> ToolAxisIKResult:
        assert kwargs["target_tool_yaw"] == 0.0
        attempts.append(kwargs["target_position"].copy())
        return ToolAxisIKResult(
            joint_positions=initial_state["joint_positions"][:-1].copy(),
            position_converged=True, tool_axis_converged=True,
            position_error=0.0, tool_axis_error=0.0, iterations=1,
            tool_yaw_converged=False, tool_yaw_error=0.2,
        )

    def unexpected_joint_targets(targets) -> None:
        pytest.fail("A yaw-rejected action must not issue arm or gripper commands.")

    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", rejected_yaw)
    monkeypatch.setattr(environment, "step_joint_targets", unexpected_joint_targets)
    monkeypatch.setattr(environment, "step_physics", physics_steps.append)

    result = adapter.step(np.array([-1.0, 0.5, 0.0, -1.0]))

    assert len(attempts) == len(IK_BACKTRACKING_SCALES)
    assert physics_steps == [PHYSICS_STEPS_PER_ACTION]
    assert result.ik_result.position_converged
    assert not result.ik_result.tool_yaw_converged
    np.testing.assert_array_equal(environment.data.ctrl, initial_controls)
    np.testing.assert_array_equal(adapter.current_target_gripper_position, accepted_target)
    np.testing.assert_array_equal(result.target_gripper_position, accepted_target)
    assert result.state["gripper_target"] == initial_state["gripper_target"]
