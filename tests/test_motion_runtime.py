"""Physics, IK, acceptance, and calibrated hardware integration contracts."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import cartesian_controller
from cartesian_controller import CartesianController, IKConvergenceError
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend, SO101JointCommandBackend
from kinematics import forward_kinematics, joint_limits, solve_ik
from robot_constants import ROBOT_JOINT_NAMES
from robot_state_provider import SimRobotStateProvider, SO101RobotStateProvider
from schemas import CommandReceipt
from schemas.config import load_config


@pytest.fixture
def environment():
    return CubeStackEnvironment()


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.json").cartesian_controller


def test_reset_opens_gripper_and_restores_xml_without_time_advance(environment):
    provider = SimRobotStateProvider(environment)
    before = provider.read()
    assert before.timestamp == 0
    np.testing.assert_array_equal(before.joint_positions, [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(before.commanded_joint_targets, before.joint_positions)
    backend = SimJointCommandBackend(environment)
    backend.send_joint_targets(np.zeros(6))
    backend.advance(0.05)
    environment.reset()
    after = provider.read()
    assert after.timestamp == 0
    np.testing.assert_array_equal(after.joint_positions, before.joint_positions)
    np.testing.assert_array_equal(after.gripper_position, before.gripper_position)
    np.testing.assert_array_equal(after.joint_velocities, np.zeros(6))


def test_read_and_send_do_not_step_and_snapshots_do_not_alias(environment):
    provider, backend = SimRobotStateProvider(environment), SimJointCommandBackend(environment)
    state = provider.read()
    state.joint_positions[:] = 17
    state.commanded_joint_targets[:] = 19
    state.gripper_position[:] = 21
    current = provider.read()
    assert np.all(np.abs(current.joint_positions) <= 1)
    qpos_before = environment.data.qpos.copy()
    receipt = backend.send_joint_targets(np.array([0.1, 0, 0, 0, 0, 1]))
    assert receipt.timestamp == 0
    np.testing.assert_array_equal(qpos_before, environment.data.qpos)
    receipt.joint_targets[:] = 55
    assert provider.read().commanded_joint_targets[0] == 0.1
    assert environment.data.time == 0
    backend.advance(0.05)
    assert environment.data.time == pytest.approx(0.05)
    measured = provider.read()
    xyz, _ = forward_kinematics(environment.model, measured.joint_positions)
    np.testing.assert_allclose(measured.gripper_position, xyz, atol=1e-12)


def test_backend_reports_joint_and_actuator_limiting(environment):
    lower, upper = joint_limits(environment.model)
    backend = SimJointCommandBackend(environment)
    requested = np.array([100, -100, 100, -100, 100, -100], dtype=float)
    receipt = backend.send_joint_targets(requested)
    np.testing.assert_array_equal(receipt.joint_targets, np.clip(requested, lower, upper))
    state = SimRobotStateProvider(environment).read()
    np.testing.assert_array_equal(receipt.joint_targets, state.commanded_joint_targets)
    with pytest.raises(ValueError, match="six finite"):
        backend.send_joint_targets(np.full(6, np.nan))
    with pytest.raises(ValueError, match="integer multiple"):
        backend.advance(0.003)
    with pytest.raises(ValueError, match="nonnegative"):
        backend.advance(-1)


def test_default_ik_works_from_xml_home_without_mutating_simulation(environment, config):
    state = SimRobotStateProvider(environment).read()
    controller = CartesianController(environment.model, config)
    with pytest.raises(RuntimeError, match="Reset"):
        controller.compute(np.zeros(4), state)
    controller.reset(state)
    live_qpos, live_time = environment.data.qpos.copy(), environment.data.time
    for action in [np.zeros(4), np.array([0, 0, -1, -1]), np.array([-1, 1, 0, 0])]:
        command = controller.compute(action, state)
        position, rotation = forward_kinematics(environment.model, command.joint_targets)
        assert np.linalg.norm(position - command.cartesian_target) <= config.position_tolerance
        angle = np.arccos(np.clip(rotation[:, 0] @ [0, 0, -1], -1, 1))
        assert angle <= config.tool_axis_tolerance
        heading = np.arcsin(abs(rotation[1, 2]))
        assert heading <= config.tool_yaw_tolerance
        np.testing.assert_array_equal(controller.prev_accepted_target, state.gripper_position)
        assert controller.gripper_target == 1
    np.testing.assert_array_equal(live_qpos, environment.data.qpos)
    assert environment.data.time == live_time


def test_strict_downward_ik_enforces_angle_and_joint_limits(environment, config):
    strict = config.model_copy(update={"tool_axis_tolerance": float(np.deg2rad(10))})
    state = SimRobotStateProvider(environment).read()
    target = np.array([0.25, 0.05, 0.05])
    joints, diagnostics = solve_ik(environment.model, state.joint_positions, target, strict)
    assert diagnostics["converged"]
    xyz, rotation = forward_kinematics(environment.model, joints)
    assert np.linalg.norm(xyz - target) <= strict.position_tolerance
    assert np.arccos(np.clip(-rotation[2, 0], -1, 1)) <= strict.tool_axis_tolerance
    assert np.arcsin(abs(rotation[1, 2])) <= strict.tool_yaw_tolerance
    lo, hi = joint_limits(environment.model)
    assert np.all(joints >= lo) and np.all(joints <= hi)


def test_downward_requirement_can_be_disabled(environment, config):
    off = config.model_copy(update={
        "require_downward": False, "target_tool_yaw": None,
        "tool_axis_tolerance": float(np.deg2rad(1)),
    })
    state = SimRobotStateProvider(environment).read()
    joints, diagnostics = solve_ik(environment.model, state.joint_positions, state.gripper_position, off)
    assert diagnostics["converged"]
    assert diagnostics["tool_axis_error"] > off.tool_axis_tolerance
    np.testing.assert_array_equal(joints, state.joint_positions)


def test_unreachable_ik_raises_diagnostics_without_committing(environment, config):
    state = SimRobotStateProvider(environment).read()
    controller = CartesianController(environment.model, config)
    controller.reset(state)
    controller.prev_accepted_target = np.array([10.0, 10.0, 10.0])
    before = controller.prev_accepted_target.copy()
    with pytest.raises(IKConvergenceError, match="4 target attempts") as exc:
        controller.compute(np.array([1, 0, 0, -1]), state)
    assert len(exc.value.diagnostics["attempts"]) == len(config.backtracking_scales)
    assert not any(attempt["converged"] for attempt in exc.value.diagnostics["attempts"])
    json.dumps(exc.value.diagnostics)
    np.testing.assert_array_equal(controller.prev_accepted_target, before)
    assert controller.gripper_target == 1


def test_backtracking_tries_smaller_delta_and_preserves_command_state(environment, config, monkeypatch):
    state = SimRobotStateProvider(environment).read()
    controller = CartesianController(environment.model, config)
    controller.reset(state)
    seen = []

    def solve(model, initial, target, cfg):
        seen.append(target.copy())
        return initial.copy(), {"converged": len(seen) == 2}

    monkeypatch.setattr(cartesian_controller, "solve_ik", solve)
    command = controller.compute(np.array([4, 0, 0, -1]), state)
    assert command.ik_diagnostics["accepted_scale"] == config.backtracking_scales[1]
    np.testing.assert_allclose(seen[0] - state.gripper_position, [config.maximum_position_delta, 0, 0])
    np.testing.assert_allclose(seen[1] - state.gripper_position, [config.maximum_position_delta * 0.5, 0, 0])
    assert controller.gripper_target == 1
    np.testing.assert_array_equal(controller.prev_accepted_target, state.gripper_position)


def test_command_acceptance_and_gripper_persistence(environment, config):
    state = SimRobotStateProvider(environment).read()
    controller = CartesianController(environment.model, config)
    controller.reset(state)
    command = controller.compute(np.array([0, 0, -1, -1]), state)
    receipt = SimJointCommandBackend(environment).send_joint_targets(command.joint_targets)
    controller.update_prev_accepted_xyz(command, receipt)
    np.testing.assert_array_equal(controller.prev_accepted_target, command.cartesian_target)
    assert controller.gripper_target == config.closed_gripper_target
    neutral = controller.compute(np.zeros(4), state)
    assert neutral.joint_targets[-1] == config.closed_gripper_target
    changed_joints = receipt.joint_targets.copy()
    changed_joints[0] += 0.04
    changed_joints[-1] = 0.3
    changed = CommandReceipt(timestamp=0.0, joint_targets=changed_joints)
    controller.update_prev_accepted_xyz(command, changed)
    expected, _ = forward_kinematics(environment.model, changed_joints)
    np.testing.assert_allclose(controller.prev_accepted_target, expected)
    assert controller.gripper_target == 0.3


class CalibratedConnection:
    def __init__(self):
        self.positions = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 1.0])
        self.velocities = np.arange(6) * 0.1
        self.targets = self.positions.copy()
        self.fail = False

    def read_joint_state(self):
        return self.positions, self.velocities, self.targets

    def send_joint_targets(self, targets):
        if self.fail:
            raise OSError("servo transmission failed")
        self.targets = np.clip(targets, -0.5, 0.5)
        return self.targets


def test_hardware_shares_adapter_returns_copies_and_propagates_failures(environment, config):
    connection = CalibratedConnection()
    provider = SO101RobotStateProvider(connection, environment.model, clock=lambda: 123.0)
    backend = SO101JointCommandBackend(connection, clock=lambda: 124.0)
    state = provider.read()
    assert state.timestamp == 123.0
    np.testing.assert_array_equal(state.joint_positions, connection.positions)
    expected, _ = forward_kinematics(environment.model, connection.positions)
    np.testing.assert_array_equal(state.gripper_position, expected)
    state.joint_positions[:] = 17
    assert connection.positions[0] == 0.01
    receipt = backend.send_joint_targets(np.ones(6))
    assert receipt.timestamp == 124.0
    np.testing.assert_array_equal(receipt.joint_targets, np.full(6, 0.5))
    receipt.joint_targets[:] = 19
    np.testing.assert_array_equal(provider.read().commanded_joint_targets, np.full(6, 0.5))
    backend.advance(0.05)
    assert environment.data.time == 0
    controller = CartesianController(environment.model, config)
    state = provider.read()
    controller.reset(state)
    before = controller.prev_accepted_target.copy()
    command = controller.compute(np.zeros(4), state)
    connection.fail = True
    with pytest.raises(OSError, match="transmission failed"):
        backend.send_joint_targets(command.joint_targets)
    np.testing.assert_array_equal(controller.prev_accepted_target, before)


def test_downward_preference_improves_orientation_inside_loose_hard_tolerance(environment, config):
    state = SimRobotStateProvider(environment).read()
    initial_angle = np.arccos(-forward_kinematics(environment.model, state.joint_positions)[1][2, 0])
    joints, result = solve_ik(environment.model, state.joint_positions, state.gripper_position, config)
    xyz, rotation = forward_kinematics(environment.model, joints)
    assert result["converged"]
    assert np.linalg.norm(xyz - state.gripper_position) <= config.position_tolerance
    final_angle = np.arccos(np.clip(-rotation[2, 0], -1, 1))
    assert final_angle < initial_angle - np.deg2rad(5)
    assert final_angle <= config.tool_axis_tolerance
