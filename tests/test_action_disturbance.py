"""Actuator drop overrides preserve intended commands and ordinary stepping."""

import json

import numpy as np
import pytest

from action_disturbance import DropDisturbanceBackend
from joint_command_backend import JointCommandBackend
from schemas import CommandReceipt, RobotState
from schemas.config import DropDisturbanceConfig


class RecordingBackend(JointCommandBackend):
    def __init__(self):
        self.sent = []
        self.advances = []
        self.fail_send = False
        self.fail_advance = False
        self.receipt = None

    def send_joint_targets(self, targets):
        if self.fail_send:
            raise OSError("transmission failed")
        self.sent.append(targets.copy())
        self.receipt = CommandReceipt(12.0, np.clip(targets, -0.8, 0.8))
        return self.receipt

    def advance(self, duration_seconds):
        if self.fail_advance:
            raise ValueError("invalid physics interval")
        self.advances.append(duration_seconds)


class CountingRNG:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def robot_state():
    return RobotState(
        timestamp=0.0,
        joint_positions=np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.4]),
        joint_velocities=np.zeros(6),
        commanded_joint_targets=np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.6]),
        gripper_position=np.array([0.25, 0.0, 0.1]),
    )


def wrapper(probability=1.0, stop_seconds=0.25, release_seconds=0.25, interval=0.05):
    backend = RecordingBackend()
    rng = CountingRNG()
    config = DropDisturbanceConfig(
        probability=probability, stop_seconds=stop_seconds, release_seconds=release_seconds,
    )
    disturbance = DropDisturbanceBackend(backend, config, rng, interval, 1.0)
    return disturbance, backend, rng


@pytest.mark.parametrize("probability,held", [(0.0, True), (1.0, False)])
def test_ineligible_steps_pass_through_and_do_not_sample(probability, held):
    disturbance, backend, rng = wrapper(probability=probability)
    intended = np.arange(6, dtype=float)
    original = intended.copy()
    disturbance.begin_step(robot_state(), held)
    receipt = disturbance.send_joint_targets(intended)
    assert receipt is backend.receipt
    np.testing.assert_array_equal(backend.sent[0], intended)
    np.testing.assert_array_equal(intended, original)
    assert rng.calls == 0
    assert not disturbance.active
    assert disturbance.events == []
    assert backend.advances == []
    disturbance.advance(0.05)
    assert backend.advances == [0.05]
    assert disturbance.steps == 1


def test_activation_freezes_measured_arm_and_preserves_commanded_squeezing_force():
    disturbance, backend, rng = wrapper()
    state = robot_state()
    measured = state.joint_positions.copy()
    commanded = state.commanded_joint_targets.copy()
    intended = np.full(6, 0.75)
    disturbance.begin_step(state, orange_held=True)
    assert disturbance.active
    assert rng.calls == 1
    receipt = disturbance.send_joint_targets(intended)
    np.testing.assert_array_equal(backend.sent[0][:-1], measured[:-1])
    assert backend.sent[0][-1] == commanded[-1]
    assert backend.sent[0][-1] != measured[-1]
    np.testing.assert_array_equal(intended, np.full(6, 0.75))
    np.testing.assert_array_equal(state.joint_positions, measured)
    np.testing.assert_array_equal(state.commanded_joint_targets, commanded)
    assert receipt is backend.receipt
    assert backend.advances == []
    assert disturbance.steps == 0
    assert disturbance.events == [{"start_step": 0, "stop_steps": 5, "release_steps": 5}]
    json.dumps(disturbance.events)


def test_two_phases_have_exact_duration_and_cannot_retrigger_while_active():
    disturbance, backend, rng = wrapper()
    state = robot_state()
    frozen = state.joint_positions.copy()
    closed = state.commanded_joint_targets[-1]
    for step in range(10):
        disturbance.begin_step(state, orange_held=True)
        receipt = disturbance.send_joint_targets(np.full(6, 0.7))
        assert receipt is backend.receipt
        np.testing.assert_array_equal(backend.sent[-1][:-1], frozen[:-1])
        assert backend.sent[-1][-1] == (closed if step < 5 else 1.0)
        assert disturbance.remaining_steps == 10 - step
        disturbance.advance(0.05)
        assert disturbance.steps == step + 1
        assert disturbance.active == (step < 9)
        state.joint_positions[:] += 0.01
        state.commanded_joint_targets[:] += 0.02
    assert len(backend.sent) == len(backend.advances) == 10
    assert rng.calls == 1
    assert len(disturbance.events) == 1
    # Observation of an unheld cube prevents retriggering after release.
    disturbance.begin_step(state, orange_held=False)
    intended = np.full(6, 0.7)
    disturbance.send_joint_targets(intended)
    disturbance.advance(0.05)
    np.testing.assert_array_equal(backend.sent[-1], intended)
    assert rng.calls == 1
    # Once grasping is observed again, a later drop is eligible.
    disturbance.begin_step(state, orange_held=True)
    assert disturbance.active
    assert rng.calls == 2
    assert disturbance.events[-1] == {"start_step": 11, "stop_steps": 5, "release_steps": 5}
    disturbance.send_joint_targets(intended)
    np.testing.assert_array_equal(backend.sent[-1][:-1], state.joint_positions[:-1])


@pytest.mark.parametrize("duration,interval,expected", [
    (0.25, 0.05, 5),
    (0.3, 0.1, 3),
    (0.251, 0.05, 6),
    (0.001, 0.05, 1),
])
def test_phase_durations_round_up_to_whole_control_intervals(duration, interval, expected):
    disturbance, _, _ = wrapper(stop_seconds=duration, release_seconds=duration, interval=interval)
    disturbance.begin_step(robot_state(), True)
    assert disturbance.events[0]["stop_steps"] == expected
    assert disturbance.events[0]["release_steps"] == expected
    assert disturbance.remaining_steps == 2 * expected


def test_sampling_occurs_once_per_eligible_action():
    disturbance, _, rng = wrapper(probability=0.2)
    rng.value = 0.2
    state = robot_state()
    for _ in range(3):
        disturbance.begin_step(state, True)
        disturbance.send_joint_targets(np.zeros(6))
        disturbance.advance(0.05)
        assert not disturbance.active
    assert rng.calls == 3
    rng.value = 0.199
    disturbance.begin_step(state, True)
    assert disturbance.active
    assert disturbance.events[0]["start_step"] == 3


def test_frozen_targets_do_not_alias_inputs_backend_arguments_or_receipts():
    disturbance, backend, _ = wrapper()
    state = robot_state()
    expected = state.joint_positions.copy()
    expected[-1] = state.commanded_joint_targets[-1]
    disturbance.begin_step(state, True)
    state.joint_positions[:] = 99.0
    state.commanded_joint_targets[:] = 99.0
    first = disturbance.send_joint_targets(np.zeros(6))
    first.joint_targets[:] = 99.0
    backend.sent[-1][:] = 99.0
    disturbance.advance(0.05)
    disturbance.send_joint_targets(np.zeros(6))
    np.testing.assert_array_equal(backend.sent[-1], expected)


@pytest.mark.parametrize("active", [False, True])
def test_failed_send_propagates_and_cannot_silently_advance(active):
    disturbance, backend, _ = wrapper()
    disturbance.begin_step(robot_state(), active)
    remaining = disturbance.remaining_steps
    backend.fail_send = True
    with pytest.raises(OSError, match="transmission failed"):
        disturbance.send_joint_targets(np.zeros(6))
    with pytest.raises(RuntimeError, match="successfully sent"):
        disturbance.advance(0.05)
    assert backend.sent == []
    assert backend.advances == []
    assert disturbance.steps == 0
    assert disturbance.remaining_steps == remaining


def test_failed_advance_does_not_consume_phase_step():
    disturbance, backend, _ = wrapper()
    disturbance.begin_step(robot_state(), True)
    disturbance.send_joint_targets(np.zeros(6))
    backend.fail_advance = True
    with pytest.raises(ValueError, match="physics interval"):
        disturbance.advance(0.05)
    assert disturbance.steps == 0
    assert disturbance.remaining_steps == 10
    assert backend.advances == []
    backend.fail_advance = False
    disturbance.advance(0.05)
    assert disturbance.steps == 1
    assert disturbance.remaining_steps == 9
    assert backend.advances == [0.05]


def test_active_override_requires_an_actuator_command_for_each_advance():
    disturbance, backend, _ = wrapper()
    disturbance.begin_step(robot_state(), True)
    with pytest.raises(RuntimeError, match="successfully sent"):
        disturbance.advance(0.05)
    disturbance.send_joint_targets(np.zeros(6))
    disturbance.advance(0.05)
    with pytest.raises(RuntimeError, match="successfully sent"):
        disturbance.advance(0.05)
    assert backend.advances == [0.05]


@pytest.mark.parametrize("targets", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_active_override_does_not_conceal_invalid_intended_commands(targets):
    disturbance, backend, _ = wrapper()
    disturbance.begin_step(robot_state(), True)
    with pytest.raises(ValueError, match="six finite"):
        disturbance.send_joint_targets(targets)
    assert backend.sent == []
    assert disturbance.remaining_steps == 10


@pytest.mark.parametrize("interval", [0.0, -0.05, np.nan, np.inf])
def test_invalid_control_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="control_interval"):
        wrapper(interval=interval)


def test_nonfinite_open_target_is_rejected():
    config = DropDisturbanceConfig(probability=0.0, stop_seconds=0.25, release_seconds=0.25)
    with pytest.raises(ValueError, match="open_gripper_target"):
        DropDisturbanceBackend(RecordingBackend(), config, np.random.default_rng(0), 0.05, np.nan)
