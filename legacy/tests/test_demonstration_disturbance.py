"""Disturbances change physical execution, never the teacher's action labels."""

from dataclasses import fields
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import torch

import scripts.generate_pickup_demonstrations as generator


class CountingRandom:
    def __init__(self, *values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


class FakeAdapter:
    def __init__(self, state):
        self.config = generator.CartesianActionConfig()
        self.reset(state)

    @property
    def current_target_gripper_position(self):
        return self._current_target_gripper_position.copy()

    def reset(self, state):
        self._current_target_gripper_position = state["gripper_position"].copy()
        self._previous_target_gripper_position = state["gripper_position"].copy()


class FakeSimulation:
    """Small deterministic actuator model, deliberately separate from Gym."""

    def __init__(self, held):
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.005))
        self.state = {
            "time": 0.0,
            "joint_positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1]),
            "gripper_position": np.array([0.3, 0.0, 0.1]),
            "gripper_target": -0.1,
            "orange_currently_held": held,
            "orange_position": np.array([0.3, 0.0, 0.075]),
            "orange_velocity": np.zeros(6),
        }
        self.commands = []
        self.fail_on_step = False

    def get_state(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.state.items()}

    def step_joint_targets(self, targets):
        if self.fail_on_step:
            raise RuntimeError("simulated actuator failure")
        targets = np.asarray(targets).copy()
        self.commands.append(targets)
        movement = targets[:3] - self.state["joint_positions"][:3]
        self.state["gripper_position"] += movement
        self.state["joint_positions"] = targets
        self.state["gripper_target"] = float(targets[-1])
        self.state["time"] += 0.05
        if targets[-1] > 0:
            self.state["orange_currently_held"] = False
            self.state["orange_position"][2] -= 0.002
        return self.get_state()


class FakeEnvironment:
    def __init__(self, held=True):
        self.simulation = FakeSimulation(held)
        self.previous_state = self.simulation.get_state()
        self.action_adapter = FakeAdapter(self.previous_state)
        self.episode_step_count = 0
        self.intended_actions = []
        self.success_next = False

    def observation(self):
        observation = np.zeros(generator.PRIVILEGED_OBSERVATION_SIZE, dtype=np.float32)
        observation[:3] = self.previous_state["gripper_position"]
        observation[3] = self.previous_state["time"]
        observation[4] = self.previous_state["gripper_target"]
        observation[5] = self.previous_state["orange_position"][2]
        return observation

    def info(self):
        return {
            "target_gripper_position": self.action_adapter.current_target_gripper_position,
            "orange_currently_held": self.previous_state["orange_currently_held"],
            "orange_grasp_hold_time": 0.0,
            "stack_stable_time": (generator.StackSuccessConfig().required_stable_time
                                  if self.success_next else 0.0),
            "is_failure": False,
            "is_success": self.success_next,
            "ik_position_converged": True,
            "ik_tool_yaw_converged": True,
        }

    def step(self, action):
        self.intended_actions.append(np.asarray(action).copy())
        adapter = self.action_adapter
        before = adapter.current_target_gripper_position
        target = before + np.asarray(action[:3]) * adapter.config.maximum_position_delta
        targets = self.previous_state["joint_positions"].copy()
        targets[:3] += target - self.previous_state["gripper_position"]
        if action[3] <= adapter.config.close_gripper_command_threshold:
            targets[-1] = adapter.config.closed_gripper_target
        elif action[3] >= adapter.config.open_gripper_command_threshold:
            targets[-1] = adapter.config.open_gripper_target
        self.previous_state = self.simulation.step_joint_targets(targets)
        # Like the production adapter, initially commit the requested IK target;
        # the disturbance must correct this to the physically frozen target.
        adapter._previous_target_gripper_position = before
        adapter._current_target_gripper_position = target.copy()
        self.episode_step_count += 1
        return self.observation(), 0.0, self.success_next, False, self.info()


def make_disturbance(monkeypatch, *draws, probability=0.002, **kwargs):
    random = CountingRandom(*draws)
    monkeypatch.setattr(generator.np.random, "default_rng", lambda seed: random)
    disturbance = generator.DropDisturbance(123, probability=probability, **kwargs)
    return disturbance, random


@pytest.mark.parametrize("draw, triggered", [(0.001999, True), (0.002, False), (0.8, False)])
def test_probability_threshold_is_point_two_percent(monkeypatch, draw, triggered):
    disturbance, random = make_disturbance(monkeypatch, draw)
    env = FakeEnvironment()
    disturbance.step(env, np.array([0.3, -0.2, 0.1, -1.0]))
    assert disturbance.active is triggered
    assert len(disturbance.events) == int(triggered)
    assert random.calls == 1


def test_unheld_steps_never_sample_or_override(monkeypatch):
    disturbance, random = make_disturbance(monkeypatch)
    env = FakeEnvironment(held=False)
    intended = np.array([0.5, -0.25, 0.75, 1.0])
    disturbance.step(env, intended)
    assert random.calls == 0
    assert not disturbance.active
    assert not disturbance.events
    np.testing.assert_array_equal(env.intended_actions[0], intended)
    assert env.simulation.commands[0][-1] == env.action_adapter.config.open_gripper_target


def test_zero_probability_disables_sampling_even_while_held(monkeypatch):
    disturbance, random = make_disturbance(monkeypatch, probability=0.0)
    env = FakeEnvironment()
    intended = np.array([0.5, -0.25, 0.75, -1.0])
    disturbance.step(env, intended)
    assert random.calls == 0
    assert not disturbance.active
    assert not disturbance.events
    np.testing.assert_array_equal(env.intended_actions[0], intended)
    np.testing.assert_allclose(env.action_adapter.current_target_gripper_position,
                               np.array([0.3, 0.0, 0.1]) + intended[:3] * 0.0025)


def test_five_stop_then_five_release_steps_preserve_time_and_arm_pose(monkeypatch):
    disturbance, random = make_disturbance(monkeypatch, 0.0)
    env = FakeEnvironment()
    initial = env.previous_state
    measured_joints = initial["joint_positions"].copy()
    measured_position = initial["gripper_position"].copy()
    intended = np.array([0.75, 0.5, -0.25, -1.0])
    original_hook = env.simulation.step_joint_targets
    for index in range(10):
        observation, _, _, _, info = disturbance.step(env, intended)
        np.testing.assert_array_equal(env.intended_actions[index], intended)
        np.testing.assert_array_equal(env.simulation.commands[index][:-1], measured_joints[:-1])
        expected_gripper = (-0.1 if index < 5
                            else env.action_adapter.config.open_gripper_target)
        assert env.simulation.commands[index][-1] == expected_gripper
        np.testing.assert_allclose(info["target_gripper_position"], measured_position)
        np.testing.assert_allclose(env.action_adapter.current_target_gripper_position, measured_position)
        assert observation[3] == pytest.approx((index + 1) * 0.05)
        assert env.simulation.step_joint_targets == original_hook
        assert disturbance.active is (index < 9)
    assert env.episode_step_count == 10
    assert random.calls == 1  # No retriggering during either phase, even while held.
    assert len(disturbance.events) == 1
    assert env.previous_state["orange_currently_held"] is False
    assert env.previous_state["orange_position"][2] < initial["orange_position"][2]

    # Override removal resumes the teacher command without target windup.
    disturbance.step(env, intended)
    assert random.calls == 1  # Still not held, so no new activation draw.
    assert env.simulation.commands[-1][-1] == env.action_adapter.config.closed_gripper_target
    np.testing.assert_allclose(env.action_adapter.current_target_gripper_position,
                               measured_position + intended[:3] * 0.0025)


def test_actuator_hook_is_restored_when_physics_raises(monkeypatch):
    disturbance, _ = make_disturbance(monkeypatch, 0.0)
    env = FakeEnvironment()
    original_hook = env.simulation.step_joint_targets
    env.simulation.fail_on_step = True
    with pytest.raises(RuntimeError, match="simulated actuator failure"):
        disturbance.step(env, np.array([0.2, 0.0, 0.0, -1.0]))
    assert env.simulation.step_joint_targets == original_hook


def test_rejected_ik_does_not_consume_disturbance_or_publish_fake_accepted_target(monkeypatch):
    disturbance, _ = make_disturbance(monkeypatch, 0.0)
    env = FakeEnvironment()
    original_hook = env.simulation.step_joint_targets
    # Simulate tracking lag so a false reset to the measured pose is detectable.
    accepted_target = np.array([0.31, 0.02, 0.08])
    env.action_adapter._current_target_gripper_position = accepted_target.copy()
    info = env.info()
    info["ik_position_converged"] = False
    # The best-effort IK fallback advances physics without invoking this hook.
    monkeypatch.setattr(env, "step", lambda action: (env.observation(), 0.0, False, False, info))
    with pytest.raises(generator.DemonstrationFailure, match="IK did not accept"):
        disturbance.step(env, np.array([0.2, 0.0, 0.0, -1.0]))
    assert disturbance.remaining_steps == 10
    assert disturbance.steps == 0
    assert not env.simulation.commands
    assert env.simulation.step_joint_targets == original_hook
    np.testing.assert_array_equal(env.action_adapter.current_target_gripper_position, accepted_target)
    np.testing.assert_array_equal(info["target_gripper_position"], accepted_target)


def test_recorder_keeps_intended_actions_actual_states_and_every_timestamp(monkeypatch):
    disturbance, _ = make_disturbance(monkeypatch, 0.0)
    env = FakeEnvironment()
    recorder = generator.EpisodeRecorder(env.observation(), env.info(), disturbance=disturbance)
    intended = np.array([0.75, 0.25, -0.5, -1.0], dtype=np.float32)
    for _ in range(10):
        assert not recorder.step(env, intended, "transport")
    np.testing.assert_array_equal(recorder.actions, np.tile(intended, (10, 1)))
    assert len(recorder.observations) == len(recorder.targets) == 11
    np.testing.assert_allclose(np.asarray(recorder.observations)[:, 3], np.arange(11) * 0.05)
    assert recorder.observations[-1][4] == pytest.approx(env.action_adapter.config.open_gripper_target)
    assert np.all(np.asarray(recorder.actions)[:, 3] == -1.0)
    assert recorder.recovery_requested
    assert recorder.stages["transport"] == 10

    # Legitimate release commands still belong in the labels. No extra tensor
    # or per-action supervision mask is needed to validate the saved episode.
    recorder.begin_recovery()
    env.success_next = True
    opening = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert recorder.step(env, opening, "release")
    episode = recorder.episode()
    episode.validate()
    np.testing.assert_array_equal(episode.actions[-1], opening)
    assert {field.name for field in fields(episode)} >= {"observations", "accepted_targets", "actions"}
    assert not hasattr(episode, "loss_mask")


def test_brief_contact_gap_does_not_request_recovery_and_begin_preserves_history():
    env = FakeEnvironment()
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    closed = np.array([0.0, 0.0, 0.0, -1.0])
    for held in (True, False, True, False):
        env.simulation.state["orange_currently_held"] = held
        recorder.step(env, closed, "transport")
        assert not recorder.recovery_requested
    env.simulation.state["orange_currently_held"] = False
    recorder.step(env, closed, "transport")
    assert recorder.recovery_requested
    before = (len(recorder.actions), len(recorder.observations), len(recorder.targets), dict(recorder.stages))
    recorder.begin_recovery()
    assert not recorder.recovery_requested
    assert before == (len(recorder.actions), len(recorder.observations), len(recorder.targets), dict(recorder.stages))


def test_legitimate_release_does_not_restart_pickup():
    env = FakeEnvironment()
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    recorder.step(env, np.array([0.0, 0.0, 0.0, -1.0]), "transport")
    for stage in ("release", "retreat", "settle_stack"):
        recorder.step(env, np.array([0.0, 0.0, 0.0, 1.0]), stage)
        assert not recorder.recovery_requested
    assert all(action[3] == 1.0 for action in recorder.actions[1:])


def test_lost_cube_stops_transport_and_requests_recovery():
    env = FakeEnvironment(held=False)
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    assert not generator.move_held_cube_to(env, recorder, np.array([0.3, 0.05, 0.08]), "transport")
    assert recorder.recovery_requested
    assert not recorder.actions


@pytest.mark.parametrize("pause_steps", [0, 2, 5])
def test_recovery_pauses_then_counts_settling_without_moving_target(monkeypatch, pause_steps):
    monkeypatch.setattr(generator, "RECOVERY_PAUSE_STEPS", pause_steps)
    env = FakeEnvironment(held=False)
    # Keep a target different from the measured pose: zero deltas must retain
    # the accepted target instead of creating a new XYZ or clearance target.
    accepted = np.array([0.31, 0.02, 0.09])
    env.action_adapter._current_target_gripper_position = accepted.copy()
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    recorder.step(env, np.array([0.0, 0.0, 0.0, -1.0]), "transport")
    history_before = np.asarray(recorder.observations).copy()
    recorder.recovery_requested = True
    recorder._grasp_seen = True
    recorder._lost_grasp_steps = 2

    assert not generator.recover_for_pickup(env, recorder)

    assert recorder.recovery_count == 1
    assert not recorder.recovery_requested
    assert not recorder._grasp_seen
    assert recorder._lost_grasp_steps == 0
    assert recorder.stages["recovery_pause"] == pause_steps
    assert recorder.stages["recovery_settle"] == generator.RECOVERY_SETTLE_STEPS
    assert "recovery_clear" not in recorder.stages
    total_steps = 1 + pause_steps + generator.RECOVERY_SETTLE_STEPS
    assert env.episode_step_count == len(recorder.actions) == total_steps
    assert len(recorder.observations) == len(recorder.targets) == total_steps + 1
    assert env.previous_state["time"] == pytest.approx(total_steps * 0.05)
    np.testing.assert_array_equal(recorder.observations[:len(history_before)], history_before)
    np.testing.assert_array_equal(recorder.actions[1:],
                                  np.tile([0.0, 0.0, 0.0, generator.OPEN_GRIPPER], (total_steps - 1, 1)))
    np.testing.assert_allclose(recorder.targets, np.tile(accepted, (total_steps + 1, 1)))


@pytest.mark.parametrize("violated_gate", ["linear", "angular", "held"])
def test_recovery_requires_consecutive_quiet_steps_after_pause(monkeypatch, violated_gate):
    monkeypatch.setattr(generator, "RECOVERY_PAUSE_STEPS", 5)
    env = FakeEnvironment(held=False)
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    original_step = env.step
    # All five pause steps are quiet, then four quiet settling steps, one
    # invalid step, and five quiet steps. Pause time cannot count as settling.
    invalid_step = generator.RECOVERY_PAUSE_STEPS + 5

    def step(action):
        observation, reward, terminated, truncated, info = original_step(action)
        velocity = np.zeros(6)
        held = False
        if env.episode_step_count == invalid_step:
            if violated_gate == "linear":
                velocity[0] = generator.RECOVERY_LINEAR_SPEED + 0.001
            elif violated_gate == "angular":
                velocity[3] = generator.RECOVERY_ANGULAR_SPEED + 0.001
            else:
                held = True
        env.previous_state["orange_velocity"] = velocity
        env.previous_state["orange_currently_held"] = held
        info["orange_currently_held"] = held
        return observation, reward, terminated, truncated, info

    monkeypatch.setattr(env, "step", step)
    assert not generator.recover_for_pickup(env, recorder)
    assert recorder.stages["recovery_pause"] == 5
    assert recorder.stages["recovery_settle"] == 10
    assert env.episode_step_count == 15


@pytest.mark.parametrize("success_step, stage", [(2, "recovery_pause"), (7, "recovery_settle")])
def test_recovery_stops_immediately_for_verified_stack(monkeypatch, success_step, stage):
    monkeypatch.setattr(generator, "RECOVERY_PAUSE_STEPS", 5)
    env = FakeEnvironment(held=False)
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    original_step = env.step

    def step(action):
        env.success_next = env.episode_step_count + 1 == success_step
        return original_step(action)

    monkeypatch.setattr(env, "step", step)
    assert generator.recover_for_pickup(env, recorder)
    assert recorder.finished and recorder.verified_success
    assert env.episode_step_count == len(recorder.actions) == success_step
    assert recorder.stages[stage] == 2
    if stage == "recovery_pause":
        assert "recovery_settle" not in recorder.stages


@pytest.mark.parametrize("timeout_step, stage", [(2, "recovery_pause"), (7, "recovery_settle")])
def test_recovery_preserves_episode_timeout(monkeypatch, timeout_step, stage):
    monkeypatch.setattr(generator, "RECOVERY_PAUSE_STEPS", 5)
    env = FakeEnvironment(held=False)
    recorder = generator.EpisodeRecorder(env.observation(), env.info())
    original_step = env.step

    def step(action):
        observation, reward, terminated, _, info = original_step(action)
        return observation, reward, terminated, env.episode_step_count == timeout_step, info

    monkeypatch.setattr(env, "step", step)
    with pytest.raises(generator.DemonstrationFailure, match=f"{stage}: episode timed out"):
        generator.recover_for_pickup(env, recorder)
    assert recorder.finished
    assert not recorder.verified_success
    assert env.episode_step_count == len(recorder.actions) == timeout_step


def test_recovery_fails_if_cube_never_settles(monkeypatch):
    monkeypatch.setattr(generator, "RECOVERY_PAUSE_STEPS", 5)
    monkeypatch.setattr(generator, "MAXIMUM_MOVE_STEPS", 8)
    env = FakeEnvironment(held=False)
    env.simulation.state["orange_velocity"][0] = generator.RECOVERY_LINEAR_SPEED + 0.001
    recorder = generator.EpisodeRecorder(env.observation(), env.info())

    with pytest.raises(generator.DemonstrationFailure, match="orange did not settle"):
        generator.recover_for_pickup(env, recorder)
    assert recorder.stages["recovery_pause"] == 5
    assert recorder.stages["recovery_settle"] == 8
    assert env.episode_step_count == 13


def test_physical_drop_recovery_finishes_stack_without_reset_or_missing_steps(monkeypatch):
    """A deterministic real MuJoCo episode exercises the entire recovery loop."""
    original_disturbance = generator.DropDisturbance

    class SingleAirborneDrop(original_disturbance):
        def step(self, environment, intended_action):
            state = environment.previous_state
            self.probability = float(not self.events and state["orange_currently_held"]
                                     and state["orange_position"][2] > 0.06)
            return super().step(environment, intended_action)

    monkeypatch.setattr(generator, "DropDisturbance", SingleAirborneDrop)
    monkeypatch.setattr(generator, "START_POSITION_HALF_RANGE", (0.0, 0.0, 0.0))
    environment = generator.make_environment()
    original_reset = environment.reset
    original_step = environment.step
    original_joint_step = environment.simulation.step_joint_targets
    resets = []
    held_states = []
    physical_commands = []

    def reset(**kwargs):
        resets.append(kwargs)
        return original_reset(**kwargs)

    def step(action):
        transition = original_step(action)
        held_states.append(bool(transition[-1]["orange_currently_held"]))
        return transition

    def joint_step(targets):
        physical_commands.append(np.array(targets, copy=True))
        return original_joint_step(targets)

    monkeypatch.setattr(environment, "reset", reset)
    monkeypatch.setattr(environment, "step", step)
    monkeypatch.setattr(environment.simulation, "step_joint_targets", joint_step)
    try:
        episode = generator.collect_episode(environment, seed=21)
        episode.validate()
        assert resets == [{"seed": 21}]
        assert episode.recovery_count == 1
        assert len(episode.disturbance_events) == 1
        assert episode.stage_steps["recovery_pause"] == generator.RECOVERY_PAUSE_STEPS
        assert "recovery_clear" not in episode.stage_steps
        assert episode.stage_steps["recovery_settle"] > 0
        assert len(episode.observations) == len(episode.actions) + 1
        assert len(physical_commands) == len(episode.actions) == environment.episode_step_count
        assert environment.previous_state["time"] == pytest.approx(len(episode.actions) * 0.05)
        event = episode.disturbance_events[0]
        release_start = event["start_step"] + event["stop_steps"]
        release_end = release_start + event["release_steps"]
        assert event["stop_steps"] == event["release_steps"] == 5
        assert all(target[-1] == environment.action_adapter.config.open_gripper_target
                   for target in physical_commands[release_start:release_end])
        # At least one forced opening physically occurred while the saved
        # intended label still requested a closed claw.
        assert any(action[3] == generator.CLOSED_GRIPPER
                   for action in episode.actions[release_start:release_end])
        assert held_states[event["start_step"] - 1]
        assert not held_states[release_end - 1]
        assert any(held_states[release_end:])  # Actual regrasp, not stage-only recovery.
        assert environment.simulation.is_success()
        assert environment.simulation.step_joint_targets == joint_step

        # Use the unmodified pretraining loader on the same real recovery
        # episode. Temporary files stay inside the repository and are removed.
        from scripts.pretrain_pickup import (
            load_episode, materialize_sequences, read_episode_records, sequence_chunks,
        )
        with tempfile.TemporaryDirectory(prefix=".test-recovery-", dir=generator.REPOSITORY_ROOT) as name:
            directory = Path(name)
            record = generator.save_episode(directory, episode, UUID(int=21), "train",
                                            generator.generation_settings(environment))
            records = read_episode_records(directory)
            assert records == json.loads(json.dumps([record]))
            assert record["recovery_count"] == 1
            assert record["disturbance_events"] == episode.disturbance_events
            assert {path.parent.name for path in directory.rglob("*.pt")} == {
                "observations", "accepted_targets", "actions",
            }
            assert len(list(directory.rglob("*.pt"))) == 3
            observations, targets, actions = load_episode(directory, record)
            for stored, original in zip((observations, targets, actions),
                                        (episode.observations, episode.accepted_targets, episode.actions)):
                torch.testing.assert_close(stored, original)
            tokens, valid, starts, labels, loss_mask = materialize_sequences(
                directory, records, 64, "recovery test", sequence_stride=48,
                show_progress=False,
            ).tensors
            previous_actions = torch.cat((torch.zeros((1, 4)), actions[:-1]))
            coverage = torch.zeros(len(actions), dtype=torch.int32)
            for row, (first, end, first_label) in enumerate(sequence_chunks(len(actions), 64, 48)):
                length = end - first
                torch.testing.assert_close(tokens[row, :length, :49], observations[first:end])
                torch.testing.assert_close(tokens[row, :length, 49:53], previous_actions[first:end])
                torch.testing.assert_close(tokens[row, :length, -3:], targets[first:end])
                torch.testing.assert_close(labels[row, :length], actions[first:end])
                assert starts[row].sum() == int(first == 0)
                assert not loss_mask[row, length:].any()
                assert not valid[row, length:].any()
                coverage[first:end] += loss_mask[row, :length].int()
            # Ordinary overlap/padding masks supervise every intended action
            # exactly once, including all disturbance timestamps.
            torch.testing.assert_close(coverage, torch.ones_like(coverage))
            assert loss_mask.sum() == len(actions)
    finally:
        environment.close()
