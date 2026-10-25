"""Demonstrations retain intended labels, exact pre-action states, and recovery."""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

import action_demonstrations as demonstrations
from action_demonstrations import (
    ActionEpisode, DemonstrationFailure, EpisodeRecorder, collect_action_episode,
    pickup_from_current_pose, reset_demonstration_environment,
)
from action_disturbance import DropDisturbanceBackend
from action_preprocessing import ActionObservationPreprocessor
from environment import CubeStackEnvironment
from kinematics import IKConvergenceError, forward_kinematics
from schemas.config import load_config
from training import build_action_windows


@pytest.fixture
def project():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    config.environment.action_data.drop_disturbance.probability = 0.0
    return config


@pytest.fixture
def recorder(project):
    environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
    reset_demonstration_environment(environment, project, np.random.default_rng(0))
    return EpisodeRecorder(project, environment, 0)


def test_reset_uses_randomized_home_and_separated_axis_aligned_ground_cubes(recorder):
    scene = recorder.scene()
    config = recorder.config
    assert scene["time"] == 0.0
    tolerance = recorder.project.cartesian_controller.position_tolerance
    assert np.all(abs(scene["gripper"] - config.start_position)
                  <= np.asarray(config.start_position_half_range) + tolerance)
    np.testing.assert_array_equal(scene["robot"].joint_positions, scene["robot"].commanded_joint_targets)
    assert scene["robot"].joint_positions[-1] == recorder.project.cartesian_controller.open_gripper_target
    assert np.linalg.norm(scene["orange"][:2] - scene["blue"][:2]) >= config.minimum_cube_separation
    for cube in scene["cubes"].values():
        np.testing.assert_array_equal(cube.orientation, [1, 0, 0, 0])
        np.testing.assert_array_equal(cube.linear_velocity, [0, 0, 0])


def test_recording_has_pre_action_ground_truth_in_new_41_value_layout(recorder):
    data = recorder.environment.data
    data.joint("orange_cube_joint").qvel[:] = [.001, .002, .003, .004, .005, .006]
    yaw = .3
    data.joint("blue_cube_joint").qpos[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(recorder.environment.model, data)
    before = recorder.scene()
    target = recorder.controller.prev_accepted_target.copy()
    intended = np.array([.1, -.1, .1, -1], dtype=np.float32)
    recorder.step(intended, "test")
    observation = recorder.observations[0]
    assert observation.shape == (41,)
    assert observation.dtype == np.float32
    np.testing.assert_allclose(observation[:6], before["robot"].joint_positions)
    np.testing.assert_allclose(observation[6:12], before["robot"].joint_velocities)
    np.testing.assert_allclose(observation[12:18], before["robot"].commanded_joint_targets)
    np.testing.assert_allclose(observation[18:21], before["gripper"])
    np.testing.assert_allclose(observation[21:24], before["orange"])
    np.testing.assert_allclose(observation[24:28], before["cubes"]["orange"].orientation)
    np.testing.assert_allclose(observation[28:31], [.001, .002, .003])
    np.testing.assert_allclose(observation[31:34], before["blue"])
    np.testing.assert_allclose(observation[34:38], before["cubes"]["blue"].orientation)
    np.testing.assert_allclose(observation[38:41], before["cubes"]["blue"].linear_velocity)
    np.testing.assert_allclose(recorder.targets[0], target)
    np.testing.assert_array_equal(recorder.actions[0], intended)
    intended[:] = 0
    assert recorder.actions[0][-1] == -1
    assert data.time == pytest.approx(recorder.interval)
    assert len(recorder.observations) == len(recorder.actions) == len(recorder.targets) == 1


@pytest.mark.parametrize("learned_observations", [False, True])
def test_override_keeps_intended_labels_and_next_actual_command_feedback(recorder, monkeypatch,
                                                                      learned_observations):
    if learned_observations:
        def observe(robot):
            result = np.zeros(37, dtype=np.float32)
            result[:21] = np.concatenate((robot.joint_positions, robot.joint_velocities,
                                         robot.commanded_joint_targets, robot.gripper_position))
            result[21:24], result[29:32] = [1, 2, 3], [4, 5, 6]
            result[24:26], result[32:34] = [.6, .8], [-.8, .6]
            return result
        recorder.observation_builder = observe
    recorder.backend.config.probability = 1.0
    recorder.backend.stop_steps = recorder.backend.release_steps = 1
    scene = recorder.scene

    def held_scene():
        state = scene()
        state["held"] = True
        return state

    monkeypatch.setattr(recorder, "scene", held_scene)
    frozen = recorder.provider.read().joint_positions.copy()
    intended = np.array([1, 0, 0, -1], dtype=np.float32)
    recorder.step(intended, "lift")
    recorder.step(intended, "lift")
    # The release command went to the simulator, but the teacher still squeezed.
    assert recorder.environment.data.actuator("gripper").ctrl[0] == 1.0
    np.testing.assert_array_equal(recorder.actions, [intended, intended])
    np.testing.assert_allclose(recorder.observations[1][12:18], frozen)
    actual_xyz, _ = forward_kinematics(recorder.environment.model, frozen)
    np.testing.assert_allclose(recorder.targets[1], actual_xyz)
    assert len(recorder.backend.events) == 1
    processed = np.stack(recorder.observations)
    if not learned_observations:
        processed = ActionObservationPreprocessor().transform_episode(processed)
    tokens, _, _, labels, _ = build_action_windows(
        torch.from_numpy(processed),
        torch.from_numpy(np.stack(recorder.targets)),
        torch.from_numpy(np.stack(recorder.actions)), history_length=4, stride=2,
    )
    torch.testing.assert_close(tokens[0, 1, 37:41], torch.from_numpy(intended))
    assert labels[0, 1, 3] == -1


@pytest.mark.parametrize("observation", [np.zeros(41), np.full(37, np.nan)])
def test_invalid_learned_observation_fails_before_command_or_physics(recorder, observation):
    recorder.observation_builder = lambda _: observation
    before = recorder.environment.data.ctrl.copy()
    with pytest.raises(ValueError, match="37 finite values"):
        recorder.step(np.array([1, 0, 0, -1]), "approach")
    np.testing.assert_array_equal(recorder.environment.data.ctrl, before)
    assert recorder.environment.data.time == 0 and not recorder.actions


def test_partial_reset_preserves_physics_and_history(recorder):
    recorder.step(np.array([0, 0, 0, 1]), "approach")
    before = recorder.environment.data.qpos.copy()
    time = recorder.environment.data.time
    target = recorder.controller.prev_accepted_target.copy()
    recorder.recovery_requested = True
    recorder.waypoint_reached_time = time
    recorder._pickup_grasp_seen = True
    recorder.grasp_seen = True
    recorder._lost_grasp_steps = 2
    recorder.begin_recovery()
    np.testing.assert_array_equal(recorder.environment.data.qpos, before)
    np.testing.assert_array_equal(recorder.controller.prev_accepted_target, target)
    assert recorder.environment.data.time == time
    assert len(recorder.actions) == len(recorder.observations) == 1
    assert recorder.grasp_seen and not recorder._pickup_grasp_seen
    assert recorder.waypoint_reached_time is None
    assert recorder._lost_grasp_steps == 0
    assert recorder.recovery_count == 1


def test_recovery_recomputes_waypoint_above_current_orange(recorder, monkeypatch):
    targets = []

    def capture_target(subject, target, gripper, stage):
        targets.append(target.copy())
        subject.recovery_requested = True
        return False

    monkeypatch.setattr(demonstrations, "move_to", capture_target)
    first_orange = recorder.scene()["orange"].copy()
    pickup_from_current_pose(recorder)
    recorder.begin_recovery()
    recorder.environment.data.joint("orange_cube_joint").qpos[:2] += [.01, -.02]
    mujoco.mj_forward(recorder.environment.model, recorder.environment.data)
    second_orange = recorder.scene()["orange"].copy()
    pickup_from_current_pose(recorder)
    for actual, cube in zip(targets, (first_orange, second_orange), strict=True):
        np.testing.assert_allclose(actual, cube + [0, 0, recorder.config.teacher.waypoint_height])
    assert not np.array_equal(targets[0], targets[1])


def test_measured_loss_requires_two_ticks_and_placement_is_exempt(recorder):
    recorder._observe_progress({"held": True, "time": 0.0}, "lift")
    recorder._observe_progress({"held": False, "time": .05}, "transport")
    assert not recorder.recovery_requested
    recorder._observe_progress({"held": False, "time": .1}, "transport")
    assert recorder.recovery_requested
    recorder.begin_recovery()
    recorder._observe_progress({"held": True, "time": .15}, "lower")
    recorder.waypoint_reached_time = 0.0
    recorder._observe_progress({"held": False, "time": 10.0}, "release")
    recorder._observe_progress({"held": False, "time": 10.05}, "retreat")
    assert not recorder.recovery_requested
    assert recorder.waypoint_retry_count == 0


def test_waypoint_deadline_retries_only_after_five_simulated_seconds(recorder):
    recorder.waypoint_reached_time = 1.0
    recorder._observe_progress({"held": False, "time": 6.0}, "close")
    assert not recorder.recovery_requested
    recorder._observe_progress({"held": False, "time": 6.05}, "close")
    assert recorder.recovery_requested and recorder.waypoint_retry_count == 1
    recorder._observe_progress({"held": False, "time": 6.1}, "close")
    assert recorder.waypoint_retry_count == 1


def test_episode_budget_bounds_repeated_recovery(recorder):
    recorder.maximum_steps = 1
    recorder.step(np.array([0, 0, 0, 1]), "recovery_pause")
    recorder.begin_recovery()
    with pytest.raises(DemonstrationFailure, match="timed out"):
        recorder.step(np.array([0, 0, 0, 1]), "recovery_pause")
    assert len(recorder.actions) == 1


def test_ik_failure_propagates_with_stage_and_step_without_recording(recorder, monkeypatch):
    error = IKConvergenceError({"attempts": [{"position_error": .01, "tool_axis_error": 0,
                                              "tool_yaw_error": 0}]})

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(recorder.controller, "compute", fail)
    with pytest.raises(IKConvergenceError) as caught:
        recorder.step(np.array([0, 0, 0, 1]), "descend")
    assert caught.value is error
    assert error.diagnostics["stage"] == "descend" and error.diagnostics["episode_step"] == 1
    assert not recorder.actions and recorder.environment.data.time == 0


@pytest.fixture(scope="module")
def successful_episode():
    project = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    project.environment.action_data.drop_disturbance.probability = 0.0
    return collect_action_episode(project=project, environment=CubeStackEnvironment(), seed=0)


def test_real_complete_episode_has_stack_and_new_training_layout(successful_episode):
    episode = successful_episode
    episode.validate()
    assert 1 < len(episode.actions) <= 900
    assert episode.observations.shape == (len(episode.actions), 41)
    assert episode.prev_accepted_targets.shape == (len(episode.actions), 3)
    assert episode.actions.shape[1] == 4
    assert episode.stable_time >= 1.0 - 1e-12
    assert episode.grasp_seen and episode.success
    assert {"approach", "descend", "close", "lift", "transport", "lower", "release", "retreat"} <= episode.stage_steps.keys()
    json.dumps(episode.metadata())
    assert episode.metadata()["observation_source"] == "simulation_ground_truth"


@pytest.mark.parametrize("source,width,valid", [
    ("simulation_ground_truth", 41, True), ("vision_model", 37, True),
    ("simulation_ground_truth", 37, False), ("vision_model", 41, False),
    ("unknown", 37, False),
])
def test_episode_source_controls_observation_layout(source, width, valid):
    episode = ActionEpisode(
        observations=torch.zeros(1, width), prev_accepted_targets=torch.zeros(1, 3),
        actions=torch.zeros(1, 4), stable_time=1., required_stable_time=1.,
        stage_steps={"settle_stack": 1}, grasp_seen=True, observation_source=source,
    )
    if valid:
        episode.validate()
        assert episode.metadata()["observation_source"] == source
    else:
        with pytest.raises(ValueError):
            episode.validate()


def test_collect_resets_vision_after_environment_reset_and_marks_episode(project, monkeypatch):
    environment = CubeStackEnvironment()
    environment.data.time = 12.0
    events = []

    class Observer:
        def reset(self):
            events.append(("reset", float(environment.data.time)))

        def observe(self, robot):
            events.append(("observe", robot.timestamp))
            return np.arange(37, dtype=np.float32)

    def one_step_success(recorder):
        recorder.step(np.array([0, 0, 0, 1]), "settle_stack")
        recorder.grasp_seen = recorder.finished = True
        recorder.scorer.stable_since = environment.data.time - recorder.config.success.required_stable_time
        return True

    monkeypatch.setattr(demonstrations, "pickup_from_current_pose", one_step_success)
    episode = collect_action_episode(project=project, environment=environment, seed=0,
                                     vision_observer=Observer())
    assert events == [("reset", 0.), ("observe", 0.)]
    assert episode.observation_source == "vision_model"
    assert episode.metadata()["observation_source"] == "vision_model"
    torch.testing.assert_close(episode.observations, torch.arange(37).float()[None])
    episode.validate()


@pytest.mark.parametrize("field,value", [
    ("grasp_seen", False), ("success", False), ("stable_time", 0.5),
    ("stable_time", float("nan")), ("stage_steps", {"approach": 1}),
    ("observations", torch.zeros(1, 49)), ("actions", torch.ones(1, 4) * 2),
])
def test_validation_rejects_malformed_or_unsuccessful_episode(successful_episode, field, value):
    with pytest.raises(ValueError):
        replace(successful_episode, **{field: value}).validate()


def test_real_override_drop_recovers_and_finishes_without_reset(project, monkeypatch):
    project.environment.action_data.drop_disturbance.probability = 1.0
    original = DropDisturbanceBackend.begin_step

    def one_drop(self, robot_state, orange_held):
        # Force exactly one actuator disturbance after a measured lift. The
        # teacher cannot see this hook or the override's private phase timer.
        original(self, robot_state, orange_held and not self.events and robot_state.gripper_position[2] > .08)

    monkeypatch.setattr(DropDisturbanceBackend, "begin_step", one_drop)
    environment = CubeStackEnvironment()
    reset_calls = []
    original_reset = environment.reset

    def reset_once():
        reset_calls.append(True)
        original_reset()

    monkeypatch.setattr(environment, "reset", reset_once)

    class Observer:
        reset_count = 0

        def __init__(self):
            self.times = []

        def reset(self):
            self.reset_count += 1
            self.times.clear()

        def observe(self, robot):
            self.times.append(robot.timestamp)
            observation = np.zeros(37, dtype=np.float32)
            observation[:21] = np.concatenate((robot.joint_positions, robot.joint_velocities,
                                               robot.commanded_joint_targets, robot.gripper_position))
            # Wrong learned poses must not steer the privileged scripted teacher.
            observation[21:24], observation[29:32] = [1, 2, 3], [4, 5, 6]
            observation[24:26], observation[32:34] = [.6, .8], [-.8, .6]
            return observation

    observer = Observer()
    episode = collect_action_episode(project=project, environment=environment, seed=1,
                                     vision_observer=observer)
    assert len(reset_calls) == 1
    assert observer.reset_count == 1
    assert episode.observation_source == "vision_model"
    assert episode.observations.shape == (len(episode.actions), 37)
    np.testing.assert_allclose(observer.times, np.arange(len(episode.actions)) / project.loop.control_hz,
                               atol=1e-10)
    torch.testing.assert_close(episode.observations[:, 21:24],
                               torch.tensor([1., 2., 3.]).expand(len(episode.actions), -1))
    assert episode.recovery_count == 1 and episode.waypoint_retry_count == 0
    assert len(episode.disturbance_events) == 1
    assert episode.stage_steps["recovery_pause"] == 5
    assert episode.stage_steps["recovery_settle"] >= 5
    assert episode.stage_steps["settle_waypoint"] == 20  # Original approach and recovery.
    assert environment.data.time == pytest.approx(len(episode.actions) / project.loop.control_hz)
    event = episode.disturbance_events[0]
    release_index = event["start_step"] + event["stop_steps"]
    assert torch.all(episode.actions[release_index:release_index + 2, 3] == -1)
    # Observation following an override records the actual open joint target.
    assert episode.observations[release_index + 1, 17] == 1.0
    episode.validate()
