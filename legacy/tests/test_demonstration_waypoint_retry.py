"""A stalled pickup retries from a fresh waypoint without resetting its episode."""

import json
from functools import partial
from pathlib import Path
import tempfile
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

import scripts.generate_pickup_demonstrations as generator


class TimedEnvironment:
    """Only the measured state and valid Gym transition needed by the teacher."""

    def __init__(self):
        self.previous_state = {
            "time": 0.0,
            "orange_currently_held": False,
            "orange_position": np.array([0.3, 0.0, 0.02]),
            "orange_velocity": np.zeros(6),
            "gripper_position": np.array([0.3, 0.0, 0.10]),
        }
        self.action_adapter = SimpleNamespace(
            config=generator.CartesianActionConfig(),
            current_target_gripper_position=self.previous_state["gripper_position"].copy(),
        )
        self.reward_config = SimpleNamespace(approach_orange_height_offset=0.08)
        self.episode_step_count = 0
        self.next_time = 0.05
        self.success = False
        self.truncated = False
        self.failure = False
        self.closed = False

    def observation(self):
        observation = np.zeros(generator.PRIVILEGED_OBSERVATION_SIZE, dtype=np.float32)
        observation[0] = self.previous_state["time"]
        return observation

    def info(self):
        return {
            "target_gripper_position": self.action_adapter.current_target_gripper_position.copy(),
            "orange_currently_held": self.previous_state["orange_currently_held"],
            "orange_grasp_hold_time": 0.0,
            "stack_stable_time": (generator.StackSuccessConfig().required_stable_time
                                  if self.success else 0.0),
            "is_failure": self.failure,
            "is_success": self.success,
            "ik_position_converged": True,
            "ik_tool_yaw_converged": True,
        }

    def step(self, action):
        self.previous_state["time"] = self.next_time
        self.next_time += 0.05
        self.episode_step_count += 1
        return self.observation(), 0.0, self.success, self.truncated, self.info()

    def close(self):
        self.closed = True


def recorder_for(environment):
    return generator.EpisodeRecorder(environment.observation(), environment.info())


def record_at(environment, recorder, time, *, held=False, stage="close"):
    environment.next_time = time
    environment.previous_state["orange_currently_held"] = held
    return recorder.step(environment, np.array([0.0, 0.0, 0.0, -1.0]), stage)


def test_waypoint_retry_default_is_five_seconds():
    assert generator.WAYPOINT_GRASP_RETRY_SECONDS == 5.0


@pytest.mark.parametrize("time", [5.1, 20.0])
def test_never_retries_before_waypoint_was_reached(time):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    record_at(env, recorder, time)
    assert recorder.waypoint_reached_time is None
    assert not recorder.recovery_requested
    assert recorder.waypoint_retry_count == 0


@pytest.mark.parametrize("elapsed, expected", [
    (0.1, False), (4.95, False), (5.0, False),
    (5.000000000000001, False), (5.05, True),
])
def test_retry_requires_more_than_five_simulated_seconds_since_waypoint(elapsed, expected):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(2.0)
    record_at(env, recorder, 2.0 + elapsed)
    assert recorder.recovery_requested is expected
    assert recorder.waypoint_retry_count == int(expected)


def test_held_cube_prevents_retry_but_does_not_reset_waypoint_clock():
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    record_at(env, recorder, 8.0, held=True)
    assert not recorder.recovery_requested
    assert recorder.waypoint_retry_count == 0
    assert recorder.waypoint_reached_time == 0.0
    record_at(env, recorder, 8.05, held=False)
    assert recorder.recovery_requested
    assert recorder.waypoint_retry_count == 1


def test_pending_recovery_is_counted_once_and_rearming_preserves_history():
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    record_at(env, recorder, 5.05)
    record_at(env, recorder, 5.10)
    assert recorder.waypoint_retry_count == 1
    history = np.asarray(recorder.observations).copy()
    recorder.begin_recovery()
    assert recorder.waypoint_reached_time is None
    assert not recorder.recovery_requested
    assert recorder.recovery_count == 1
    assert recorder.waypoint_retry_count == 1
    np.testing.assert_array_equal(recorder.observations, history)
    record_at(env, recorder, 20.0, stage="recovery_settle")
    assert not recorder.recovery_requested
    recorder.mark_waypoint_reached(20.0)
    record_at(env, recorder, 24.95)
    assert not recorder.recovery_requested
    record_at(env, recorder, 25.05)
    assert recorder.recovery_requested
    assert recorder.waypoint_retry_count == 2
    assert len(recorder.observations) == len(recorder.targets) == len(recorder.actions) + 1


def test_an_existing_drop_recovery_does_not_increment_waypoint_retry_counter():
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    recorder.recovery_requested = True
    record_at(env, recorder, 6.0)
    assert recorder.recovery_requested
    assert recorder.waypoint_retry_count == 0


@pytest.mark.parametrize("stage", ["release", "retreat", "settle_stack"])
def test_intentional_placement_disarms_timer_even_before_contacts_release(stage):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    record_at(env, recorder, 8.0, held=True, stage=stage)
    assert recorder.waypoint_reached_time is None
    record_at(env, recorder, 8.05, held=False, stage=stage)
    assert not recorder.recovery_requested
    assert recorder.waypoint_retry_count == 0


@pytest.mark.parametrize("outcome", ["success", "truncated", "failure"])
def test_terminal_or_invalid_transition_does_not_record_a_waypoint_retry(outcome):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    setattr(env, outcome, True)
    if outcome == "success":
        assert record_at(env, recorder, 6.0)
    else:
        with pytest.raises(generator.DemonstrationFailure):
            record_at(env, recorder, 6.0)
    assert recorder.waypoint_retry_count == 0


PICKUP_STAGES = ["approach", "settle_waypoint", "descend", "settle", "close", "lift", "settle_lift"]


@pytest.mark.parametrize("retry_stage", PICKUP_STAGES)
def test_every_pickup_phase_yields_immediately_when_recovery_is_requested(monkeypatch, retry_stage):
    env = TimedEnvironment()
    env.previous_state["time"] = 2.0
    env.previous_state["orange_currently_held"] = True
    recorder = recorder_for(env)
    stages = []

    def run_phase(environment, active_recorder, *arguments):
        stage = arguments[-1]
        stages.append(stage)
        if stage == retry_stage:
            active_recorder.recovery_requested = True
        return False

    monkeypatch.setattr(generator, "move_to", run_phase)
    monkeypatch.setattr(generator, "hold_position", run_phase)
    assert not generator.pickup_from_current_pose(env, recorder)
    assert stages == PICKUP_STAGES[:PICKUP_STAGES.index(retry_stage) + 1]
    assert recorder.recovery_requested
    assert recorder.waypoint_reached_time == (None if retry_stage == "approach" else 2.0)


def test_retry_waits_then_uses_current_cube_for_a_new_waypoint(monkeypatch):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    record_at(env, recorder, 5.05)
    history = np.asarray(recorder.observations).copy()
    assert not generator.recover_for_pickup(env, recorder)
    assert recorder.waypoint_reached_time is None
    assert recorder.recovery_count == recorder.waypoint_retry_count == 1
    assert recorder.stages["recovery_pause"] == generator.RECOVERY_PAUSE_STEPS
    assert recorder.stages["recovery_settle"] == generator.RECOVERY_SETTLE_STEPS
    np.testing.assert_array_equal(recorder.observations[:len(history)], history)
    moved_cube = np.array([0.32, -0.07, 0.02])
    env.previous_state["orange_position"] = moved_cube.copy()
    approach_targets = []
    new_waypoint_time = env.previous_state["time"]

    def move(environment, active_recorder, target, gripper, stage):
        assert stage == "approach"
        approach_targets.append(target.copy())
        return False

    def hold(environment, active_recorder, gripper, steps, stage):
        assert stage == "settle_waypoint"
        assert active_recorder.waypoint_reached_time == new_waypoint_time
        active_recorder.recovery_requested = True  # End the focused control-flow test here.
        return False

    monkeypatch.setattr(generator, "move_to", move)
    monkeypatch.setattr(generator, "hold_position", hold)
    assert not generator.pickup_from_current_pose(env, recorder)
    np.testing.assert_allclose(approach_targets, [moved_cube + [0.0, 0.0, 0.08]])
    assert recorder.waypoint_retry_count == 1
    assert env.episode_step_count == len(recorder.actions)


def test_retry_count_is_saved_and_printed_for_the_completed_episode(monkeypatch, capsys):
    env = TimedEnvironment()
    recorder = recorder_for(env)
    recorder.mark_waypoint_reached(0.0)
    record_at(env, recorder, 5.05)
    recorder.begin_recovery()
    env.success = True
    assert record_at(env, recorder, 5.10, stage="settle_stack")
    episode = recorder.episode()
    assert episode.waypoint_retry_count == 1
    monkeypatch.setattr(generator, "make_environment", lambda: env)
    monkeypatch.setattr(generator, "generation_settings", lambda environment: {})
    monkeypatch.setattr(generator, "collect_episode", lambda environment, seed: episode)
    monkeypatch.setattr(generator, "uuid4", lambda: UUID(int=1))
    with tempfile.TemporaryDirectory(prefix=".test-waypoint-retry-", dir=generator.REPOSITORY_ROOT) as directory:
        records = generator.generate_dataset(1, Path(directory))
        assert records[0]["waypoint_retry_count"] == 1
        saved = json.loads((Path(directory) / "manifests" / f"{UUID(int=1)}.json").read_text())
        assert saved["episode"]["waypoint_retry_count"] == 1
        assert len(list(Path(directory).rglob("*.pt"))) == 3
    assert "waypoint_retries=1" in capsys.readouterr().out
    assert env.closed


def test_physical_missed_descent_retries_and_finishes_without_resetting_history(monkeypatch):
    """Keep the first descent still long enough to require an actual timed retry."""
    monkeypatch.setattr(generator, "START_POSITION_HALF_RANGE", (0.0, 0.0, 0.0))
    monkeypatch.setattr(generator, "DropDisturbance", partial(generator.DropDisturbance, probability=0.0))
    original_move = generator.move_to
    stalled_once = False

    def move(environment, recorder, target, gripper, stage):
        nonlocal stalled_once
        if stage == "descend" and not stalled_once:
            stalled_once = True
            with monkeypatch.context() as patch:
                patch.setattr(generator, "action_toward", lambda *_: np.array(
                    [0.0, 0.0, 0.0, generator.OPEN_GRIPPER], dtype=np.float32))
                return original_move(environment, recorder, target, gripper, stage)
        return original_move(environment, recorder, target, gripper, stage)

    monkeypatch.setattr(generator, "move_to", move)
    env = generator.make_environment()
    original_reset = env.reset
    reset_seeds = []

    def reset(*, seed):
        reset_seeds.append(seed)
        return original_reset(seed=seed)

    monkeypatch.setattr(env, "reset", reset)
    try:
        episode = generator.collect_episode(env, seed=21)
        episode.validate()
        assert stalled_once
        assert reset_seeds == [21]
        assert episode.waypoint_retry_count == episode.recovery_count == 1
        assert not episode.disturbance_events
        assert env.simulation.is_success()
        assert episode.stage_steps["recovery_pause"] == generator.RECOVERY_PAUSE_STEPS
        assert episode.stage_steps["recovery_settle"] >= generator.RECOVERY_SETTLE_STEPS
        assert len(episode.actions) == env.episode_step_count <= generator.MAXIMUM_EPISODE_STEPS
        assert len(episode.observations) == len(episode.accepted_targets) == len(episode.actions) + 1
        assert env.previous_state["time"] == pytest.approx(len(episode.actions) * 0.05)
    finally:
        env.close()
