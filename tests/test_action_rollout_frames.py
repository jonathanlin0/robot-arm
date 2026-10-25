"""Frame observers see completed rollout states without changing evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

import action_evaluation
from action_evaluation import ActionRolloutFrame, ActionRolloutMetrics, evaluate_action_rollouts
from kinematics import IKConvergenceError
from schemas.config import load_config
from test_action_evaluation import FakeRenderer, ImagePoseModel, ObservingActionModel


@pytest.fixture
def project():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    config.environment.episode_seconds = 0.1
    config.training.action.rollout_validation.episodes = 2
    return config


def test_timeout_frames_observe_reset_and_completed_ticks_without_changing_inference(project):
    events = []
    states = []
    renders = []

    def run(observe):
        action, vision = ObservingActionModel(), ImagePoseModel()
        created = []

        def renderer_factory(*args, **kwargs):
            renderer = FakeRenderer(*args, **kwargs)
            created.append(renderer)
            return renderer

        def on_frame(environment, frame):
            events.append((frame, float(environment.data.time), len(action.histories)))
            states.append(environment.data.qpos.copy())
            assert torch.is_inference_mode_enabled()

        metrics = evaluate_action_rollouts(
            project=project, action_policy=action, vision_model=vision,
            device=torch.device("cpu"), renderer_factory=renderer_factory,
            on_frame=on_frame if observe else None,
        )
        renders.append(created)
        return metrics, action.histories, vision.inputs

    baseline = run(False)
    observed = run(True)
    assert observed[0] == baseline[0] == ActionRolloutMetrics(episodes=2, timeouts=2)
    assert [event[0] for event in events] == [
        ActionRolloutFrame(0, 0, "running"),
        ActionRolloutFrame(0, 1, "running"),
        ActionRolloutFrame(0, 2, "timeout"),
        ActionRolloutFrame(1, 0, "running"),
        ActionRolloutFrame(1, 1, "running"),
        ActionRolloutFrame(1, 2, "timeout"),
    ]
    assert [event[1] for event in events] == pytest.approx([0, 0.05, 0.1, 0, 0.05, 0.1])
    assert [event[2] for event in events] == [0, 1, 2, 2, 3, 4]
    for baseline_values, observed_values in zip(baseline[1:], observed[1:], strict=True):
        assert len(baseline_values) == len(observed_values) == 4
        for baseline_inputs, observed_inputs in zip(baseline_values, observed_values, strict=True):
            for baseline_tensor, observed_tensor in zip(baseline_inputs, observed_inputs, strict=True):
                torch.testing.assert_close(observed_tensor, baseline_tensor, rtol=0, atol=0)
    for baseline_renderer, observed_renderer in zip(*renders, strict=True):
        assert baseline_renderer.closed and observed_renderer.closed
        for baseline_snapshot, observed_snapshot in zip(
            baseline_renderer.snapshots, observed_renderer.snapshots, strict=True,
        ):
            assert baseline_snapshot[:2] == observed_snapshot[:2]
            np.testing.assert_array_equal(baseline_snapshot[2], observed_snapshot[2])
    assert not np.array_equal(states[0], states[3])


@pytest.mark.parametrize("success_step", [1, 2])
def test_success_is_reported_once_including_when_it_coincides_with_time_limit(project, monkeypatch,
                                                                           success_step):
    project.training.action.rollout_validation.episodes = 1
    outcomes = iter([False] * (success_step - 1) + [True])
    monkeypatch.setattr(action_evaluation.StackSuccessTracker, "update", lambda self, data: next(outcomes))
    events = []
    action, vision = ObservingActionModel(), ImagePoseModel()
    metrics = evaluate_action_rollouts(
        project=project, action_policy=action, vision_model=vision,
        device=torch.device("cpu"), renderer_factory=FakeRenderer,
        on_frame=lambda environment, frame: events.append(frame),
        on_episode=lambda metrics: events.append(metrics),
    )
    assert events == [
        *[ActionRolloutFrame(0, step, "running") for step in range(success_step)],
        ActionRolloutFrame(0, success_step, "success"),
        ActionRolloutMetrics(episodes=1, successes=1),
    ]
    assert metrics == events[-1]
    assert len(action.histories) == len(vision.inputs) == success_step


@pytest.mark.parametrize("failure_stage", ["reset", "first_tick", "second_tick"])
def test_ik_frame_reports_only_completed_ticks_and_next_episode_resets(project, monkeypatch,
                                                                     failure_stage):
    events = []
    action, vision = ObservingActionModel(), ImagePoseModel()
    calls = 0
    method = "reset_rollout_environment" if failure_stage == "reset" else "compute"
    owner = action_evaluation if failure_stage == "reset" else action_evaluation.CartesianController
    original = getattr(owner, method)
    fail_call = 2 if failure_stage == "second_tick" else 1

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise IKConvergenceError({"attempts": [{
                "position_error": 1.0, "tool_axis_error": 1.0, "tool_yaw_error": 1.0,
            }]})
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, fail_once)
    metrics = evaluate_action_rollouts(
        project=project, action_policy=action, vision_model=vision,
        device=torch.device("cpu"), renderer_factory=FakeRenderer,
        on_frame=lambda environment, frame: events.append((frame, float(environment.data.time))),
    )
    completed = int(failure_stage == "second_tick")
    expected = [] if failure_stage == "reset" else [ActionRolloutFrame(0, 0, "running")]
    if completed:
        expected.append(ActionRolloutFrame(0, 1, "running"))
    expected.extend([
        ActionRolloutFrame(0, completed, "ik_failure"),
        ActionRolloutFrame(1, 0, "running"),
        ActionRolloutFrame(1, 1, "running"),
        ActionRolloutFrame(1, 2, "timeout"),
    ])
    assert [event[0] for event in events] == expected
    assert events[-4][1] == pytest.approx(completed * 0.05)
    assert [event[1] for event in events[-3:]] == pytest.approx([0, 0.05, 0.1])
    assert metrics == ActionRolloutMetrics(episodes=2, ik_failures=1, timeouts=1)
    assert len(action.histories) == len(vision.inputs) == 2 + (
        0 if failure_stage == "reset" else fail_call
    )
    assert action.histories[-2][0].shape[1] == 1


@pytest.mark.parametrize("failure_step", [0, 1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, IKConvergenceError])
def test_observer_errors_propagate_and_restore_models_and_renderers(project, failure_step, error_type):
    renderers = []
    progress = []
    action, vision = ObservingActionModel(), ImagePoseModel()
    action.eval()
    failure = RuntimeError("observer failed") if error_type is RuntimeError else IKConvergenceError({
        "attempts": [{"position_error": 1.0, "tool_axis_error": 1.0, "tool_yaw_error": 1.0}],
    })

    def renderer_factory(*args, **kwargs):
        renderer = FakeRenderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    def on_frame(environment, frame):
        if frame.step == failure_step:
            raise failure

    with pytest.raises(error_type) as caught:
        evaluate_action_rollouts(
            project=project, action_policy=action, vision_model=vision,
            device=torch.device("cpu"), renderer_factory=renderer_factory,
            on_frame=on_frame, on_episode=progress.append,
        )
    assert caught.value is failure
    assert not progress
    assert len(renderers) == 2 and all(renderer.closed for renderer in renderers)
    assert not action.training and vision.training and not vision.mixed_mode.training
    assert len(action.histories) == len(vision.inputs) == failure_step
