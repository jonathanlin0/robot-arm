"""Learned action inputs use camera predictions and simulation-time history."""

from pathlib import Path

import numpy as np
import pytest
import torch

from action_demonstrations import EpisodeRecorder, reset_demonstration_environment
from action_observation import VisionActionObserver
from environment import CubeStackEnvironment
from robot_state_provider import SimRobotStateProvider
from schemas.config import load_config


class PredictedPoses(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Deliberately unrelated to physical cube positions and orientations.
        self.prediction = torch.tensor([[1., 2., 3., 3., 4., 8., 9., 10., 1e-9, 0.]])
        self.inputs = []

    def forward(self, wrist, external, joints):
        assert not torch.is_grad_enabled() and not self.training
        self.inputs.append((wrist.clone(), external.clone(), joints.clone()))
        return self.prediction


class Renderer:
    def __init__(self, model, *, height, width):
        self.shape = (height, width, 3)
        self.camera = None
        self.times = []
        self.close_count = 0

    def update_scene(self, data, *, camera):
        self.camera = camera
        self.times.append(float(data.time))

    def render(self):
        return np.full(self.shape, 255 if self.camera == "wrist_cam" else 127, dtype=np.uint8)

    def close(self):
        self.close_count += 1


@pytest.fixture
def setup():
    environment, model, renderers = CubeStackEnvironment(), PredictedPoses(), []

    def factory(*args, **kwargs):
        renderer = Renderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    observer = VisionActionObserver(environment, model, torch.device("cpu"), .5, factory)
    return environment, model, renderers, observer


def test_camera_predictions_fill_policy_features_without_ground_truth_or_double_normalization(setup):
    environment, model, renderers, observer = setup
    robot = SimRobotStateProvider(environment).read()
    with observer:
        observation = observer.observe(robot)
        assert observation.shape == (37,) and observation.dtype == np.float32
        np.testing.assert_array_equal(observation[:21], np.concatenate((
            robot.joint_positions, robot.joint_velocities, robot.commanded_joint_targets,
            robot.gripper_position,
        )).astype(np.float32))
        np.testing.assert_array_equal(observation[21:24], [1, 2, 3])
        np.testing.assert_allclose(observation[24:26], [.6, .8])
        np.testing.assert_array_equal(observation[26:29], 0)
        np.testing.assert_array_equal(observation[29:32], [8, 9, 10])
        np.testing.assert_allclose(observation[32:34], [.001, 0])
        np.testing.assert_array_equal(observation[34:37], 0)
        assert not np.allclose(observation[21:24], environment.data.body("orange_cube").xpos)
        wrist, external, joints = model.inputs[0]
        assert wrist.shape == (1, 3, 128, 128) and external.shape == (1, 3, 300, 300)
        torch.testing.assert_close(wrist, torch.ones_like(wrist))
        torch.testing.assert_close(external, torch.full_like(external, 127 / 255))
        torch.testing.assert_close(joints, torch.from_numpy(robot.joint_positions).float()[None])
        assert [renderer.camera for renderer in renderers] == ["wrist_cam", "external"]
        assert environment.data.time == 0
    assert [renderer.close_count for renderer in renderers] == [1, 1]


def test_velocity_uses_learned_positions_simulation_time_and_reset_reuses_renderers(setup):
    environment, model, renderers, observer = setup
    telemetry = SimRobotStateProvider(environment)
    with observer:
        environment.data.time = 7.0
        observer.observe(telemetry.read())
        model.prediction[0, :3] += torch.tensor([.5, -.25, 1.])
        model.prediction[0, 5:8] += torch.tensor([-.5, 1., .25])
        environment.data.time = 7.25
        moved = observer.observe(telemetry.read())
        np.testing.assert_array_equal(moved[26:29], [2, -1, 4])
        np.testing.assert_array_equal(moved[34:37], [-2, 4, 1])
        assert observer.provider.previous_estimates["orange"].timestamp == 7.25
        assert all(renderer.times == [7.0, 7.25] for renderer in renderers)
        observer.reset()
        environment.reset()
        reset = observer.observe(telemetry.read())
        np.testing.assert_array_equal(reset[26:29], 0)
        np.testing.assert_array_equal(reset[34:37], 0)
        assert observer.provider.previous_estimates["orange"].timestamp == 0
        assert len(renderers) == 2 and not any(renderer.close_count for renderer in renderers)


def test_recorder_captures_pre_action_vision_and_recovery_keeps_velocity_history(setup):
    environment, model, renderers, observer = setup
    project = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    project.environment.action_data.drop_disturbance.probability = 0
    reset_demonstration_environment(environment, project, np.random.default_rng(0))
    with observer:
        recorder = EpisodeRecorder(project, environment, 0, observation_builder=observer.observe)
        before = recorder.scene()
        recorder.step(np.array([.1, 0, 0, 1]), "approach")
        np.testing.assert_allclose(recorder.observations[0][:6], before["robot"].joint_positions)
        np.testing.assert_array_equal(recorder.observations[0][21:24], [1, 2, 3])
        assert all(renderer.times == [0.] for renderer in renderers)
        # Physical velocities cannot leak into the next learned observation.
        environment.data.joint("orange_cube_joint").qvel[:3] = [.1, .1, .1]
        model.prediction[0, :3] += torch.tensor([.01, .02, .03])
        previous = observer.provider.previous_estimates
        recorder.begin_recovery()
        assert observer.provider.previous_estimates is previous
        recorder.step(np.array([0, 0, 0, 1]), "recovery_pause")
        assert len(recorder.actions) == 2 and recorder.recovery_count == 1
        np.testing.assert_allclose(recorder.observations[1][26:29], [.2, .4, .6], atol=3e-6)
        assert all(renderer.times == pytest.approx([0., .05]) for renderer in renderers)


def test_second_renderer_allocation_failure_closes_first(setup):
    environment, model, renderers, _ = setup

    def factory(*args, **kwargs):
        if renderers:
            raise RuntimeError("second allocation failed")
        renderer = Renderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    observer = VisionActionObserver(environment, model, torch.device("cpu"), .5, factory)
    with pytest.raises(RuntimeError, match="second allocation"):
        with observer:
            pytest.fail("Construction should fail.")
    assert renderers[0].close_count == 1
    assert not observer.renderers


@pytest.mark.parametrize("failure_point", ["render", "inference", "close"])
def test_observation_and_cleanup_errors_close_both_renderers(setup, monkeypatch, failure_point):
    environment, model, renderers, observer = setup

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        with observer:
            if failure_point == "render":
                monkeypatch.setattr(renderers[1], "render", fail)
            elif failure_point == "inference":
                monkeypatch.setattr(model, "forward", fail)
            else:
                # ExitStack captures the bound callback at construction time.
                observer._resources.callback(fail)
            observer.observe(SimRobotStateProvider(environment).read())
    assert [renderer.close_count for renderer in renderers] == [1, 1]
    assert not observer.renderers and observer.provider.previous_estimates is None


def test_observer_requires_open_context_and_rejects_nested_entry(setup):
    environment, _, renderers, observer = setup
    robot = SimRobotStateProvider(environment).read()
    with pytest.raises(RuntimeError, match="Open VisionActionObserver"):
        observer.observe(robot)
    with observer:
        with pytest.raises(RuntimeError, match="already open"):
            observer.__enter__()
    with pytest.raises(RuntimeError, match="Open VisionActionObserver"):
        observer.observe(robot)
    assert len(renderers) == 2
