"""Headless scoring and short image-to-action validation-loop contracts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

import action_evaluation
from action_evaluation import (
    ActionRolloutMetrics, StackSuccessTracker, evaluate_action_rollouts,
    reset_rollout_environment,
)
from environment import CubeStackEnvironment
from kinematics import IKConvergenceError
from robot_constants import ROBOT_JOINT_NAMES, TOOL_FRAME_SITE_NAME
from schemas.config import load_config


@pytest.fixture
def project():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    config.environment.episode_seconds = 0.1
    config.training.action.rollout_validation.episodes = 2
    return config


def make_stack(project, orange_half_height=0.02):
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><worldbody>
          <geom name="table" type="box" pos="0 0 -.025" size="1 1 .025"/>
          <body name="orange_cube" pos="0 0 {0.0399 + orange_half_height}">
            <freejoint name="orange_cube_joint"/>
            <geom name="orange_cube_geom" type="box" size=".02 .02 {orange_half_height}"/>
          </body>
          <body name="blue_cube" pos="0 0 .01995">
            <freejoint name="blue_cube_joint"/>
            <geom name="blue_cube_geom" type="box" size=".02 .02 .02"/>
          </body>
          <body name="gripper" pos=".3 0 .06"><body name="finger">
            <geom name="finger_geom" type="sphere" size=".015"/>
          </body></body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tracker = StackSuccessTracker(model, project.training.action.rollout_validation.success)
    return model, data, tracker


@pytest.fixture
def stack(project):
    return make_stack(project)


def test_success_requires_a_full_stable_interval_and_restarts_after_motion(stack):
    model, data, tracker = stack
    assert tracker.conditions_met(data)
    assert not tracker.update(data)
    data.time = 0.99
    assert not tracker.update(data)
    data.time = 1.0
    assert tracker.update(data)
    data.joint("orange_cube_joint").qvel[0] = 0.02
    data.time = 1.05
    assert not tracker.update(data)
    data.joint("orange_cube_joint").qvel[:] = 0
    data.time = 1.1
    assert not tracker.update(data)
    data.time = 2.09
    assert not tracker.update(data)
    data.time = 2.1
    assert tracker.update(data)
    tracker.reset()
    assert not tracker.update(data)


@pytest.mark.parametrize("condition", ["offset", "height", "angular_speed", "unsupported", "gripper", "nonfinite"])
def test_invalid_or_unreleased_stacks_are_not_successful(stack, condition):
    model, data, tracker = stack
    if condition == "offset":
        data.joint("orange_cube_joint").qpos[0] = 0.015
    elif condition == "height":
        data.joint("orange_cube_joint").qpos[2] = 0.08
    elif condition == "angular_speed":
        data.joint("blue_cube_joint").qvel[3] = 0.06
    elif condition == "unsupported":
        for name in ("orange", "blue"):
            data.joint(f"{name}_cube_joint").qpos[2] += 0.1
    elif condition == "gripper":
        model.body_pos[model.body("gripper").id] = [0.03, 0, 0.0599]
    mujoco.mj_forward(model, data)
    if condition == "nonfinite":
        data.joint("orange_cube_joint").qvel[0] = np.nan
    assert not tracker.conditions_met(data)


def test_stack_height_comes_from_scene_geometry(project):
    _, data, tracker = make_stack(project, orange_half_height=0.03)
    assert tracker.expected_vertical_distance == pytest.approx(0.05)
    assert tracker.conditions_met(data)


def test_randomized_home_and_ground_resets_are_repeatable_separated_and_reset_physics(project):
    environment = CubeStackEnvironment(project.environment.scene_path)
    config = project.training.action.rollout_validation
    controller = project.cartesian_controller
    controller.open_gripper_target = 0.9

    def sequence(seed):
        rng = np.random.default_rng(seed)
        states = []
        for _ in range(3):
            environment.data.time = 17
            environment.data.qvel[:] = 1
            for name in ROBOT_JOINT_NAMES:
                environment.data.joint(name).qpos[0] += 0.1
            environment.data.ctrl[:] = 0.5
            reset_rollout_environment(environment, config, rng, controller_config=controller)
            states.append(environment.data.qpos.copy())
            assert environment.data.time == 0
            np.testing.assert_array_equal(environment.data.qvel, 0)
            joints = np.array([environment.data.joint(name).qpos[0] for name in ROBOT_JOINT_NAMES])
            targets = np.array([environment.data.actuator(name).ctrl[0] for name in ROBOT_JOINT_NAMES])
            np.testing.assert_array_equal(joints, targets)
            assert joints[-1] == controller.open_gripper_target
            gripper_position = environment.data.site(TOOL_FRAME_SITE_NAME).xpos
            assert np.all(abs(gripper_position - config.start_position)
                          <= np.asarray(config.start_position_half_range) + controller.position_tolerance)
            orange, blue = (environment.data.joint(f"{name}_cube_joint").qpos for name in ("orange", "blue"))
            assert np.linalg.norm(orange[:2] - blue[:2]) >= config.minimum_cube_separation
            for cube in (orange, blue):
                assert config.cube_x_range[0] <= cube[0] <= config.cube_x_range[1]
                assert config.cube_y_range[0] <= cube[1] <= config.cube_y_range[1]
                assert cube[2] == pytest.approx(0.02)
                np.testing.assert_array_equal(cube[3:], [1, 0, 0, 0])
        return np.array(states)

    first = sequence(config.seed)
    np.testing.assert_array_equal(first, sequence(config.seed))
    assert not np.array_equal(first, sequence(config.seed + 1))
    assert not np.array_equal(first[0], first[1])


def test_validation_and_demonstration_defaults_produce_the_same_seeded_start(project):
    from action_demonstrations import reset_demonstration_environment

    demonstration = CubeStackEnvironment(project.environment.scene_path)
    validation = CubeStackEnvironment(project.environment.scene_path)
    seed = project.training.action.rollout_validation.seed
    reset_demonstration_environment(demonstration, project, np.random.default_rng(seed))
    reset_rollout_environment(validation, project.training.action.rollout_validation,
                              np.random.default_rng(seed), controller_config=project.cartesian_controller)
    np.testing.assert_array_equal(validation.data.qpos, demonstration.data.qpos)
    np.testing.assert_array_equal(validation.data.qvel, demonstration.data.qvel)
    np.testing.assert_array_equal(validation.data.ctrl, demonstration.data.ctrl)
    assert validation.data.time == demonstration.data.time == 0.0


def test_failed_home_ik_is_reported_before_installing_joint_targets(project, monkeypatch):
    environment = CubeStackEnvironment(project.environment.scene_path)
    initial = environment.data.qpos.copy()

    def failed_ik(model, initial_positions, target_position, config):
        np.testing.assert_array_equal(initial_positions[:-1], 0)
        assert initial_positions[-1] == config.open_gripper_target
        assert config is project.cartesian_controller
        return np.ones(len(ROBOT_JOINT_NAMES)), {
            "converged": False, "position_error": 1.0, "tool_axis_error": 1.0, "tool_yaw_error": 1.0,
        }

    monkeypatch.setattr(action_evaluation, "solve_ik", failed_ik)
    with pytest.raises(IKConvergenceError):
        reset_rollout_environment(environment, project.training.action.rollout_validation,
                                  np.random.default_rng(43), controller_config=project.cartesian_controller)
    np.testing.assert_array_equal(environment.data.qpos, initial)
    assert environment.data.time == 0.0


class ImagePoseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.mixed_mode = torch.nn.Dropout()
        self.mixed_mode.eval()

    def forward(self, wrist, external, joint_positions):
        assert not self.training and not self.mixed_mode.training
        assert torch.is_inference_mode_enabled()
        self.inputs.append((wrist.clone(), external.clone(), joint_positions.clone()))
        return torch.tensor([[0.11, 0.12, 0.13, 1, 0, 0.21, -0.22, 0.23, 1, 0]],
                            dtype=torch.float32, device=wrist.device)


class ObservingActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.histories = []

    def predict_action(self, tokens, valid, episode_start, deterministic):
        assert deterministic and not self.training
        assert torch.is_inference_mode_enabled()
        self.histories.append((tokens.clone(), valid.clone(), episode_start.clone()))
        return torch.tensor([[-0.2, 0, -0.2, 0]], dtype=torch.float32, device=tokens.device)


class FakeRenderer:
    def __init__(self, model, *, height, width):
        self.height, self.width = height, width
        self.snapshots = []
        self.closed = False

    def update_scene(self, data, *, camera):
        self.snapshots.append((float(data.time), camera, data.qpos.copy()))

    def render(self):
        return np.full((self.height, self.width, 3), 128, dtype=np.uint8)

    def close(self):
        self.closed = True


def test_image_based_rollouts_reset_histories_advance_physics_and_never_wait(project, monkeypatch):
    renderers = []
    action, vision = ObservingActionModel(), ImagePoseModel()
    progress = []
    original_runner = action_evaluation.LoopRunner

    def runner_factory(**kwargs):
        # The runtime pacing hook is deliberately an immediate no-op.
        assert kwargs["wait"](1000) is None
        return original_runner(**kwargs)

    def renderer_factory(*args, **kwargs):
        renderer = FakeRenderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    monkeypatch.setattr(action_evaluation, "LoopRunner", runner_factory)
    metrics = evaluate_action_rollouts(project=project, action_policy=action, vision_model=vision,
                                      device=torch.device("cpu"), on_episode=progress.append,
                                      renderer_factory=renderer_factory)
    assert metrics == ActionRolloutMetrics(episodes=2, timeouts=2)
    assert [value.episodes for value in progress] == [1, 2]
    assert action.training and vision.training and not vision.mixed_mode.training
    assert all(renderer.closed for renderer in renderers)
    assert [(renderer.height, renderer.width) for renderer in renderers] == [(128, 128), (300, 300)]
    assert [snapshot[0] for snapshot in renderers[0].snapshots] == pytest.approx([0, 0.05, 0, 0.05])
    assert not np.array_equal(renderers[0].snapshots[0][2], renderers[0].snapshots[2][2])
    assert len(action.histories) == len(vision.inputs) == 4
    for index, (tokens, valid, starts) in enumerate(action.histories):
        length = index % 2 + 1
        assert tokens.shape == (1, length, 44)
        assert valid.tolist() == [[True] * length]
        assert starts.tolist() == [[True] + [False] * (length - 1)]
        np.testing.assert_allclose(tokens[0, -1, 21:24], [0.11, 0.12, 0.13])
        np.testing.assert_allclose(tokens[0, -1, 29:32], [0.21, -0.22, 0.23])
        if length == 1:
            np.testing.assert_array_equal(tokens[0, 0, 26:29], 0)
            np.testing.assert_array_equal(tokens[0, 0, 37:41], 0)
    for wrist, external, joints in vision.inputs:
        assert wrist.shape == (1, 3, 128, 128)
        assert external.shape == (1, 3, 300, 300)
        assert joints.shape == (1, 6)
        assert wrist[0, 0, 0, 0].item() == pytest.approx(128 / 255)


def test_success_ik_failure_and_timeout_all_count_in_denominator(project, monkeypatch):
    project.training.action.rollout_validation.episodes = 3
    original_compute = action_evaluation.CartesianController.compute
    calls = 0

    def compute(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IKConvergenceError({"attempts": [{"position_error": 1, "tool_axis_error": 1,
                                                     "tool_yaw_error": 1}]})
        return original_compute(self, *args, **kwargs)

    outcomes = iter([True, False, False])
    monkeypatch.setattr(action_evaluation.CartesianController, "compute", compute)
    monkeypatch.setattr(StackSuccessTracker, "update", lambda self, data: next(outcomes))
    metrics = evaluate_action_rollouts(project=project, action_policy=ObservingActionModel(),
                                      vision_model=ImagePoseModel(), device=torch.device("cpu"),
                                      renderer_factory=FakeRenderer)
    assert metrics == ActionRolloutMetrics(episodes=3, successes=1, ik_failures=1, timeouts=1)
    assert metrics.success_rate_percent == pytest.approx(100 / 3)


def test_reset_ik_failure_counts_one_attempt_without_rendering_or_stepping_it(project, monkeypatch):
    action, vision = ObservingActionModel(), ImagePoseModel()
    original_reset = action_evaluation.reset_rollout_environment
    reset_calls = []
    progress = []
    renderers = []

    def reset(environment, config, rng, *, controller_config):
        reset_calls.append(controller_config)
        if len(reset_calls) == 1:
            raise IKConvergenceError({"attempts": [{
                "position_error": 1.0, "tool_axis_error": 1.0, "tool_yaw_error": 1.0,
            }]})
        return original_reset(environment, config, rng, controller_config=controller_config)

    def renderer_factory(*args, **kwargs):
        renderer = FakeRenderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    def record_progress(metrics):
        progress.append(metrics)
        if metrics.episodes == 1:
            assert not action.histories and not vision.inputs
            assert all(not renderer.snapshots for renderer in renderers)

    monkeypatch.setattr(action_evaluation, "reset_rollout_environment", reset)
    metrics = evaluate_action_rollouts(project=project, action_policy=action, vision_model=vision,
                                      device=torch.device("cpu"), on_episode=record_progress,
                                      renderer_factory=renderer_factory)
    assert reset_calls == [project.cartesian_controller] * 2
    assert progress == [ActionRolloutMetrics(episodes=1, ik_failures=1),
                        ActionRolloutMetrics(episodes=2, ik_failures=1, timeouts=1)]
    assert metrics == progress[-1] and metrics.success_rate_percent == 0.0
    assert len(action.histories) == len(vision.inputs) == 2
    assert [tokens.shape[1] for tokens, _, _ in action.histories] == [1, 2]
    assert [snapshot[0] for snapshot in renderers[0].snapshots] == pytest.approx([0, 0.05])
    assert all(renderer.closed for renderer in renderers)


@pytest.mark.parametrize("failure", ["second_renderer", "render", "policy", "progress"])
def test_errors_propagate_close_resources_and_restore_modes(project, monkeypatch, failure):
    renderers = []
    action, vision = ObservingActionModel(), ImagePoseModel()
    action.eval()

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    def renderer_factory(*args, **kwargs):
        if failure == "second_renderer" and renderers:
            fail()
        renderer = FakeRenderer(*args, **kwargs)
        if failure == "render":
            renderer.render = fail
        renderers.append(renderer)
        return renderer

    if failure == "policy":
        monkeypatch.setattr(action, "predict_action", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        evaluate_action_rollouts(project=project, action_policy=action, vision_model=vision,
                                 device=torch.device("cpu"), renderer_factory=renderer_factory,
                                 on_episode=fail if failure == "progress" else None)
    assert renderers and all(renderer.closed for renderer in renderers)
    assert not action.training and vision.training and not vision.mixed_mode.training


def test_subtick_duration_is_rejected_and_partial_tick_does_not_overrun(project):
    project.environment.episode_seconds = 0.01
    with pytest.raises(ValueError, match="at least one control tick"):
        evaluate_action_rollouts(project=project, action_policy=ObservingActionModel(),
                                 vision_model=ImagePoseModel(), device=torch.device("cpu"),
                                 renderer_factory=FakeRenderer)
    project.environment.episode_seconds = 0.09
    action = ObservingActionModel()
    evaluate_action_rollouts(project=project, action_policy=action, vision_model=ImagePoseModel(),
                             device=torch.device("cpu"), renderer_factory=FakeRenderer)
    assert len(action.histories) == 2  # One 50-ms tick per episode, without overshooting 90 ms.
