"""A complete image-to-command simulation loop without a graphics context."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from camera_feed_getter import SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from cartesian_controller import CartesianController
from cube_state_provider import CubeStateProvider
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from loop_runner import LoopRunner
from robot_state_provider import SimRobotStateProvider
from schemas import CAMERA_IMAGE_SHAPES, OBSERVATION_FIELDS, CameraFrame
from schemas.config import load_config
from trained_policy import TrainedPolicy


class TestClock:
    __test__ = False

    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def wait(self, duration):
        self.waits.append(duration)
        self.now += duration


class ImagePoseModel(torch.nn.Module):
    """Return image-dependent estimates deliberately unlike simulation truth."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, wrist, external, joint_positions):
        self.inputs.append((wrist.clone(), external.clone(), joint_positions.clone()))
        poses = torch.tensor(
            [[0.11, 0.12, 0.13, 3, 4, 0.21, -0.22, 0.23, 0, -2]],
            dtype=torch.float32,
        )
        # A changed image changes the estimate; no MuJoCo state is accessible.
        poses[:, 0] += wrist[:, 0, 0, 0] * 0.01
        return poses


class ObservingActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.histories = []

    def predict_action(self, tokens, valid, episode_start, deterministic):
        assert deterministic
        self.histories.append((tokens.clone(), valid.clone(), episode_start.clone()))
        return torch.tensor([[-0.2, 0.0, -0.2, 0.0]], dtype=torch.float32)


def test_two_full_ticks_use_image_estimates_and_share_one_simulator():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    environment = CubeStackEnvironment(scene_path=config.environment.scene_path)
    clock = TestClock()
    buffers = {
        name: CameraFrameBuffer(
            camera_id=name, maximum_frame_age_seconds=config.cameras.maximum_frame_age_seconds,
            clock=clock,
        ) for name in ("wrist", "external")
    }
    poses, actions = ImagePoseModel(), ObservingActionModel()
    policy = TrainedPolicy(policy=actions, observation_fields=OBSERVATION_FIELDS, history_length=4)
    controller = CartesianController(robot_model=environment.model, config=config.cartesian_controller)
    state_provider = SimRobotStateProvider(environment=environment)
    backend = SimJointCommandBackend(environment=environment)
    getter = SimCameraFeedGetter(environment=environment, buffers=buffers)
    cube_provider = CubeStateProvider(
        model=poses, device=torch.device("cpu"),
        maximum_frame_age_seconds=config.cameras.maximum_frame_age_seconds,
    )
    runner = LoopRunner(
        robot_state_provider=state_provider, camera_feed_getter=getter,
        cube_state_provider=cube_provider, trained_policy=policy,
        controller=controller, backend=backend, config=config.loop,
        deterministic=True, clock=clock, wait=clock.wait,
    )
    runner.reset()
    initial_state = state_provider.read()
    assert environment.data.time == 0.0

    def publish(wrist_value):
        for name, value in (("wrist", wrist_value), ("external", 127)):
            buffers[name].record_read(
                success=True,
                frame=CameraFrame(camera_id=name, timestamp=float(environment.data.time),
                                  rgb=np.full(CAMERA_IMAGE_SHAPES[name], value, dtype=np.uint8)),
            )

    publish(255)
    first = runner.step()
    assert first is not None
    assert first.robot_state.timestamp == first.receipt.timestamp == 0.0
    assert environment.data.time == pytest.approx(0.05)
    assert clock.now == pytest.approx(0.05)
    np.testing.assert_allclose(first.cube_states["orange"].position, [0.12, 0.12, 0.13])
    np.testing.assert_array_equal(first.cube_states["orange"].linear_velocity, np.zeros(3))
    np.testing.assert_array_equal(first.receipt.joint_targets, state_provider.read().commanded_joint_targets)
    issued_xyz = controller.prev_accepted_target.copy()
    tokens, valid, starts = actions.histories[0]
    assert tokens.dtype == torch.float32 and tokens.shape == (1, 1, 44)
    assert valid.tolist() == [[True]] and starts.tolist() == [[True]]
    np.testing.assert_allclose(tokens[0, 0, 21:24], first.cube_states["orange"].position)
    assert first.cube_states["orange"].orientation is None
    assert first.cube_states["blue"].orientation is None
    np.testing.assert_allclose(tokens[0, 0, 24:26], [0.6, 0.8], atol=1e-7)
    np.testing.assert_allclose(tokens[0, 0, 32:34], [0., -1.], atol=1e-7)
    np.testing.assert_array_equal(tokens[0, 0, 37:41], np.zeros(4))
    np.testing.assert_allclose(tokens[0, 0, 41:44], initial_state.gripper_position, atol=1e-7)

    # Alter hidden simulator cube state independently of the synthetic camera
    # images. Runtime perception must still consume only image/model outputs.
    with environment.state_lock:
        environment.data.joint("orange_cube_joint").qpos[:3] = [0.38, -0.19, 0.02]
        environment.data.joint("blue_cube_joint").qpos[:3] = [0.39, 0.18, 0.02]
        mujoco.mj_forward(environment.model, environment.data)
    publish(0)
    second = runner.step()
    assert second is not None
    assert environment.data.time == pytest.approx(0.10)
    assert clock.now == pytest.approx(0.10)
    assert len(clock.waits) == 2
    assert second.robot_state.timestamp == second.receipt.timestamp == pytest.approx(0.05)
    np.testing.assert_allclose(second.cube_states["orange"].position, [0.11, 0.12, 0.13])
    np.testing.assert_allclose(second.cube_states["orange"].linear_velocity, [-0.2, 0, 0], atol=2e-7)
    np.testing.assert_allclose(second.cube_states["blue"].position, [0.21, -0.22, 0.23])
    tokens, valid, starts = actions.histories[1]
    assert tokens.shape == (1, 2, 44)
    assert valid.tolist() == [[True, True]] and starts.tolist() == [[True, False]]
    np.testing.assert_allclose(tokens[0, 1, 21:24], second.cube_states["orange"].position)
    np.testing.assert_allclose(tokens[0, 1, 26:29], [-0.2, 0, 0], atol=2e-7)
    np.testing.assert_array_equal(tokens[0, 1, 37:41], first.action)
    np.testing.assert_allclose(tokens[0, 1, 41:44], issued_xyz, atol=1e-7)
    for wrist, external, joints in poses.inputs:
        assert wrist.shape == (1, 3, 128, 128)
        assert external.shape == (1, 3, 300, 300)
        assert wrist.dtype == external.dtype == joints.dtype == torch.float32
        assert joints.shape == (1, 6)
        assert external[0, 0, 0, 0].item() == pytest.approx(127 / 255)
    assert getter.environment is state_provider.environment is backend.environment is environment
