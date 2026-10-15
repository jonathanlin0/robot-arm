"""Verify model input/output ordering and estimate-time velocity tracking."""

import numpy as np
import pytest
import torch

from cube_state_provider import CubeStateProvider
from schemas import CameraFrame, RobotState
from vision_errors import StaleVisionDataError, VisionDataUnavailableError


class PoseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pose = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]])
        self.inputs = None

    def forward(self, wrist, external, joints):
        self.inputs = wrist, external, joints
        assert not torch.is_grad_enabled()
        return self.pose


def inputs(timestamp=1.0):
    frames = {name: CameraFrame(camera_id=name, timestamp=timestamp,
                               rgb=np.full((300, 300, 3), fill, dtype=np.uint8))
              for name, fill in (("wrist", 255), ("external", 127))}
    state = RobotState(timestamp=timestamp, joint_positions=np.arange(6, dtype=np.float64),
                       joint_velocities=np.zeros(6), commanded_joint_targets=np.zeros(6),
                       gripper_position=np.zeros(3))
    return frames, state


def test_pose_input_normalization_order_and_output_mapping():
    model = PoseModel()
    provider = CubeStateProvider(model, torch.device("cpu"))
    frames, state = inputs()
    result = provider.estimate(frames, state, current_time=1.5)
    wrist, external, joints = model.inputs
    assert wrist.shape == external.shape == (1, 3, 300, 300)
    assert wrist.dtype == external.dtype == joints.dtype == torch.float32
    assert joints.shape == (1, 6)
    torch.testing.assert_close(wrist, torch.ones_like(wrist))
    torch.testing.assert_close(external, torch.full_like(external, 127 / 255))
    torch.testing.assert_close(joints, torch.arange(6, dtype=torch.float32).unsqueeze(0))
    assert np.all(frames["external"].rgb == 127)
    np.testing.assert_array_equal(result["orange"].position, [1, 2, 3])
    np.testing.assert_array_equal(result["orange"].orientation, [4, 5, 6, 7])
    np.testing.assert_array_equal(result["blue"].position, [8, 9, 10])
    np.testing.assert_array_equal(result["blue"].orientation, [11, 12, 13, 14])
    for cube in result.values():
        assert cube.timestamp == 1.5
        np.testing.assert_array_equal(cube.linear_velocity, np.zeros(3))


def test_velocities_use_estimate_time_and_repeated_time_preserves_baseline():
    model = PoseModel()
    provider = CubeStateProvider(model, torch.device("cpu"))
    frames, state = inputs()
    first = provider.estimate(frames, state, 1.0)
    first["orange"].position[:] = 900  # Caller mutation cannot change history.
    model.pose[:, :3] += 1
    moved = provider.estimate(frames, state, 1.25)
    np.testing.assert_array_equal(moved["orange"].linear_velocity, [4, 4, 4])
    model.pose[:, :3] += 100
    repeated = provider.estimate(frames, state, 1.25)
    backwards = provider.estimate(frames, state, 1.125)
    np.testing.assert_array_equal(repeated["orange"].linear_velocity, [4, 4, 4])
    np.testing.assert_array_equal(backwards["orange"].linear_velocity, [4, 4, 4])
    assert provider.previous_estimates["orange"].timestamp == 1.25
    model.pose[:, :3] -= 99
    later = provider.estimate(frames, state, 1.5)
    np.testing.assert_array_equal(later["orange"].linear_velocity, [4, 4, 4])
    provider.reset()
    reset = provider.estimate(frames, state, 1.5)
    np.testing.assert_array_equal(reset["orange"].linear_velocity, [0, 0, 0])


def test_stale_existing_view_precedes_missing_view_and_joint_readings():
    provider = CubeStateProvider(PoseModel(), torch.device("cpu"))
    frames, state = inputs()
    del frames["wrist"]
    with pytest.raises(StaleVisionDataError):
        provider.estimate(frames, None, 2.01)
    with pytest.raises(VisionDataUnavailableError):
        provider.estimate(frames, state, 2.0)
    frames, state = inputs()
    with pytest.raises(VisionDataUnavailableError):
        provider.estimate(frames, None, 2.0)
    state.joint_positions = None
    with pytest.raises(VisionDataUnavailableError):
        provider.estimate(frames, state, 2.0)


def test_exact_age_limit_allowed_and_pose_values_not_gated():
    model = PoseModel()
    model.pose[:, 0] = float("nan")
    provider = CubeStateProvider(model, torch.device("cpu"))
    frames, state = inputs()
    result = provider.estimate(frames, state, 2.0)
    assert np.isnan(result["orange"].position[0])
    # No quaternion normalization or plausibility filter is silently applied.
    np.testing.assert_array_equal(result["orange"].orientation, [4, 5, 6, 7])
