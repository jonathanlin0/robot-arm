"""Shared learned cube-pose estimation and position-based velocity tracking."""

from __future__ import annotations

import numpy as np
import torch

from schemas import CameraFrames, CubeState, CubeStates, RobotState
from vision_errors import MAXIMUM_FRAME_AGE_SECONDS, StaleVisionDataError, VisionDataUnavailableError


class CubeStateProvider:
    """Estimate both cube poses from the latest images and joint positions."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 maximum_frame_age_seconds: float = MAXIMUM_FRAME_AGE_SECONDS) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.maximum_frame_age_seconds = maximum_frame_age_seconds
        self.previous_estimates: CubeStates | None = None

    def reset(self) -> None:
        """Clear velocity history without affecting camera acquisition health."""
        self.previous_estimates = None

    def estimate(self, frames: CameraFrames, robot_state: RobotState,
                 current_time: float) -> CubeStates:
        """Use latest readings without synchronization or pose-value gating.

        Images become BCHW float32 / 255; joints retain their radian units.
        The model returns orange XYZ/quaternion followed by blue XYZ/quaternion.
        Repeated/nonincreasing estimate times preserve the velocity baseline.
        """
        required = ("wrist", "external")
        # Freshness must take precedence over missing inputs in another view.
        for name in required:
            if name in frames and current_time - frames[name].timestamp > self.maximum_frame_age_seconds:
                raise StaleVisionDataError(f"Camera {name!r} frame is older than the permitted age.")
        if any(name not in frames for name in required):
            raise VisionDataUnavailableError("Both wrist and external frames are required.")
        if robot_state is None or robot_state.joint_positions is None:
            raise VisionDataUnavailableError("Joint positions are unavailable.")

        images = [torch.from_numpy(np.ascontiguousarray(frames[name].rgb.transpose(2, 0, 1)))
                  .to(device=self.device, dtype=torch.float32).unsqueeze(0).div(255.0)
                  for name in required]
        joints = torch.as_tensor(robot_state.joint_positions, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        with torch.inference_mode():
            self.model.eval()
            poses = self.model(images[0], images[1], joints)[0].to(device="cpu", dtype=torch.float32).numpy()

        estimates: CubeStates = {}
        for name, offset in (("orange", 0), ("blue", 7)):
            position = poses[offset:offset + 3].copy()
            velocity = np.zeros(3, dtype=np.float32)
            if self.previous_estimates is not None:
                previous = self.previous_estimates[name]
                elapsed = current_time - previous.timestamp
                velocity = ((position - previous.position) / elapsed if elapsed > 0
                            else previous.linear_velocity.copy())
            estimates[name] = CubeState(
                timestamp=current_time, position=position,
                orientation=poses[offset + 3:offset + 7].copy(), linear_velocity=velocity,
            )
        if (self.previous_estimates is None
                or current_time > self.previous_estimates["orange"].timestamp):
            self.previous_estimates = {
                name: CubeState(timestamp=state.timestamp, position=state.position.copy(),
                                orientation=state.orientation.copy(), linear_velocity=state.linear_velocity.copy())
                for name, state in estimates.items()
            }
        return estimates
