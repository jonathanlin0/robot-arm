"""Read timestamped wrist/external RGB frames without acquiring or waiting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from camera_frame_buffer import CameraFrameBuffer
from environment import CubeStackEnvironment
from schemas import CameraFrames
from vision_errors import VisionDataUnavailableError


SIM_CAMERA_NAMES = {"wrist": "wrist_cam", "external": "external"}


def _capture_latest(buffers: dict[str, CameraFrameBuffer], current_time: float) -> CameraFrames:
    frames = {}
    missing = []
    # Check every required camera before reporting missing data. In particular,
    # an empty wrist buffer cannot conceal an external camera's fatal failure.
    for camera_id in SIM_CAMERA_NAMES:
        if camera_id not in buffers:
            missing.append(camera_id)
            continue
        try:
            frames[camera_id] = buffers[camera_id].get_latest(current_time)
        except VisionDataUnavailableError:
            missing.append(camera_id)
    if missing:
        raise VisionDataUnavailableError(f"Required camera frames unavailable: {', '.join(missing)}.")
    return frames


class CameraFeedGetter(ABC):
    """Snapshot independent latest-frame buffers; never perform camera I/O."""

    buffers: dict[str, CameraFrameBuffer]

    @abstractmethod
    def capture(self) -> CameraFrames:
        """Copy each view, preserving its original capture timestamp."""
        raise NotImplementedError


class SimCameraFeedGetter(CameraFeedGetter):
    """Read externally produced simulation images using simulation time."""

    def __init__(self, environment: CubeStackEnvironment,
                 buffers: dict[str, CameraFrameBuffer]) -> None:
        self.environment = environment
        self.buffers = buffers

    def capture(self) -> CameraFrames:
        with self.environment.state_lock:
            current_time = float(self.environment.data.time)
        return _capture_latest(self.buffers, current_time)


class PhysicalCameraFeedGetter(CameraFeedGetter):
    """Read already-buffered RGB images using the hardware session clock."""

    def __init__(self, buffers: dict[str, CameraFrameBuffer], clock: Callable[[], float]) -> None:
        self.buffers = buffers
        self.clock = clock

    def capture(self) -> CameraFrames:
        return _capture_latest(self.buffers, self.clock())
