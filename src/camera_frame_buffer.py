"""Thread-safe latest-frame storage for independent camera producers."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from time import monotonic

from schemas import CameraFrame
from vision_errors import MAXIMUM_FRAME_AGE_SECONDS, StaleVisionDataError, VisionDataUnavailableError


CAMERA_FPS = 30.0


def copy_frame(frame: CameraFrame) -> CameraFrame:
    """Detach an observation from memory that a producer or consumer may reuse."""
    return CameraFrame(camera_id=frame.camera_id, timestamp=frame.timestamp, rgb=frame.rgb.copy())


class CameraFrameBuffer:
    """Retain one image and latch fatal acquisition failures until session reset.

    Frame age uses the supplied sensor clock; failed-read duration independently
    uses monotonic wall time. A later successful read cannot erase a timeout.
    """

    def __init__(
        self,
        camera_id: str,
        maximum_frame_age_seconds: float = MAXIMUM_FRAME_AGE_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.camera_id = camera_id
        self.maximum_frame_age_seconds = maximum_frame_age_seconds
        self.clock = clock
        self.lock = Lock()
        self.latest_frame: CameraFrame | None = None
        self.last_camera_read_failure: float | None = None
        self.failure: RuntimeError | None = None

    def reset(self) -> None:
        """Clear the session while the application has paused its producer."""
        with self.lock:
            self.latest_frame = None
            self.last_camera_read_failure = None
            self.failure = None

    def _check_failed_read_duration(self, now: float) -> None:
        if (self.failure is None and self.last_camera_read_failure is not None
                and now - self.last_camera_read_failure > self.maximum_frame_age_seconds):
            self.failure = StaleVisionDataError(
                f"Camera {self.camera_id!r} has failed to read for "
                f"{now - self.last_camera_read_failure:.3f}s "
                f"(limit {self.maximum_frame_age_seconds:.3f}s)."
            )

    def record_read(self, success: bool, frame: CameraFrame | None) -> None:
        """Publish a successful frame or record the first failed-read time."""
        if success and (frame is None or frame.camera_id != self.camera_id):
            raise ValueError("A successful read requires a frame with the buffer's camera ID.")
        with self.lock:
            now = self.clock()
            # Check before clearing on recovery, including when nobody polled.
            self._check_failed_read_duration(now)
            if success:
                self.latest_frame = copy_frame(frame)
                self.last_camera_read_failure = None
            elif self.last_camera_read_failure is None:
                self.last_camera_read_failure = now

    def record_failure(self, failure: RuntimeError) -> None:
        """Expose an unexpected worker failure to the controlling thread."""
        with self.lock:
            if self.failure is None:
                self.failure = failure

    def get_latest(self, current_time: float) -> CameraFrame:
        """Copy the latest image immediately, checking fatal health first."""
        with self.lock:
            self._check_failed_read_duration(self.clock())
            if self.failure is not None:
                raise self.failure
            if self.latest_frame is None:
                raise VisionDataUnavailableError(f"Camera {self.camera_id!r} has no frame yet.")
            age = current_time - self.latest_frame.timestamp
            if age > self.maximum_frame_age_seconds:
                self.failure = StaleVisionDataError(
                    f"Camera {self.camera_id!r} frame is {age:.3f}s old "
                    f"(limit {self.maximum_frame_age_seconds:.3f}s)."
                )
                raise self.failure
            return copy_frame(self.latest_frame)
