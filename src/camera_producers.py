"""Independent camera workers; getters only read their latest-frame buffers.

Applications own producers and camera connections. Use start()/stop() (or a
context manager), and reset_session() when the associated episode is reset.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import mujoco

from camera_feed_getter import SIM_CAMERA_NAMES
from camera_frame_buffer import CAMERA_FPS, CameraFrameBuffer
from environment import CubeStackEnvironment
from schemas import CAMERA_IMAGE_SHAPE, CameraFrame
from vision_errors import CameraAcquisitionError


class _CameraProducer:
    """Manage worker lifetime and serialize acquisition with session reset."""

    def __init__(self, buffers: dict[str, CameraFrameBuffer], fps: float) -> None:
        if fps <= 0:
            raise ValueError("Camera FPS must be positive.")
        self.buffers = buffers
        self.fps = fps
        self._stop_event = Event()
        self._capture_lock = Lock()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start acquisition independently of policy and getter calls."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Join the worker; applications release their hardware connection later."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise CameraAcquisitionError(
                    "Camera worker did not stop; unblock the camera's pending read before closing."
                )
            self._thread = None

    def _publish_failure(self, exception: Exception) -> None:
        failure = CameraAcquisitionError(f"{type(self).__name__} failed: {exception}")
        failure.__cause__ = exception
        for buffer in self.buffers.values():
            buffer.record_failure(failure)

    def _run(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class SimCameraProducer(_CameraProducer):
    """Render both views at a fixed cadence without advancing the simulation.

    One worker owns renderer creation, use, and destruction. Physics is copied
    under the shared environment lock, then rendered outside that lock.
    """

    def __init__(self, environment: CubeStackEnvironment,
                 buffers: dict[str, CameraFrameBuffer], fps: float = CAMERA_FPS,
                 renderer_factory: Callable[..., Any] = mujoco.Renderer) -> None:
        super().__init__(buffers=buffers, fps=fps)
        self.environment = environment
        self.renderer_factory = renderer_factory

    def reset_session(self, reset_environment: bool = True) -> None:
        """Atomically reset physics and clear images relative to production.

        Reset policy/perception/controller histories separately after this call.
        A frame copied before reset cannot be published into the new session.
        """
        with self._capture_lock:
            if reset_environment:
                self.environment.reset()
            for buffer in self.buffers.values():
                buffer.reset()

    def _run(self) -> None:
        renderer = None
        try:
            renderer = self.renderer_factory(
                self.environment.model, height=CAMERA_IMAGE_SHAPE[0], width=CAMERA_IMAGE_SHAPE[1]
            )
            snapshot = mujoco.MjData(self.environment.model)
            while not self._stop_event.is_set():
                started = monotonic()
                with self._capture_lock:
                    with self.environment.state_lock:
                        mujoco.mj_copyData(snapshot, self.environment.model, self.environment.data)
                        timestamp = float(snapshot.time)
                    for camera_id, camera_name in SIM_CAMERA_NAMES.items():
                        renderer.update_scene(snapshot, camera=camera_name)
                        rgb = renderer.render()
                        self.buffers[camera_id].record_read(
                            success=True,
                            frame=CameraFrame(camera_id=camera_id, timestamp=timestamp, rgb=rgb),
                        )
                self._stop_event.wait(max(0.0, 1.0 / self.fps - (monotonic() - started)))
        except Exception as exception:
            self._publish_failure(exception)
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as exception:
                    self._publish_failure(exception)


class PhysicalCameraProducer(_CameraProducer):
    """Continuously drain a configured OpenCV camera into its single RGB slot.

    The application opens/configures the source at 30 FPS and 300x300. read()
    normally supplies the camera cadence: successful reads are not artificially
    delayed, allowing queued driver frames to be consumed promptly. Failed
    reads retry at fps to avoid spinning on a disconnected camera.
    """

    def __init__(self, camera_id: str, camera: Any, buffer: CameraFrameBuffer,
                 fps: float = CAMERA_FPS, clock: Callable[[], float] = monotonic) -> None:
        super().__init__(buffers={camera_id: buffer}, fps=fps)
        self.camera_id = camera_id
        self.camera = camera
        self.buffer = buffer
        self.clock = clock

    def reset_session(self) -> None:
        """Clear this camera's buffer while production is serialized."""
        with self._capture_lock:
            self.buffer.reset()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._capture_lock:
                    success, image = self.camera.read()
                    frame = None
                    if success:
                        # OpenCV provides BGR. Copy because camera memory may be reused.
                        frame = CameraFrame(camera_id=self.camera_id, timestamp=self.clock(),
                                            rgb=image[:, :, ::-1].copy())
                    self.buffer.record_read(success=bool(success), frame=frame)
                if not success:
                    self._stop_event.wait(1.0 / self.fps)
        except Exception as exception:
            self._publish_failure(exception)
