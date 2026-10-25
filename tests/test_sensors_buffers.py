"""Camera freshness/failure semantics shared by simulation and hardware."""

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from camera_feed_getter import PhysicalCameraFeedGetter, SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from schemas import CAMERA_IMAGE_SHAPES, CameraFrame
from vision_errors import CameraAcquisitionError, StaleVisionDataError, VisionDataUnavailableError


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def frame(name="wrist", timestamp=0.0):
    return CameraFrame(camera_id=name, timestamp=timestamp,
                       rgb=np.zeros(CAMERA_IMAGE_SHAPES[name], dtype=np.uint8))


@pytest.mark.parametrize("camera_id, wrong_shape", [
    ("wrist", (300, 300, 3)), ("external", (128, 128, 3)),
])
def test_frame_rejects_other_cameras_resolution(camera_id, wrong_shape):
    with pytest.raises(ValueError, match=f"Camera '{camera_id}' requires RGB image shape"):
        CameraFrame(camera_id=camera_id, timestamp=0.0,
                    rgb=np.zeros(wrong_shape, dtype=np.uint8))


def test_frame_rejects_unknown_camera_id():
    with pytest.raises(ValueError, match="Unknown camera ID"):
        CameraFrame(camera_id="unknown", timestamp=0.0,
                    rgb=np.zeros((128, 128, 3), dtype=np.uint8))


def test_missing_image_is_recoverable_and_success_resets_failed_read_streak():
    clock = Clock()
    buffer = CameraFrameBuffer("wrist", clock=clock)
    with pytest.raises(VisionDataUnavailableError):
        buffer.get_latest(0.0)
    buffer.record_read(False, None)
    clock.now = 0.5
    buffer.record_read(False, None)
    assert buffer.last_camera_read_failure == 0.0
    buffer.record_read(True, frame(timestamp=0.5))
    assert buffer.last_camera_read_failure is None
    buffer.record_read(False, None)
    assert buffer.last_camera_read_failure == 0.5
    assert buffer.get_latest(0.5).timestamp == 0.5


def test_reader_also_expires_failure_when_producer_stops_reporting():
    clock = Clock()
    buffer = CameraFrameBuffer("wrist", clock=clock)
    buffer.record_read(False, None)
    clock.now = 1.0
    with pytest.raises(VisionDataUnavailableError):
        buffer.get_latest(0.0)
    clock.now = 1.001
    with pytest.raises(StaleVisionDataError):
        buffer.get_latest(0.0)


def test_recovery_cannot_hide_timeout_even_without_intermediate_getter():
    clock = Clock()
    buffer = CameraFrameBuffer("wrist", clock=clock)
    buffer.record_read(False, None)
    clock.now = 1.1
    buffer.record_read(True, frame(timestamp=0.0))
    assert buffer.last_camera_read_failure is None
    with pytest.raises(StaleVisionDataError):
        buffer.get_latest(0.0)
    buffer.record_read(True, frame(timestamp=0.5))
    with pytest.raises(StaleVisionDataError):
        buffer.get_latest(0.5)
    buffer.reset()
    assert buffer.failure is None
    assert buffer.latest_frame is None
    assert buffer.last_camera_read_failure is None
    buffer.record_read(True, frame(timestamp=0.0))
    assert buffer.get_latest(0.0).timestamp == 0.0


def test_frame_age_has_exact_one_second_boundary_and_latches():
    buffer = CameraFrameBuffer("wrist", clock=Clock(800.0))
    buffer.record_read(True, frame(timestamp=4.0))
    assert buffer.get_latest(5.0).timestamp == 4.0
    with pytest.raises(StaleVisionDataError):
        buffer.get_latest(5.001)
    buffer.record_read(True, frame(timestamp=5.001))
    with pytest.raises(StaleVisionDataError):
        buffer.get_latest(5.001)


def test_publish_and_capture_both_copy_image_storage_and_preserve_timestamp():
    buffer = CameraFrameBuffer("wrist")
    original = frame(timestamp=8.0)
    buffer.record_read(True, original)
    original.rgb[:] = 250
    first = buffer.get_latest(8.1)
    assert not np.any(first.rgb)
    first.rgb[:] = 170
    second = buffer.get_latest(8.2)
    assert not np.any(second.rgb)
    assert second.timestamp == 8.0
    buffer.record_read(False, None)
    assert buffer.get_latest(8.2).timestamp == 8.0


def test_bad_publish_is_a_programming_error():
    buffer = CameraFrameBuffer("wrist")
    with pytest.raises(ValueError):
        buffer.record_read(True, None)
    with pytest.raises(ValueError):
        buffer.record_read(True, frame(name="external"))


@pytest.mark.parametrize("missing_wrist", [True, False])
def test_missing_camera_never_hides_other_cameras_fatal_error(missing_wrist):
    buffers = {"external": CameraFrameBuffer("external")}
    if not missing_wrist:
        buffers["wrist"] = CameraFrameBuffer("wrist")
    buffers["external"].record_failure(CameraAcquisitionError("renderer unavailable"))
    with pytest.raises(CameraAcquisitionError, match="renderer unavailable"):
        PhysicalCameraFeedGetter(buffers, Clock()).capture()


def test_success_of_other_camera_does_not_clear_failure_timer():
    clock = Clock()
    buffers = {name: CameraFrameBuffer(name, clock=clock) for name in ("wrist", "external")}
    buffers["wrist"].record_read(False, None)
    clock.now = 1.01
    buffers["external"].record_read(True, frame("external", 1.01))
    with pytest.raises(StaleVisionDataError):
        PhysicalCameraFeedGetter(buffers, clock).capture()


def test_sim_getter_uses_simulation_time_and_does_not_advance_or_render():
    buffers = {name: CameraFrameBuffer(name) for name in ("wrist", "external")}
    for name, buffer in buffers.items():
        buffer.record_read(True, frame(name, 20.0))
    environment = SimpleNamespace(data=SimpleNamespace(time=20.1), state_lock=Lock())
    getter = SimCameraFeedGetter(environment, buffers)
    frames = getter.capture()
    assert set(frames) == {"wrist", "external"}
    assert environment.data.time == 20.1
    frames["wrist"].rgb[:] = 42
    assert not np.any(getter.capture()["wrist"].rgb)
