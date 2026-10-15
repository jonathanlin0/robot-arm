"""Worker lifecycle, coherent simulation snapshots, and failure forwarding."""

from threading import Event, Lock, get_ident
from time import monotonic
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from camera_feed_getter import PhysicalCameraFeedGetter, SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from camera_producers import PhysicalCameraProducer, SimCameraProducer
from environment import DEFAULT_SCENE_PATH
from vision_errors import CameraAcquisitionError


def make_buffers():
    return {name: CameraFrameBuffer(name) for name in ("wrist", "external")}


def test_physical_worker_reads_independently_converts_bgr_and_leaves_connection_owned():
    read_started = Event()
    allow_read = Event()
    supplied = np.zeros((300, 300, 3), dtype=np.uint8)
    supplied[:] = [10, 20, 30]

    class Camera:
        count = 0

        def read(self):
            self.count += 1
            if self.count > 1:
                read_started.set()
                allow_read.wait(timeout=1.0)
                return False, None
            return True, supplied

        def release(self):
            pytest.fail("Producer must not release an application-owned connection.")

    camera = Camera()
    buffer = CameraFrameBuffer("wrist")
    producer = PhysicalCameraProducer("wrist", camera, buffer, fps=30, clock=lambda: 4.0)
    producer.start()
    try:
        assert read_started.wait(timeout=2.0)
        latest = buffer.get_latest(4.0)
        np.testing.assert_array_equal(latest.rgb[0, 0], [30, 20, 10])
        assert latest.timestamp == 4.0
        assert camera.count >= 2  # No getter was needed to initiate acquisition.
        np.testing.assert_array_equal(supplied[0, 0], [10, 20, 30])
    finally:
        allow_read.set()
        producer.stop()


def test_unexpected_read_failure_is_raised_on_getter_thread():
    raised = Event()

    class BrokenCamera:
        def read(self):
            raised.set()
            raise OSError("unplugged")

    buffer = CameraFrameBuffer("wrist")
    producer = PhysicalCameraProducer("wrist", BrokenCamera(), buffer)
    producer.start()
    assert raised.wait(timeout=1.0)
    producer.stop()
    getter = PhysicalCameraFeedGetter({"wrist": buffer}, monotonic)
    with pytest.raises(CameraAcquisitionError, match="unplugged") as exception:
        getter.capture()
    assert isinstance(exception.value.__cause__, OSError)
    producer.reset_session()
    assert buffer.failure is None


def test_renderer_setup_failure_is_published_to_both_buffers():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    environment = SimpleNamespace(model=model)
    buffers = make_buffers()

    def broken_renderer(*args, **kwargs):
        raise RuntimeError("No graphics context")

    producer = SimCameraProducer(environment, buffers, renderer_factory=broken_renderer)
    producer.start()
    producer.stop()
    for buffer in buffers.values():
        with pytest.raises(CameraAcquisitionError, match="No graphics context"):
            buffer.get_latest(0.0)


def test_simulation_worker_owns_renderer_copies_coherent_state_and_resets_safely():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.time = 4.0
    environment = SimpleNamespace(model=model, data=data, state_lock=Lock())
    reset_calls = []

    def reset_environment():
        with environment.state_lock:
            reset_calls.append(True)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

    environment.reset = reset_environment
    buffers = make_buffers()
    both_rendered = Event()
    calls = []
    owner_thread = []

    class Renderer:
        def __init__(self, model_arg, height, width):
            owner_thread.append(get_ident())
            assert model_arg is model
            assert height == width == 300
            calls.append("create")

        def update_scene(self, snapshot, camera):
            assert get_ident() == owner_thread[0]
            assert snapshot is not data
            assert snapshot.time == 4.0
            self.camera = camera
            calls.append(camera)

        def render(self):
            assert get_ident() == owner_thread[0]
            return np.full((300, 300, 3), 12, dtype=np.uint8)

        def close(self):
            assert get_ident() == owner_thread[0]
            calls.append("close")

    # Set the completion event after publication of the second frame.
    original_publish = buffers["external"].record_read

    def publish_external(success, frame):
        original_publish(success, frame)
        both_rendered.set()

    buffers["external"].record_read = publish_external
    producer = SimCameraProducer(environment, buffers, fps=30, renderer_factory=Renderer)
    producer.start()
    try:
        assert both_rendered.wait(timeout=2.0)
        result = SimCameraFeedGetter(environment, buffers).capture()
        assert result["wrist"].timestamp == result["external"].timestamp == 4.0
        assert data.time == 4.0  # Rendering never advances physics.
    finally:
        producer.stop()
    assert owner_thread[0] != get_ident()
    assert calls[0] == "create" and calls[-1] == "close"
    assert "wrist_cam" in calls and "external" in calls
    producer.reset_session()
    assert reset_calls == [True]
    assert data.time == 0.0
    assert all(buffer.latest_frame is None for buffer in buffers.values())
