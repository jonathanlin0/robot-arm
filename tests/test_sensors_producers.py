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


@pytest.mark.parametrize("camera_id, size", [("wrist", 128), ("external", 300)])
def test_physical_worker_reads_independently_converts_bgr_and_leaves_connection_owned(camera_id, size):
    read_started = Event()
    allow_read = Event()
    supplied = np.zeros((size, size, 3), dtype=np.uint8)
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
    buffer = CameraFrameBuffer(camera_id)
    producer = PhysicalCameraProducer(camera_id, camera, buffer, fps=30, clock=lambda: 4.0)
    producer.start()
    try:
        assert read_started.wait(timeout=2.0)
        latest = buffer.get_latest(4.0)
        assert latest.rgb.shape == (size, size, 3)
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


@pytest.mark.parametrize("failure", ["create_second", "render_first", "close_first"])
def test_simulation_worker_closes_all_created_renderers_after_failure(failure):
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    data = mujoco.MjData(model)
    environment = SimpleNamespace(model=model, data=data, state_lock=Lock())
    buffers = make_buffers()
    created, closed = [], []

    class Renderer:
        def __init__(self, model_arg, height, width):
            self.size = height
            if failure == "create_second" and height == 300:
                raise RuntimeError(failure)
            created.append(height)

        def update_scene(self, snapshot, camera):
            pass

        def render(self):
            if failure == "render_first":
                raise RuntimeError(failure)
            producer._stop_event.set()  # End after this pair, then test cleanup.
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)

        def close(self):
            closed.append(self.size)
            if failure == "close_first" and self.size == 128:
                raise RuntimeError(failure)

    producer = SimCameraProducer(environment, buffers, renderer_factory=Renderer)
    producer._run()
    assert created == ([128] if failure == "create_second" else [128, 300])
    assert closed == created
    for buffer in buffers.values():
        with pytest.raises(CameraAcquisitionError, match=failure):
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
    snapshots = []

    class Renderer:
        def __init__(self, model_arg, height, width):
            owner_thread.append(get_ident())
            assert model_arg is model
            assert height == width
            self.size = height
            calls.append(("create", height))

        def update_scene(self, snapshot, camera):
            assert get_ident() == owner_thread[0]
            assert snapshot is not data
            assert snapshot.time == 4.0
            self.camera = camera
            assert self.size == {"wrist_cam": 128, "external": 300}[camera]
            snapshots.append(snapshot)
            calls.append(camera)

        def render(self):
            assert get_ident() == owner_thread[0]
            return np.full((self.size, self.size, 3), 12, dtype=np.uint8)

        def close(self):
            assert get_ident() == owner_thread[0]
            calls.append(("close", self.size))

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
        assert result["wrist"].rgb.shape == (128, 128, 3)
        assert result["external"].rgb.shape == (300, 300, 3)
        assert data.time == 4.0  # Rendering never advances physics.
    finally:
        producer.stop()
    assert owner_thread[0] == owner_thread[1] != get_ident()
    assert calls[:2] == [("create", 128), ("create", 300)]
    assert calls[-2:] == [("close", 128), ("close", 300)]
    assert snapshots and all(snapshot is snapshots[0] for snapshot in snapshots)
    assert "wrist_cam" in calls and "external" in calls
    producer.reset_session()
    assert reset_calls == [True]
    assert data.time == 0.0
    assert all(buffer.latest_frame is None for buffer in buffers.values())
