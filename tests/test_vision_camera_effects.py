"""Camera color, depth of field, and finite exposure contracts without OpenGL."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from camera_feed_getter import SIM_CAMERA_NAMES
from environment import CubeStackEnvironment
from schemas.config import load_config
from vision_materials import configure_vision_materials
from vision_data import AugmentedRenderer, _apply_focus_blur, randomize_visual_setup


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parameters():
    config = load_config(ROOT / "configs/default.json").environment.vision_data
    config.exposure_range = (1.0, 1.0)
    config.image_noise_std = 0.0
    config.blur_probability = 0.0
    config.focus_blur_scale = 0.0
    config.motion_blur_probability = 0.0
    return config


@pytest.fixture
def moving_scene():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <camera name="external" pos="0 0 1"/>
            <camera name="wrist_cam" pos="0 0 1"/>
            <body name="moving_object">
              <joint name="translation" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.01" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.3
    data.qvel[0] = 10.0
    data.time = 0.5
    mujoco.mj_forward(model, data)
    return model, data


def camera_setup(**overrides):
    return {
        name: {
            "lens_distortion": 0.0,
            "white_balance_gains": [1.0, 1.0, 1.0],
            "gamma": 1.0,
            "focus_distance": 0.4,
            "motion_blur_exposure": 0.02,
            **overrides,
        }
        for name in SIM_CAMERA_NAMES
    }


class ColorRenderer:
    """Record rendered state and expose metric depth like MuJoCo's renderer."""

    def __init__(self, model, height, width):
        self.height, self.width = height, width
        self.depth_enabled = False
        self.frames = []
        self.closed = False

    def update_scene(self, data, camera):
        self.position = float(data.qpos[0])
        self.time = float(data.time)
        self.camera = camera

    def enable_depth_rendering(self):
        self.depth_enabled = True

    def disable_depth_rendering(self):
        self.depth_enabled = False

    def render(self):
        self.frames.append((self.camera, self.position, self.time, self.depth_enabled))
        if self.depth_enabled:
            return np.full((self.height, self.width), 0.4, dtype=np.float32)
        return np.broadcast_to(np.array([32, 96, 192], dtype=np.uint8),
                               (self.height, self.width, 3)).copy()

    def close(self):
        self.closed = True


class MovingStripeRenderer(ColorRenderer):
    def render(self):
        image = super().render()
        if not self.depth_enabled:
            image[:] = 0
            column = int(round(100 * self.position))
            if 0 <= column < self.width:
                image[:, column] = 255
        return image


def make_renderer(model, parameters, setup=None, factory=ColorRenderer):
    return AugmentedRenderer(
        model=model, height=9, width=41,
        setup=camera_setup() if setup is None else setup,
        config=parameters, rng=np.random.default_rng(7), renderer_factory=factory,
    )


def textured_image():
    yy, xx = np.indices((25, 25))
    return np.repeat((((xx + yy) % 2) * 255)[..., None], 3, axis=2).astype(np.float32)


def test_focus_preserves_focused_plane_and_blurs_both_nearer_and_farther_texture():
    image = textured_image()
    focused = _apply_focus_blur(image, np.full(image.shape[:2], 0.4), 0.4, 0.6, 2.0)
    assert focused.dtype == np.float32
    np.testing.assert_array_equal(focused, image)
    for depth in (0.2, 2.0):
        blurred = _apply_focus_blur(image, np.full(image.shape[:2], depth), 0.4, 0.6, 2.0)
        assert blurred.std() < image.std() * 0.6
        assert np.isfinite(blurred).all()
        assert 0 <= blurred.min() <= blurred.max() <= 255


def test_focus_uses_inverse_distance_and_respects_zero_strength():
    image = textured_image()
    nearer = _apply_focus_blur(image, np.full(image.shape[:2], 0.25), 1 / 3, 0.6, 2.0)
    farther = _apply_focus_blur(image, np.full(image.shape[:2], 0.5), 1 / 3, 0.6, 2.0)
    np.testing.assert_allclose(nearer, farther, atol=1e-4)
    for scale, maximum in ((0.0, 2.0), (0.6, 0.0)):
        result = _apply_focus_blur(image, np.full(image.shape[:2], 0.1), 0.4, scale, maximum)
        np.testing.assert_array_equal(result, image)


def test_focus_handles_black_pixels_zero_depth_and_sky_depth():
    depth = np.broadcast_to(np.array([0.0, 0.4, np.inf, 1e10, np.nan, -1.0], dtype=np.float32), (8, 6))
    for image in (np.zeros((8, 6, 3), dtype=np.float32),
                  np.full((8, 6, 3), 127.0, dtype=np.float32)):
        result = _apply_focus_blur(image, depth, 0.4, 0.6, 2.0)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, image, atol=1e-5)


def test_new_camera_settings_are_sampled_independently_and_reproducibly():
    project = load_config(ROOT / "configs/default.json")
    parameters = project.environment.vision_data
    first = randomize_visual_setup(CubeStackEnvironment(project.environment.scene_path,
                                                        configure_spec=configure_vision_materials),
                                   parameters, np.random.default_rng(42))
    repeated = randomize_visual_setup(CubeStackEnvironment(project.environment.scene_path,
                                                        configure_spec=configure_vision_materials),
                                      parameters, np.random.default_rng(42))
    assert first == repeated
    ranges = {
        "gamma": parameters.gamma_range,
        "focus_distance": parameters.focus_distance_range,
        "motion_blur_exposure": parameters.motion_blur_exposure_range,
    }
    for name in SIM_CAMERA_NAMES:
        gains = np.asarray(first[name]["white_balance_gains"])
        assert gains.shape == (3,)
        assert np.all(gains >= parameters.white_balance_gain_range[0])
        assert np.all(gains <= parameters.white_balance_gain_range[1])
        for field, (lower, upper) in ranges.items():
            assert lower <= first[name][field] <= upper
    for field in (*ranges, "white_balance_gains"):
        assert first["wrist"][field] != first["external"][field]


def test_white_balance_and_gamma_are_per_camera_and_stable_across_frames(parameters, moving_scene):
    model, data = moving_scene
    setup = camera_setup()
    setup["wrist"].update(white_balance_gains=[1.5, 1.0, 0.5], gamma=2.0)
    original_setup = deepcopy(setup)
    renderer = make_renderer(model, parameters, setup)
    expected = np.rint(255 * (np.array([48, 96, 96], dtype=np.float32) / 255) ** 2).astype(np.uint8)
    for camera in ("external", "wrist_cam", "external", "wrist_cam"):
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        assert image.dtype == np.uint8
        assert image.shape == (9, 41, 3)
        target = expected if camera == "wrist_cam" else np.array([32, 96, 192])
        np.testing.assert_array_equal(image, np.broadcast_to(target, image.shape))
    assert setup == original_setup
    renderer.close()
    assert renderer.renderer.closed


@pytest.mark.parametrize("camera", ["wrist_cam", "external"])
def test_motion_blur_integrates_moving_geometry_without_advancing_live_state(parameters, moving_scene, camera):
    model, data = moving_scene
    parameters.motion_blur_probability = 1.0
    parameters.motion_blur_samples = 3
    parameters.focus_blur_scale = 0.6
    original = (data.qpos.copy(), data.qvel.copy(), data.qacc.copy(), data.time)
    renderer = make_renderer(model, parameters, factory=MovingStripeRenderer)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    lit_columns = np.flatnonzero(image[4, :, 0])
    np.testing.assert_array_equal(lit_columns, [10, 20, 30])
    np.testing.assert_array_equal(image[:, lit_columns], 85)
    np.testing.assert_array_equal(data.qpos, original[0])
    np.testing.assert_array_equal(data.qvel, original[1])
    np.testing.assert_array_equal(data.qacc, original[2])
    assert data.time == original[3]
    assert renderer.renderer.position == pytest.approx(original[0][0])
    assert not renderer.renderer.depth_enabled
    rgb_frames = [frame[:3] for frame in renderer.renderer.frames if not frame[3]]
    depth_frames = [frame[:3] for frame in renderer.renderer.frames if frame[3]]
    assert len(rgb_frames) == 3
    assert rgb_frames == depth_frames
    assert all(frame[0] == camera for frame in rgb_frames)


@pytest.mark.parametrize("disabled_by", ["static", "probability", "exposure"])
def test_motion_blur_keeps_static_or_disabled_frames_sharp(parameters, moving_scene, disabled_by):
    model, data = moving_scene
    parameters.motion_blur_probability = 1.0
    setup = camera_setup()
    if disabled_by == "static":
        data.qvel[:] = 0
    elif disabled_by == "probability":
        parameters.motion_blur_probability = 0.0
    else:
        for camera in setup.values():
            camera["motion_blur_exposure"] = 0.0
    renderer = make_renderer(model, parameters, setup, MovingStripeRenderer)
    renderer.update_scene(data, camera="external")
    image = renderer.render()
    np.testing.assert_array_equal(np.flatnonzero(image[4, :, 0]), [30])
    np.testing.assert_array_equal(image[:, 30], 255)
    assert len(renderer.renderer.frames) == 1


def test_shutter_interval_does_not_extend_before_start_of_sequence(parameters, moving_scene):
    model, data = moving_scene
    data.time = 0.01
    parameters.motion_blur_probability = 1.0
    parameters.motion_blur_samples = 3
    renderer = make_renderer(model, parameters, factory=MovingStripeRenderer)
    renderer.update_scene(data, camera="external")
    image = renderer.render()
    np.testing.assert_array_equal(np.flatnonzero(image[4, :, 0]), [20, 25, 30])
    np.testing.assert_allclose([frame[2] for frame in renderer.renderer.frames], [0.0, 0.005, 0.01])
    assert data.time == 0.01


def test_depth_failure_restores_rgb_mode_and_current_scene(parameters, moving_scene):
    class FailedDepthRenderer(ColorRenderer):
        def render(self):
            if self.depth_enabled:
                raise RuntimeError("depth acquisition failed")
            return super().render()

    model, data = moving_scene
    parameters.motion_blur_probability = 1.0
    parameters.focus_blur_scale = 0.6
    renderer = make_renderer(model, parameters, factory=FailedDepthRenderer)
    renderer.update_scene(data, camera="external")
    with pytest.raises(RuntimeError, match="depth acquisition failed"):
        renderer.render()
    assert not renderer.renderer.depth_enabled
    assert renderer.renderer.position == pytest.approx(0.3)
    assert renderer.renderer.time == 0.5
    assert data.qpos[0] == pytest.approx(0.3)
    assert data.time == 0.5
