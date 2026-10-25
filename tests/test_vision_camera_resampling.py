"""Persistent scenes resample cameras relative to the original mounts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_feed_getter import SIM_CAMERA_NAMES
from environment import CubeStackEnvironment
from schemas.config import load_config
from vision_data import (
    AugmentedRenderer, capture_camera_baseline, randomize_camera_setup, randomize_visual_setup,
)
from vision_materials import configure_vision_materials


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project():
    return load_config(ROOT / "configs/default.json")


def scene(project):
    return CubeStackEnvironment(project.environment.scene_path,
                                configure_spec=configure_vision_materials)


def test_camera_only_sampling_matches_original_full_setup(project):
    full_environment = scene(project)
    camera_environment = scene(project)
    baseline = capture_camera_baseline(camera_environment)
    full_setup = randomize_visual_setup(full_environment, project.environment.vision_data,
                                       np.random.default_rng(71))
    camera_setup = randomize_camera_setup(camera_environment, project.environment.vision_data,
                                         np.random.default_rng(71), baseline=baseline)
    assert set(camera_setup) == set(SIM_CAMERA_NAMES)
    for name in SIM_CAMERA_NAMES:
        assert camera_setup[name] == full_setup[name]
    np.testing.assert_array_equal(camera_environment.model.cam_intrinsic,
                                  full_environment.model.cam_intrinsic)


def test_resampling_does_not_accumulate_mount_or_intrinsic_changes(project):
    environment = scene(project)
    baseline = capture_camera_baseline(environment)
    original_baseline = deepcopy(baseline)
    config = project.environment.vision_data
    first_setup = randomize_camera_setup(environment, config, np.random.default_rng(39),
                                        baseline=baseline)
    first_intrinsics = environment.model.cam_intrinsic.copy()
    for seed in range(30):
        randomize_camera_setup(environment, config, np.random.default_rng(seed),
                               baseline=baseline)
        for name, sim_name in SIM_CAMERA_NAMES.items():
            camera = environment.model.camera(sim_name)
            assert np.all(np.abs(camera.pos - baseline[name]["position"])
                          <= np.asarray(config.camera_position_jitter) + 1e-12)
            angle = 2 * np.arccos(np.clip(
                abs(np.dot(camera.quat, baseline[name]["orientation"])), 0, 1,
            ))
            assert angle <= config.camera_angle_jitter + 1e-12
    repeated = randomize_camera_setup(environment, config, np.random.default_rng(39),
                                     baseline=baseline)
    assert repeated == first_setup
    np.testing.assert_array_equal(environment.model.cam_intrinsic, first_intrinsics)
    for name, fields in baseline.items():
        for field, values in fields.items():
            np.testing.assert_array_equal(values, original_baseline[name][field])


def test_baseline_restores_original_lens_ratios_after_intrinsic_changes(project):
    environment = scene(project)
    baseline = capture_camera_baseline(environment)
    config = project.environment.vision_data
    randomize_camera_setup(environment, config, np.random.default_rng(5), baseline=baseline)
    first_intrinsics = environment.model.cam_intrinsic.copy()
    environment.model.cam_intrinsic[:] *= 3
    randomize_camera_setup(environment, config, np.random.default_rng(5), baseline=baseline)
    np.testing.assert_array_equal(environment.model.cam_intrinsic, first_intrinsics)


def test_camera_resampling_preserves_surfaces_lights_and_physics_state(project):
    environment = scene(project)
    baseline = capture_camera_baseline(environment)
    config = project.environment.vision_data
    randomize_visual_setup(environment, config, np.random.default_rng(12))
    model_fields = (
        "light_pos", "light_diffuse", "light_ambient", "light_castshadow",
        "geom_rgba", "mat_rgba", "mat_specular", "mat_shininess", "mat_reflectance",
        "tex_data", "cam_sensorsize",
    )
    original_model = {name: getattr(environment.model, name).copy() for name in model_fields}
    original_data = {name: getattr(environment.data, name).copy()
                     for name in ("qpos", "qvel", "ctrl")}
    original_time = environment.data.time
    original_headlight = environment.model.vis.headlight.diffuse.copy()
    original_haze = environment.model.vis.rgba.haze.copy()
    randomize_camera_setup(environment, config, np.random.default_rng(14), baseline=baseline)
    for name, value in original_model.items():
        np.testing.assert_array_equal(getattr(environment.model, name), value)
    for name, value in original_data.items():
        np.testing.assert_array_equal(getattr(environment.data, name), value)
    assert environment.data.time == original_time
    np.testing.assert_array_equal(environment.model.vis.headlight.diffuse, original_headlight)
    np.testing.assert_array_equal(environment.model.vis.rgba.haze, original_haze)
    # The model mounts and derived camera positions agree immediately: callers
    # can render the new camera without first advancing physics.
    for sim_name in SIM_CAMERA_NAMES.values():
        camera = environment.model.camera(sim_name)
        body = environment.data.body(environment.model.cam_bodyid[camera.id])
        expected = body.xpos + body.xmat.reshape(3, 3) @ camera.pos
        np.testing.assert_allclose(environment.data.cam_xpos[camera.id], expected, atol=1e-12)


def test_captured_baseline_is_independent_of_model_buffers(project):
    environment = scene(project)
    baseline = capture_camera_baseline(environment)
    snapshot = deepcopy(baseline)
    randomize_visual_setup(environment, project.environment.vision_data, np.random.default_rng(17))
    for name, fields in baseline.items():
        for field, values in fields.items():
            np.testing.assert_array_equal(values, snapshot[name][field])


def test_persistent_renderer_refreshes_lens_maps_and_effects_without_recreation(project):
    config = project.environment.vision_data
    config.focus_blur_scale = 0
    config.motion_blur_probability = 0
    config.exposure_range = (1, 1)
    config.image_noise_std = 0
    config.blur_probability = 0
    setup = {name: {"lens_distortion": 0, "gamma": 1, "white_balance_gains": [1, 1, 1],
                    "focus_distance": .4, "motion_blur_exposure": 0}
             for name in SIM_CAMERA_NAMES}

    class ConstantRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.full(self.shape, 128, dtype=np.uint8)

    renderer = AugmentedRenderer(None, 7, 9, setup, config,
                                 np.random.default_rng(3), ConstantRenderer)
    underlying = renderer.renderer
    renderer.update_scene(SimpleNamespace(time=0), "wrist_cam")
    np.testing.assert_array_equal(renderer.render(), 128)
    original_map = renderer.maps["wrist_cam"][0].copy()
    updated = deepcopy(setup)
    updated["wrist"]["lens_distortion"] = .1
    updated["wrist"]["gamma"] = 2
    updated["external"]["focus_distance"] = .8
    renderer.update_camera_setup(updated)
    assert renderer.renderer is underlying
    assert not np.array_equal(renderer.maps["wrist_cam"][0], original_map)
    assert renderer.parameters["external"]["focus_distance"] == .8
    np.testing.assert_array_equal(renderer.render(), 64)
