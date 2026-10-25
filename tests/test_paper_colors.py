"""Construction-paper colors stay near scene-defined bases throughout a sequence."""

from colorsys import hsv_to_rgb, rgb_to_hsv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from action_evaluation import evaluate_action_rollouts
from environment import CubeStackEnvironment
from schemas.config import PaperColorVariationConfig, load_config
from test_action_evaluation import FakeRenderer, ImagePoseModel, ObservingActionModel
from vision_data import collect_sequence, randomize_visual_setup
from vision_materials import PAPER_GEOM_NAMES, configure_vision_materials, sample_paper_color


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project():
    return load_config(ROOT / "configs/default.json")


def source_colors(project):
    model = CubeStackEnvironment(project.environment.scene_path).model
    return {name: model.geom(name).rgba[:3].copy() for name in PAPER_GEOM_NAMES}


def test_sample_uses_centered_triangular_draws_and_metadata_reproduces_color(project):
    settings = project.environment.vision_data.paper_color_variation
    values = [0.005, -0.04, 0.06, -0.01]

    class ScriptedTriangular:
        def __init__(self):
            self.calls = []

        def triangular(self, lower, mode, upper):
            self.calls.append((lower, mode, upper))
            return values[len(self.calls) - 1]

    rng = ScriptedTriangular()
    base = np.array([0.3, 0.6, 0.4], dtype=np.float32)
    actual, variations = sample_paper_color(base, settings, rng)
    assert rng.calls == [(-maximum, 0.0, maximum) for maximum in settings.model_dump().values()]
    assert variations == {"hue_shift": 0.005, "saturation_scale": 0.96,
                          "brightness_scale": 1.06, "warmth": -0.01}
    hue, saturation, brightness = rgb_to_hsv(*base.astype(float))
    expected = np.array(hsv_to_rgb(hue + 0.005, saturation * 0.96, brightness * 1.06))
    expected *= [0.99, 1.0, 1.01]
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    json.dumps(variations)


def test_zero_variation_preserves_exact_source_color_without_drawing(project):
    settings = PaperColorVariationConfig(hue_shift=0.0, saturation_scale=0.0,
                                         brightness_scale=0.0, warmth=0.0)

    class NoDraws:
        def triangular(self, *args):
            pytest.fail("Disabled variation must not draw random values.")

    for base in source_colors(project).values():
        original = base.copy()
        rgb, variations = sample_paper_color(base, settings, NoDraws())
        np.testing.assert_array_equal(rgb, base)
        np.testing.assert_array_equal(base, original)
        assert rgb.dtype == np.float32 and not np.shares_memory(rgb, base)
        assert variations == {"hue_shift": 0.0, "saturation_scale": 1.0,
                              "brightness_scale": 1.0, "warmth": 0.0}


def test_default_variation_stays_in_paper_color_families_and_is_seeded(project):
    settings = project.environment.vision_data.paper_color_variation
    bases = source_colors(project)
    np.testing.assert_array_equal(bases["wall_x_min"], bases["wall_x_max"])
    np.testing.assert_array_equal(bases["wall_y_min"], bases["wall_y_max"])
    red, green, brown = bases["wall_x_min"], bases["wall_y_min"], bases["table"]
    assert red[0] > max(red[1:]) and green[1] > max(green[::2])
    assert brown[0] > brown[1] > brown[2]

    def sample(seed):
        rng = np.random.default_rng(seed)
        return {name: [sample_paper_color(base, settings, rng) for _ in range(250)]
                for name, base in bases.items()}

    first, repeated, different = sample(10), sample(10), sample(11)
    for name, draws in first.items():
        colors = np.array([color for color, _ in draws])
        np.testing.assert_array_equal(colors, [color for color, _ in repeated[name]])
        assert not np.array_equal(colors, [color for color, _ in different[name]])
        assert np.all((colors >= 0.0) & (colors <= 1.0))
        assert len(np.unique(colors, axis=0)) == len(colors)
        if name.startswith("wall_y"):
            assert np.all(colors[:, 1] > colors[:, 0]) and np.all(colors[:, 1] > colors[:, 2])
        elif name.startswith("wall_x"):
            assert np.all(colors[:, 0] > colors[:, 1]) and np.all(colors[:, 0] > colors[:, 2])
        else:
            assert np.all(colors[:, 0] > colors[:, 1]) and np.all(colors[:, 1] > colors[:, 2])
        for _, variations in draws:
            for key, maximum in settings.model_dump().items():
                deviation = variations[key] - (1.0 if key.endswith("_scale") else 0.0)
                assert abs(deviation) <= maximum
        # A triangular distribution concentrates draws toward the source color.
        deviations = np.array([variation["hue_shift"] for _, variation in draws])
        assert abs(deviations.mean()) < settings.hue_shift * 0.1
        assert np.mean(abs(deviations) < settings.hue_shift / 2) > 0.65


def test_visual_setup_reads_custom_scene_colors_and_records_actual_float32_values(project):
    environment = CubeStackEnvironment(project.environment.scene_path, configure_spec=configure_vision_materials)
    base_model = CubeStackEnvironment(project.environment.scene_path).model
    # Reading the fresh model, rather than a hardcoded palette, supports alternate scenes.
    custom = np.array([0.2, 0.35, 0.65], dtype=np.float32)
    environment.model.geom("table").rgba[:3] = custom
    expected_bases = {name: environment.model.geom(name).rgba[:3].copy() for name in PAPER_GEOM_NAMES}
    metadata = randomize_visual_setup(environment, project.environment.vision_data, np.random.default_rng(22))
    assert set(metadata["paper_colors"]) == set(PAPER_GEOM_NAMES)
    json.dumps(metadata)
    for name, record in metadata["paper_colors"].items():
        np.testing.assert_array_equal(record["base_rgb"], expected_bases[name])
        np.testing.assert_array_equal(record["rgb"], environment.model.geom(name).rgba[:3])
    for name in ("orange_cube_geom", "blue_cube_geom"):
        np.testing.assert_array_equal(environment.model.geom(name).rgba, base_model.geom(name).rgba)


def test_paper_metadata_matches_all_frames_and_is_fixed_through_sequence(project):
    project.environment.episode_seconds = 0.15
    config = project.environment.vision_data
    config.samples_per_sequence = 3
    config.sample_interval_steps = 1
    config.focus_blur_scale = 0.0
    config.motion_blur_probability = 0.0
    renderers = []

    class RecordingRenderer:
        def __init__(self, model, *, height, width):
            self.model = model
            self.shape = (height, width, 3)
            self.snapshots = []
            self.closed = False
            renderers.append(self)

        def update_scene(self, data, camera):
            self.snapshots.append({name: self.model.geom(name).rgba[:3].copy() for name in PAPER_GEOM_NAMES})

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            self.closed = True

    record = collect_sequence(project, "paper-colors", "ground", np.random.default_rng(18), RecordingRenderer)
    assert len(renderers) == 2 and all(renderer.closed for renderer in renderers)
    assert all(len(renderer.snapshots) == 3 for renderer in renderers)
    metadata = record["camera_setup"]["paper_colors"]
    for renderer in renderers:
        for snapshot in renderer.snapshots:
            for name, color in snapshot.items():
                np.testing.assert_array_equal(color, metadata[name]["rgb"])
    for name, base in source_colors(project).items():
        np.testing.assert_array_equal(metadata[name]["base_rgb"], base)


def test_action_validation_uses_fixed_scene_paper_colors_and_materials(project):
    project.environment.episode_seconds = 0.1
    project.training.action.rollout_validation.episodes = 2
    for name, value in (("specular", 0.3), ("shininess", 0.4), ("reflectance", 0.5)):
        setattr(project.environment.vision_data, f"{name}_mean", value)
        setattr(project.environment.vision_data, f"{name}_variance", 0.0)
    expected = source_colors(project)
    snapshots = []

    def observe(environment, frame):
        snapshots.append({name: environment.model.geom(name).rgba[:3].copy() for name in PAPER_GEOM_NAMES})
        for name in PAPER_GEOM_NAMES:
            material_id = int(environment.model.geom(name).matid[0])
            for setting, value in (("specular", 0.2), ("shininess", 0.1), ("reflectance", 0.0)):
                assert getattr(environment.model, f"mat_{setting}")[material_id] == pytest.approx(value)

    evaluate_action_rollouts(project=project, action_policy=ObservingActionModel(), vision_model=ImagePoseModel(),
                             device=torch.device("cpu"), renderer_factory=FakeRenderer, on_frame=observe)
    assert len(snapshots) == 6
    for snapshot in snapshots:
        for name, rgb in snapshot.items():
            np.testing.assert_array_equal(rgb, expected[name])
