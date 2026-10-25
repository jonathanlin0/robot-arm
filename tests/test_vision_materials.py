"""Material domain randomization contracts, checked without an OpenGL context."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment import CubeStackEnvironment
from schemas.config import load_config
from vision_data import collect_sequence, randomize_visual_setup
from vision_materials import (
    PAPER_GEOM_NAMES,
    SURFACE_GEOM_NAMES,
    configure_vision_materials,
    draw_clipped_val_from_gaussian,
    randomize_materials,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes/so101_two_cube_stack.xml"
SETTINGS = ("specular", "shininess", "reflectance")


class ScriptedNormal:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def normal(self, mean, scale):
        self.calls.append((mean, scale))
        return next(self.values)


@pytest.mark.parametrize("accepted", [0.0, 0.43, 1.0])
def test_gaussian_rejects_outside_values_accepts_inclusive_bounds_and_uses_variance(accepted):
    rng = ScriptedNormal([-0.01, 1.01, accepted])
    actual = draw_clipped_val_from_gaussian(0.65, 0.55, 0.0, 1.0, rng)
    assert actual == accepted
    assert len(rng.calls) == 3  # Reject and redraw; do not clamp to an endpoint.
    for mean, scale in rng.calls:
        assert mean == 0.65
        assert scale == pytest.approx(np.sqrt(0.55))


@pytest.mark.parametrize("mean,variance,lower,upper", [
    (0.65, -1.0, 0.0, 1.0),
    (0.65, 0.55, 1.0, 0.0),
    (0.65, 0.55, 0.5, 0.5),
    (1.1, 0.0, 0.0, 1.0),
    (-0.1, 0.0, 0.0, 1.0),
    (float("nan"), 0.55, 0.0, 1.0),
    (0.65, float("inf"), 0.0, 1.0),
    (0.65, 0.55, float("-inf"), 1.0),
    (0.65, 0.55, 0.0, float("nan")),
])
def test_invalid_or_impossible_distributions_fail_before_sampling(mean, variance, lower, upper):
    rng = ScriptedNormal([])
    with pytest.raises(ValueError):
        draw_clipped_val_from_gaussian(mean, variance, lower, upper, rng)
    assert rng.calls == []


@pytest.mark.parametrize("mean,lower,upper", [(0.65, 0.0, 1.0), (0.4, 0.4, 0.4), (0.0, 0.0, 1.0)])
def test_zero_variance_returns_valid_constant_without_drawing(mean, lower, upper):
    rng = ScriptedNormal([])
    assert draw_clipped_val_from_gaussian(mean, 0.0, lower, upper, rng) == mean
    assert rng.calls == []


def configured_environment():
    return CubeStackEnvironment(SCENE, configure_spec=configure_vision_materials)


def arm_material_names(model):
    arm_geoms = model.body_rootid[model.geom_bodyid] == model.body("base").id
    return [model.material(material_id).name for material_id in sorted(set(model.geom_matid[arm_geoms]))
            if material_id >= 0 and model.material(material_id).name]


def scene_geoms(environment):
    scene = mujoco.MjvScene(environment.model, maxgeom=environment.model.ngeom * 2)
    mujoco.mjv_updateScene(environment.model, environment.data, mujoco.MjvOption(), None,
                          mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    return {geom.objid: geom for geom in scene.geoms[:scene.ngeom]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM}


def test_material_creation_preserves_xml_physics_and_effective_colors():
    source_paths = [SCENE, ROOT / "models/so101/so101.xml"]
    source_bytes = {path: path.read_bytes() for path in source_paths}
    original = CubeStackEnvironment(SCENE)
    augmented = configured_environment()
    before, after = original.model, augmented.model
    assert after.nmat == before.nmat + len(SURFACE_GEOM_NAMES)
    assert after.ngeom == before.ngeom
    for name in ("geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat",
                 "geom_friction", "geom_contype", "geom_conaffinity", "body_mass",
                 "body_inertia", "jnt_type", "jnt_range", "actuator_gainprm"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    np.testing.assert_array_equal(augmented.data.qpos, original.data.qpos)
    before_geoms, after_geoms = scene_geoms(original), scene_geoms(augmented)
    ids = []
    for geom_name in SURFACE_GEOM_NAMES:
        geom = after.geom(geom_name)
        material_id = int(geom.matid[0])
        ids.append(material_id)
        assert material_id >= before.nmat
        assert after.material(material_id).name == f"vision_{geom_name}_material"
        np.testing.assert_allclose(after.mat_rgba[material_id], before.geom(geom_name).rgba)
        np.testing.assert_allclose(after_geoms[geom.id].rgba, before_geoms[geom.id].rgba)
        if geom_name in PAPER_GEOM_NAMES:
            baseline_material = int(before.geom(geom_name).matid[0])
            assert all(getattr(before, f"mat_{name}")[baseline_material] == 0.0 for name in SETTINGS)
    assert len(set(ids)) == len(SURFACE_GEOM_NAMES)
    for path in source_paths:
        assert path.read_bytes() == source_bytes[path]
    # An ordinary runtime environment continues to use the unmodified scene.
    assert CubeStackEnvironment(SCENE).model.nmat == before.nmat


def test_arm_and_all_surface_materials_vary_independently():
    parameters = load_config(ROOT / "configs/default.json").environment.vision_data
    baseline = CubeStackEnvironment(SCENE).model
    arm_names = arm_material_names(baseline)

    def configure_with_unused_material(spec):
        configure_vision_materials(spec)
        spec.add_material(name="unused_material", specular=0.23, shininess=0.34, reflectance=0.45)

    environment = CubeStackEnvironment(SCENE, configure_spec=configure_with_unused_material)
    model = environment.model
    arm_ids = [model.material(name).id for name in arm_names]
    # Nonzero sentinels prove the arm's reflectance is preserved, not just set to zero.
    model.mat_reflectance[arm_ids] = np.linspace(0.1, 0.7, len(arm_ids))
    original_reflectance = model.mat_reflectance[arm_ids].copy()
    metadata = randomize_materials(model, parameters, np.random.default_rng(11))
    surface_names = [f"vision_{name}_material" for name in SURFACE_GEOM_NAMES]
    assert set(metadata) == set(arm_names + surface_names)
    np.testing.assert_array_equal(model.mat_reflectance[arm_ids], original_reflectance)
    for material_name, settings in metadata.items():
        assert set(settings) == set(SETTINGS if material_name in surface_names else SETTINGS[:2])
        material_id = model.material(material_name).id
        for setting, value in settings.items():
            assert 0.0 <= value <= 1.0
            assert getattr(model, f"mat_{setting}")[material_id] == pytest.approx(value)
    for setting in SETTINGS:
        values = [item[setting] for item in metadata.values() if setting in item]
        assert len(set(values)) == len(values)
    unused_id = model.material("unused_material").id
    for setting, expected in zip(SETTINGS, (0.23, 0.34, 0.45), strict=True):
        assert getattr(model, f"mat_{setting}")[unused_id] == pytest.approx(expected)
    # Verify the renderer's scene receives the settings through each geom's material.
    visible = scene_geoms(environment)
    for geom_name in SURFACE_GEOM_NAMES:
        geom = model.geom(geom_name)
        settings = metadata[model.material(int(geom.matid[0])).name]
        for setting, expected in settings.items():
            assert getattr(visible[geom.id], setting) == pytest.approx(expected)


def test_visual_setup_material_metadata_is_seeded_and_retains_wall_and_cube_colors():
    parameters = load_config(ROOT / "configs/default.json").environment.vision_data
    environments = [configured_environment() for _ in range(3)]
    setups = [randomize_visual_setup(environment, parameters, np.random.default_rng(seed))
              for environment, seed in zip(environments, (7, 7, 8), strict=True)]
    assert setups[0]["materials"] == setups[1]["materials"]
    assert setups[0]["materials"] != setups[2]["materials"]
    json.dumps(setups[0]["materials"])
    baseline = CubeStackEnvironment(SCENE).model
    for environment in environments:
        visible = scene_geoms(environment)
        for geom_name in SURFACE_GEOM_NAMES:
            geom = environment.model.geom(geom_name)
            expected = (baseline.geom(geom_name).rgba if "cube" in geom_name else geom.rgba)
            np.testing.assert_allclose(visible[geom.id].rgba, expected)


def test_generation_records_material_settings_once_and_holds_them_through_rollout():
    project = load_config(ROOT / "configs/default.json")
    project.environment.episode_seconds = 0.15
    parameters = project.environment.vision_data
    parameters.samples_per_sequence = 3
    parameters.sample_interval_steps = 1
    parameters.focus_blur_scale = 0.0
    parameters.motion_blur_probability = 0.0
    expected_settings = dict(zip(SETTINGS, (0.3, 0.4, 0.5), strict=True))
    for name, value in expected_settings.items():
        setattr(parameters, f"{name}_mean", value)
        setattr(parameters, f"{name}_variance", 0.0)
    renderers = []

    class MaterialRecordingRenderer:
        def __init__(self, model, height, width):
            self.model = model
            self.shape = (height, width, 3)
            self.frames = []
            self.closed = False
            renderers.append(self)

        def update_scene(self, data, camera):
            self.frames.append((float(data.time), camera,
                                [getattr(self.model, f"mat_{name}").copy() for name in SETTINGS]))

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            self.closed = True

    record = collect_sequence(project=project, sequence_id="materials", mode="ground",
                              rng=np.random.default_rng(27), renderer_factory=MaterialRecordingRenderer)
    assert len(renderers) == 2
    assert all(renderer.closed for renderer in renderers)
    assert all(len(renderer.frames) == 3 for renderer in renderers)
    assert all(len({time for time, _, _ in renderer.frames}) == 3 for renderer in renderers)
    renderer = renderers[0]
    frames = [frame for instance in renderers for frame in instance.frames]
    metadata = record["camera_setup"]["materials"]
    assert len(metadata) == len(arm_material_names(CubeStackEnvironment(SCENE).model)) + len(SURFACE_GEOM_NAMES)
    for geom_name in SURFACE_GEOM_NAMES:
        material_id = int(renderer.model.geom(geom_name).matid[0])
        assert metadata[renderer.model.material(material_id).name] == pytest.approx(expected_settings)
    for _, _, arrays in frames:
        for first, actual in zip(renderer.frames[0][2], arrays, strict=True):
            np.testing.assert_array_equal(actual, first)
        for material_name, settings in metadata.items():
            material_id = renderer.model.material(material_name).id
            for setting, value in settings.items():
                assert arrays[SETTINGS.index(setting)][material_id] == pytest.approx(value)
    # Generation cannot alter the separate scene used for validation.
    baseline = CubeStackEnvironment(SCENE).model
    for geom_name in PAPER_GEOM_NAMES:
        material_id = int(baseline.geom(geom_name).matid[0])
        assert all(getattr(baseline, f"mat_{name}")[material_id] == 0.0 for name in SETTINGS)
