"""Saved vision data, fixed setup augmentation, and asynchronous sample contracts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from camera_feed_getter import SIM_CAMERA_NAMES
from environment import CubeStackEnvironment
from schemas import CameraFrame
from schemas.config import load_config
from vision_data import (
    AugmentedRenderer, collect_sequence, cube_pose_labels, generate_vision_data,
    randomize_cube_poses, randomize_visual_setup,
)
from vision_errors import CameraAcquisitionError


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    project = load_config(ROOT / "configs/default.json")
    project.environment.episode_seconds = 0.15
    project.environment.vision_data.sequences = 2
    project.environment.vision_data.sample_interval_steps = 1
    project.training.vision.data_dir = tmp_path / "vision"
    return project


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class FakeRenderer:
    def __init__(self, model, height, width):
        self.height, self.width = height, width
        self.closed = False
        self.depth_rendering = False

    def update_scene(self, data, camera):
        self.camera = camera

    def render(self):
        if self.depth_rendering:
            return np.full((self.height, self.width), 0.3, dtype=np.float32)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(0, 255, self.width).astype(np.uint8)
        image[..., 1] = 32 if self.camera == "wrist_cam" else 96
        return image

    def enable_depth_rendering(self):
        self.depth_rendering = True

    def disable_depth_rendering(self):
        self.depth_rendering = False

    def close(self):
        self.closed = True


class InitialFramesProducer:
    """Publish once so collection must retain original times on reused frames."""

    instances = []

    def __init__(self, environment, buffers, fps, renderer_factory):
        self.environment, self.buffers = environment, buffers
        self.renderer = renderer_factory(environment.model, height=300, width=300)
        self.fps = fps
        self.stopped = False
        self.instances.append(self)

    def start(self):
        for name, sim_name in SIM_CAMERA_NAMES.items():
            self.renderer.update_scene(self.environment.data, camera=sim_name)
            self.buffers[name].record_read(True, CameraFrame(
                camera_id=name, timestamp=float(self.environment.data.time), rgb=self.renderer.render(),
            ))

    def stop(self):
        self.stopped = True
        self.renderer.close()


def test_additive_saved_sequences_split_whole_setups_and_preserve_latest_timestamps(config):
    config.environment.vision_data.cube_pose_mode_probabilities = (1.0, 0.0, 0.0, 0.0, 0.0)
    clock = FakeClock()
    paths = generate_vision_data(project=config, renderer_factory=FakeRenderer,
                                 producer_factory=InitialFramesProducer, clock=clock, wait=clock.wait)
    assert {path.parent.name for path in paths} == {"train", "validation"}
    initial_bytes = {path: path.read_bytes() for path in paths}
    setups = []
    for path in paths:
        record = torch.load(path, weights_only=True)
        assert record["sequence_id"] == path.stem
        assert record["wrist_images"].shape == record["external_images"].shape == (3, 300, 300, 3)
        assert record["wrist_images"].dtype == torch.uint8
        assert record["joint_positions"].shape == (3, 6)
        assert record["joint_positions"].dtype == torch.float32
        assert record["cube_poses"].shape == (3, 14)
        assert record["cube_poses"].dtype == torch.float32
        assert record["phases"] == [record["cube_pose_mode"]] * 3
        torch.testing.assert_close(record["robot_timestamps"], torch.tensor([0., .05, .10], dtype=torch.float64))
        assert torch.count_nonzero(record["wrist_timestamps"]) == 0
        assert torch.count_nonzero(record["external_timestamps"]) == 0
        setups.append(record["camera_setup"])
    assert setups[0] != setups[1]
    more = generate_vision_data(project=config, renderer_factory=FakeRenderer,
                                producer_factory=InitialFramesProducer, clock=clock, wait=clock.wait)
    assert not set(paths).intersection(more)
    assert len(list(config.training.vision.data_dir.rglob("*.pt"))) == 4
    assert all(path.read_bytes() == initial_bytes[path] for path in paths)
    assert all(producer.stopped for producer in InitialFramesProducer.instances)
    assert all(producer.fps == config.cameras.fps for producer in InitialFramesProducer.instances)


@pytest.mark.parametrize("selected_mode", ["ground", "elevated", "stacked", "task", "carry"])
def test_generation_honors_mode_probabilities_and_excludes_zero_weight_modes(config, monkeypatch, selected_mode):
    parameters = config.environment.vision_data
    parameters.sequences = 6
    parameters.cube_pose_mode_probabilities = tuple(
        1.0 if mode == selected_mode else 0.0 for mode in parameters.cube_pose_modes
    )
    collected_modes = []

    def collect_stub(*, mode, sequence_id, **kwargs):
        collected_modes.append(mode)
        return {
            "sequence_id": sequence_id, "cube_pose_mode": mode,
            "cube_poses": torch.zeros((1, 14)), "final_scenario_metadata": None,
        }

    monkeypatch.setattr("vision_data.collect_sequence", collect_stub)
    paths = generate_vision_data(project=config)

    assert collected_modes == [selected_mode] * parameters.sequences
    assert len(paths) == parameters.sequences
    assert all(torch.load(path, weights_only=True)["cube_pose_mode"] == selected_mode for path in paths)


@pytest.mark.parametrize("mode", ["ground", "elevated", "stacked", "task", "carry"])
def test_cube_modes_and_label_order(config, mode):
    environment = CubeStackEnvironment(config.environment.scene_path)
    randomize_cube_poses(environment, config.environment.vision_data, np.random.default_rng(5), mode)
    labels = cube_pose_labels(environment)
    for index, name in enumerate(("orange", "blue")):
        pose = labels[index * 7:(index + 1) * 7]
        np.testing.assert_allclose(pose[:3], environment.data.body(f"{name}_cube").xpos)
        np.testing.assert_allclose(pose[3:], environment.data.body(f"{name}_cube").xquat)
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.0)
    if mode in ("ground", "task", "carry"):
        assert labels[2] == pytest.approx(.02)
        assert labels[9] == pytest.approx(.02)
        assert np.linalg.norm(labels[:2] - labels[7:9]) >= config.environment.vision_data.minimum_cube_separation
        if mode in ("task", "carry"):
            np.testing.assert_array_equal(labels[3:7], [1, 0, 0, 0])
            np.testing.assert_array_equal(labels[10:14], [1, 0, 0, 0])
    elif mode == "elevated":
        assert .02 <= labels[2] <= .16
        assert .02 <= labels[9] <= .16
    else:
        np.testing.assert_allclose(labels[:2], labels[7:9])
        assert labels[2] - labels[9] == pytest.approx(.04)


def test_setup_ranges_and_relative_mounts_are_fixed_for_sequence(config):
    environment = CubeStackEnvironment(config.environment.scene_path)
    original = {name: environment.model.camera(sim_name).pos.copy()
                for name, sim_name in SIM_CAMERA_NAMES.items()}
    parameters = config.environment.vision_data
    setup = randomize_visual_setup(environment, parameters, np.random.default_rng(10))
    for name, sim_name in SIM_CAMERA_NAMES.items():
        camera = environment.model.camera(sim_name)
        assert np.all(np.abs(camera.pos - original[name]) <= np.asarray(parameters.camera_position_jitter))
        assert parameters.fov_range[0] <= camera.fovy[0] <= parameters.fov_range[1]
        assert parameters.lens_distortion_range[0] <= setup[name]["lens_distortion"] <= parameters.lens_distortion_range[1]
        np.testing.assert_allclose(setup[name]["position"], camera.pos)
        np.testing.assert_allclose(setup[name]["orientation"], camera.quat)
    assert parameters.texture_contrast_range[0] <= setup["texture_contrast"] <= parameters.texture_contrast_range[1]


@pytest.mark.parametrize("target_fovy", [65.0, 100.0])
def test_fov_randomization_changes_rendered_frustum_for_both_cameras(config, target_fovy):
    environment = CubeStackEnvironment(config.environment.scene_path)
    model = environment.model
    original_sensor_sizes = model.cam_sensorsize.copy()
    original_intrinsics = model.cam_intrinsic.copy()
    config.environment.vision_data.fov_range = (target_fovy, target_fovy)
    setup = randomize_visual_setup(environment, config.environment.vision_data, np.random.default_rng(10))

    # Scene construction uses the renderer's camera projection without needing
    # an OpenGL context. Reading cam_fovy alone misses calibrated-lens overrides.
    scene = mujoco.MjvScene(model, maxgeom=1000)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    for name, sim_name in SIM_CAMERA_NAMES.items():
        camera_id = model.camera(sim_name).id
        camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(model, environment.data, mujoco.MjvOption(), None,
                              camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        projection = scene.camera[0]
        rendered_fovy = np.degrees(
            np.arctan(projection.frustum_top / projection.frustum_near)
            - np.arctan(projection.frustum_bottom / projection.frustum_near)
        )
        assert rendered_fovy == pytest.approx(target_fovy, abs=1e-5), name
        assert setup[name]["fovy"] == pytest.approx(rendered_fovy, abs=1e-5)
        if original_sensor_sizes[camera_id, 1] > 0:
            intrinsic = model.cam_intrinsic[camera_id]
            original = original_intrinsics[camera_id]
            assert intrinsic[0] / intrinsic[1] == pytest.approx(original[0] / original[1])
    np.testing.assert_array_equal(model.cam_sensorsize, original_sensor_sizes)
    np.testing.assert_array_equal(model.cam_intrinsic[:, 2:], original_intrinsics[:, 2:])


@pytest.mark.parametrize("jitter", [(0.0, 0.07, 0.02), (0.03, 0.0, 0.0)])
def test_camera_position_jitter_uses_each_mount_axis_and_preserves_zero_axes(config, jitter):
    environment = CubeStackEnvironment(config.environment.scene_path)
    original = {name: environment.model.camera(sim_name).pos.copy()
                for name, sim_name in SIM_CAMERA_NAMES.items()}
    config.environment.vision_data.camera_position_jitter = jitter
    randomize_visual_setup(environment, config.environment.vision_data, np.random.default_rng(10))

    bounds = np.asarray(jitter)
    for name, sim_name in SIM_CAMERA_NAMES.items():
        offset = environment.model.camera(sim_name).pos - original[name]
        assert np.all(np.abs(offset) <= bounds)
        np.testing.assert_array_equal(offset[bounds == 0], 0)
        assert np.all(offset[bounds > 0] != 0)


@pytest.mark.parametrize("jitter", [(0.1, 0.1, 0.1), (0.0, 0.07, 0.02), (0.03, 0.0, 0.0)])
def test_light_position_jitter_respects_each_axis_and_stays_fixed_during_simulation(config, jitter):
    environment = CubeStackEnvironment(config.environment.scene_path)
    model = environment.model
    assert model.nlight > 0
    original = model.light_pos.copy()
    config.environment.vision_data.light_position_jitter = jitter
    randomize_visual_setup(environment, config.environment.vision_data, np.random.default_rng(10))

    bounds = np.asarray(jitter)
    offset = model.light_pos - original
    assert np.all(np.abs(offset) <= bounds)
    np.testing.assert_array_equal(offset[:, bounds == 0], 0)
    assert np.all(offset[:, bounds > 0] != 0)
    light_id = model.light("overhead").id
    direction_to_center = np.array([0.20, 0.0, 0.0]) - environment.data.light_xpos[light_id]
    direction_to_center /= np.linalg.norm(direction_to_center)
    np.testing.assert_allclose(environment.data.light_xdir[light_id], direction_to_center, atol=1e-12)
    sampled_positions = model.light_pos.copy()
    sampled_world_positions = environment.data.light_xpos.copy()
    for _ in range(5):
        mujoco.mj_step(model, environment.data)
        np.testing.assert_array_equal(model.light_pos, sampled_positions)
        np.testing.assert_array_equal(environment.data.light_xpos, sampled_world_positions)
        np.testing.assert_allclose(environment.data.light_xdir[light_id], direction_to_center, atol=1e-12)


@pytest.mark.parametrize("saturation", [0.0, 0.06])
def test_lights_use_independent_intensities_and_share_a_near_white_tint(config, saturation):
    environment = CubeStackEnvironment(config.environment.scene_path)
    parameters = config.environment.vision_data
    parameters.light_intensity_range = (0.7, 0.7)
    parameters.ambient_intensity_range = (0.25, 0.25)
    parameters.light_saturation_range = (saturation, saturation)
    randomize_visual_setup(environment, parameters, np.random.default_rng(21))

    model = environment.model
    lights = list(zip(model.light_diffuse, model.light_ambient))
    lights.append((model.vis.headlight.diffuse, model.vis.headlight.ambient))
    assert model.nlight > 0
    for diffuse, ambient in lights:
        assert max(diffuse) == pytest.approx(0.7)
        assert max(ambient) == pytest.approx(0.25)
        tint = diffuse / 0.7
        np.testing.assert_allclose(ambient / 0.25, tint, atol=1e-7)
        assert max(tint) - min(tint) == pytest.approx(saturation, abs=1e-7)
        assert min(tint) >= 1 - saturation - 1e-7


def test_zero_ambient_intensity_disables_ambient_light(config):
    environment = CubeStackEnvironment(config.environment.scene_path)
    parameters = config.environment.vision_data
    parameters.ambient_intensity_range = (0.0, 0.0)
    randomize_visual_setup(environment, parameters, np.random.default_rng(3))
    np.testing.assert_array_equal(environment.model.light_ambient, 0)
    np.testing.assert_array_equal(environment.model.vis.headlight.ambient, 0)


@pytest.mark.parametrize("enabled", [True, False])
def test_shadow_setting_overrides_every_scene_light(config, enabled):
    environment = CubeStackEnvironment(config.environment.scene_path)
    parameters = config.environment.vision_data
    parameters.shadows_enabled = enabled
    assert environment.model.nlight > 0
    environment.model.light_castshadow[:] = not enabled
    randomize_visual_setup(environment, parameters, np.random.default_rng(4))
    assert np.all(environment.model.light_castshadow == enabled)


def test_identity_lens_and_image_augmentation_preserve_rgb_shape(config):
    parameters = config.environment.vision_data
    parameters.image_noise_std = 0.0
    parameters.blur_probability = 0.0
    parameters.exposure_range = (1.0, 1.0)
    parameters.focus_blur_scale = 0.0
    parameters.motion_blur_probability = 0.0
    setup = {name: {"lens_distortion": 0.0, "white_balance_gains": [1.0, 1.0, 1.0],
                    "gamma": 1.0, "focus_distance": 0.3, "motion_blur_exposure": 0.0}
             for name in SIM_CAMERA_NAMES}
    renderer = AugmentedRenderer(model=None, height=9, width=9, setup=setup, config=parameters,
                                 rng=np.random.default_rng(1), renderer_factory=FakeRenderer)
    renderer.update_scene(None, camera="wrist_cam")
    first = renderer.render()
    maps_before = tuple(mapping.copy() for mapping in renderer.maps["wrist_cam"])
    second = renderer.render()
    assert first.shape == (9, 9, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, renderer.renderer.render())
    np.testing.assert_array_equal(first, second)
    for before, after in zip(maps_before, renderer.maps["wrist_cam"]):
        np.testing.assert_array_equal(before, after)
    renderer.close()
    assert renderer.renderer.closed


def test_missing_initial_frames_times_out_and_closes_worker(config):
    class NoFrames(InitialFramesProducer):
        def start(self):
            pass

    clock = FakeClock()
    with pytest.raises(CameraAcquisitionError, match="Initial camera frames"):
        collect_sequence(project=config, sequence_id="failure", mode="ground", rng=np.random.default_rng(1),
                         renderer_factory=FakeRenderer, producer_factory=NoFrames, clock=clock, wait=clock.wait)
    assert NoFrames.instances[-1].stopped


def test_producer_fatal_error_is_not_retried_as_missing_data(config):
    class FailedProducer(InitialFramesProducer):
        def start(self):
            for buffer in self.buffers.values():
                buffer.record_failure(CameraAcquisitionError("failed initialization"))

    clock = FakeClock()
    with pytest.raises(CameraAcquisitionError, match="failed initialization"):
        collect_sequence(project=config, sequence_id="failure", mode="ground", rng=np.random.default_rng(1),
                         renderer_factory=FakeRenderer, producer_factory=FailedProducer, clock=clock, wait=clock.wait)
    assert clock.now == 0.0
    assert FailedProducer.instances[-1].stopped


@pytest.mark.parametrize("mode, scenario_name", [
    ("task", "ScriptedVisionScenario"), ("carry", "ScriptedCarryScenario"),
])
def test_scripted_modes_record_observed_phases_and_never_use_random_commands(config, monkeypatch,
                                                                          mode, scenario_name):
    instances = []
    rng = np.random.default_rng(1)

    class Scenario:
        def __init__(self, environment, backend, controller_config, config, **kwargs):
            self.environment, self.backend = environment, backend
            self.controller_config, self.config = controller_config, config
            self.extra_arguments = kwargs
            self.phase = "approach"
            self.steps = 0
            self.commands = []
            instances.append(self)

        def metadata(self):
            return {"phase": self.phase, "ever_grasped": False, "successful_stack": False}

        def step(self):
            targets = np.array([0., 0., 0., 0., 0., 1.])
            self.commands.append(self.backend.send_joint_targets(targets=targets))
            self.steps += 1
            self.phase = ("approach", "descend", "grasp", "carry")[self.steps]

    monkeypatch.setattr(f"vision_data.{scenario_name}", Scenario)
    config.environment.vision_data.joint_step_std = 1e6  # Must not affect scripted commands.
    clock = FakeClock()
    record = collect_sequence(project=config, sequence_id=f"{mode}-example", mode=mode, rng=rng,
                              renderer_factory=FakeRenderer, producer_factory=InitialFramesProducer,
                              clock=clock, wait=clock.wait)
    scenario = instances[0]
    assert scenario.config is config.environment.vision_data.task
    assert scenario.controller_config is config.cartesian_controller
    if mode == "carry":
        assert scenario.extra_arguments["carry_config"] is config.environment.vision_data.carry
        assert scenario.extra_arguments["rng"] is rng
    else:
        assert scenario.extra_arguments == {}
    assert scenario.steps == 3
    assert scenario.environment.data.time == pytest.approx(.15)
    assert record["phases"] == ["approach", "descend", "grasp"]
    assert [item["phase"] for item in record["scenario_metadata"]] == record["phases"]
    assert not any(item["ever_grasped"] for item in record["scenario_metadata"])
    assert record["final_scenario_metadata"]["phase"] == "carry"
    assert record["final_scenario_metadata"]["successful_stack"] is False
    assert "actions" not in record  # These are pose-labelled images, not policy supervision.
    np.testing.assert_array_equal(scenario.environment.data.ctrl, scenario.commands[-1].joint_targets)


def test_invalid_separation_is_reported_before_producing_frames(config):
    config.environment.vision_data.cube_x_range = (.3, .3)
    config.environment.vision_data.cube_y_range = (0., 0.)
    environment = CubeStackEnvironment(config.environment.scene_path)
    with pytest.raises(ValueError, match="minimum_cube_separation"):
        randomize_cube_poses(environment, config.environment.vision_data, np.random.default_rng(1), "task")
