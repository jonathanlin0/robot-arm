"""Generate paired wrist 128x128 and external 300x300 images with varied setups.

Physics scenes combine varied placements, random arm motion, and optional
scripted task attempts. Their phase metadata does not imply a successful grasp
or stack. Ground-truth labels stay in this training-data module; the runtime
perception provider always uses images.
"""

from collections.abc import Callable
from colorsys import hsv_to_rgb
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from camera_feed_getter import SIM_CAMERA_NAMES
from cube_yaw import CubeYawEncoder
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from robot_state_provider import SimRobotStateProvider
from schemas.arrays import CAMERA_IMAGE_SHAPES
from schemas.config import ProjectConfig, VisionDataGenerationConfig
from vision_materials import (
    PAPER_GEOM_NAMES, SURFACE_GEOM_NAMES, configure_vision_materials, randomize_materials,
    sample_paper_color,
)
from vision_scenarios import ScriptedCarryScenario, ScriptedVisionScenario


def _quaternion(rng: np.random.Generator, yaw_only: bool) -> np.ndarray:
    if yaw_only:
        angle = rng.uniform(-np.pi, np.pi)
        return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    quat = rng.normal(size=4)
    return quat / np.linalg.norm(quat)


def randomize_cube_poses(environment: CubeStackEnvironment, config: VisionDataGenerationConfig,
                         rng: np.random.Generator, mode: str) -> None:
    """Place resting, falling, or initially stacked cubes, then let physics run."""
    if mode not in config.cube_pose_modes:
        raise ValueError(f"Cube pose mode {mode!r} is not enabled in the config.")
    with environment.state_lock:
        # Separate initial cubes so the teacher begins with an approachable cube,
        # and baseline scenes do not start with interpenetrating free bodies.
        if mode == "stacked":
            xy = np.array([[rng.uniform(*config.cube_x_range), rng.uniform(*config.cube_y_range)]] * 2)
        else:
            extent = np.array([np.ptp(config.cube_x_range), np.ptp(config.cube_y_range)])
            if np.linalg.norm(extent) <= config.minimum_cube_separation:
                raise ValueError("Cube spawn ranges cannot accommodate minimum_cube_separation.")
            for _ in range(100):
                xy = np.column_stack((rng.uniform(*config.cube_x_range, size=2),
                                      rng.uniform(*config.cube_y_range, size=2)))
                if np.linalg.norm(xy[0] - xy[1]) >= config.minimum_cube_separation:
                    break
            else:
                raise ValueError("Could not sample separated cubes; enlarge spawn ranges or reduce separation.")
        for index, name in enumerate(("blue", "orange")):
            half_size = environment.model.geom(f"{name}_cube_geom").size.copy()
            orientation = (np.array([1.0, 0.0, 0.0, 0.0]) if mode in ("task", "carry")
                           else _quaternion(rng=rng, yaw_only=mode != "elevated"))
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, orientation)
            support_height = np.abs(rotation.reshape(3, 3)[2]) @ half_size
            position = np.array([*xy[index], support_height])
            if mode == "elevated":
                position[2] = max(support_height, rng.uniform(*config.cube_height_range))
            if mode == "stacked" and name == "orange":
                position[2] += 2 * environment.model.geom("blue_cube_geom").size[2]
            joint = environment.data.joint(f"{name}_cube_joint")
            joint.qpos[:] = np.concatenate((position, orientation))
            joint.qvel[:] = 0.0
        mujoco.mj_forward(environment.model, environment.data)


def _sample_light_colors(config: VisionDataGenerationConfig,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Use one near-white tint with separate diffuse and ambient brightness."""
    tint = np.array(hsv_to_rgb(rng.uniform(), rng.uniform(*config.light_saturation_range), 1.0))
    diffuse = tint * rng.uniform(*config.light_intensity_range)
    ambient = tint * rng.uniform(*config.ambient_intensity_range)
    return diffuse, ambient


def capture_camera_baseline(environment: CubeStackEnvironment) -> dict[str, dict[str, np.ndarray]]:
    """Copy original camera mounts/intrinsics before any visual randomization.

    Keep this baseline for the lifetime of a scene when resampling its cameras
    repeatedly. Offsets remain relative to the original mounts, not the last
    sampled camera positions and orientations.
    """
    with environment.state_lock:
        return {
            name: {
                "position": environment.model.camera(sim_name).pos.copy(),
                "orientation": environment.model.camera(sim_name).quat.copy(),
                "intrinsics": environment.model.cam_intrinsic[
                    environment.model.camera(sim_name).id
                ].copy(),
            }
            for name, sim_name in SIM_CAMERA_NAMES.items()
        }


def _randomize_camera_setup(environment: CubeStackEnvironment, config: VisionDataGenerationConfig,
                            rng: np.random.Generator,
                            baseline: dict[str, dict[str, np.ndarray]] | None) -> dict[str, Any]:
    """Mutate cameras under the caller's state lock, preserving sampling order."""
    model = environment.model
    setup = {}
    position_jitter = np.asarray(config.camera_position_jitter)
    for name, sim_name in SIM_CAMERA_NAMES.items():
        camera = model.camera(sim_name)
        if baseline is not None:
            camera.pos[:] = baseline[name]["position"]
            camera.quat[:] = baseline[name]["orientation"]
            model.cam_intrinsic[camera.id] = baseline[name]["intrinsics"]
        # Offsets are relative to each camera's parent body (the wrist
        # mount follows the arm; the external mount is in world axes).
        camera.pos[:] += rng.uniform(-position_jitter, position_jitter)
        # A local axis-angle perturbation preserves wrist mounting semantics.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-config.camera_angle_jitter, config.camera_angle_jitter)
        delta = np.concatenate(([np.cos(angle / 2)], axis * np.sin(angle / 2)))
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, camera.quat.copy(), delta)
        camera.quat[:] = rotated
        camera.fovy[0] = rng.uniform(*config.fov_range)
        sensor_height = model.cam_sensorsize[camera.id, 1]
        if sensor_height > 0:
            # Explicit intrinsics take precedence over fovy in MuJoCo.
            # Scale both focal lengths to keep the lens's X/Y ratio.
            focal_lengths = model.cam_intrinsic[camera.id, :2]
            focal_y = sensor_height / (2 * np.tan(np.deg2rad(camera.fovy[0]) / 2))
            focal_lengths *= focal_y / focal_lengths[1]
        setup[name] = {
            "position": camera.pos.tolist(), "orientation": camera.quat.tolist(),
            "fovy": float(camera.fovy[0]),
            "lens_distortion": float(rng.uniform(*config.lens_distortion_range)),
            "white_balance_gains": rng.uniform(*config.white_balance_gain_range, size=3).tolist(),
            "gamma": float(rng.uniform(*config.gamma_range)),
            "focus_distance": float(rng.uniform(*config.focus_distance_range)),
            "motion_blur_exposure": float(rng.uniform(*config.motion_blur_exposure_range)),
        }
    return setup


def randomize_camera_setup(environment: CubeStackEnvironment, config: VisionDataGenerationConfig,
                           rng: np.random.Generator, *,
                           baseline: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """Resample camera mounts, lenses, and image effects without changing surfaces.

    Supply a baseline captured on the fresh scene for repeated calls. Without a
    baseline, offsets are relative to the current mounts, as for a one-time
    visual setup. The returned settings can configure ``AugmentedRenderer``.
    """
    with environment.state_lock:
        setup = _randomize_camera_setup(environment, config, rng, baseline)
        mujoco.mj_forward(environment.model, environment.data)
    return setup


def randomize_visual_setup(environment: CubeStackEnvironment, config: VisionDataGenerationConfig,
                           rng: np.random.Generator) -> dict[str, Any]:
    """Set mounts, lenses, lights and materials once on a fresh sequence scene.

    The scene's original geom colors supply the paper palette; this function
    must be called once per newly constructed environment, never per frame.
    """
    model = environment.model
    light_position_jitter = np.asarray(config.light_position_jitter)
    with environment.state_lock:
        setup = _randomize_camera_setup(environment, config, rng, baseline=None)
        for light in range(model.nlight):
            # The scene's targetbody mode re-aims the overhead spotlight at
            # the tabletop center when mj_forward applies this position.
            model.light_pos[light] += rng.uniform(-light_position_jitter, light_position_jitter)
            diffuse, ambient = _sample_light_colors(config=config, rng=rng)
            model.light_diffuse[light] = diffuse
            model.light_ambient[light] = ambient
            model.light_castshadow[light] = config.shadows_enabled
        diffuse, ambient = _sample_light_colors(config=config, rng=rng)
        model.vis.headlight.diffuse[:] = diffuse
        model.vis.headlight.ambient[:] = ambient
        model.vis.rgba.haze[:3] = rng.uniform(size=3)
        setup["paper_colors"] = {}
        for name in PAPER_GEOM_NAMES:
            geom = model.geom(name)
            base_rgb = geom.rgba[:3].copy()
            rgb, variations = sample_paper_color(base_rgb, config.paper_color_variation, rng)
            geom.rgba[:3] = rgb
            setup["paper_colors"][name] = {
                "base_rgb": base_rgb.tolist(), "rgb": geom.rgba[:3].tolist(),
                "variations": variations,
            }
        contrast = rng.uniform(*config.texture_contrast_range)
        # Randomize the scene's skybox texture and non-cube material contrast.
        # Do this before renderer creation so its GPU textures match the model.
        texture = model.tex_data.astype(np.float32)
        texture += rng.normal(0.0, 255.0 * contrast, size=texture.shape)
        model.tex_data[:] = np.clip(texture, 0, 255).astype(np.uint8)
        # Preserve the new surface materials' colors; cube identity colors and
        # independently sampled wall/table colors remain on their geoms.
        material_mask = np.ones(model.nmat, dtype=bool)
        for name in SURFACE_GEOM_NAMES:
            material_id = int(model.geom(name).matid[0])
            if material_id >= 0:
                material_mask[material_id] = False
        model.mat_rgba[material_mask, :3] = np.clip(
            model.mat_rgba[material_mask, :3]
            * rng.uniform(1 - contrast, 1 + contrast, size=(int(material_mask.sum()), 1)),
            0, 1,
        )
        setup["materials"] = randomize_materials(model=model, config=config, rng=rng)
        mujoco.mj_forward(model, environment.data)
        setup["texture_contrast"] = float(contrast)
    return setup


def cube_pose_labels(environment: CubeStackEnvironment,
                     encoders: dict[str, CubeYawEncoder] | None = None) -> np.ndarray:
    """Read XYZ/yaw labels, retaining each cube's previous face only for ties.

    A collector supplies encoders for its whole sequence. A standalone call
    starts fresh and resolves initial ties using deterministic world geometry.
    """
    if encoders is None:
        encoders = {name: CubeYawEncoder() for name in ("orange", "blue")}
    with environment.state_lock:
        return np.concatenate([
            component.copy()
            for name in ("orange", "blue")
            for component in (environment.data.body(f"{name}_cube").xpos,
                              encoders[name].encode(environment.data.body(f"{name}_cube").xquat))
        ]).astype(np.float32)


def _apply_focus_blur(image: np.ndarray, depth: np.ndarray, focus_distance: float,
                      scale: float, max_sigma: float) -> np.ndarray:
    """Approximate defocus using metric depth and interpolated Gaussian levels.

    Focus distance is measured along the camera's optical axis. This models
    distance-dependent softness, not a full aperture/bokeh simulation.
    """
    image = image.astype(np.float32)
    if scale == 0 or max_sigma == 0:
        return image
    depth = np.asarray(depth, dtype=np.float32)
    inverse_depth = np.zeros_like(depth)
    np.divide(1.0, depth, out=inverse_depth, where=np.isfinite(depth) & (depth > 0))
    sigma = np.clip(scale * np.abs(inverse_depth - np.float32(1.0 / focus_distance)),
                    0.0, max_sigma)
    # Interpolating five levels avoids a separate convolution at every pixel.
    level = sigma * (4.0 / max_sigma)
    result = np.zeros_like(image)
    for index, blur_sigma in enumerate(np.linspace(0.0, max_sigma, 5)):
        weight = np.maximum(0.0, 1.0 - np.abs(level - index))
        if not np.any(weight):
            continue
        blurred = image if index == 0 else cv2.GaussianBlur(
            image, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
        result += blurred * weight[..., None]
    return result


class AugmentedRenderer:
    """Apply fixed camera settings and short shutter exposures to copied scenes."""

    def __init__(self, model: mujoco.MjModel, height: int, width: int,
                 setup: dict[str, Any], config: VisionDataGenerationConfig,
                 rng: np.random.Generator, renderer_factory: Callable[..., Any]) -> None:
        self.renderer = renderer_factory(model, height=height, width=width)
        self.model = model
        self.config = config
        self.rng = rng
        self.height, self.width = height, width
        self.update_camera_setup(setup)
        self.camera = None
        self.data = None
        self._motion_data = None

    def update_camera_setup(self, setup: dict[str, Any]) -> None:
        """Refresh lens/effect parameters after resampling an existing camera."""
        height, width = self.height, self.width
        self.parameters = {sim_name: setup[name] for name, sim_name in SIM_CAMERA_NAMES.items()}
        self.maps = {}
        yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        for camera_id, camera_name in SIM_CAMERA_NAMES.items():
            scale = 1 + setup[camera_id]["lens_distortion"] * (xx * xx + yy * yy)
            self.maps[camera_name] = (
                ((xx * scale + 1) * (width - 1) / 2).astype(np.float32),
                ((yy * scale + 1) * (height - 1) / 2).astype(np.float32),
            )
    def update_scene(self, data: mujoco.MjData, camera: str) -> None:
        self.camera = camera
        self.data = data
        self.renderer.update_scene(data, camera=camera)

    def _render_focused_frame(self) -> np.ndarray:
        image = self.renderer.render().astype(np.float32)
        if self.config.focus_blur_scale == 0 or self.config.focus_blur_max_sigma == 0:
            return image
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        return _apply_focus_blur(
            image, depth, self.parameters[self.camera]["focus_distance"],
            self.config.focus_blur_scale, self.config.focus_blur_max_sigma,
        )

    def _render_exposure(self) -> np.ndarray:
        exposure = self.parameters[self.camera]["motion_blur_exposure"]
        if (self.config.motion_blur_probability == 0 or exposure == 0
                or self.data.time <= 0 or not np.any(self.data.qvel)
                or self.rng.random() >= self.config.motion_blur_probability):
            return self._render_focused_frame()
        exposure = min(exposure, self.data.time)
        if self._motion_data is None:
            self._motion_data = mujoco.MjData(self.model)
        accumulated = None
        try:
            # Approximate the shutter interval ending at the capture timestamp
            # using constant joint velocities. Live simulation data is untouched.
            for offset in np.linspace(-exposure, 0.0, self.config.motion_blur_samples):
                mujoco.mj_copyData(self._motion_data, self.model, self.data)
                mujoco.mj_integratePos(self.model, self._motion_data.qpos,
                                      self._motion_data.qvel, float(offset))
                self._motion_data.time = self.data.time + offset
                mujoco.mj_forward(self.model, self._motion_data)
                self.renderer.update_scene(self._motion_data, camera=self.camera)
                # Matching depth at each sample keeps moving edges aligned.
                image = self._render_focused_frame()
                accumulated = image if accumulated is None else accumulated + image
            return accumulated / self.config.motion_blur_samples
        finally:
            self.renderer.update_scene(self.data, camera=self.camera)

    def render(self) -> np.ndarray:
        parameters = self.parameters[self.camera]
        image = cv2.remap(self._render_exposure(), *self.maps[self.camera],
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        image *= self.rng.uniform(*self.config.exposure_range)
        image *= np.asarray(parameters["white_balance_gains"], dtype=np.float32)
        image += self.rng.normal(0.0, self.config.image_noise_std, size=image.shape)
        image = 255.0 * np.power(np.clip(image, 0.0, 255.0) / 255.0, parameters["gamma"])
        if self.rng.random() < self.config.blur_probability:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0,
                                     borderType=cv2.BORDER_REFLECT_101)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.renderer.close()


def collect_sequence(project: ProjectConfig, sequence_id: str, mode: str,
                     rng: np.random.Generator, renderer_factory: Callable[..., Any]) -> dict[str, Any]:
    """Collect a fixed number of random observations during a full rollout.

    Choose distinct steps from the configured capture cadence before starting
    the simulator. Both views are rendered synchronously at those steps, with
    labels and metadata from the same simulation state. Control and physics
    run every step at the configured cadence in simulated time, without sleeps.
    """
    config = project.environment.vision_data
    steps = int(np.ceil(project.environment.episode_seconds * project.loop.control_hz))
    candidates = np.arange(0, steps, config.sample_interval_steps)
    if config.samples_per_sequence > len(candidates):
        raise ValueError(
            f"samples_per_sequence={config.samples_per_sequence} exceeds the "
            f"{len(candidates)} available sample steps for this rollout."
        )
    # A separate stream keeps the robot trajectory independent of sample count.
    sampling_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
    sampled_steps = set(sampling_rng.choice(
        candidates, size=config.samples_per_sequence, replace=False,
    ).tolist())
    environment = CubeStackEnvironment(scene_path=project.environment.scene_path,
                                       configure_spec=configure_vision_materials)
    randomize_cube_poses(environment=environment, config=config, rng=rng, mode=mode)
    setup = randomize_visual_setup(environment=environment, config=config, rng=rng)
    # Camera effects must not change the RNG stream used for robot motion.
    image_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
    robot = SimRobotStateProvider(environment=environment)
    backend = SimJointCommandBackend(environment=environment)
    scenario = None
    if mode == "task":
        scenario = ScriptedVisionScenario(
            environment=environment, backend=backend,
            controller_config=project.cartesian_controller, config=config.task,
        )
    elif mode == "carry":
        scenario = ScriptedCarryScenario(
            environment=environment, backend=backend,
            controller_config=project.cartesian_controller, config=config.task,
            carry_config=config.carry, rng=rng,
        )
    phases = []
    scenario_metadata = []
    samples = {key: [] for key in ("wrist_images", "external_images", "joint_positions", "cube_poses",
                                  "robot_timestamps", "wrist_timestamps", "external_timestamps")}
    period = 1.0 / project.loop.control_hz
    yaw_encoders = {name: CubeYawEncoder() for name in ("orange", "blue")}
    with ExitStack() as resources:
        renderers = {}
        for name in SIM_CAMERA_NAMES:
            height, width, _ = CAMERA_IMAGE_SHAPES[name]
            renderer = AugmentedRenderer(
                model=environment.model, height=height, width=width,
                setup=setup, config=config, rng=image_rng, renderer_factory=renderer_factory,
            )
            resources.callback(renderer.close)
            renderers[name] = renderer
        for step in range(steps):
            # Physics stays fixed while face labels, robot state, and both
            # camera views are read. Motion blur uses copied states only.
            with environment.state_lock:
                # Update at unsaved control ticks too. Only a vertical-score
                # tie retains the previous face; a clearly better face wins.
                poses = cube_pose_labels(environment=environment, encoders=yaw_encoders)
                if step in sampled_steps:
                    state = robot.read()
                    for name, sim_name in SIM_CAMERA_NAMES.items():
                        renderer = renderers[name]
                        renderer.update_scene(environment.data, camera=sim_name)
                        samples[f"{name}_images"].append(renderer.render().copy())
                        samples[f"{name}_timestamps"].append(state.timestamp)
            if step in sampled_steps:
                samples["joint_positions"].append(state.joint_positions.astype(np.float32))
                samples["cube_poses"].append(poses)
                samples["robot_timestamps"].append(state.timestamp)
                phases.append(scenario.phase if scenario is not None else mode)
                if scenario is not None:
                    scenario_metadata.append(scenario.metadata())
            if scenario is not None:
                scenario.step()
            else:
                state = robot.read()
                targets = state.commanded_joint_targets + rng.normal(0.0, config.joint_step_std, size=6)
                backend.send_joint_targets(targets=targets)
            backend.advance(duration_seconds=period)
    return {
        **{name: torch.from_numpy(np.stack(values)) for name, values in samples.items()},
        "sequence_id": sequence_id, "cube_pose_mode": mode, "camera_setup": setup,
        "pose_representation": "leveled_yaw_cos4_sin4_previous_face_ties",
        "phases": phases, "scenario_metadata": scenario_metadata,
        "final_scenario_metadata": scenario.metadata() if scenario is not None else None,
        "config": project.model_dump(mode="json"),
    }


def generate_vision_data(project: ProjectConfig,
                         renderer_factory: Callable[..., Any] = mujoco.Renderer) -> list[Path]:
    """Add whole camera-setup sequences to train/validation without overwriting."""
    config = project.environment.vision_data
    split_rng = np.random.default_rng(project.training.vision.seed)
    order = split_rng.permutation(config.sequences)
    training_count = max(1, min(config.sequences - 1, round(config.sequences * config.train_fraction)))
    training_indices = set(order[:training_count])
    paths = []
    with tqdm(total=config.sequences * config.samples_per_sequence,
              desc="Generating vision data", unit="sample", dynamic_ncols=True) as progress:
        for index in range(config.sequences):
            episode_id = uuid4()
            rng = np.random.default_rng(np.random.SeedSequence([project.training.vision.seed, episode_id.int]))
            mode = str(rng.choice(config.cube_pose_modes, p=config.cube_pose_mode_probabilities))
            record = collect_sequence(project=project, sequence_id=str(episode_id), mode=mode, rng=rng,
                                      renderer_factory=renderer_factory)
            split = "train" if index in training_indices else "validation"
            path = project.training.vision.data_dir / split / f"{episode_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects existing files even in the improbable UUID collision.
            with path.open("xb") as stream:
                torch.save(record, stream)
            paths.append(path)
            sample_count = len(record["cube_poses"])
            progress.update(sample_count)
            tqdm.write(f"[{index + 1}/{config.sequences}] {split} {episode_id} "
                       f"samples={sample_count} mode={mode}")
            outcome = record["final_scenario_metadata"]
            if outcome is not None:
                tqdm.write(f"  measured grasp={outcome['grasp_seen']} lift={outcome['lift_seen']} "
                           f"drop={outcome['drop_seen']} retries={outcome['recovery_count']} "
                           f"stack_seen={outcome['stack_seen']}; last_failure={outcome['last_failure']}")
    return paths
