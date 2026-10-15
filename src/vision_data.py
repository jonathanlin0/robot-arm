"""Generate additive, sequence-split perception data with varied visual setups.

Physics scenes combine varied placements, random arm motion, and optional
scripted task attempts. Their phase metadata does not imply a successful grasp
or stack. Ground-truth labels stay in this training-data module; the runtime
perception provider always uses images.
"""

from collections.abc import Callable
from colorsys import hsv_to_rgb
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import cv2
import mujoco
import numpy as np
import torch

from camera_feed_getter import SIM_CAMERA_NAMES, SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from camera_producers import SimCameraProducer
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from robot_state_provider import SimRobotStateProvider
from schemas.config import ProjectConfig, VisionDataGenerationConfig
from vision_errors import CameraAcquisitionError, VisionDataUnavailableError
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


def randomize_visual_setup(environment: CubeStackEnvironment, config: VisionDataGenerationConfig,
                           rng: np.random.Generator) -> dict[str, Any]:
    """Set mounts, lens parameters, lights and materials once for a sequence."""
    model = environment.model
    setup = {}
    position_jitter = np.asarray(config.camera_position_jitter)
    light_position_jitter = np.asarray(config.light_position_jitter)
    with environment.state_lock:
        for name, sim_name in SIM_CAMERA_NAMES.items():
            camera = model.camera(sim_name)
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
        for name in ("table", "wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"):
            # Floor and walls share the muted palette, each sampled independently.
            model.geom(name).rgba[:3] = hsv_to_rgb(
                rng.uniform(), rng.uniform(*config.wall_saturation_range),
                rng.uniform(*config.wall_brightness_range),
            )
        contrast = rng.uniform(*config.texture_contrast_range)
        # Randomize the scene's skybox texture and non-cube material contrast.
        # Do this before renderer creation so its GPU textures match the model.
        texture = model.tex_data.astype(np.float32)
        texture += rng.normal(0.0, 255.0 * contrast, size=texture.shape)
        model.tex_data[:] = np.clip(texture, 0, 255).astype(np.uint8)
        model.mat_rgba[:, :3] = np.clip(
            model.mat_rgba[:, :3] * rng.uniform(1 - contrast, 1 + contrast, size=(model.nmat, 1)),
            0, 1,
        )
        mujoco.mj_forward(model, environment.data)
        setup["texture_contrast"] = float(contrast)
    return setup


def cube_pose_labels(environment: CubeStackEnvironment) -> np.ndarray:
    """Read orange XYZ/wxyz then blue XYZ/wxyz at the current robot timestamp."""
    with environment.state_lock:
        return np.concatenate([
            component.copy()
            for name in ("orange", "blue")
            for component in (environment.data.body(f"{name}_cube").xpos,
                              environment.data.body(f"{name}_cube").xquat)
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
        self.parameters = {sim_name: setup[name] for name, sim_name in SIM_CAMERA_NAMES.items()}
        self.maps = {}
        yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        for camera_id, camera_name in SIM_CAMERA_NAMES.items():
            scale = 1 + setup[camera_id]["lens_distortion"] * (xx * xx + yy * yy)
            self.maps[camera_name] = (
                ((xx * scale + 1) * (width - 1) / 2).astype(np.float32),
                ((yy * scale + 1) * (height - 1) / 2).astype(np.float32),
            )
        self.camera = None
        self.data = None
        self._motion_data = None

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
                     rng: np.random.Generator, renderer_factory: Callable[..., Any],
                     producer_factory: Callable[..., Any], clock: Callable[[], float],
                     wait: Callable[[float], None]) -> dict[str, Any]:
    """Collect latest camera samples while commands and physics run at control_hz."""
    config = project.environment.vision_data
    environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
    randomize_cube_poses(environment=environment, config=config, rng=rng, mode=mode)
    setup = randomize_visual_setup(environment=environment, config=config, rng=rng)
    # Keep augmentation RNG local to its worker, separate from control sampling.
    image_rng = np.random.default_rng(int(rng.integers(0, 2**63)))

    def make_renderer(model, height, width):
        return AugmentedRenderer(model=model, height=height, width=width, setup=setup,
                                 config=config, rng=image_rng, renderer_factory=renderer_factory)

    buffers = {
        name: CameraFrameBuffer(camera_id=name,
                                maximum_frame_age_seconds=project.cameras.maximum_frame_age_seconds,
                                clock=clock)
        for name in SIM_CAMERA_NAMES
    }
    producer = producer_factory(environment=environment, buffers=buffers,
                                fps=project.cameras.fps, renderer_factory=make_renderer)
    getter = SimCameraFeedGetter(environment=environment, buffers=buffers)
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
    steps = int(np.ceil(project.environment.episode_seconds * project.loop.control_hz))
    producer.start()
    try:
        # Startup is bounded; fatal renderer failures immediately escape capture().
        deadline = clock() + project.cameras.maximum_frame_age_seconds
        while True:
            try:
                getter.capture()
                break
            except VisionDataUnavailableError:
                if clock() >= deadline:
                    raise CameraAcquisitionError("Initial camera frames were not produced before timeout.")
                wait(min(1.0 / project.cameras.fps, max(0.0, deadline - clock())))
        for step in range(steps):
            started = clock()
            if step % config.sample_interval_steps == 0:
                frames = getter.capture()
                # Only this thread advances physics, so these labels and joints
                # describe the same observation time without matching image times.
                with environment.state_lock:
                    state = robot.read()
                    poses = cube_pose_labels(environment=environment)
                samples["joint_positions"].append(state.joint_positions.astype(np.float32))
                samples["cube_poses"].append(poses)
                samples["robot_timestamps"].append(state.timestamp)
                phases.append(scenario.phase if scenario is not None else mode)
                if scenario is not None:
                    scenario_metadata.append(scenario.metadata())
                for name in SIM_CAMERA_NAMES:
                    samples[f"{name}_images"].append(frames[name].rgb)
                    samples[f"{name}_timestamps"].append(frames[name].timestamp)
            if scenario is not None:
                scenario.step()
            else:
                state = robot.read()
                targets = state.commanded_joint_targets + rng.normal(0.0, config.joint_step_std, size=6)
                backend.send_joint_targets(targets=targets)
            backend.advance(duration_seconds=period)
            wait(max(0.0, period - (clock() - started)))
    finally:
        producer.stop()
    return {
        **{name: torch.from_numpy(np.stack(values)) for name, values in samples.items()},
        "sequence_id": sequence_id, "cube_pose_mode": mode, "camera_setup": setup,
        "phases": phases, "scenario_metadata": scenario_metadata,
        "final_scenario_metadata": scenario.metadata() if scenario is not None else None,
        "config": project.model_dump(mode="json"),
    }


def generate_vision_data(project: ProjectConfig, renderer_factory: Callable[..., Any] = mujoco.Renderer,
                         producer_factory: Callable[..., Any] = SimCameraProducer,
                         clock: Callable[[], float] = monotonic,
                         wait: Callable[[float], None] = sleep) -> list[Path]:
    """Add whole camera-setup sequences to train/validation without overwriting."""
    config = project.environment.vision_data
    split_rng = np.random.default_rng(project.training.vision.seed)
    order = split_rng.permutation(config.sequences)
    training_count = max(1, min(config.sequences - 1, round(config.sequences * config.train_fraction)))
    training_indices = set(order[:training_count])
    paths = []
    for index in range(config.sequences):
        episode_id = uuid4()
        rng = np.random.default_rng(np.random.SeedSequence([project.training.vision.seed, episode_id.int]))
        mode = str(rng.choice(config.cube_pose_modes, p=config.cube_pose_mode_probabilities))
        record = collect_sequence(project=project, sequence_id=str(episode_id), mode=mode, rng=rng,
                                  renderer_factory=renderer_factory, producer_factory=producer_factory,
                                  clock=clock, wait=wait)
        split = "train" if index in training_indices else "validation"
        path = project.training.vision.data_dir / split / f"{episode_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing files even in the improbable UUID collision.
        with path.open("xb") as stream:
            torch.save(record, stream)
        paths.append(path)
        print(f"[{index + 1}/{config.sequences}] {split} {episode_id} "
              f"samples={len(record['cube_poses'])} mode={mode}")
        outcome = record["final_scenario_metadata"]
        if outcome is not None:
            print(f"  measured grasp={outcome['grasp_seen']} lift={outcome['lift_seen']} "
                  f"drop={outcome['drop_seen']} retries={outcome['recovery_count']} "
                  f"stack_seen={outcome['stack_seen']}; last_failure={outcome['last_failure']}")
    return paths
