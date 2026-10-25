"""Uniform gripper-frame targets with physical carrying and vision labels.

Each successful iteration saves one paired-camera example. A scene persists
between successful moves; errors recreate it before another attempt. Surfaces,
lighting, and textures use the original vision generator's per-scene setup.
Cameras and blue positions are resampled per attempt. Targets are uniform;
successful saved positions may differ in distribution when moves fail.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
import traceback
from typing import Any
from uuid import UUID, uuid4

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from action_demonstrations import (
    CLOSED_GRIPPER, DemonstrationFailure, EpisodeRecorder, move_to,
    pickup_from_current_pose, reset_demonstration_environment,
)
from camera_feed_getter import SIM_CAMERA_NAMES
from cube_yaw import CubeYawEncoder
from environment import CubeStackEnvironment
from kinematics import IKConvergenceError
from schemas.arrays import CAMERA_IMAGE_SHAPES
from schemas.config import ProjectConfig
from vision_data import (
    AugmentedRenderer, capture_camera_baseline, cube_pose_labels,
    randomize_camera_setup, randomize_visual_setup,
)
from vision_materials import configure_vision_materials


TARGET_Z_RANGE_M = (0.03, 0.08)
DEFAULT_SAMPLES = 5000
MAX_CONSECUTIVE_FAILURES = 20


class GenerationLog:
    """Write only intentional events to JSON lines and tqdm's console writer."""

    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id = Path(path), run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        record = {"time": datetime.now().astimezone().isoformat(), "run_id": self.run_id,
                  "event": event, **fields}
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        # No stdout/stderr redirection: progress-bar redraws never enter the log.
        console_fields = {k: v for k, v in fields.items() if k not in ("traceback", "ik_diagnostics", "config")}
        tqdm.write(f"[{event}] {json.dumps(console_fields, allow_nan=False)}")

    def close(self) -> None:
        self.stream.close()


def sample_target(project: ProjectConfig, rng: np.random.Generator) -> np.ndarray:
    config = project.environment.vision_data
    bounds = np.asarray((config.cube_x_range, config.cube_y_range, TARGET_Z_RANGE_M))
    return rng.uniform(bounds[:, 0], bounds[:, 1])


class WorkspaceRecorder(EpisodeRecorder):
    """Reuse tested teacher motion while updating yaw history at every tick."""

    def __init__(self, project, environment, seed, encoders):
        super().__init__(project, environment, seed)
        self.encoders = encoders
        table = environment.model.geom("table").id
        top = float(environment.data.geom_xpos[table, 2] + environment.model.geom_size[table, 2])
        # A new recorder may begin mid-carry: the table threshold must not be
        # derived from the elevated orange starting height.
        self.initial_heights = {name: top + float(environment.model.geom(f"{name}_cube_geom").size[2])
                                for name in ("orange", "blue")}
        self.grasp_seen = self._pickup_grasp_seen = bool(self.scene()["held"])

    def step(self, action, stage):
        result = super().step(action, stage)
        cube_pose_labels(self.environment, self.encoders)
        return result


class WorkspaceScene:
    """Own physics, persistent renderers, visual setup, and one dataset split."""

    def __init__(self, project, split, scene_rng, image_rng, renderer_factory):
        self.project, self.split, self.scene_id = project, split, str(uuid4())
        self.environment = CubeStackEnvironment(project.environment.scene_path,
                                                configure_spec=configure_vision_materials)
        reset_demonstration_environment(self.environment, project, scene_rng)
        self.camera_baseline = capture_camera_baseline(self.environment)
        self.setup = randomize_visual_setup(self.environment, project.environment.vision_data, scene_rng)
        self.encoders = {name: CubeYawEncoder() for name in ("orange", "blue")}
        cube_pose_labels(self.environment, self.encoders)
        self.resources = ExitStack()
        self.renderers = {}
        try:
            for name in SIM_CAMERA_NAMES:
                height, width, _ = CAMERA_IMAGE_SHAPES[name]
                renderer = AugmentedRenderer(self.environment.model, height, width, self.setup,
                                             project.environment.vision_data, image_rng, renderer_factory)
                self.resources.callback(renderer.close)
                self.renderers[name] = renderer
        except BaseException:
            self.resources.close()
            raise

    def close(self):
        self.resources.close()

    def randomize_cameras(self, rng):
        cameras = randomize_camera_setup(self.environment, self.project.environment.vision_data,
                                         rng, baseline=self.camera_baseline)
        self.setup = {**self.setup, **cameras}
        for renderer in self.renderers.values():
            renderer.update_camera_setup(self.setup)

    def move_blue(self, rng, target):
        """Resample blue on the table, clear of orange and the target grasp."""
        environment, config = self.environment, self.project.environment.vision_data
        current = environment.data.body("orange_cube").xpos[:2].copy()
        for _ in range(1000):
            xy = rng.uniform([config.cube_x_range[0], config.cube_y_range[0]],
                             [config.cube_x_range[1], config.cube_y_range[1]])
            if min(np.linalg.norm(xy - current), np.linalg.norm(xy - target[:2])) >= config.minimum_cube_separation:
                break
        else:
            raise DemonstrationFailure("Could not sample a separated blue-cube position.")
        table = environment.model.geom("table").id
        z = float(environment.data.geom_xpos[table, 2] + environment.model.geom_size[table, 2]
                  + environment.model.geom("blue_cube_geom").size[2])
        yaw = rng.uniform(-np.pi, np.pi)
        joint = environment.data.joint("blue_cube_joint")
        joint.qpos[:] = [*xy, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        joint.qvel[:] = 0
        mujoco.mj_forward(environment.model, environment.data)
        self.encoders["blue"].reset()
        cube_pose_labels(environment, self.encoders)

    def collect(self, sample_id, target):
        environment = self.environment
        recorder = WorkspaceRecorder(self.project, environment, UUID(sample_id).int, self.encoders)
        picked_up = not recorder.scene()["held"]
        if picked_up:
            pickup_from_current_pose(recorder)
        if recorder.recovery_requested or not recorder.scene()["held"]:
            raise DemonstrationFailure("Pickup failed to produce a measured two-jaw grasp.")
        # This is deliberately the rigid gripperframe target. No cube offset.
        move_to(recorder, target, CLOSED_GRIPPER, "transport")
        scene = recorder.scene()
        error = float(np.linalg.norm(scene["gripper"] - target))
        if recorder.recovery_requested or not scene["held"]:
            raise DemonstrationFailure("Orange grasp was lost before reaching the target.")
        if not np.isfinite(error) or error > self.project.environment.action_data.teacher.position_tolerance:
            raise DemonstrationFailure(f"Gripper missed target by {error * 1000:.3f} mm.")
        with environment.state_lock:
            poses = cube_pose_labels(environment, self.encoders)
            robot = recorder.provider.read()
            images = {}
            for name, sim_name in SIM_CAMERA_NAMES.items():
                renderer = self.renderers[name]
                renderer.update_scene(environment.data, camera=sim_name)
                frame = renderer.render().copy()
                if frame.shape != CAMERA_IMAGE_SHAPES[name] or frame.dtype != np.uint8:
                    raise ValueError(f"{name}: renderer returned incompatible image shape or dtype.")
                images[f"{name}_images"] = torch.from_numpy(frame[None])
        timestamps = torch.tensor([robot.timestamp], dtype=torch.float64)
        metadata = {"phase": "carry", "currently_held": True, "picked_up_this_iteration": picked_up,
                    "gripper_target_xyz_m": target.tolist(), "actual_gripper_xyz_m": scene["gripper"].tolist(),
                    "gripper_error_mm": error * 1000, "stage_steps": dict(recorder.stages)}
        return {
            **images, "joint_positions": torch.from_numpy(robot.joint_positions.astype(np.float32)[None]),
            "cube_poses": torch.from_numpy(poses[None]),
            "robot_timestamps": timestamps, "wrist_timestamps": timestamps.clone(),
            "external_timestamps": timestamps.clone(),
            "uuid": sample_id, "sequence_id": sample_id, "scene_id": self.scene_id, "split": self.split,
            "cube_pose_mode": "uniform_gripper_workspace", "camera_setup": self.setup,
            "pose_representation": "leveled_yaw_cos4_sin4_previous_face_ties",
            "phases": ["carry"], "scenario_metadata": [metadata], "final_scenario_metadata": metadata,
            "config": self.project.model_dump(mode="json"),
        }


def save_example(record, directory: Path) -> Path:
    """Publish one UUID file atomically; preserve all existing data."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{record['uuid']}.pt"
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=f".{record['uuid']}-", suffix=".tmp",
                                dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def generate_workspace_vision_data(*, project: ProjectConfig, samples: int = DEFAULT_SAMPLES,
                                   log_file: Path | None = None, max_attempts: int | None = None,
                                   renderer_factory: Callable[..., Any] = mujoco.Renderer,
                                   scene_factory=WorkspaceScene) -> list[Path]:
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if max_attempts is None:
        max_attempts = samples * 10
    if type(max_attempts) is not int or max_attempts < samples:
        raise ValueError("max_attempts must be an integer at least as large as samples")
    project = project.model_copy(deep=True)
    config = project.environment.vision_data
    project.environment.action_data.drop_disturbance.probability = 0.0
    project.environment.action_data.cube_x_range = config.cube_x_range
    project.environment.action_data.cube_y_range = config.cube_y_range
    project.environment.action_data.minimum_cube_separation = config.minimum_cube_separation
    # Per-iteration recorder, independent of how long this scene has existed.
    project.environment.episode_seconds = max(project.environment.episode_seconds, 45.0)
    training_count = (1 if samples == 1 else max(1, min(samples - 1, round(samples * config.train_fraction))))
    run_id = str(uuid4())
    streams = np.random.SeedSequence([project.training.vision.seed, UUID(run_id).int]).spawn(5)
    target_rng, scene_rng, camera_rng, blue_rng, image_rng = map(np.random.default_rng, streams)
    if log_file is None:
        log_file = Path(__file__).resolve().parents[1] / "logs" / (
            f"vision-workspace-{datetime.now():%Y%m%d-%H%M%S}-{run_id[:8]}.log")
    log = GenerationLog(log_file, run_id)
    paths = []
    attempts = failures = consecutive_failures = 0
    scene = None
    started = perf_counter()

    def create_scene(split):
        # Creation is also bounded: persistent renderer/config errors must not
        # trap a run forever. Every failure remains visible in the event log.
        for creation_attempt in range(1, MAX_CONSECUTIVE_FAILURES + 1):
            try:
                created = scene_factory(project, split, scene_rng, image_rng, renderer_factory)
            except Exception as error:
                log.write("SCENE_CREATE_FAILURE", creation_attempt=creation_attempt, split=split,
                          error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            else:
                try:
                    log.write("SCENE_CREATED", scene_id=created.scene_id, split=split)
                except BaseException:
                    created.close()
                    raise
                return created
        raise RuntimeError("Scene creation failed repeatedly; see the generation log.")

    try:
        log.write("RUN_START", requested_samples=samples, training_samples=training_count,
                  validation_samples=samples - training_count, max_attempts=max_attempts,
                  data_dir=str(project.training.vision.data_dir), log_file=str(log_file),
                  target_x_range_m=list(config.cube_x_range), target_y_range_m=list(config.cube_y_range),
                  target_z_range_m=list(TARGET_Z_RANGE_M), config=project.model_dump(mode="json"))
        # The first scene exists before entering the sampling loop.
        scene = create_scene("train")
        with tqdm(total=samples, desc="Generating workspace vision data", unit="sample", dynamic_ncols=True) as progress:
            while len(paths) < samples:
                if attempts >= max_attempts:
                    raise RuntimeError(f"Reached {max_attempts} attempts with only {len(paths)}/{samples} saved samples.")
                sample_id = str(uuid4())
                target = sample_target(project, target_rng)
                attempts += 1
                failed = False
                try:
                    scene.randomize_cameras(camera_rng)
                    scene.move_blue(blue_rng, target)
                    record = scene.collect(sample_id, target)
                    record["run_id"] = run_id
                    record["attempt"] = attempts
                    path = save_example(record, project.training.vision.data_dir / scene.split)
                except Exception as error:
                    failed = True
                    failures += 1
                    consecutive_failures += 1
                    fields = {"uuid": sample_id, "scene_id": scene.scene_id, "attempt": attempts,
                              "saved": len(paths), "target_xyz_m": target.tolist(),
                              "error_type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()}
                    if isinstance(error, IKConvergenceError):
                        fields["ik_diagnostics"] = error.diagnostics
                    log.write("SAMPLE_FAILURE", **fields)
                else:
                    paths.append(path)
                    consecutive_failures = 0
                    log.write("SAMPLE_SAVED", uuid=sample_id, scene_id=scene.scene_id, split=scene.split,
                              path=str(path), saved=len(paths), requested=samples, attempt=attempts,
                              target_xyz_m=target.tolist(), **record["final_scenario_metadata"])
                    progress.update(1)
                progress.set_postfix(attempts=attempts, failures=failures, refresh=False)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"Stopped after {consecutive_failures} consecutive failed samples.")
                next_split = "train" if len(paths) < training_count else "validation"
                # Recreate at the END of a failed iteration, before another
                # target is sampled. A split boundary also requires a new
                # independent scene to avoid sharing setups across splits.
                if len(paths) < samples and attempts < max_attempts and (failed or scene.split != next_split):
                    scene.close()
                    scene = None
                    scene = create_scene(next_split)
        log.write("RUN_COMPLETE", saved=len(paths), attempts=attempts, failures=failures,
                  wall_seconds=perf_counter() - started)
        return paths
    except BaseException as error:
        log.write("RUN_ABORTED", saved=len(paths), attempts=attempts, failures=failures,
                  error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if scene is not None:
                scene.close()
        finally:
            log.close()
