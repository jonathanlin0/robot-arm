#!/usr/bin/env mjpython

"""Render a grid with 3 rows and 4 columns of policy rollouts or saved demonstrations.

Run from the repository root after training has saved the PPO checkpoint:

    mjpython scripts/demo.py

Render the checkpoint produced by a particular W&B sweep run with:

    mjpython scripts/demo.py --repeat-wandb m58de6rc

View up to twelve randomly selected training demonstrations without a model:

    mjpython scripts/demo.py --pretrain-data

This displays saved states directly, including the full recorded episode,
and writes ``stack_demo.mp4``. Shorter panels hold their final frame; unused
panels are black. The terminal lists the selected episode UUIDs.

Override the worker count with, for example, ``--workers 3``. Each worker
creates independent MuJoCo and rendering state. The parent process combines
the twelve temporary panel videos and deletes them after the final video has
been created.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from gymnasium import spaces
import mujoco
import numpy as np
from stable_baselines3 import PPO


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from environment import PHYSICS_STEPS_PER_ACTION  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402
from action_observation_history import ActionObservationHistoryWrapper  # noqa: E402
from randomization import CubeSpawnConfig  # noqa: E402
from rewards import StackRewardConfig  # noqa: E402


CHECKPOINT_PATH = Path("checkpoints/pretraining/default.zip")
PRETRAINING_DATA_DIRECTORY = Path("data")
WANDB_CHECKPOINT_DIRECTORY = Path("checkpoints/wandb")
OUTPUT_PATH = Path("stack_demo.mp4")
TEMPORARY_DIRECTORY_ROOT = Path(".tmp")

GRID_ROWS = 3
GRID_COLUMNS = 4
ROLLOUT_COUNT = GRID_ROWS * GRID_COLUMNS
DEFAULT_WORKER_COUNT = 5
BASE_EVALUATION_SEED = 0
MAXIMUM_EPISODE_STEPS = 400

PANEL_WIDTH = 320
PANEL_HEIGHT = 240
STATUS_BORDER_WIDTH = 4
GRIPPERFRAME_MARKER_RADIUS = 0.006
GRIPPERFRAME_MARKER_COLOR = np.array(
    [0.65, 0.0, 1.0, 1.0],
    dtype=np.float32,
)
# Match the radius used to decide that the pregrasp waypoint was reached.
APPROACH_TARGET_MARKER_RADIUS = 0.010
# The cube has a 0.02 m half-width. A normal marker at its mathematical
# center would be completely occluded, so this slightly larger sphere leaves
# a small visible cap on the cube faces without moving the marked position.
ORANGE_CENTER_TARGET_MARKER_RADIUS = 0.021
APPROACH_TARGET_MARKER_COLOR = np.array(
    [0.0, 1.0, 0.0, 1.0],
    dtype=np.float32,
)

RUNNING = "running"
SUCCESS = "success"
FAILURE = "failure"
TRUNCATED = "truncated"

STATUS_BORDER_COLORS = {
    RUNNING: np.array([230, 230, 230], dtype=np.uint8),
    SUCCESS: np.array([40, 210, 70], dtype=np.uint8),
    FAILURE: np.array([230, 50, 50], dtype=np.uint8),
    TRUNCATED: np.array([245, 165, 35], dtype=np.uint8),
}

_worker_policy: PPO | None = None
_worker_policy_checkpoint_path: Path | None = None


@dataclass(frozen=True)
class RolloutTask:
    """Inputs needed by one spawned rollout worker."""

    rollout_index: int
    seed: int
    output_path: Path
    checkpoint_path: Path = CHECKPOINT_PATH


@dataclass(frozen=True)
class RolloutResult:
    """Small rollout summary returned from a worker to the parent."""

    rollout_index: int
    seed: int
    status: str
    episode_steps: int
    episode_reward: float
    output_path: Path


@dataclass(frozen=True)
class DemonstrationTask:
    """One recorded training episode, padded to the grid's common duration."""

    rollout_index: int
    output_path: Path
    record: dict
    data_directory: Path
    frame_count: int


def select_demonstrations(data_directory: Path, count: int = ROLLOUT_COUNT) -> list[dict]:
    """Randomly choose distinct, committed and complete successful train episodes."""
    import json
    import random
    from uuid import UUID

    if type(count) is not int or count < 1:
        raise ValueError("demonstration count must be a positive integer")
    directory = Path(data_directory)
    candidates: dict[str, dict] = {}
    paths = sorted((directory / "manifests").glob("*.json"))
    if (directory / "manifest.json").is_file():
        paths.insert(0, directory / "manifest.json")
    for path in paths:
        try:
            content = json.loads(path.read_text())
            version = content.get("format_version")
            records = (content["episodes"] if version == 1 else
                       [content["episode"]] if version == 2 else [])
        except (OSError, ValueError, KeyError, AttributeError):
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            try:
                identifier = record["uuid"]
                episode_id = UUID(identifier)
                if (str(episode_id) != identifier or record["split"] != "train"
                        or record["success"] is not True or type(record["steps"]) is not int
                        or record["steps"] < 1 or type(record["seed"]) is not int
                        or record["seed"] < 0):
                    continue
                if version == 2 and (path.stem != identifier or record["seed"] != episode_id.int):
                    continue
                if all((directory / "train" / name / f"{identifier}.pt").is_file()
                       for name in ("observations", "accepted_targets", "actions")):
                    candidates[identifier] = record
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    if not candidates:
        raise ValueError(f"No completed successful training demonstrations found in {directory}.")
    return random.sample(list(candidates.values()), min(count, len(candidates)))


def load_observations(data_directory: Path, record: dict) -> np.ndarray:
    """Load only the recorded states needed for playback, without modifying data."""
    from uuid import UUID
    import torch

    try:
        identifier = record["uuid"]
        if (str(UUID(identifier)) != identifier or record["split"] != "train"
                or type(record["steps"]) is not int or record["steps"] < 1):
            raise ValueError("invalid training episode metadata")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("Recorded playback requires a valid training episode UUID and steps.") from error
    path = Path(data_directory) / "train" / "observations" / f"{identifier}.pt"
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    expected_shape = (record["steps"] + 1, 49)
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape
            or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
        raise ValueError(f"{path} must be a finite float32 tensor with shape {expected_shape}.")
    return tensor.numpy()


def apply_recorded_observation(
    model: mujoco.MjModel, data: mujoco.MjData, observation: np.ndarray,
) -> None:
    """Restore the 49-value observation's physical state, then refresh geometry."""
    from robot_constants import ROBOT_JOINT_NAMES
    from randomization import ORANGE_CUBE_JOINT, BLUE_CUBE_JOINT

    state = np.asarray(observation)
    if state.shape != (49,) or not np.isfinite(state).all():
        raise ValueError("recorded observation must be a finite vector of length 49")
    for index, name in enumerate(ROBOT_JOINT_NAMES):
        data.joint(name).qpos[0] = state[index]
        data.joint(name).qvel[0] = state[6 + index]
        data.actuator(name).ctrl[0] = state[12 + index]
    for name, offset in ((ORANGE_CUBE_JOINT, 21), (BLUE_CUBE_JOINT, 34)):
        data.joint(name).qpos[:] = state[offset:offset + 7]
        data.joint(name).qvel[:] = state[offset + 7:offset + 13]
    mujoco.mj_forward(model, data)


def render_demonstration(task: DemonstrationTask) -> RolloutResult:
    """Render every saved state, including the initial and terminal observations."""
    observations = load_observations(task.data_directory, task.record)
    if type(task.frame_count) is not int or task.frame_count < len(observations):
        raise ValueError("frame_count must include every recorded observation")
    settings = task.record["settings"]
    interval = float(settings["action_interval"])
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("recorded action_interval must be finite and positive")
    scene_path = Path(settings["scene"])
    if not scene_path.is_absolute():
        scene_path = Path(__file__).resolve().parents[1] / scene_path
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    spawn_config = CubeSpawnConfig(**settings["spawn_config"])
    terminal_status = SUCCESS if task.record["success"] is True else FAILURE
    renderer = None
    writer = None
    try:
        renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
        writer = RawVideoWriter(
            output_path=task.output_path, width=PANEL_WIDTH, height=PANEL_HEIGHT,
            frames_per_second=1.0 / interval,
        )
        camera = create_overview_camera()
        for index, observation in enumerate(observations):
            data.time = index * interval
            apply_recorded_observation(model, data, observation)
            renderer.update_scene(data, camera=camera)
            add_spawn_area_outline(renderer.scene, spawn_config)
            add_gripperframe_marker(renderer.scene, data)
            status = terminal_status if index == len(observations) - 1 else RUNNING
            frame = add_status_border(renderer.render(), status)
            writer.write(frame)
        for _ in range(task.frame_count - len(observations)):
            writer.write(frame)
    except BaseException:
        if writer is not None:
            writer.close(check_return_code=False)
        raise
    else:
        writer.close()
    finally:
        if renderer is not None:
            renderer.close()
    return RolloutResult(task.rollout_index, task.record["seed"], terminal_status,
                         task.record["steps"], 0.0, task.output_path)


class RawVideoWriter:
    """Stream RGB frames to FFmpeg to produce one panel video."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        frames_per_second: float,
    ):
        if width < 1 or height < 1:
            raise ValueError("video dimensions must be positive")
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError("video dimensions must be even for yuv420p")
        if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
            raise ValueError(
                "frames_per_second must be finite and positive"
            )

        self.output_path = Path(output_path)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{frames_per_second:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "FFmpeg was not found. Install it with `brew install "
                "ffmpeg`."
            ) from error

        if self._process.stdin is None:
            self._process.kill()
            self._process.wait()
            raise RuntimeError("FFmpeg did not provide a writable input pipe.")

        self._stdin = self._process.stdin
        self._expected_shape = (height, width, 3)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        """Write one RGB uint8 frame to the encoded video."""
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")

        rgb_frame = np.asarray(frame)
        if rgb_frame.shape != self._expected_shape:
            raise ValueError(
                f"video frame must have shape {self._expected_shape}; "
                f"received {rgb_frame.shape}"
            )
        if rgb_frame.dtype != np.uint8:
            raise ValueError("video frame must have dtype uint8")

        try:
            self._stdin.write(np.ascontiguousarray(rgb_frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(
                f"FFmpeg stopped while encoding {self.output_path}."
            ) from error

    def close(self, *, check_return_code: bool = True) -> None:
        """Finish encoding and optionally verify that FFmpeg succeeded."""
        if self._closed:
            return

        self._closed = True
        try:
            self._stdin.close()
        except BrokenPipeError:
            # The process return code below provides the useful failure signal.
            pass

        return_code = self._process.wait()
        if check_return_code and return_code != 0:
            raise RuntimeError(
                f"FFmpeg exited with status {return_code} while creating "
                f"{self.output_path}."
            )


def parse_worker_count(value: str) -> int:
    """Parse a worker count in the useful range for the rollout grid."""
    try:
        worker_count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "worker count must be an integer"
        ) from error

    if not 1 <= worker_count <= ROLLOUT_COUNT:
        raise argparse.ArgumentTypeError(
            f"worker count must be between 1 and {ROLLOUT_COUNT}"
        )
    return worker_count


def parse_wandb_run_id(value: str) -> str:
    """Require a W&B run ID that is safe to use as one path component."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise argparse.ArgumentTypeError(
            "W&B run ID may contain only letters, numbers, underscores, "
            "and hyphens"
        )
    return value


def resolve_checkpoint_path(wandb_run_id: str | None) -> Path:
    """Return the default checkpoint or one saved by a W&B sweep run."""
    if wandb_run_id is None:
        return CHECKPOINT_PATH
    return (
        WANDB_CHECKPOINT_DIRECTORY
        / wandb_run_id
        / CHECKPOINT_PATH.name
    )


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line options without starting any worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=parse_worker_count,
        default=DEFAULT_WORKER_COUNT,
        help=(
            "number of spawned rollout workers "
            f"(default: {DEFAULT_WORKER_COUNT})"
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--repeat-wandb",
        metavar="RUN_ID",
        type=parse_wandb_run_id,
        help="render the checkpoint saved by the specified W&B run",
    )
    source.add_argument(
        "--pretrain-data",
        action="store_true",
        help=f"render up to {ROLLOUT_COUNT} randomly selected saved training demonstrations",
    )
    return parser.parse_args(arguments)


def temporary_rollout_directory(process_id: int | None = None) -> Path:
    """Return a unique, repository-local directory for this parent run."""
    if process_id is None:
        process_id = os.getpid()
    if process_id < 1:
        raise ValueError("process_id must be positive")
    return TEMPORARY_DIRECTORY_ROOT / f"policy_grid_{process_id}"


def rollout_video_paths(directory: Path) -> list[Path]:
    """Return the ordered intermediate video paths for the grid."""
    return [
        directory / f"rollout_{rollout_index:02d}.mp4"
        for rollout_index in range(ROLLOUT_COUNT)
    ]


def cleanup_temporary_directory(directory: Path) -> None:
    """Delete only the temporary directory created for one render run."""
    if directory.exists():
        shutil.rmtree(directory)


def create_overview_camera() -> mujoco.MjvCamera:
    """Create the fixed overview used for every grid panel."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.25, 0.0, 0.12)
    camera.distance = 0.75
    camera.azimuth = 160.0
    camera.elevation = -25.0
    return camera


def add_status_border(frame: np.ndarray, status: str) -> np.ndarray:
    """Return a frame whose border identifies rollout terminal status."""
    if status not in STATUS_BORDER_COLORS:
        raise ValueError(f"unknown rollout status: {status!r}")

    bordered_frame = np.asarray(frame).copy()
    expected_shape = (PANEL_HEIGHT, PANEL_WIDTH, 3)
    if bordered_frame.shape != expected_shape:
        raise ValueError(
            f"panel frame must have shape {expected_shape}; received "
            f"{bordered_frame.shape}"
        )
    if bordered_frame.dtype != np.uint8:
        raise ValueError("panel frame must have dtype uint8")

    color = STATUS_BORDER_COLORS[status]
    border_width = STATUS_BORDER_WIDTH
    bordered_frame[:border_width, :, :] = color
    bordered_frame[-border_width:, :, :] = color
    bordered_frame[:, :border_width, :] = color
    bordered_frame[:, -border_width:, :] = color
    return bordered_frame


def _add_sphere_marker(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    color: np.ndarray,
    name: str,
) -> None:
    """Append one emissive decorative sphere to a populated render scene."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(
            f"render scene has no capacity for the {name} marker"
        )

    marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    marker.emission = 1.0
    marker.category = mujoco.mjtCatBit.mjCAT_DECOR
    marker.segid = -1
    scene.ngeom += 1


def add_spawn_area_outline(
    scene: mujoco.MjvScene,
    spawn_config: CubeSpawnConfig,
) -> None:
    """Outline the cube-center spawn bounds on the tabletop for playback."""
    if scene.ngeom + 4 > scene.maxgeom:
        raise RuntimeError("render scene has no capacity for the spawn-area outline")

    x_min, x_max = spawn_config.x_range
    y_min, y_max = spawn_config.y_range
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_x, half_y = (x_max - x_min) / 2, (y_max - y_min) / 2
    # Two-millimeter-wide flat strokes just above the z=0 tabletop avoid
    # flickering from coincident surfaces. Decorative geoms affect rendering only.
    for position, half_size in (
        ((center_x, y_min, 0.0005), (half_x, 0.001, 0.0001)),
        ((center_x, y_max, 0.0005), (half_x, 0.001, 0.0001)),
        ((x_min, center_y, 0.0005), (0.001, half_y, 0.0001)),
        ((x_max, center_y, 0.0005), (0.001, half_y, 0.0001)),
    ):
        line = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            line,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(half_size, dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        line.specular = 0.0
        line.category = mujoco.mjtCatBit.mjCAT_DECOR
        line.segid = -1
        scene.ngeom += 1


def add_gripperframe_marker(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
) -> None:
    """Draw a bright purple sphere at the Cartesian-control site."""
    _add_sphere_marker(
        scene,
        np.asarray(data.site("gripperframe").xpos, dtype=np.float64),
        GRIPPERFRAME_MARKER_RADIUS,
        GRIPPERFRAME_MARKER_COLOR,
        "gripperframe",
    )


def active_orange_approach_target(
    data: mujoco.MjData,
    *,
    waypoint_reached: bool,
    approach_height_offset: float,
) -> np.ndarray:
    """Return the reward's current waypoint or orange-center target."""
    target = np.asarray(
        data.body("orange_cube").xpos,
        dtype=np.float64,
    ).copy()
    if not waypoint_reached:
        target[2] += approach_height_offset
    return target


def add_active_orange_approach_target_marker(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
    *,
    waypoint_reached: bool,
    approach_height_offset: float,
) -> None:
    """Draw the active approach target as a bright-green sphere."""
    target = active_orange_approach_target(
        data,
        waypoint_reached=waypoint_reached,
        approach_height_offset=approach_height_offset,
    )
    marker_radius = (
        ORANGE_CENTER_TARGET_MARKER_RADIUS
        if waypoint_reached
        else APPROACH_TARGET_MARKER_RADIUS
    )
    _add_sphere_marker(
        scene,
        target,
        marker_radius,
        APPROACH_TARGET_MARKER_COLOR,
        "orange approach target",
    )


def rollout_status(
    terminated: bool,
    truncated: bool,
    info: dict[str, object],
) -> str:
    """Translate one Gymnasium terminal transition into a panel status."""
    if bool(info["is_success"]):
        return SUCCESS
    if bool(info["is_failure"]) or terminated:
        return FAILURE
    if truncated:
        return TRUNCATED
    return RUNNING


def get_worker_policy(checkpoint_path: Path = CHECKPOINT_PATH) -> PPO:
    """Load one CPU policy per spawned process and reuse it for its tasks."""
    global _worker_policy, _worker_policy_checkpoint_path
    requested_checkpoint_path = Path(checkpoint_path)
    if (
        _worker_policy is None
        or _worker_policy_checkpoint_path != requested_checkpoint_path
    ):
        import torch

        # Without this, every process can create a full CPU thread pool.
        torch.set_num_threads(1)
        _worker_policy = PPO.load(
            requested_checkpoint_path,
            device="cpu",
        )
        _worker_policy_checkpoint_path = requested_checkpoint_path
    return _worker_policy


def reset_state_dependent_noise_if_due(
    policy: PPO,
    policy_step: int,
) -> None:
    """Match PPO training's gSDE noise-resampling cadence during playback."""
    if policy_step < 0:
        raise ValueError("policy_step must be nonnegative")
    if not policy.use_sde:
        return

    resample_frequency = policy.sde_sample_freq
    beginning_of_episode = policy_step == 0
    scheduled_resample = (
        resample_frequency > 0
        and policy_step % resample_frequency == 0
    )
    if beginning_of_episode or scheduled_resample:
        policy.policy.reset_noise(n_envs=1)


def create_policy_environment(
    environment: CubeStackGymEnvironment,
    policy: PPO,
) -> ActionObservationHistoryWrapper:
    """Reconstruct the history inputs recorded in the trained checkpoint."""
    observation_space = policy.observation_space
    if (
        not isinstance(observation_space, spaces.Dict)
        or "tokens" not in observation_space.spaces
        or len(observation_space["tokens"].shape or ()) != 2
    ):
        raise ValueError(
            "Playback requires a transformer checkpoint with action/observation "
            "history. Retrain older flat-observation checkpoints."
        )

    history_length = observation_space["tokens"].shape[0]
    training_config = getattr(policy, "pickup_training_config", {})
    if training_config.get("history_length", history_length) != history_length:
        raise ValueError(
            "Checkpoint history length disagrees with its saved training config."
        )
    policy_environment = ActionObservationHistoryWrapper(
        environment,
        history_length=history_length,
    )
    if policy_environment.observation_space != observation_space:
        token_dim = policy_environment.observation_space["tokens"].shape[1]
        raise ValueError(
            "Checkpoint history observations do not match this environment. "
            f"Expected {token_dim}-value tokens ordered as "
            "[observation, previous action, previous accepted XYZ target]. "
            "Retrain checkpoints saved with the older token layout."
        )
    return policy_environment


def render_rollout(task: RolloutTask | DemonstrationTask) -> RolloutResult:
    """Render one fixed-length rollout inside a spawned worker process."""
    if isinstance(task, DemonstrationTask):
        return render_demonstration(task)
    policy = get_worker_policy(task.checkpoint_path)
    environment: CubeStackGymEnvironment | None = None
    renderer: mujoco.Renderer | None = None
    video_writer: RawVideoWriter | None = None

    status = RUNNING
    episode_steps = 0
    episode_reward = 0.0
    frozen_terminal_frame: np.ndarray | None = None

    try:
        training_config = getattr(policy, "pickup_training_config", {})
        environment = CubeStackGymEnvironment(
            maximum_episode_steps=training_config.get(
                "maximum_episode_steps",
                MAXIMUM_EPISODE_STEPS,
            ),
            reward_config=StackRewardConfig(
                **training_config.get("reward_config", {})
            ),
            start_at_orange_waypoint=bool(
                training_config.get("start_at_orange_waypoint", False)
            ),
            # Keep playback starts comparable across training curricula.
            recovery_start_probability=0.0,
        )
        policy_environment = create_policy_environment(environment, policy)
        observation, _ = policy_environment.reset(seed=task.seed)
        approach_height_offset = (
            environment.reward_config.approach_orange_height_offset
        )
        renderer = mujoco.Renderer(
            environment.simulation.model,
            height=PANEL_HEIGHT,
            width=PANEL_WIDTH,
        )

        control_interval = (
            PHYSICS_STEPS_PER_ACTION
            * environment.simulation.model.opt.timestep
        )
        video_writer = RawVideoWriter(
            output_path=task.output_path,
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
            frames_per_second=1.0 / control_interval,
        )
        camera = create_overview_camera()

        for video_step in range(1, MAXIMUM_EPISODE_STEPS + 1):
            if frozen_terminal_frame is None:
                reset_state_dependent_noise_if_due(
                    policy,
                    episode_steps,
                )
                action, _ = policy.predict(
                    observation,
                    deterministic=True,
                )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = policy_environment.step(action)
                episode_steps += 1
                episode_reward += float(reward)

                if terminated or truncated:
                    status = rollout_status(
                        terminated,
                        truncated,
                        info,
                    )

                renderer.update_scene(
                    environment.simulation.data,
                    camera=camera,
                )
                add_spawn_area_outline(
                    renderer.scene,
                    environment.simulation.spawn_config,
                )
                add_gripperframe_marker(
                    renderer.scene,
                    environment.simulation.data,
                )
                if not environment.reward_calculator.confirmed_grasp_seen:
                    add_active_orange_approach_target_marker(
                        renderer.scene,
                        environment.simulation.data,
                        waypoint_reached=bool(
                            info["orange_pregrasp_waypoint_reached"]
                        ),
                        approach_height_offset=approach_height_offset,
                    )
                frame = add_status_border(renderer.render(), status)

                if terminated or truncated:
                    frozen_terminal_frame = frame
            else:
                frame = frozen_terminal_frame

            video_writer.write(frame)

            if video_step % 100 == 0:
                print(
                    f"Rollout {task.rollout_index + 1:02d}/"
                    f"{ROLLOUT_COUNT}: rendered {video_step}/"
                    f"{MAXIMUM_EPISODE_STEPS} frames",
                    flush=True,
                )

    except BaseException:
        if video_writer is not None:
            video_writer.close(check_return_code=False)
        raise
    else:
        if video_writer is None:
            raise RuntimeError("video writer was not created")
        video_writer.close()
    finally:
        if renderer is not None:
            renderer.close()
        if environment is not None:
            environment.close()

    return RolloutResult(
        rollout_index=task.rollout_index,
        seed=task.seed,
        status=status,
        episode_steps=episode_steps,
        episode_reward=episode_reward,
        output_path=task.output_path,
    )


def run_parallel_rollouts(
    tasks: list[RolloutTask | DemonstrationTask],
    worker_count: int,
) -> list[RolloutResult]:
    """Run all rollout tasks in a promptly cancellable spawned pool."""
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=worker_count)
    results: list[RolloutResult] = []

    try:
        for result in pool.imap_unordered(render_rollout, tasks):
            results.append(result)
            print(
                f"Completed rollout {result.rollout_index + 1:02d}/"
                f"{len(tasks)}",
                flush=True,
            )
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    return sorted(results, key=lambda result: result.rollout_index)


def build_xstack_command(
    input_paths: Sequence[Path],
    output_path: Path,
) -> list[str]:
    """Build the FFmpeg command that combines panels in row-major order."""
    if len(input_paths) != ROLLOUT_COUNT:
        raise ValueError(
            f"expected {ROLLOUT_COUNT} rollout videos; "
            f"received {len(input_paths)}"
        )

    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for input_path in input_paths:
        command.extend(["-i", str(input_path)])

    xstack_filter = (
        f"xstack=inputs={ROLLOUT_COUNT}:"
        f"grid={GRID_COLUMNS}x{GRID_ROWS}:shortest=1[grid]"
    )
    command.extend(
        [
            "-filter_complex",
            xstack_filter,
            "-map",
            "[grid]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def combine_rollout_videos(
    input_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Combine completed rollout videos into the final grid."""
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"rollout video does not exist: {missing_paths[0]}"
        )

    subprocess.run(
        build_xstack_command(input_paths, output_path),
        check=True,
    )


def print_rollout_summary(results: Sequence[RolloutResult]) -> None:
    """Print each held-out rollout result and the overall success count."""
    print("\nRollout results:")
    for result in results:
        print(
            f"{result.rollout_index + 1:02d}: seed={result.seed} "
            f"status={result.status:<9} "
            f"steps={result.episode_steps:3d} "
            f"reward={result.episode_reward:+.6f}"
        )

    success_count = sum(result.status == SUCCESS for result in results)
    print(f"\nSuccesses: {success_count}/{ROLLOUT_COUNT}")
    print(f"Saved video: {OUTPUT_PATH}")


def write_blank_panel(output_path: Path, frame_count: int, frames_per_second: float) -> None:
    """Fill an unused grid cell for the full demonstration video duration."""
    writer = RawVideoWriter(output_path, PANEL_WIDTH, PANEL_HEIGHT, frames_per_second)
    frame = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    try:
        for _ in range(frame_count):
            writer.write(frame)
    except BaseException:
        writer.close(check_return_code=False)
        raise
    else:
        writer.close()


def main(arguments: Sequence[str] | None = None) -> None:
    options = parse_arguments(arguments)
    records = []
    if options.pretrain_data:
        records = select_demonstrations(PRETRAINING_DATA_DIRECTORY)
        intervals = [float(record["settings"]["action_interval"]) for record in records]
        if (not np.all(np.isfinite(intervals)) or min(intervals) <= 0
                or not np.allclose(intervals, intervals[0], rtol=0, atol=1e-12)):
            raise ValueError("Selected demonstrations must have the same positive action interval.")
        frame_count = max(record["steps"] for record in records) + 1
        frames_per_second = 1.0 / intervals[0]
    else:
        checkpoint_path = resolve_checkpoint_path(options.repeat_wandb)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PPO checkpoint does not exist: {checkpoint_path}. "
                "Run training first."
            )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Install it with `brew install ffmpeg`."
        )

    temporary_directory = temporary_rollout_directory()
    temporary_paths = rollout_video_paths(temporary_directory)
    staging_output_path = temporary_directory / OUTPUT_PATH.name
    if options.pretrain_data:
        tasks = [
            DemonstrationTask(index, temporary_paths[index], record,
                              PRETRAINING_DATA_DIRECTORY, frame_count)
            for index, record in enumerate(records)
        ]
        for index, record in enumerate(records):
            print(f"Panel {index + 1:02d}: training episode {record['uuid']} "
                  f"({record['steps']} recorded actions)", flush=True)
        source_description = f"saved training data in {PRETRAINING_DATA_DIRECTORY}"
    else:
        tasks = [
            RolloutTask(
                rollout_index=rollout_index,
                seed=BASE_EVALUATION_SEED + rollout_index,
                output_path=temporary_paths[rollout_index],
                checkpoint_path=checkpoint_path,
            )
            for rollout_index in range(ROLLOUT_COUNT)
        ]
        source_description = str(checkpoint_path)
    worker_count = min(options.workers, len(tasks))

    temporary_directory.mkdir(parents=True, exist_ok=False)
    try:
        print(
            f"Rendering {len(tasks)} rollouts with "
            f"{worker_count} spawned worker processes from "
            f"{source_description}",
            flush=True,
        )
        results = run_parallel_rollouts(tasks, worker_count)
        if options.pretrain_data:
            for path in temporary_paths[len(tasks):]:
                write_blank_panel(path, frame_count, frames_per_second)
        combine_rollout_videos(temporary_paths, staging_output_path)
        staging_output_path.replace(OUTPUT_PATH)
        if options.pretrain_data:
            print(f"Rendered {len(records)} saved demonstrations. Saved video: {OUTPUT_PATH}")
        else:
            print_rollout_summary(results)
    finally:
        cleanup_temporary_directory(temporary_directory)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
