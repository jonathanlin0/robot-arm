"""Generate additive, full-episode action demonstrations in the active schema.

Frozen vision supplies cube features from simulated cameras; robot telemetry
and scripted teacher actions come from the simulator. Only successful episodes
are published to the dataset. No trained action model is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import mujoco
import torch
from tqdm import tqdm

from action_demonstrations import ActionEpisode, DemonstrationFailure, collect_action_episode
from action_observation import VisionActionObserver
from environment import CubeStackEnvironment
from kinematics import IKConvergenceError
from schemas.config import ProjectConfig
from training import load_vision_checkpoint, resolve_device


def generation_settings(project: ProjectConfig) -> dict[str, Any]:
    return {
        "scene_path": str(project.environment.scene_path),
        "episode_seconds": project.environment.episode_seconds,
        "control_hz": project.loop.control_hz,
        "cartesian_controller": project.cartesian_controller.model_dump(mode="json"),
        "action_data": project.environment.action_data.model_dump(mode="json"),
    }


def save_action_episode(*, episode: ActionEpisode, project: ProjectConfig,
                        episode_id: UUID, split: str,
                        vision_checkpoint: Path | None = None) -> dict[str, Any]:
    """Atomically publish one complete episode without replacing existing data."""
    if split not in ("train", "validation"):
        raise ValueError("Action episodes must belong to train or validation.")
    episode.validate()
    metadata = {
        **episode.metadata(), "schema_version": 2 if episode.observation_source == "vision_model" else 1,
        "uuid": str(episode_id),
        "seed": episode_id.int, "split": split, "task": "stack_orange_on_blue",
        "observation_source": episode.observation_source,
        "cube_velocity_source": ("vision_position_differences" if episode.observation_source == "vision_model"
                                 else "simulation_ground_truth"),
        "action_source": "scripted_teacher_intended",
        "settings": generation_settings(project),
    }
    if vision_checkpoint is not None:
        metadata["vision_checkpoint"] = str(vision_checkpoint.expanduser().resolve())
    # Restrict metadata to portable values accepted by weights_only loaders.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    record = {
        "observations": episode.observations,
        "prev_accepted_targets": episode.prev_accepted_targets,
        "actions": episode.actions,
        "metadata": metadata,
    }
    directory = project.training.action.data_dir / split
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{episode_id}.pt"
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=f".{episode_id}-", suffix=".tmp",
                                dir=directory, delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is an atomic no-clobber publication. The
        # trainer's *.pt glob cannot see a partially written temporary file.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {**metadata, "path": str(destination)}


def generate_action_data(*, project: ProjectConfig, examples: int,
                         validation_examples: int | None = None,
                         vision_checkpoint: Path | None = None, device: str | None = None,
                         renderer_factory: Callable[..., Any] = mujoco.Renderer) -> list[dict[str, Any]]:
    """Record vision-based successes with a frozen checkpoint loaded once.

    The teacher and success checks retain privileged physics state. Saved cube
    inputs use the same camera inference, yaw normalization, and estimated
    position-derived velocity as runtime. Failed perception never falls back
    to simulator observations. The checkpoint defaults to vision training's
    configured output path; the device defaults to its training device.
    """
    if type(examples) is not int or examples < 1:
        raise ValueError("examples must be a positive integer")
    if validation_examples is not None and (
        type(validation_examples) is not int or not 0 <= validation_examples <= examples
    ):
        raise ValueError("validation_examples must be an integer between zero and examples")
    config = project.environment.action_data
    validation_count = (math.floor(examples * (1.0 - config.train_fraction) + 0.5)
                        if validation_examples is None else validation_examples)
    splits = ["train"] * (examples - validation_count) + ["validation"] * validation_count
    np.random.default_rng().shuffle(splits)
    checkpoint = (project.training.vision.checkpoint_path if vision_checkpoint is None
                  else vision_checkpoint).expanduser().resolve()
    inference_device = resolve_device(project.training.vision.device if device is None else device)
    tqdm.write(f"Loading vision checkpoint for action data generation: {checkpoint} ({inference_device})")
    # Fail before creating a simulator, renderers, or output files if the saved
    # model is missing or incompatible with the current ten-output contract.
    vision = load_vision_checkpoint(checkpoint, inference_device)
    vision.requires_grad_(False).eval()
    records = []
    attempts = failures = 0
    environment = None
    vision_observer = None
    with ExitStack() as resources, tqdm(total=examples, desc="Generating action episodes",
                                       unit="episode", dynamic_ncols=True) as progress:
        for index, split in enumerate(splits, start=1):
            for _ in range(config.maximum_attempts_per_example):
                episode_id = uuid4()
                attempts += 1
                try:
                    if environment is None:
                        environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
                        vision_observer = resources.enter_context(VisionActionObserver(
                            environment=environment, vision_model=vision, device=inference_device,
                            maximum_frame_age_seconds=project.cameras.maximum_frame_age_seconds,
                            renderer_factory=renderer_factory,
                        ))
                    episode = collect_action_episode(project=project, environment=environment, seed=episode_id.int,
                                                     vision_observer=vision_observer)
                except IKConvergenceError as error:
                    diagnostic = {
                        "uuid": str(episode_id), "seed": episode_id.int,
                        "error_type": type(error).__name__, "message": str(error),
                        "attempts": attempts, "completed_episodes": len(records),
                        "discarded_attempts": failures, "settings": generation_settings(project),
                        "vision_checkpoint": str(checkpoint),
                        "diagnostics": error.diagnostics,
                    }
                    path = project.training.action.data_dir / "diagnostics" / f"ik-failure-{episode_id}.json"
                    tqdm.write(f"[IK_CONVERGENCE_FAILURE] episode={episode_id} "
                               f"stage={error.diagnostics.get('stage', 'unknown')} "
                               f"step={error.diagnostics.get('episode_step', 'unknown')}: {error}")
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("x", encoding="utf-8") as stream:
                            json.dump(diagnostic, stream, indent=2, allow_nan=False)
                            stream.write("\n")
                        tqdm.write(f"Diagnostic saved to {path}")
                    except OSError as log_error:
                        tqdm.write(f"Could not save IK diagnostic: {log_error}")
                    tqdm.write(f"Generation stopped; {len(records)} new completed episodes remain saved.")
                    raise
                except DemonstrationFailure as error:
                    failures += 1
                    progress.set_postfix(attempts=attempts, discarded=failures, refresh=False)
                    tqdm.write(f"Discarded {episode_id}: {error}")
                    continue
                if episode.observation_source != "vision_model":
                    raise ValueError("Action data generation requires vision_model observations.")
                record = save_action_episode(episode=episode, project=project, episode_id=episode_id,
                                             split=split, vision_checkpoint=checkpoint)
                records.append(record)
                progress.update(1)
                progress.set_postfix(attempts=attempts, discarded=failures, refresh=False)
                tqdm.write(f"[{index}/{examples}] {split} {episode_id} steps={record['steps']} "
                           f"stack_stable={record['stable_time']:.3f}s "
                           f"disturbances={len(record['disturbance_events'])} "
                           f"recoveries={record['recovery_count']} waypoint_retries={record['waypoint_retry_count']}")
                break
            else:
                raise RuntimeError(
                    f"Could not collect example {index} after {config.maximum_attempts_per_example} attempts. "
                    f"The {len(records)} new completed episodes remain saved."
                )
    tqdm.write(f"Added {examples} episodes ({examples - validation_count} train, "
               f"{validation_count} validation); {failures}/{attempts} attempts discarded. "
               f"Saved in {project.training.action.data_dir}.")
    return records
