#!/usr/bin/env python3
"""Behavior cloning for home-to-stack demonstrations, with PPO-compatible saving.

    .venv/bin/python scripts/pretrain_pickup.py
    .venv/bin/python scripts/pretrain_pickup.py --device mps
    .venv/bin/python scripts/pretrain_pickup.py --save
    .venv/bin/python scripts/pretrain_pickup.py --save pickup_v1
    .venv/bin/python scripts/pretrain_pickup.py --wandb
    .venv/bin/python scripts/pretrain_pickup.py --wandb SWEEP_ID
    .venv/bin/python scripts/pretrain_pickup.py --smoke
    .venv/bin/python scripts/pretrain_pickup.py --test

Install tqdm if needed: .venv/bin/python -m pip install tqdm

All supervised sequences and labels are materialized from saved .pt episodes
before creating DataLoaders. No simulator rollout is used to generate labels.
Each real timestep predicts its demonstrated action with causal attention.
Sequences are right-padded to at most 384 tokens; longer episodes use chunks
with stride 256. Repeated overlap supplies context but contributes loss only
in its first chunk. Padding never contributes to the loss.
Observations have 49 values, excluding the pregrasp waypoint-reached flag;
history tokens add four previous-action and three accepted-target values.
Pickup-only and older 50-value demonstrations are rejected without modifying
their files.
The existing data/test split is used as VALIDATION for loss and sweep selection;
it is therefore no longer an untouched final test set for these experiments.

Validation success is measured by running the learned deterministic policy in
MuJoCo on the validation episodes' recorded seeds, starting at the fixed gripper
home position configured in environment.py and ending with the orange cube
released in a stable stack on the blue cube. Saved demonstration starts may
be randomized. IK failures during validation episodes count as unsuccessful
trials; they do not abort supervised training.
Validation uses its own configurable jaw-alignment tolerance (5 degrees by
default), overriding the teacher's recorded tolerance for reset and actions.
The critic and exploration weights are frozen; only the shared transformer,
actor MLP, and bounded action-mean head are trained. Without --save, weights
and metrics stay in memory. --save [NAME] writes the final model, config, and
metric history to checkpoints/pretraining/NAME.zip, readable by PPO.load.
Omitting NAME uses default.zip. Use src/train.py --pretrained [PATH] to
initialize PPO from the learned actor; omitting PATH loads that default ZIP.
With --wandb --save NAME, each trial appends its W&B run ID to NAME.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Sequence
from uuid import UUID

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRETRAINING_CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "pretraining"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from action_observation_history import ActionObservationHistoryWrapper  # noqa: E402
from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy  # noqa: E402
from cartesian_actions import CartesianActionConfig  # noqa: E402
from environment import DEFAULT_SCENE_PATH, DEFAULT_START_POSITION, PHYSICS_STEPS_PER_ACTION  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402
from kinematics import IKConvergenceError  # noqa: E402
from observations import PRIVILEGED_OBSERVATION_SIZE  # noqa: E402
from randomization import CubeSpawnConfig  # noqa: E402
from rewards import StackRewardConfig  # noqa: E402
from success import StackSuccessConfig  # noqa: E402
from temporal_features import ActionObservationTransformer  # noqa: E402


# Live validation may accept a larger jaw-plane error than the scripted teacher.
VALIDATION_CLAW_YAW_TOLERANCE_DEGREES = 5.0


# Edit these defaults to configure ordinary training. Sweep parameters override
# only the fields named in WANDB_SWEEP_CONFIG; all other defaults still apply.
@dataclass(frozen=True)
class PretrainingConfig:
    data_directory: Path = REPOSITORY_ROOT / "data"
    seed: int = 0
    device: str = "auto"  # Prefer MPS when available; otherwise use CPU.
    cpu_threads: int = 4

    epochs: int = 600
    batch_size: int = 8  # Episodes/chunks per batch, not individual action labels.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_gradient_norm: float = 1.0
    gripper_loss_weight: float = 1.0

    history_length: int = 384
    sequence_stride: int = 256  # New chunk starts; overlap is context only.
    transformer_embedding_dim: int = 160
    transformer_layers: int = 6
    transformer_heads: int = 4
    transformer_feedforward_dim: int = 512
    # The frozen exploration MLP uses the same hidden sizes as the actor.
    actor_dim: int = 128
    actor_layers: int = 2

    print_interval_epochs: int = 10
    validation_interval_epochs: int = 10  # Simulator success evaluation cadence.
    validation_episodes: int = 200  # At most this many saved validation scenes.
    validation_claw_yaw_tolerance_degrees: float = VALIDATION_CLAW_YAW_TOLERANCE_DEGREES
    maximum_episode_steps: int = 900  # 45 seconds at 20 actions per second.
    dataloader_workers: int = 0  # Tensors are already in RAM; no disk work per batch.
    sweep_run_count: int | None = None  # None keeps the W&B agent running.
    save_name: str | None = None  # --save NAME opts into saving the final epoch.

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_directory", Path(self.data_directory).expanduser().resolve())
        for name in (
            "cpu_threads", "epochs", "batch_size", "history_length", "sequence_stride",
            "transformer_embedding_dim", "transformer_layers", "transformer_heads",
            "transformer_feedforward_dim", "actor_dim", "actor_layers",
            "print_interval_epochs", "validation_interval_epochs", "validation_episodes",
            "maximum_episode_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("seed", "dataloader_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.seed >= 2 ** 32:
            raise ValueError("The model/shuffle seed must be below 2**32 (episode seeds may be larger).")
        if self.sweep_run_count is not None and (
            type(self.sweep_run_count) is not int or self.sweep_run_count < 1
        ):
            raise ValueError("sweep_run_count must be None or a positive integer.")
        for name in ("learning_rate", "max_gradient_norm", "gripper_loss_weight", "weight_decay",
                     "validation_claw_yaw_tolerance_degrees"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (value == 0 and name != "weight_decay")):
                raise ValueError(f"{name} must be finite and positive (weight_decay may be zero).")
        if self.transformer_embedding_dim % self.transformer_heads:
            raise ValueError("transformer_embedding_dim must be divisible by transformer_heads.")
        if self.sequence_stride > self.history_length:
            raise ValueError("sequence_stride must not exceed history_length (no gaps between chunks).")
        if self.device not in ("auto", "cpu", "mps"):
            raise ValueError("device must be auto, cpu, or mps.")
        if self.save_name is not None:
            checkpoint_path(self.save_name)


WANDB_ENTITY_NAME = "jonathanlin"
WANDB_PROJECT_NAME = "robotics-pick_up_cube"
WANDB_SWEEP_CONFIG: dict[str, Any] = {
    "name": "PRETRAINING-waypoint_to_pickup",
    "program": "scripts/pretrain_pickup.py",
    "method": "bayes",
    "metric": {"name": "validation_success_rate", "goal": "maximize"},
    "parameters": {
        # This is the prior over learning rates; the search METHOD is Bayesian.
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [4, 8, 16]},
        "transformer_embedding_dim": {"values": [128, 160]},
        "transformer_layers": {"values": [2, 3, 4]},
        "transformer_feedforward_dim": {"values": [256, 512, 768]},
        "gripper_loss_weight": {"values": [0.5, 1.0, 2.0]},
    },
}

INPUT_NAMES = ("tokens", "valid", "episode_start")
TOKEN_SIZE = PRIVILEGED_OBSERVATION_SIZE + 4 + 3


# Dataset preparation: stored tensors only, never simulator-generated labels.


def read_episode_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    legacy = directory / "manifest.json"
    if legacy.is_file():
        content = json.loads(legacy.read_text())
        if content.get("format_version") != 1 or not isinstance(content.get("episodes"), list):
            raise ValueError("Invalid legacy manifest.json.")
        records.extend(content["episodes"])
    for path in sorted((directory / "manifests").glob("*.json")):
        content = json.loads(path.read_text())
        if content.get("format_version") != 2:
            raise ValueError(f"Unsupported manifest format: {path}")
        record = content["episode"]
        if path.stem != record["uuid"] or record["seed"] != UUID(record["uuid"]).int:
            raise ValueError(f"UUID/seed mismatch in {path}")
        records.append(record)
    if not records:
        raise ValueError(f"No demonstration manifests found in {directory}.")

    identifiers, seeds = set(), set()
    signature = None
    for record in records:
        identifier = record["uuid"]
        if str(UUID(identifier)) != identifier or identifier in identifiers:
            raise ValueError("Invalid or duplicate episode UUID.")
        seed = record["seed"]
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError("Invalid or duplicate seed: train/validation scenes must be disjoint.")
        if record["split"] not in ("train", "test"):
            raise ValueError("Each episode must belong to train or test (used as validation).")
        if type(record["steps"]) is not int or record["steps"] < 1:
            raise ValueError("Episode steps must be a positive integer.")
        identifiers.add(identifier)
        seeds.add(seed)
        settings = record["settings"]
        if settings.get("task") != "stack_orange_on_blue":
            raise ValueError("Pretraining requires full stack_orange_on_blue demonstrations; "
                             "legacy pickup-only data is unsupported. Generate a new dataset.")
        if (settings.get("start_at_orange_waypoint") is not False
                or settings.get("recovery_start_probability") != 0):
            raise ValueError("Full stacking pretraining requires home starts without recovery starts.")
        for name in ("start_position", "start_position_half_range"):
            if name not in settings:
                continue  # Older demonstrations predate start-distribution metadata.
            values = settings[name]
            if (not isinstance(values, (list, tuple)) or len(values) != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in values)
                    or (name == "start_position_half_range" and any(value < 0 for value in values))):
                raise ValueError(f"Demonstration {name} must be a finite XYZ vector"
                                 + (" with nonnegative values." if name.endswith("half_range") else "."))
        recorded_success_config = settings.get("success_config")
        if (not isinstance(recorded_success_config, dict)
                or set(recorded_success_config) != {field.name for field in fields(StackSuccessConfig)}):
            raise ValueError("Stack demonstrations require a complete success_config.")
        try:
            success_config = StackSuccessConfig(**recorded_success_config)
        except (TypeError, AssertionError) as error:
            raise ValueError("Stack demonstrations require a valid success_config.") from error
        stable_time = record.get("final_stack_stable_time")
        if (record["success"] is not True or isinstance(stable_time, bool)
                or not isinstance(stable_time, (int, float)) or not math.isfinite(stable_time)
                or stable_time < success_config.required_stable_time):
            raise ValueError("Pretraining requires successful released, stable stack demonstrations.")
        maximum_steps = settings.get("maximum_episode_steps")
        if type(maximum_steps) is not int or maximum_steps < record["steps"]:
            raise ValueError("Demonstration steps must fit the recorded maximum_episode_steps.")
        # Start distributions may differ across demonstrations. Their recorded
        # observations remain supervised inputs; live evaluation uses a fixed start.
        environment_keys = (
            "scene", "action_config", "spawn_config", "waypoint_height",
            "success_config", "maximum_episode_steps", "action_interval",
        )
        signature_settings = {key: settings[key] for key in environment_keys}
        # Older records omit optional action settings; compare their effective
        # defaults without rewriting the recorded demonstration metadata.
        signature_settings["action_config"] = asdict(
            CartesianActionConfig(**settings["action_config"])
        )
        current_signature = json.dumps(signature_settings, sort_keys=True)
        if signature is not None and current_signature != signature:
            raise ValueError("Episodes use different environment settings; use a consistent dataset.")
        signature = current_signature
        for name in ("observations", "accepted_targets", "actions"):
            if not (directory / record["split"] / name / f"{identifier}.pt").is_file():
                raise ValueError(f"Missing {name} tensor for episode {identifier}.")
    return records


def load_episode(directory: Path, record: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    steps = record["steps"]
    shapes = ((steps + 1, PRIVILEGED_OBSERVATION_SIZE), (steps + 1, 3), (steps, 4))
    tensors = []
    for name, shape in zip(("observations", "accepted_targets", "actions"), shapes):
        path = directory / record["split"] / name / f"{record['uuid']}.pt"
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError(f"{path} must be a finite float32 tensor with shape {shape}.")
        tensors.append(tensor)
    if torch.any(tensors[2].abs() > 1):
        raise ValueError("Demonstration actions must be normalized to [-1, 1].")
    return tuple(tensors)


def sequence_chunks(steps: int, history_length: int, sequence_stride: int) -> list[tuple[int, int, int]]:
    """Return (start, end, first_label) episode indices, with exclusive ends.

    Each action is supervised exactly once. Later chunks retain their overlap
    as attention context, and stop once the episode's final action is covered.
    """
    for name, value in (("steps", steps), ("history_length", history_length),
                        ("sequence_stride", sequence_stride)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if sequence_stride > history_length:
        raise ValueError("sequence_stride must not exceed history_length.")
    chunks = []
    first_label = 0
    for start in range(0, steps, sequence_stride):
        end = min(start + history_length, steps)
        chunks.append((start, end, first_label))
        if end == steps:
            break
        first_label = end
    return chunks


def materialize_sequences(directory: Path, records: Sequence[dict[str, Any]],
                          history_length: int, description: str,
                          *, sequence_stride: int = 256, show_progress: bool = True) -> TensorDataset:
    """Build all padded inputs, per-timestep labels, and loss masks in RAM."""
    chunk_indices = [sequence_chunks(record["steps"], history_length, sequence_stride)
                     for record in records]
    count = sum(len(chunks) for chunks in chunk_indices)
    if not count:
        raise ValueError(f"No examples found for {description}.")
    sequences = torch.zeros((count, history_length, TOKEN_SIZE), dtype=torch.float32)
    valid = torch.zeros((count, history_length), dtype=torch.float32)
    episode_start = torch.zeros_like(valid)
    labels = torch.zeros((count, history_length, 4), dtype=torch.float32)
    loss_mask = torch.zeros((count, history_length), dtype=torch.bool)
    row = 0
    with tqdm(total=count, desc=description, unit="sequences", disable=not show_progress) as progress:
        for record, chunks in zip(records, chunk_indices):
            observations, targets, actions = load_episode(directory, record)
            previous_actions = torch.cat((torch.zeros((1, 4)), actions[:-1]))
            # Terminal observation/target have no label and are excluded here.
            tokens = torch.cat((observations[:-1], previous_actions, targets[:-1]), dim=1)
            for first, end, first_label in chunks:
                length = end - first
                sequences[row, :length] = tokens[first:end]
                valid[row, :length] = 1
                episode_start[row, 0] = float(first == 0)
                labels[row, :length] = actions[first:end]
                loss_mask[row, first_label - first:length] = True
                row += 1
                progress.update(1)
    return TensorDataset(sequences, valid, episode_start, labels, loss_mask)


@dataclass
class PreparedData:
    training: TensorDataset
    validation: TensorDataset
    validation_records: list[dict[str, Any]]
    history_length: int
    sequence_stride: int


def prepare_data(config: PretrainingConfig, *, episode_limit: int | None = None,
                 show_progress: bool = True) -> PreparedData:
    records = read_episode_records(config.data_directory)
    training_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "test"]
    if episode_limit is not None:
        training_records = training_records[:episode_limit]
        validation_records = validation_records[:episode_limit]
    training = materialize_sequences(config.data_directory, training_records, config.history_length,
                                     "Calculating training tensors", sequence_stride=config.sequence_stride,
                                     show_progress=show_progress)
    validation = materialize_sequences(config.data_directory, validation_records, config.history_length,
                                       "Calculating validation tensors", sequence_stride=config.sequence_stride,
                                       show_progress=show_progress)
    return PreparedData(training, validation, validation_records, config.history_length, config.sequence_stride)


# Model and supervised optimization. No PPO rollout buffer or PPO updates.


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python session; use --device cpu or auto.")
    return torch.device(requested)


def create_policy(config: PretrainingConfig, device: torch.device) -> TanhBoundedMeanActorCriticPolicy:
    observation_space = spaces.Dict({
        "tokens": spaces.Box(-np.inf, np.inf, (config.history_length, TOKEN_SIZE), dtype=np.float32),
        "valid": spaces.Box(0, 1, (config.history_length,), dtype=np.float32),
        "episode_start": spaces.Box(0, 1, (config.history_length,), dtype=np.float32),
    })
    policy = TanhBoundedMeanActorCriticPolicy(
        observation_space, spaces.Box(-1, 1, (4,), dtype=np.float32),
        lr_schedule=lambda _: config.learning_rate,
        features_extractor_class=ActionObservationTransformer,
        features_extractor_kwargs={
            "embedding_dim": config.transformer_embedding_dim,
            "layer_count": config.transformer_layers,
            "head_count": config.transformer_heads,
            "feedforward_dim": config.transformer_feedforward_dim,
        },
        net_arch={"pi": [config.actor_dim] * config.actor_layers,
                  "vf": [config.actor_dim] * config.actor_layers},
        activation_fn=torch.nn.ReLU, share_features_extractor=True,
        use_sde=True, use_expln=True, squash_output=False,
        sde_log_std_init=(*([math.log(0.2 / 7.1)] * 3), math.log(0.5 / 7.1)),
    ).to(device)
    policy.requires_grad_(False)
    # The feature extractor is SHARED with the critic. Unfreeze it once while
    # leaving the separate value MLP/head and exploration log_std frozen.
    for component in (policy.features_extractor, policy.mlp_extractor.policy_net, policy.action_net):
        component.requires_grad_(True)
    # Demonstration MSE trains only action means. Keep the exploration MLP
    # explicitly frozen here; PPO fine-tuning trains a fresh copy later.
    policy.exploration_mlp.requires_grad_(False)
    policy.optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    return policy


def action_means(policy: TanhBoundedMeanActorCriticPolicy,
                 observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Predict a bounded action at every sequence position in one forward pass."""
    features = policy.features_extractor.forward_sequence(observations)
    latent = policy.mlp_extractor.forward_actor(features)
    return torch.tanh(policy.action_net(latent))


def run_epoch(policy: TanhBoundedMeanActorCriticPolicy, loader: DataLoader,
              config: PretrainingConfig, *, training: bool) -> dict[str, float]:
    policy.set_training_mode(training)
    totals = torch.zeros(3, device=policy.device)
    count = 0
    weights = torch.tensor([1, 1, 1, config.gripper_loss_weight], device=policy.device)
    with torch.set_grad_enabled(training):
        for tokens, valid, episode_start, labels, loss_mask in loader:
            # Trim all-padding columns on CPU before transferring this batch.
            # Tensors were already constructed; this does not generate new data.
            length = int(valid.sum(dim=1).max().item())
            observations = {name: tensor[:, :length].to(policy.device)
                            for name, tensor in zip(INPUT_NAMES, (tokens, valid, episode_start))}
            labels = labels[:, :length].to(policy.device)
            loss_mask = loss_mask[:, :length].to(device=policy.device, dtype=torch.bool)
            loss_mask = loss_mask & (observations["valid"] > 0.5)
            batch_count = int(loss_mask.sum().item())
            if not batch_count:
                raise ValueError("Every batch must contain supervised real timesteps.")
            if training:
                policy.optimizer.zero_grad(set_to_none=True)
            # Select BEFORE MSE: padding/overlap labels do not enter the loss,
            # even if their unused values are nonfinite. Context still gets
            # gradients through attention from the supervised later actions.
            predictions = action_means(policy, observations)
            squared_errors = (predictions[loss_mask] - labels[loss_mask]).square()
            loss = (squared_errors * weights).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy.parameters() if parameter.requires_grad],
                    config.max_gradient_norm, error_if_nonfinite=True,
                )
                policy.optimizer.step()
            totals += torch.stack((loss.detach(), squared_errors[:, :3].mean().detach(),
                                   squared_errors[:, 3].mean().detach())) * batch_count
            count += batch_count
    if not count:
        raise ValueError("A training/validation loader must not be empty.")
    values = (totals / count).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Training produced a nonfinite loss.")
    return dict(zip(("loss", "xyz_mse", "gripper_mse"), values))


def make_validation_environment(config: PretrainingConfig,
                                settings: dict[str, Any]) -> ActionObservationHistoryWrapper:
    scene = Path(settings["scene"])
    if not scene.is_absolute():
        scene = REPOSITORY_ROOT / scene
    environment = CubeStackGymEnvironment(
        scene_path=scene,
        maximum_episode_steps=config.maximum_episode_steps,
        start_at_orange_waypoint=False, recovery_start_probability=0.0,
        start_position=DEFAULT_START_POSITION, start_position_half_range=(0.0, 0.0, 0.0),
        action_config=replace(
            CartesianActionConfig(**settings["action_config"]),
            tool_yaw_tolerance=math.radians(config.validation_claw_yaw_tolerance_degrees),
        ),
        spawn_config=CubeSpawnConfig(**settings["spawn_config"]),
        reward_config=StackRewardConfig(approach_orange_height_offset=settings["waypoint_height"]),
        success_config=StackSuccessConfig(**settings["success_config"]),
    )
    action_interval = PHYSICS_STEPS_PER_ACTION * environment.simulation.model.opt.timestep
    if not math.isclose(action_interval, settings["action_interval"]):
        environment.close()
        raise ValueError("The simulator action interval differs from the recorded demonstrations.")
    return ActionObservationHistoryWrapper(environment, history_length=config.history_length)


def evaluate_success(policy: TanhBoundedMeanActorCriticPolicy,
                     records: Sequence[dict[str, Any]], config: PretrainingConfig) -> dict[str, float | int]:
    """Measure completed stacks and episodes with a valid grasp on held-out scenes."""
    selected = records[:config.validation_episodes]
    if not selected:
        raise ValueError("Success evaluation requires validation episodes.")
    required_stable_time = selected[0]["settings"]["success_config"]["required_stable_time"]
    old_mode, old_threads = policy.training, torch.get_num_threads()
    environment = None
    successes = grasp_successes = reset_failures = ik_failures = 0
    try:
        torch.set_num_threads(1)  # Single-environment inference and MuJoCo.
        policy.set_training_mode(False)
        environment = make_validation_environment(config, selected[0]["settings"])
        with torch.no_grad():
            for record in selected:
                try:
                    history, _ = environment.reset(seed=record["seed"])
                except IKConvergenceError:
                    reset_failures += 1
                    ik_failures += 1
                    continue
                except RuntimeError:
                    # A failed start counts as a failed trial, not a removed denominator.
                    reset_failures += 1
                    continue
                for _ in range(config.maximum_episode_steps):
                    length = int(history["valid"].sum())
                    inputs = {name: torch.as_tensor(history[name][:length], device=policy.device).unsqueeze(0)
                              for name in INPUT_NAMES}
                    # Inference uses only the latest real output; training
                    # above applies the same actor head at every real position.
                    action = policy._predict(inputs, deterministic=True)[0].cpu().numpy()
                    if not np.isfinite(action).all():
                        raise RuntimeError("Policy produced nonfinite validation actions.")
                    try:
                        history, _, terminated, truncated, info = environment.step(action)
                    except IKConvergenceError:
                        # An imperfect policy can request an unsolvable pose.
                        # End this trial as a failure, retaining its denominator
                        # and any grasp already seen, then evaluate the next seed.
                        ik_failures += 1
                        break
                    if terminated or truncated:
                        successes += int(bool(
                            terminated and not truncated and info["is_success"]
                            and not info["is_failure"] and not info["orange_currently_held"]
                            and info["stack_stable_time"] >= required_stable_time
                        ))
                        break
                # This physics-step latch means both jaws held orange clear of
                # the table at least once, even if it was later dropped/released.
                # Count episodes, not individual grasps or held action steps.
                grasp_successes += int(environment.unwrapped.simulation.confirmed_grasp_seen)
    finally:
        if environment is not None:
            environment.close()
        policy.set_training_mode(old_mode)
        torch.set_num_threads(old_threads)
    return {"validation_success_rate": successes / len(selected),
            "validation_successes": successes, "validation_episodes": len(selected),
            "validation_grasp_successes": grasp_successes,
            "validation_grasp_success_rate": grasp_successes / len(selected),
            "validation_reset_failures": reset_failures,
            "validation_ik_failures": ik_failures}


def checkpoint_path(save_name: str) -> Path:
    """Keep named archives inside checkpoints/pretraining; accept an optional .zip."""
    if (not isinstance(save_name, str) or not save_name.strip()
            or save_name != save_name.strip() or any(character in save_name for character in ("/", "\\", "\0"))):
        raise ValueError("--save must be a filename, such as pickup_v1 or pickup_v1.zip, not a path.")
    stem = save_name[:-4] if save_name.endswith(".zip") else save_name
    if stem in ("", ".", ".."):
        raise ValueError("--save requires a nonempty model name.")
    return PRETRAINING_CHECKPOINT_DIRECTORY / f"{stem}.zip"


def save_checkpoint(policy: TanhBoundedMeanActorCriticPolicy, config: PretrainingConfig,
                    history: list[dict[str, Any]], records: Sequence[dict[str, Any]]) -> Path:
    """Export learned weights and results in the same ZIP format as PPO training.

    Behavior cloning uses AdamW on only the actor/transformer. A fresh PPO
    container gives the archive an ordinary, empty Adam optimizer containing
    ALL parameters, so PPO.load can later train the critic and exploration too.
    The pretraining optimizer and expanded demonstration tensors are not saved.
    """
    import gymnasium as gym

    if config.save_name is None:
        raise ValueError("Set save_name (or --save NAME) before exporting a checkpoint.")
    destination = checkpoint_path(config.save_name)
    if not history or not records:
        raise ValueError("Saving requires completed training metrics and environment metadata.")

    class SpacesOnlyEnvironment(gym.Env):
        # PPO construction only needs spaces; exporting must not reset physics.
        observation_space = policy.observation_space
        action_space = policy.action_space

    parameters = policy._get_constructor_parameters()
    for key in ("observation_space", "action_space", "lr_schedule", "use_sde"):
        parameters.pop(key)
    model = PPO(
        type(policy), SpacesOnlyEnvironment(), device="cpu", verbose=0,
        policy_kwargs=parameters, use_sde=policy.use_sde,
        learning_rate=config.learning_rate, n_steps=512, batch_size=256,
        sde_sample_freq=8,
    )
    model.policy.load_state_dict(policy.state_dict(), strict=True)
    model.policy.reset_noise()
    model.pretraining_config = asdict(config)
    model.pretraining_config["data_directory"] = str(config.data_directory)
    model.pretraining_history = history
    # Retain the source demonstration's settings as provenance, including any
    # randomized starts; playback below uses the fixed evaluation start.
    model.pretraining_environment_settings = records[0]["settings"]
    # Existing rendering/diagnostic scripts use this metadata to recreate starts.
    model.pickup_training_config = {
        "history_length": config.history_length,
        "transformer_embedding_dim": config.transformer_embedding_dim,
        "transformer_layers": config.transformer_layers,
        "transformer_heads": config.transformer_heads,
        "transformer_feedforward_dim": config.transformer_feedforward_dim,
        "model_dim": config.actor_dim, "model_layers": config.actor_layers,
        "maximum_episode_steps": config.maximum_episode_steps,
        "task": "stack_orange_on_blue",
        "start_at_orange_waypoint": False, "recovery_start_probability": 0.0,
        "start_position": list(DEFAULT_START_POSITION),
        "start_position_half_range": [0.0, 0.0, 0.0],
        "success_config": records[0]["settings"]["success_config"],
        "reward_config": asdict(StackRewardConfig(
            approach_orange_height_offset=records[0]["settings"]["waypoint_height"],
        )),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after the entire archive has been written successfully.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}-",
                                     suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        model.save(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def train_pretraining(config: PretrainingConfig, *, prepared: PreparedData | None = None,
                      wandb_run: Any = None, show_progress: bool = True) -> list[dict[str, Any]]:
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        device = resolve_device(config.device)
        data = prepared if prepared is not None else prepare_data(config, show_progress=show_progress)
        if data.history_length != config.history_length or data.sequence_stride != config.sequence_stride:
            raise ValueError("Prepared sequences do not match the configured history length/stride.")
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(data.training, batch_size=config.batch_size, shuffle=True,
                                  generator=generator, num_workers=config.dataloader_workers)
        validation_loader = DataLoader(data.validation, batch_size=config.batch_size, shuffle=False,
                                       num_workers=config.dataloader_workers)
        policy = create_policy(config, device)
        training_actions = int(data.training.tensors[-1].sum().item())
        validation_actions = int(data.validation.tensors[-1].sum().item())
        saving = (f"Final checkpoint: {checkpoint_path(config.save_name)}"
                  if config.save_name is not None else "No weights will be saved.")
        print(f"Device: {device}; training sequences: {len(data.training):,} ({training_actions:,} actions); "
              f"validation sequences: {len(data.validation):,} ({validation_actions:,} actions). {saving}", flush=True)
        if wandb_run is not None:
            wandb_run.config.update({"resolved_device": str(device), "training_sequences": len(data.training),
                                     "validation_sequences": len(data.validation),
                                     "training_action_labels": training_actions,
                                     "validation_action_labels": validation_actions})
            wandb_run.define_metric("epoch")
            wandb_run.define_metric("*", step_metric="epoch")
        history = []
        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(policy, train_loader, config, training=True)
            validation_metrics = run_epoch(policy, validation_loader, config, training=False)
            metrics = {"epoch": epoch,
                       **{f"train_{key}": value for key, value in train_metrics.items()},
                       **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            if epoch % config.validation_interval_epochs == 0 or epoch == config.epochs:
                metrics.update(evaluate_success(policy, data.validation_records, config))
            metrics["epoch_seconds"] = time.perf_counter() - started
            history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics)
            if (epoch % config.print_interval_epochs == 0 or epoch == config.epochs
                    or "validation_success_rate" in metrics):
                success = ""
                if "validation_success_rate" in metrics:
                    success = (f" | validation success={metrics['validation_success_rate']:.1%}"
                               f" | orange grasp={metrics['validation_grasp_successes']}/"
                               f"{metrics['validation_episodes']}"
                               f" ({metrics['validation_grasp_success_rate']:.1%})"
                               f" | IK failures={metrics['validation_ik_failures']}/"
                               f"{metrics['validation_episodes']}")
                print(f"Epoch {epoch}/{config.epochs} | train loss={metrics['train_loss']:.6f} "
                      f"| validation loss={metrics['validation_loss']:.6f}{success}", flush=True)
        if wandb_run is not None:
            # Optimize the final evaluated policy, rather than its luckiest earlier score.
            wandb_run.summary["validation_success_rate"] = history[-1]["validation_success_rate"]
            wandb_run.summary["validation_grasp_successes"] = history[-1]["validation_grasp_successes"]
            wandb_run.summary["validation_grasp_success_rate"] = history[-1]["validation_grasp_success_rate"]
            wandb_run.summary["validation_ik_failures"] = history[-1]["validation_ik_failures"]
        if config.save_name is not None:
            destination = save_checkpoint(policy, config, history, data.validation_records)
            print(f"Saved final pretraining checkpoint: {destination}", flush=True)
        return history
    finally:
        torch.set_num_threads(previous_threads)


def run_wandb_sweep(config: PretrainingConfig, sweep_id: str | None = None) -> None:
    import wandb

    valid_fields = {field.name for field in fields(PretrainingConfig)}
    unknown = set(WANDB_SWEEP_CONFIG["parameters"]) - valid_fields
    if unknown:
        raise ValueError(f"Unknown sweep fields: {sorted(unknown)}")
    # Cache already-materialized tensors within this agent. Changing architecture
    # or batch size does not require rebuilding identical sequence tensors.
    cache: dict[tuple[Path, int, int], PreparedData] = {}

    def trial() -> None:
        defaults = asdict(config)
        defaults["data_directory"] = str(config.data_directory)
        with wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                        config=defaults, save_code=False) as run:
            sampled = {name: run.config[name] for name in valid_fields if name in run.config}
            trial_config = replace(config, **sampled)
            if trial_config.save_name is not None:
                # Every trial has its own archive, including parallel workers.
                stem = checkpoint_path(trial_config.save_name).stem
                trial_config = replace(trial_config, save_name=f"{stem}-{run.id}")
            key = (trial_config.data_directory, trial_config.history_length, trial_config.sequence_stride)
            if key not in cache:
                # Keep only one expanded dataset when sweeping history length.
                cache.clear()
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(trial_config.cpu_threads)
                    cache[key] = prepare_data(trial_config)
                finally:
                    torch.set_num_threads(previous_threads)
            train_pretraining(trial_config, prepared=cache[key], wandb_run=run)

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=WANDB_SWEEP_CONFIG,
                               project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME)
    elif "/" not in sweep_id:
        sweep_id = f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}"
    wandb.agent(sweep_id, function=trial, count=config.sweep_run_count)


# Small end-to-end smoke run and dedicated test-suite entry point.


def run_smoke(config: PretrainingConfig) -> list[dict[str, Any]]:
    """Exercise saved data -> learning -> real simulator, not a quality benchmark."""
    smoke_config = replace(
        config, epochs=2, batch_size=4, history_length=8, sequence_stride=4,
        transformer_embedding_dim=16, transformer_layers=1, transformer_heads=2,
        transformer_feedforward_dim=32, actor_dim=16, actor_layers=1,
        cpu_threads=1, validation_episodes=1, maximum_episode_steps=20,
        dataloader_workers=0,
    )
    prepared = prepare_data(smoke_config, episode_limit=2)
    result = train_pretraining(smoke_config, prepared=prepared)
    if len(result) != 2 or not all(math.isfinite(row["train_loss"]) for row in result):
        raise AssertionError("Smoke training did not return finite losses for two epochs.")
    if not 0 <= result[-1]["validation_success_rate"] <= 1:
        raise AssertionError("Invalid validation success rate.")
    print("Smoke passed: tensor preparation, optimization, and simulator evaluation."
          + ("" if smoke_config.save_name is not None else " No weights saved."))
    return result


def run_tests() -> bool:
    """Run the dedicated pretraining tests in a separate process."""
    import subprocess

    test_files = sorted((REPOSITORY_ROOT / "test").glob("test_pretrain_*.py"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *(str(path) for path in test_files), "-q"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return result.returncode == 0


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wandb", nargs="?", const="", default=None, metavar="SWEEP_ID",
                      help="Create a Bayesian sweep, or join one with an optional existing ID.")
    mode.add_argument("--smoke", action="store_true", help="Tiny end-to-end run using at most two episodes per split.")
    mode.add_argument("--test", action="store_true", help="Run test/test_pretrain_*.py without W&B or saved weights.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--sweep-count", type=int, default=None)
    parser.add_argument("--save", dest="save_name", nargs="?", const="default", default=None, metavar="NAME",
                        help="Save the final policy and results as checkpoints/pretraining/NAME.zip; "
                             "omitting NAME uses default.zip. "
                             "W&B trials append their run ID.")
    parsed = parser.parse_args(arguments)
    if parsed.save_name is not None:
        if parsed.test:
            parser.error("--save cannot be combined with --test; use --smoke to test a saved model.")
        try:
            checkpoint_path(parsed.save_name)
        except ValueError as error:
            parser.error(str(error))
    return parsed


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_arguments(arguments)
    if args.test:
        raise SystemExit(0 if run_tests() else 1)
    overrides = {name: getattr(args, name) for name in ("device", "epochs", "batch_size", "learning_rate")
                 if getattr(args, name) is not None}
    if args.data_dir is not None:
        overrides["data_directory"] = args.data_dir
    if args.sweep_count is not None:
        overrides["sweep_run_count"] = args.sweep_count
    if args.save_name is not None:
        overrides["save_name"] = args.save_name
    config = PretrainingConfig(**overrides)
    if args.smoke:
        run_smoke(config)
    elif args.wandb is not None:
        run_wandb_sweep(config, args.wandb if args.wandb != "" else None)
    else:
        train_pretraining(config)


if __name__ == "__main__":
    main()
