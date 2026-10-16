"""Offline supervised training for the shared perception and action models.

Tensor files are loaded once before building DataLoaders; training never rolls
out a simulator to derive labels. Each dataset has train/ and validation/
subdirectories with .pt dictionaries. Vision shards contain wrist_images,
external_images, joint_positions, cube_poses. Images are RGB uint8 NHWC or
float32 NCHW in [0, 1]. Action files each contain one episode's observations
[T,41], prev_accepted_targets [T,3], and intended actions [T,4]. Rows refer to
observations before executing the action at that same row.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import RandAugment
from tqdm import tqdm

from action_policy import ActionPolicy, masked_action_loss
from schemas import HISTORY_TOKEN_SIZE, OBSERVATION_SIZE
from schemas.config import ActionPolicyConfig, ActionTrainingConfig, ProjectConfig, SupervisedTrainingConfig, VisionModelConfig, VisionTrainingConfig
from vision_augmentation import AugmentedVisionDataset
from vision_model import VisionModel, cube_pose_loss


CHECKPOINT_FORMAT_VERSION = 1


def _tensor_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise ValueError(f"No tensor files found in {directory}.")
    return files


def _load_tensors(path: Path) -> dict[str, Tensor]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a tensor dictionary.")
    return record


def _require_tensor(record: dict[str, Tensor], key: str, dimensions: tuple[int | None, ...], path: Path) -> Tensor:
    value = record.get(key)
    if not isinstance(value, Tensor) or value.ndim != len(dimensions):
        raise ValueError(f"{path}: {key} must be a tensor with shape {dimensions}.")
    if any(size is not None and actual != size for actual, size in zip(value.shape, dimensions)):
        raise ValueError(f"{path}: {key} must have shape {dimensions}; got {tuple(value.shape)}.")
    return value.detach().cpu()


def load_vision_dataset(directory: Path, show_progress: bool) -> TensorDataset:
    """Materialize image/pose tensors; normalize pixels later per minibatch.

    Keeping source uint8 images until a batch is used avoids making the whole
    image dataset four times larger. This performs no image resizing/cropping.
    All shards must use the same image dtype and layout.
    """
    wrist_parts, external_parts, joint_parts, pose_parts = [], [], [], []
    for path in tqdm(_tensor_files(directory), desc=f"Loading vision {directory.name}", unit="shard", disable=not show_progress):
        record = _load_tensors(path)
        pair = []
        for key in ("wrist_images", "external_images"):
            value = record.get(key)
            if not isinstance(value, Tensor):
                raise ValueError(f"{path}: {key} must be a tensor.")
            if value.dtype == torch.uint8:
                value = _require_tensor(record, key, (None, 300, 300, 3), path).permute(0, 3, 1, 2).contiguous()
            elif value.dtype == torch.float32:
                value = _require_tensor(record, key, (None, 3, 300, 300), path)
            else:
                raise ValueError(f"{path}: {key} must be RGB uint8 NHWC or normalized float32 NCHW.")
            pair.append(value)
        count = pair[0].shape[0]
        if count < 1 or pair[1].shape[0] != count or pair[0].dtype != pair[1].dtype:
            raise ValueError(f"{path}: both views must have the same nonzero count and dtype.")
        if wrist_parts and pair[0].dtype != wrist_parts[0].dtype:
            raise ValueError("All vision shards must have the same image dtype.")
        wrist_parts.append(pair[0])
        external_parts.append(pair[1])
        joint_parts.append(_require_tensor(record, "joint_positions", (count, 6), path).float())
        pose_parts.append(_require_tensor(record, "cube_poses", (count, 14), path).float())
    return TensorDataset(*(torch.cat(parts, dim=0) for parts in (wrist_parts, external_parts, joint_parts, pose_parts)))


def build_action_windows(observations: Tensor, prev_accepted_targets: Tensor, actions: Tensor,
                         history_length: int, stride: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build padded causal sequences, supervising each episode action once.

    With context 384 and stride 256, windows are [0:384], [256:640], ... .
    The overlap provides context in later windows, but its repeated labels are
    masked out. A chunk retains its original preceding action, while only the
    actual episode's first token receives an episode-start embedding.
    """
    count = len(observations)
    if count < 1 or observations.shape != (count, OBSERVATION_SIZE):
        raise ValueError(f"observations must have nonempty shape [time, {OBSERVATION_SIZE}].")
    if prev_accepted_targets.shape != (count, 3) or actions.shape != (count, 4):
        raise ValueError("Action episodes must have matching [time, 3] targets and [time, 4] actions.")
    if not 1 <= stride <= history_length:
        raise ValueError("stride must be between 1 and history_length.")
    observations = observations.detach().cpu().float()
    prev_accepted_targets = prev_accepted_targets.detach().cpu().float()
    actions = actions.detach().cpu().float()
    previous = torch.zeros_like(actions)
    previous[1:] = actions[:-1]
    episode_tokens = torch.cat((observations, previous, prev_accepted_targets), dim=-1)
    windows = []
    start, previous_end = 0, 0
    while previous_end < count:
        end = min(start + history_length, count)
        length = end - start
        tokens = torch.zeros(history_length, HISTORY_TOKEN_SIZE)
        valid = torch.zeros(history_length, dtype=torch.bool)
        episode_start = torch.zeros_like(valid)
        labels = torch.zeros(history_length, 4)
        loss_mask = torch.zeros_like(valid)
        tokens[:length] = episode_tokens[start:end]
        valid[:length] = True
        episode_start[0] = start == 0
        labels[:length] = actions[start:end]
        loss_mask[max(0, previous_end - start):length] = True
        windows.append((tokens, valid, episode_start, labels, loss_mask))
        previous_end = end
        start += stride
    return tuple(torch.stack(items) for items in zip(*windows))


def load_action_dataset(directory: Path, history_length: int, stride: int, show_progress: bool) -> TensorDataset:
    parts = []
    for path in tqdm(_tensor_files(directory), desc=f"Calculating {directory.name} tensors", unit="episode", disable=not show_progress):
        record = _load_tensors(path)
        observations = _require_tensor(record, "observations", (None, OBSERVATION_SIZE), path)
        count = observations.shape[0]
        targets = _require_tensor(record, "prev_accepted_targets", (count, 3), path)
        actions = _require_tensor(record, "actions", (count, 4), path)
        parts.append(build_action_windows(observations, targets, actions, history_length=history_length, stride=stride))
    return TensorDataset(*(torch.cat(group, dim=0) for group in zip(*parts)))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return device


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def create_loader(dataset: Dataset, config: SupervisedTrainingConfig, training: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed) if training else None
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training and config.shuffle_train,
                      num_workers=config.num_workers, generator=generator, pin_memory=config.pin_memory,
                      drop_last=False)


def create_loaders(training: Dataset, validation: Dataset,
                   config: SupervisedTrainingConfig) -> tuple[DataLoader, DataLoader]:
    return (
        create_loader(dataset=training, config=config, training=True),
        create_loader(dataset=validation, config=config, training=False),
    )


def create_vision_training_loader(config: VisionTrainingConfig, show_progress: bool,
                                  directory: Path | None = None) -> DataLoader:
    """Load training tensors once and apply fresh augmentation as samples are read."""
    dataset = AugmentedVisionDataset(
        dataset=load_vision_dataset(directory if directory is not None else config.data_dir / "train",
                                    show_progress=show_progress),
        transform=RandAugment(num_ops=config.randaugment.num_ops, magnitude=config.randaugment.magnitude,
                              num_magnitude_bins=31),
    )
    return create_loader(dataset=dataset, config=config, training=True)


def run_action_epoch(model: ActionPolicy, loader: DataLoader, device: torch.device,
                     optimizer: torch.optim.Optimizer | None,
                     on_batch: Callable[[float], None] | None = None) -> float:
    model.train(optimizer is not None)
    total_loss, total_count = 0.0, 0
    for tokens, valid, episode_start, labels, loss_mask in loader:
        # Remove columns padded for every item in this batch, without changing
        # relative token positions or the individual padding/loss masks.
        length = int(valid.sum(dim=1).max())
        tokens, valid, episode_start, labels, loss_mask = (
            tensor[:, :length].to(device) for tensor in (tokens, valid, episode_start, labels, loss_mask)
        )
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(tokens, valid, episode_start)
            loss = masked_action_loss(prediction, labels, loss_mask)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        count = int(loss_mask.sum())
        total_loss += float(loss.detach()) * count
        total_count += count
        if on_batch is not None and total_count:
            on_batch(total_loss / total_count)
    if total_count == 0:
        raise ValueError("Action loader contains no supervised timestamps.")
    return total_loss / total_count


def run_vision_epoch(model: VisionModel, loader: DataLoader, device: torch.device,
                     optimizer: torch.optim.Optimizer | None,
                     on_batch: Callable[[float], None] | None = None, *,
                     position_weight: float = 200.0, orientation_weight: float = 1.0) -> float:
    model.train(optimizer is not None)
    total_loss, total_count = 0.0, 0
    for wrist, external, joints, poses in loader:
        images_are_bytes = wrist.dtype == torch.uint8
        wrist, external = wrist.to(device=device, dtype=torch.float32), external.to(device=device, dtype=torch.float32)
        if images_are_bytes:
            wrist, external = wrist / 255.0, external / 255.0
        joints, poses = joints.to(device), poses.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(wrist, external, joints)
            loss = cube_pose_loss(prediction, poses, position_weight=position_weight,
                                  orientation_weight=orientation_weight)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        count = len(joints)
        total_loss += float(loss.detach()) * count
        total_count += count
        if on_batch is not None and total_count:
            on_batch(total_loss / total_count)
    if total_count == 0:
        raise ValueError("Vision loader contains no examples.")
    return total_loss / total_count


def save_model_checkpoint(model: ActionPolicy | VisionModel, model_type: str, checkpoint_path: Path,
                          epoch: int, validation_loss: float) -> None:
    """Save one self-contained file, overwriting a previous checkpoint name."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": model_type,
        "model_config": model.config.model_dump(mode="json"),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": epoch,
        "validation_loss": validation_loss,
    }, checkpoint_path)


def _read_checkpoint(path: Path, model_type: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {path}.")
    if checkpoint.get("model_type") != model_type:
        raise ValueError(f"Expected {model_type} checkpoint: {path}.")
    return checkpoint


def load_action_checkpoint(path: Path, device: str | torch.device) -> ActionPolicy:
    checkpoint = _read_checkpoint(path, "action")
    config = ActionPolicyConfig.model_validate_json(json.dumps(checkpoint["model_config"]))
    model = ActionPolicy(config=config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def load_vision_checkpoint(path: Path, device: str | torch.device) -> VisionModel:
    checkpoint = _read_checkpoint(path, "vision")
    config = VisionModelConfig.model_validate_json(json.dumps(checkpoint["model_config"]))
    # The complete frozen ResNet weights are already in this checkpoint. Avoid
    # downloading ImageNet weights merely to overwrite them during loading.
    model = VisionModel(config=config.model_copy(update={"backbone_weights": None}), backbone=None)
    model.config = config
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def train_action(project: ProjectConfig) -> list[dict[str, float | int]]:
    config: ActionTrainingConfig = project.training.action
    seed_training(seed=config.seed)
    device = resolve_device(name=config.device)
    training = load_action_dataset(config.data_dir / "train", history_length=project.action_policy.history_length,
                                   stride=config.stride, show_progress=True)
    validation = load_action_dataset(config.data_dir / "validation", history_length=project.action_policy.history_length,
                                     stride=config.stride, show_progress=True)
    loaders = create_loaders(training=training, validation=validation, config=config)
    model = ActionPolicy(config=project.action_policy).to(device)
    return _train(model=model, model_type="action", loaders=loaders, config=config, device=device)


def train_vision(project: ProjectConfig, log_file: Path | None = None) -> list[dict[str, float | int]]:
    config = project.training.vision
    seed_training(seed=config.seed)
    device = resolve_device(name=config.device)
    training_loader = create_vision_training_loader(config=config, show_progress=True)
    validation = load_vision_dataset(config.data_dir / "validation", show_progress=True)
    loaders = (training_loader, create_loader(dataset=validation, config=config, training=False))
    model = VisionModel(config=project.vision_model, backbone=None).to(device)
    return _train(model=model, model_type="vision", loaders=loaders, config=config, device=device, log_file=log_file)


def _train(model: ActionPolicy | VisionModel, model_type: str, loaders: tuple[DataLoader, DataLoader],
           config: SupervisedTrainingConfig, device: torch.device,
           log_file: Path | None = None) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=config.learning_rate, betas=config.optimizer_betas, eps=config.optimizer_eps,
                                 weight_decay=config.weight_decay)
    run_epoch = run_action_epoch if model_type == "action" else partial(
        run_vision_epoch, position_weight=config.position_loss_weight,
        orientation_weight=config.orientation_loss_weight,
    )
    metrics = []
    training_loss = validation_loss = float("nan")
    print(f"Device: {device}; training examples: {len(loaders[0].dataset):,}; validation examples: {len(loaders[1].dataset):,}")
    validation_runs = config.epochs // config.validation_interval + bool(config.epochs % config.validation_interval)
    total_batches = config.epochs * len(loaders[0]) + validation_runs * len(loaders[1])

    def loss_text(loss: float) -> str:
        return "NaN" if np.isnan(loss) else f"{loss:.6f}"

    with ExitStack() as stack:
        log_stream = None
        if log_file is not None:
            log_file = log_file.expanduser()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_stream = stack.enter_context(log_file.open("a", encoding="utf-8"))

        def log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
            if log_stream is not None:
                log_stream.write(f"epoch={epoch} train_loss={loss_text(train_loss)} val_loss={loss_text(val_loss)}\n")
                log_stream.flush()

        log_epoch(0, training_loss, validation_loss)
        progress = stack.enter_context(tqdm(
            total=total_batches, desc=f"Epoch 0/{config.epochs}", unit="batch", dynamic_ncols=True,
            postfix={"train_loss": "NaN", "val_loss": "NaN"},
        ))
        def update_training(loss: float) -> None:
            progress.set_postfix(train_loss=loss_text(loss), val_loss=loss_text(validation_loss), refresh=False)
            progress.update(1)

        def update_validation(_loss: float) -> None:
            progress.update(1)

        for epoch in range(1, config.epochs + 1):
            epoch_validation_loss = float("nan")
            progress.set_description(f"Epoch {epoch}/{config.epochs} training", refresh=False)
            training_loss = run_epoch(model=model, loader=loaders[0], device=device, optimizer=optimizer,
                                      on_batch=update_training)
            if epoch % config.validation_interval == 0 or epoch == config.epochs:
                progress.set_description(f"Epoch {epoch}/{config.epochs} validation")
                validation_loss = epoch_validation_loss = run_epoch(
                    model=model, loader=loaders[1], device=device, optimizer=None, on_batch=update_validation,
                )
                metrics.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
            progress.set_description(f"Epoch {epoch}/{config.epochs}", refresh=False)
            progress.set_postfix(train_loss=loss_text(training_loss), val_loss=loss_text(validation_loss))
            log_epoch(epoch, training_loss, epoch_validation_loss)
    save_model_checkpoint(model=model, model_type=model_type, checkpoint_path=config.checkpoint_path,
                          epoch=config.epochs, validation_loss=validation_loss)
    print(f"Saved {model_type} checkpoint: {config.checkpoint_path}")
    return metrics
