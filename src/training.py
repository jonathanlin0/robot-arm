"""Offline supervised training for the shared perception and action models.

Tensor files are loaded once before building DataLoaders; training never rolls
out a simulator to derive labels. Action validation also evaluates the current
policy in image-based simulated episodes. Each dataset has train/ and validation/
subdirectories with .pt dictionaries. Vision shards contain wrist_images,
external_images, joint_positions, cube_poses [T,10] (XYZ and yaw pairs per cube).
Legacy [T,14] quaternion labels are converted in memory. Images are RGB uint8 NHWC or
float32 NCHW in [0, 1]. Action files each contain one episode's observations
[T,37] (vision-based policy features) or legacy [T,41] simulator observations,
prev_accepted_targets [T,3], and intended actions [T,4]. Rows refer to
observations before executing the action at that same row.
Cube quaternions are converted to the same yaw features as vision labels
before building overlapping windows of 44-value policy tokens. Tied faces
retain the previous face within each recorded episode.
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
from tqdm import tqdm

from action_policy import ActionPolicy, masked_action_loss
from action_preprocessing import ActionObservationPreprocessor
from action_evaluation import ActionRolloutMetrics, evaluate_action_rollouts
from cube_yaw import CubeYawEncoder
from schemas import CAMERA_IMAGE_SHAPES, HISTORY_TOKEN_SIZE, OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE
from schemas.config import ActionPolicyConfig, ActionTrainingConfig, ProjectConfig, SupervisedTrainingConfig, VisionModelConfig, VisionTrainingConfig
from vision_augmentation import AugmentedVisionDataset, RandAugmentWithoutSolarize
from vision_model import VisionModel, cube_pose_loss


CHECKPOINT_FORMAT_VERSION = 3
VISION_CHECKPOINT_FORMAT_VERSION = 3


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


def convert_vision_pose_labels(poses: Tensor) -> Tensor:
    """Read XYZ/yaw labels, or derive them from saved legacy XYZ/quaternions.

    Return CPU float32 without editing stored tensors. Legacy samples must be
    supplied in chronological order within a shard; face ties retain the prior
    face among those saved samples. New files already contain the yaw labels
    calculated during simulation, including tracking across unsaved frames.
    """
    if not isinstance(poses, Tensor) or poses.ndim < 1 or poses.shape[-1] not in (10, 14):
        raise ValueError("cube_poses must have shape [..., 10] (XYZ/yaw) or [..., 14] (legacy XYZ/quaternion).")
    values = poses.detach().cpu().float()
    if not torch.isfinite(values).all():
        raise ValueError("cube_poses must contain finite values.")
    if values.shape[-1] == 10:
        return values
    raw = values.numpy().reshape(-1, 14)
    converted = np.empty((len(raw), 10), dtype=np.float32)
    encoders = (CubeYawEncoder(), CubeYawEncoder())
    for index, row in enumerate(raw):
        for encoder, (old_offset, new_offset) in zip(encoders, ((0, 0), (7, 5))):
            converted[index, new_offset:new_offset + 3] = row[old_offset:old_offset + 3]
            try:
                converted[index, new_offset + 3:new_offset + 5] = encoder.encode(row[old_offset + 3:old_offset + 7])
            except ValueError as exc:
                raise ValueError(f"cube_poses row {index}, cube {old_offset // 7}: {exc}") from exc
    return torch.from_numpy(converted.reshape(*values.shape[:-1], 10))


def load_vision_dataset(directory: Path, show_progress: bool) -> TensorDataset:
    """Materialize image/pose tensors; normalize pixels later per minibatch.

    Keeping source uint8 images until a batch is used avoids making the whole
    image dataset four times larger. This performs no image resizing/cropping.
    Each shard is one sequence. Every saved row is retained; selection happens
    during data generation, not loading.
    Wrist images are 128x128 and external images are 300x300.
    All shards must use the same image dtype and layout.
    """
    wrist_parts, external_parts, joint_parts, pose_parts = [], [], [], []
    for path in tqdm(_tensor_files(directory), desc=f"Loading vision {directory.name}", unit="shard", disable=not show_progress):
        record = _load_tensors(path)
        pair = []
        for camera_id in ("wrist", "external"):
            key = f"{camera_id}_images"
            height, width, channels = CAMERA_IMAGE_SHAPES[camera_id]
            value = record.get(key)
            if not isinstance(value, Tensor):
                raise ValueError(f"{path}: {key} must be a tensor.")
            if value.dtype == torch.uint8:
                dimensions = (None, height, width, channels)
            elif value.dtype == torch.float32:
                dimensions = (None, channels, height, width)
            else:
                raise ValueError(f"{path}: {key} must be RGB uint8 NHWC or normalized float32 NCHW.")
            try:
                value = _require_tensor(record, key, dimensions, path)
            except ValueError as exc:
                raise ValueError(f"{exc} Regenerate vision data with external 300x300 and wrist 128x128 "
                                 "images; existing images are not resized by the loader.") from exc
            if value.dtype == torch.uint8:
                value = value.permute(0, 3, 1, 2)
            pair.append(value)
        count = pair[0].shape[0]
        if count < 1 or pair[1].shape[0] != count or pair[0].dtype != pair[1].dtype:
            raise ValueError(f"{path}: both views must have the same nonzero count and dtype.")
        if wrist_parts and pair[0].dtype != wrist_parts[0].dtype:
            raise ValueError("All vision shards must have the same image dtype.")
        joints = _require_tensor(record, "joint_positions", (count, 6), path).float()
        poses = _require_tensor(record, "cube_poses", (count, None), path)
        try:
            poses = convert_vision_pose_labels(poses)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        wrist_parts.append(pair[0].contiguous())
        external_parts.append(pair[1].contiguous())
        joint_parts.append(joints)
        pose_parts.append(poses)
    return TensorDataset(*(torch.cat(parts, dim=0) for parts in (wrist_parts, external_parts, joint_parts, pose_parts)))


def build_action_windows(observations: Tensor, prev_accepted_targets: Tensor, actions: Tensor,
                         history_length: int, stride: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build padded causal sequences, supervising each episode action once.

    With context 384 and stride 256, windows are [0:384], [256:640], ... .
    The overlap provides context in later windows, but its repeated labels are
    masked out. A chunk retains its original preceding action, while only the
    actual episode's first token receives an episode-start embedding.
    Observations must already contain the 37 policy features, including
    yaw pairs rather than raw quaternion components. Preprocess the full
    episode first to preserve the previous-face tie-break across windows.
    """
    count = len(observations)
    if count < 1 or observations.shape != (count, POLICY_OBSERVATION_SIZE):
        raise ValueError(f"Processed observations must have nonempty shape [time, {POLICY_OBSERVATION_SIZE}].")
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
    """Load direct policy features or convert legacy raw episodes, then window.

    Direct 37-value observations already contain runtime-normalized vision yaw
    and estimated velocities. Preserve them exactly, including epsilon-sized
    yaw pairs; another normalization would change what runtime supplied.
    """
    parts = []
    preprocessor = ActionObservationPreprocessor()
    for path in tqdm(_tensor_files(directory), desc=f"Calculating {directory.name} tensors", unit="episode", disable=not show_progress):
        record = _load_tensors(path)
        observations = _require_tensor(record, "observations", (None, None), path)
        count = observations.shape[0]
        targets = _require_tensor(record, "prev_accepted_targets", (count, 3), path)
        actions = _require_tensor(record, "actions", (count, 4), path)
        if observations.shape[1] == POLICY_OBSERVATION_SIZE:
            processed = observations.float()
            if not torch.isfinite(processed).all():
                raise ValueError(f"{path}: observations must contain finite policy features.")
        elif observations.shape[1] == OBSERVATION_SIZE:
            try:
                processed = torch.from_numpy(preprocessor.transform_episode(observations.float().numpy()))
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
        else:
            raise ValueError(f"{path}: observations must have shape [time, {POLICY_OBSERVATION_SIZE}] "
                             f"or legacy [time, {OBSERVATION_SIZE}].")
        parts.append(build_action_windows(processed, targets, actions, history_length=history_length, stride=stride))
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
    """Load every saved training sample, then augment it on access."""
    dataset = AugmentedVisionDataset(
        dataset=load_vision_dataset(directory if directory is not None else config.data_dir / "train",
                                    show_progress=show_progress),
        transform=RandAugmentWithoutSolarize(
            num_ops=config.randaugment.num_ops, magnitude=config.randaugment.magnitude,
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
                     position_weight: float = 200.0, orientation_weight: float = 1.0,
                     on_validation_position_error: Callable[[float], None] | None = None) -> float:
    """Return mean loss, optionally reporting validation mean XYZ distance in meters.

    Reuse the loss predictions to average each cube's Euclidean position error,
    first over the two cubes per example and then over all examples. This metric
    is independent of orientation and the training loss weights.
    """
    if optimizer is not None and on_validation_position_error is not None:
        raise ValueError("Position distance is a validation-only metric; optimizer must be None.")
    model.train(optimizer is not None)
    total_loss, total_count = 0.0, 0
    total_position_error = 0.0
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
        if on_validation_position_error is not None:
            predicted_xyz = prediction.detach().reshape(count, 2, 5)[..., :3]
            expected_xyz = poses.reshape(count, 2, 5)[..., :3]
            distances = torch.linalg.vector_norm(predicted_xyz - expected_xyz, dim=-1).mean(dim=-1)
            # Transfer before casting: a combined MPS-to-CPU float64 conversion
            # can silently return zeros on MPS. Accumulate in float64 on CPU.
            total_position_error += float(distances.cpu().double().sum())
        total_loss += float(loss.detach()) * count
        total_count += count
        if on_batch is not None and total_count:
            on_batch(total_loss / total_count)
    if total_count == 0:
        raise ValueError("Vision loader contains no examples.")
    if on_validation_position_error is not None:
        on_validation_position_error(total_position_error / total_count)
    return total_loss / total_count


def save_model_checkpoint(model: ActionPolicy | VisionModel, model_type: str, checkpoint_path: Path,
                          epoch: int, validation_loss: float,
                          validation_rollout_metrics: dict[str, float | int] | None = None) -> None:
    """Save one self-contained file, overwriting a previous checkpoint name."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": VISION_CHECKPOINT_FORMAT_VERSION if model_type == "vision" else CHECKPOINT_FORMAT_VERSION,
        "model_type": model_type,
        "model_config": model.config.model_dump(mode="json"),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": epoch,
        "validation_loss": validation_loss,
    }
    if validation_rollout_metrics is not None:
        checkpoint["validation_rollout_metrics"] = validation_rollout_metrics
    torch.save(checkpoint, checkpoint_path)


def _read_checkpoint(path: Path, model_type: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}.")
    if checkpoint.get("model_type") != model_type:
        raise ValueError(f"Expected {model_type} checkpoint: {path}.")
    expected_version = VISION_CHECKPOINT_FORMAT_VERSION if model_type == "vision" else CHECKPOINT_FORMAT_VERSION
    if checkpoint.get("format_version") != expected_version:
        if model_type == "action" and checkpoint.get("format_version") == 1:
            raise ValueError(
                f"Legacy 48-input action checkpoint is incompatible with the 44-input cube-yaw model: {path}. "
                "Retrain the action model using the existing raw demonstration dataset."
            )
        if model_type == "action" and checkpoint.get("format_version") == 2:
            raise ValueError(
                f"5%-hysteresis cube-yaw action checkpoint is incompatible with direct-yaw inputs and tie-only tracking: {path}. "
                "Retrain the action model using the existing raw demonstration dataset."
            )
        if model_type == "vision" and checkpoint.get("format_version") == 1:
            raise ValueError(
                f"Legacy ResNet-50 vision checkpoint is incompatible with ResNet-34 and the wrist CNN: {path}. "
                "Train a new vision checkpoint using the regenerated dataset."
            )
        if model_type == "vision" and checkpoint.get("format_version") == 2:
            raise ValueError(
                f"Quaternion-output vision checkpoint is incompatible with the 10-output direct-yaw model: {path}. "
                "Train a new vision checkpoint; existing quaternion-label datasets are converted when loaded."
            )
        raise ValueError(f"Unsupported checkpoint format: {path}.")
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
    # The complete fine-tuned ResNet and wrist CNN weights are in this checkpoint. Avoid
    # downloading ImageNet weights merely to overwrite them during loading.
    model = VisionModel(config=config.model_copy(update={"backbone_weights": None}), backbone=None)
    model.config = config
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def train_action(project: ProjectConfig, log_file: Path | None = None) -> list[dict[str, float | int]]:
    config: ActionTrainingConfig = project.training.action
    seed_training(seed=config.seed)
    device = resolve_device(name=config.device)
    training = load_action_dataset(config.data_dir / "train", history_length=project.action_policy.history_length,
                                   stride=config.stride, show_progress=True)
    validation = load_action_dataset(config.data_dir / "validation", history_length=project.action_policy.history_length,
                                     stride=config.stride, show_progress=True)
    loaders = create_loaders(training=training, validation=validation, config=config)
    print(f"Loading vision checkpoint for rollout validation: {project.training.vision.checkpoint_path}")
    vision = load_vision_checkpoint(project.training.vision.checkpoint_path, device)
    vision.requires_grad_(False).eval()
    model = ActionPolicy(config=project.action_policy).to(device)
    evaluator = partial(evaluate_action_rollouts, project=project, action_policy=model,
                        vision_model=vision, device=device)
    return _train(model=model, model_type="action", loaders=loaders, config=config, device=device,
                  log_file=log_file, rollout_evaluator=evaluator)


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
           log_file: Path | None = None,
           rollout_evaluator: Callable[..., ActionRolloutMetrics] | None = None) -> list[dict[str, float | int]]:
    if rollout_evaluator is not None and (model_type != "action" or not isinstance(config, ActionTrainingConfig)):
        raise ValueError("Rollout validation requires action training settings.")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=config.learning_rate, betas=config.optimizer_betas, eps=config.optimizer_eps,
                                 weight_decay=config.weight_decay)
    scheduler = None
    if model_type == "vision" and config.lr_scheduler is not None:
        settings = config.lr_scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings.factor, patience=settings.patience,
            threshold=settings.threshold, threshold_mode="rel", min_lr=settings.min_lr,
        )
    run_epoch = run_action_epoch if model_type == "action" else partial(
        run_vision_epoch, position_weight=config.position_loss_weight,
        orientation_weight=config.orientation_loss_weight,
    )
    metrics = []
    training_loss = validation_loss = float("nan")
    validation_position_error = float("nan")
    validation_success_rate = float("nan")
    validation_rollout_metrics = None
    print(f"Device: {device}; training examples: {len(loaders[0].dataset):,}; validation examples: {len(loaders[1].dataset):,}")
    validation_runs = config.epochs // config.validation_interval + bool(config.epochs % config.validation_interval)
    total_batches = config.epochs * len(loaders[0]) + validation_runs * len(loaders[1])

    def loss_text(loss: float) -> str:
        return "NaN" if np.isnan(loss) else f"{loss:.6f}"

    def progress_fields(train_loss: float) -> dict[str, str]:
        fields = {"train_loss": loss_text(train_loss), "val_loss": loss_text(validation_loss)}
        if model_type == "vision":
            fields["val_position_error_m"] = loss_text(validation_position_error)
            fields["learning_rate"] = f"{optimizer.param_groups[0]['lr']:.8g}"
        if rollout_evaluator is not None:
            fields["val_success_rate_pct"] = loss_text(validation_success_rate)
        return fields

    with ExitStack() as stack:
        log_stream = None
        if log_file is not None:
            log_file = log_file.expanduser()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_stream = stack.enter_context(log_file.open("a", encoding="utf-8"))

        def log_epoch(epoch: int, train_loss: float, val_loss: float,
                      position_error: float = float("nan"),
                      success_rate: float = float("nan"), *,
                      learning_rate: float = config.learning_rate) -> None:
            if log_stream is not None:
                line = f"epoch={epoch} train_loss={loss_text(train_loss)} val_loss={loss_text(val_loss)}"
                if model_type == "vision":
                    line += f" val_position_error_m={loss_text(position_error)}"
                if rollout_evaluator is not None:
                    line += f" val_success_rate_pct={loss_text(success_rate)}"
                if model_type == "vision":
                    line += (f" learning_rate={learning_rate:.8g}"
                             f" next_learning_rate={optimizer.param_groups[0]['lr']:.8g}")
                log_stream.write(line + "\n")
                log_stream.flush()

        log_epoch(0, training_loss, validation_loss)
        progress = stack.enter_context(tqdm(
            total=total_batches, desc=f"Epoch 0/{config.epochs}", unit="batch", dynamic_ncols=True,
            postfix=progress_fields(training_loss),
        ))
        def update_training(loss: float) -> None:
            progress.set_postfix(**progress_fields(loss), refresh=False)
            progress.update(1)

        def update_validation(_loss: float) -> None:
            progress.update(1)

        def update_position_error(distance: float) -> None:
            nonlocal validation_position_error
            validation_position_error = distance

        for epoch in range(1, config.epochs + 1):
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            epoch_validation_loss = float("nan")
            epoch_position_error = float("nan")
            epoch_success_rate = float("nan")
            progress.set_description(f"Epoch {epoch}/{config.epochs} training", refresh=False)
            training_loss = run_epoch(model=model, loader=loaders[0], device=device, optimizer=optimizer,
                                      on_batch=update_training)
            if epoch % config.validation_interval == 0 or epoch == config.epochs:
                progress.set_description(f"Epoch {epoch}/{config.epochs} validation")
                validation_options = ({"on_validation_position_error": update_position_error}
                                      if model_type == "vision" else {})
                validation_loss = epoch_validation_loss = run_epoch(
                    model=model, loader=loaders[1], device=device, optimizer=None, on_batch=update_validation,
                    **validation_options,
                )
                epoch_metrics = {"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss}
                if model_type == "vision":
                    epoch_position_error = validation_position_error
                    epoch_metrics["validation_position_error_m"] = epoch_position_error
                if rollout_evaluator is not None:
                    progress.set_description(f"Epoch {epoch}/{config.epochs} rollout validation")
                    with tqdm(total=config.rollout_validation.episodes, unit="episode", leave=False,
                              desc="Action validation rollouts", dynamic_ncols=True) as rollout_progress:
                        def update_rollout(result: ActionRolloutMetrics) -> None:
                            rollout_progress.set_postfix(
                                successes=result.successes, ik_failures=result.ik_failures,
                                timeouts=result.timeouts, success_rate_pct=f"{result.success_rate_percent:.2f}",
                                refresh=False,
                            )
                            rollout_progress.update(1)

                        result = rollout_evaluator(on_episode=update_rollout)
                    if result.episodes != config.rollout_validation.episodes:
                        raise ValueError("Rollout validation did not evaluate the configured number of episodes.")
                    validation_success_rate = epoch_success_rate = result.success_rate_percent
                    validation_rollout_metrics = {
                        "episodes": result.episodes, "successes": result.successes,
                        "ik_failures": result.ik_failures, "timeouts": result.timeouts,
                        "success_rate_percent": result.success_rate_percent,
                    }
                    epoch_metrics.update(
                        validation_success_rate_percent=epoch_success_rate,
                        validation_rollouts=result.episodes, validation_successes=result.successes,
                        validation_ik_failures=result.ik_failures, validation_timeouts=result.timeouts,
                    )
                # Patience counts fresh validation checks, not training epochs.
                # The updated rate is first used by the following training epoch.
                if scheduler is not None:
                    scheduler.step(epoch_validation_loss)
                if model_type == "vision":
                    epoch_metrics.update(learning_rate=epoch_learning_rate,
                                         next_learning_rate=optimizer.param_groups[0]["lr"])
                metrics.append(epoch_metrics)
            progress.set_description(f"Epoch {epoch}/{config.epochs}", refresh=False)
            progress.set_postfix(**progress_fields(training_loss))
            log_epoch(epoch, training_loss, epoch_validation_loss, epoch_position_error, epoch_success_rate,
                      learning_rate=epoch_learning_rate)
    checkpoint_options = ({"validation_rollout_metrics": validation_rollout_metrics}
                          if validation_rollout_metrics is not None else {})
    save_model_checkpoint(model=model, model_type=model_type, checkpoint_path=config.checkpoint_path,
                          epoch=config.epochs, validation_loss=validation_loss, **checkpoint_options)
    print(f"Saved {model_type} checkpoint: {config.checkpoint_path}")
    return metrics
