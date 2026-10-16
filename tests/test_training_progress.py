"""Batch progress and displayed losses follow the actual training schedule."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import training
from schemas.config import load_config


class RecordingProgress:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.unit = kwargs["unit"]
        self.description = kwargs.get("desc", "")
        self.descriptions = [self.description]
        self.postfix = dict(kwargs.get("postfix", {}))
        self.postfixes = [dict(self.postfix)] if self.postfix else []
        self.n = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def set_description(self, desc, refresh=True):
        self.description = desc
        self.descriptions.append(desc)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self.postfix = dict(ordered_dict or {}, **kwargs)
        self.postfixes.append(dict(self.postfix))

    def update(self, n=1):
        self.n += n


def progress_setup(monkeypatch, tmp_path, model_type, epochs=3):
    config = getattr(load_config(Path(__file__).parents[1] / "configs/default.json").training, model_type)
    config.epochs = epochs
    config.validation_interval = 2
    config.checkpoint_path = tmp_path / "trained.pt"
    loaders = (
        DataLoader(TensorDataset(torch.zeros(3, 1)), batch_size=2),
        DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=2),
    )
    bars, checkpoints = [], []

    def create_bar(**kwargs):
        bar = RecordingProgress(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(training, "tqdm", create_bar)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: checkpoints.append(kwargs))
    return config, loaders, bars, checkpoints


@pytest.mark.parametrize("model_type", ["vision", "action"])
@pytest.mark.parametrize("epochs", [3, 4])
def test_training_progress_reports_batches_and_latest_completed_validation(monkeypatch, tmp_path, model_type, epochs):
    config, loaders, bars, checkpoints = progress_setup(monkeypatch, tmp_path, model_type, epochs)
    if model_type == "vision":
        config.position_loss_weight = 7.5
        config.orientation_loss_weight = 0.25
    observations = []
    epoch = 0

    def run_epoch(*, model, loader, device, optimizer, on_batch, **loss_weights):
        nonlocal epoch
        expected_weights = ({"position_weight": 7.5, "orientation_weight": 0.25}
                            if model_type == "vision" else {})
        assert loss_weights == expected_weights
        is_training = optimizer is not None
        if is_training:
            epoch += 1
        for batch in range(1, len(loader) + 1):
            loss = (10 * epoch + batch) / 100 if is_training else epoch / 50
            on_batch(loss)
            bar = bars[0]
            observations.append((epoch, is_training, loss, bar.n, bar.description, dict(bar.postfix)))
        return loss

    monkeypatch.setattr(training, f"run_{model_type}_epoch", run_epoch)
    metrics = training._train(model=nn.Linear(1, 1), model_type=model_type, loaders=loaders,
                              config=config, device=torch.device("cpu"))

    assert len(bars) == 1
    bar = bars[0]
    validation_epochs = [value for value in range(1, epochs + 1) if value % 2 == 0 or value == epochs]
    assert bar.total == epochs * len(loaders[0]) + len(validation_epochs) * len(loaders[1])
    assert bar.n == bar.total and bar.closed
    assert bar.unit == "batch"
    assert bar.descriptions[0] == f"Epoch 0/{epochs}"
    assert bar.postfixes[0] == {"train_loss": "NaN", "val_loss": "NaN"}

    last_validation = None
    training_loss = None
    for batch_number, (current_epoch, is_training, loss, completed, description, postfix) in enumerate(observations, 1):
        assert completed == batch_number
        assert f"Epoch {current_epoch}/{epochs}" in description
        assert ("training" if is_training else "validation") in description.lower()
        if is_training:
            training_loss = loss
        assert float(postfix["train_loss"]) == pytest.approx(training_loss)
        if last_validation is None:
            assert postfix["val_loss"] == "NaN"
        else:
            assert float(postfix["val_loss"]) == pytest.approx(last_validation)
        if not is_training:
            last_validation = loss

    assert metrics == [{"epoch": value, "training_loss": (10 * value + 2) / 100,
                        "validation_loss": value / 50} for value in validation_epochs]
    assert float(bar.postfix["train_loss"]) == pytest.approx(metrics[-1]["training_loss"])
    assert float(bar.postfix["val_loss"]) == pytest.approx(metrics[-1]["validation_loss"])
    assert len(checkpoints) == 1
    assert checkpoints[0]["epoch"] == epochs
    assert checkpoints[0]["validation_loss"] == metrics[-1]["validation_loss"]


def test_training_progress_closes_when_a_batch_fails(monkeypatch, tmp_path):
    config, loaders, bars, checkpoints = progress_setup(monkeypatch, tmp_path, "vision")

    def fail_epoch(*, on_batch, **kwargs):
        on_batch(0.25)
        raise RuntimeError("batch failed")

    monkeypatch.setattr(training, "run_vision_epoch", fail_epoch)
    with pytest.raises(RuntimeError, match="batch failed"):
        training._train(model=nn.Linear(1, 1), model_type="vision", loaders=loaders,
                        config=config, device=torch.device("cpu"))
    assert bars[0].closed and bars[0].n == 1
    assert checkpoints == []


@pytest.mark.parametrize("loss_weights, expected_losses", [
    ({}, [500.5, 1400.5]),
    ({"position_weight": 3.0, "orientation_weight": 4.0}, [9.5, 23.0]),
])
def test_vision_batch_callback_reports_example_weighted_running_loss(loss_weights, expected_losses):
    class PoseModel(nn.Module):
        def forward(self, wrist, external, joints):
            prediction = torch.zeros(len(joints), 14)
            prediction[:, :3] = joints[:, :1]
            prediction[:, 7:10] = joints[:, :1]
            prediction[:, 4] = prediction[:, 11] = 1.0
            return prediction

    image = torch.zeros(3, 3, 1, 1)
    joints = torch.zeros(3, 6)
    joints[:, 0] = torch.tensor([1.0, 2.0, 4.0])
    poses = torch.zeros(3, 14)
    poses[:, 3] = poses[:, 10] = 1.0
    loader = DataLoader(TensorDataset(image, image.clone(), joints, poses), batch_size=2)
    losses = []
    result = training.run_vision_epoch(PoseModel(), loader, torch.device("cpu"), optimizer=None,
                                       on_batch=losses.append, **loss_weights)
    # The unweighted position averages are 2.5 and 7.0; orientation MSE is 0.5.
    assert losses == pytest.approx(expected_losses)
    assert result == pytest.approx(losses[-1])


def test_action_batch_callback_weights_supervised_timestamps():
    class ActionModel(nn.Module):
        def forward(self, tokens, valid, episode_start):
            return tokens[..., :4]

    tokens = torch.zeros(3, 2, 48)
    tokens[0, :, :4] = 1.0
    tokens[1, :, :4] = 2.0
    tokens[2, :, :4] = 4.0
    valid = torch.ones(3, 2, dtype=torch.bool)
    mask = torch.tensor([[True, True], [True, False], [True, False]])
    loader = DataLoader(TensorDataset(tokens, valid, torch.zeros_like(valid), torch.zeros(3, 2, 4), mask),
                        batch_size=2)
    losses = []
    result = training.run_action_epoch(ActionModel(), loader, torch.device("cpu"), optimizer=None,
                                       on_batch=losses.append)
    assert losses == pytest.approx([2.0, 5.5])
    assert result == pytest.approx(losses[-1])
