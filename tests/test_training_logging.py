"""Training logs expose completed-epoch losses and validation distance in meters."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import training
from schemas.config import load_config


ROOT = Path(__file__).parents[1]


class SilentProgress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


@pytest.fixture
def train_setup(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/default.json").training.vision
    config.epochs = 5
    config.validation_interval = 2
    config.lr_scheduler = None
    config.checkpoint_path = tmp_path / "model.pt"
    loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
    checkpoints = []
    monkeypatch.setattr(training, "tqdm", SilentProgress)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: checkpoints.append(kwargs))
    arguments = dict(model=nn.Linear(1, 1), model_type="vision", loaders=(loader, loader),
                     config=config, device=torch.device("cpu"))
    return arguments, checkpoints


def test_log_flushes_completed_epochs_and_marks_unmeasured_validation_nan(monkeypatch, tmp_path, train_setup):
    arguments, checkpoints = train_setup
    path = tmp_path / "nested" / "logs" / "vision.log"
    expected_lines = [
        "epoch=0 train_loss=NaN val_loss=NaN val_position_error_m=NaN",
        "epoch=1 train_loss=0.100000 val_loss=NaN val_position_error_m=NaN",
        "epoch=2 train_loss=0.200000 val_loss=0.020000 val_position_error_m=0.002000",
        "epoch=3 train_loss=0.300000 val_loss=NaN val_position_error_m=NaN",
        "epoch=4 train_loss=0.400000 val_loss=0.040000 val_position_error_m=0.004000",
        "epoch=5 train_loss=0.500000 val_loss=0.050000 val_position_error_m=0.005000",
    ]
    expected_lines = [line + " learning_rate=0.0001 next_learning_rate=0.0001" for line in expected_lines]
    epoch = 0
    validation_epochs = []

    def run_epoch(*, optimizer, on_batch, **kwargs):
        nonlocal epoch
        if optimizer is not None:
            assert "on_validation_position_error" not in kwargs
            epoch += 1
            loss = epoch / 10
        else:
            validation_epochs.append(epoch)
            loss = epoch / 100
        # A separate reader can already see all completed epochs while this
        # epoch is running. There are no partial or stale validation records.
        assert path.read_text(encoding="utf-8").splitlines() == expected_lines[:epoch]
        on_batch(loss)
        assert path.read_text(encoding="utf-8").splitlines() == expected_lines[:epoch]
        if optimizer is None:
            kwargs["on_validation_position_error"](epoch / 1000)
        return loss

    monkeypatch.setattr(training, "run_vision_epoch", run_epoch)
    metrics = training._train(**arguments, log_file=path)

    assert path.read_text(encoding="utf-8") == "\n".join(expected_lines) + "\n"
    assert validation_epochs == [2, 4, 5]
    assert metrics == [dict(epoch=value, training_loss=value / 10, validation_loss=value / 100,
                            validation_position_error_m=value / 1000,
                            learning_rate=0.0001, next_learning_rate=0.0001)
                       for value in validation_epochs]
    assert len(checkpoints) == 1
    assert checkpoints[0]["epoch"] == 5
    assert checkpoints[0]["validation_loss"] == pytest.approx(0.05)


def test_existing_log_is_preserved_and_new_run_is_appended(monkeypatch, tmp_path, train_setup):
    arguments, _ = train_setup
    arguments["config"].epochs = 1
    path = tmp_path / "vision.log"
    path.write_text("previous training run\n", encoding="utf-8")

    def run_epoch(**kwargs):
        if kwargs["optimizer"] is None:
            kwargs["on_validation_position_error"](0.012345678)
        return 1 / 3

    monkeypatch.setattr(training, "run_vision_epoch", run_epoch)

    training._train(**arguments, log_file=path)

    assert path.read_text(encoding="utf-8") == (
        "previous training run\n"
        "epoch=0 train_loss=NaN val_loss=NaN val_position_error_m=NaN learning_rate=0.0001 next_learning_rate=0.0001\n"
        "epoch=1 train_loss=0.333333 val_loss=0.333333 val_position_error_m=0.012346 learning_rate=0.0001 next_learning_rate=0.0001\n"
    )


def test_omitted_log_file_does_not_open_or_create_a_log(monkeypatch, tmp_path, train_setup):
    arguments, checkpoints = train_setup

    def run_epoch(**kwargs):
        if kwargs["optimizer"] is None:
            kwargs["on_validation_position_error"](0.015)
        return 0.25

    monkeypatch.setattr(training, "run_vision_epoch", run_epoch)

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Logging is optional; no path should be opened.")

    monkeypatch.setattr(Path, "open", unexpected_open)
    metrics = training._train(**arguments)

    assert [row["epoch"] for row in metrics] == [2, 4, 5]
    assert [row["validation_position_error_m"] for row in metrics] == [0.015] * 3
    assert len(checkpoints) == 1
    assert list(tmp_path.iterdir()) == []


def test_log_closes_and_keeps_only_completed_epochs_after_failure(monkeypatch, tmp_path, train_setup):
    arguments, checkpoints = train_setup
    path = tmp_path / "vision.log"
    original_open = Path.open
    handles = []

    def record_open(self, mode="r", *args, **kwargs):
        handle = original_open(self, mode, *args, **kwargs)
        if self == path and "a" in mode:
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", record_open)
    epoch = 0

    def fail_second_epoch(*, on_batch, **kwargs):
        nonlocal epoch
        epoch += 1
        on_batch(0.25)
        if epoch == 2:
            raise RuntimeError("training batch failed")
        return 0.25

    monkeypatch.setattr(training, "run_vision_epoch", fail_second_epoch)
    with pytest.raises(RuntimeError, match="training batch failed"):
        training._train(**arguments, log_file=path)

    assert len(handles) == 1 and handles[0].closed
    assert path.read_text(encoding="utf-8") == (
        "epoch=0 train_loss=NaN val_loss=NaN val_position_error_m=NaN learning_rate=0.0001 next_learning_rate=0.0001\n"
        "epoch=1 train_loss=0.250000 val_loss=NaN val_position_error_m=NaN learning_rate=0.0001 next_learning_rate=0.0001\n"
    )
    assert checkpoints == []


def test_action_log_has_no_vision_position_metric(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/default.json").training.action
    config.epochs = 1
    config.checkpoint_path = tmp_path / "action.pt"
    loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
    path = tmp_path / "action.log"

    def run_epoch(*, on_batch, **kwargs):
        assert "on_validation_position_error" not in kwargs
        on_batch(0.25)
        return 0.25

    monkeypatch.setattr(training, "tqdm", SilentProgress)
    monkeypatch.setattr(training, "run_action_epoch", run_epoch)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: None)
    metrics = training._train(model=nn.Linear(1, 1), model_type="action", loaders=(loader, loader),
                              config=config, device=torch.device("cpu"), log_file=path)

    assert path.read_text(encoding="utf-8") == (
        "epoch=0 train_loss=NaN val_loss=NaN\n"
        "epoch=1 train_loss=0.250000 val_loss=0.250000\n"
    )
    assert metrics == [{"epoch": 1, "training_loss": 0.25, "validation_loss": 0.25}]


@pytest.mark.parametrize("log_option", [None, "--log", "--log-file"])
def test_vision_cli_forwards_default_or_explicit_log_path(monkeypatch, tmp_path, log_option):
    script = ROOT / "scripts/train_vision.py"
    spec = importlib.util.spec_from_file_location("train_vision_logging_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = object()
    config_paths, calls = [], []

    def load_configuration(*, path):
        config_paths.append(path)
        return config

    monkeypatch.setattr(module, "load_config", load_configuration)
    monkeypatch.setattr(module, "train_vision", lambda **kwargs: calls.append(kwargs))
    arguments = [str(script)]
    log_path = tmp_path / "training logs" / "vision.log" if log_option else Path("logs/default.log")
    if log_option:
        arguments.extend([log_option, str(log_path)])
    monkeypatch.setattr(sys, "argv", arguments)

    module.main()

    assert config_paths == [Path("configs/default.json")]
    assert calls == [{"project": config, "log_file": log_path}]


@pytest.mark.parametrize("with_log", [False, True])
def test_vision_training_entrypoint_forwards_log_path(monkeypatch, tmp_path, with_log):
    project = load_config(ROOT / "configs/default.json")
    device = torch.device("cpu")
    model = nn.Linear(1, 1)
    train_loader, validation_data, validation_loader = object(), object(), object()
    calls = []
    monkeypatch.setattr(training, "seed_training", lambda **kwargs: None)
    monkeypatch.setattr(training, "resolve_device", lambda **kwargs: device)
    monkeypatch.setattr(training, "create_vision_training_loader", lambda **kwargs: train_loader)
    monkeypatch.setattr(training, "load_vision_dataset", lambda *args, **kwargs: validation_data)
    monkeypatch.setattr(training, "create_loader", lambda **kwargs: validation_loader)
    monkeypatch.setattr(training, "VisionModel", lambda **kwargs: model)

    def train(**kwargs):
        calls.append(kwargs)
        return [{"epoch": 1, "training_loss": 0.2, "validation_loss": 0.1}]

    monkeypatch.setattr(training, "_train", train)
    path = tmp_path / "vision.log" if with_log else None
    result = training.train_vision(project=project, **({"log_file": path} if with_log else {}))

    assert calls == [{"model": model, "model_type": "vision", "loaders": (train_loader, validation_loader),
                      "config": project.training.vision, "device": device, "log_file": path}]
    assert result == [{"epoch": 1, "training_loss": 0.2, "validation_loss": 0.1}]
