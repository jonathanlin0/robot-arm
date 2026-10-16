"""Position diagnostics use the validation split and report absolute axis errors."""

import importlib.util
from pathlib import Path
import re
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from schemas.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "vision_position_error.py"


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("vision_position_error", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordedPredictions(nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, wrist, external, joints):
        self.calls.append({
            "wrist": wrist.clone(), "external": external.clone(),
            "training": self.training, "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
        })
        return self.predictions[joints[:, 0].long()] + self.offset


def make_loader(targets, wrist=None, external=None):
    count = len(targets)
    if wrist is None:
        wrist = torch.zeros(count, 3, 2, 2, dtype=torch.uint8)
    if external is None:
        external = wrist.clone()
    joints = torch.zeros(count, 6)
    joints[:, 0] = torch.arange(count)
    return DataLoader(TensorDataset(wrist, external, joints, targets), batch_size=2)


def test_axis_mae_averages_examples_and_ignores_orientation(diagnostic):
    targets = torch.arange(42, dtype=torch.float32).reshape(3, 14) / 100
    predictions = targets.clone()
    predictions[:, :3] += torch.tensor([
        [0.01, -0.02, 0.03], [-0.01, 0.04, -0.09], [0.10, -0.15, 0.18],
    ])
    predictions[:, 7:10] += torch.tensor([
        [0.02, -0.01, 0.04], [-0.04, 0.02, -0.08], [0.12, -0.09, 0.24],
    ])
    predictions[:, 3:7] = 1e6
    predictions[:, 10:14] = -1e6
    model = RecordedPredictions(predictions).train()

    count, error = diagnostic.evaluate_position_error(model, make_loader(targets), torch.device("cpu"))

    assert count == 3
    assert error.shape == (2, 3)
    torch.testing.assert_close(error.float(), torch.tensor([[0.04, 0.07, 0.10], [0.06, 0.04, 0.12]]))
    assert len(model.calls) == 2
    assert all(not call["training"] and not call["grad_enabled"] and call["inference_mode"]
               for call in model.calls)
    assert model.offset.grad is None
    assert not error.requires_grad


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_images_match_training_normalization(diagnostic, dtype):
    wrist = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 3, 1, 1).expand(3, 3, 2, 2).clone()
    external = 255 - wrist
    if dtype == torch.float32:
        wrist = wrist.float() / 255
        external = external.float() / 255
    expected_wrist = wrist.float() / 255 if dtype == torch.uint8 else wrist
    expected_external = external.float() / 255 if dtype == torch.uint8 else external
    targets = torch.zeros(3, 14)
    model = RecordedPredictions(targets)

    diagnostic.evaluate_position_error(model, make_loader(targets, wrist, external), torch.device("cpu"))

    torch.testing.assert_close(torch.cat([call["wrist"] for call in model.calls]), expected_wrist)
    torch.testing.assert_close(torch.cat([call["external"] for call in model.calls]), expected_external)


def test_empty_validation_loader_is_an_error(diagnostic):
    targets = torch.zeros(0, 14)
    model = RecordedPredictions(targets)

    with pytest.raises(ValueError, match=r"(?i)(empty|no .*examples|no .*samples|zero)"):
        diagnostic.evaluate_position_error(model, make_loader(targets), torch.device("cpu"))


@pytest.mark.parametrize("explicit_checkpoint", [False, True])
def test_cli_uses_selected_checkpoint_and_unaugmented_validation_data(
    diagnostic, monkeypatch, tmp_path, capsys, explicit_checkpoint,
):
    project = load_config(REPO_ROOT / "configs" / "default.json")
    settings = project.training.vision
    settings.data_dir = tmp_path / "data"
    settings.checkpoint_path = tmp_path / "trained" / "default.pt"
    settings.device = "auto"
    selected = tmp_path / "other.pt" if explicit_checkpoint else settings.checkpoint_path
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.touch()
    arguments = [str(SCRIPT)]
    if explicit_checkpoint:
        arguments.extend(["--checkpoint", str(selected)])
    monkeypatch.setattr(sys, "argv", arguments)

    dataset = TensorDataset(torch.zeros(3, 14))
    loader, model = object(), object()
    device = torch.device("cpu")
    calls = {}

    def config(path):
        calls["config"] = path
        return project

    def checkpoint(path, device):
        calls["checkpoint"] = (path, device)
        return model

    def validation_dataset(directory, show_progress):
        calls["dataset"] = (directory, show_progress)
        return dataset

    def validation_loader(dataset, config, training):
        calls["loader"] = (dataset, config, training)
        return loader

    def resolve(name):
        calls["device"] = name
        return device

    def evaluate(model, loader, device):
        calls["evaluate"] = (model, loader, device)
        return 3, torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])

    monkeypatch.setattr(diagnostic, "load_config", config)
    monkeypatch.setattr(diagnostic, "load_vision_checkpoint", checkpoint)
    monkeypatch.setattr(diagnostic, "load_vision_dataset", validation_dataset)
    monkeypatch.setattr(diagnostic, "create_loader", validation_loader)
    monkeypatch.setattr(diagnostic, "resolve_device", resolve)
    monkeypatch.setattr(diagnostic, "evaluate_position_error", evaluate)

    diagnostic.main()

    assert calls["config"] == REPO_ROOT / "configs" / "default.json"
    assert calls["checkpoint"] == (selected, device)
    assert calls["dataset"] == (settings.data_dir / "validation", True)
    assert calls["loader"] == (dataset, settings, False)
    assert calls["device"] == settings.device
    assert calls["evaluate"] == (model, loader, device)
    output = capsys.readouterr().out.lower()
    assert "cm" in output
    for label, expected in (("orange", [1, 2, 3]), ("blue", [4, 5, 6]), ("combined", [2.5, 3.5, 4.5])):
        row = next(line for line in output.splitlines() if label in line)
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", row)]
        assert values == pytest.approx(expected)
