"""Validation diagnostics split exactly the position/orientation training loss."""

import importlib.util
from math import cos, pi, sin
from pathlib import Path
import re
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from schemas.config import load_config
from vision_model import cube_pose_loss, cube_pose_loss_components


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "vision_loss_breakdown.py"


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("vision_loss_breakdown", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def known_poses():
    targets = torch.zeros(3, 2, 5)
    # Pair magnitude is irrelevant; a quarter-turn has the same encoded yaw.
    targets[..., 3] = 3
    prediction = targets.clone()
    prediction[:, 0, :3] = torch.tensor([[0.3, 0, 0], [0, -0.6, 0], [0, 0, 0.9]])
    prediction[:, 1, :3] = torch.tensor([[0.1, -0.1, 0.1], [0.2, 0.2, -0.2], [-0.3, 0.3, 0.3]])
    prediction[0, 0, 3:] = torch.tensor([5., 0])
    prediction[1, 0, 3:] = torch.tensor([2., 0])
    prediction[2, 0, 3:] = 2 * torch.tensor([cos(2 * pi / 3), sin(2 * pi / 3)])  # 30 degrees.
    prediction[0, 1, 3:] = torch.tensor([-4., 0])  # 45 degrees.
    prediction[1, 1, 3:] = torch.tensor([2., 0])
    prediction[2, 1, 3:] = torch.tensor([7., 0])
    return prediction.reshape(3, 10), targets.reshape(3, 10)


def test_components_have_expected_xyz_and_circular_yaw_values():
    prediction, target = known_poses()
    position, orientation = cube_pose_loss_components(prediction, target)

    torch.testing.assert_close(position, torch.tensor([[0.03, 0.01], [0.12, 0.04], [0.27, 0.09]]))
    thirty_degrees = 1 - cos(2 * pi / 3)
    forty_five_degrees = 2.0
    torch.testing.assert_close(orientation, torch.tensor([[0., forty_five_degrees], [0., 0.], [thirty_degrees, 0.]]))
    torch.testing.assert_close((position + orientation).mean(),
                               cube_pose_loss(prediction, target, position_weight=1.0, orientation_weight=1.0))


@pytest.mark.parametrize("shape", [(10,), (3, 10), (2, 3, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_components_support_batch_dimensions_and_have_finite_gradients(shape, dtype):
    generator = torch.Generator().manual_seed(19)
    prediction = torch.randn(shape, dtype=dtype, generator=generator).requires_grad_()
    target = torch.randn(shape, dtype=dtype, generator=generator).requires_grad_()
    position, orientation = cube_pose_loss_components(prediction, target)
    assert position.shape == orientation.shape == (*shape[:-1], 2)
    assert position.dtype == orientation.dtype == dtype
    predicted_poses = prediction.reshape(*shape[:-1], 2, 5)
    target_poses = target.reshape(*shape[:-1], 2, 5)
    expected_position = (predicted_poses[..., :3] - target_poses[..., :3]).square().mean(-1)
    torch.testing.assert_close(position, expected_position, rtol=0, atol=0)
    assert (orientation > 0).all()

    expected = (position + orientation).mean()
    actual = cube_pose_loss(prediction, target, position_weight=1.0, orientation_weight=1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for gradient in torch.autograd.grad(actual, (prediction, target)):
        assert torch.isfinite(gradient).all()
        assert gradient[..., [3, 4, 8, 9]].abs().sum() > 0


def test_circular_yaw_loss_gradients_match_finite_differences():
    generator = torch.Generator().manual_seed(47)
    prediction = torch.randn(2, 10, dtype=torch.float64, generator=generator).requires_grad_()
    target = torch.randn(2, 10, dtype=torch.float64, generator=generator).requires_grad_()
    assert torch.autograd.gradcheck(cube_pose_loss_components, (prediction, target))


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


@pytest.mark.parametrize("weights", [None, (7.5, 3.25), (0.0, 2.0), (6.0, 0.0)])
def test_evaluation_weights_uneven_batches_and_matches_training_loss(diagnostic, weights):
    predictions, targets = known_poses()
    model = RecordedPredictions(predictions).train()

    arguments = {} if weights is None else {"position_weight": weights[0], "orientation_weight": weights[1]}
    position_weight, orientation_weight = (200.0, 1.0) if weights is None else weights
    count, components = diagnostic.evaluate_loss_breakdown(model, make_loader(targets), torch.device("cpu"),
                                                            **arguments)

    thirty_degrees = 1 - cos(2 * pi / 3)
    forty_five_degrees = 2.0
    assert count == 3
    assert components.shape == (2, 2)
    expected = torch.tensor([
        [0.14, thirty_degrees / 3],
        [0.14 / 3, forty_five_degrees / 3],
    ]) * torch.tensor([position_weight, orientation_weight])
    torch.testing.assert_close(components.float(), expected)
    torch.testing.assert_close(components.sum(-1).mean().float(), cube_pose_loss(predictions, targets, **arguments))
    assert len(model.calls) == 2
    assert all(not call["training"] and not call["grad_enabled"] and call["inference_mode"]
               for call in model.calls)
    assert model.offset.grad is None
    assert not components.requires_grad


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_evaluation_preserves_training_image_normalization(diagnostic, dtype):
    wrist = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 3, 1, 1).expand(3, 3, 2, 2).clone()
    external = 255 - wrist
    if dtype == torch.float32:
        wrist = wrist.float() / 255
        external = external.float() / 255
    expected_wrist = wrist.float() / 255 if dtype == torch.uint8 else wrist
    expected_external = external.float() / 255 if dtype == torch.uint8 else external
    predictions, targets = known_poses()
    model = RecordedPredictions(predictions)

    diagnostic.evaluate_loss_breakdown(model, make_loader(targets, wrist, external), torch.device("cpu"))

    torch.testing.assert_close(torch.cat([call["wrist"] for call in model.calls]), expected_wrist)
    torch.testing.assert_close(torch.cat([call["external"] for call in model.calls]), expected_external)


def test_empty_validation_loader_is_an_error(diagnostic):
    targets = torch.zeros(0, 10)
    with pytest.raises(ValueError, match=r"(?i)(empty|no .*examples|no .*samples|zero)"):
        diagnostic.evaluate_loss_breakdown(RecordedPredictions(targets), make_loader(targets), torch.device("cpu"))


@pytest.mark.parametrize("explicit_checkpoint,zero_loss", [(False, False), (True, False), (False, True)])
def test_cli_checkpoint_selection_validation_split_and_loss_shares(
    diagnostic, monkeypatch, tmp_path, capsys, explicit_checkpoint, zero_loss,
):
    project = load_config(REPO_ROOT / "configs" / "default.json")
    settings = project.training.vision
    settings.data_dir = tmp_path / "data"
    settings.checkpoint_path = tmp_path / "trained" / "default.pt"
    settings.device = "auto"
    settings.position_loss_weight = 7.0
    settings.orientation_loss_weight = 3.0
    selected = tmp_path / "other.pt" if explicit_checkpoint else settings.checkpoint_path
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.touch()
    arguments = [str(SCRIPT)]
    if explicit_checkpoint:
        arguments.extend(["--checkpoint", str(selected)])
    monkeypatch.setattr(sys, "argv", arguments)

    dataset = TensorDataset(torch.zeros(3, 10))
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

    def evaluate(model, loader, device, position_weight, orientation_weight):
        calls["evaluate"] = (model, loader, device, position_weight, orientation_weight)
        components = torch.zeros(2, 2) if zero_loss else torch.tensor([[0.01, 0.03], [0.09, 0.07]])
        return 3, components * torch.tensor([position_weight, orientation_weight])

    monkeypatch.setattr(diagnostic, "load_config", config)
    monkeypatch.setattr(diagnostic, "load_vision_checkpoint", checkpoint)
    monkeypatch.setattr(diagnostic, "load_vision_dataset", validation_dataset)
    monkeypatch.setattr(diagnostic, "create_loader", validation_loader)
    monkeypatch.setattr(diagnostic, "resolve_device", resolve)
    monkeypatch.setattr(diagnostic, "evaluate_loss_breakdown", evaluate)

    diagnostic.main()

    assert calls["config"] == REPO_ROOT / "configs" / "default.json"
    assert calls["checkpoint"] == (selected, device)
    assert calls["dataset"] == (settings.data_dir / "validation", True)
    assert calls["loader"] == (dataset, settings, False)
    assert calls["device"] == settings.device
    assert calls["evaluate"] == (model, loader, device, 7.0, 3.0)
    output = capsys.readouterr().out.lower()
    assert "weighted" in output
    # The combined percentage comes from combined losses, not mean cube percentages.
    expected_rows = (("orange", [0.07, 0.09, 0.16, 43.75, 56.25]),
                     ("blue", [0.63, 0.21, 0.84, 75., 25.]),
                     ("combined", [0.35, 0.15, 0.5, 70., 30.]))
    for label, expected in expected_rows:
        row = next(line for line in output.splitlines() if line.lstrip().startswith(label))
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", row)]
        if zero_loss:
            assert values == [0., 0., 0.]
            assert row.count("n/a") == 2
        else:
            assert values == pytest.approx(expected, abs=0.01)
