"""Validation distance is a mean in meters, separate from the weighted loss."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import training


class IndexedPoseModel(nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = nn.Parameter(predictions.clone())
        self.forward_states = []

    def forward(self, wrist, external, joints):
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        return self.predictions[joints[:, 0].long()]


def distance_fixture():
    # Nonzero world positions make confusing the prediction with its error fail.
    targets = torch.tensor([
        [0.10, -0.10, 0.02, 1, 0, -0.10, 0.10, 0.04, 1, 0],
        [0.12, -0.08, 0.03, 1, 0, -0.08, 0.12, 0.05, 1, 0],
        [0.14, -0.06, 0.04, 1, 0, -0.06, 0.14, 0.06, 1, 0],
    ], dtype=torch.float32)
    predictions = targets.clone()
    predictions[:, :3] += torch.tensor([
        [-0.003, 0.004, -0.012],  # 13 mm: a 3-D 3-4-12 triangle.
        [0.001, 0, 0],
        [0, -0.030, 0],
    ])
    predictions[:, 5:8] += torch.tensor([
        [0, -0.003, 0.004],  # 5 mm.
        [0, 0, -0.002],
        [0.040, 0, 0],
    ])
    return targets, predictions


def loader_for(targets, batch_size=2):
    count = len(targets)
    image = torch.zeros(count, 3, 1, 1)
    joints = torch.zeros(count, 6)
    joints[:, 0] = torch.arange(count)
    return DataLoader(TensorDataset(image, image.clone(), joints, targets), batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("device_name", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable")),
])
def test_validation_distance_averages_euclidean_errors_over_examples_and_both_cubes(batch_size, device_name):
    targets, predictions = distance_fixture()
    device = torch.device(device_name)
    model = IndexedPoseModel(predictions).to(device)
    distances, running_losses = [], []

    loss = training.run_vision_epoch(
        model, loader_for(targets, batch_size), device, optimizer=None,
        on_batch=running_losses.append, on_validation_position_error=distances.append,
    )

    # Average each example's two cube distances, then all three examples.
    expected = ((0.013 + 0.005) / 2 + (0.001 + 0.002) / 2 + (0.030 + 0.040) / 2) / 3
    assert distances == pytest.approx([expected], abs=1e-8)
    assert isinstance(distances[0], float)
    assert isinstance(loss, float)
    assert loss == pytest.approx(running_losses[-1])
    # The uneven 2+1 batches must not receive equal weight, and measuring the
    # distance must reuse the same validation forward passes as measuring loss.
    expected_batches = (3 + batch_size - 1) // batch_size
    assert model.forward_states == [(False, False)] * expected_batches
    assert model.predictions.grad is None


@pytest.mark.parametrize("position_weight, orientation_weight", [(200.0, 1.0), (0.0, 500.0), (3.0, 0.0)])
def test_validation_distance_ignores_yaw_and_loss_weights(position_weight, orientation_weight):
    targets, predictions = distance_fixture()
    # Change only rotations; the reported position distance remains in meters.
    predictions[:, 3:5] = torch.tensor([0.98, 0.10])
    predictions[:, 8:10] = torch.tensor([-0.70, 0.30])
    distances = []

    training.run_vision_epoch(
        IndexedPoseModel(predictions), loader_for(targets), torch.device("cpu"), optimizer=None,
        position_weight=position_weight, orientation_weight=orientation_weight,
        on_validation_position_error=distances.append,
    )

    assert distances == pytest.approx([0.091 / 6], abs=1e-8)


def test_training_does_not_compute_validation_distance(monkeypatch):
    targets, predictions = distance_fixture()
    model = IndexedPoseModel(predictions)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.predictions.detach().clone()

    # Isolate the distance calculation from any norms needed by orientation loss.
    monkeypatch.setattr(training, "cube_pose_loss", lambda prediction, target, **kwargs:
                        (prediction - target).square().mean())

    def unexpected_distance(*args, **kwargs):
        raise AssertionError("Training must not calculate the validation distance")

    monkeypatch.setattr(torch.linalg, "vector_norm", unexpected_distance)
    loss = training.run_vision_epoch(model, loader_for(targets), torch.device("cpu"), optimizer=optimizer)

    assert loss > 0
    assert model.forward_states == [(True, True), (True, True)]
    assert not torch.equal(model.predictions.detach(), before)


def test_training_rejects_validation_distance_callback_before_updating_model():
    targets, predictions = distance_fixture()
    model = IndexedPoseModel(predictions).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    distances = []

    with pytest.raises(ValueError):
        training.run_vision_epoch(
            model, loader_for(targets), torch.device("cpu"), optimizer=optimizer,
            on_validation_position_error=distances.append,
        )

    assert not model.training
    assert model.forward_states == []
    assert model.predictions.grad is None
    assert torch.equal(model.predictions.detach(), predictions)
    assert distances == []


def test_empty_validation_loader_reports_no_distance():
    targets = torch.zeros(0, 10)
    model = IndexedPoseModel(targets)
    distances = []

    with pytest.raises(ValueError, match="Vision loader contains no examples"):
        training.run_vision_epoch(
            model, loader_for(targets), torch.device("cpu"), optimizer=None,
            on_validation_position_error=distances.append,
        )

    assert distances == []
    assert model.forward_states == []
