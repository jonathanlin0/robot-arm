"""Validation maps preserve each cube center and its full 3-D position error."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "vision_position_error_maps.py"


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("vision_position_error_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndexedPredictions(nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, wrist, external, joints):
        self.calls.append({
            "wrist": wrist.clone(),
            "external": external.clone(),
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
        })
        return self.predictions[joints[:, 0].long()] + self.offset


def make_loader(targets, wrist=None, external=None, batch_size=2):
    count = len(targets)
    if wrist is None:
        wrist = torch.zeros(count, 3, 2, 2, dtype=torch.uint8)
    if external is None:
        external = wrist.clone()
    joints = torch.zeros(count, 6)
    joints[:, 0] = torch.arange(count)
    return DataLoader(TensorDataset(wrist, external, joints, targets), batch_size=batch_size)


def position_fixture():
    targets = torch.tensor([
        [0.10, -0.10, 0.02, 1, 0, -0.10, 0.10, 0.04, 1, 0],
        [0.12, -0.08, 0.03, 1, 0, -0.08, 0.12, 0.05, 1, 0],
        [0.14, -0.06, 0.04, 1, 0, -0.06, 0.14, 0.06, 1, 0],
    ], dtype=torch.float32)
    predictions = targets.clone()
    predictions[:, :3] += torch.tensor([
        [-0.003, 0.004, -0.012],  # 13 mm, including the Z error.
        [0.001, 0, 0],
        [0, -0.030, 0],
    ])
    predictions[:, 5:8] += torch.tensor([
        [0, -0.003, 0.004],
        [0, 0, -0.002],
        [0.040, 0, 0],
    ])
    # Orientation contributes nothing to the positional diagnostic.
    predictions[:, 3:5] = 1e6
    predictions[:, 8:10] = -1e6
    return targets, predictions


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_collect_preserves_every_cube_position_and_euclidean_error(diagnostic, batch_size):
    targets, predictions = position_fixture()
    model = IndexedPredictions(predictions).train()

    true_xyz, predicted_xyz, distances = diagnostic.collect_position_predictions(
        model, make_loader(targets, batch_size=batch_size), torch.device("cpu"),
    )

    assert true_xyz.shape == predicted_xyz.shape == (3, 2, 3)
    assert distances.shape == (3, 2)
    assert all(isinstance(values, np.ndarray) for values in (true_xyz, predicted_xyz, distances))
    np.testing.assert_array_equal(true_xyz, targets.reshape(3, 2, 5)[..., :3].numpy())
    np.testing.assert_array_equal(predicted_xyz, predictions.reshape(3, 2, 5)[..., :3].numpy())
    np.testing.assert_allclose(distances, [[0.013, 0.005], [0.001, 0.002], [0.030, 0.040]], atol=1e-8)
    np.testing.assert_array_equal(distances, np.linalg.norm(predicted_xyz - true_xyz, axis=-1))
    assert true_xyz.dtype == predicted_xyz.dtype == distances.dtype == np.float64
    assert len(model.calls) == (3 + batch_size - 1) // batch_size
    assert all(not call["training"] and not call["grad_enabled"] and call["inference_mode"]
               for call in model.calls)
    assert model.offset.grad is None


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_collect_scales_byte_images_once_and_leaves_normalized_floats_unchanged(diagnostic, dtype):
    wrist = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 3, 1, 1).expand(3, 3, 2, 2).clone()
    external = 255 - wrist
    if dtype == torch.float32:
        wrist = wrist.float() / 255
        external = external.float() / 255
    expected_wrist = wrist.float() / 255 if dtype == torch.uint8 else wrist
    expected_external = external.float() / 255 if dtype == torch.uint8 else external
    targets = torch.zeros(3, 10)
    model = IndexedPredictions(targets)

    diagnostic.collect_position_predictions(
        model, make_loader(targets, wrist, external), torch.device("cpu"),
    )

    torch.testing.assert_close(torch.cat([call["wrist"] for call in model.calls]), expected_wrist)
    torch.testing.assert_close(torch.cat([call["external"] for call in model.calls]), expected_external)


def test_collect_rejects_empty_validation_data(diagnostic):
    targets = torch.zeros(0, 10)
    model = IndexedPredictions(targets)

    with pytest.raises(ValueError, match=r"(?i)(empty|no .*examples|no .*samples|zero)"):
        diagnostic.collect_position_predictions(model, make_loader(targets), torch.device("cpu"))

    assert model.calls == []


@pytest.mark.parametrize("width", [9, 14])
def test_collect_rejects_incompatible_prediction_shapes(diagnostic, width):
    targets = torch.zeros(3, 10)
    model = IndexedPredictions(torch.zeros(3, width))

    with pytest.raises(ValueError, match=r"(?i)(shape|10|output)"):
        diagnostic.collect_position_predictions(model, make_loader(targets), torch.device("cpu"))


@pytest.mark.parametrize("source, value", [("prediction", float("nan")), ("prediction", float("inf")),
                                         ("target", float("nan")), ("target", float("inf"))])
def test_collect_does_not_silently_hide_nonfinite_positions(diagnostic, source, value):
    targets, predictions = position_fixture()
    (predictions if source == "prediction" else targets)[1, 5] = value
    model = IndexedPredictions(predictions)

    with pytest.raises(ValueError, match=r"(?i)(finite|nan|inf)"):
        diagnostic.collect_position_predictions(model, make_loader(targets), torch.device("cpu"))


def test_xy_bins_average_distance_per_prediction_and_keep_sparse_cells_unknown(diagnostic):
    x_edges, y_edges, means, counts = diagnostic.bin_xy_mean(
        np.array([0.1, 0.2, 0.3, 2.0]),
        np.array([0.1, 0.2, 0.3, 2.0]),
        np.array([1.0, 1.0, 10.0, 7.0]),
        bin_width_cm=1.0,
    )

    # Every prediction contributes, including the maximum exactly on a bin edge.
    assert counts.sum() == 4
    assert counts.shape == means.shape == (len(x_edges) - 1, len(y_edges) - 1)
    assert counts[0, 0] == 3
    assert means[0, 0] == pytest.approx(4.0)
    assert counts[-1, -1] == 1
    assert means[-1, -1] == pytest.approx(7.0)
    assert np.isnan(means[counts == 0]).all()
    assert np.count_nonzero(counts) == 2
    assert np.sum(means[counts > 0] * counts[counts > 0]) / counts.sum() == pytest.approx(4.75)


def test_z_bins_keep_constant_height_and_sparse_height_samples(diagnostic):
    edges, means, counts = diagnostic.bin_z_mean(
        np.array([2.0, 2.0, 2.0]), np.array([2.0, 4.0, 9.0]), bin_width_cm=1.0,
    )

    assert counts.sum() == 3
    assert np.count_nonzero(counts) == 1
    np.testing.assert_allclose(means[counts > 0], [5.0])
    assert edges[0] <= 2.0 <= edges[-1]

    edges, means, counts = diagnostic.bin_z_mean(
        np.array([0.1, 0.2, 3.0]), np.array([1.0, 3.0, 7.0]), bin_width_cm=1.0,
    )
    assert counts.sum() == 3
    assert counts[0] == 2 and counts[-1] == 1
    assert means[0] == pytest.approx(2.0) and means[-1] == pytest.approx(7.0)
    assert np.isnan(means[counts == 0]).all()


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_bin_width_must_be_finite_and_positive(diagnostic, width):
    with pytest.raises(ValueError):
        diagnostic.bin_xy_mean(np.array([0.0]), np.array([0.0]), np.array([1.0]), bin_width_cm=width)
    with pytest.raises(ValueError):
        diagnostic.bin_z_mean(np.array([0.0]), np.array([1.0]), bin_width_cm=width)


def test_plots_write_viewable_pngs_with_constant_z_and_sparse_xy(diagnostic, tmp_path):
    from PIL import Image

    # Both cubes have identical heights; distant XY values leave empty map bins.
    true_xyz = np.array([
        [[-0.1, -0.1, 0.02], [0.1, 0.1, 0.02]],
        [[-0.1, -0.1, 0.02], [0.1, 0.1, 0.02]],
    ])
    errors = np.array([[0.001, 0.002], [0.005, 0.010]])

    paths = diagnostic.plot_position_error_maps(true_xyz, errors, tmp_path, epoch=100)

    assert len(paths) == 2
    for path in paths:
        assert path.parent == tmp_path
        with Image.open(path) as picture:
            assert picture.format == "PNG"
            assert picture.width > 100 and picture.height > 100
            picture.verify()
