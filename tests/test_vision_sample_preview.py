"""The sample preview displays the training loader's augmented image stream."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import RandAugment

from schemas.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def preview():
    spec = importlib.util.spec_from_file_location(
        "vision_sample_preview", REPO_ROOT / "scripts" / "temp_view_vision_samples.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path):
    settings = load_config(REPO_ROOT / "configs" / "default.json").training.vision
    return settings.model_copy(update={
        "data_dir": tmp_path, "num_workers": 0, "batch_size": 7,
        "shuffle_train": False, "pin_memory": False,
    })


def image_dataset(count, dtype):
    values = torch.arange(count, dtype=torch.uint8)
    wrist = (10 + 3 * values).reshape(-1, 1, 1, 1).expand(-1, 3, 128, 128)
    external = (200 - 3 * values).reshape(-1, 1, 1, 1).expand(-1, 3, 300, 300)
    if dtype == torch.float32:
        wrist, external = wrist.float() / 255, external.float() / 255
    return TensorDataset(wrist, external, torch.zeros(count, 6), torch.zeros(count, 10))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("camera", ["wrist", "external"])
def test_grid_uses_first_twenty_loader_images_across_batches(preview, config, monkeypatch, dtype, camera):
    # Deliberately nonsequential order: the preview must preserve the sampler's order.
    order = [22, 0, 21, 1, 20, 2, 19, 3, 18, 4, 17, 5, 16, 6, 15, 7, 14, 8, 13, 9, 12, 10, 11]
    loader = DataLoader(image_dataset(23, dtype), sampler=order, batch_size=7)
    monkeypatch.setattr(preview, "create_vision_training_loader", lambda *args, **kwargs: loader)

    grid = preview.build_grid(config, camera)

    assert grid.mode == "RGB"
    assert grid.size == (1500, 1296)
    for slot, sample_index in enumerate(order[:20]):
        expected = 10 + 3 * sample_index if camera == "wrist" else 200 - 3 * sample_index
        x, y = (slot % 5) * 300 + 150, (slot // 5) * 324 + 24 + 150
        assert grid.getpixel((x, y)) == (expected, expected, expected)


def test_grid_requires_twenty_samples(preview, config, monkeypatch):
    loader = DataLoader(image_dataset(19, torch.uint8), batch_size=7)
    monkeypatch.setattr(preview, "create_vision_training_loader", lambda *args, **kwargs: loader)

    with pytest.raises(ValueError, match=r"20.*19"):
        preview.build_grid(config, "external")


@pytest.mark.parametrize("pose_width", [10, 14])
def test_training_loader_reads_variable_shards_without_validation_data(config, monkeypatch, pose_width):
    from training import create_vision_training_loader

    train_dir = config.data_dir / "train"
    train_dir.mkdir()
    for index, count in enumerate((1, 3)):
        poses = torch.zeros(count, pose_width)
        poses[:, 3] = poses[:, pose_width // 2 + 3] = 1.0
        torch.save({
            "wrist_images": torch.full((count, 128, 128, 3), 20 + index, dtype=torch.uint8),
            "external_images": torch.full((count, 300, 300, 3), 50 + index, dtype=torch.uint8),
            "joint_positions": torch.zeros(count, 6),
            "cube_poses": poses,
        }, train_dir / f"{index}.pt")

    calls = []

    def augment(transform, image):
        calls.append((transform.num_ops, transform.magnitude))
        return 255 - image

    monkeypatch.setattr(RandAugment, "forward", augment)
    loader = create_vision_training_loader(config, show_progress=False)
    batch = next(iter(loader))

    assert len(loader.dataset) == 4
    assert batch[0][:, 0, 0, 0].tolist() == [235, 234, 234, 234]
    assert batch[1][:, 0, 0, 0].tolist() == [205, 204, 204, 204]
    assert batch[3].shape == (4, 10)
    torch.testing.assert_close(batch[3][:, [3, 8]], torch.ones(4, 2))
    assert calls == [(config.randaugment.num_ops, config.randaugment.magnitude)] * 8
