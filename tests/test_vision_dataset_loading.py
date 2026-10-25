"""Vision loaders preserve all saved rows and their paired views and labels."""

from pathlib import Path

import pytest
import torch
from torch import nn

from schemas.config import load_config
from training import load_vision_dataset


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def write_sequence(directory, name, count, offset=0):
    directory.mkdir(parents=True, exist_ok=True)
    ids = torch.arange(offset, offset + count)
    record = {
        "wrist_images": ids[:, None, None, None].expand(-1, 128, 128, 3).to(torch.uint8),
        "external_images": (ids + 100)[:, None, None, None].expand(-1, 300, 300, 3).to(torch.uint8),
        "joint_positions": (ids + 1000)[:, None].expand(-1, 6).float(),
        "cube_poses": (ids + 2000)[:, None].expand(-1, 10).float(),
    }
    torch.save(record, directory / name)
    return ids


def assert_aligned(tensors):
    wrist, external, joints, poses = tensors
    ids = joints[:, 0] - 1000
    assert wrist.shape == (len(ids), 3, 128, 128)
    assert external.shape == (len(ids), 3, 300, 300)
    torch.testing.assert_close(wrist, ids[:, None, None, None].expand_as(wrist).to(torch.uint8))
    torch.testing.assert_close(external, (ids + 100)[:, None, None, None].expand_as(external).to(torch.uint8))
    torch.testing.assert_close(joints, (ids + 1000)[:, None].expand_as(joints))
    torch.testing.assert_close(poses, (ids + 2000)[:, None].expand_as(poses))
    return ids.long()


def test_load_keeps_every_legacy_and_new_row_in_order_and_aligned(tmp_path):
    originals = [
        write_sequence(tmp_path, "a.pt", 45),
        write_sequence(tmp_path, "b.pt", 5, offset=60),
        write_sequence(tmp_path, "c.pt", 4, offset=80),
    ]

    before = torch.random.get_rng_state().clone()
    dataset = load_vision_dataset(tmp_path, show_progress=False)

    assert len(dataset) == 54
    assert torch.equal(before, torch.random.get_rng_state())
    loaded_ids = assert_aligned(dataset.tensors)
    torch.testing.assert_close(loaded_ids, torch.cat(originals))


def test_training_entrypoint_uses_every_saved_training_and_validation_row(tmp_path, monkeypatch):
    import training

    project = load_config(Path(__file__).parents[1] / "configs" / "default.json")
    config = project.training.vision
    config.data_dir = tmp_path
    config.device, config.batch_size, config.num_workers = "cpu", 4, 0
    config.seed, config.shuffle_train = 19, False
    config.randaugment.num_ops = 0
    for split in ("train", "validation"):
        write_sequence(tmp_path / split, "a.pt", 45)
        write_sequence(tmp_path / split, "b.pt", 5, offset=60)

    def inspect_training(*, model, model_type, loaders, config, device, log_file=None):
        assert model_type == "vision"
        train_loader, validation_loader = loaders
        assert len(train_loader.dataset) == 50
        assert len(validation_loader.dataset) == 50
        train_tensors = tuple(torch.cat(parts) for parts in zip(*train_loader))
        validation_tensors = tuple(torch.cat(parts) for parts in zip(*validation_loader))
        expected_ids = torch.cat((torch.arange(45), torch.arange(60, 65)))
        for tensors in (train_tensors, validation_tensors):
            loaded_ids = assert_aligned(tensors)
            torch.testing.assert_close(loaded_ids, expected_ids)
        return []

    monkeypatch.setattr(training, "VisionModel", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(training, "_train", inspect_training)
    assert training.train_vision(project) == []


def test_legacy_quaternion_shards_and_direct_yaw_shards_mix_without_rewriting_files(tmp_path):
    legacy_path = tmp_path / "a_legacy.pt"
    direct_path = tmp_path / "b_yaw.pt"
    common = {
        "wrist_images": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
        "external_images": torch.zeros(1, 300, 300, 3, dtype=torch.uint8),
        "joint_positions": torch.zeros(1, 6),
    }
    half_angle = torch.tensor(torch.pi / 16)
    legacy = torch.tensor([[0.1, 0.2, 0.3, half_angle.cos(), 0, 0, half_angle.sin(),
                            0.4, 0.5, 0.6, 1, 0, 0, 0]])
    direct = torch.tensor([[0.7, 0.8, 0.9, -0.6, 0.8, 0.3, 0.2, 0.1, 0, -1]])
    torch.save({**common, "cube_poses": legacy}, legacy_path)
    torch.save({**common, "cube_poses": direct}, direct_path)
    before = {path: path.read_bytes() for path in (legacy_path, direct_path)}

    dataset = load_vision_dataset(tmp_path, show_progress=False)

    labels = dataset.tensors[3]
    assert labels.shape == (2, 10)
    torch.testing.assert_close(labels[0], torch.tensor([0.1, 0.2, 0.3, 0., 1., 0.4, 0.5, 0.6, 1., 0.]),
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(labels[1], direct[0])
    assert all(path.read_bytes() == original for path, original in before.items())


def test_legacy_zero_quaternion_labels_fail_with_the_shard_path(tmp_path):
    path = tmp_path / "invalid_quaternion.pt"
    torch.save({"wrist_images": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
                "external_images": torch.zeros(1, 300, 300, 3, dtype=torch.uint8),
                "joint_positions": torch.zeros(1, 6), "cube_poses": torch.zeros(1, 14)}, path)
    with pytest.raises(ValueError, match="invalid_quaternion.pt"):
        load_vision_dataset(tmp_path, show_progress=False)
