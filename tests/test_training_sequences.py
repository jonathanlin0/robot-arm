"""Offline tensor construction and overlap masking for full-episode training."""

import pytest
import torch

from training import build_action_windows, load_action_dataset, load_vision_dataset


@pytest.mark.parametrize("count,expected_windows", [(1, 1), (383, 1), (384, 1), (385, 2), (640, 2), (641, 3), (1024, 4)])
def test_sequence_windows_cover_each_label_once(count, expected_windows):
    observations = torch.zeros(count, 41)
    observations[:, 0] = torch.arange(count)
    actions = torch.zeros(count, 4)
    actions[:, 0] = torch.arange(count) / count
    targets = torch.ones(count, 3) * 0.2
    tokens, valid, starts, labels, loss_mask = build_action_windows(observations, targets, actions, history_length=384, stride=256)
    assert tokens.shape == (expected_windows, 384, 48)
    assert loss_mask.sum() == count
    assert not (loss_mask & ~valid).any()
    assert starts.sum() == 1 and starts[0, 0]
    covered = tokens[..., 0][loss_mask]
    torch.testing.assert_close(covered, torch.arange(count, dtype=torch.float32))
    torch.testing.assert_close(labels[loss_mask], actions)
    for window in range(expected_windows):
        length = int(valid[window].sum())
        assert valid[window, :length].all() and not valid[window, length:].any()
        assert torch.count_nonzero(tokens[window, length:]) == 0
        assert tokens[window, 0, 0] == window * 256
        if window > 0:
            torch.testing.assert_close(tokens[window, 0, 41:45], actions[window * 256 - 1])
    assert torch.count_nonzero(tokens[0, 0, 41:45]) == 0


def test_invalid_window_stride_rejected():
    with pytest.raises(ValueError, match="stride"):
        build_action_windows(torch.zeros(3, 41), torch.zeros(3, 3), torch.zeros(3, 4), history_length=2, stride=3)


def test_action_dataset_is_fully_materialized_and_episode_histories_are_isolated(tmp_path):
    for index, count in enumerate((3, 7)):
        torch.save({"observations": torch.full((count, 41), float(index)),
                    "prev_accepted_targets": torch.zeros(count, 3), "actions": torch.ones(count, 4)}, tmp_path / f"{index}.pt")
    dataset = load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)
    assert len(dataset) == 4
    assert int(dataset.tensors[-1].sum()) == 10
    assert dataset.tensors[2].sum() == 2  # Once for each episode, not each chunk.
    for path in tmp_path.glob("*.pt"):
        path.unlink()
    assert dataset[0][0].shape == (4, 48)  # No later file reads are necessary.


def test_dataset_schema_rejects_old_observation_layout(tmp_path):
    torch.save({"observations": torch.zeros(3, 42), "prev_accepted_targets": torch.zeros(3, 3),
                "actions": torch.zeros(3, 4)}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="observations"):
        load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)


def test_vision_dataset_converts_layout_and_preserves_byte_storage(tmp_path):
    wrist = torch.zeros(2, 300, 300, 3, dtype=torch.uint8)
    wrist[..., 0] = 255
    torch.save({"wrist_images": wrist, "external_images": wrist.clone(),
                "joint_positions": torch.zeros(2, 6), "cube_poses": torch.zeros(2, 14)}, tmp_path / "images.pt")
    dataset = load_vision_dataset(tmp_path, show_progress=False)
    assert dataset.tensors[0].shape == (2, 3, 300, 300)
    assert dataset.tensors[0].dtype == torch.uint8
    assert (dataset.tensors[0][:, 0] == 255).all()
    assert (dataset.tensors[0][:, 1:] == 0).all()


def test_missing_dataset_reports_directory(tmp_path):
    with pytest.raises(ValueError, match="No tensor files"):
        load_vision_dataset(tmp_path, show_progress=False)
