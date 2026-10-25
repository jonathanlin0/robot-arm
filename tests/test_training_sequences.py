"""Offline tensor construction and overlap masking for full-episode training."""

import numpy as np
import pytest
import torch

from action_preprocessing import ActionObservationPreprocessor
from schemas import CubeState, RobotState
from trained_policy import TrainedPolicy
from training import build_action_windows, load_action_dataset, load_vision_dataset


def raw_observations(count, value=0.0):
    observations = torch.full((count, 41), value)
    observations[:, 24:28] = torch.tensor([1., 0., 0., 0.])
    observations[:, 34:38] = torch.tensor([1., 0., 0., 0.])
    return observations


def tilted_observations(angles):
    """Tilt near a top-face tie; pitch makes the two levelled yaws distinct."""
    observations = raw_observations(len(angles))
    beta = np.deg2rad(20) / 2
    for index, angle in enumerate(angles):
        alpha = np.deg2rad(angle) / 2
        # R = Ry(20 degrees) @ Rx(angle), in wxyz quaternion order.
        observations[index, 24:28] = torch.tensor([
            np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha),
            np.sin(beta) * np.cos(alpha), -np.sin(beta) * np.sin(alpha),
        ])
    return observations


@pytest.mark.parametrize("count,expected_windows", [(1, 1), (383, 1), (384, 1), (385, 2), (640, 2), (641, 3), (1024, 4)])
def test_sequence_windows_cover_each_label_once(count, expected_windows):
    observations = torch.zeros(count, 37)
    observations[:, 0] = torch.arange(count)
    actions = torch.zeros(count, 4)
    actions[:, 0] = torch.arange(count) / count
    targets = torch.ones(count, 3) * 0.2
    tokens, valid, starts, labels, loss_mask = build_action_windows(observations, targets, actions, history_length=384, stride=256)
    assert tokens.shape == (expected_windows, 384, 44)
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
            torch.testing.assert_close(tokens[window, 0, 37:41], actions[window * 256 - 1])
    assert torch.count_nonzero(tokens[0, 0, 37:41]) == 0


def test_invalid_window_stride_rejected():
    with pytest.raises(ValueError, match="stride"):
        build_action_windows(torch.zeros(3, 37), torch.zeros(3, 3), torch.zeros(3, 4), history_length=2, stride=3)


def test_action_dataset_is_fully_materialized_and_episode_histories_are_isolated(tmp_path):
    for index, count in enumerate((3, 7)):
        torch.save({"observations": raw_observations(count, float(index)),
                    "prev_accepted_targets": torch.zeros(count, 3), "actions": torch.ones(count, 4)}, tmp_path / f"{index}.pt")
    dataset = load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)
    assert len(dataset) == 4
    assert int(dataset.tensors[-1].sum()) == 10
    assert dataset.tensors[2].sum() == 2  # Once for each episode, not each chunk.
    for path in tmp_path.glob("*.pt"):
        path.unlink()
    assert dataset[0][0].shape == (4, 44)  # No later file reads are necessary.


@pytest.mark.parametrize("width", [36, 38, 40, 42])
def test_dataset_schema_rejects_widths_other_than_policy_37_or_legacy_41(tmp_path, width):
    torch.save({"observations": torch.zeros(3, width), "prev_accepted_targets": torch.zeros(3, 3),
                "actions": torch.zeros(3, 4)}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="observations"):
        load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)


def test_direct_vision_features_are_preserved_exactly_in_windows_and_runtime(tmp_path):
    count = 7
    observations = torch.arange(count * 37, dtype=torch.float32).reshape(count, 37) / 100
    observations[:, 24:26] = torch.tensor([
        [0., 0.], [0.1, 0.], [1e-7, 0.], [0., -1.], [0.6, 0.8], [-1., 0.], [0., 0.1],
    ])
    observations[:, 32:34] = observations[:, 24:26].flip(0)
    actions = torch.linspace(-1, 1, count * 4).reshape(count, 4)
    targets = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 100
    torch.save({"observations": observations, "actions": actions, "prev_accepted_targets": targets,
                "metadata": {"schema_version": 2, "observation_source": "vision_model"}},
               tmp_path / "vision.pt")

    tokens, valid, starts, labels, loss_mask = load_action_dataset(
        tmp_path, history_length=4, stride=2, show_progress=False,
    ).tensors
    assert tokens.shape == (3, 4, 44)
    assert loss_mask.sum() == count and starts.sum() == 1
    assert not (loss_mask & ~valid).any()
    offline = tokens[loss_mask]
    torch.testing.assert_close(offline[:, :37], observations, rtol=0, atol=0)
    torch.testing.assert_close(offline[:, 37:41], torch.cat((torch.zeros(1, 4), actions[:-1])))
    torch.testing.assert_close(offline[:, 41:], targets)
    torch.testing.assert_close(labels[loss_mask], actions)
    torch.testing.assert_close(tokens[0, 2:4, :37], tokens[1, :2, :37], rtol=0, atol=0)
    assert not torch.count_nonzero(tokens[~valid])

    class RecordedActions:
        index = 0

        def predict(self, inputs, deterministic):
            action = actions[self.index].numpy().copy()
            self.index += 1
            return action, None

    runtime = TrainedPolicy(RecordedActions(), history_length=4)
    for index, row in enumerate(observations.numpy()):
        timestamp = index * 0.05
        robot = RobotState(timestamp, row[:6], row[6:12], row[12:18], row[18:21])
        cubes = {
            "orange": CubeState(timestamp, row[21:24], None, row[26:29], row[24:26]),
            "blue": CubeState(timestamp, row[29:32], None, row[34:37], row[32:34]),
        }
        issued = runtime.predict(robot, cubes, targets[index].numpy(), deterministic=True)
        np.testing.assert_array_equal(runtime.tokens[runtime.valid_length - 1], offline[index].numpy())
        runtime.record_action(issued)


def test_mixed_legacy_and_vision_episodes_share_tokens_without_cross_episode_history(tmp_path):
    raw = tilted_observations([44, 45, 45])
    direct = torch.full((5, 37), 0.2)
    direct[:, 24:26] = torch.tensor([0.1, 0.])
    direct[:, 32:34] = torch.tensor([0., 0.])
    for name, observations, action in (("0-legacy", raw, 0.25), ("1-vision", direct, -0.75)):
        count = len(observations)
        torch.save({"observations": observations, "actions": torch.full((count, 4), action),
                    "prev_accepted_targets": torch.zeros(count, 3)}, tmp_path / f"{name}.pt")

    tokens, valid, starts, labels, loss_mask = load_action_dataset(tmp_path, 4, 2, False).tensors
    assert tokens.shape == (3, 4, 44)
    assert loss_mask.sum() == 8 and starts.sum() == 2
    expected_legacy = torch.from_numpy(ActionObservationPreprocessor().transform_episode(raw.numpy()))
    torch.testing.assert_close(tokens[loss_mask][:3, :37], expected_legacy)
    torch.testing.assert_close(tokens[loss_mask][3:, :37], direct, rtol=0, atol=0)
    torch.testing.assert_close(labels[loss_mask][:3], torch.full((3, 4), 0.25))
    torch.testing.assert_close(labels[loss_mask][3:], torch.full((5, 4), -0.75))
    assert not torch.count_nonzero(tokens[starts][:, 37:41])
    assert not (loss_mask & ~valid).any()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_direct_policy_observations_reject_nonfinite_features(tmp_path, nonfinite):
    observations = torch.zeros(2, 37)
    observations[1, 24] = nonfinite
    torch.save({"observations": observations, "actions": torch.zeros(2, 4),
                "prev_accepted_targets": torch.zeros(2, 3)}, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="finite"):
        load_action_dataset(tmp_path, 4, 2, False)


def test_nontied_yaw_features_match_overlaps_and_runtime_tokens(tmp_path):
    raw = tilted_observations([44, 46, 46, 46, 46, 46, 48, 48])
    targets = torch.arange(len(raw) * 3, dtype=torch.float32).reshape(-1, 3) / 100
    actions = torch.zeros(len(raw), 4)
    actions[:, 0] = torch.linspace(0.1, 0.8, len(raw))
    actions[:, 3] = 1
    torch.save({"observations": raw, "prev_accepted_targets": targets, "actions": actions},
               tmp_path / "episode.pt")
    expected = ActionObservationPreprocessor().transform_episode(raw.numpy())
    chunk_fresh = ActionObservationPreprocessor().transform_episode(raw[2:].numpy())
    # A frame with a unique best face is independent of earlier frames/chunking.
    np.testing.assert_allclose(expected[2:], chunk_fresh, atol=1e-7)
    dataset = load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)
    tokens, valid, starts, labels, loss_mask = dataset.tensors
    assert tokens.shape == (3, 4, 44)
    offline_tokens = tokens[loss_mask]
    np.testing.assert_allclose(offline_tokens[:, :37], expected, atol=1e-7)
    torch.testing.assert_close(tokens[0, 2:4, :37], tokens[1, :2, :37])
    torch.testing.assert_close(tokens[1, 2:4, :37], tokens[2, :2, :37])
    torch.testing.assert_close(labels[loss_mask], actions)

    class RecordedActions:
        index = 0

        def predict(self, inputs, deterministic):
            action = actions[self.index].numpy().copy()
            self.index += 1
            return action, None

    runtime = TrainedPolicy(RecordedActions(), history_length=4)
    direct_runtime = TrainedPolicy(RecordedActions(), history_length=4)

    def states(row, index):
        timestamp = index * 0.05
        robot = RobotState(timestamp, row[:6], row[6:12], row[12:18], row[18:21])
        cubes = {
            "orange": CubeState(timestamp, row[21:24], row[24:28], row[28:31]),
            "blue": CubeState(timestamp, row[31:34], row[34:38], row[38:41]),
        }
        return robot, cubes

    for index, row in enumerate(raw.numpy()):
        robot, cubes = states(row, index)
        issued = runtime.predict(robot, cubes, targets[index].numpy(), deterministic=True)
        np.testing.assert_allclose(runtime.tokens[runtime.valid_length - 1], offline_tokens[index], atol=1e-7)
        runtime.record_action(issued)
        for name, yaw_slice in (("orange", slice(24, 26)), ("blue", slice(32, 34))):
            cubes[name].orientation = None
            cubes[name].yaw = expected[index, yaw_slice]
        direct_issued = direct_runtime.predict(robot, cubes, targets[index].numpy(), deterministic=True)
        np.testing.assert_allclose(direct_runtime.tokens[direct_runtime.valid_length - 1],
                                   offline_tokens[index], atol=1e-7)
        direct_runtime.record_action(direct_issued)

    # Reset changes history bookkeeping while preserving this frame's features.
    runtime.reset()
    runtime.policy.index = 0
    robot, cubes = states(raw[2].numpy(), 0)
    runtime.predict(robot, cubes, targets[2].numpy(), deterministic=True)
    np.testing.assert_allclose(runtime.tokens[0, :37], chunk_fresh[0], atol=1e-7)
    np.testing.assert_array_equal(runtime.tokens[0, 37:41], 0)
    assert runtime.episode_start_mask[0]


def test_loading_next_episode_resets_tied_face_selection(tmp_path):
    first = tilted_observations([44, 45, 45])
    second = tilted_observations([45, 45])
    for name, raw in (("first", first), ("second", second)):
        torch.save({"observations": raw, "prev_accepted_targets": torch.zeros(len(raw), 3),
                    "actions": torch.zeros(len(raw), 4)}, tmp_path / f"{name}.pt")
    dataset = load_action_dataset(tmp_path, history_length=4, stride=2, show_progress=False)
    tokens = dataset.tensors[0]
    fresh_second = ActionObservationPreprocessor().transform_episode(second.numpy())
    np.testing.assert_allclose(tokens[1, :2, :37], fresh_second, atol=1e-7)
    assert not np.allclose(tokens[0, 1, 24:26], tokens[1, 0, 24:26])


def test_exact_face_ties_keep_full_episode_context_before_windowing(tmp_path):
    raw = tilted_observations([44, 45, 45, 45, 45, 45, 46, 46])
    expected = ActionObservationPreprocessor().transform_episode(raw.numpy())
    fresh_chunk = ActionObservationPreprocessor().transform_episode(raw[2:].numpy())
    assert not np.allclose(expected[2, 24:26], fresh_chunk[0, 24:26])
    torch.save({"observations": raw, "prev_accepted_targets": torch.zeros(len(raw), 3),
                "actions": torch.zeros(len(raw), 4)}, tmp_path / "ties.pt")

    tokens, _, _, _, loss_mask = load_action_dataset(
        tmp_path, history_length=4, stride=2, show_progress=False,
    ).tensors
    np.testing.assert_allclose(tokens[loss_mask][:, :37], expected, atol=1e-7)
    torch.testing.assert_close(tokens[0, 2:4, :37], tokens[1, :2, :37])

    class Action:
        def predict(self, inputs, deterministic):
            return np.zeros(4), None

    runtime = TrainedPolicy(Action(), history_length=4)
    for index, row in enumerate(raw.numpy()):
        robot = RobotState(index * 0.05, row[:6], row[6:12], row[12:18], row[18:21])
        cubes = {
            "orange": CubeState(robot.timestamp, row[21:24], None, row[28:31], expected[index, 24:26]),
            "blue": CubeState(robot.timestamp, row[31:34], None, row[38:41], expected[index, 32:34]),
        }
        runtime.predict(robot, cubes, np.zeros(3))
        np.testing.assert_allclose(runtime.tokens[runtime.valid_length - 1, :37], expected[index], atol=1e-7)


def test_vision_dataset_converts_layout_and_preserves_byte_storage(tmp_path):
    wrist = torch.zeros(2, 128, 128, 3, dtype=torch.uint8)
    wrist[..., 0] = 255
    external = torch.arange(2 * 300 * 300 * 3).reshape(2, 300, 300, 3).remainder(256).to(torch.uint8)
    torch.save({"wrist_images": wrist, "external_images": external,
                "joint_positions": torch.zeros(2, 6), "cube_poses": torch.zeros(2, 10)}, tmp_path / "images.pt")
    dataset = load_vision_dataset(tmp_path, show_progress=False)
    assert dataset.tensors[0].shape == (2, 3, 128, 128)
    assert dataset.tensors[0].dtype == torch.uint8
    assert (dataset.tensors[0][:, 0] == 255).all()
    assert (dataset.tensors[0][:, 1:] == 0).all()
    torch.testing.assert_close(dataset.tensors[1], external.permute(0, 3, 1, 2))
    assert dataset.tensors[1].shape == (2, 3, 300, 300)
    assert dataset.tensors[1].dtype == torch.uint8


def test_vision_dataset_preserves_normalized_float_images(tmp_path):
    image = torch.full((2, 3, 128, 128), 0.123456, dtype=torch.float32)
    external = torch.rand(2, 3, 300, 300)
    torch.save({"wrist_images": image, "external_images": external,
                "joint_positions": torch.zeros(2, 6), "cube_poses": torch.zeros(2, 10)}, tmp_path / "images.pt")
    dataset = load_vision_dataset(tmp_path, show_progress=False)
    torch.testing.assert_close(dataset.tensors[0], image)
    torch.testing.assert_close(dataset.tensors[1], external, rtol=0, atol=0)


@pytest.mark.parametrize("key,incorrect_size", [("wrist_images", 300), ("external_images", 128)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_vision_dataset_rejects_old_camera_resolution(tmp_path, key, incorrect_size, dtype):
    record = {"wrist_images": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
              "external_images": torch.zeros(1, 300, 300, 3, dtype=torch.uint8),
              "joint_positions": torch.zeros(1, 6), "cube_poses": torch.zeros(1, 10)}
    record[key] = torch.zeros(1, incorrect_size, incorrect_size, 3, dtype=torch.uint8)
    if dtype == torch.float32:
        for image_key in ("wrist_images", "external_images"):
            record[image_key] = record[image_key].permute(0, 3, 1, 2).float()
    torch.save(record, tmp_path / "old_images.pt")
    with pytest.raises(ValueError, match=rf"{key}.*Regenerate vision data"):
        load_vision_dataset(tmp_path, show_progress=False)


def test_missing_dataset_reports_directory(tmp_path):
    with pytest.raises(ValueError, match="No tensor files"):
        load_vision_dataset(tmp_path, show_progress=False)
