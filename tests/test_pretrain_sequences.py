"""Chunk coverage, chronological inputs, and per-timestep masked optimization."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pretrain_helpers import module, small_config
from pretrain_pickup import (
    PretrainingConfig,
    PRIVILEGED_OBSERVATION_SIZE,
    TOKEN_SIZE,
    materialize_sequences,
    run_epoch,
    sequence_chunks,
)


def episode_tensors(steps, offset=0):
    observations = (torch.arange(steps + 1, dtype=torch.float32) + offset)[:, None].repeat(
        1, PRIVILEGED_OBSERVATION_SIZE
    )
    targets = observations[:, :3] + 0.25
    actions = torch.linspace(-0.8, 0.8, steps)[:, None].repeat(1, 4)
    return observations, targets, actions


@pytest.mark.parametrize("steps", [1, 255, 256, 383, 384, 385, 512, 600, 640, 641, 896, 897, 1025])
def test_every_action_has_exactly_one_label_and_chunks_stop_at_episode_end(monkeypatch, steps):
    tensors = episode_tensors(steps)
    monkeypatch.setattr(module, "load_episode", lambda directory, record: tensors)
    dataset = materialize_sequences(Path("unused"), [{"steps": steps}], 384, "test", show_progress=False)
    inputs, valid, starts, labels, loss_mask = dataset.tensors
    chunks = sequence_chunks(steps, 384, 256)
    coverage = torch.zeros(steps, dtype=torch.int32)
    assert len(dataset) == (1 if steps <= 384 else 1 + (steps - 384 + 255) // 256)
    assert inputs.shape == (len(chunks), 384, TOKEN_SIZE)
    assert loss_mask.dtype == torch.bool
    observations, targets, actions = tensors
    for row, (start, end, first_label) in enumerate(chunks):
        length = end - start
        assert start == row * 256
        assert end == min(start + 384, steps)
        assert valid[row].sum() == length
        assert valid[row, :length].all() and not valid[row, length:].any()
        assert loss_mask[row].sum() > 0
        assert not loss_mask[row, :first_label - start].any()
        assert loss_mask[row, first_label - start:length].all()
        assert not loss_mask[row, length:].any()
        coverage[start:end] += loss_mask[row, :length].int()
        assert starts[row].sum() == (1 if start == 0 else 0)
        assert inputs[row, length:].count_nonzero() == 0
        assert labels[row, length:].count_nonzero() == 0
        torch.testing.assert_close(inputs[row, :length, :PRIVILEGED_OBSERVATION_SIZE], observations[start:end])
        torch.testing.assert_close(inputs[row, :length, -3:], targets[start:end])
        previous = torch.cat((torch.zeros(1, 4), actions[:-1]))[start:end]
        torch.testing.assert_close(inputs[row, :length, PRIVILEGED_OBSERVATION_SIZE:-3], previous)
        torch.testing.assert_close(labels[row, :length], actions[start:end])
    torch.testing.assert_close(coverage, torch.ones_like(coverage))
    assert loss_mask.sum() == steps
    assert chunks[-1][1] == steps


def test_requested_384_context_and_256_stride_boundaries():
    assert sequence_chunks(600, 384, 256) == [(0, 384, 0), (256, 600, 384)]
    assert sequence_chunks(700, 384, 256) == [(0, 384, 0), (256, 640, 384), (512, 700, 640)]
    assert sequence_chunks(320, 384, 256) == [(0, 320, 0)]
    assert sequence_chunks(640, 384, 256) == [(0, 384, 0), (256, 640, 384)]
    config = PretrainingConfig()
    assert (config.history_length, config.sequence_stride, config.batch_size) == (384, 256, 8)


@pytest.mark.parametrize("steps, context, stride", [
    (0, 384, 256), (True, 384, 256), (2, 0, 1), (2, 4, 0),
    (2, 4, 5), (2, 4, True), (2, 4.0, 2), (2, 4, 1.5),
])
def test_invalid_sequence_settings_are_rejected(steps, context, stride):
    with pytest.raises(ValueError):
        sequence_chunks(steps, context, stride)


@pytest.mark.parametrize("stride", [0, -1, True, 385, 1.5])
def test_config_rejects_invalid_stride(stride):
    with pytest.raises(ValueError):
        PretrainingConfig(sequence_stride=stride)


def test_sequences_do_not_cross_episodes_or_reset_history_at_chunk_boundaries(monkeypatch):
    episodes = {"first": episode_tensors(7, 100), "second": episode_tensors(3, 200)}
    records = [{"uuid": "first", "steps": 7}, {"uuid": "second", "steps": 3}]
    monkeypatch.setattr(module, "load_episode", lambda directory, record: episodes[record["uuid"]])
    tensors = materialize_sequences(Path("unused"), records, 4, "test", sequence_stride=2,
                                    show_progress=False).tensors
    inputs, valid, starts, labels, loss_mask = tensors
    assert len(inputs) == 4
    assert starts[:, 0].tolist() == [1, 0, 0, 1]
    assert valid.sum(1).tolist() == [4, 4, 3, 3]
    assert loss_mask.sum(1).tolist() == [4, 2, 1, 3]
    assert inputs[:, 0, 0].tolist() == [100, 102, 104, 200]
    torch.testing.assert_close(inputs[1, 0, PRIVILEGED_OBSERVATION_SIZE:-3], episodes["first"][2][1])
    assert inputs[3, 0, PRIVILEGED_OBSERVATION_SIZE:-3].count_nonzero() == 0
    assert inputs[valid.bool(), 0].max() == 202  # Terminal observations never become inputs.


def test_padding_and_overlap_labels_have_no_loss_or_prediction_gradient(monkeypatch):
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    loss_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]], dtype=torch.bool)
    tokens = torch.zeros(2, 6, TOKEN_SIZE)
    starts = torch.zeros_like(valid)
    labels = torch.full((2, 6, 4), float("nan"))
    labels[loss_mask] = 0
    predictions = torch.nn.Parameter(torch.ones(2, 6, 4))
    policy = SimpleNamespace(device=torch.device("cpu"), set_training_mode=Mock(),
                             parameters=lambda: iter([predictions]),
                             optimizer=torch.optim.SGD([predictions], lr=0))
    seen_shapes = []

    def predict(policy, observations):
        seen_shapes.append(observations["tokens"].shape)
        return predictions[:, :observations["tokens"].shape[1]]

    monkeypatch.setattr(module, "action_means", predict)
    config = small_config(history_length=6, sequence_stride=4, gripper_loss_weight=2,
                          max_gradient_norm=100)
    result = run_epoch(policy, DataLoader(TensorDataset(tokens, valid, starts, labels, loss_mask),
                                         batch_size=2), config, training=True)
    assert seen_shapes == [(2, 5, TOKEN_SIZE)]  # Trim the all-padding final column.
    assert result == {"loss": 1.25, "xyz_mse": 1.0, "gripper_mse": 1.0}
    expected = torch.zeros_like(predictions)
    expected[loss_mask] = torch.tensor([1., 1., 1., 2.]) * (2 / (6 * 4))
    torch.testing.assert_close(predictions.grad, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_epoch_metrics_weight_real_action_count_not_sequence_count(monkeypatch, batch_size):
    valid = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    labels = torch.tensor([1., 3.])[:, None, None].expand(2, 5, 4)
    dataset = TensorDataset(torch.zeros(2, 5, TOKEN_SIZE), valid, torch.zeros_like(valid), labels, valid.bool())
    policy = SimpleNamespace(device=torch.device("cpu"), set_training_mode=Mock())
    monkeypatch.setattr(module, "action_means", lambda policy, inputs:
                        torch.zeros((*inputs["tokens"].shape[:2], 4)))
    result = run_epoch(policy, DataLoader(dataset, batch_size=batch_size),
                       small_config(gripper_loss_weight=2), training=False)
    assert result["xyz_mse"] == pytest.approx(46 / 6)
    assert result["gripper_mse"] == pytest.approx(46 / 6)
    assert result["loss"] == pytest.approx((46 / 6) * 1.25)


def test_prepared_sequences_reject_different_stride_before_training():
    prepared = SimpleNamespace(history_length=4, sequence_stride=1)
    with pytest.raises(ValueError, match="stride"):
        module.train_pretraining(small_config(sequence_stride=2), prepared=prepared)
