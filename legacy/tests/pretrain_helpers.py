"""Shared deterministic fixtures for standalone pretraining tests."""

from pathlib import Path
from dataclasses import asdict
from torch.utils.data import TensorDataset
from typing import Any
from unittest.mock import Mock
import json
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pretrain_pickup as module
from pretrain_pickup import (
    CartesianActionConfig,
    CubeSpawnConfig,
    DEFAULT_SCENE_PATH,
    PretrainingConfig,
    StackRewardConfig,
    StackSuccessConfig,
    TOKEN_SIZE,
)


def small_config(**changes: Any) -> PretrainingConfig:
    defaults = dict(device="cpu", cpu_threads=1, epochs=2, batch_size=4,
                    history_length=4, sequence_stride=2, transformer_embedding_dim=8,
                    transformer_layers=1, transformer_heads=2,
                    transformer_feedforward_dim=16, actor_dim=8, actor_layers=1,
                    weight_decay=0.0, validation_episodes=2, maximum_episode_steps=3)
    return PretrainingConfig(**(defaults | changes))


def settings() -> dict[str, Any]:
    return json.loads(json.dumps({
        "task": "stack_orange_on_blue",
        "scene": str(DEFAULT_SCENE_PATH), "start_at_orange_waypoint": False,
        "recovery_start_probability": 0.0,
        "action_config": asdict(CartesianActionConfig()),
        "spawn_config": asdict(CubeSpawnConfig()),
        "waypoint_height": StackRewardConfig().approach_orange_height_offset,
        "success_config": asdict(StackSuccessConfig()),
        "maximum_episode_steps": 600, "action_interval": 0.05,
    }))


def small_dataset(count: int = 7, history_length: int = 4) -> TensorDataset:
    generator = torch.Generator().manual_seed(41)
    tokens = torch.randn((count, history_length, TOKEN_SIZE), generator=generator) * 0.1
    valid = torch.ones((count, history_length))
    starts = torch.zeros_like(valid)
    starts[:, 0] = 1
    labels = torch.tensor([0.25, -0.3, 0.4, -0.6]).repeat(count, history_length, 1)
    loss_mask = valid.bool()
    return TensorDataset(tokens, valid, starts, labels, loss_mask)


def fake_run(config: dict[str, Any] | None = None) -> Any:
    run = Mock()
    run.id = "test-run"
    run.config = {} if config is None else config
    run.summary = {}
    run.__enter__ = Mock(return_value=run)
    run.__exit__ = Mock(return_value=False)
    run.log_artifact.side_effect = AssertionError("Models must not be uploaded.")
    return run


_saved_thread_counts = []

def setUpModule():
    """Keep these small CPU tests fast and restore caller settings afterward."""
    _saved_thread_counts.append(torch.get_num_threads())
    torch.set_num_threads(1)

def tearDownModule():
    torch.set_num_threads(_saved_thread_counts.pop())
