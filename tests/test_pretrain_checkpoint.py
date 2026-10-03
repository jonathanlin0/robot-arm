"""Pretraining checkpoint behavior."""

from dataclasses import asdict, replace
from gymnasium import spaces
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
import copy
import numpy as np
import tempfile
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, settings, small_config, small_dataset,
)
from pretrain_pickup import (
    DEFAULT_START_POSITION,
    INPUT_NAMES,
    action_means,
    checkpoint_path,
    create_policy,
    run_epoch,
    save_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pretraining-checkpoint-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "checkpoints" / "pretraining"
        patcher = patch.object(module, "PRETRAINING_CHECKPOINT_DIRECTORY", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = small_config(save_name="learned", learning_rate=0.005)
        torch.manual_seed(24)
        self.policy = create_policy(self.config, torch.device("cpu"))
        run_epoch(self.policy, DataLoader(small_dataset(), batch_size=7), self.config, training=True)
        self.policy.set_training_mode(False)
        self.history = [dict(epoch=1, train_loss=0.25, validation_loss=0.3, validation_success_rate=0.5)]
        self.records = [dict(seed=7, settings=settings())]

    def test_named_checkpoint_paths_reject_escape_and_create_no_directories(self):
        self.assertEqual(checkpoint_path("pickup_v1"), self.directory / "pickup_v1.zip")
        self.assertEqual(checkpoint_path("pickup_v1.zip"), self.directory / "pickup_v1.zip")
        for invalid in (None, 7, "", " ", ".", "..", ".zip", "../escape", "child/name",
                        "/absolute/name", "child\\name", "name\0bad", " leading", "trailing "):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                checkpoint_path(invalid)
        self.assertFalse(self.directory.exists())
        with self.assertRaises(ValueError):
            save_checkpoint(self.policy, replace(self.config, save_name=None), self.history, self.records)
        for history, records in (([], self.records), (self.history, [])):
            with self.subTest(history=history, records=records), self.assertRaises(ValueError):
                save_checkpoint(self.policy, self.config, history, records)
        self.assertFalse(self.directory.exists())

    def test_export_roundtrip_preserves_means_metrics_and_fresh_ppo_optimizer(self):
        import zipfile
        from stable_baselines3 import PPO
        inputs = dict(zip(INPUT_NAMES, small_dataset().tensors[:3]))
        with torch.no_grad():
            expected = action_means(self.policy, inputs)
        old_optimizer = self.policy.optimizer
        before = {name: parameter.requires_grad for name, parameter in self.policy.named_parameters()}
        with patch.object(module, "CubeStackGymEnvironment", side_effect=AssertionError("Export must not run physics")), \
                patch.object(module, "make_validation_environment", side_effect=AssertionError("Export must not run physics")):
            path = save_checkpoint(self.policy, self.config, self.history, self.records)
        self.assertEqual(path, self.directory / "learned.zip")
        self.assertEqual(list(self.directory.iterdir()), [path])
        with zipfile.ZipFile(path) as archive:
            self.assertTrue({"data", "policy.pth", "policy.optimizer.pth"}.issubset(archive.namelist()))
        loaded = PPO.load(path, device="cpu")
        loaded.policy.set_training_mode(False)
        with torch.no_grad():
            actual = action_means(loaded.policy, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        action, _ = loaded.predict(
            {name: tensor[0].numpy() for name, tensor in inputs.items()}, deterministic=True
        )
        np.testing.assert_allclose(action, expected[0, -1].numpy(), rtol=1e-6, atol=1e-7)
        self.assertEqual(loaded.pretraining_history, self.history)
        self.assertEqual(loaded.pretraining_environment_settings, self.records[0]["settings"])
        self.assertEqual(loaded.pretraining_config["save_name"], "learned")
        self.assertEqual(loaded.pretraining_config["history_length"], self.config.history_length)
        self.assertEqual(loaded.pretraining_config["actor_dim"], self.config.actor_dim)
        self.assertEqual(loaded.pickup_training_config["history_length"], self.config.history_length)
        self.assertEqual(loaded.pickup_training_config["task"], "stack_orange_on_blue")
        self.assertFalse(loaded.pickup_training_config["start_at_orange_waypoint"])
        self.assertEqual(loaded.pickup_training_config["recovery_start_probability"], 0.0)
        self.assertEqual(loaded.pickup_training_config["start_position"], list(DEFAULT_START_POSITION))
        self.assertEqual(loaded.pickup_training_config["start_position_half_range"], [0.0, 0.0, 0.0])
        self.assertEqual(loaded.pickup_training_config["success_config"],
                         self.records[0]["settings"]["success_config"])
        self.assertEqual(loaded.num_timesteps, 0)
        self.assertTrue(all(parameter.requires_grad for parameter in loaded.policy.parameters()))
        for name, parameter in loaded.policy.exploration_mlp.named_parameters():
            torch.testing.assert_close(parameter, self.policy.exploration_mlp.state_dict()[name],
                                       rtol=0, atol=0)
        self.assertIs(type(loaded.policy.optimizer), torch.optim.Adam)
        self.assertFalse(loaded.policy.optimizer.state)
        optimized = {id(parameter) for group in loaded.policy.optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimized, {id(parameter) for parameter in loaded.policy.parameters()})
        self.assertIs(self.policy.optimizer, old_optimizer)
        self.assertEqual(before, {name: parameter.requires_grad for name, parameter in self.policy.named_parameters()})

    def test_randomized_source_metadata_is_preserved_while_playback_uses_fixed_start(self):
        from stable_baselines3 import PPO

        self.records[0]["settings"].update(start_position=[0.38, 0.01, 0.24],
                                           start_position_half_range=[0.04, 0.04, 0.02])
        original = copy.deepcopy(self.records)
        path = save_checkpoint(self.policy, self.config, self.history, self.records)
        loaded = PPO.load(path, device="cpu")

        self.assertEqual(loaded.pretraining_environment_settings, original[0]["settings"])
        self.assertEqual(loaded.pickup_training_config["start_position"], list(DEFAULT_START_POSITION))
        self.assertEqual(loaded.pickup_training_config["start_position_half_range"], [0.0, 0.0, 0.0])
        self.assertEqual(self.records, original)

    def test_failed_export_preserves_existing_archive_and_removes_partial_file(self):
        from stable_baselines3 import PPO
        path = save_checkpoint(self.policy, self.config, self.history, self.records)
        original = path.read_bytes()
        def interrupted_save(model, destination, *args, **kwargs):
            Path(destination).write_bytes(b"incomplete archive")
            raise OSError("simulated write failure")
        with patch.object(PPO, "save", new=interrupted_save), self.assertRaises(OSError):
            save_checkpoint(self.policy, self.config, self.history, self.records)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_full_384_position_checkpoint_runs_standard_ppo_prediction(self):
        from stable_baselines3 import PPO

        config = small_config(history_length=384, sequence_stride=256, save_name="full-history")
        policy = create_policy(config, torch.device("cpu"))
        policy.set_training_mode(False)
        inputs = dict(zip(INPUT_NAMES, small_dataset(count=1, history_length=384).tensors[:3]))
        for tensor in inputs.values():
            tensor[:, 3:] = 0
        with torch.no_grad():
            expected = action_means(policy, inputs)[0, 2].numpy()

        path = save_checkpoint(policy, config, self.history, self.records)
        loaded = PPO.load(path, device="cpu")
        self.assertEqual(loaded.observation_space["tokens"].shape, (384, 56))
        self.assertEqual(loaded.pretraining_config["sequence_stride"], 256)
        action, _ = loaded.predict(
            {name: tensor[0].numpy() for name, tensor in inputs.items()}, deterministic=True
        )
        np.testing.assert_allclose(action, expected, rtol=1e-6, atol=1e-7)

    def test_main_training_adopts_architecture_and_transfers_only_actor_components(self):
        from stable_baselines3 import PPO
        import train as ppo_training
        path = save_checkpoint(self.policy, self.config, self.history, self.records)
        requested = ppo_training.PPOTrainingConfig(pretrained_checkpoint=path, learning_rate=0.0007,
                                                   recovery_start_probability=0.25, device="cpu")
        effective, source = ppo_training.load_pretraining_checkpoint(requested)
        for source_name, target_name in (("history_length", "history_length"),
                                         ("transformer_embedding_dim", "transformer_embedding_dim"),
                                         ("transformer_layers", "transformer_layers"),
                                         ("transformer_heads", "transformer_heads"),
                                         ("transformer_feedforward_dim", "transformer_feedforward_dim"),
                                         ("actor_dim", "model_dim"), ("actor_layers", "model_layers")):
            self.assertEqual(getattr(effective, target_name), getattr(self.config, source_name))
        self.assertEqual(effective.learning_rate, requested.learning_rate)
        self.assertEqual(effective.recovery_start_probability, requested.recovery_start_probability)
        self.assertEqual(effective.reward_config, requested.reward_config)
        unchanged, absent = ppo_training.load_pretraining_checkpoint(replace(requested, pretrained_checkpoint=None))
        self.assertIsNone(absent)
        self.assertEqual(unchanged.model_dim, requested.model_dim)

        destination = PPO.load(path, device="cpu")
        with torch.no_grad():
            for parameter in destination.policy.parameters():
                parameter.add_(0.25)
        before = {name: parameter.detach().clone() for name, parameter in destination.policy.named_parameters()}
        optimizer = destination.policy.optimizer
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        destination.num_timesteps = 123
        ppo_training.initialize_from_pretraining(destination, source)
        source_parameters = dict(source.policy.named_parameters())
        for name, parameter in destination.policy.named_parameters():
            if name.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net.")):
                torch.testing.assert_close(parameter, source_parameters[name], rtol=0, atol=0)
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertIs(destination.policy.optimizer, optimizer)
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertEqual(destination.num_timesteps, 123)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        exploration = list(destination.policy.exploration_mlp.parameters())
        self.assertTrue(exploration)
        self.assertTrue(all(parameter.requires_grad for parameter in exploration))
        self.assertTrue(all(id(parameter) in optimized for parameter in exploration))
        inputs = dict(zip(INPUT_NAMES, small_dataset().tensors[:3]))
        source.policy.set_training_mode(False)
        destination.policy.set_training_mode(False)
        with torch.no_grad():
            torch.testing.assert_close(action_means(destination.policy, inputs),
                                       action_means(source.policy, inputs), rtol=0, atol=0)
        destination.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        with self.assertRaises(ValueError):
            ppo_training.initialize_from_pretraining(destination, source)
        source.pretraining_config["transformer_embedding_dim"] *= 2
        with patch.object(ppo_training.PPO, "load", return_value=source), self.assertRaises(ValueError):
            ppo_training.load_pretraining_checkpoint(requested)
