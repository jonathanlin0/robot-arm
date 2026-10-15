"""Pretraining optimization behavior."""

from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, small_config, small_dataset,
)
from pretrain_pickup import (
    INPUT_NAMES,
    action_means,
    create_policy,
    run_epoch,
)


class OptimizationTests(unittest.TestCase):
    def test_action_means_match_deterministic_policy(self):
        torch.manual_seed(5)
        policy = create_policy(small_config(), torch.device("cpu"))
        policy.set_training_mode(False)
        batch = small_dataset().tensors
        observations = dict(zip(INPUT_NAMES, batch[:3]))
        with torch.no_grad():
            actual = action_means(policy, observations)
            for index in range(observations["tokens"].shape[1]):
                prefix = {name: tensor[:, :index + 1] for name, tensor in observations.items()}
                expected = policy._predict(prefix, deterministic=True)
                torch.testing.assert_close(actual[:, index], expected)
        self.assertTrue(torch.all(actual.abs() <= 1))

    def test_only_transformer_actor_and_mean_train_and_tiny_fit_improves(self):
        torch.manual_seed(3)
        config = small_config(learning_rate=0.01, batch_size=7)
        policy = create_policy(config, torch.device("cpu"))
        loader = DataLoader(small_dataset(), batch_size=7)
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
        trainable = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
        optimized = {id(parameter) for group in policy.optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(trainable, optimized)
        exploration = list(policy.exploration_mlp.parameters())
        self.assertTrue(exploration)
        self.assertTrue(all(not parameter.requires_grad for parameter in exploration))
        self.assertTrue(all(id(parameter) not in optimized for parameter in exploration))
        initial = run_epoch(policy, loader, config, training=False)["loss"]
        for _ in range(15):
            run_epoch(policy, loader, config, training=True)
        final = run_epoch(policy, loader, config, training=False)["loss"]
        self.assertLess(final, initial * 0.5)
        parameters = dict(policy.named_parameters())
        for prefix in ("features_extractor.", "mlp_extractor.policy_net.", "action_net."):
            self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in parameters.items()
                                if name.startswith(prefix)), prefix)
        for name, parameter in parameters.items():
            if name == "log_std" or name.startswith(("mlp_extractor.value_net.", "value_net.",
                                                     "exploration_mlp.")):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_loss_weights_short_final_batch_and_empty_loader(self):
        config = small_config(gripper_loss_weight=2.0)
        tensors = list(small_dataset(count=3).tensors)
        tensors[3] = torch.tensor([[1.0] * 4, [1.0] * 4, [3.0] * 4])[:, None].repeat(1, 4, 1)
        policy = SimpleNamespace(device=torch.device("cpu"), set_training_mode=Mock())
        with patch.object(module, "action_means", side_effect=lambda policy, inputs: torch.zeros((*inputs["tokens"].shape[:2], 4))):
            result = run_epoch(policy, DataLoader(TensorDataset(*tensors), batch_size=2), config, training=False)
        self.assertAlmostEqual(result["loss"], 55 / 12, places=5)
        self.assertAlmostEqual(result["xyz_mse"], 11 / 3, places=5)
        self.assertAlmostEqual(result["gripper_mse"], 11 / 3, places=5)
        empty = TensorDataset(*(tensor[:0] for tensor in tensors))
        with self.assertRaises(ValueError):
            run_epoch(policy, DataLoader(empty, batch_size=2), config, training=False)

