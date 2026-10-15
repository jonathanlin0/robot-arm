"""Pretraining config behavior."""

from pathlib import Path
from unittest.mock import Mock, patch
import contextlib
import io
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, small_config,
)
from pretrain_pickup import (
    PretrainingConfig,
    REPOSITORY_ROOT,
    checkpoint_path,
    main,
    parse_arguments,
    resolve_device,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_validation_and_devices(self):
        self.assertEqual(PretrainingConfig().print_interval_epochs, 10)
        self.assertEqual(PretrainingConfig().device, "auto")
        self.assertEqual(PretrainingConfig().maximum_episode_steps, 900)
        self.assertEqual(PretrainingConfig().history_length, 384)
        self.assertEqual(PretrainingConfig().sequence_stride, 256)
        self.assertEqual(PretrainingConfig().batch_size, 8)
        self.assertTrue(PretrainingConfig(data_directory="data").data_directory.is_absolute())
        for changes in ({"epochs": 0}, {"batch_size": True}, {"cpu_threads": 0},
                        {"seed": -1}, {"seed": 2 ** 32}, {"dataloader_workers": -1},
                        {"learning_rate": 0}, {"weight_decay": -1},
                        {"gripper_loss_weight": float("nan")},
                        {"max_gradient_norm": float("inf")},
                        {"transformer_embedding_dim": 9}, {"sequence_stride": 0},
                        {"sequence_stride": 5},
                        {"sweep_run_count": 0}, {"device": "cuda"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                small_config(**changes)
        for available, expected in ((False, "cpu"), (True, "mps")):
            with patch.object(torch.backends.mps, "is_available", return_value=available):
                self.assertEqual(resolve_device("auto").type, expected)
                self.assertEqual(resolve_device("cpu").type, "cpu")
                if not available:
                    with self.assertRaises(RuntimeError):
                        resolve_device("mps")

    def test_cli_modes_and_explicit_overrides(self):
        self.assertIsNone(parse_arguments([]).wandb)
        self.assertEqual(parse_arguments(["--wandb"]).wandb, "")
        self.assertEqual(parse_arguments(["--wandb", "trial-id"]).wandb, "trial-id")
        for arguments in (["--wandb", "--smoke"], ["--test", "--smoke"], ["--device", "cuda"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(arguments)
        with patch.object(module, "train_pretraining") as train:
            main(["--device", "cpu", "--epochs", "3", "--batch-size", "5",
                  "--learning-rate", "0.0002", "--data-dir", str(REPOSITORY_ROOT / "data")])
            config = train.call_args.args[0]
            self.assertEqual((config.device, config.epochs, config.batch_size), ("cpu", 3, 5))
            self.assertEqual(config.learning_rate, 0.0002)
        for arguments, expected in ((["--wandb"], None), (["--wandb", "abc"], "abc")):
            with patch.object(module, "run_wandb_sweep") as sweep:
                main(arguments + ["--sweep-count", "2"])
                self.assertEqual(sweep.call_args.args[1], expected)
                self.assertEqual(sweep.call_args.args[0].sweep_run_count, 2)
        with patch.object(module, "run_smoke") as smoke:
            main(["--smoke", "--device", "cpu"])
            self.assertEqual(smoke.call_args.args[0].device, "cpu")

    def test_test_mode_dispatches_to_dedicated_suite_and_propagates_failure(self):
        for passed, exit_code in ((True, 0), (False, 1)):
            with self.subTest(passed=passed), \
                    patch.object(module, "run_tests", return_value=passed) as run, \
                    self.assertRaises(SystemExit) as result:
                main(["--test"])
            self.assertEqual(result.exception.code, exit_code)
            run.assert_called_once_with()

    def test_save_cli_defaults_and_valid_names_are_opt_in(self):
        self.assertIsNone(PretrainingConfig().save_name)
        self.assertIsNone(parse_arguments([]).save_name)
        self.assertEqual(parse_arguments(["--save"]).save_name, "default")
        self.assertEqual(parse_arguments(["--save", "--device", "cpu"]).save_name, "default")
        self.assertEqual(parse_arguments(["--save", "pickup_v1.zip"]).save_name, "pickup_v1.zip")
        for arguments in (["--save", "../outside"], ["--save", ""], ["--save", "--test"],
                          ["--save", "model", "--test"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_arguments(arguments)
        for arguments, target, expected_name in ((["--save"], "train_pretraining", "default"),
                                  (["--save", "normal"], "train_pretraining", "normal"),
                                  (["--smoke", "--save"], "run_smoke", "default"),
                                  (["--smoke", "--save", "smoke.zip"], "run_smoke", "smoke.zip")):
            with patch.object(module, target) as run:
                main(arguments + ["--device", "cpu"])
                self.assertEqual(run.call_args.args[0].save_name, expected_name)
                self.assertEqual(checkpoint_path(expected_name).name,
                                 expected_name if expected_name.endswith(".zip") else expected_name + ".zip")
        with self.assertRaises(ValueError):
            small_config(save_name="../outside")

    def test_pretrained_cli_defaults_and_explicit_paths(self):
        import train as ppo_training
        self.assertIsNone(ppo_training.parse_arguments([]).pretrained)
        for arguments, expected in ((["--pretrained"], Path("checkpoints/pretraining/default.zip")),
                                    (["--pretrained", "checkpoints/pretraining/custom.zip"],
                                     Path("checkpoints/pretraining/custom.zip"))):
            with self.subTest(arguments=arguments):
                self.assertEqual(ppo_training.parse_arguments(arguments).pretrained, expected)
                with patch.object(ppo_training, "train_ppo") as train:
                    ppo_training.main(arguments)
                    self.assertEqual(train.call_args.args[0].pretrained_checkpoint, expected)
