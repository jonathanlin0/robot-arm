"""Pretraining orchestration behavior."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import contextlib
import io
import sys
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, fake_run, settings, small_config, small_dataset,
)
from pretrain_pickup import (
    PreparedData,
    WANDB_ENTITY_NAME,
    WANDB_PROJECT_NAME,
    checkpoint_path,
    run_smoke,
    run_wandb_sweep,
    train_pretraining,
)


class OrchestrationTests(unittest.TestCase):
    def test_training_cadence_final_evaluation_and_no_checkpoint(self):
        config = small_config(epochs=21)
        data = PreparedData(small_dataset(), small_dataset(), [dict(seed=7, settings=settings())], 4, 2)
        run = fake_run()
        output = io.StringIO()
        def epoch(policy, loader, configuration, *, training):
            return dict(loss=0.5 if training else 0.25, xyz_mse=0.2, gripper_mse=0.3)
        success = dict(validation_success_rate=0.25, validation_successes=1,
                       validation_episodes=4, validation_reset_failures=0,
                       validation_ik_failures=2,
                       validation_grasp_successes=3, validation_grasp_success_rate=0.75)
        threads = torch.get_num_threads()
        with patch.object(module, "create_policy", return_value=Mock()), \
                patch.object(module, "run_epoch", side_effect=epoch) as run_epoch_mock, \
                patch.object(module, "evaluate_success", return_value=success) as evaluate, \
                patch.object(torch, "save", side_effect=AssertionError("No model saving")), \
                contextlib.redirect_stdout(output):
            rows = train_pretraining(config, prepared=data, wandb_run=run, show_progress=False)
        self.assertEqual(len(rows), 21)
        self.assertEqual(run_epoch_mock.call_count, 42)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual([row["epoch"] for row in rows if "validation_success_rate" in row], [10, 20, 21])
        printed = [line for line in output.getvalue().splitlines() if line.startswith("Epoch ")]
        self.assertEqual([line.split(" | ")[0] for line in printed], ["Epoch 10/21", "Epoch 20/21", "Epoch 21/21"])
        self.assertTrue(all("train loss=" in line and "validation loss=" in line for line in printed))
        self.assertTrue(all(" | orange grasp=3/4 (75.0%)" in line for line in printed))
        self.assertTrue(all(" | IK failures=2/4" in line for line in printed))
        self.assertEqual(run.log.call_count, 21)
        self.assertEqual(run.summary["validation_success_rate"], 0.25)
        self.assertEqual(run.summary["validation_grasp_successes"], 3)
        self.assertEqual(run.summary["validation_grasp_success_rate"], 0.75)
        self.assertEqual(run.summary["validation_ik_failures"], 2)
        self.assertEqual([call.args[0] for call in run.log.call_args_list], rows)
        for row in rows:
            if row["epoch"] in (10, 20, 21):
                self.assertEqual(row["validation_grasp_successes"], 3)
                self.assertEqual(row["validation_grasp_success_rate"], 0.75)
                self.assertEqual(row["validation_ik_failures"], 2)
            else:
                self.assertNotIn("validation_grasp_successes", row)
                self.assertNotIn("validation_grasp_success_rate", row)
                self.assertNotIn("validation_ik_failures", row)
        run.log_artifact.assert_not_called()
        self.assertEqual(torch.get_num_threads(), threads)
        for changes in ({"history_length": 3}, {"sequence_stride": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train_pretraining(config, prepared=replace(data, **changes), show_progress=False)

    def test_every_evaluation_prints_grasp_count_when_print_cadence_differs(self):
        config = small_config(epochs=7, validation_interval_epochs=3, print_interval_epochs=5)
        data = PreparedData(small_dataset(), small_dataset(), [dict(seed=7, settings=settings())], 4, 2)
        output = io.StringIO()
        results = [dict(validation_success_rate=(count - 1) / 4,
                        validation_successes=count - 1,
                        validation_episodes=4, validation_reset_failures=0,
                        validation_ik_failures=3 - count,
                        validation_grasp_successes=count,
                        validation_grasp_success_rate=count / 4)
                   for count in (1, 2, 3)]
        with patch.object(module, "create_policy", return_value=Mock()), \
                patch.object(module, "run_epoch", return_value=dict(loss=0.5, xyz_mse=0.2, gripper_mse=0.3)), \
                patch.object(module, "evaluate_success", side_effect=results) as evaluate, \
                patch.object(module, "save_checkpoint") as save, \
                contextlib.redirect_stdout(output):
            rows = train_pretraining(config, prepared=data, show_progress=False)

        self.assertEqual(evaluate.call_count, 3)
        printed = [line for line in output.getvalue().splitlines() if line.startswith("Epoch ")]
        self.assertEqual([line.split(" | ")[0] for line in printed],
                         ["Epoch 3/7", "Epoch 5/7", "Epoch 6/7", "Epoch 7/7"])
        self.assertIn(" | orange grasp=1/4 (25.0%)", printed[0])
        self.assertIn(" | IK failures=2/4", printed[0])
        self.assertNotIn("orange grasp=", printed[1])
        self.assertNotIn("validation success=", printed[1])
        self.assertNotIn("IK failures=", printed[1])
        self.assertIn(" | orange grasp=2/4 (50.0%)", printed[2])
        self.assertIn(" | IK failures=1/4", printed[2])
        self.assertIn(" | orange grasp=3/4 (75.0%)", printed[3])
        self.assertIn(" | IK failures=0/4", printed[3])
        self.assertEqual([row["epoch"] for row in rows if "validation_grasp_successes" in row], [3, 6, 7])
        save.assert_not_called()

    def test_smoke_limits_before_loading_and_has_no_saved_output(self):
        data = PreparedData(small_dataset(history_length=8), small_dataset(history_length=8), [], 8, 4)
        rows = [dict(train_loss=1.0), dict(train_loss=0.8, validation_success_rate=0.0,
                                         validation_grasp_successes=1, validation_grasp_success_rate=1.0,
                                         validation_ik_failures=1,
                                         validation_episodes=1)]
        with patch.object(module, "prepare_data", return_value=data) as prepare, \
                patch.object(module, "train_pretraining", return_value=rows) as train, \
                patch.object(torch, "save", side_effect=AssertionError("No saving")), \
                contextlib.redirect_stdout(io.StringIO()):
            run_smoke(small_config())
        config = prepare.call_args.args[0]
        self.assertEqual(prepare.call_args.kwargs["episode_limit"], 2)
        self.assertEqual((config.epochs, config.history_length, config.validation_episodes), (2, 8, 1))
        self.assertEqual(config.sequence_stride, 4)
        self.assertEqual(config.maximum_episode_steps, 20)
        self.assertIs(train.call_args.kwargs["prepared"], data)

    def test_wandb_bayesian_objective_default_merge_and_cache(self):
        config = small_config(epochs=7, sweep_run_count=2)
        runs = [fake_run({"learning_rate": 0.0002, "batch_size": 16}),
                fake_run({"learning_rate": 0.0003, "batch_size": 8})]
        wandb = SimpleNamespace(init=Mock(side_effect=runs), sweep=Mock(return_value="new-id"), agent=Mock())
        wandb.agent.side_effect = lambda identifier, function, count: [function() for _ in range(count)]
        data = PreparedData(small_dataset(), small_dataset(), [], 4, 2)
        with patch.dict(sys.modules, {"wandb": wandb}), \
                patch.object(module, "prepare_data", return_value=data) as prepare, \
                patch.object(module, "train_pretraining") as train, \
                patch.object(torch, "save", side_effect=AssertionError("No saving")):
            run_wandb_sweep(config)
        sweep = wandb.sweep.call_args.kwargs["sweep"]
        self.assertEqual(sweep["name"], "PRETRAINING-waypoint_to_pickup")
        self.assertEqual(sweep["method"], "bayes")
        self.assertEqual(sweep["metric"], {"name": "validation_success_rate", "goal": "maximize"})
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(train.call_count, 2)
        for index, call in enumerate(train.call_args_list):
            sampled = call.args[0]
            self.assertEqual(sampled.epochs, config.epochs)
            self.assertEqual(sampled.learning_rate, runs[index].config["learning_rate"])
            self.assertEqual(sampled.batch_size, runs[index].config["batch_size"])
            self.assertIs(call.kwargs["prepared"], data)
            self.assertFalse(wandb.init.call_args_list[index].kwargs["save_code"])
            runs[index].log_artifact.assert_not_called()

    def test_wandb_existing_id_sequence_cache_invalidation_and_unknown_fields(self):
        config = small_config(sweep_run_count=3)
        wandb = SimpleNamespace(init=Mock(side_effect=[fake_run({"history_length": 4}),
                                                     fake_run({"history_length": 8}),
                                                     fake_run({"history_length": 8, "sequence_stride": 4})]),
                                sweep=Mock(), agent=Mock())
        wandb.agent.side_effect = lambda identifier, function, count: [function() for _ in range(count)]
        with patch.dict(sys.modules, {"wandb": wandb}), \
                patch.object(module, "prepare_data", side_effect=lambda value: SimpleNamespace(history_length=value.history_length, sequence_stride=value.sequence_stride)) as prepare, \
                patch.object(module, "train_pretraining"):
            run_wandb_sweep(config, "existing")
        self.assertEqual(prepare.call_count, 3)
        wandb.sweep.assert_not_called()
        self.assertEqual(wandb.agent.call_args.args[0], f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/existing")
        wandb.agent = Mock()
        with patch.dict(sys.modules, {"wandb": wandb}):
            run_wandb_sweep(config, "entity/project/id")
            self.assertEqual(wandb.agent.call_args.args[0], "entity/project/id")
            with patch.object(module, "WANDB_SWEEP_CONFIG", {"parameters": {"typo": {"value": 1}}}), \
                    self.assertRaises(ValueError):
                run_wandb_sweep(config)

    def test_named_wandb_exports_use_distinct_run_ids(self):
        config = small_config(save_name="pickup.zip", sweep_run_count=2)
        runs = [fake_run(), fake_run()]
        runs[0].id, runs[1].id = "run-a", "run-b"
        wandb = SimpleNamespace(init=Mock(side_effect=runs), sweep=Mock(return_value="sweep"), agent=Mock())
        wandb.agent.side_effect = lambda identifier, function, count: [function() for _ in range(count)]
        with patch.dict(sys.modules, {"wandb": wandb}), \
                patch.object(module, "prepare_data", return_value=Mock()), \
                patch.object(module, "train_pretraining") as train:
            run_wandb_sweep(config)
        self.assertEqual([checkpoint_path(call.args[0].save_name).name for call in train.call_args_list],
                         ["pickup-run-a.zip", "pickup-run-b.zip"])
        self.assertEqual(config.save_name, "pickup.zip")
        for run in runs:
            run.log_artifact.assert_not_called()

    def test_export_happens_once_after_final_epoch_only_when_requested(self):
        data = PreparedData(small_dataset(), small_dataset(), [dict(seed=7, settings=settings())], 4, 2)
        metrics = dict(loss=0.2, xyz_mse=0.1, gripper_mse=0.3)
        for save_name in (None, "final"):
            events = []
            config = small_config(epochs=11, save_name=save_name)
            policy = Mock()
            def epoch(*args, training, **kwargs):
                events.append("train" if training else "validation")
                return metrics
            def evaluate(*args):
                events.append("success")
                return dict(validation_success_rate=0.75, validation_successes=3,
                            validation_grasp_successes=4, validation_grasp_success_rate=1.0,
                            validation_ik_failures=1,
                            validation_episodes=4, validation_reset_failures=0)
            def export(exported_policy, exported_config, history, records):
                events.append("save")
                self.assertIs(exported_policy, policy)
                self.assertEqual(exported_config, config)
                self.assertEqual(len(history), 11)
                self.assertEqual(history[-1]["epoch"], 11)
                self.assertEqual(history[-1]["validation_success_rate"], 0.75)
                self.assertEqual(history[-1]["validation_grasp_successes"], 4)
                self.assertEqual(history[-1]["validation_grasp_success_rate"], 1.0)
                self.assertEqual(history[-1]["validation_ik_failures"], 1)
                self.assertEqual(history[-1]["validation_episodes"], 4)
                self.assertIs(records, data.validation_records)
                return checkpoint_path(config.save_name)
            with self.subTest(save_name=save_name), \
                    patch.object(module, "create_policy", return_value=policy), \
                    patch.object(module, "run_epoch", side_effect=epoch), \
                    patch.object(module, "evaluate_success", side_effect=evaluate), \
                    patch.object(module, "save_checkpoint", side_effect=export) as save, \
                    patch.object(torch, "save", side_effect=AssertionError("Export is mocked")), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_pretraining(config, prepared=data, show_progress=False)
            if save_name is None:
                save.assert_not_called()
            else:
                save.assert_called_once()
                self.assertEqual(events[-1], "save")
