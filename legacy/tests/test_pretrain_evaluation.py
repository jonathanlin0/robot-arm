"""Pretraining evaluation behavior."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, settings, small_config, small_dataset,
)
from kinematics import IKConvergenceError
from pretrain_pickup import (
    DEFAULT_START_POSITION,
    INPUT_NAMES,
    StackSuccessConfig,
    evaluate_success,
    make_validation_environment,
)


def ik_failure():
    """Use the actual typed solver exception, including its diagnostic payload."""
    return IKConvergenceError({
        "tool_axis_tolerance": np.deg2rad(100.0),
        "attempts": [{"position_error": 0.000415,
                      "tool_axis_error": np.deg2rad(100.19), "tool_yaw_error": 0.0}],
    })


class EvaluationTests(unittest.TestCase):
    def test_policy_actions_success_denominator_and_mode_restoration(self):
        config = small_config(validation_episodes=6)
        history = dict(zip(INPUT_NAMES, [tensor[0].numpy() for tensor in small_dataset().tensors[:3]]))
        actions, seeds = [], []
        env = SimpleNamespace(close=Mock())
        simulation = SimpleNamespace(confirmed_grasp_seen=False)
        env.unwrapped = SimpleNamespace(simulation=simulation)
        policy = SimpleNamespace(device=torch.device("cpu"), training=True)
        policy.set_training_mode = lambda value: setattr(policy, "training", value)
        def reset(*, seed):
            seeds.append(seed)
            env.seed = seed
            simulation.confirmed_grasp_seen = False
            if seed == 11:
                raise RuntimeError("scene preparation failed")
            return history, {}
        def step(action):
            actions.append(action.copy())
            simulation.confirmed_grasp_seen = env.seed in (10, 14)
            return history, 0, env.seed != 12, env.seed == 12, dict(
                is_success=True, is_failure=env.seed == 13, orange_currently_held=env.seed == 14,
                stack_stable_time=0.1 if env.seed == 15 else StackSuccessConfig().required_stable_time,
                orange_grasp_hold_time=0.0)
        env.reset, env.step = reset, step
        records = [dict(seed=seed, settings=settings()) for seed in range(10, 17)]
        predicted = torch.tensor([[0.25, -0.4, 0.7, -1.0]])
        policy._predict = Mock(return_value=predicted)
        threads = torch.get_num_threads()
        with patch.object(module, "make_validation_environment", return_value=env):
            result = evaluate_success(policy, records, config)
        self.assertEqual(result, dict(validation_success_rate=1 / 6, validation_successes=1,
                                      validation_episodes=6, validation_reset_failures=1,
                                      validation_ik_failures=0,
                                      validation_grasp_successes=2, validation_grasp_success_rate=2 / 6))
        self.assertEqual(seeds, [10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(actions, predicted.numpy().repeat(5, axis=0))
        self.assertTrue(policy.training)
        self.assertEqual(torch.get_num_threads(), threads)
        env.close.assert_called_once()
        self.assertEqual(policy._predict.call_count, 5)
        self.assertTrue(all(call.kwargs["deterministic"] for call in policy._predict.call_args_list))

    def test_confirmed_grasps_count_once_per_episode_after_release_failure_and_timeout(self):
        config = small_config(validation_episodes=6, maximum_episode_steps=3)
        history = dict(zip(INPUT_NAMES, [tensor[0].numpy() for tensor in small_dataset().tensors[:3]]))
        simulation = SimpleNamespace(confirmed_grasp_seen=False)
        env = SimpleNamespace(unwrapped=SimpleNamespace(simulation=simulation), close=Mock())
        policy = SimpleNamespace(device=torch.device("cpu"), training=False,
                                 _predict=Mock(return_value=torch.zeros((1, 4))))
        policy.set_training_mode = lambda value: setattr(policy, "training", value)
        observed_holds = {}
        seeds = []

        def reset(*, seed):
            seeds.append(seed)
            env.seed, env.steps = seed, 0
            if seed == 11:
                # A failed reset can leave the previous episode's latch set.
                # It must still count as a failed trial, not another grasp.
                self.assertTrue(simulation.confirmed_grasp_seen)
                raise RuntimeError("scene preparation failed")
            simulation.confirmed_grasp_seen = False
            observed_holds[seed] = []
            return history, {}

        def step(action):
            env.steps += 1
            if env.seed in (10, 12, 14, 15):
                simulation.confirmed_grasp_seen = True
            currently_held = env.seed in (10, 12, 14) and env.steps == 1
            observed_holds[env.seed].append(currently_held)
            succeeded = env.seed == 10 and env.steps == 3
            failed = env.seed in (12, 13) and env.steps == 2
            # Seed 14 reaches the loop's maximum without an environment flag.
            # Seed 15 grasps and releases between policy observations.
            truncated = env.seed == 15 and env.steps == 3
            return history, 0.0, succeeded or failed, truncated, {
                "is_success": succeeded,
                "is_failure": failed,
                "orange_currently_held": currently_held,
                "stack_stable_time": StackSuccessConfig().required_stable_time if succeeded else 0.0,
            }

        env.reset, env.step = reset, step
        records = [dict(seed=seed, settings=settings()) for seed in range(10, 17)]
        with patch.object(module, "make_validation_environment", return_value=env):
            result = evaluate_success(policy, records, config)

        self.assertEqual(seeds, [10, 11, 12, 13, 14, 15])
        self.assertEqual(result["validation_grasp_successes"], 4)
        self.assertEqual(result["validation_grasp_success_rate"], 4 / 6)
        self.assertEqual(result["validation_successes"], 1)
        self.assertEqual(result["validation_success_rate"], 1 / 6)
        self.assertEqual(result["validation_reset_failures"], 1)
        self.assertEqual(result["validation_ik_failures"], 0)
        self.assertEqual(result["validation_episodes"], 6)
        self.assertEqual(observed_holds[15], [False, False, False])
        self.assertEqual(observed_holds[13], [False, False])
        self.assertEqual(policy._predict.call_count, 13)
        env.close.assert_called_once()
        self.assertFalse(policy.training)

    def test_ik_failures_keep_denominator_preserve_grasps_and_continue_next_scene(self):
        config = small_config(validation_episodes=6, maximum_episode_steps=3)
        history = dict(zip(INPUT_NAMES, [tensor[0].numpy() for tensor in small_dataset().tensors[:3]]))
        simulation = SimpleNamespace(confirmed_grasp_seen=False)
        env = SimpleNamespace(unwrapped=SimpleNamespace(simulation=simulation), close=Mock())
        policy = SimpleNamespace(device=torch.device("cpu"), training=True,
                                 _predict=Mock(return_value=torch.zeros((1, 4))))
        policy.set_training_mode = lambda value: setattr(policy, "training", value)
        seeds, steps = [], []

        def reset(*, seed):
            seeds.append(seed)
            env.seed, env.steps = seed, 0
            if seed == 11:
                # This failed reset retains the previous grasp latch. Do not
                # count it as a second grasp or retry/replace the failed scene.
                self.assertTrue(simulation.confirmed_grasp_seen)
                raise ik_failure()
            simulation.confirmed_grasp_seen = False
            if seed == 14:
                raise RuntimeError("unrelated scene preparation failed")
            return history, {}

        def step(action):
            env.steps += 1
            steps.append((env.seed, env.steps))
            if env.seed == 10 and env.steps == 1:
                simulation.confirmed_grasp_seen = True
            if (env.seed == 10 and env.steps == 2) or env.seed == 12:
                raise ik_failure()
            succeeded = env.seed == 13
            if succeeded:
                simulation.confirmed_grasp_seen = True
            return history, 0.0, succeeded, env.seed == 15, {
                "is_success": succeeded, "is_failure": False,
                "orange_currently_held": False,
                "stack_stable_time": StackSuccessConfig().required_stable_time,
            }

        env.reset, env.step = reset, step
        records = [dict(seed=seed, settings=settings()) for seed in range(10, 17)]
        threads = torch.get_num_threads()
        with patch.object(module, "make_validation_environment", return_value=env) as make:
            result = evaluate_success(policy, records, config)

        self.assertEqual(result, {
            "validation_success_rate": 1 / 6, "validation_successes": 1,
            "validation_episodes": 6, "validation_reset_failures": 2,
            "validation_ik_failures": 3,
            "validation_grasp_successes": 2, "validation_grasp_success_rate": 2 / 6,
        })
        self.assertEqual(seeds, [10, 11, 12, 13, 14, 15])
        self.assertEqual(steps, [(10, 1), (10, 2), (12, 1), (13, 1), (15, 1)])
        self.assertEqual(policy._predict.call_count, 5)
        make.assert_called_once()
        env.close.assert_called_once()
        self.assertTrue(policy.training)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_unrelated_step_exceptions_propagate_and_restore_resources(self):
        history = dict(zip(INPUT_NAMES, [tensor[0].numpy() for tensor in small_dataset().tensors[:3]]))
        for error in (RuntimeError("unexpected simulation failure"), ValueError("invalid action")):
            with self.subTest(error=type(error).__name__):
                policy = SimpleNamespace(device=torch.device("cpu"), training=False,
                                         _predict=Mock(return_value=torch.zeros((1, 4))))
                policy.set_training_mode = lambda value: setattr(policy, "training", value)
                env = SimpleNamespace(reset=Mock(return_value=(history, {})),
                                      step=Mock(side_effect=error), close=Mock())
                threads = torch.get_num_threads()
                with patch.object(module, "make_validation_environment", return_value=env), \
                        self.assertRaises(type(error)) as raised:
                    evaluate_success(policy, [dict(seed=7, settings=settings())], small_config())
                self.assertIs(raised.exception, error)
                env.reset.assert_called_once_with(seed=7)
                env.step.assert_called_once()
                env.close.assert_called_once()
                self.assertFalse(policy.training)
                self.assertEqual(torch.get_num_threads(), threads)

    def test_constructor_ik_failure_remains_fatal_and_restores_mode_and_threads(self):
        policy = SimpleNamespace(training=True)
        policy.set_training_mode = lambda value: setattr(policy, "training", value)
        error = ik_failure()
        threads = torch.get_num_threads()
        with patch.object(module, "make_validation_environment", side_effect=error) as make, \
                self.assertRaises(IKConvergenceError) as raised:
            evaluate_success(policy, [dict(seed=7, settings=settings())], small_config())
        self.assertIs(raised.exception, error)
        make.assert_called_once()
        self.assertTrue(policy.training)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_validation_environment_recreates_home_start_and_recorded_success_config(self):
        recorded = settings()
        recorded["success_config"]["required_stable_time"] = 0.75
        config = small_config()
        environment = make_validation_environment(config, recorded)
        try:
            base = environment.unwrapped
            self.assertFalse(base.start_at_orange_waypoint)
            self.assertEqual(base.recovery_start_config.probability, 0.0)
            self.assertEqual(base.maximum_episode_steps, config.maximum_episode_steps)
            self.assertEqual(asdict(base.simulation.success_config), recorded["success_config"])
            history, info = environment.reset(seed=7)
            self.assertEqual(history["tokens"].shape, (config.history_length, 56))
            self.assertEqual(info["episode_start_type"], "home")
        finally:
            environment.close()

    def test_live_validation_uses_fixed_center_despite_randomized_demonstration_starts(self):
        recorded = settings()
        recorded.update(start_position=[0.38, 0.01, 0.24],
                        start_position_half_range=[0.04, 0.04, 0.02])
        environment = make_validation_environment(small_config(), recorded)
        try:
            np.testing.assert_array_equal(environment.unwrapped.simulation.start_position,
                                          DEFAULT_START_POSITION)
            np.testing.assert_array_equal(environment.unwrapped.simulation.start_position_half_range,
                                          [0.0, 0.0, 0.0])
            for seed in (3, 7, 11):
                history, info = environment.reset(seed=seed)
                np.testing.assert_allclose(history["tokens"][0, 18:21], DEFAULT_START_POSITION, atol=0.005)
                np.testing.assert_allclose(info["target_gripper_position"], DEFAULT_START_POSITION, atol=0.005)
            self.assertEqual(recorded["start_position"], [0.38, 0.01, 0.24])
            self.assertEqual(recorded["start_position_half_range"], [0.04, 0.04, 0.02])
        finally:
            environment.close()

    def test_invalid_actions_restore_mode_and_close_environment(self):
        config = small_config()
        history = dict(zip(INPUT_NAMES, [tensor[0].numpy() for tensor in small_dataset().tensors[:3]]))
        policy = SimpleNamespace(device=torch.device("cpu"), training=False)
        policy.set_training_mode = lambda value: setattr(policy, "training", value)
        env = SimpleNamespace(reset=Mock(return_value=(history, {})), step=Mock(), close=Mock())
        policy._predict = Mock(return_value=torch.full((1, 4), float("nan")))
        threads = torch.get_num_threads()
        with patch.object(module, "make_validation_environment", return_value=env), \
                self.assertRaises(RuntimeError):
            evaluate_success(policy, [dict(seed=7, settings=settings())], config)
        env.step.assert_not_called()
        env.close.assert_called_once()
        self.assertFalse(policy.training)
        self.assertEqual(torch.get_num_threads(), threads)
        with self.assertRaises(ValueError):
            evaluate_success(policy, [], config)
