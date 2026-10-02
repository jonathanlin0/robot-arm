from collections import deque
from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import train
from train import (
    ROLLOUT_METRIC_FILE_NAMES,
    GRIPPER_ACTION_INDEX,
    PROMISING_RUN_ROLLOUT_WINDOW,
    GripperStandardDeviationFloorCallback,
    PPOTrainingConfig,
    RolloutMetricsCallback,
)
from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy
from action_observation_history import ActionObservationHistoryWrapper
from cartesian_actions import CartesianActionConfig
from rewards import StackRewardConfig
from temporal_features import ActionObservationTransformer


class FakeEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePPO:
    def __init__(self) -> None:
        self.learned_timesteps: int | None = None
        self.learned_callback: BaseCallback | None = None
        self.learn_calls: list[tuple[int, bool]] = []
        self.saved_path: Path | None = None

    def learn(
        self,
        *,
        total_timesteps: int,
        callback: BaseCallback,
        reset_num_timesteps: bool = True,
    ) -> "FakePPO":
        self.learned_timesteps = total_timesteps
        self.learned_callback = callback
        self.learn_calls.append((total_timesteps, reset_num_timesteps))
        return self

    def save(self, path: Path) -> None:
        self.saved_path = path


@pytest.mark.parametrize("start_at_orange_waypoint", [False, True])
def test_create_validation_environment_uses_training_config(
    monkeypatch: pytest.MonkeyPatch,
    start_at_orange_waypoint: bool,
) -> None:
    constructor_arguments: dict[str, object] = {}
    expected_environment = FakeEnvironment()
    wrapped_environment = FakeEnvironment()
    wrapper_arguments = {}

    def fake_environment_constructor(**kwargs) -> FakeEnvironment:
        constructor_arguments.update(kwargs)
        return expected_environment

    monkeypatch.setattr(
        train,
        "CubeStackGymEnvironment",
        fake_environment_constructor,
    )
    def fake_history_wrapper(env, **kwargs):
        wrapper_arguments.update(env=env, **kwargs)
        return wrapped_environment

    monkeypatch.setattr(train, "ActionObservationHistoryWrapper", fake_history_wrapper)
    config = PPOTrainingConfig(
        seed=17,
        maximum_episode_steps=123,
        history_length=17,
        start_at_orange_waypoint=start_at_orange_waypoint,
        recovery_start_probability=0.3,
        recovery_xy_offset_range=(0.025, 0.045),
        recovery_height_offset_range=(0.025, 0.035),
        recovery_closed_gripper_probability=0.6,
    )

    environment = train.create_validation_environment(config)

    assert environment is wrapped_environment
    assert wrapper_arguments == {"env": expected_environment, "history_length": 17}
    assert constructor_arguments == {
        "seed": 17,
        "maximum_episode_steps": 123,
        "reward_config": config.reward_config,
        "start_at_orange_waypoint": start_at_orange_waypoint,
        "recovery_start_probability": 0.3,
        "recovery_xy_offset_range": (0.025, 0.045),
        "recovery_height_offset_range": (0.025, 0.035),
        "recovery_closed_gripper_probability": 0.6,
    }


@pytest.mark.parametrize("start_at_orange_waypoint", [False, True])
def test_create_training_environment_uses_subprocess_workers(
    monkeypatch: pytest.MonkeyPatch,
    start_at_orange_waypoint: bool,
) -> None:
    constructor_arguments: dict[str, object] = {}
    expected_environment = FakeEnvironment()

    def fake_make_vec_env(**kwargs) -> FakeEnvironment:
        constructor_arguments.update(kwargs)
        return expected_environment

    monkeypatch.setattr(train, "make_vec_env", fake_make_vec_env)
    config = PPOTrainingConfig(
        seed=18,
        maximum_episode_steps=234,
        environment_count=4,
        start_at_orange_waypoint=start_at_orange_waypoint,
        recovery_start_probability=0.4,
        recovery_xy_offset_range=(0.035, 0.055),
        recovery_height_offset_range=(0.03, 0.04),
        recovery_closed_gripper_probability=0.7,
    )

    environment = train.create_training_environment(config)

    assert environment is expected_environment
    assert constructor_arguments == {
        "env_id": train.CubeStackGymEnvironment,
        "n_envs": 4,
        "seed": 18,
        "env_kwargs": {
            "maximum_episode_steps": 234,
            "reward_config": config.reward_config,
            "start_at_orange_waypoint": start_at_orange_waypoint,
            "recovery_start_probability": 0.4,
            "recovery_xy_offset_range": (0.035, 0.055),
            "recovery_height_offset_range": (0.03, 0.04),
            "recovery_closed_gripper_probability": 0.7,
        },
        "vec_env_cls": SubprocVecEnv,
        "vec_env_kwargs": {"start_method": "spawn"},
        "wrapper_class": ActionObservationHistoryWrapper,
        "wrapper_kwargs": {"history_length": config.history_length},
        "monitor_kwargs": {
            "info_keywords": ("orange_pregrasp_waypoint_reached",),
        },
    }


def test_training_defaults_use_4096_transitions_per_rollout() -> None:
    config = PPOTrainingConfig()

    assert config.environment_count == 8
    assert config.start_at_orange_waypoint
    assert config.recovery_start_probability == 0.0
    assert config.recovery_xy_offset_range == (0.03, 0.05)
    assert config.recovery_height_offset_range == (0.03, 0.05)
    assert config.recovery_closed_gripper_probability == 0.5
    assert config.rollout_steps == 512
    assert config.environment_count * config.rollout_steps == 4_096
    assert config.model_dim == 128
    assert config.model_layers == 2
    assert config.history_length == 384
    assert config.transformer_embedding_dim == 128
    assert config.transformer_layers == 3
    assert config.transformer_heads == 4
    assert config.transformer_feedforward_dim == 512
    assert config.entropy_coefficient == pytest.approx(0.0001)
    assert config.target_kl == pytest.approx(0.03)
    assert config.log_std_init == pytest.approx(math.log(0.2))
    assert config.sde_xyz_log_std_init == pytest.approx(math.log(0.2 / 7.1))
    assert config.sde_gripper_log_std_init == pytest.approx(math.log(0.5 / 7.1))
    assert config.use_expln
    assert config.minimum_gripper_standard_deviation == pytest.approx(0.5)
    assert config.use_state_dependent_exploration
    assert config.exploration_noise_resample_steps == 8
    assert config.evaluation_interval_rollouts == 25
    assert config.evaluation_episodes == 100
    assert config.evaluation_seed == 20_000
    assert config.reward_config == StackRewardConfig()


def test_vector_auto_resets_start_every_worker_at_waypoint_with_clean_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_environment_factory = train.CubeStackGymEnvironment

    def best_effort_environment(**kwargs):
        return real_environment_factory(
            **kwargs, action_config=CartesianActionConfig(require_downward=False)
        )

    monkeypatch.setattr(train, "CubeStackGymEnvironment", best_effort_environment)
    monkeypatch.setattr(
        train, "SubprocVecEnv", lambda env_fns, **kwargs: DummyVecEnv(env_fns)
    )
    environment = train.create_training_environment(
        PPOTrainingConfig(
            seed=17,
            environment_count=2,
            maximum_episode_steps=1,
            history_length=4,
            start_at_orange_waypoint=True,
            recovery_start_probability=0.0,
        )
    )
    try:
        observations = environment.reset()
        for _ in range(3):
            assert observations["tokens"].shape == (2, 4, 56)
            np.testing.assert_array_equal(observations["valid"].sum(axis=1), [1, 1])
            np.testing.assert_array_equal(observations["episode_start"][:, 0], [1, 1])
            for index, wrapped in enumerate(environment.envs):
                gym_environment = wrapped.unwrapped
                state = gym_environment.previous_state
                waypoint = state["orange_position"].copy()
                waypoint[2] += gym_environment.reward_config.approach_orange_height_offset
                assert np.linalg.norm(state["gripper_position"] - waypoint) <= 0.005
                assert gym_environment.reward_calculator.orange_pregrasp_waypoint_reached
                assert gym_environment.episode_step_count == 0
                assert gym_environment.simulation.get_hold_time() == 0.0
                np.testing.assert_array_equal(
                    observations["tokens"][index, 0, :49],
                    gym_environment.observation_builder.build(state),
                )
                np.testing.assert_array_equal(
                    observations["tokens"][index, 0, -7:-3], np.zeros(4)
                )
                np.testing.assert_allclose(
                    observations["tokens"][index, 0, -3:],
                    gym_environment.action_adapter.current_target_gripper_position,
                )

            observations, _, dones, infos = environment.step(np.zeros((2, 4)))

            assert dones.all()
            for info in infos:
                assert info["TimeLimit.truncated"]
                assert info["episode"]["l"] == 1
                assert info["episode"]["orange_pregrasp_waypoint_reached"]
                assert not info["is_success"]
                assert info["terminal_observation"]["valid"].sum() == 2
    finally:
        environment.close()


@pytest.mark.parametrize("overrides", [
    {"history_length": 0},
    {"transformer_layers": -1},
    {"transformer_heads": 0},
    {"transformer_embedding_dim": 127},
    {"transformer_feedforward_dim": 0},
    {"transformer_layers": 1.5},
    {"history_length": True},
])
def test_training_config_rejects_invalid_temporal_architecture(overrides) -> None:
    with pytest.raises(ValueError):
        PPOTrainingConfig(**overrides)


@pytest.mark.parametrize("name", ["sde_xyz_log_std_init", "sde_gripper_log_std_init"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "-3"])
def test_training_config_rejects_invalid_sde_initialization(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        PPOTrainingConfig(**{name: value})


@pytest.mark.parametrize("overrides", [
    {"recovery_start_probability": -0.1},
    {"recovery_start_probability": 1.1},
    {"recovery_start_probability": float("nan")},
    {"recovery_start_probability": True},
    {"recovery_closed_gripper_probability": -0.1},
    {"recovery_closed_gripper_probability": 1.1},
    {"recovery_closed_gripper_probability": float("inf")},
    {"recovery_xy_offset_range": (0.05, 0.03)},
    {"recovery_xy_offset_range": (-0.01, 0.03)},
    {"recovery_xy_offset_range": (0.03,)},
    {"recovery_height_offset_range": (0.04, 0.02)},
    {"recovery_height_offset_range": (0.02, float("nan"))},
    {"recovery_height_offset_range": (0.02, 0.08)},
    {"recovery_height_offset_range": (0.02, 0.09)},
    {"reward_config": StackRewardConfig(approach_orange_height_offset=0.03)},
])
def test_training_config_rejects_invalid_recovery_starts(overrides) -> None:
    with pytest.raises(ValueError):
        PPOTrainingConfig(**({"recovery_start_probability": 0.25} | overrides))


@pytest.mark.parametrize("overrides", [
    {"start_at_orange_waypoint": False},
    {"recovery_start_probability": 0.0},
])
def test_disabled_recovery_does_not_constrain_waypoint_height(overrides) -> None:
    config = PPOTrainingConfig(
        reward_config=StackRewardConfig(approach_orange_height_offset=0.03),
        **({"recovery_start_probability": 0.25} | overrides),
    )

    assert config.reward_config.approach_orange_height_offset == 0.03


@pytest.mark.parametrize("use_sde", [False, True])
@pytest.mark.parametrize("use_wandb", [False, True])
@pytest.mark.parametrize("start_at_orange_waypoint", [False, True])
def test_train_ppo_validates_trains_saves_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_sde: bool,
    use_wandb: bool,
    start_at_orange_waypoint: bool,
) -> None:
    validation_environment = FakeEnvironment()
    training_environment = FakeEnvironment()
    model = FakePPO()
    checker_calls: list[tuple[FakeEnvironment, bool]] = []
    ppo_arguments: dict[str, object] = {}
    factory_configs: list[PPOTrainingConfig] = []

    def create_validation(config):
        factory_configs.append(config)
        return validation_environment

    def create_training(config):
        factory_configs.append(config)
        return training_environment

    monkeypatch.setattr(
        train,
        "create_validation_environment",
        create_validation,
    )
    monkeypatch.setattr(
        train,
        "create_training_environment",
        create_training,
    )
    monkeypatch.setattr(
        train,
        "check_env",
        lambda env, warn: checker_calls.append((env, warn)),
    )

    def fake_ppo_constructor(**kwargs) -> FakePPO:
        ppo_arguments.update(kwargs)
        return model

    monkeypatch.setattr(train, "PPO", fake_ppo_constructor)
    checkpoint_path = tmp_path / "nested" / "test_policy"
    config = PPOTrainingConfig(
        seed=29,
        total_timesteps=321,
        maximum_episode_steps=234,
        learning_rate=1e-4,
        model_dim=256,
        model_layers=3,
        history_length=12,
        transformer_embedding_dim=48,
        transformer_layers=2,
        transformer_heads=3,
        transformer_feedforward_dim=96,
        rollout_steps=128,
        batch_size=32,
        training_epochs=4,
        clip_range=0.1,
        entropy_coefficient=0.007,
        sde_xyz_log_std_init=-3.2,
        sde_gripper_log_std_init=-2.4,
        minimum_gripper_standard_deviation=0.4,
        use_state_dependent_exploration=use_sde,
        exploration_noise_resample_steps=6,
        device="cpu",
        checkpoint_path=checkpoint_path,
        metrics_directory=tmp_path / "metrics",
        start_at_orange_waypoint=start_at_orange_waypoint,
        recovery_start_probability=0.25,
        evaluation_interval_rollouts=100,
    )

    wandb_run = SimpleNamespace(config={}) if use_wandb else None
    returned_model = train.train_ppo(config, wandb_run=wandb_run)

    assert returned_model is model
    assert checker_calls == [(validation_environment, True)]
    assert ppo_arguments == {
        "policy": TanhBoundedMeanActorCriticPolicy,
        "env": training_environment,
        "learning_rate": 1e-4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 4,
        "clip_range": 0.1,
        "ent_coef": 0.007,
        "target_kl": 0.03,
        "use_sde": use_sde,
        "sde_sample_freq": 6,
        "policy_kwargs": {
            "features_extractor_class": ActionObservationTransformer,
            "features_extractor_kwargs": {
                "embedding_dim": 48,
                "layer_count": 2,
                "head_count": 3,
                "feedforward_dim": 96,
            },
            "share_features_extractor": True,
            "activation_fn": torch.nn.ReLU,
            "net_arch": {
                "pi": [256, 256, 256],
                "vf": [256, 256, 256],
            },
            "log_std_init": math.log(0.2),
            "sde_log_std_init": (-3.2, -3.2, -3.2, -2.4) if use_sde else None,
            "use_expln": True,
            "squash_output": False,
        },
        "seed": 29,
        "device": "cpu",
        "verbose": 0,
    }
    assert model.learned_timesteps == 321
    assert model.learn_calls == [(321, True)]
    assert isinstance(model.learned_callback, CallbackList)
    assert len(model.learned_callback.callbacks) == (1 if use_sde else 2)
    if not use_sde:
        floor_callback = model.learned_callback.callbacks[0]
        assert isinstance(floor_callback, GripperStandardDeviationFloorCallback)
        assert floor_callback.minimum_log_standard_deviation == pytest.approx(math.log(0.4))
    metrics_callback = model.learned_callback.callbacks[-1]
    assert isinstance(metrics_callback, RolloutMetricsCallback)
    assert metrics_callback.metrics_directory == tmp_path / "metrics"
    assert metrics_callback.wandb_run is wandb_run
    assert metrics_callback.evaluation_config is config
    assert model.saved_path == checkpoint_path
    assert config.start_at_orange_waypoint is start_at_orange_waypoint
    assert len(factory_configs) == 2
    assert all(factory_config is config for factory_config in factory_configs)
    assert model.pickup_training_config == asdict(config)
    if wandb_run is not None:
        assert wandb_run.config == {
            "environment_start_at_orange_waypoint": start_at_orange_waypoint,
            "environment_recovery_start_probability": (
                0.25 if start_at_orange_waypoint else 0.0
            ),
            "environment_recovery_xy_offset_range": (0.03, 0.05),
            "environment_recovery_height_offset_range": (0.03, 0.05),
            "environment_recovery_closed_gripper_probability": 0.5,
            "evaluation_interval_rollouts": 100,
            "evaluation_episodes": 100,
            "evaluation_seed": 20_000,
            "evaluation_recovery_start_probability": 0.0,
            "evaluation_deterministic": True,
        }
    assert checkpoint_path.parent.is_dir()
    assert validation_environment.closed
    assert training_environment.closed


def test_train_ppo_closes_environment_when_training_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_environment = FakeEnvironment()
    training_environment = FakeEnvironment()
    model = FakePPO()

    monkeypatch.setattr(
        train,
        "create_validation_environment",
        lambda config: validation_environment,
    )
    monkeypatch.setattr(
        train,
        "create_training_environment",
        lambda config: training_environment,
    )
    monkeypatch.setattr(train, "check_env", lambda env, warn: None)
    monkeypatch.setattr(train, "PPO", lambda **kwargs: model)

    def fail_during_learning(
        *,
        total_timesteps: int,
        callback: BaseCallback,
    ) -> FakePPO:
        raise RuntimeError("training failed")

    monkeypatch.setattr(model, "learn", fail_during_learning)

    with pytest.raises(RuntimeError, match="training failed"):
        train.train_ppo(PPOTrainingConfig(total_timesteps=1))

    assert validation_environment.closed
    assert training_environment.closed


@pytest.mark.parametrize(
    ("episode_reward_means", "success_rates"),
    [
        (
            [
                float(value)
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW)
            ],
            [0.0] * PROMISING_RUN_ROLLOUT_WINDOW,
        ),
        (
            [
                float(PROMISING_RUN_ROLLOUT_WINDOW - value)
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW)
            ],
            [
                float(value) / PROMISING_RUN_ROLLOUT_WINDOW
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW)
            ],
        ),
    ],
)
def test_train_ppo_extends_promising_runs(
    episode_reward_means: list[float],
    success_rates: list[float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_environment = FakeEnvironment()
    training_environment = FakeEnvironment()
    model = FakePPO()
    monkeypatch.setattr(
        train,
        "create_validation_environment",
        lambda config: validation_environment,
    )
    monkeypatch.setattr(
        train,
        "create_training_environment",
        lambda config: training_environment,
    )
    monkeypatch.setattr(train, "check_env", lambda env, warn: None)
    monkeypatch.setattr(train, "PPO", lambda **kwargs: model)

    def learn_with_rollout_trends(
        *,
        total_timesteps: int,
        callback: CallbackList,
        reset_num_timesteps: bool = True,
    ) -> FakePPO:
        model.learn_calls.append((total_timesteps, reset_num_timesteps))
        if len(model.learn_calls) == 1:
            metrics_callback = next(
                child_callback
                for child_callback in callback.callbacks
                if isinstance(child_callback, RolloutMetricsCallback)
            )
            assert isinstance(metrics_callback, RolloutMetricsCallback)
            metrics_callback.recent_episode_reward_means.extend(
                episode_reward_means
            )
            metrics_callback.recent_success_rates.extend(success_rates)
        return model

    monkeypatch.setattr(model, "learn", learn_with_rollout_trends)

    train.train_ppo(
        PPOTrainingConfig(
            total_timesteps=1_000,
            checkpoint_path=tmp_path / "policy",
            metrics_directory=tmp_path / "metrics",
        )
    )

    assert model.learn_calls == [
        (1_000, True),
        (train.PROMISING_RUN_EXTENSION_TIMESTEPS, False),
    ]
    assert validation_environment.closed
    assert training_environment.closed


@pytest.mark.parametrize(
    ("episode_reward_means", "success_rates"),
    [
        (
            [
                float(PROMISING_RUN_ROLLOUT_WINDOW - value)
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW)
            ],
            [0.0] * PROMISING_RUN_ROLLOUT_WINDOW,
        ),
        (
            [2.5] * PROMISING_RUN_ROLLOUT_WINDOW,
            [0.25] * PROMISING_RUN_ROLLOUT_WINDOW,
        ),
        (
            [
                float(value)
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW - 1)
            ],
            [
                float(value) / (PROMISING_RUN_ROLLOUT_WINDOW - 1)
                for value in range(PROMISING_RUN_ROLLOUT_WINDOW - 1)
            ],
        ),
    ],
)
def test_train_ppo_does_not_extend_without_positive_full_window(
    episode_reward_means: list[float],
    success_rates: list[float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_environment = FakeEnvironment()
    training_environment = FakeEnvironment()
    model = FakePPO()
    monkeypatch.setattr(
        train,
        "create_validation_environment",
        lambda config: validation_environment,
    )
    monkeypatch.setattr(
        train,
        "create_training_environment",
        lambda config: training_environment,
    )
    monkeypatch.setattr(train, "check_env", lambda env, warn: None)
    monkeypatch.setattr(train, "PPO", lambda **kwargs: model)

    def learn_with_nonpositive_trends(
        *,
        total_timesteps: int,
        callback: CallbackList,
        reset_num_timesteps: bool = True,
    ) -> FakePPO:
        model.learn_calls.append((total_timesteps, reset_num_timesteps))
        metrics_callback = next(
            child_callback
            for child_callback in callback.callbacks
            if isinstance(child_callback, RolloutMetricsCallback)
        )
        assert isinstance(metrics_callback, RolloutMetricsCallback)
        metrics_callback.recent_episode_reward_means.extend(
            episode_reward_means
        )
        metrics_callback.recent_success_rates.extend(success_rates)
        return model

    monkeypatch.setattr(model, "learn", learn_with_nonpositive_trends)

    train.train_ppo(
        PPOTrainingConfig(
            total_timesteps=1_000,
            checkpoint_path=tmp_path / "policy",
            metrics_directory=tmp_path / "metrics",
        )
    )

    assert model.learn_calls == [(1_000, True)]
    assert validation_environment.closed
    assert training_environment.closed


def read_metric_values(metric_path: Path) -> list[float]:
    return [
        float(line)
        for line in metric_path.read_text(encoding="utf-8").splitlines()
    ]


def test_rollout_metrics_callback_writes_and_appends_rolling_means(
    tmp_path: Path,
) -> None:
    callback = RolloutMetricsCallback(tmp_path)
    episode_information = [
        {"l": 400, "r": 10.0, "orange_pregrasp_waypoint_reached": True},
        {"l": 200, "r": 30.0, "orange_pregrasp_waypoint_reached": False},
    ]
    successes = [False, True]
    callback.model = SimpleNamespace(
        ep_info_buffer=episode_information,
        ep_success_buffer=successes,
    )

    callback._on_training_start()
    callback._on_rollout_end()

    episode_information.append(
        {"l": 300, "r": 20.0, "orange_pregrasp_waypoint_reached": False}
    )
    successes.append(True)
    callback._on_rollout_end()

    assert read_metric_values(tmp_path / "ep_len_mean.txt") == [
        300.0,
        300.0,
    ]
    assert read_metric_values(tmp_path / "ep_rew_mean.txt") == [
        20.0,
        20.0,
    ]
    assert read_metric_values(tmp_path / "success_rate.txt") == [
        0.5,
        2.0 / 3.0,
    ]
    assert read_metric_values(tmp_path / "orange_waypoint_reach_rate.txt") == [
        0.5,
        1.0 / 3.0,
    ]
    assert list(callback.recent_episode_reward_means) == [20.0, 20.0]
    assert list(callback.recent_success_rates) == [0.5, 2.0 / 3.0]


def test_rollout_metrics_callback_saves_pickup_fraction_and_max_hold_time(
    tmp_path: Path,
) -> None:
    callback = RolloutMetricsCallback(tmp_path)
    callback.model = SimpleNamespace(
        ep_info_buffer=[
            {"l": 400, "r": 2.5, "orange_pregrasp_waypoint_reached": False}
        ],
        ep_success_buffer=[False],
    )

    callback._on_training_start()
    callback._on_rollout_start()
    callback.locals = {
        "infos": [
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
            {
                "orange_currently_held": True,
                "orange_grasp_hold_time": 0.15,
            },
        ]
    }
    assert callback._on_step()
    callback.locals = {
        "infos": [
            {
                "orange_currently_held": True,
                "orange_grasp_hold_time": 0.4,
            },
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
        ]
    }
    assert callback._on_step()
    callback._on_rollout_end()

    callback._on_rollout_start()
    callback.locals = {
        "infos": [
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
        ]
    }
    assert callback._on_step()
    callback._on_rollout_end()

    assert read_metric_values(
        tmp_path / "orange_currently_held.txt"
    ) == [0.5, 0.0]
    assert read_metric_values(
        tmp_path / "orange_grasp_hold_time.txt"
    ) == [0.4, 0.0]


def test_waypoint_rate_uses_completed_monitored_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScriptedEnvironment(gym.Env):
        def __init__(self, **kwargs) -> None:
            self.observation_space = gym.spaces.Box(
                -1.0, 1.0, shape=(1,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(4,), dtype=np.float32
            )
            self.episode_index = -1

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.episode_index += 1
            self.step_count = 0
            return np.zeros(1, dtype=np.float32), {
                "target_gripper_position": np.zeros(3),
            }

        def step(self, action):
            self.step_count += 1
            # The first episode stays at the waypoint for three steps but
            # times out without picking up. The second fails without reaching
            # it; the third reaches it and completes the pickup task.
            episode_length, outcome, waypoint_reached = (
                (3, "truncated", True),
                (2, "failure", False),
                (1, "success", True),
            )[self.episode_index]
            done = self.step_count == episode_length
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                done and outcome != "truncated",
                done and outcome == "truncated",
                {
                    "target_gripper_position": np.zeros(3),
                    "orange_pregrasp_waypoint_reached": waypoint_reached,
                    "orange_currently_held": False,
                    "orange_grasp_hold_time": 0.0,
                    "is_success": done and outcome == "success",
                    "is_failure": done and outcome == "failure",
                },
            )

    monkeypatch.setattr(train, "CubeStackGymEnvironment", ScriptedEnvironment)
    monkeypatch.setattr(
        train, "SubprocVecEnv", lambda env_fns, **kwargs: DummyVecEnv(env_fns)
    )
    environment = train.create_training_environment(
        PPOTrainingConfig(environment_count=1)
    )
    callback = RolloutMetricsCallback(tmp_path)
    # A shorter version of SB3's rolling episode window makes eviction clear.
    callback.model = SimpleNamespace(
        ep_info_buffer=deque(maxlen=2),
        ep_success_buffer=deque(maxlen=2),
    )
    terminal_infos = []
    try:
        environment.reset()
        callback._on_training_start()
        for _ in range(6):
            callback._on_rollout_start()
            _, _, dones, infos = environment.step(np.zeros((1, 4)))
            callback.locals = {"infos": infos}
            assert callback._on_step()
            BaseAlgorithm._update_info_buffer(callback.model, infos, dones)
            callback._on_rollout_end()
            if dones[0]:
                terminal_infos.append(infos[0])
    finally:
        environment.close()

    rates = read_metric_values(tmp_path / "orange_waypoint_reach_rate.txt")
    assert math.isnan(rates[0]) and math.isnan(rates[1])
    assert rates[2:] == [1.0, 1.0, 0.5, 0.5]
    assert terminal_infos[0]["TimeLimit.truncated"]
    assert not terminal_infos[0]["is_success"]
    assert terminal_infos[1]["is_failure"]
    assert terminal_infos[2]["is_success"]
    assert [
        info["episode"]["orange_pregrasp_waypoint_reached"]
        for info in terminal_infos
    ] == [True, False, True]
    assert read_metric_values(tmp_path / "success_rate.txt")[2:] == [
        0.0, 0.0, 0.0, 0.5
    ]


def test_rollout_metric_trend_windows_keep_only_latest_values() -> None:
    callback = RolloutMetricsCallback("unused")
    supplied_value_count = PROMISING_RUN_ROLLOUT_WINDOW + 50

    callback.recent_episode_reward_means.extend(
        float(value) for value in range(supplied_value_count)
    )
    callback.recent_success_rates.extend(
        float(value) / supplied_value_count
        for value in range(supplied_value_count)
    )

    assert (
        len(callback.recent_episode_reward_means)
        == PROMISING_RUN_ROLLOUT_WINDOW
    )
    assert callback.recent_episode_reward_means[0] == pytest.approx(50.0)
    assert callback.recent_episode_reward_means[-1] == pytest.approx(
        supplied_value_count - 1
    )
    assert len(callback.recent_success_rates) == PROMISING_RUN_ROLLOUT_WINDOW
    assert callback.recent_success_rates[0] == pytest.approx(
        50.0 / supplied_value_count
    )
    assert callback.recent_success_rates[-1] == pytest.approx(
        (supplied_value_count - 1) / supplied_value_count
    )


def test_callback_preserves_metrics_when_training_continues(
    tmp_path: Path,
) -> None:
    callback = RolloutMetricsCallback(tmp_path)
    callback.model = SimpleNamespace(
        ep_info_buffer=[
            {"l": 400, "r": 2.5, "orange_pregrasp_waypoint_reached": True}
        ],
        ep_success_buffer=[True],
    )

    callback._on_training_start()
    callback._on_rollout_end()
    callback._on_training_start()

    assert callback.rollout_count == 1
    assert list(callback.recent_episode_reward_means) == [2.5]
    assert list(callback.recent_success_rates) == [1.0]
    assert read_metric_values(tmp_path / "ep_rew_mean.txt") == [2.5]
    assert read_metric_values(tmp_path / "orange_waypoint_reach_rate.txt") == [1.0]


def test_rollout_metrics_callback_writes_aligned_nan_values_when_empty(
    tmp_path: Path,
) -> None:
    callback = RolloutMetricsCallback(tmp_path)
    callback.model = SimpleNamespace(
        ep_info_buffer=[],
        ep_success_buffer=[],
    )

    callback._on_training_start()
    callback._on_rollout_end()

    for file_name in ROLLOUT_METRIC_FILE_NAMES.values():
        values = read_metric_values(tmp_path / file_name)
        assert len(values) == 1
        assert values[0] != values[0]


def test_rollout_metrics_callback_clears_previous_run(
    tmp_path: Path,
) -> None:
    callback = RolloutMetricsCallback(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for file_name in ROLLOUT_METRIC_FILE_NAMES.values():
        (tmp_path / file_name).write_text("123.0\n", encoding="utf-8")

    callback._on_training_start()

    for file_name in ROLLOUT_METRIC_FILE_NAMES.values():
        assert (tmp_path / file_name).read_text(encoding="utf-8") == ""


def test_gripper_standard_deviation_floor_clamps_only_gripper() -> None:
    callback = GripperStandardDeviationFloorCallback(0.5)
    initial_standard_deviations = torch.tensor([0.2, 0.3, 0.4, 0.1])
    log_standard_deviations = torch.nn.Parameter(
        initial_standard_deviations.log()
    )
    callback.model = SimpleNamespace(
        policy=SimpleNamespace(log_std=log_standard_deviations)
    )

    callback._on_rollout_start()

    resulting_standard_deviations = log_standard_deviations.detach().exp()
    torch.testing.assert_close(
        resulting_standard_deviations[:GRIPPER_ACTION_INDEX],
        initial_standard_deviations[:GRIPPER_ACTION_INDEX],
    )
    assert resulting_standard_deviations[GRIPPER_ACTION_INDEX] == pytest.approx(
        0.5
    )


def test_gripper_standard_deviation_floor_preserves_larger_value() -> None:
    callback = GripperStandardDeviationFloorCallback(0.5)
    log_standard_deviations = torch.nn.Parameter(
        torch.tensor([0.2, 0.3, 0.4, 0.8]).log()
    )
    callback.model = SimpleNamespace(
        policy=SimpleNamespace(log_std=log_standard_deviations)
    )

    callback._on_training_end()

    assert log_standard_deviations.detach().exp()[3] == pytest.approx(0.8)


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, float("nan")])
def test_gripper_standard_deviation_callback_requires_positive_floor(
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GripperStandardDeviationFloorCallback(invalid_value)


def test_rollout_metrics_callback_logs_each_rollout_to_wandb(
    tmp_path: Path,
) -> None:
    logged_values: list[tuple[dict[str, float | int], int]] = []
    wandb_run = SimpleNamespace(
        log=lambda values, step: logged_values.append((values, step))
    )
    callback = RolloutMetricsCallback(tmp_path, wandb_run=wandb_run)
    callback.model = SimpleNamespace(
        ep_info_buffer=[
            {"l": 400, "r": 20.0, "orange_pregrasp_waypoint_reached": True}
        ],
        ep_success_buffer=[False],
        policy=SimpleNamespace(action_dist=SimpleNamespace(
            distribution=torch.distributions.Normal(
                torch.zeros(2, 4), torch.ones(2, 4) * 0.5,
            ),
        )),
    )

    callback._on_training_start()
    callback._on_rollout_start()
    callback.locals = {
        "actions": np.array([[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "clipped_actions": np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "infos": [
            {
                "orange_currently_held": True,
                "orange_grasp_hold_time": 0.25,
            },
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
        ]
    }
    assert callback._on_step()
    callback.num_timesteps = 2_048
    callback._on_rollout_end()
    callback._on_rollout_start()
    callback.locals = {
        "actions": np.zeros((2, 4)),
        "clipped_actions": np.zeros((2, 4)),
        "infos": [
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
            {
                "orange_currently_held": False,
                "orange_grasp_hold_time": 0.0,
            },
        ]
    }
    assert callback._on_step()
    callback.num_timesteps = 4_096
    callback._on_rollout_end()

    assert logged_values == [
        (
            {
                "ep_len_mean": 400.0,
                "ep_rew_mean": 20.0,
                "success_rate": 0.0,
                "orange_waypoint_reach_rate": 1.0,
                "orange_currently_held": 0.5,
                "orange_grasp_hold_time": 0.25,
                "action_std_x": 0.5,
                "action_std_y": 0.5,
                "action_std_z": 0.5,
                "action_std_gripper": 0.5,
                "action_clip_fraction": 0.25,
                "rollout": 1,
                "total_timesteps": 2_048,
            },
            1,
        ),
        (
            {
                "ep_len_mean": 400.0,
                "ep_rew_mean": 20.0,
                "success_rate": 0.0,
                "orange_waypoint_reach_rate": 1.0,
                "orange_currently_held": 0.0,
                "orange_grasp_hold_time": 0.0,
                "action_std_x": 0.5,
                "action_std_y": 0.5,
                "action_std_z": 0.5,
                "action_std_gripper": 0.5,
                "action_clip_fraction": 0.0,
                "rollout": 2,
                "total_timesteps": 4_096,
            },
            2,
        ),
    ]


def test_create_reward_config_overrides_only_named_values() -> None:
    config = train.create_reward_config(
        {
            "grasp_reward": 8.0,
            "ik_failure_penalty": -1.25,
        }
    )

    assert config.grasp_reward == pytest.approx(8.0)
    assert config.ik_failure_penalty == pytest.approx(-1.25)
    assert (
        config.approach_orange_progress_weight
        == StackRewardConfig().approach_orange_progress_weight
    )


def test_create_reward_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="not_a_reward_field"):
        train.create_reward_config({"not_a_reward_field": 1.0})


def test_wandb_sweep_ranges_include_defaults_and_map_to_training_config() -> None:
    sweep_config = train.WANDB_SWEEP_CONFIG
    defaults = PPOTrainingConfig()
    transformer_ranges = {
        "transformer_embedding_dim": {
            "distribution": "q_uniform", "min": 128, "max": 160, "q": 4,
        },
        "transformer_feedforward_dim": {
            "distribution": "int_uniform", "min": 512, "max": 768,
        },
        "transformer_layers": {
            "distribution": "int_uniform", "min": 3, "max": 4,
        },
    }

    train._validate_sweep_parameter_names()
    assert sweep_config["method"] == "bayes"
    assert sweep_config["metric"] == {
        "name": "success_rate",
        "goal": "maximize",
    }
    assert {
        f"{train.MODEL_PARAMETER_PREFIX}{name}" for name in transformer_ranges
    } <= sweep_config["parameters"].keys()
    assert sweep_config["parameters"]["model_sde_xyz_log_std_init"] == {
        "distribution": "uniform", "min": math.log(0.1 / 7.1), "max": -2.6,
    }
    assert sweep_config["parameters"]["model_target_kl"] == {
        "distribution": "uniform", "min": 0.005, "max": 0.03,
    }
    assert sweep_config["parameters"]["model_batch_size"] == {"values": [256, 512]}
    assert sweep_config["parameters"]["model_learning_rate"]["min"] == 2e-5
    for parameter_name, bounds in sweep_config["parameters"].items():
        if parameter_name.startswith(train.REWARD_PARAMETER_PREFIX):
            name = parameter_name.removeprefix(train.REWARD_PARAMETER_PREFIX)
            for value in bounds["values"]:
                assert getattr(train.create_reward_config({name: value}), name) == value
            continue
        assert parameter_name.startswith(train.MODEL_PARAMETER_PREFIX)
        name = parameter_name.removeprefix(train.MODEL_PARAMETER_PREFIX)
        field_name = train.MODEL_CONFIG_FIELDS[name]
        if "values" in bounds:
            assert getattr(defaults, field_name) in bounds["values"]
            sampled_values = bounds["values"]
        else:
            assert bounds["min"] < bounds["max"]
            assert bounds["min"] <= getattr(defaults, field_name) <= bounds["max"]
            sampled_values = (bounds["min"], bounds["max"])
        # A logarithmic parameter is sampled uniformly in log space; applying
        # log_uniform to its negative values would be invalid.
        if name in transformer_ranges:
            assert bounds == transformer_ranges[name]
        elif name.endswith("log_std_init") or name == "target_kl":
            assert bounds["distribution"] == "uniform"
        elif "values" not in bounds:
            assert bounds["distribution"] == "log_uniform_values"
            assert bounds["min"] > 0
        if name in transformer_ranges:
            sampled_values = range(bounds["min"], bounds["max"] + 1, bounds.get("q", 1))
        for sampled_value in sampled_values:
            # Quantized uniform samples can arrive from W&B as floats.
            value = float(sampled_value) if bounds.get("distribution") == "q_uniform" else sampled_value
            overrides = train.model_config_overrides({name: value})
            config = PPOTrainingConfig(**overrides)
            assert getattr(config, field_name) == pytest.approx(sampled_value)
            if name == "target_kl":
                assert isinstance(config.target_kl, float)
            if name == "batch_size":
                assert isinstance(config.batch_size, int)
            if name in transformer_ranges:
                assert isinstance(getattr(config, field_name), int)
                assert config.transformer_embedding_dim % config.transformer_heads == 0


def test_run_wandb_sweep_starts_unbounded_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_calls: list[dict[str, Any]] = []
    agent_calls: list[tuple[str, object]] = []
    validation_calls: list[bool] = []
    monkeypatch.setattr(
        train,
        "_validate_sweep_parameter_names",
        lambda: validation_calls.append(True),
    )
    monkeypatch.setattr(
        train.wandb,
        "sweep",
        lambda **kwargs: sweep_calls.append(kwargs) or "sweep-123",
    )
    monkeypatch.setattr(
        train.wandb,
        "agent",
        lambda sweep_id, function: agent_calls.append(
            (sweep_id, function)
        ),
    )

    train.run_wandb_sweep()

    assert validation_calls == [True]
    assert sweep_calls == [
        {
            "sweep": train.WANDB_SWEEP_CONFIG,
            "project": train.WANDB_PROJECT_NAME,
        }
    ]
    assert agent_calls == [("sweep-123", train.run_wandb_trial)]


@pytest.mark.parametrize(
    ("sweep_id", "expected_agent_id"),
    [
        (
            "existing-123",
            f"{train.WANDB_ENTITY_NAME}/{train.WANDB_PROJECT_NAME}/existing-123",
        ),
        ("other-team/other-project/existing-123", "other-team/other-project/existing-123"),
    ],
)
def test_run_wandb_sweep_joins_existing_sweep_without_creating_one(
    monkeypatch: pytest.MonkeyPatch,
    sweep_id: str,
    expected_agent_id: str,
) -> None:
    agent_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        train,
        "_validate_sweep_parameter_names",
        lambda: pytest.fail("Joining a sweep must not validate local sweep settings"),
    )
    monkeypatch.setattr(
        train.wandb,
        "sweep",
        lambda **kwargs: pytest.fail("Joining a sweep must not create another sweep"),
    )
    monkeypatch.setattr(
        train.wandb,
        "agent",
        lambda sweep_id, function: agent_calls.append((sweep_id, function)),
    )

    train.run_wandb_sweep(sweep_id)

    assert agent_calls == [(expected_agent_id, train.run_wandb_trial)]


class FakeWandbRun:
    def __init__(self, config: dict[str, float]) -> None:
        self.id = "test-run-id"
        self.config = config

    def __enter__(self) -> "FakeWandbRun":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_run_wandb_trial_applies_sampled_reward_and_model_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_values = {
        parameter_name: (
            parameter_config["min"]
            if "min" in parameter_config
            else parameter_config["values"][0]
        )
        for parameter_name, parameter_config in (
            train.WANDB_SWEEP_CONFIG["parameters"].items()
        )
    }
    fake_run = FakeWandbRun(sampled_values)
    init_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        train.wandb,
        "init",
        lambda **kwargs: init_calls.append(kwargs) or fake_run,
    )
    training_calls: list[tuple[PPOTrainingConfig, object]] = []

    monkeypatch.setattr(
        train,
        "train_ppo",
        lambda config, wandb_run: training_calls.append(
            (config, wandb_run)
        ),
    )

    train.run_wandb_trial()

    assert init_calls == [
        {
            "project": train.WANDB_PROJECT_NAME,
            "config": {
                **{
                    f"reward_{name}": value
                    for name, value in asdict(StackRewardConfig()).items()
                },
                **{
                    f"model_{name}": getattr(PPOTrainingConfig(), field_name)
                    for name, field_name in train.MODEL_CONFIG_FIELDS.items()
                },
            },
        }
    ]
    training_config, passed_run = training_calls[0]
    assert passed_run is fake_run
    for parameter_name, sampled_value in sampled_values.items():
        if parameter_name.startswith(train.REWARD_PARAMETER_PREFIX):
            field_name = parameter_name.removeprefix(train.REWARD_PARAMETER_PREFIX)
            assert getattr(training_config.reward_config, field_name) == pytest.approx(sampled_value)
            continue
        field_name = train.MODEL_CONFIG_FIELDS[
            parameter_name.removeprefix(train.MODEL_PARAMETER_PREFIX)
        ]
        assert getattr(training_config, field_name) == pytest.approx(sampled_value)
    assert training_config.reward_config == train.create_reward_config({
        name.removeprefix(train.REWARD_PARAMETER_PREFIX): value
        for name, value in sampled_values.items()
        if name.startswith(train.REWARD_PARAMETER_PREFIX)
    })
    defaults = PPOTrainingConfig()
    for field_name in (
        "model_dim", "model_layers", "history_length", "transformer_heads",
        "use_state_dependent_exploration", "exploration_noise_resample_steps",
    ):
        assert getattr(training_config, field_name) == getattr(defaults, field_name)
    assert training_config.checkpoint_path == Path(
        "checkpoints/wandb/test-run-id/ppo_cube_stacker"
    )
    assert training_config.metrics_directory == Path(
        "data/wandb/test-run-id"
    )


def test_wandb_trial_applies_configured_model_parameters(monkeypatch) -> None:
    model_values = {
        "model_history_length": 12,
        "model_transformer_embedding_dim": 48,
        "model_transformer_layers": 2,
        "model_transformer_heads": 3,
        "model_transformer_feedforward_dim": 96,
        "model_sde_xyz_log_std_init": -3.7,
        "model_sde_gripper_log_std_init": -2.6,
        "model_target_kl": 0.017,
        "model_batch_size": 512,
    }
    for name, value in model_values.items():
        monkeypatch.setitem(train.WANDB_SWEEP_CONFIG["parameters"], name, {"value": value})
    train._validate_sweep_parameter_names()
    fake_run = FakeWandbRun(model_values)
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: fake_run)
    calls = []
    monkeypatch.setattr(train, "train_ppo", lambda config, wandb_run: calls.append(config))

    train.run_wandb_trial()

    config = calls[0]
    assert config.history_length == 12
    assert config.transformer_embedding_dim == 48
    assert config.transformer_layers == 2
    assert config.transformer_heads == 3
    assert config.transformer_feedforward_dim == 96
    assert config.sde_xyz_log_std_init == pytest.approx(-3.7)
    assert config.sde_gripper_log_std_init == pytest.approx(-2.6)
    assert config.target_kl == pytest.approx(0.017)
    assert config.batch_size == 512
    assert isinstance(config.batch_size, int)


@pytest.mark.parametrize(
    ("arguments", "expected_call"),
    [
        ([], ("normal", None)),
        (["--wandb"], ("wandb", None)),
        (["--wandb", "existing-123"], ("wandb", "existing-123")),
        (
            ["--wandb", "other-team/other-project/existing-123"],
            ("wandb", "other-team/other-project/existing-123"),
        ),
    ],
)
def test_main_routes_between_normal_and_wandb_training(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected_call: tuple[str, str | None],
) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(train, "train_ppo", lambda: calls.append(("normal", None)))
    monkeypatch.setattr(
        train,
        "run_wandb_sweep",
        lambda sweep_id=None, *, pretrained_checkpoint=None: calls.append(("wandb", sweep_id)),
    )

    train.main(arguments)

    assert calls == [expected_call]


@pytest.mark.parametrize("pretrained_arguments", [
    [], ["--pretrained"], ["--pretrained", "checkpoints/pretraining/custom.zip"],
])
def test_main_retrains_with_saved_wandb_reward_config(
    monkeypatch: pytest.MonkeyPatch,
    pretrained_arguments: list[str],
) -> None:
    api_paths: list[str] = []
    source_run = SimpleNamespace(
        config={
            "reward_approach_orange_progress_weight": 1.75,
            "reward_grasp_reward": 8.5,
            "reward_ik_failure_penalty": -1.25,
            "model_dim": 512,
            "model_layers": 4,
            "model_learning_rate": 7e-4,
            "model_history_length": 32,
            "model_transformer_embedding_dim": 96,
            "model_transformer_layers": 4,
            "model_transformer_heads": 3,
            "model_transformer_feedforward_dim": 192,
            "model_sde_xyz_log_std_init": -3.8,
            "model_sde_gripper_log_std_init": -2.7,
            "model_target_kl": 0.0125,
            "model_batch_size": 512,
            "unrelated_wandb_value": 123,
        }
    )
    api = SimpleNamespace(
        run=lambda path: api_paths.append(path) or source_run
    )
    monkeypatch.setattr(train.wandb, "Api", lambda: api)
    training_configs: list[PPOTrainingConfig] = []
    monkeypatch.setattr(
        train,
        "train_ppo",
        lambda config: training_configs.append(config),
    )

    train.main(["--repeat-wandb", "source-run-id", *pretrained_arguments])

    assert api_paths == [
        f"{train.WANDB_ENTITY_NAME}/"
        f"{train.WANDB_PROJECT_NAME}/source-run-id"
    ]
    assert len(training_configs) == 1
    reward_config = training_configs[0].reward_config
    assert reward_config.approach_orange_progress_weight == pytest.approx(
        1.75
    )
    assert reward_config.grasp_reward == pytest.approx(8.5)
    assert reward_config.ik_failure_penalty == pytest.approx(-1.25)
    assert (
        reward_config.grasp_candidate_reward
        == StackRewardConfig().grasp_candidate_reward
    )
    assert training_configs[0].model_dim == 512
    assert training_configs[0].model_layers == 4
    assert training_configs[0].learning_rate == pytest.approx(7e-4)
    assert training_configs[0].history_length == 32
    assert training_configs[0].transformer_embedding_dim == 96
    assert training_configs[0].transformer_layers == 4
    assert training_configs[0].transformer_heads == 3
    assert training_configs[0].transformer_feedforward_dim == 192
    assert training_configs[0].sde_xyz_log_std_init == pytest.approx(-3.8)
    assert training_configs[0].sde_gripper_log_std_init == pytest.approx(-2.7)
    assert training_configs[0].target_kl == pytest.approx(0.0125)
    assert training_configs[0].batch_size == 512
    expected_checkpoint = (
        None if not pretrained_arguments else Path(
            pretrained_arguments[1] if len(pretrained_arguments) == 2
            else "checkpoints/pretraining/default.zip"
        )
    )
    assert training_configs[0].pretrained_checkpoint == expected_checkpoint


@pytest.mark.parametrize(
    "wandb_arguments",
    [
        ["--wandb"],
        ["--wandb", "existing-123"],
        ["--wandb", "other-team/other-project/existing-123"],
    ],
)
def test_training_mode_arguments_are_mutually_exclusive(
    wandb_arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        train.parse_arguments(
            [*wandb_arguments, "--repeat-wandb", "source-run-id"]
        )
    assert error.value.code == 2
    assert "not allowed with argument --wandb" in capsys.readouterr().err


@pytest.mark.parametrize("legacy_schema", ["flat", "history"])
def test_pretraining_initialization_rejects_waypoint_observation_checkpoints(
    legacy_schema: str,
) -> None:
    environment = train.create_validation_environment(
        PPOTrainingConfig(history_length=3)
    )
    try:
        current_space = environment.observation_space
        if legacy_schema == "flat":
            legacy_space = gym.spaces.Box(-np.inf, np.inf, (50,), dtype=np.float32)
        else:
            legacy_space = deepcopy(current_space)
            legacy_space["tokens"] = gym.spaces.Box(
                -np.inf, np.inf, (3, 57), dtype=np.float32
            )
        model = SimpleNamespace(
            observation_space=current_space, action_space=environment.action_space
        )
        pretrained = SimpleNamespace(
            observation_space=legacy_space, action_space=environment.action_space
        )

        with pytest.raises(ValueError, match="observation/action spaces do not match"):
            train.initialize_from_pretraining(model, pretrained)
    finally:
        environment.close()


@pytest.fixture
def pretrained_sweep_architecture(monkeypatch):
    """A validated checkpoint whose architecture differs from scratch defaults."""
    architecture = {
        "history_length": 32,
        "transformer_embedding_dim": 96,
        "transformer_layers": 2,
        "transformer_heads": 3,
        "transformer_feedforward_dim": 192,
        "model_dim": 64,
        "model_layers": 1,
    }
    loaded_configs = []

    def load(config):
        assert config.pretrained_checkpoint is not None
        loaded_configs.append(config)
        return replace(config, **architecture), SimpleNamespace()

    monkeypatch.setattr(train, "load_pretraining_checkpoint", load)
    return architecture, loaded_configs


@pytest.mark.parametrize(("arguments", "sweep_id", "checkpoint"), [
    (["--wandb", "--pretrained"], None, "default.zip"),
    (["--pretrained", "--wandb"], None, "default.zip"),
    (["--wandb", "existing-123", "--pretrained"], "existing-123", "default.zip"),
    (["--wandb", "--pretrained", "checkpoints/pretraining/custom.zip"], None, "custom.zip"),
    (["--pretrained", "checkpoints/pretraining/custom.zip", "--wandb", "team/project/sweep"],
     "team/project/sweep", "custom.zip"),
])
def test_main_combines_wandb_with_default_or_named_pretraining(
    monkeypatch, arguments, sweep_id, checkpoint,
):
    calls = []
    monkeypatch.setattr(train, "run_wandb_sweep", lambda sweep_id=None, *, pretrained_checkpoint=None:
                        calls.append((sweep_id, pretrained_checkpoint)))
    monkeypatch.setattr(train, "train_ppo", lambda *args, **kwargs:
                        pytest.fail("A sweep must route through its agent."))

    train.main(arguments)

    assert calls == [(sweep_id, Path("checkpoints/pretraining") / checkpoint)]


def test_pretrained_sweep_pins_architecture_without_changing_scratch_sweep(
    monkeypatch, pretrained_sweep_architecture,
):
    _, loads = pretrained_sweep_architecture
    original = deepcopy(train.WANDB_SWEEP_CONFIG)
    checkpoint = Path("checkpoints/pretraining/clone.zip")
    sweep_calls, trial_calls = [], []
    monkeypatch.setattr(train.wandb, "sweep", lambda **kwargs:
                        sweep_calls.append(kwargs) or "new-warm-start-sweep")
    monkeypatch.setattr(train, "run_wandb_trial", lambda pretrained_checkpoint=None:
                        trial_calls.append(pretrained_checkpoint))

    def agent(sweep_id, function):
        assert sweep_id == "new-warm-start-sweep"
        function()
        function()

    monkeypatch.setattr(train.wandb, "agent", agent)
    train.run_wandb_sweep(pretrained_checkpoint=checkpoint)

    assert len(loads) == 1
    assert loads[0].pretrained_checkpoint.resolve() == checkpoint.resolve()
    assert trial_calls == [checkpoint.resolve(), checkpoint.resolve()]
    assert train.WANDB_SWEEP_CONFIG == original
    assert sweep_calls[0]["project"] == train.WANDB_PROJECT_NAME
    expected = deepcopy(original)
    expected["parameters"].update({
        "model_history_length": {"value": 32},
        "model_transformer_embedding_dim": {"value": 96},
        "model_transformer_layers": {"value": 2},
        "model_transformer_heads": {"value": 3},
        "model_transformer_feedforward_dim": {"value": 192},
        "model_dim": {"value": 64},
        "model_layers": {"value": 1},
    })
    assert sweep_calls[0]["sweep"] == expected


@pytest.mark.parametrize("sweep_id", ["existing-123", "team/project/existing-123"])
def test_pretrained_agent_for_existing_sweep_forwards_checkpoint_every_trial(
    monkeypatch, pretrained_sweep_architecture, sweep_id,
):
    checkpoint = Path("checkpoints/pretraining/clone.zip")
    calls = []
    monkeypatch.setattr(train.wandb, "sweep", lambda **kwargs:
                        pytest.fail("An existing sweep must not be recreated."))
    monkeypatch.setattr(train, "run_wandb_trial", lambda pretrained_checkpoint=None:
                        calls.append(pretrained_checkpoint))

    def agent(identifier, function):
        expected = (sweep_id if "/" in sweep_id
                    else f"{train.WANDB_ENTITY_NAME}/{train.WANDB_PROJECT_NAME}/{sweep_id}")
        assert identifier == expected
        function()
        function()

    monkeypatch.setattr(train.wandb, "agent", agent)
    train.run_wandb_sweep(sweep_id, pretrained_checkpoint=checkpoint)
    assert calls == [checkpoint.resolve(), checkpoint.resolve()]


def test_pretrained_trials_reload_source_apply_samples_and_save_per_run(
    monkeypatch, pretrained_sweep_architecture,
):
    architecture, loads = pretrained_sweep_architecture
    checkpoint = Path("checkpoints/pretraining/clone.zip")
    init_calls, training_calls = [], []
    samples = {
        "model_learning_rate": 8e-6, "model_batch_size": 512,
        "model_target_kl": 0.007, "model_sde_xyz_log_std_init": -4.0,
        "model_transformer_embedding_dim": 96.0,
        "reward_approach_orange_progress_weight": 0, "reward_grasp_reward": 9.5,
    }

    def initialize(**kwargs):
        init_calls.append(kwargs)
        run = FakeWandbRun(kwargs["config"] | samples)
        run.id = f"trial-{len(init_calls)}"
        return run

    monkeypatch.setattr(train.wandb, "init", initialize)
    monkeypatch.setattr(train, "train_ppo", lambda config, wandb_run:
                        training_calls.append((config, wandb_run)))

    train.run_wandb_trial(checkpoint)
    train.run_wandb_trial(checkpoint)

    assert len(loads) == len(training_calls) == 2
    for index, (config, run) in enumerate(training_calls, start=1):
        assert config.pretrained_checkpoint.resolve() == checkpoint.resolve()
        assert config.learning_rate == pytest.approx(8e-6)
        assert config.batch_size == 512
        assert config.target_kl == pytest.approx(0.007)
        assert config.sde_xyz_log_std_init == pytest.approx(-4.0)
        assert config.reward_config.approach_orange_progress_weight == 0
        assert config.reward_config.grasp_reward == pytest.approx(9.5)
        for name, value in architecture.items():
            assert getattr(config, name) == value
        assert config.recovery_start_probability == PPOTrainingConfig().recovery_start_probability
        assert config.checkpoint_path == Path(f"checkpoints/wandb/trial-{index}/ppo_cube_stacker")
        assert config.metrics_directory == Path(f"data/wandb/trial-{index}")
        assert run.id == f"trial-{index}"
        assert init_calls[index - 1]["config"]["pretrained_checkpoint"] == str(checkpoint.resolve())
        assert init_calls[index - 1]["config"]["model_transformer_heads"] == 3


@pytest.mark.parametrize(("parameter", "sample"), [
    ("model_transformer_embedding_dim", 128),
    ("model_transformer_heads", 4),
    ("model_dim", 128),
    ("model_history_length", 64),
])
def test_existing_sweep_cannot_override_pretrained_architecture(
    monkeypatch, pretrained_sweep_architecture, parameter, sample,
):
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs:
                        FakeWandbRun(kwargs["config"] | {parameter: sample}))
    monkeypatch.setattr(train, "train_ppo", lambda *args, **kwargs:
                        pytest.fail("An incompatible sampled architecture must fail before training."))
    with pytest.raises(ValueError, match="architecture"):
        train.run_wandb_trial(Path("checkpoints/pretraining/clone.zip"))


@pytest.mark.parametrize("sweep_id", [None, "existing-123"])
def test_invalid_pretraining_checkpoint_stops_before_sweep_or_agent(
    monkeypatch, sweep_id,
):
    def load(config):
        raise ValueError("Invalid checkpoint metadata")

    monkeypatch.setattr(train, "load_pretraining_checkpoint", load)
    monkeypatch.setattr(train.wandb, "sweep", lambda **kwargs:
                        pytest.fail("Invalid checkpoints must not create sweeps."))
    monkeypatch.setattr(train.wandb, "agent", lambda *args, **kwargs:
                        pytest.fail("Invalid checkpoints must not start agents."))
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        train.run_wandb_sweep(sweep_id, pretrained_checkpoint=Path("missing.zip"))
