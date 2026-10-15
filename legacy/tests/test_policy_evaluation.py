"""Periodic serial evaluation must not become part of PPO's training data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

import train
from cartesian_actions import CartesianActionConfig
from train import PPOTrainingConfig, RolloutMetricsCallback, evaluate_no_var_success_rate


class EvaluationEnvironment:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.seeds = []
        self.active = False
        self.closed = False

    def reset(self, *, seed):
        assert not self.active, "Evaluation episodes must finish before the next reset."
        self.active = True
        self.seeds.append(seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        assert self.active
        if self.fail:
            raise RuntimeError("evaluation failure")
        self.steps += 1
        done = self.steps == 2
        self.active = not done
        success = self.seeds[-1] % 4 == 0
        return (
            np.zeros(3, dtype=np.float32), 0.0,
            done and success, done and not success,
            # Nonterminal flags must not be counted as episode successes.
            {"is_success": success if done else True},
        )

    def close(self):
        self.closed = True


class EvaluationPolicy(torch.nn.Module):
    def set_training_mode(self, mode):
        self.train(mode)


def make_model():
    policy = EvaluationPolicy()
    observations = []

    def predict(observation, *, deterministic):
        assert deterministic is True
        assert torch.get_num_threads() == 1
        assert not policy.training
        assert not torch.is_grad_enabled()
        observations.append(observation)
        torch.rand(1)  # Even incidental evaluation RNG use must be isolated.
        return np.zeros(4), None

    return SimpleNamespace(
        policy=policy, predict=predict, observations=observations,
        num_timesteps=12345, ep_success_buffer=[False],
    )


def test_evaluation_runs_100_complete_episodes_in_series_on_fixed_seeds(monkeypatch):
    environments = []
    configs = []

    def factory(config):
        configs.append(config)
        environment = EvaluationEnvironment()
        environments.append(environment)
        return environment

    monkeypatch.setattr(train, "create_validation_environment", factory)
    model = make_model()
    config = PPOTrainingConfig(maximum_episode_steps=2, recovery_start_probability=0.25)
    original_rng = torch.get_rng_state().clone()
    original_threads = torch.get_num_threads()

    for _ in range(2):
        assert evaluate_no_var_success_rate(model, config) == 0.25
        assert model.policy.training
        assert torch.get_num_threads() == original_threads
        torch.testing.assert_close(torch.get_rng_state(), original_rng)

    assert len(environments) == 2  # One ordinary environment per evaluation.
    for environment in environments:
        assert environment.seeds == list(range(20_000, 20_100))
        assert environment.closed
        assert not environment.active
    assert len(model.observations) == 400
    assert model.num_timesteps == 12345
    assert model.ep_success_buffer == [False]
    assert config.recovery_start_probability == 0.25
    assert all(c.recovery_start_probability == 0.0 for c in configs)
    assert all(c.history_length == config.history_length for c in configs)
    assert all(c.start_at_orange_waypoint for c in configs)
    assert all(c.reward_config == config.reward_config for c in configs)


@pytest.mark.parametrize("training_mode", [False, True])
def test_evaluation_failure_closes_environment_and_restores_training_state(
    monkeypatch, training_mode,
):
    environment = EvaluationEnvironment(fail=True)
    monkeypatch.setattr(train, "create_validation_environment", lambda config: environment)
    model = make_model()
    model.policy.set_training_mode(training_mode)
    original_rng = torch.get_rng_state().clone()
    original_threads = torch.get_num_threads()

    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate_no_var_success_rate(model, PPOTrainingConfig())

    assert environment.closed
    assert model.policy.training is training_mode
    assert torch.get_num_threads() == original_threads
    torch.testing.assert_close(torch.get_rng_state(), original_rng)


def test_callback_evaluates_first_and_every_100_rollouts_at_the_same_wandb_step(
    tmp_path, monkeypatch,
):
    calls = []
    logged = []
    config = PPOTrainingConfig(evaluation_interval_rollouts=100)

    def evaluate(model, received_config):
        assert received_config is config
        calls.append(callback.rollout_count)
        return len(calls) / 4

    monkeypatch.setattr(train, "evaluate_no_var_success_rate", evaluate)
    callback = RolloutMetricsCallback(
        tmp_path,
        wandb_run=SimpleNamespace(log=lambda values, step: logged.append((values, step))),
        evaluation_config=config,
    )
    callback.model = SimpleNamespace(
        ep_info_buffer=[{"l": 400, "r": 2.0, "orange_pregrasp_waypoint_reached": True}],
        ep_success_buffer=[False],
    )
    callback._on_training_start()
    for rollout in range(1, 201):
        if rollout == 150:
            # train_ppo may call learn() again to extend a promising run.
            callback._on_training_start()
        callback.num_timesteps = rollout * 4096
        callback._on_rollout_end()

    assert calls == [1, 100, 200]
    assert len(logged) == 200
    evaluation_logs = [(values, step) for values, step in logged if "no_var_success_rate" in values]
    assert [step for _, step in evaluation_logs] == [1, 100, 200]
    assert [values["no_var_success_rate"] for values, _ in evaluation_logs] == [0.25, 0.5, 0.75]
    assert all(values["rollout"] == step for values, step in logged)
    assert all(values["success_rate"] == 0.0 for values, _ in logged)
    values = np.loadtxt(tmp_path / "no_var_success_rate.txt")
    assert values.shape == (200,)
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(values)), [0, 99, 199])
    np.testing.assert_array_equal(values[[0, 99, 199]], [0.25, 0.5, 0.75])


@pytest.mark.parametrize("overrides", [
    {"evaluation_interval_rollouts": 0},
    {"evaluation_interval_rollouts": 1.5},
    {"evaluation_episodes": 0},
    {"evaluation_episodes": True},
    {"evaluation_seed": -1},
    {"evaluation_seed": 1.5},
])
def test_evaluation_settings_reject_invalid_values(overrides):
    with pytest.raises(ValueError):
        PPOTrainingConfig(**overrides)


def test_real_transformer_ppo_can_train_through_serial_evaluation(tmp_path, monkeypatch):
    """Exercise episode resets and a gradient update after evaluation."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    real_environment_factory = train.CubeStackGymEnvironment

    def best_effort_environment(**kwargs):
        return real_environment_factory(
            **kwargs, action_config=CartesianActionConfig(require_downward=False)
        )

    monkeypatch.setattr(train, "CubeStackGymEnvironment", best_effort_environment)
    monkeypatch.setattr(train, "SubprocVecEnv", lambda env_fns, **kwargs: DummyVecEnv(env_fns))
    config = PPOTrainingConfig(
        environment_count=1, maximum_episode_steps=2, total_timesteps=8,
        rollout_steps=4, batch_size=4, training_epochs=1,
        history_length=4, transformer_embedding_dim=16, transformer_layers=1,
        transformer_heads=2, transformer_feedforward_dim=32, model_dim=16,
        model_layers=1, evaluation_interval_rollouts=1, evaluation_episodes=2,
        evaluation_seed=20_000, checkpoint_path=tmp_path / "policy",
        metrics_directory=tmp_path / "metrics", recovery_start_probability=1.0,
    )
    model = train.train_ppo(config)

    assert model.num_timesteps == 8
    assert model._n_updates == 2
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "metrics" / "no_var_success_rate.txt"), [0.0, 0.0],
    )
    loaded = PPO.load(tmp_path / "policy", device="cpu")
    assert loaded.pickup_training_config["evaluation_episodes"] == 2
    assert loaded.pickup_training_config["recovery_start_probability"] == 1.0
