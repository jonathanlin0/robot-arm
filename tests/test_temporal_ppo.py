"""Exercise temporal context through real MuJoCo rollouts and PPO updates."""

from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from action_observation_history import ActionObservationHistoryWrapper
from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy
from cartesian_actions import CartesianActionConfig
from gym_environment import CubeStackGymEnvironment
from temporal_features import ActionObservationTransformer


class RecordedCubeEnvironment(CubeStackGymEnvironment):
    """Record the observations and commands at the actual simulator interface."""

    def __init__(self, horizon: int) -> None:
        # Audit PPO/history mechanics independently of home-pose orientation limits.
        super().__init__(
            maximum_episode_steps=horizon,
            action_config=CartesianActionConfig(require_downward=False),
        )
        self.episode_tokens: list[np.ndarray] = []
        self.terminal_tokens: list[np.ndarray] = []
        self.reset_count = 0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.episode_tokens = [
            np.concatenate(
                (
                    observation,
                    np.zeros(4),
                    self.action_adapter.current_target_gripper_position,
                )
            )
        ]
        self.reset_count += 1
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_tokens.append(
            np.concatenate(
                (
                    observation,
                    action.copy(),
                    self.action_adapter.current_target_gripper_position,
                )
            )
        )
        if terminated or truncated:
            self.terminal_tokens = list(self.episode_tokens)
        return observation, reward, terminated, truncated, info


def vector_environment() -> DummyVecEnv:
    return DummyVecEnv(
        [
            lambda horizon=horizon: ActionObservationHistoryWrapper(
                RecordedCubeEnvironment(horizon), history_length=4
            )
            for horizon in (3, 5)
        ]
    )


def assert_matches_simulator_history(observation: dict, recorded: list) -> None:
    expected = np.asarray(recorded[-4:], dtype=np.float32)
    count = len(expected)
    np.testing.assert_array_equal(observation["tokens"][:count], expected)
    np.testing.assert_array_equal(observation["tokens"][count:], 0)
    np.testing.assert_array_equal(observation["valid"], np.arange(4) < count)
    expected_start = np.zeros(4)
    expected_start[0] = float(len(recorded) <= 4)
    np.testing.assert_array_equal(observation["episode_start"], expected_start)


class AuditTemporalRollouts(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rollouts_checked = 0
        self.terminals_checked = 0
        self.clipped_components = 0
        self.history_snapshots: list[dict] = []
        self.initial_history_reference = None
        self.initial_history_copy = None

    def _on_rollout_start(self) -> None:
        self.history_snapshots = []
        for index, environment in enumerate(self.training_env.envs):
            observation = {
                key: value[index] for key, value in self.model._last_obs.items()
            }
            assert_matches_simulator_history(
                observation, environment.unwrapped.episode_tokens
            )

    def _on_step(self) -> bool:
        self.history_snapshots.append(
            {key: value.copy() for key, value in self.model._last_obs.items()}
        )
        if self.initial_history_reference is None:
            self.initial_history_reference = self.model._last_obs
            self.initial_history_copy = self.history_snapshots[-1]
        self.clipped_components += np.count_nonzero(
            self.locals["actions"] != self.locals["clipped_actions"]
        )
        for index, environment in enumerate(self.training_env.envs):
            simulator = environment.unwrapped
            observation = {
                key: value[index] for key, value in self.locals["new_obs"].items()
            }
            assert_matches_simulator_history(observation, simulator.episode_tokens)
            if self.locals["dones"][index]:
                terminal = self.locals["infos"][index]["terminal_observation"]
                assert_matches_simulator_history(terminal, simulator.terminal_tokens)
                np.testing.assert_array_equal(
                    terminal["tokens"][int(terminal["valid"].sum()) - 1, -7:-3],
                    self.locals["clipped_actions"][index],
                )
                self.terminals_checked += 1
            else:
                np.testing.assert_array_equal(
                    observation["tokens"][int(observation["valid"].sum()) - 1, -7:-3],
                    self.locals["clipped_actions"][index],
                )
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        assert isinstance(buffer, DictRolloutBuffer)
        for key, values in buffer.observations.items():
            np.testing.assert_array_equal(
                values, np.stack([snapshot[key] for snapshot in self.history_snapshots])
            )
            np.testing.assert_array_equal(
                self.initial_history_reference[key], self.initial_history_copy[key]
            )

        # This happens before PPO updates the weights. Shuffled minibatches must
        # reproduce collection-time likelihoods even after switching train mode.
        self.model.policy.set_training_mode(True)
        try:
            with torch.no_grad():
                for batch in buffer.get(batch_size=3):
                    values, log_probabilities, entropy = self.model.policy.evaluate_actions(
                        batch.observations, batch.actions
                    )
                    torch.testing.assert_close(
                        log_probabilities, batch.old_log_prob, rtol=1e-5, atol=1e-5
                    )
                    torch.testing.assert_close(
                        values.flatten(), batch.old_values, rtol=1e-5, atol=1e-5
                    )
                    assert torch.isfinite(entropy).all()
        finally:
            self.model.policy.set_training_mode(False)
        self.rollouts_checked += 1


@pytest.mark.parametrize("use_sde", [False, True])
def test_temporal_ppo_rollouts_updates_and_checkpoint(
    tmp_path: Path, use_sde: bool
) -> None:
    environment = vector_environment()
    fresh_environment = vector_environment()
    try:
        model = PPO(
            TanhBoundedMeanActorCriticPolicy,
            environment,
            n_steps=4,
            batch_size=4,
            n_epochs=2,
            learning_rate=1e-3,
            use_sde=use_sde,
            sde_sample_freq=2,
            seed=17,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": ActionObservationTransformer,
                "features_extractor_kwargs": {
                    "embedding_dim": 16,
                    "layer_count": 1,
                    "head_count": 2,
                    "feedforward_dim": 32,
                },
                "share_features_extractor": True,
                "activation_fn": torch.nn.ReLU,
                "net_arch": {"pi": [16], "vf": [16]},
                "log_std_init": 1.0,
                "use_expln": True,
                "squash_output": False,
            },
        )
        features = model.policy.features_extractor
        assert environment.observation_space["tokens"].shape == (4, 56)
        assert features.token_projection.in_features == 56
        initial_projection = features.token_projection.weight.detach().clone()
        initial_exploration = (
            {
                name: parameter.detach().clone()
                for name, parameter in model.policy.exploration_mlp.named_parameters()
            }
            if use_sde else None
        )
        assert model.policy.pi_features_extractor is model.policy.vf_features_extractor
        callback = AuditTemporalRollouts()

        model.learn(total_timesteps=16, callback=callback)

        assert callback.rollouts_checked == 2
        assert callback.terminals_checked >= 3
        assert callback.clipped_components > 0
        assert all(env.unwrapped.reset_count >= 2 for env in environment.envs)
        assert not torch.equal(initial_projection, features.token_projection.weight)
        if use_sde:
            assert any(
                not torch.equal(parameter, initial_exploration[name])
                for name, parameter in model.policy.exploration_mlp.named_parameters()
            )
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())

        checkpoint = tmp_path / "temporal_policy"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=fresh_environment, device="cpu")
        assert loaded.policy.activation_fn is torch.nn.ReLU
        loaded_features = loaded.policy.features_extractor
        assert isinstance(loaded_features, ActionObservationTransformer)
        assert loaded_features.history_length == 4
        assert loaded_features.features_dim == 16
        assert len(loaded_features.encoder_layers) == 1
        assert loaded_features.encoder_layers[0].self_attn.num_heads == 2
        assert loaded_features.encoder_layers[0].linear1.out_features == 32
        assert loaded.policy.pi_features_extractor is loaded.policy.vf_features_extractor
        if use_sde:
            for name, parameter in model.policy.exploration_mlp.named_parameters():
                torch.testing.assert_close(
                    loaded.policy.exploration_mlp.state_dict()[name],
                    parameter,
                    rtol=0,
                    atol=0,
                )
        observations = fresh_environment.reset()
        for _ in range(6):
            expected_action, _ = model.predict(observations, deterministic=True)
            loaded_action, _ = loaded.predict(observations, deterministic=True)
            np.testing.assert_allclose(loaded_action, expected_action, rtol=1e-6, atol=1e-6)
            observations, _, _, _ = fresh_environment.step(loaded_action)
    finally:
        environment.close()
        fresh_environment.close()
