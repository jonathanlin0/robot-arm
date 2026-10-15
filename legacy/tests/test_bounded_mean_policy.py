import math
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor

from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy
from temporal_features import ActionObservationTransformer


def make_policy(**kwargs: object) -> TanhBoundedMeanActorCriticPolicy:
    return TanhBoundedMeanActorCriticPolicy(
        gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [8], "vf": [8]},
        **kwargs,
    )


@pytest.mark.parametrize("use_sde", [False, True])
def test_policy_tanh_bounds_only_the_gaussian_mean(use_sde: bool) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )
    policy = TanhBoundedMeanActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [8], "vf": [8]},
        use_sde=use_sde,
        squash_output=False,
    )

    intended_unbounded_means = torch.tensor(
        [10.0, -10.0, 2.0, -2.0],
        dtype=torch.float32,
    )
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(intended_unbounded_means)

    observation = torch.zeros((1, 3), dtype=torch.float32)
    distribution = policy.get_distribution(observation)
    expected_means = torch.tanh(intended_unbounded_means).unsqueeze(0)

    assert torch.allclose(distribution.distribution.mean, expected_means)
    assert torch.all(distribution.distribution.mean <= 1.0)
    assert torch.all(distribution.distribution.mean >= -1.0)
    entropy = distribution.entropy()
    assert entropy is not None
    assert torch.all(torch.isfinite(entropy))


def test_ppo_checkpoint_preserves_bounded_mean_policy(
    tmp_path: Path,
) -> None:
    environment = gym.make("Pendulum-v1")
    checkpoint_path = tmp_path / "bounded_mean_policy"
    try:
        model = PPO(
            policy=TanhBoundedMeanActorCriticPolicy,
            env=environment,
            n_steps=2,
            batch_size=2,
            use_sde=True,
            policy_kwargs={"squash_output": False},
        )
        model.save(checkpoint_path)
    finally:
        environment.close()

    loaded_model = PPO.load(checkpoint_path, device="cpu")

    assert isinstance(
        loaded_model.policy,
        TanhBoundedMeanActorCriticPolicy,
    )


@pytest.mark.parametrize("use_expln", [False, True])
def test_per_action_sde_initialization_controls_effective_action_sd(
    use_expln: bool,
) -> None:
    coefficient_stds = torch.tensor([0.2, 0.2, 0.2, 0.5]) / 7.1
    policy = make_policy(
        use_sde=True,
        use_expln=use_expln,
        sde_log_std_init=coefficient_stds.log().tolist(),
    )
    latent = torch.full((1, 8), 7.1 / math.sqrt(8))

    distribution = policy._get_action_dist_from_latent(latent, latent)

    bounded_feature_norm = math.sqrt(8) * math.tanh(7.1 / math.sqrt(8))
    expected_noise_sd = coefficient_stds * bounded_feature_norm
    expected_variance = expected_noise_sd.unsqueeze(0).square()
    expected_sd = (expected_variance + distribution.epsilon).sqrt()
    torch.testing.assert_close(distribution.distribution.stddev, expected_sd)
    # The sampler must use the new coefficients immediately, before PPO's
    # first rollout calls reset_noise().
    torch.testing.assert_close(
        distribution.weights_dist.stddev,
        coefficient_stds.expand_as(policy.log_std),
    )

    torch.manual_seed(123)
    sample_count = 8192
    policy.reset_noise(sample_count)
    distribution = policy._get_action_dist_from_latent(
        latent.expand(sample_count, -1), latent.expand(sample_count, -1)
    )
    noise = distribution.sample() - distribution.distribution.mean
    torch.testing.assert_close(
        noise.std(dim=0),
        expected_noise_sd,
        rtol=0.04,
        atol=0.0,
    )


def test_sde_bounds_noise_features_without_changing_mean_or_likelihood() -> None:
    policy = make_policy(
        use_sde=True,
        activation_fn=torch.nn.ReLU,
        log_std_init=math.log(0.1),
    )
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.weight[:, :4].copy_(
            torch.diag(torch.tensor([0.01, 0.02, 0.03, 0.04]))
        )
        policy.action_net.bias.zero_()

    # Large ReLU features should stop amplifying noise while the mean still
    # responds to their original magnitudes.
    latent = torch.tensor([[10.0] * 8, [100.0] * 8])
    distribution = policy._get_action_dist_from_latent(latent, latent)
    expected_mean = torch.tanh(
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]])
    )
    expected_sd = torch.full_like(
        expected_mean, math.sqrt(8 * 0.1**2 + distribution.epsilon)
    )
    torch.testing.assert_close(distribution.distribution.mean, expected_mean)
    torch.testing.assert_close(distribution.distribution.stddev, expected_sd)

    samples = distribution.sample()
    expected_noise = torch.ones_like(latent) @ distribution.exploration_mat
    torch.testing.assert_close(samples - expected_mean, expected_noise)
    expected_gaussian = torch.distributions.Normal(expected_mean, expected_sd)
    torch.testing.assert_close(
        distribution.log_prob(samples), expected_gaussian.log_prob(samples).sum(dim=1)
    )
    torch.testing.assert_close(
        distribution.entropy(), expected_gaussian.entropy().sum(dim=1)
    )


def test_custom_sde_coefficients_remain_trainable_and_survive_policy_reload(
    tmp_path: Path,
) -> None:
    initialization = (math.log(0.2 / 7.1),) * 3 + (math.log(0.5 / 7.1),)
    policy = make_policy(use_sde=True, sde_log_std_init=initialization)
    initial_coefficients = policy.log_std.detach().clone()
    latent = torch.full((2, 8), 0.5)
    initial_sd = (
        policy._get_action_dist_from_latent(latent, latent)
        .distribution.stddev.detach().clone()
    )

    policy.optimizer.zero_grad()
    loss = -policy._get_action_dist_from_latent(latent, latent).entropy().mean()
    loss.backward()
    assert policy.log_std.requires_grad
    assert torch.all(policy.log_std.grad != 0)
    policy.optimizer.step()

    learned_coefficients = policy.log_std.detach().clone()
    assert torch.all(learned_coefficients > initial_coefficients)
    learned_sd = (
        policy._get_action_dist_from_latent(latent, latent)
        .distribution.stddev.detach().clone()
    )
    assert torch.all(learned_sd > initial_sd)
    checkpoint_path = tmp_path / "per_action_sde_policy.pt"
    policy.save(checkpoint_path)

    loaded = TanhBoundedMeanActorCriticPolicy.load(checkpoint_path, device="cpu")

    assert loaded.sde_log_std_init == initialization
    torch.testing.assert_close(loaded.log_std, learned_coefficients)
    torch.testing.assert_close(
        loaded._get_action_dist_from_latent(latent, latent).distribution.stddev,
        learned_sd,
    )


def test_ppo_reload_preserves_learned_custom_sde_coefficients(
    tmp_path: Path,
) -> None:
    environment = gym.make("Pendulum-v1")
    initialization = (math.log(0.2 / 7.1),)
    try:
        model = PPO(
            policy=TanhBoundedMeanActorCriticPolicy,
            env=environment,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            use_sde=True,
            seed=123,
            policy_kwargs={"sde_log_std_init": initialization},
        )
        initial_coefficients = model.policy.log_std.detach().clone()
        model.learn(total_timesteps=16)
        learned_coefficients = model.policy.log_std.detach().clone()
        assert not torch.equal(initial_coefficients, learned_coefficients)
        checkpoint_path = tmp_path / "per_action_sde_ppo"
        model.save(checkpoint_path)
    finally:
        environment.close()

    loaded = PPO.load(checkpoint_path, device="cpu")

    assert loaded.policy.sde_log_std_init == initialization
    torch.testing.assert_close(loaded.policy.log_std, learned_coefficients)
    torch.testing.assert_close(
        loaded.policy.action_dist.weights_dist.stddev,
        learned_coefficients.exp(),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_sde": False, "sde_log_std_init": (-3.0,) * 4}, "use_sde=True"),
        (
            {"use_sde": True, "full_std": False, "sde_log_std_init": (-3.0,) * 4},
            "full_std=True",
        ),
        ({"use_sde": True, "sde_log_std_init": (-3.0,) * 3}, "action dimension"),
        ({"use_sde": True, "sde_log_std_init": ()}, "action dimension"),
        ({"use_sde": True, "sde_log_std_init": (math.nan,) * 4}, "finite"),
        ({"use_sde": True, "sde_log_std_init": (math.inf,) * 4}, "finite"),
        ({"use_sde": True, "sde_log_std_init": -3.0}, "sequence"),
    ],
)
def test_custom_sde_initialization_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_policy(**kwargs)


@pytest.mark.parametrize("use_sde", [False, True])
def test_without_custom_sde_initialization_preserves_global_log_std(
    use_sde: bool,
) -> None:
    policy = make_policy(use_sde=use_sde, log_std_init=-1.5)
    torch.testing.assert_close(policy.log_std, torch.full_like(policy.log_std, -1.5))


def make_transformer_policy(
    *, share_features_extractor: bool = True,
) -> tuple[TanhBoundedMeanActorCriticPolicy, dict[str, torch.Tensor]]:
    """Use the production transformer with a small, reproducible history batch."""
    torch.manual_seed(123)
    observation_space = gym.spaces.Dict({
        "tokens": gym.spaces.Box(-np.inf, np.inf, shape=(4, 7), dtype=np.float32),
        "valid": gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        "episode_start": gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
    })
    policy = TanhBoundedMeanActorCriticPolicy(
        observation_space,
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [8], "vf": [8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=ActionObservationTransformer,
        features_extractor_kwargs={
            "embedding_dim": 8,
            "layer_count": 1,
            "head_count": 2,
            "feedforward_dim": 16,
        },
        share_features_extractor=share_features_extractor,
        use_sde=True,
        log_std_init=-2.0,
    )
    observations = {
        "tokens": torch.randn(4, 4, 7),
        "valid": torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]),
        "episode_start": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
    }
    return policy, observations


def parameter_copies(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in module.parameters()
    )


def parameters_changed(
    module: torch.nn.Module, before: list[torch.Tensor],
) -> bool:
    return any(
        not torch.equal(parameter.detach(), previous)
        for parameter, previous in zip(module.parameters(), before, strict=True)
    )


@pytest.mark.parametrize("entrypoint", ["forward", "evaluate_actions", "predict_values"])
def test_critic_update_preserves_shared_transformer_and_actor(
    entrypoint: str,
) -> None:
    policy, observations = make_transformer_policy()
    protected_modules = (
        policy.features_extractor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
        policy.exploration_mlp,
    )
    protected_parameters = [parameter_copies(module) for module in protected_modules]
    critic_modules = (policy.mlp_extractor.value_net, policy.value_net)
    critic_parameters = [parameter_copies(module) for module in critic_modules]
    initial_log_std = policy.log_std.detach().clone()
    with torch.no_grad():
        initial_mean = policy.get_distribution(observations).distribution.mean.clone()

    policy.optimizer.zero_grad(set_to_none=True)
    if entrypoint == "forward":
        _, values, _ = policy(observations, deterministic=True)
    elif entrypoint == "evaluate_actions":
        values, _, _ = policy.evaluate_actions(observations, torch.zeros(4, 4))
    else:
        values = policy.predict_values(observations)
    targets = values.detach() + 1.0
    torch.nn.functional.mse_loss(values, targets).backward()

    for module in protected_modules:
        assert not has_nonzero_gradient(module)
    assert policy.log_std.grad is None
    for module in critic_modules:
        assert has_nonzero_gradient(module)
    policy.optimizer.step()

    for module, before in zip(protected_modules, protected_parameters, strict=True):
        assert not parameters_changed(module, before)
    for module, before in zip(critic_modules, critic_parameters, strict=True):
        assert parameters_changed(module, before)
    torch.testing.assert_close(policy.log_std, initial_log_std, rtol=0, atol=0)
    with torch.no_grad():
        final_mean = policy.get_distribution(observations).distribution.mean
    torch.testing.assert_close(final_mean, initial_mean, rtol=0, atol=0)


def test_actor_likelihood_still_updates_shared_transformer_actor_and_sde() -> None:
    policy, observations = make_transformer_policy()
    actor_modules = (
        policy.features_extractor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
        policy.exploration_mlp,
    )
    actor_parameters = [parameter_copies(module) for module in actor_modules]
    initial_log_std = policy.log_std.detach().clone()
    policy.optimizer.zero_grad(set_to_none=True)
    actions = torch.tensor([[0.4, -0.2, 0.7, -0.6]] * 4)
    _, log_probabilities, _ = policy.evaluate_actions(observations, actions)
    (-log_probabilities.mean()).backward()

    for module in actor_modules:
        assert has_nonzero_gradient(module)
    assert not has_nonzero_gradient(policy.mlp_extractor.value_net)
    assert not has_nonzero_gradient(policy.value_net)
    assert policy.log_std.requires_grad
    assert policy.log_std.grad is not None
    assert torch.count_nonzero(policy.log_std.grad) > 0
    policy.optimizer.step()

    for module, before in zip(actor_modules, actor_parameters, strict=True):
        assert parameters_changed(module, before)
    assert not torch.equal(policy.log_std.detach(), initial_log_std)


def test_exploration_entropy_trains_only_noise_features_and_coefficients() -> None:
    policy, observations = make_transformer_policy()
    exploration_before = parameter_copies(policy.exploration_mlp)
    coefficients_before = policy.log_std.detach().clone()
    protected_modules = (
        policy.features_extractor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
        policy.mlp_extractor.value_net,
        policy.value_net,
    )
    protected_before = [parameter_copies(module) for module in protected_modules]

    policy.optimizer.zero_grad(set_to_none=True)
    entropy = policy.get_distribution(observations).entropy()
    (-entropy.mean()).backward()

    assert policy.action_dist.learn_features
    assert has_nonzero_gradient(policy.exploration_mlp)
    assert policy.log_std.grad is not None
    assert torch.count_nonzero(policy.log_std.grad) > 0
    for module in protected_modules:
        assert not has_nonzero_gradient(module)
    policy.optimizer.step()

    assert parameters_changed(policy.exploration_mlp, exploration_before)
    assert not torch.equal(policy.log_std.detach(), coefficients_before)
    for module, before in zip(protected_modules, protected_before, strict=True):
        assert not parameters_changed(module, before)


def test_mean_loss_does_not_update_exploration_branch() -> None:
    policy, observations = make_transformer_policy()
    exploration_before = parameter_copies(policy.exploration_mlp)
    coefficients_before = policy.log_std.detach().clone()
    actor_modules = (
        policy.features_extractor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
    )
    actor_before = [parameter_copies(module) for module in actor_modules]

    policy.optimizer.zero_grad(set_to_none=True)
    mean = policy.get_distribution(observations).distribution.mean
    torch.nn.functional.mse_loss(mean, torch.full_like(mean, 0.5)).backward()

    assert all(parameter.grad is None for parameter in policy.exploration_mlp.parameters())
    assert policy.log_std.grad is None
    for module in actor_modules:
        assert has_nonzero_gradient(module)
    policy.optimizer.step()

    assert not parameters_changed(policy.exploration_mlp, exploration_before)
    torch.testing.assert_close(policy.log_std, coefficients_before, rtol=0, atol=0)
    for module, before in zip(actor_modules, actor_before, strict=True):
        assert parameters_changed(module, before)


def test_actor_and_exploration_mlp_outputs_are_independent() -> None:
    policy, observations = make_transformer_policy()
    with torch.no_grad():
        initial_distribution = policy.get_distribution(observations).distribution
        initial_mean = initial_distribution.mean.clone()
        initial_sd = initial_distribution.stddev.clone()

        # Alter the actor features without changing the shared transformer.
        for parameter in policy.mlp_extractor.policy_net.parameters():
            parameter.add_(0.5)
        actor_distribution = policy.get_distribution(observations).distribution
        torch.testing.assert_close(actor_distribution.stddev, initial_sd, rtol=0, atol=0)
        assert not torch.equal(actor_distribution.mean, initial_mean)
        actor_mean = actor_distribution.mean.clone()

        for parameter in policy.exploration_mlp.parameters():
            parameter.zero_()
        exploration_distribution = policy.get_distribution(observations).distribution
        torch.testing.assert_close(exploration_distribution.mean, actor_mean, rtol=0, atol=0)
        assert not torch.equal(exploration_distribution.stddev, initial_sd)


@pytest.mark.parametrize("share_features_extractor", [True, False])
def test_exploration_distribution_matches_all_policy_entrypoints(
    share_features_extractor: bool,
) -> None:
    policy, observations = make_transformer_policy(
        share_features_extractor=share_features_extractor
    )
    policy.set_training_mode(False)
    policy.reset_noise(len(observations["tokens"]))
    with torch.no_grad():
        actions, values, log_probabilities = policy(observations)
        evaluated_values, evaluated_log_probabilities, entropy = policy.evaluate_actions(
            observations, actions
        )
        distribution = policy.get_distribution(observations)
        torch.testing.assert_close(distribution.sample(), actions)
        torch.testing.assert_close(distribution.log_prob(actions), log_probabilities)
        torch.testing.assert_close(distribution.entropy(), entropy)
        deterministic_actions, _, _ = policy(observations, deterministic=True)
        torch.testing.assert_close(deterministic_actions, distribution.distribution.mean)
        torch.testing.assert_close(evaluated_values, values)
        torch.testing.assert_close(evaluated_log_probabilities, log_probabilities)


def test_exploration_mlp_is_registered_and_uses_actor_shape_without_shared_weights() -> None:
    policy, _ = make_transformer_policy()
    actor = policy.mlp_extractor.policy_net
    exploration = policy.exploration_mlp
    assert [(type(layer)) for layer in exploration] == [(type(layer)) for layer in actor]
    assert [parameter.shape for parameter in exploration.parameters()] == [
        parameter.shape for parameter in actor.parameters()
    ]
    actor_parameters = {id(parameter) for parameter in actor.parameters()}
    optimizer_parameters = {
        id(parameter)
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter in exploration.parameters():
        assert parameter.requires_grad
        assert id(parameter) not in actor_parameters
        assert id(parameter) in optimizer_parameters


def test_non_sde_policy_does_not_construct_exploration_mlp() -> None:
    policy = make_policy(use_sde=False)
    assert policy.exploration_mlp is None
    assert not any(key.startswith("exploration_mlp.") for key in policy.state_dict())


def test_separate_critic_transformer_still_receives_value_gradients() -> None:
    policy, observations = make_transformer_policy(share_features_extractor=False)
    actor_before = parameter_copies(policy.pi_features_extractor)
    critic_before = parameter_copies(policy.vf_features_extractor)
    assert policy.pi_features_extractor is not policy.vf_features_extractor
    policy.optimizer.zero_grad(set_to_none=True)
    values, _, _ = policy.evaluate_actions(observations, torch.zeros(4, 4))
    torch.nn.functional.mse_loss(values, values.detach() + 1.0).backward()

    assert has_nonzero_gradient(policy.vf_features_extractor)
    assert not has_nonzero_gradient(policy.pi_features_extractor)
    policy.optimizer.step()
    assert parameters_changed(policy.vf_features_extractor, critic_before)
    assert not parameters_changed(policy.pi_features_extractor, actor_before)


def test_critic_gradient_gate_preserves_old_mlp_weights_and_forward_values() -> None:
    policy, observations = make_transformer_policy()
    original_extractor = MlpExtractor(
        policy.features_dim, policy.net_arch, policy.activation_fn, device="cpu"
    )
    # A pre-change checkpoint uses stock SB3 MLP keys and tensor shapes.
    policy.mlp_extractor.load_state_dict(original_extractor.state_dict(), strict=True)
    with torch.no_grad():
        features = policy.extract_features(observations)
        expected_latent_actor, expected_latent_critic = original_extractor(features)
        actor, critic = policy.mlp_extractor(features)
        expected_values = policy.value_net(expected_latent_critic)
        actions, forward_values, forward_log_probabilities = policy(
            observations, deterministic=True
        )
        evaluated_values, evaluated_log_probabilities, _ = policy.evaluate_actions(
            observations, actions
        )
        predicted_values = policy.predict_values(observations)

    torch.testing.assert_close(actor, expected_latent_actor, rtol=0, atol=0)
    torch.testing.assert_close(critic, expected_latent_critic, rtol=0, atol=0)
    for values in (forward_values, evaluated_values, predicted_values):
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(
        forward_log_probabilities, evaluated_log_probabilities, rtol=0, atol=0
    )


def test_policy_reload_preserves_shared_critic_gradient_gate() -> None:
    policy, observations = make_transformer_policy()
    with torch.no_grad():
        expected_actions, expected_values, _ = policy(observations, deterministic=True)
        expected_sd = policy.get_distribution(observations).distribution.stddev.clone()
    expected_exploration = parameter_copies(policy.exploration_mlp)
    with TemporaryDirectory(
        prefix=".test-critic-gradients-", dir=Path(__file__).resolve().parents[1]
    ) as directory:
        path = Path(directory) / "policy.pt"
        policy.save(path)
        loaded = TanhBoundedMeanActorCriticPolicy.load(path, device="cpu")

    actions, values, _ = loaded(observations, deterministic=True)
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.get_distribution(observations).distribution.stddev,
        expected_sd,
        rtol=0,
        atol=0,
    )
    assert not parameters_changed(loaded.exploration_mlp, expected_exploration)
    loaded.optimizer.zero_grad(set_to_none=True)
    values.sum().backward()
    assert not has_nonzero_gradient(loaded.features_extractor)
    assert not has_nonzero_gradient(loaded.exploration_mlp)
    assert has_nonzero_gradient(loaded.mlp_extractor.value_net)
    assert has_nonzero_gradient(loaded.value_net)
    loaded.optimizer.zero_grad(set_to_none=True)
    loaded.get_distribution(observations).entropy().sum().backward()
    assert has_nonzero_gradient(loaded.exploration_mlp)
    assert not has_nonzero_gradient(loaded.features_extractor)
