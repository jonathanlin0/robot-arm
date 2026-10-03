"""Actor-critic policy variants used by the robot control task."""

import math
from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class CriticDetachedMlpExtractor(MlpExtractor):
    """Train the critic MLP without sending its gradients into shared features."""

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        # Detach the input, keeping the critic MLP and value head trainable.
        # TEMP
        return super().forward_critic(features.detach())


class TanhBoundedMeanActorCriticPolicy(ActorCriticPolicy):
    """Keep each Gaussian action mean inside the normalized action bounds.

    This bounds only the distribution mean. Exploration noise remains
    unsquashed, so stochastic samples can still leave ``[-1, 1]`` and are
    clipped by Stable-Baselines3 before reaching the environment. Keeping the
    distribution itself unsquashed preserves its analytical entropy while
    preventing the deterministic mean from drifting to values such as -6.5.

    For gSDE, a separate exploration MLP reads detached transformer features.
    Its output passes through tanh before entering the noise calculation.
    PPO trains this MLP and the noise weights without sending their variance
    gradients into the transformer or the actor MLP. Its hidden layers match
    the actor's configured widths, with independent weights.

    ``sde_log_std_init`` optionally initializes one gSDE noise-weight log
    standard deviation per action. These are trainable starting values, not
    limits on the effective action standard deviations, which also depend on
    the tanh-bounded exploration features.
    """

    def __init__(
        self,
        *args: Any,
        sde_log_std_init: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if sde_log_std_init is not None:
            try:
                sde_log_std_init = tuple(float(value) for value in sde_log_std_init)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "sde_log_std_init must be a sequence of finite numbers."
                ) from error
            if not all(math.isfinite(value) for value in sde_log_std_init):
                raise ValueError("sde_log_std_init values must be finite.")

        super().__init__(*args, **kwargs)
        self.sde_log_std_init = sde_log_std_init
        if sde_log_std_init is None:
            return

        if not isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise ValueError("sde_log_std_init requires use_sde=True.")
        if not self.action_dist.full_std:
            raise ValueError("sde_log_std_init requires full_std=True.")
        if len(sde_log_std_init) != self.action_dist.action_dim:
            raise ValueError(
                "sde_log_std_init must contain one value per action dimension "
                f"({self.action_dist.action_dim})."
            )

        # Keep the Parameter registered with the optimizer and only change its
        # initial values. Checkpoint state loading subsequently restores the
        # learned values without applying this initialization again.
        with torch.no_grad():
            initial_values = self.log_std.new_tensor(sde_log_std_init)
            self.log_std.copy_(initial_values.expand_as(self.log_std))
        # SB3 sampled weights before applying the per-action initialization.
        self.reset_noise()

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters["sde_log_std_init"] = self.sde_log_std_init
        return parameters

    def _build_mlp_extractor(self) -> None:
        """Build actor, critic, and independent gSDE feature networks."""
        # TEMP
        if self.share_features_extractor:
            # All value paths (forward, evaluate_actions, predict_values) use
            # forward_critic, so one detach covers both PPO and direct calls.
            self.mlp_extractor = CriticDetachedMlpExtractor(
                self.features_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
        else:
            super()._build_mlp_extractor()

        self.exploration_mlp = None
        if self.use_sde:
            actor_layers = self.net_arch.get("pi", []) if isinstance(self.net_arch, dict) else self.net_arch
            layers: list[torch.nn.Module] = []
            input_dim = self.features_dim
            # Keep an actual trainable noise head even for a linear actor.
            for output_dim in actor_layers or [self.features_dim]:
                layers.extend((torch.nn.Linear(input_dim, output_dim), self.activation_fn()))
                input_dim = output_dim
            self.exploration_mlp = torch.nn.Sequential(*layers).to(self.device)
            if self.ortho_init:
                self.exploration_mlp.apply(partial(self.init_weights, gain=math.sqrt(2)))
            # SB3 normally detaches gSDE features. Let PPO train this separate
            # branch; its input is detached in _get_latents instead.
            self.action_dist.learn_features = True
            self.dist_kwargs["learn_features"] = True

    def _get_latents(
        self, observations: PyTorchObs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract once and route features into the three independent heads."""
        features = self.extract_features(observations)
        if self.share_features_extractor:
            actor_features = critic_features = features
        else:
            actor_features, critic_features = features
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        latent_vf = self.mlp_extractor.forward_critic(critic_features)
        latent_sde = (
            self.exploration_mlp(actor_features.detach())
            if self.exploration_mlp is not None else None
        )
        return latent_pi, latent_vf, latent_sde

    def forward(
        self, obs: PyTorchObs, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect actions and values using the same distribution as training."""
        latent_pi, latent_vf, latent_sde = self._get_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, self.value_net(latent_vf), log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Backpropagate PPO likelihoods through mean and exploration heads."""
        latent_pi, latent_vf, latent_sde = self._get_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Use the same mean and exploration heads for policy prediction."""
        latent_pi, _, latent_sde = self._get_latents(obs)
        return self._get_action_dist_from_latent(latent_pi, latent_sde)

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
        latent_sde: torch.Tensor | None = None,
    ) -> Distribution:
        """Build the continuous-action distribution with a bounded mean."""
        mean_actions = torch.tanh(self.action_net(latent_pi))

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std,
            )
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            if latent_sde is None:
                raise ValueError("gSDE requires features from the exploration MLP.")
            # Use the same bounded features for noise sampling and variance
            # calculation so PPO's log probabilities match the policy.
            noise_features = torch.tanh(latent_sde)
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std,
                noise_features,
            )

        raise TypeError(
            "TanhBoundedMeanActorCriticPolicy requires a continuous Box "
            "action space backed by a Gaussian distribution."
        )
