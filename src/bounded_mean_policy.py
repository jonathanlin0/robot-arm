"""Actor-critic policy variants used by the robot control task."""

import torch
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy


class TanhBoundedMeanActorCriticPolicy(ActorCriticPolicy):
    """Keep each Gaussian action mean inside the normalized action bounds.

    This bounds only the distribution mean. Exploration noise remains
    unsquashed, so stochastic samples can still leave ``[-1, 1]`` and are
    clipped by Stable-Baselines3 before reaching the environment. Keeping the
    distribution itself unsquashed preserves its analytical entropy while
    preventing the deterministic mean from drifting to values such as -6.5.
    """

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
    ) -> Distribution:
        """Build the continuous-action distribution with a bounded mean."""
        mean_actions = torch.tanh(self.action_net(latent_pi))

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std,
            )
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std,
                latent_pi,
            )

        raise TypeError(
            "TanhBoundedMeanActorCriticPolicy requires a continuous Box "
            "action space backed by a Gaussian distribution."
        )
