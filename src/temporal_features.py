"""Learn policy features from observation, action, and accepted-target histories."""

from collections.abc import Mapping

from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActionObservationTransformer(BaseFeaturesExtractor):
    """Encode ``[observation, previous action, previous accepted XYZ target]``.

    ``valid`` distinguishes real timesteps from padding. ``episode_start``
    marks the initial timestep, where no previous action exists. Histories
    must contain at least one real timestep and never cross episode boundaries.
    The most recent real token provides a shared representation for PPO's
    actor and critic. All dropout is disabled so PPO reevaluates action
    probabilities without introducing a second source of sampling noise.
    ``forward_sequence`` exposes every causal representation for supervised
    learning, with sequence lengths up to the observation space's history size.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embedding_dim: int = 128,
        layer_count: int = 3,
        head_count: int = 4,
        feedforward_dim: int = 512,
    ) -> None:
        architecture = {
            "embedding_dim": embedding_dim,
            "layer_count": layer_count,
            "head_count": head_count,
            "feedforward_dim": feedforward_dim,
        }
        for name, value in architecture.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if embedding_dim % head_count:
            raise ValueError("embedding_dim must be divisible by head_count.")
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("Transformer observations must use a Dict space.")
        required = {"tokens", "valid", "episode_start"}
        if set(observation_space.spaces) != required:
            raise ValueError(
                "Transformer observations require tokens, valid, and episode_start."
            )
        token_space = observation_space["tokens"]
        if (
            not isinstance(token_space, spaces.Box)
            or len(token_space.shape) != 2
            or min(token_space.shape) <= 0
        ):
            raise ValueError("tokens must be a Box with shape (history, token_dim).")
        history_length, token_dim = token_space.shape
        for name in ("valid", "episode_start"):
            marker_space = observation_space[name]
            if (
                not isinstance(marker_space, spaces.Box)
                or marker_space.shape != (history_length,)
            ):
                raise ValueError(f"{name} must be a Box with shape (history,).")

        super().__init__(observation_space, features_dim=embedding_dim)
        self.history_length = history_length
        self.token_dim = token_dim
        self.token_projection = nn.Linear(token_dim, embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(history_length, embedding_dim)
        )
        self.episode_start_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.episode_start_embedding, std=0.02)
        # Construct separately so each layer starts with independent parameters.
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=head_count,
                    dim_feedforward=feedforward_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layer_count)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.register_buffer(
            "causal_mask",
            torch.ones(history_length, history_length, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def _sequence_features(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        validate_prefix: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = observations["tokens"]
        valid = observations["valid"] > 0.5
        episode_start = observations["episode_start"] > 0.5
        if (
            tokens.ndim != 3
            or tokens.shape[0] < 1
            or not 1 <= tokens.shape[1] <= self.history_length
            or tokens.shape[2] != self.token_dim
        ):
            raise ValueError(
                "tokens must have shape (batch, history, token_dim), with a "
                "nonempty batch and 1 <= history <= the configured history_length."
            )
        if valid.shape != tokens.shape[:2] or episode_start.shape != valid.shape:
            raise ValueError("valid and episode_start must have shape (batch, history).")
        if validate_prefix:
            # One scalar check for a supervised batch. The online history wrapper
            # guarantees this contract, so forward avoids synchronizing a device
            # tensor to the CPU on every policy decision.
            has_gap = (~valid[:, :-1] & valid[:, 1:]).any(dim=1)
            if not torch.all(valid[:, 0] & ~has_gap):
                raise ValueError("valid must contain a nonempty, contiguous prefix in every row.")
        length = tokens.shape[1]

        # Ignore padded values before projection, including any non-finite data.
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.token_projection(tokens) + self.position_embedding[:length]
        start_markers = (episode_start & valid).unsqueeze(-1)
        encoded = encoded + start_markers * self.episode_start_embedding
        for layer in self.encoder_layers:
            encoded = layer(
                encoded,
                src_mask=self.causal_mask[:length, :length],
                src_key_padding_mask=~valid,
            )
        features = self.final_norm(encoded)
        return features.masked_fill(~valid.unsqueeze(-1), 0.0), valid

    def forward_sequence(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return normalized causal features with shape ``(batch, history, dim)``.

        Each row must contain at least one valid token, followed only by right
        padding. Shorter tensors use the same learned position embeddings as
        their padded equivalents. Padding outputs are zero and carry no gradient;
        callers should also mask padded action labels when calculating their loss.
        """
        features, _ = self._sequence_features(observations, validate_prefix=True)
        return features

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features, valid = self._sequence_features(observations, validate_prefix=False)

        # The history wrapper guarantees nonempty, contiguous valid prefixes.
        # Tensor indexing avoids CPU synchronization for every policy decision.
        last_indices = valid.sum(dim=1) - 1
        return features[
            torch.arange(features.shape[0], device=features.device), last_indices
        ]
