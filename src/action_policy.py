"""Causal transformer producing an action prediction at every real timestamp."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from schemas import HISTORY_TOKEN_SIZE
from schemas.config import ActionPolicyConfig


class ActionPolicy(nn.Module):
    """Encode 44-value history tokens and predict bounded Cartesian actions.

    valid and episode_start are separate boolean [batch, time] masks. Histories
    contain a contiguous real prefix followed by right padding. The forward
    method predicts every timestamp for supervised training; predict_action
    selects the last real timestamp for live control.
    """

    def __init__(self, config: ActionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        transformer = config.transformer
        self.token_projection = nn.Linear(HISTORY_TOKEN_SIZE, transformer.embedding_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, config.history_length, transformer.embedding_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.episode_start_embedding = nn.Embedding(2, transformer.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer.embedding_dim, nhead=transformer.heads,
            dim_feedforward=transformer.feedforward_dim, dropout=transformer.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer.layers, norm=nn.LayerNorm(transformer.embedding_dim),
            enable_nested_tensor=False,
        )
        # Encoder clones begin with identical parameters in PyTorch. Initialize
        # each layer separately so the new network does not start symmetrically.
        for encoder_layer in self.transformer.layers:
            for parameter in encoder_layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        modules: list[nn.Module] = []
        width = transformer.embedding_dim
        for hidden_width in config.mlp_hidden_dims:
            modules.extend((nn.Linear(width, hidden_width), nn.ReLU()))
            width = hidden_width
        modules.extend((nn.Linear(width, 4), nn.Tanh()))
        self.action_head = nn.Sequential(*modules)

    def forward(self, tokens: Tensor, valid: Tensor, episode_start: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != HISTORY_TOKEN_SIZE:
            raise ValueError(f"tokens must have shape [batch, time, {HISTORY_TOKEN_SIZE}].")
        batch, length, _ = tokens.shape
        if not 1 <= length <= self.config.history_length:
            raise ValueError("History length exceeds the configured context window or is empty.")
        if valid.shape != (batch, length) or episode_start.shape != valid.shape:
            raise ValueError("History masks must match tokens' batch/time dimensions.")
        if valid.dtype != torch.bool or episode_start.dtype != torch.bool:
            raise ValueError("History masks must be boolean tensors.")
        if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Every history must contain a nonempty real prefix with right padding.")
        features = self.token_projection(tokens)
        features = features + self.position_embedding[:, :length] + self.episode_start_embedding(episode_start.long())
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        features = self.transformer(features, mask=causal, src_key_padding_mask=~valid)
        return self.action_head(features)

    def predict_action(self, tokens: Tensor, valid: Tensor, episode_start: Tensor, deterministic: bool) -> Tensor:
        actions = self(tokens, valid, episode_start)
        last_real = valid.sum(dim=1) - 1
        action = actions[torch.arange(actions.shape[0], device=actions.device), last_real]
        if not deterministic:
            # Optional inference exploration is explicit, independent Gaussian
            # noise. This supervised model does not train a PPO variance branch.
            action = action + torch.randn_like(action) * self.config.action_std
        return action.clamp(-1.0, 1.0)


def masked_action_loss(prediction: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """Mean squared action error over supervised real tokens, excluding padding."""
    if prediction.shape != labels.shape or prediction.shape[-1] != 4 or loss_mask.shape != prediction.shape[:-1]:
        raise ValueError("Action predictions, labels, and loss masks have incompatible shapes.")
    if loss_mask.dtype != torch.bool or not bool(loss_mask.any()):
        raise ValueError("A batch must contain at least one supervised real token.")
    # Index before arithmetic so arbitrary padded labels do not affect the loss.
    return (prediction[loss_mask] - labels[loss_mask]).square().mean()
