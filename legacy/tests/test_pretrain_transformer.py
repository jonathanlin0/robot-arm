"""Causal sequence features shared by pretraining and online policy inference."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from temporal_features import ActionObservationTransformer


TOKEN_DIM = 56
FEATURE_DIM = 16


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def extractor(maximum_length: int = 8) -> ActionObservationTransformer:
    observation_space = gym.spaces.Dict({
        "tokens": gym.spaces.Box(
            -np.inf, np.inf, shape=(maximum_length, TOKEN_DIM), dtype=np.float32,
        ),
        "valid": gym.spaces.Box(0, 1, shape=(maximum_length,), dtype=np.float32),
        "episode_start": gym.spaces.Box(0, 1, shape=(maximum_length,), dtype=np.float32),
    })
    return ActionObservationTransformer(
        observation_space, embedding_dim=FEATURE_DIM, layer_count=2,
        head_count=2, feedforward_dim=32,
    )


def observations(length: int = 8, counts: tuple[int, ...] = (3, 8)) -> dict:
    valid = torch.arange(length)[None, :] < torch.tensor(counts)[:, None]
    starts = torch.zeros(len(counts), length)
    starts[:, 0] = 1
    return {
        "tokens": torch.randn(len(counts), length, TOKEN_DIM),
        "valid": valid.float(),
        "episode_start": starts,
    }


def test_each_position_is_causal_in_tokens_and_start_markers() -> None:
    torch.manual_seed(19)
    model = extractor()
    inputs = observations(counts=(8,))
    expected = model.forward_sequence(inputs)

    for position in range(7):
        changed = {name: tensor.clone() for name, tensor in inputs.items()}
        changed["tokens"][:, position + 1:] += 10 * torch.randn_like(
            changed["tokens"][:, position + 1:]
        )
        changed["episode_start"][:, position + 1:] = 1
        actual = model.forward_sequence(changed)
        torch.testing.assert_close(actual[:, :position + 1], expected[:, :position + 1])
        assert not torch.allclose(actual[:, -1], expected[:, -1])


def test_padding_is_inert_even_with_nonfinite_tokens_and_start_markers() -> None:
    model = extractor()
    inputs = observations()
    expected = model.forward_sequence(inputs)
    changed = {name: tensor.clone() for name, tensor in inputs.items()}
    padding = ~inputs["valid"].bool()
    changed["tokens"][padding] = float("nan")
    changed["tokens"][0, -1] = float("inf")
    changed["episode_start"][padding] = 1

    actual = model.forward_sequence(changed)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[padding]) == 0


def test_latest_matches_sequence_and_every_valid_position_is_normalized() -> None:
    torch.manual_seed(23)
    model = extractor()
    with torch.no_grad():
        model.final_norm.weight.copy_(torch.linspace(0.5, 2.0, FEATURE_DIM))
        model.final_norm.bias.copy_(torch.linspace(-1.0, 1.0, FEATURE_DIM))
    inputs = observations(counts=(1, 3, 8))
    encoded = []
    hook = model.encoder_layers[-1].register_forward_hook(
        lambda _module, _arguments, output: encoded.append(output.detach())
    )
    try:
        sequence = model.forward_sequence(inputs)
    finally:
        hook.remove()

    valid = inputs["valid"].bool()
    normalized = F.layer_norm(
        encoded[0], (FEATURE_DIM,), model.final_norm.weight,
        model.final_norm.bias, model.final_norm.eps,
    )
    torch.testing.assert_close(sequence[valid], normalized[valid])
    assert not torch.allclose(sequence[valid], encoded[0][valid])
    latest_indices = inputs["valid"].sum(dim=1).long() - 1
    torch.testing.assert_close(model(inputs), sequence[torch.arange(3), latest_indices])


@pytest.mark.parametrize("length", [1, 2, 5, 8])
def test_short_inputs_match_padded_prefixes_and_online_latest(length: int) -> None:
    model = extractor()
    padded = observations(counts=(length,))
    short = {name: tensor[:, :length] for name, tensor in padded.items()}
    sequence = model.forward_sequence(short)

    assert sequence.shape == (1, length, FEATURE_DIM)
    torch.testing.assert_close(sequence, model.forward_sequence(padded)[:, :length])
    torch.testing.assert_close(model(short), model(padded))
    torch.testing.assert_close(model(short), sequence[:, -1])


def test_full_384_position_capacity_and_bounded_inputs() -> None:
    model = extractor(384)
    inputs = observations(length=384, counts=(1, 384))
    features = model.forward_sequence(inputs)

    assert features.shape == (2, 384, FEATURE_DIM)
    assert torch.isfinite(features).all()
    assert model.position_embedding.shape == (384, FEATURE_DIM)
    with pytest.raises(ValueError, match="history_length"):
        model.forward_sequence(observations(length=385, counts=(385,)))


@pytest.mark.parametrize("method", ["forward", "forward_sequence"])
@pytest.mark.parametrize(
    "field, shape",
    [
        ("tokens", (2, 8)),
        ("tokens", (2, 8, TOKEN_DIM + 1)),
        ("tokens", (2, 0, TOKEN_DIM)),
        ("tokens", (0, 8, TOKEN_DIM)),
        ("tokens", (2, 9, TOKEN_DIM)),
        ("valid", (2, 7)),
        ("episode_start", (2, 8, 1)),
    ],
)
def test_malformed_shapes_are_rejected(method: str, field: str, shape: tuple) -> None:
    inputs = observations()
    inputs[field] = torch.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        getattr(extractor(), method)(inputs)


@pytest.mark.parametrize("valid", [[0] * 8, [0, 1, 1, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 1, 0, 0, 0, 0]])
def test_sequence_rejects_empty_left_padded_or_gapped_histories(valid: list) -> None:
    inputs = observations()
    inputs["valid"][0] = torch.tensor(valid)
    with pytest.raises(ValueError, match="nonempty, contiguous prefix"):
        extractor().forward_sequence(inputs)


def test_sequence_loss_reaches_all_valid_tokens_and_parameters_not_padding() -> None:
    torch.manual_seed(29)
    model = extractor()
    inputs = observations(counts=(3, 5))
    inputs["tokens"].requires_grad_()
    sequence = model.forward_sequence(inputs)
    weights = torch.arange(1, FEATURE_DIM + 1, dtype=sequence.dtype)
    (sequence * weights).sum().backward()

    valid = inputs["valid"].bool()
    gradient = inputs["tokens"].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient[valid].abs().sum(dim=-1) > 0)
    assert torch.count_nonzero(gradient[~valid]) == 0
    assert torch.count_nonzero(model.position_embedding.grad[5:]) == 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_single_position_loss_cannot_reach_future_tokens() -> None:
    model = extractor()
    inputs = observations(counts=(8,))
    inputs["tokens"].requires_grad_()
    features = model.forward_sequence(inputs)
    (features[:, 2] * torch.arange(FEATURE_DIM)).sum().backward()

    gradient = inputs["tokens"].grad
    assert torch.all(gradient[0, :3].abs().sum(dim=-1) > 0)
    assert torch.count_nonzero(gradient[0, 3:]) == 0


def test_sequence_api_preserves_existing_checkpoint_parameter_names() -> None:
    model = extractor()
    layer_names = {
        "self_attn.in_proj_weight", "self_attn.in_proj_bias",
        "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
    }
    expected = {
        "position_embedding", "episode_start_embedding",
        "token_projection.weight", "token_projection.bias",
        "final_norm.weight", "final_norm.bias",
    } | {f"encoder_layers.{index}.{name}" for index in range(2) for name in layer_names}
    state = model.state_dict()
    assert set(state) == expected
    assert state["position_embedding"].shape == (8, FEATURE_DIM)
    restored = extractor()
    restored.load_state_dict(state, strict=True)
    inputs = observations(length=5, counts=(2, 5))
    torch.testing.assert_close(restored.forward_sequence(inputs), model.forward_sequence(inputs))
    torch.testing.assert_close(restored(inputs), model(inputs))
