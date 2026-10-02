import gymnasium as gym
import numpy as np
import pytest
import torch

from temporal_features import ActionObservationTransformer


TOKEN_DIM = 56  # Observation (49), previous action (4), accepted target XYZ (3).


def history_space(length: int = 5) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "tokens": gym.spaces.Box(
                -np.inf, np.inf, shape=(length, TOKEN_DIM), dtype=np.float32
            ),
            "valid": gym.spaces.Box(0, 1, shape=(length,), dtype=np.float32),
            "episode_start": gym.spaces.Box(0, 1, shape=(length,), dtype=np.float32),
        }
    )


def extractor(length: int = 5) -> ActionObservationTransformer:
    return ActionObservationTransformer(
        history_space(length),
        embedding_dim=16,
        layer_count=3,
        head_count=4,
        feedforward_dim=32,
    )


def history(length: int = 5, counts: tuple[int, ...] = (1, 3, 5)) -> dict:
    valid = torch.arange(length)[None, :] < torch.tensor(counts)[:, None]
    episode_start = torch.zeros(len(counts), length)
    episode_start[:, 0] = 1
    return {
        "tokens": torch.randn(len(counts), length, TOKEN_DIM),
        "valid": valid.float(),
        "episode_start": episode_start,
    }


def test_full_and_short_histories_produce_finite_features() -> None:
    model = extractor()
    observations = history()
    # A full sliding window need not contain the episode's initial timestep.
    observations["episode_start"][-1].zero_()
    features = model(observations)

    assert features.shape == (3, 16)
    assert model.features_dim == 16
    assert torch.isfinite(features).all()


def test_padding_values_and_start_markers_cannot_change_features() -> None:
    model = extractor()
    observations = history()
    expected = model(observations)
    altered = {key: value.clone() for key, value in observations.items()}
    padding = ~observations["valid"].bool()
    altered["tokens"][padding] = float("nan")
    altered["episode_start"][padding] = 1

    torch.testing.assert_close(model(altered), expected)


def test_batched_histories_match_individual_histories() -> None:
    model = extractor()
    observations = history()
    batch_features = model(observations)
    for index in range(3):
        single = {key: value[index : index + 1] for key, value in observations.items()}
        torch.testing.assert_close(model(single)[0], batch_features[index])


def test_causal_mask_prevents_future_tokens_changing_earlier_encodings() -> None:
    model = extractor()
    observations = history(counts=(5,))
    outputs = []
    hook = model.encoder_layers[-1].register_forward_hook(
        lambda _module, _inputs, output: outputs.append(output.detach().clone())
    )
    try:
        model(observations)
        observations["tokens"][:, 3:] = 20 * torch.randn(1, 2, TOKEN_DIM)
        model(observations)
    finally:
        hook.remove()

    torch.testing.assert_close(outputs[0][:, :3], outputs[1][:, :3])
    assert not torch.allclose(outputs[0][:, -1], outputs[1][:, -1])


def test_gradient_reaches_past_observations_actions_and_targets_but_not_padding() -> None:
    torch.manual_seed(7)
    model = extractor()
    observations = history(counts=(3,))
    observations["tokens"].requires_grad_()
    features = model(observations)
    (features * torch.arange(16)).sum().backward()
    gradient = observations["tokens"].grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[0, :2, :49].abs().sum() > 0
    assert gradient[0, :2, 49:53].abs().sum() > 0
    assert gradient[0, :2, 53:].abs().sum() > 0
    assert gradient[0, 3:].abs().sum() == 0
    assert model.episode_start_embedding.grad.abs().sum() > 0


def test_episode_start_is_distinguishable_from_a_real_zero_action() -> None:
    model = extractor()
    observations = history(counts=(1,))
    observations["tokens"][:, 0, 49:53].zero_()
    initial_features = model(observations)
    observations["episode_start"].zero_()

    assert not torch.allclose(model(observations), initial_features)


def test_train_and_eval_features_match_without_dropout() -> None:
    model = extractor()
    observations = history()
    model.train()
    training_features = model(observations)
    repeated_features = model(observations)
    model.eval()
    with torch.no_grad():
        evaluation_features = model(observations)

    torch.testing.assert_close(training_features, repeated_features, rtol=0, atol=0)
    torch.testing.assert_close(training_features, evaluation_features)


def test_layers_have_independent_initial_values() -> None:
    model = extractor()
    assert not torch.equal(
        model.encoder_layers[0].self_attn.in_proj_weight,
        model.encoder_layers[1].self_attn.in_proj_weight,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"layer_count": -1},
        {"head_count": 0},
        {"feedforward_dim": 0},
        {"layer_count": 1.5},
        {"layer_count": True},
        {"embedding_dim": 15, "head_count": 4},
    ],
)
def test_invalid_architecture_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ActionObservationTransformer(history_space(), **kwargs)


@pytest.mark.parametrize("name", ["valid", "episode_start"])
def test_marker_spaces_must_match_history_length(name: str) -> None:
    space = history_space()
    space[name] = gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32)
    with pytest.raises(ValueError, match=name):
        ActionObservationTransformer(space)


def test_default_architecture_processes_64_token_window() -> None:
    model = ActionObservationTransformer(history_space(64))
    features = model(history(length=64, counts=(1, 64)))

    assert len(model.encoder_layers) == 3
    assert features.shape == (2, 128)
    assert torch.isfinite(features).all()
