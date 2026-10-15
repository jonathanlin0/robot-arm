import numpy as np
import pytest

from randomization import CubeSpawnConfig, sample_cube_positions


def test_sampled_positions_are_reproducible_and_valid() -> None:
    config = CubeSpawnConfig()
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    for _ in range(500):
        first_orange, first_blue = sample_cube_positions(first_rng, config)
        second_orange, second_blue = sample_cube_positions(second_rng, config)

        np.testing.assert_array_equal(first_orange, second_orange)
        np.testing.assert_array_equal(first_blue, second_blue)

        for position in (first_orange, first_blue):
            assert position.shape == (2,)
            assert np.all(np.isfinite(position))
            assert config.x_range[0] <= position[0] <= config.x_range[1]
            assert config.y_range[0] <= position[1] <= config.y_range[1]

        center_distance = np.linalg.norm(first_orange - first_blue)
        assert center_distance >= config.minimum_center_distance


def test_impossible_spawn_region_raises() -> None:
    config = CubeSpawnConfig(
        x_range=(0.0, 0.001),
        y_range=(0.0, 0.001),
        minimum_center_distance=1.0,
        maximum_attempts=3,
    )

    with pytest.raises(RuntimeError, match="Could not sample valid"):
        sample_cube_positions(np.random.default_rng(42), config)


@pytest.mark.parametrize(
    ("config_arguments", "expected_message"),
    [
        ({"x_range": (0.4, 0.2)}, "x_range"),
        ({"y_range": (0.1, 0.1)}, "y_range"),
        ({"minimum_center_distance": -0.01}, "minimum_center_distance"),
        ({"maximum_attempts": 0}, "maximum_attempts"),
    ],
)
def test_invalid_spawn_config_raises(
    config_arguments: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message): # note: match parameter is a weak match, not a strict string match
        CubeSpawnConfig(**config_arguments)
