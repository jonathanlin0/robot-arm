import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from action_observation_history import ActionObservationHistoryWrapper


class CountingEnvironment(gym.Env):
    """Small deterministic environment exposing action and reset boundaries."""

    def __init__(self, *, length: int = 10, offset: float = 0.0) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.5], dtype=np.float32),
            high=np.array([1.0, 0.5], dtype=np.float32),
        )
        self.length = length
        self.offset = offset
        self.steps = 0
        self.reset_count = 0
        self.actions: list[np.ndarray] = []
        self.last_options = None
        self.target_position = np.zeros(3, dtype=np.float32)

    def _update_target(self) -> None:
        # Simulate a processed target that differs from the action increment.
        # Reuse the array to ensure histories never alias controller state.
        self.target_position[:] = [
            self.offset + 0.5,
            0.25 * self.reset_count,
            -0.125 * self.steps,
        ]

    def _observation(self) -> np.ndarray:
        return np.array(
            [self.offset + self.steps, self.reset_count], dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_options = options
        self.reset_count += 1
        self.steps = 0
        self._update_target()
        return self._observation(), {
            "reset_count": self.reset_count,
            "target_gripper_position": self.target_position,
        }

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        self._update_target()
        return (
            self._observation(),
            1.25,
            False,
            self.steps >= self.length,
            {"step": self.steps, "target_gripper_position": self.target_position},
        )


def test_tokens_pair_current_observation_with_previous_action_and_processed_target():
    base = CountingEnvironment(offset=10)
    env = ActionObservationHistoryWrapper(base, history_length=3)
    initial, info = env.reset(seed=17, options={"example": True})

    assert info["reset_count"] == 1
    np.testing.assert_array_equal(info["target_gripper_position"], [10.5, 0.25, 0])
    assert base.last_options == {"example": True}
    assert env.observation_space.contains(initial)
    np.testing.assert_array_equal(initial["tokens"][0], [10, 1, 0, 0, 10.5, 0.25, 0])
    np.testing.assert_array_equal(initial["tokens"][1:], 0)
    np.testing.assert_array_equal(initial["valid"], [1, 0, 0])
    np.testing.assert_array_equal(initial["episode_start"], [1, 0, 0])

    requested = np.array([2.0, -0.75], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(requested)

    np.testing.assert_array_equal(base.actions[-1], [1.0, -0.5])
    np.testing.assert_array_equal(requested, [2.0, -0.75])
    np.testing.assert_array_equal(
        observation["tokens"][1], [11, 1, 1, -0.5, 10.5, 0.25, -0.125]
    )
    np.testing.assert_array_equal(observation["valid"], [1, 1, 0])
    np.testing.assert_array_equal(observation["episode_start"], [1, 0, 0])
    assert env.observation_space.contains(observation)
    assert reward == 1.25
    assert not terminated
    assert not truncated
    assert info["step"] == 1
    np.testing.assert_array_equal(info["target_gripper_position"], [10.5, 0.25, -0.125])


def test_window_evicts_oldest_token_and_its_episode_start_marker():
    env = ActionObservationHistoryWrapper(CountingEnvironment(), history_length=3)
    env.reset()
    env.step(np.array([0.1, 0.2]))
    full, *_ = env.step(np.array([0.3, 0.4]))
    np.testing.assert_array_equal(full["tokens"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(full["episode_start"], [1, 0, 0])

    evicted, *_ = env.step(np.array([0.5, -0.4]))
    np.testing.assert_allclose(
        evicted["tokens"],
        [
            [1, 1, 0.1, 0.2, 0.5, 0.25, -0.125],
            [2, 1, 0.3, 0.4, 0.5, 0.25, -0.25],
            [3, 1, 0.5, -0.4, 0.5, 0.25, -0.375],
        ],
    )
    np.testing.assert_array_equal(evicted["valid"], [1, 1, 1])
    np.testing.assert_array_equal(evicted["episode_start"], [0, 0, 0])


def test_returned_histories_are_independent_snapshots():
    env = ActionObservationHistoryWrapper(CountingEnvironment(), history_length=2)
    initial, _ = env.reset()
    first, *_ = env.step(np.array([0.2, 0.3]))
    expected_first = {key: value.copy() for key, value in first.items()}
    env.step(np.array([0.4, 0.5]))
    env.reset()

    np.testing.assert_array_equal(
        initial["tokens"], [[0, 1, 0, 0, 0.5, 0.25, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    for key in first:
        np.testing.assert_array_equal(first[key], expected_first[key])
        first[key].fill(99)
    after_mutation, *_ = env.step(np.zeros(2))
    np.testing.assert_array_equal(
        after_mutation["tokens"],
        [[0, 2, 0, 0, 0.5, 0.5, 0], [1, 2, 0, 0, 0.5, 0.5, -0.125]],
    )
    np.testing.assert_array_equal(after_mutation["valid"], [1, 1])
    np.testing.assert_array_equal(after_mutation["episode_start"], [1, 0])


def test_reset_clears_all_prior_tokens_and_distinguishes_real_zero_action():
    env = ActionObservationHistoryWrapper(CountingEnvironment(), history_length=4)
    env.reset()
    env.step(np.ones(2))
    env.step(np.ones(2))
    fresh, _ = env.reset()
    np.testing.assert_array_equal(fresh["tokens"][0], [0, 2, 0, 0, 0.5, 0.5, 0])
    np.testing.assert_array_equal(fresh["tokens"][1:], 0)
    np.testing.assert_array_equal(fresh["valid"], [1, 0, 0, 0])
    after_zero, *_ = env.step(np.zeros(2))
    np.testing.assert_array_equal(after_zero["tokens"][1, 2:4], [0, 0])
    np.testing.assert_array_equal(after_zero["episode_start"], [1, 0, 0, 0])


def test_length_one_retains_current_observation_previous_action_and_target():
    env = ActionObservationHistoryWrapper(CountingEnvironment(), history_length=1)
    initial, _ = env.reset()
    np.testing.assert_array_equal(initial["episode_start"], [1])
    current, *_ = env.step(np.array([0.25, -0.25]))
    np.testing.assert_array_equal(
        current["tokens"], [[1, 1, 0.25, -0.25, 0.5, 0.25, -0.125]]
    )
    np.testing.assert_array_equal(current["episode_start"], [0])


def test_tokens_use_reported_target_even_when_action_does_not_move_it(monkeypatch):
    base = CountingEnvironment()
    env = ActionObservationHistoryWrapper(base, history_length=2)
    initial, _ = env.reset()
    monkeypatch.setattr(base, "_update_target", lambda: None)

    current, *_ = env.step(np.array([0.75, -0.5]))

    np.testing.assert_array_equal(current["tokens"][1, 2:4], [0.75, -0.5])
    np.testing.assert_array_equal(current["tokens"][1, -3:], initial["tokens"][0, -3:])
    np.testing.assert_array_equal(current["tokens"][1, :2], [1, 1])


def test_mutating_info_target_cannot_change_current_or_future_history():
    env = ActionObservationHistoryWrapper(CountingEnvironment(), history_length=3)
    initial, initial_info = env.reset()
    initial_info["target_gripper_position"].fill(42)
    first, _, _, _, first_info = env.step(np.zeros(2))
    first_info["target_gripper_position"].fill(99)
    second, *_ = env.step(np.zeros(2))

    np.testing.assert_array_equal(initial["tokens"][0, -3:], [0.5, 0.25, 0])
    np.testing.assert_array_equal(first["tokens"][1, -3:], [0.5, 0.25, -0.125])
    np.testing.assert_array_equal(
        second["tokens"][:, -3:],
        [[0.5, 0.25, 0], [0.5, 0.25, -0.125], [0.5, 0.25, -0.25]],
    )


@pytest.mark.parametrize("boundary", ["reset", "step"])
@pytest.mark.parametrize(
    "bad_info",
    [
        {},
        {"target_gripper_position": None},
        {"target_gripper_position": 0.0},
        {"target_gripper_position": [0.0, 0.0]},
        {"target_gripper_position": [[0.0, 0.0, 0.0]]},
        {"target_gripper_position": [0.0, np.nan, 0.0]},
        {"target_gripper_position": [0.0, 0.0, np.inf]},
        {"target_gripper_position": ["invalid", "target", "position"]},
    ],
)
def test_missing_or_invalid_target_requires_reset(monkeypatch, boundary, bad_info):
    base = CountingEnvironment()
    env = ActionObservationHistoryWrapper(base)
    env.reset()
    original = getattr(base, boundary)

    def invalid_info(*args, **kwargs):
        result = list(original(*args, **kwargs))
        result[-1] = bad_info.copy()
        return tuple(result)

    monkeypatch.setattr(base, boundary, invalid_info)
    with pytest.raises(ValueError, match="target_gripper_position"):
        if boundary == "reset":
            env.reset()
        else:
            env.step(np.zeros(2))
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))


@pytest.mark.parametrize("history_length", [0, -1, 1.5, True])
def test_history_length_must_be_positive_integer(history_length):
    with pytest.raises(ValueError, match="positive integer"):
        ActionObservationHistoryWrapper(CountingEnvironment(), history_length=history_length)


def test_step_requires_reset_at_both_episode_boundaries():
    env = ActionObservationHistoryWrapper(CountingEnvironment(length=1))
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))
    env.reset()
    _, _, _, truncated, _ = env.step(np.zeros(2))
    assert truncated
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))
    env.reset()
    env.step(np.zeros(2))


@pytest.mark.parametrize("action", [[0.0], [0.0, np.nan], [np.inf, 0.0]])
def test_invalid_actions_do_not_reach_the_environment(action):
    base = CountingEnvironment()
    env = ActionObservationHistoryWrapper(base)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array(action))
    assert base.actions == []


def test_vector_workers_reset_independently_and_preserve_terminal_history():
    env = DummyVecEnv(
        [
            lambda: ActionObservationHistoryWrapper(
                CountingEnvironment(length=1, offset=10), history_length=3
            ),
            lambda: ActionObservationHistoryWrapper(
                CountingEnvironment(length=3, offset=20), history_length=3
            ),
        ]
    )
    try:
        env.reset()
        observation, rewards, dones, infos = env.step(
            np.array([[0.2, 0.3], [-0.4, 0.5]], dtype=np.float32)
        )
        np.testing.assert_array_equal(dones, [True, False])
        np.testing.assert_array_equal(rewards, [1.25, 1.25])
        np.testing.assert_array_equal(observation["valid"], [[1, 0, 0], [1, 1, 0]])
        np.testing.assert_array_equal(
            observation["tokens"][0, 0], [10, 2, 0, 0, 10.5, 0.5, 0]
        )
        np.testing.assert_allclose(
            observation["tokens"][1, 1], [21, 1, -0.4, 0.5, 20.5, 0.25, -0.125]
        )
        terminal = infos[0]["terminal_observation"]
        np.testing.assert_allclose(
            terminal["tokens"],
            [
                [10, 1, 0, 0, 10.5, 0.25, 0],
                [11, 1, 0.2, 0.3, 10.5, 0.25, -0.125],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        np.testing.assert_array_equal(terminal["valid"], [1, 1, 0])
        np.testing.assert_array_equal(terminal["episode_start"], [1, 0, 0])
        assert infos[0]["TimeLimit.truncated"]

        env.step(np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(terminal["tokens"][:, 0], [10, 11, 0])
        np.testing.assert_array_equal(
            terminal["tokens"][:, -3:], [[10.5, 0.25, 0], [10.5, 0.25, -0.125], [0, 0, 0]]
        )
    finally:
        env.close()
