"""Episode-local observation/action/target windows for training and playback."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


HistoryObservation = dict[str, np.ndarray]
TARGET_GRIPPER_POSITION_SIZE = 3


class ActionObservationHistoryWrapper(gym.Wrapper):
    """Expose ``[observation_t, action_(t-1), accepted_target_xyz_(t-1)]``.

    The target comes from ``info['target_gripper_position']`` after the
    controller processes the previous action, including workspace constraints
    and IK outcomes. The initial token uses the controller's reset target and
    a zero previous action.

    Valid tokens occupy the beginning of the window in chronological order;
    unused slots are zero padded. Once full, the window evicts its oldest
    token. ``episode_start`` marks the initial token's missing previous action,
    distinguishing it from a real zero command. Every returned observation is
    an independent snapshot suitable for storage in a PPO rollout buffer.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        history_length: int = 384,
    ) -> None:
        super().__init__(env)
        if (
            isinstance(history_length, bool)
            or not isinstance(history_length, (int, np.integer))
            or history_length < 1
        ):
            raise ValueError("history_length must be a positive integer.")
        if not isinstance(env.observation_space, spaces.Box) or len(
            env.observation_space.shape
        ) != 1:
            raise ValueError("History requires a flat Box observation space.")
        if not isinstance(env.action_space, spaces.Box) or len(
            env.action_space.shape
        ) != 1:
            raise ValueError("History requires a flat Box action space.")

        self.history_length = int(history_length)
        self.action_size = env.action_space.shape[0]
        self.observation_size = env.observation_space.shape[0]
        token_size = (
            self.observation_size + self.action_size + TARGET_GRIPPER_POSITION_SIZE
        )
        self.observation_space = spaces.Dict(
            {
                "tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.history_length, token_size),
                    dtype=np.float32,
                ),
                "valid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.history_length,),
                    dtype=np.float32,
                ),
                "episode_start": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.history_length,),
                    dtype=np.float32,
                ),
            }
        )
        self._tokens = np.zeros(
            (self.history_length, token_size), dtype=np.float32
        )
        self._episode_start = np.zeros(self.history_length, dtype=np.float32)
        self._valid_count = 0
        self._needs_reset = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[HistoryObservation, dict[str, Any]]:
        self._needs_reset = True
        observation, info = self.env.reset(seed=seed, options=options)
        self._tokens.fill(0.0)
        self._episode_start.fill(0.0)
        self._valid_count = 0
        self._append(
            np.zeros(self.action_size, dtype=np.float32),
            observation,
            self._target_from_info(info),
            episode_start=True,
        )
        self._needs_reset = False
        return self._snapshot(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[HistoryObservation, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before starting an episode.")
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"received {action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Actions must contain only finite values.")
        issued_action = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        # If the environment advances but returns invalid history data, require
        # a reset instead of silently continuing with an incomplete history.
        self._needs_reset = True
        observation, reward, terminated, truncated, info = self.env.step(
            issued_action.copy()
        )
        self._append(
            issued_action,
            observation,
            self._target_from_info(info),
            episode_start=False,
        )
        self._needs_reset = bool(terminated or truncated)
        return self._snapshot(), reward, terminated, truncated, info

    @staticmethod
    def _target_from_info(info: dict[str, Any]) -> np.ndarray:
        key = "target_gripper_position"
        if key not in info:
            raise ValueError(
                "History requires info['target_gripper_position'] on reset and step."
            )
        message = (
            "info['target_gripper_position'] must be a finite XYZ vector "
            f"with shape {(TARGET_GRIPPER_POSITION_SIZE,)}."
        )
        try:
            target = np.asarray(info[key], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(message) from error
        if target.shape != (TARGET_GRIPPER_POSITION_SIZE,) or not np.all(
            np.isfinite(target)
        ):
            raise ValueError(message)
        return target

    def _append(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        target_gripper_position: np.ndarray,
        *,
        episode_start: bool,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape {(self.observation_size,)}, "
                f"received {observation.shape}."
            )
        if self._valid_count == self.history_length:
            self._tokens[:-1] = self._tokens[1:]
            self._episode_start[:-1] = self._episode_start[1:]
            index = self.history_length - 1
        else:
            index = self._valid_count
            self._valid_count += 1
        action_end = self.observation_size + self.action_size
        self._tokens[index, : self.observation_size] = observation
        self._tokens[index, self.observation_size : action_end] = action
        self._tokens[index, action_end:] = target_gripper_position
        self._episode_start[index] = float(episode_start)

    def _snapshot(self) -> HistoryObservation:
        valid = np.zeros(self.history_length, dtype=np.float32)
        valid[: self._valid_count] = 1.0
        return {
            "tokens": self._tokens.copy(),
            "valid": valid,
            "episode_start": self._episode_start.copy(),
        }
