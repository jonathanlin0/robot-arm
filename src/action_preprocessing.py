"""Conversion of recorded quaternion states to action yaw features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cube_yaw import CubeYawEncoder, _quaternion_rotation
from schemas.arrays import OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE


class ActionObservationPreprocessor:
    """Convert raw 41-value states to 37 policy features without editing data.

    Each cube quaternion becomes a levelled-face ``[cos(4*yaw), sin(4*yaw)]``
    pair. At numerical face ties, retain the preceding frame's face. Process
    whole episodes before slicing windows to preserve those tie decisions.
    """

    def __init__(self) -> None:
        self.orange = CubeYawEncoder()
        self.blue = CubeYawEncoder()

    def reset(self) -> None:
        self.orange.reset()
        self.blue.reset()

    def transform(self, observation: NDArray) -> NDArray[np.float32]:
        raw = np.asarray(observation, dtype=np.float64)
        if raw.shape != (OBSERVATION_SIZE,) or not np.isfinite(raw).all():
            raise ValueError(f"Raw action observation must contain {OBSERVATION_SIZE} finite values.")
        # Validate both quaternions before advancing either face history.
        orange_rotation = _quaternion_rotation(raw[24:28])
        blue_rotation = _quaternion_rotation(raw[34:38])
        orange = self.orange._encode_rotation(orange_rotation)
        blue = self.blue._encode_rotation(blue_rotation)
        return np.concatenate((raw[:24], orange, raw[28:34], blue, raw[38:])).astype(np.float32)

    def transform_episode(self, observations: NDArray) -> NDArray[np.float32]:
        """Reset at an actual episode boundary and encode rows in time order."""
        self.reset()
        raw = np.asarray(observations)
        if raw.ndim != 2 or raw.shape[1] != OBSERVATION_SIZE:
            raise ValueError(f"Raw action episode must have shape [time, {OBSERVATION_SIZE}].")
        result = np.empty((len(raw), POLICY_OBSERVATION_SIZE), dtype=np.float32)
        for index, observation in enumerate(raw):
            result[index] = self.transform(observation)
        return result
