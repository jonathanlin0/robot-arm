"""Quarter-turn-invariant cube headings and shared normalization.

The heading uses the most horizontal cube face, levelled by the smallest
rotation to the ground plane. At numerical face ties, an episode encoder keeps
the previous geometric face. It omits tilt and can jump when faces change.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F


_GEOMETRIC_TIE_TOLERANCE = 1e-6
_MINIMUM_QUATERNION_NORM = 1e-12
_YAW_NORMALIZATION_EPS = 1e-6


def normalize_yaw_tensor(yaw: Tensor) -> Tensor:
    """Normalize [..., 2] pairs with a finite, differentiable zero guard.

    Pairs shorter than epsilon are divided by epsilon, matching ``F.normalize``;
    an exactly zero prediction remains zero rather than inventing a heading.
    """
    if yaw.ndim == 0 or yaw.shape[-1] != 2:
        raise ValueError("Cube yaw pairs must have shape [..., 2].")
    return F.normalize(yaw, dim=-1, eps=_YAW_NORMALIZATION_EPS)


def normalize_yaw_pair(yaw: NDArray) -> NDArray[np.float32]:
    """Numpy version of the shared [..., 2] yaw normalization rule."""
    value = np.asarray(yaw, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 2:
        raise ValueError("Cube yaw pairs must have shape [..., 2].")
    if not np.isfinite(value).all():
        raise ValueError("Cube yaw pairs must contain finite values.")
    # hypot avoids overflow for finite large pairs and underflow for tiny ones.
    norm = np.hypot(value[..., 0], value[..., 1])[..., None]
    return (value / np.maximum(norm, _YAW_NORMALIZATION_EPS)).astype(np.float32)


def _quaternion_rotation(quaternion: NDArray) -> NDArray[np.float64]:
    """Normalize a finite wxyz quaternion without overflow or underflow."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Cube orientation must contain four finite wxyz quaternion values.")
    scale = float(np.abs(value).max())
    if scale == 0:
        raise ValueError("Cube orientation quaternion has zero or near-zero norm.")
    scaled = value / scale
    norm = float(np.linalg.norm(scaled))
    if scale <= _MINIMUM_QUATERNION_NORM / norm:
        raise ValueError("Cube orientation quaternion has zero or near-zero norm.")
    w, x, y, z = scaled / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _geometric_argmax(scores: NDArray, normals: NDArray) -> int:
    """Break numerical score ties lexicographically by world-space normals."""
    candidates = np.flatnonzero(scores >= np.max(scores) - _GEOMETRIC_TIE_TOLERANCE)
    for coordinate in range(3):
        values = normals[candidates, coordinate]
        candidates = candidates[values >= np.max(values) - _GEOMETRIC_TIE_TOLERANCE]
        if len(candidates) == 1:
            break
    return int(candidates[0])


class CubeYawEncoder:
    """Choose the most vertical face, retaining the previous face only at ties.

    A clearly more vertical face always wins: there is no switching margin.
    Match previous faces by world-space normals, so local axis permutations and
    signs cannot change the match. This nearest-normal association assumes
    modest inter-frame motion; large rotations can make a face match ambiguous.
    The first frame (or a tie excluding the previous face) uses geometric ties.
    """

    def __init__(self) -> None:
        self._selected_normal: NDArray[np.float64] | None = None

    @property
    def selected_normal(self) -> NDArray[np.float64] | None:
        """Return a copy of the selected upward normal, if a frame was encoded."""
        return None if self._selected_normal is None else self._selected_normal.copy()

    def reset(self) -> None:
        self._selected_normal = None

    def encode(self, quaternion: NDArray) -> NDArray[np.float32]:
        return self._encode_rotation(_quaternion_rotation(quaternion))

    def _encode_rotation(self, rotation: NDArray[np.float64]) -> NDArray[np.float32]:
        normals = rotation.T.copy()
        normals[normals[:, 2] < 0] *= -1
        scores = normals[:, 2]
        selected = _geometric_argmax(scores, normals)
        if self._selected_normal is not None:
            previous = _geometric_argmax(np.abs(normals @ self._selected_normal), normals)
            if scores[previous] >= np.max(scores) - _GEOMETRIC_TIE_TOLERANCE:
                selected = previous
        normal = normals[selected]

        # Rodrigues' shortest rotation from upward n to world Z maps an edge v
        # in n's plane to XY = v.xy - n.xy*v.z/(1+n.z). Unlike projection, it
        # preserves perpendicular edges; either edge/sign has the same pair.
        # The chosen axis has n.z near or above 1/sqrt(3), so this is safe.
        edge = rotation[:, 1 if selected == 0 else 0]
        horizontal = edge[:2] - normal[:2] * edge[2] / (1 + normal[2])
        horizontal /= np.linalg.norm(horizontal)
        cosine, sine = horizontal
        cos_double = cosine * cosine - sine * sine
        sin_double = 2 * cosine * sine
        result = np.array([cos_double * cos_double - sin_double * sin_double,
                           2 * cos_double * sin_double], dtype=np.float32)
        self._selected_normal = normal.copy()
        return result


def quaternion_to_yaw(quaternion: NDArray) -> NDArray[np.float32]:
    """Encode one wxyz quaternion without history as [cos(4*yaw), sin(4*yaw)].

    Quaternion sign, scale and all 24 local cube rotations leave the result
    unchanged. Use a persistent ``CubeYawEncoder`` for previous-face tie breaks
    across an episode; single frames break ties by world-space geometry.
    """
    return CubeYawEncoder().encode(quaternion)
