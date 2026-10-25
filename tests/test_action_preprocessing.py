"""Geometry, symmetry, and previous-face ties in action yaw features."""

from itertools import permutations, product

import mujoco
import numpy as np
import pytest
import torch

from action_preprocessing import ActionObservationPreprocessor
from cube_yaw import CubeYawEncoder, normalize_yaw_pair, normalize_yaw_tensor, quaternion_to_yaw


def rx(degrees):
    cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])


def ry(degrees):
    cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def rz(degrees):
    cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def quaternion(rotation):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, rotation.reshape(-1))
    return result


def cube_symmetries():
    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.eye(3)[:, permutation] * np.array(signs)[None, :]
            if np.linalg.det(matrix) > 0:
                matrices.append(matrix)
    assert len(matrices) == 24
    return matrices


def yaw_features(degrees):
    angle = 4 * np.deg2rad(degrees)
    return [np.cos(angle), np.sin(angle)]


def raw_observation(orange_rotation=None, blue_rotation=None):
    raw = np.arange(41, dtype=np.float32)
    raw[24:28] = quaternion(np.eye(3) if orange_rotation is None else orange_rotation)
    raw[34:38] = quaternion(np.eye(3) if blue_rotation is None else blue_rotation)
    return raw


@pytest.mark.parametrize("yaw", [-180, -60, -1, 0, 30, 44, 46, 89, 90, 180])
def test_upright_yaw_is_a_continuous_quarter_turn_encoding(yaw):
    result = CubeYawEncoder().encode(quaternion(rz(yaw)))
    assert result.shape == (2,) and result.dtype == np.float32
    np.testing.assert_allclose(result, yaw_features(yaw), atol=1e-7)
    np.testing.assert_allclose(CubeYawEncoder().encode(quaternion(rz(yaw + 90))), result, atol=1e-7)
    assert np.linalg.norm(result) == pytest.approx(1, abs=1e-7)


def test_yaw_wrap_has_no_feature_discontinuity():
    low = CubeYawEncoder().encode(quaternion(rz(0.1)))
    high = CubeYawEncoder().encode(quaternion(rz(89.9)))
    assert np.linalg.norm(low - high) < 0.015


def test_all_24_local_cube_symmetries_signs_and_scales_are_equivalent():
    rotation = rz(31) @ ry(20) @ rx(35)
    expected_encoder = CubeYawEncoder()
    expected = expected_encoder.encode(quaternion(rotation))
    for symmetry in cube_symmetries():
        q = quaternion(rotation @ symmetry)
        for scale in (1, -1, 7, -0.03, 1e-10, 1e300):
            encoder = CubeYawEncoder()
            actual = encoder.encode(q * scale)
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            np.testing.assert_allclose(encoder.selected_normal, expected_encoder.selected_normal, atol=1e-12)


def test_tilt_is_levelled_by_shortest_rotation_not_naive_projection():
    # Tilt around a diagonal world axis without adding twist, then rotate the
    # face within its own plane. Inverting that shortest tilt recovers yaw=30.
    axis = np.array([1., 2., 0.]) / np.sqrt(5)
    angle = np.deg2rad(30)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    tilt = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    rotation = tilt @ rz(30)
    encoder = CubeYawEncoder()
    np.testing.assert_allclose(encoder.encode(quaternion(rotation)), yaw_features(30), atol=1e-7)
    np.testing.assert_allclose(encoder.selected_normal, rotation[:, 2], atol=1e-12)
    naive_heading = np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
    assert abs(naive_heading - 30) > 1
    assert abs(rotation[:2, 0] @ rotation[:2, 1]) > 0.01


def test_previous_face_is_retained_only_at_numerical_ties():
    encoder = CubeYawEncoder()
    encoder.encode(quaternion(ry(20) @ rx(44)))
    for angle in (45, 45 + 1e-6):
        rotation = ry(20) @ rx(angle)
        encoder.encode(quaternion(rotation))
        np.testing.assert_allclose(encoder.selected_normal, rotation[:, 2], atol=1e-12)
    # A tiny but definite advantage switches immediately; no 5% margin remains.
    rotation = ry(20) @ rx(45.01)
    encoder.encode(quaternion(rotation))
    np.testing.assert_allclose(encoder.selected_normal, rotation[:, 1], atol=1e-12)
    rotation = ry(20) @ rx(45)
    encoder.encode(quaternion(rotation))
    np.testing.assert_allclose(encoder.selected_normal, rotation[:, 1], atol=1e-12)
    rotation = ry(20) @ rx(44.99)
    encoder.encode(quaternion(rotation))
    np.testing.assert_allclose(encoder.selected_normal, rotation[:, 2], atol=1e-12)


def test_standalone_conversion_uses_geometry_and_has_no_cross_call_state():
    q = quaternion(ry(20) @ rx(45))
    expected = quaternion_to_yaw(q)
    quaternion_to_yaw(quaternion(ry(20) @ rx(44)))
    np.testing.assert_array_equal(quaternion_to_yaw(q), expected)
    encoder = CubeYawEncoder()
    encoder.encode(quaternion(ry(20) @ rx(44)))
    assert np.linalg.norm(encoder.encode(q) - expected) > 0.5


def test_geometric_tracking_survives_permuted_vision_axis_labels():
    baseline, changed = CubeYawEncoder(), CubeYawEncoder()
    symmetries = cube_symmetries()
    angles = [44, 45, 45, 45.01, 45, 44.99, 45, 46, 45, 44]
    for index, angle in enumerate(angles):
        rotation = rz(17) @ ry(20) @ rx(angle)
        expected = baseline.encode(quaternion(rotation))
        q = quaternion(rotation @ symmetries[(index * 7) % len(symmetries)])
        actual = changed.encode((q * (-3 if index % 2 else 7)).astype(np.float32))
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        np.testing.assert_allclose(changed.selected_normal, baseline.selected_normal, atol=2e-7)


def test_exact_face_ties_use_geometry_not_axis_number():
    # Two faces tie at vertical score=2/3 and share an X coordinate; positive
    # world Y breaks that tie. Their levelled headings differ, so arbitrary
    # local-axis ordering would produce different encoded features.
    third, two_thirds, root_two = 1 / 3, 2 / 3, np.sqrt(2)
    rotation = np.array([[root_two * two_thirds, -third / root_two, -third / root_two],
                         [0, 1 / root_two, -1 / root_two],
                         [third, two_thirds, two_thirds]])
    expected = yaw_features(-np.rad2deg(np.arctan(1 / 7)))
    for symmetry in cube_symmetries():
        encoder = CubeYawEncoder()
        actual = encoder.encode(quaternion(rotation @ symmetry).astype(np.float32))
        np.testing.assert_allclose(encoder.selected_normal, rotation[:, 1], atol=2e-7)
        np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_near_vertical_normal_and_upward_choice_are_finite():
    for rotation in (rz(37) @ rx(1e-9), rz(37) @ rx(180 - 1e-9)):
        encoder = CubeYawEncoder()
        actual = encoder.encode(quaternion(rotation))
        assert np.isfinite(actual).all() and encoder.selected_normal[2] > 0
        np.testing.assert_allclose(actual, yaw_features(37), atol=1e-7)


@pytest.mark.parametrize("invalid", [np.zeros(4), np.full(4, 1e-14), [1, 0, 0],
                                     [np.nan, 0, 0, 0], [np.inf, 0, 0, 0], np.ones((2, 2))])
def test_invalid_quaternion_is_rejected_without_changing_tracking(invalid):
    encoder = CubeYawEncoder()
    encoder.encode(quaternion(ry(20) @ rx(44)))
    before = encoder.selected_normal
    with pytest.raises(ValueError, match="quaternion"):
        encoder.encode(invalid)
    np.testing.assert_array_equal(encoder.selected_normal, before)


def test_reset_discards_tie_history_and_normal_snapshots_do_not_alias():
    encoder = CubeYawEncoder()
    assert encoder.selected_normal is None
    encoder.encode(quaternion(ry(20) @ rx(44)))
    q = quaternion(ry(20) @ rx(45))
    history_dependent = encoder.encode(q)
    encoder.selected_normal[:] = 99
    assert np.linalg.norm(encoder.selected_normal) == pytest.approx(1)
    encoder.reset()
    assert encoder.selected_normal is None
    fresh = encoder.encode(q)
    assert np.linalg.norm(fresh - history_dependent) > 0.5
    np.testing.assert_allclose(fresh, CubeYawEncoder().encode(q))


def test_observation_layout_replaces_only_quaternions_and_preserves_source():
    raw = raw_observation(rz(30), rz(-60))
    before = raw.copy()
    result = ActionObservationPreprocessor().transform(raw)
    assert result.shape == (37,) and result.dtype == np.float32
    np.testing.assert_array_equal(result[:24], raw[:24])
    np.testing.assert_allclose(result[24:26], yaw_features(30), atol=1e-7)
    np.testing.assert_array_equal(result[26:32], raw[28:34])
    np.testing.assert_allclose(result[32:34], yaw_features(-60), atol=1e-7)
    np.testing.assert_array_equal(result[34:], raw[38:])
    np.testing.assert_array_equal(raw, before)
    result[:] = 0
    np.testing.assert_array_equal(raw, before)


def test_episode_transform_matches_live_processing_and_resets_between_episodes():
    raw = np.stack([raw_observation(ry(20) @ rx(angle), ry(-10) @ rx(90 - angle))
                    for angle in (44, 45, 45, 46, 45, 44)])
    before = raw.copy()
    processor = ActionObservationPreprocessor()
    full_episode = processor.transform_episode(raw)
    processor.reset()
    live = np.stack([processor.transform(row) for row in raw])
    np.testing.assert_array_equal(full_episode, live)
    np.testing.assert_array_equal(processor.transform_episode(raw), full_episode)
    np.testing.assert_array_equal(raw, before)
    # Starting an overlapping chunk afresh loses earlier face selection. The
    # training loader therefore must convert the full episode before splitting.
    fresh_chunk = processor.transform_episode(raw[1:])
    assert np.linalg.norm(fresh_chunk[0, 24:26] - full_episode[1, 24:26]) > 0.5
    empty = processor.transform_episode(np.empty((0, 41)))
    assert empty.shape == (0, 37) and empty.dtype == np.float32
    assert processor.orange.selected_normal is processor.blue.selected_normal is None


def test_bad_second_cube_does_not_advance_first_cubes_tracking():
    processor = ActionObservationPreprocessor()
    processor.transform(raw_observation(ry(20) @ rx(44)))
    before = processor.orange.selected_normal
    raw = raw_observation(ry(20) @ rx(48))
    raw[34:38] = 0
    with pytest.raises(ValueError, match="near-zero"):
        processor.transform(raw)
    np.testing.assert_array_equal(processor.orange.selected_normal, before)


def test_malformed_observations_and_episodes_are_rejected():
    processor = ActionObservationPreprocessor()
    for raw in (np.zeros(40), np.zeros((1, 41)), np.full(41, np.nan)):
        with pytest.raises(ValueError, match="41 finite values"):
            processor.transform(raw)
    for episode in (np.zeros(41), np.zeros((2, 40))):
        with pytest.raises(ValueError, match="shape"):
            processor.transform_episode(episode)


def test_shared_numpy_and_tensor_normalization_agree_including_zero_guard():
    values = np.array([[[3., 4.], [-3., -4.], [0., 0.]],
                       [[3e-8, 4e-8], [1e-6, 0.], [0., -7.]]], dtype=np.float32)
    actual = normalize_yaw_pair(values)
    expected = normalize_yaw_tensor(torch.from_numpy(values)).numpy()
    assert actual.shape == values.shape and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(actual[0, 2], [0., 0.])
    np.testing.assert_allclose(actual[1, 0], [0.03, 0.04], rtol=1e-6)
    np.testing.assert_allclose(actual[0, 0], [0.6, 0.8], rtol=1e-6)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_yaw_normalization_rejects_wrong_pair_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        normalize_yaw_pair(np.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        normalize_yaw_tensor(torch.zeros(shape))


def test_numpy_yaw_normalization_rejects_nonfinite_inputs():
    for invalid in ([np.nan, 0.], [np.inf, 0.]):
        with pytest.raises(ValueError, match="finite"):
            normalize_yaw_pair(invalid)


def test_tensor_yaw_normalization_has_finite_nonzero_directional_gradient():
    value = torch.tensor([[3., 4.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    normalized = normalize_yaw_tensor(value)
    normalized[:, 0].sum().backward()
    assert normalized.dtype == torch.float64
    assert torch.isfinite(value.grad).all()
    assert value.grad[0].abs().sum() > 0
    assert value.grad[1, 0] == pytest.approx(1e6)
    assert value.grad[1, 1] == 0
