"""Causality, padding, online history, and gradients of the action network."""

import numpy as np
import pytest
import torch

from action_policy import ActionPolicy, masked_action_loss
from schemas import CubeState, RobotState, OBSERVATION_FIELDS
from schemas.config import ActionPolicyConfig, TransformerConfig
from trained_policy import TrainedPolicy, build_observation, build_policy_observation


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(length=8, std=0.05):
    return ActionPolicyConfig(history_length=length,
        transformer=TransformerConfig(embedding_dim=16, feedforward_dim=32, heads=4, layers=2, dropout=0.0),
        mlp_hidden_dims=(12,), action_std=std, deterministic=True)


def state():
    robot = RobotState(timestamp=0.0, joint_positions=np.arange(6, dtype=np.float64),
                       joint_velocities=np.arange(6, dtype=np.float64) + 10,
                       commanded_joint_targets=np.arange(6, dtype=np.float64) + 20,
                       gripper_position=np.array([0.2, 0.0, 0.1]))
    cubes = {name: CubeState(timestamp=0.0, position=np.array([x, 0., 0.02]),
              orientation=np.array([1., 0., 0., 0.]), linear_velocity=np.zeros(3))
             for name, x in (("orange", 0.25), ("blue", 0.35))}
    return robot, cubes


def test_observation_order_and_float32_do_not_mutate_source():
    robot, cubes = state()
    result = build_observation(robot, cubes, observation_fields=OBSERVATION_FIELDS)
    assert result.shape == (41,)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:6], robot.joint_positions)
    np.testing.assert_array_equal(result[6:12], robot.joint_velocities)
    np.testing.assert_array_equal(result[12:18], robot.commanded_joint_targets)
    np.testing.assert_allclose(result[18:21], robot.gripper_position)
    np.testing.assert_allclose(result[21:24], cubes["orange"].position)
    np.testing.assert_allclose(result[31:34], cubes["blue"].position)
    assert robot.joint_positions.dtype == np.float64
    del cubes["orange"]
    with pytest.raises(KeyError):
        build_observation(robot, cubes)


def test_causal_attention_excludes_future_inputs():
    torch.manual_seed(10)
    model = ActionPolicy(config()).eval()
    tokens = torch.randn(2, 8, 44)
    valid = torch.ones(2, 8, dtype=torch.bool)
    starts = torch.zeros_like(valid)
    starts[:, 0] = True
    changed = tokens.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20
    with torch.no_grad():
        original = model(tokens, valid, starts)
        other = model(changed, valid, starts)
    torch.testing.assert_close(original[:, :4], other[:, :4], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(original[:, 4:], other[:, 4:])
    assert (original.abs() <= 1).all()


def test_padding_and_variable_length_produce_same_last_real_prediction():
    torch.manual_seed(7)
    model = ActionPolicy(config()).eval()
    tokens = torch.randn(1, 8, 44)
    valid = torch.tensor([[True, True, True, False, False, False, False, False]])
    starts = torch.zeros_like(valid)
    starts[:, 0] = True
    changed = tokens.clone()
    changed[:, 3:] *= 100
    with torch.no_grad():
        expected = model.predict_action(tokens, valid, starts, deterministic=True)
        actual = model.predict_action(changed, valid, starts, deterministic=True)
        short = model.predict_action(tokens[:, :3], valid[:, :3], starts[:, :3], deterministic=True)
    torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(expected, short, rtol=1e-5, atol=1e-6)


def test_invalid_padding_layout_is_rejected():
    model = ActionPolicy(config())
    with pytest.raises(ValueError, match="right padding"):
        model(torch.zeros(1, 3, 44), torch.tensor([[True, False, True]]), torch.zeros(1, 3, dtype=torch.bool))


def test_old_raw_quaternion_tokens_are_rejected():
    model = ActionPolicy(config())
    with pytest.raises(ValueError, match="44"):
        model(torch.zeros(1, 3, 48), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool))


def test_duck_typed_runtime_accepts_direct_yaw_without_preprocessing_configuration():
    class Policy:
        def predict(self, inputs, deterministic):
            return np.zeros(4), None

    runtime = TrainedPolicy(Policy(), history_length=3)
    robot, cubes = state()
    for cube in cubes.values():
        cube.orientation = None
        cube.yaw = np.array([0.6, 0.8])
    runtime.predict(robot, cubes, robot.gripper_position)
    np.testing.assert_allclose(runtime.tokens[0, 24:26], [0.6, 0.8])
    np.testing.assert_allclose(runtime.tokens[0, 32:34], [0.6, 0.8])


def test_policy_observation_prefers_direct_yaw_and_raw_storage_requires_quaternion():
    robot, cubes = state()
    cubes["orange"].orientation = np.full(4, np.nan)
    cubes["orange"].yaw = np.array([-0.6, 0.8])
    cubes["blue"].orientation = None
    cubes["blue"].yaw = np.array([0., -1.])
    result = build_policy_observation(robot, cubes)
    assert result.shape == (37,) and result.dtype == np.float32
    np.testing.assert_allclose(result[24:26], [-0.6, 0.8])
    np.testing.assert_allclose(result[32:34], [0., -1.])
    np.testing.assert_array_equal(cubes["orange"].yaw, [-0.6, 0.8])
    with pytest.raises(ValueError, match="quaternion"):
        build_observation(robot, cubes)
    cubes["blue"].yaw = None
    with pytest.raises(ValueError, match="requires yaw or quaternion"):
        build_policy_observation(robot, cubes)


def test_every_supervised_token_contributes_and_padding_has_no_gradient():
    prediction = torch.zeros(1, 4, 4, requires_grad=True)
    labels = torch.ones_like(prediction)
    labels[:, 2:] = float("nan")
    loss = masked_action_loss(prediction, labels, torch.tensor([[True, True, False, False]]))
    loss.backward()
    assert loss == 1
    assert (prediction.grad[:, :2] != 0).all()
    assert (prediction.grad[:, 2:] == 0).all()
    model = ActionPolicy(config())
    real = torch.ones(2, 5, dtype=torch.bool)
    loss = masked_action_loss(model(torch.randn(2, 5, 44), real, torch.zeros_like(real)), torch.ones(2, 5, 4) * 0.4, real)
    loss.backward()
    assert model.token_projection.weight.grad.abs().sum() > 0
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.action_head[0].weight.grad.abs().sum() > 0


def test_runtime_history_records_only_successfully_issued_actions_and_rolls():
    model = ActionPolicy(config(length=3)).eval()
    runtime = TrainedPolicy(model, observation_fields=OBSERVATION_FIELDS, history_length=3)
    robot, cubes = state()
    target = np.array([0.2, 0., 0.1])
    issued = np.array([0.1, 0.2, 0.3, -1.], dtype=np.float32)
    runtime.predict(robot, cubes, target, deterministic=True)
    assert runtime.episode_start_mask.tolist() == [True, False, False]
    assert runtime.valid_mask.tolist() == [True, False, False]
    np.testing.assert_array_equal(runtime.tokens[0, 37:41], 0.)
    runtime.predict(robot, cubes, target, deterministic=True)
    np.testing.assert_array_equal(runtime.tokens[1, 37:41], 0.)
    runtime.record_action(issued)
    issued[0] = 99  # The recorder must own its previous-action snapshot.
    runtime.predict(robot, cubes, target, deterministic=True)
    assert runtime.tokens[2, 37] == pytest.approx(0.1)
    np.testing.assert_allclose(runtime.tokens[2, 41:], target)
    runtime.predict(robot, cubes, target, deterministic=True)
    assert not runtime.episode_start_mask.any()
    assert runtime.valid_length == 3
    runtime.reset()
    assert not runtime.valid_mask.any()
    assert np.count_nonzero(runtime.tokens) == 0
    runtime.predict(robot, cubes, target, deterministic=True)
    assert runtime.episode_start_mask[0]


def test_exploration_std_is_explicit_and_model_knobs_are_used():
    model = ActionPolicy(config(std=0.0)).eval()
    tokens = torch.randn(1, 3, 44)
    valid = torch.ones(1, 3, dtype=torch.bool)
    starts = torch.zeros_like(valid)
    torch.testing.assert_close(model.predict_action(tokens, valid, starts, deterministic=True),
                               model.predict_action(tokens, valid, starts, deterministic=False))
    model.config.action_std = 0.2
    assert not torch.allclose(model.predict_action(tokens, valid, starts, deterministic=True),
                              model.predict_action(tokens, valid, starts, deterministic=False))
    assert model.token_projection.out_features == 16
    assert len(model.transformer.layers) == 2
    assert model.transformer.layers[0].linear1.out_features == 32
    assert model.transformer.layers[0].self_attn.num_heads == 4
    assert model.action_head[0].out_features == 12
