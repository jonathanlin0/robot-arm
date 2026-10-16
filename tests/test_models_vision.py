"""Frozen backbone, trainable adapter, shared encoding, and pose-loss contracts."""

import pytest
import torch
from torch import nn

from schemas.config import VisionModelConfig
from vision_model import VisionModel, cube_pose_loss


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyBackbone(nn.Module):
    """Cheap differentiable stand-in with the actual pooled feature width."""
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.projection = nn.Linear(3, 2048)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return self.projection(self.norm(image).mean(dim=(2, 3)))


def config():
    return VisionModelConfig(adapter_kernel_size=3, backbone_weights=None, mlp_hidden_layers=2,
                             mlp_hidden_dim=16, dropout=0.1)


def test_shared_frozen_backbone_does_not_block_adapter_gradient():
    torch.manual_seed(8)
    backbone = TinyBackbone()
    model = VisionModel(config=config(), backbone=backbone).train()
    wrist, external = torch.rand(2, 3, 300, 300), torch.rand(2, 3, 300, 300)
    before = backbone.norm.running_mean.clone()
    prediction = model(wrist, external, torch.randn(2, 6))
    assert prediction.shape == (2, 14)
    prediction.square().mean().backward()
    assert backbone.calls == 2
    assert not backbone.training and not backbone.norm.training
    assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
    assert model.input_adapter.weight.grad.abs().sum() > 0
    assert model.pose_head[0].weight.grad.abs().sum() > 0
    torch.testing.assert_close(before, backbone.norm.running_mean)
    model.eval().train()
    assert not backbone.training


def test_fusion_order_is_wrist_then_external_then_joint_positions():
    model = VisionModel(config=config(), backbone=TinyBackbone()).eval()
    wrist, external, joints = torch.zeros(1, 3, 300, 300), torch.ones(1, 3, 300, 300), torch.arange(6.).reshape(1, 6)
    fused = []
    handle = model.pose_head.register_forward_pre_hook(lambda _, args: fused.append(args[0]))
    model(wrist, external, joints)
    handle.remove()
    torch.testing.assert_close(fused[0][:, :2048], model.encode_image(wrist))
    torch.testing.assert_close(fused[0][:, 2048:4096], model.encode_image(external))
    torch.testing.assert_close(fused[0][:, 4096:], joints)
    assert model.pose_head[0].in_features == 4102
    assert model.pose_head[0].out_features == 16
    assert len([layer for layer in model.pose_head.modules() if isinstance(layer, nn.Linear)]) == 3


def test_hidden_dropout_masks_and_scales_during_training_but_is_identity_in_eval():
    torch.manual_seed(17)
    model = VisionModel(config=config(), backbone=TinyBackbone()).train()
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    hidden_linears = [layer for layer in model.pose_head.modules() if isinstance(layer, nn.Linear)][:-1]
    assert len(dropout_layers) == len(hidden_linears) == model.config.mlp_hidden_layers
    assert all(layer.p == model.config.dropout and layer.training for layer in dropout_layers)
    assert not model.backbone.training
    # Known positive activations distinguish dropout zeros from ReLU zeros.
    with torch.no_grad():
        for layer in hidden_linears:
            layer.weight.zero_()
            layer.bias.fill_(1.0)
    activations = []
    handles = [layer.register_forward_hook(
        lambda _, inputs, output: activations.append((inputs[0].detach().clone(), output.detach().clone()))
    ) for layer in dropout_layers]
    features = torch.ones(64, 4102)
    model.pose_head(features)
    assert len(activations) == model.config.mlp_hidden_layers
    for before, after in activations:
        assert (before == 1).all()
        kept = after != 0
        assert kept.any() and not kept.all()
        torch.testing.assert_close(after[kept], before[kept] / (1 - model.config.dropout))

    model.eval()
    activations.clear()
    first = model.pose_head(features)
    second = model.pose_head(features)
    assert all(not layer.training for layer in dropout_layers)
    assert not model.backbone.training
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for before, after in activations:
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_zero_dropout_keeps_training_head_deterministic():
    model = VisionModel(config=config().model_copy(update={"dropout": 0.0}), backbone=TinyBackbone()).train()
    assert all(layer.p == 0.0 for layer in model.modules() if isinstance(layer, nn.Dropout))
    features = torch.randn(8, 4102)
    torch.testing.assert_close(model.pose_head(features), model.pose_head(features), rtol=0, atol=0)


def test_head_without_hidden_layers_does_not_drop_raw_pose_outputs():
    model = VisionModel(config=config().model_copy(update={"mlp_hidden_layers": 0}), backbone=TinyBackbone()).train()
    assert not any(isinstance(layer, nn.Dropout) for layer in model.modules())
    assert model.pose_head(torch.zeros(2, 4102)).shape == (2, 14)


def test_adapter_begins_as_identity_then_applies_imagenet_normalization():
    model = VisionModel(config=config(), backbone=TinyBackbone())
    image = torch.rand(1, 3, 300, 300)
    torch.testing.assert_close(model.input_adapter(image), image)
    with pytest.raises(ValueError, match="300"):
        model(torch.zeros(1, 3, 224, 224), image, torch.zeros(1, 6))


def test_quaternion_sign_does_not_change_pose_loss():
    target = torch.tensor([[0.2, 0., 0.02, 1., 0., 0., 0., 0.3, 0., 0.02, 1., 0., 0., 0.]])
    prediction = target.clone()
    prediction[:, 3:7] *= -1
    prediction[:, 10:14] *= -1
    assert cube_pose_loss(prediction, target) == 0
    prediction[:, 0] += 0.3
    assert cube_pose_loss(prediction, target, position_weight=1.0, orientation_weight=1.0) == pytest.approx(0.015)


def test_pose_loss_defaults_to_position_weight_200_and_orientation_weight_one():
    target = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]] * 2).reshape(1, 14)
    prediction = torch.tensor([[0.3, -0.6, 0.9, 1., 1., 0., 0.]] * 2).reshape(1, 14)
    position_mse = (0.3 ** 2 + 0.6 ** 2 + 0.9 ** 2) / 3
    orientation_mse = (1 - 2 ** -0.5) / 2

    assert cube_pose_loss(prediction, target).item() == pytest.approx(200 * position_mse + orientation_mse)


@pytest.mark.parametrize("position_weight,orientation_weight", [(7.5, 3.25), (0.0, 2.0), (6.0, 0.0)])
def test_pose_loss_weights_scale_corresponding_gradients(position_weight, orientation_weight):
    target = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]] * 2, dtype=torch.float64).reshape(1, 14)
    prediction = torch.tensor([[0.3, -0.6, 0.9, 1., 1., 0., 0.]] * 2,
                              dtype=torch.float64).reshape(1, 14).requires_grad_()
    baseline = cube_pose_loss(prediction, target, position_weight=1.0, orientation_weight=1.0)
    baseline_gradient, = torch.autograd.grad(baseline, prediction)
    weighted = cube_pose_loss(prediction, target, position_weight=position_weight,
                              orientation_weight=orientation_weight)
    weighted_gradient, = torch.autograd.grad(weighted, prediction)
    position_indices, orientation_indices = [0, 1, 2, 7, 8, 9], [3, 4, 5, 6, 10, 11, 12, 13]

    torch.testing.assert_close(weighted_gradient[:, position_indices],
                               baseline_gradient[:, position_indices] * position_weight)
    torch.testing.assert_close(weighted_gradient[:, orientation_indices],
                               baseline_gradient[:, orientation_indices] * orientation_weight)
    expected = position_weight * 0.42 + orientation_weight * (1 - 2 ** -0.5) / 2
    assert weighted.item() == pytest.approx(expected)


def test_runtime_outputs_are_raw_without_pose_value_checks():
    model = VisionModel(config=config(), backbone=TinyBackbone())
    with torch.no_grad():
        model.pose_head[-1].weight.zero_()
        model.pose_head[-1].bias.fill_(123.0)
        output = model(torch.zeros(1, 3, 300, 300), torch.zeros(1, 3, 300, 300), torch.zeros(1, 6))
    assert (output == 123).all()


def test_real_resnet_forward_and_adapter_backward_without_download():
    config = VisionModelConfig(adapter_kernel_size=3, backbone_weights=None, mlp_hidden_layers=1,
                               mlp_hidden_dim=8, dropout=0.1)
    model = VisionModel(config=config, backbone=None).train()
    output = model(torch.rand(1, 3, 300, 300), torch.rand(1, 3, 300, 300), torch.zeros(1, 6))
    output.square().mean().backward()
    assert output.shape == (1, 14)
    assert torch.isfinite(output).all()
    assert model.input_adapter.weight.grad is not None
    assert torch.isfinite(model.input_adapter.weight.grad).all()
    assert model.input_adapter.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(not layer.training for layer in model.backbone.modules() if isinstance(layer, nn.BatchNorm2d))
