"""Fine-tuned external backbone, configurable wrist CNN, and pose-loss contracts."""

from math import cos, pi, sin

import numpy as np
import pytest
import torch
from torch import nn

from schemas import CAMERA_IMAGE_SHAPES
from schemas.config import VisionModelConfig, WristCNNConfig
from vision_model import VisionModel, cube_pose_loss, cube_pose_loss_components


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
        self.projection = nn.Linear(3, 512)
        self.calls = 0
        self.last_image = None

    def forward(self, image):
        self.calls += 1
        self.last_image = image.detach().clone()
        return self.projection(self.norm(image).mean(dim=(2, 3)))


def config():
    return VisionModelConfig(adapter_kernel_size=3, backbone_weights=None, mlp_hidden_layers=2,
                             mlp_hidden_dim=16, dropout=0.1,
                             wrist_cnn=WristCNNConfig(layers=2, channels=(16, 16), embedding_dim=128))


def images(batch=2, camera_id="wrist"):
    height, width, channels = CAMERA_IMAGE_SHAPES[camera_id]
    return torch.rand(batch, channels, height, width)


def test_both_encoders_adapter_and_head_receive_gradients():
    torch.manual_seed(8)
    backbone = TinyBackbone()
    model = VisionModel(config=config(), backbone=backbone).train()
    wrist, external = images(), images(camera_id="external")
    before = backbone.norm.running_mean.clone()
    prediction = model(wrist, external, torch.randn(2, 6))
    assert prediction.shape == (2, 10)
    prediction.square().mean().backward()
    assert backbone.calls == 1
    assert backbone.training and backbone.norm.training
    assert all(p.requires_grad and p.grad is not None for p in backbone.parameters())
    assert backbone.projection.weight.grad.abs().sum() > 0
    assert model.wrist_encoder[0].weight.grad.abs().sum() > 0
    assert model.wrist_encoder[-1].weight.grad.abs().sum() > 0
    assert model.input_adapter.weight.grad.abs().sum() > 0
    assert model.pose_head[0].weight.grad.abs().sum() > 0
    assert not torch.equal(before, backbone.norm.running_mean)
    model.eval().train()
    assert backbone.training


def test_batch_norm_tracks_training_batches_and_stops_in_eval():
    model = VisionModel(config=config(), backbone=TinyBackbone()).train()
    norm = model.backbone.norm
    model(images(), images(camera_id="external"), torch.zeros(2, 6))
    assert norm.num_batches_tracked.item() == 1
    before = norm.running_mean.clone()
    model.eval()
    with torch.inference_mode():
        model(images(), images(camera_id="external"), torch.zeros(2, 6))
    torch.testing.assert_close(norm.running_mean, before, rtol=0, atol=0)
    assert norm.num_batches_tracked.item() == 1
    assert not model.wrist_encoder.training and not model.backbone.training


def test_fusion_order_is_wrist_then_external_then_joint_positions():
    model = VisionModel(config=config(), backbone=TinyBackbone()).eval()
    wrist, external, joints = images(1), images(1, "external"), torch.arange(6.).reshape(1, 6)
    fused = []
    handle = model.pose_head.register_forward_pre_hook(lambda _, args: fused.append(args[0]))
    model(wrist, external, joints)
    handle.remove()
    torch.testing.assert_close(fused[0][:, :128], model.encode_wrist_image(wrist))
    torch.testing.assert_close(fused[0][:, 128:640], model.encode_external_image(external))
    torch.testing.assert_close(fused[0][:, 640:], joints)
    assert model.pose_head[0].in_features == 646
    assert model.pose_head[0].out_features == 16
    assert len([layer for layer in model.pose_head.modules() if isinstance(layer, nn.Linear)]) == 3


@pytest.mark.parametrize("channels,embedding_dim", [((8,), 32), ((4, 8, 24), 96)])
def test_wrist_cnn_configuration_controls_layers_channels_and_embedding(channels, embedding_dim):
    custom = config().model_copy(update={"wrist_cnn": WristCNNConfig(
        layers=len(channels), channels=channels, embedding_dim=embedding_dim)})
    model = VisionModel(config=custom, backbone=TinyBackbone()).eval()
    convolutions = [layer for layer in model.wrist_encoder if isinstance(layer, nn.Conv2d)]
    assert len(convolutions) == len(channels)
    assert [layer.in_channels for layer in convolutions] == [3, *channels[:-1]]
    assert [layer.out_channels for layer in convolutions] == list(channels)
    assert all(layer.kernel_size == (3, 3) and layer.stride == (2, 2) for layer in convolutions)
    assert model.wrist_encoder[-1].in_features == channels[-1] * 16
    assert model.encode_wrist_image(images()).shape == (2, embedding_dim)
    assert model.pose_head[0].in_features == embedding_dim + 512 + 6


def test_hidden_dropout_masks_and_scales_during_training_but_is_identity_in_eval():
    torch.manual_seed(17)
    model = VisionModel(config=config(), backbone=TinyBackbone()).train()
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    hidden_linears = [layer for layer in model.pose_head.modules() if isinstance(layer, nn.Linear)][:-1]
    assert len(dropout_layers) == len(hidden_linears) == model.config.mlp_hidden_layers
    assert all(layer.p == model.config.dropout and layer.training for layer in dropout_layers)
    assert model.backbone.training
    # Known positive activations distinguish dropout zeros from ReLU zeros.
    with torch.no_grad():
        for layer in hidden_linears:
            layer.weight.zero_()
            layer.bias.fill_(1.0)
    activations = []
    handles = [layer.register_forward_hook(
        lambda _, inputs, output: activations.append((inputs[0].detach().clone(), output.detach().clone()))
    ) for layer in dropout_layers]
    features = torch.ones(64, 646)
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
    features = torch.randn(8, 646)
    torch.testing.assert_close(model.pose_head(features), model.pose_head(features), rtol=0, atol=0)


def test_head_without_hidden_layers_does_not_drop_raw_pose_outputs():
    model = VisionModel(config=config().model_copy(update={"mlp_hidden_layers": 0}), backbone=TinyBackbone()).train()
    assert not any(isinstance(layer, nn.Dropout) for layer in model.modules())
    assert model.pose_head(torch.zeros(2, 646)).shape == (2, 10)


def test_only_external_image_passes_through_identity_adapter_resize_and_imagenet_normalization():
    model = VisionModel(config=config(), backbone=TinyBackbone())
    image = images(1, "external")
    torch.testing.assert_close(model.input_adapter(image), image)
    adapted = []
    handle = model.input_adapter.register_forward_hook(lambda _, args, output: adapted.append(args[0]))
    model(images(1), image, torch.zeros(1, 6))
    handle.remove()
    assert len(adapted) == model.backbone.calls == 1
    assert adapted[0].shape == (1, 3, 300, 300)
    assert model.backbone.last_image.shape == (1, 3, 224, 224)
    torch.testing.assert_close(adapted[0], image)
    resized = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    torch.testing.assert_close(model.backbone.last_image, (resized - model.image_mean) / model.image_std)
    with pytest.raises(ValueError, match=r"wrist.*128.*128"):
        model(torch.zeros(1, 3, 224, 224), image, torch.zeros(1, 6))
    with pytest.raises(ValueError, match=r"external.*300.*300"):
        model(images(1), torch.zeros(1, 3, 128, 128), torch.zeros(1, 6))


def identity_targets(dtype=torch.float32):
    return torch.tensor([[0., 0., 0., 1., 0.] * 2], dtype=dtype)


def test_position_and_orientation_components_keep_cube_order_and_leading_dimensions():
    target = identity_targets().expand(2, 3, 10).clone()
    prediction = target.clone()
    prediction[..., 0] = 0.3
    prediction[..., 5:8] = 0.6
    prediction[..., 8:10] = torch.tensor([0., 9.])
    position, orientation = cube_pose_loss_components(prediction, target)
    assert position.shape == orientation.shape == (2, 3, 2)
    torch.testing.assert_close(position[..., 0], torch.full((2, 3), 0.03))
    torch.testing.assert_close(position[..., 1], torch.full((2, 3), 0.36))
    torch.testing.assert_close(orientation[..., 0], torch.zeros(2, 3))
    torch.testing.assert_close(orientation[..., 1], torch.ones(2, 3))


def test_pose_loss_defaults_to_position_weight_200_and_orientation_weight_one():
    target = identity_targets()
    prediction = torch.tensor([[0.3, -0.6, 0.9, cos(pi / 3), sin(pi / 3)] * 2])
    position_mse = (0.3 ** 2 + 0.6 ** 2 + 0.9 ** 2) / 3
    orientation_mse = 1 - cos(pi / 3)
    assert cube_pose_loss(prediction, target).item() == pytest.approx(200 * position_mse + orientation_mse)


@pytest.mark.parametrize("position_weight,orientation_weight", [(7.5, 3.25), (0.0, 2.0), (6.0, 0.0)])
def test_pose_loss_weights_scale_corresponding_gradients(position_weight, orientation_weight):
    target = identity_targets(torch.float64)
    prediction = torch.tensor([[0.3, -0.6, 0.9, cos(pi / 3), sin(pi / 3)] * 2],
                              dtype=torch.float64).requires_grad_()
    baseline = cube_pose_loss(prediction, target, position_weight=1.0, orientation_weight=1.0)
    baseline_gradient, = torch.autograd.grad(baseline, prediction)
    weighted = cube_pose_loss(prediction, target, position_weight=position_weight,
                              orientation_weight=orientation_weight)
    weighted_gradient, = torch.autograd.grad(weighted, prediction)
    position_indices, orientation_indices = [0, 1, 2, 5, 6, 7], [3, 4, 8, 9]
    torch.testing.assert_close(weighted_gradient[:, position_indices],
                               baseline_gradient[:, position_indices] * position_weight)
    torch.testing.assert_close(weighted_gradient[:, orientation_indices],
                               baseline_gradient[:, orientation_indices] * orientation_weight)
    assert baseline_gradient[:, orientation_indices].abs().sum() > 0
    expected = position_weight * 0.42 + orientation_weight * (1 - cos(pi / 3))
    assert weighted.item() == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pose_loss_normalizes_both_predictions_and_targets_without_scaling_xyz(dtype):
    target = identity_targets(dtype)
    prediction = target.clone()
    prediction[:, 3:5] = torch.tensor([3., 4.], dtype=dtype)
    prediction[:, 8:10] = torch.tensor([-4., 3.], dtype=dtype)
    target[:, 8:10] = torch.tensor([0., 1.], dtype=dtype)
    base_position, base_orientation = cube_pose_loss_components(prediction, target)
    prediction[:, 3:5] *= 0.02
    prediction[:, 8:10] *= 7.5
    target[:, 3:5] *= 20
    target[:, 8:10] *= 0.25
    position, orientation = cube_pose_loss_components(prediction, target)
    torch.testing.assert_close(position, base_position)
    torch.testing.assert_close(orientation, base_orientation)
    torch.testing.assert_close(orientation, torch.full((1, 2), 0.4, dtype=dtype))


@pytest.mark.parametrize("quarter_turns", [-2, -1, 1, 2, 3])
def test_encoded_quarter_turn_yaws_have_zero_loss(quarter_turns):
    physical_yaw = np.deg2rad(np.array([17., -31.]))
    target = np.zeros((2, 5))
    prediction = target.copy()
    target[:, 3:] = np.stack([np.cos(4 * physical_yaw), np.sin(4 * physical_yaw)], axis=-1)
    shifted = physical_yaw + quarter_turns * pi / 2
    prediction[:, 3:] = np.stack([np.cos(4 * shifted), np.sin(4 * shifted)], axis=-1)
    loss = cube_pose_loss(torch.tensor(prediction.reshape(10)), torch.tensor(target.reshape(10)))
    assert loss.item() == pytest.approx(0., abs=1e-25)


def test_opposite_yaw_pairs_are_different_headings_not_a_quaternion_sign_ambiguity():
    target = identity_targets()
    prediction = target.clone()
    prediction[:, [3, 8]] = -1
    _, orientation = cube_pose_loss_components(prediction, target)
    torch.testing.assert_close(orientation, torch.full((1, 2), 2.))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalized_yaw_mse_matches_independent_angular_error_formula(dtype):
    generator = torch.Generator().manual_seed(29)
    prediction = torch.randn(3, 2, 5, dtype=dtype, generator=generator)
    target = torch.randn(3, 2, 5, dtype=dtype, generator=generator)
    predicted_angle = np.arctan2(prediction[..., 4].numpy(), prediction[..., 3].numpy())
    target_angle = np.arctan2(target[..., 4].numpy(), target[..., 3].numpy())
    expected = 1 - np.cos(predicted_angle - target_angle)
    _, orientation = cube_pose_loss_components(prediction.reshape(3, 10), target.reshape(3, 10))
    assert orientation.dtype == dtype
    torch.testing.assert_close(orientation, torch.tensor(expected, dtype=dtype), rtol=1e-5, atol=1e-7)


def test_zero_and_tiny_predicted_yaws_have_finite_loss_and_learning_gradients():
    target = identity_targets()
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = cube_pose_loss(prediction, target)
    loss.backward()
    assert loss.item() == pytest.approx(0.5)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert (prediction.grad[:, [3, 8]] < 0).all()
    assert (prediction.grad[:, [4, 9]] == 0).all()
    tiny = torch.zeros_like(target)
    tiny[:, [3, 8]] = 5e-8
    tiny.requires_grad_()
    tiny_loss = cube_pose_loss(tiny, target)
    tiny_loss.backward()
    assert tiny_loss.item() == pytest.approx((1 - 0.05) ** 2 / 2)
    assert torch.isfinite(tiny.grad).all() and (tiny.grad[:, [3, 8]] < 0).all()


def test_normalized_yaw_loss_passes_gradient_check():
    prediction = torch.tensor([[0.2, -0.1, 0.04, 3., 4., 0.3, 0., 0.02, -4., 3.]],
                              dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: cube_pose_loss(value, identity_targets(torch.float64)),
                                   (prediction,))


@pytest.mark.parametrize("prediction_shape,target_shape", [((14,), (14,)), ((10,), (2, 10)),
                                                          ((2, 9), (2, 9)), ((), ())])
def test_pose_loss_rejects_wrong_shapes(prediction_shape, target_shape):
    with pytest.raises(ValueError, match=r"matching.*10"):
        cube_pose_loss_components(torch.zeros(prediction_shape), torch.zeros(target_shape))


def test_runtime_outputs_are_raw_without_pose_value_checks():
    model = VisionModel(config=config(), backbone=TinyBackbone())
    with torch.no_grad():
        model.pose_head[-1].weight.zero_()
        model.pose_head[-1].bias.fill_(123.0)
        output = model(images(1), images(1, "external"), torch.zeros(1, 6))
    assert (output == 123).all()


def test_real_resnet34_forward_backward_and_optimizer_step_without_download():
    model = VisionModel(config=config().model_copy(update={"mlp_hidden_layers": 1, "mlp_hidden_dim": 8}),
                        backbone=None).train()
    assert [len(model.backbone.layer1), len(model.backbone.layer2),
            len(model.backbone.layer3), len(model.backbone.layer4)] == [3, 4, 6, 3]
    assert model.backbone.layer1[0].__class__.__name__ == "BasicBlock"
    assert isinstance(model.backbone.fc, nn.Identity)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.backbone.conv1.weight.detach().clone()
    output = model(images(1), images(1, "external"), torch.zeros(1, 6))
    output.square().mean().backward()
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
    assert model.input_adapter.weight.grad is not None
    assert torch.isfinite(model.input_adapter.weight.grad).all()
    assert model.input_adapter.weight.grad.abs().sum() > 0
    assert all(parameter.requires_grad and parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(layer.training for layer in model.backbone.modules() if isinstance(layer, nn.BatchNorm2d))
    optimizer.step()
    assert not torch.equal(before, model.backbone.conv1.weight)
