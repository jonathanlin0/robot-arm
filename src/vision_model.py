"""Shared two-camera pose regressor with a frozen ResNet-50 feature encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

from schemas.config import VisionModelConfig


class VisionModel(nn.Module):
    """Map wrist/external RGB images and joint positions to two cube poses.

    Images are float32 NCHW tensors in [0, 1], already 300x300. Pose output
    ordering is orange XYZ/quaternion then blue XYZ/quaternion, in meters and
    (w, x, y, z). No runtime clipping or plausibility checks alter predictions.
    """

    def __init__(self, config: VisionModelConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        kernel = config.adapter_kernel_size
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("adapter_kernel_size must be a positive odd integer.")
        self.input_adapter = nn.Conv2d(3, 3, kernel_size=kernel, padding=kernel // 2)
        # Begin with unchanged RGB values; training can learn a useful adapter.
        nn.init.dirac_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        if backbone is None:
            weights = None if config.backbone_weights is None else ResNet50_Weights[config.backbone_weights]
            backbone = resnet50(weights=weights)
            backbone.fc = nn.Identity()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        modules: list[nn.Module] = []
        width = 2048 * 2 + 6
        for _ in range(config.mlp_hidden_layers):
            modules.extend((nn.Linear(width, config.mlp_hidden_dim), nn.ReLU(),
                            nn.Dropout(p=config.dropout)))
            width = config.mlp_hidden_dim
        modules.append(nn.Linear(width, 14))
        self.pose_head = nn.Sequential(*modules)

    def train(self, mode: bool = True) -> VisionModel:
        super().train(mode)
        # Frozen backbone includes BatchNorm statistics, even while the adapter
        # and pose MLP train. Autograd still passes through it to the adapter.
        self.backbone.eval()
        return self

    def encode_image(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 300, 300):
            raise ValueError("Vision images must have shape [batch, 3, 300, 300].")
        image = self.input_adapter(image)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - self.image_mean) / self.image_std
        # Do not use no_grad here: the trainable input adapter needs gradients
        # through the frozen ResNet computation.
        return self.backbone(image)

    def forward(self, wrist_images: Tensor, external_images: Tensor, joint_positions: Tensor) -> Tensor:
        if joint_positions.ndim != 2 or joint_positions.shape[1] != 6:
            raise ValueError("joint_positions must have shape [batch, 6].")
        wrist = self.encode_image(wrist_images)
        external = self.encode_image(external_images)
        return self.pose_head(torch.cat((wrist, external, joint_positions), dim=-1))


def cube_pose_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Position MSE plus sign-invariant unit-quaternion MSE, averaged per cube.

    q and -q describe the same orientation. Normalization is confined to the
    loss; inference returns the network's raw 14 values.
    """
    if prediction.shape != target.shape or prediction.shape[-1] != 14:
        raise ValueError("Cube pose predictions and labels must have matching [..., 14] shapes.")
    losses = []
    for offset in (0, 7):
        position_error = (prediction[..., offset:offset + 3] - target[..., offset:offset + 3]).square().mean(-1)
        quaternion = F.normalize(prediction[..., offset + 3:offset + 7], dim=-1, eps=1e-6)
        expected = F.normalize(target[..., offset + 3:offset + 7], dim=-1, eps=1e-6)
        orientation_error = torch.minimum((quaternion - expected).square().mean(-1),
                                          (quaternion + expected).square().mean(-1))
        losses.append(position_error + orientation_error)
    return torch.stack(losses, dim=-1).mean()
