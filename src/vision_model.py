"""Two-camera pose regressor with a fine-tuned ResNet-34 and small wrist CNN."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34

from cube_yaw import normalize_yaw_tensor
from schemas import CAMERA_IMAGE_SHAPES
from schemas.config import VisionModelConfig


class VisionModel(nn.Module):
    """Map wrist/external RGB images and joint positions to two cube poses.

    Images are float32 NCHW tensors in [0, 1]: wrist 128x128, external 300x300.
    The external image uses ResNet-34 and the wrist image uses its own small CNN.
    The 10 raw outputs are orange XYZ/yaw then blue XYZ/yaw. XYZ is in meters;
    each yaw is [cos(4*yaw), sin(4*yaw)] before shared loss/runtime normalization.
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
            weights = None if config.backbone_weights is None else ResNet34_Weights[config.backbone_weights]
            backbone = resnet34(weights=weights)
            backbone.fc = nn.Identity()
        self.backbone = backbone
        self.backbone.requires_grad_(True)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        wrist_layers: list[nn.Module] = []
        channels = 3
        for output_channels in config.wrist_cnn.channels:
            wrist_layers.extend((nn.Conv2d(channels, output_channels, kernel_size=3,
                                          stride=2, padding=1), nn.ReLU()))
            channels = output_channels
        wrist_layers.extend((nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                             nn.Linear(channels * 4 * 4, config.wrist_cnn.embedding_dim)))
        self.wrist_encoder = nn.Sequential(*wrist_layers)

        modules: list[nn.Module] = []
        width = config.wrist_cnn.embedding_dim + 512 + 6
        for _ in range(config.mlp_hidden_layers):
            modules.extend((nn.Linear(width, config.mlp_hidden_dim), nn.ReLU(),
                            nn.Dropout(p=config.dropout)))
            width = config.mlp_hidden_dim
        modules.append(nn.Linear(width, 10))
        self.pose_head = nn.Sequential(*modules)

    @staticmethod
    def _validate_image(image: Tensor, camera_id: str) -> None:
        height, width, channels = CAMERA_IMAGE_SHAPES[camera_id]
        if image.ndim != 4 or tuple(image.shape[1:]) != (channels, height, width):
            raise ValueError(f"{camera_id} images must have shape [batch, {channels}, {height}, {width}]; "
                             f"got {tuple(image.shape)}.")

    def encode_external_image(self, image: Tensor) -> Tensor:
        self._validate_image(image, "external")
        image = self.input_adapter(image)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - self.image_mean) / self.image_std
        return self.backbone(image)

    def encode_wrist_image(self, image: Tensor) -> Tensor:
        self._validate_image(image, "wrist")
        return self.wrist_encoder(image)

    def forward(self, wrist_images: Tensor, external_images: Tensor, joint_positions: Tensor) -> Tensor:
        if joint_positions.ndim != 2 or joint_positions.shape[1] != 6:
            raise ValueError("joint_positions must have shape [batch, 6].")
        wrist = self.encode_wrist_image(wrist_images)
        external = self.encode_external_image(external_images)
        return self.pose_head(torch.cat((wrist, external, joint_positions), dim=-1))


def cube_pose_loss_components(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Return unreduced position/yaw MSE, each shaped [..., 2] per cube.

    Normalize predicted and target yaw pairs with the same rule used by live
    action observations. The fourth-angle targets already identify quarter-turn
    equivalents; negating a yaw pair is a different orientation (45 degrees).
    """
    if prediction.shape != target.shape or prediction.ndim == 0 or prediction.shape[-1] != 10:
        raise ValueError("Cube pose predictions and labels must have matching [..., 10] shapes.")
    position_errors, orientation_errors = [], []
    for offset in (0, 5):
        position_error = (prediction[..., offset:offset + 3] - target[..., offset:offset + 3]).square().mean(-1)
        yaw = normalize_yaw_tensor(prediction[..., offset + 3:offset + 5])
        expected = normalize_yaw_tensor(target[..., offset + 3:offset + 5])
        orientation_error = (yaw - expected).square().mean(-1)
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)
    return torch.stack(position_errors, dim=-1), torch.stack(orientation_errors, dim=-1)


def cube_pose_loss(prediction: Tensor, target: Tensor,
                   position_weight: float = 200.0, orientation_weight: float = 1.0) -> Tensor:
    """Weighted position and normalized fourth-angle yaw MSE, averaged per cube."""
    position_error, orientation_error = cube_pose_loss_components(prediction, target)
    return (position_weight * position_error + orientation_weight * orientation_error).mean()
