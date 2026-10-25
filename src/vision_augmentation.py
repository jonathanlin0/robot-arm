"""Fresh RandAugment without solarization for saved two-camera training samples."""

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import RandAugment


class RandAugmentWithoutSolarize(RandAugment):
    """Keep torchvision's augmentation choices except intensity inversion."""

    def _augmentation_space(self, num_bins: int, image_size: tuple[int, int]):
        operations = super()._augmentation_space(num_bins, image_size)
        operations.pop("Solarize", None)
        return operations


class AugmentedVisionDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Augment each view on access; retain the original stored images and labels."""

    def __init__(self, dataset: TensorDataset, transform: RandAugment) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def _augment_image(self, image: Tensor) -> Tensor:
        if self.transform.num_ops == 0:
            return image
        if image.dtype == torch.uint8:
            return self.transform(image)
        # RandAugment requires byte images. Float shards remain float32 [0, 1]
        # after augmentation, so the training loop does not scale them twice.
        pixels = image.clamp(0, 1).mul(255).round().to(dtype=torch.uint8)
        return self.transform(pixels).to(dtype=image.dtype).div(255)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        wrist, external, joints, poses = self.dataset[index]
        return self._augment_image(wrist), self._augment_image(external), joints, poses
