"""Image augmentation preserves pose labels and is confined to training access."""

from pathlib import Path

import pytest
from pydantic import ValidationError
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.transforms import RandAugment

from schemas.config import RandAugmentConfig, load_config
from vision_augmentation import AugmentedVisionDataset, RandAugmentWithoutSolarize


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def image_dataset(dtype):
    wrist = torch.arange(2 * 3 * 128 * 128).reshape(2, 3, 128, 128).remainder(256).to(torch.uint8)
    external = torch.arange(2 * 3 * 300 * 300).reshape(2, 3, 300, 300).remainder(256).to(torch.uint8)
    if dtype == torch.float32:
        wrist, external = wrist.float() / 255, external.float() / 255
    return TensorDataset(wrist, external.flip(-1), torch.arange(12).reshape(2, 6).float(),
                         torch.arange(20).reshape(2, 10).float())


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_each_access_augments_both_cameras_independently_without_changing_sources(dtype):
    class CountedTransform(RandAugment):
        def __init__(self):
            super().__init__(num_ops=2, magnitude=9)
            self.calls = []

        def forward(self, image):
            assert image.dtype == torch.uint8
            self.calls.append(image.clone())
            return torch.full_like(image, 20 * len(self.calls))

    source = image_dataset(dtype)
    originals = tuple(tensor.clone() for tensor in source.tensors)
    transform = CountedTransform()
    dataset = AugmentedVisionDataset(source, transform)

    first, second = dataset[0], dataset[0]

    assert len(dataset) == len(source)
    assert len(transform.calls) == 4
    scale = 255 if dtype == torch.float32 else 1
    for index, (image, value) in enumerate(zip((*first[:2], *second[:2]), (20, 40, 60, 80))):
        assert image.shape == source.tensors[index % 2][0].shape
        assert image.dtype == dtype
        torch.testing.assert_close(image, torch.full_like(image, value / scale))
    for index, received in enumerate(transform.calls):
        expected = originals[index % 2][0]
        if dtype == torch.float32:
            expected = expected.mul(255).round().to(torch.uint8)
        torch.testing.assert_close(received, expected)
    for actual, original in zip(source.tensors, originals):
        torch.testing.assert_close(actual, original)
    for sample in (first, second):
        torch.testing.assert_close(sample[2], originals[2][0])
        torch.testing.assert_close(sample[3], originals[3][0])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_real_augmentation_retains_image_dtype_shape_and_range(dtype):
    source = image_dataset(dtype)
    dataset = AugmentedVisionDataset(source, RandAugmentWithoutSolarize(num_ops=2, magnitude=9))
    with torch.random.fork_rng():
        torch.manual_seed(21)
        for index in range(len(dataset)):
            for view, image in enumerate(dataset[index][:2]):
                assert image.dtype == dtype
                assert image.shape == source.tensors[view][index].shape
                assert torch.isfinite(image).all()
                assert image.min() >= 0
                assert image.max() <= (1 if dtype == torch.float32 else 255)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_zero_operations_preserves_images_exactly(dtype):
    source = image_dataset(dtype)
    if dtype == torch.float32:
        # Values between byte levels must survive disabling augmentation.
        source.tensors[0].fill_(0.123456)
    dataset = AugmentedVisionDataset(source, RandAugmentWithoutSolarize(num_ops=0, magnitude=9))
    for actual, expected in zip(dataset[0], source[0]):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("field,value", [("num_ops", -1), ("num_ops", "2"),
                                        ("magnitude", -1), ("magnitude", 31), ("magnitude", 5.5)])
def test_invalid_randaugment_settings_are_rejected(field, value):
    settings = {"num_ops": 2, "magnitude": 9, field: value}
    with pytest.raises(ValidationError, match=field):
        RandAugmentConfig.model_validate(settings)


@pytest.mark.parametrize("settings", [None, {"num_ops": 1, "magnitude": 13}])
def test_vision_entrypoint_augments_training_only_and_honors_configuration(tmp_path, monkeypatch, settings):
    import training

    project = load_config(Path(__file__).parents[1] / "configs" / "default.json")
    config = project.training.vision
    assert config.randaugment.num_ops == 2
    assert config.randaugment.magnitude == 7
    if settings is not None:
        config.randaugment = RandAugmentConfig.model_validate(settings)
    config.data_dir = tmp_path / "data"
    config.device, config.num_workers, config.batch_size = "cpu", 0, 2
    config.shuffle_train = False
    record = {
        "wrist_images": torch.full((2, 128, 128, 3), 30, dtype=torch.uint8),
        "external_images": torch.full((2, 300, 300, 3), 70, dtype=torch.uint8),
        "joint_positions": torch.arange(12).reshape(2, 6).float(),
        "cube_poses": torch.arange(20).reshape(2, 10).float(),
    }
    for split in ("train", "validation"):
        directory = config.data_dir / split
        directory.mkdir(parents=True)
        torch.save(record, directory / "shard.pt")

    calls = []

    def augment(transform, image):
        calls.append((transform.num_ops, transform.magnitude))
        return 255 - image

    def inspect_training(*, model, model_type, loaders, config, device, log_file=None):
        assert model_type == "vision"
        assert type(loaders[0].dataset.transform) is RandAugmentWithoutSolarize
        train_batch = next(iter(loaders[0]))
        assert torch.all(train_batch[0] == 225)
        assert torch.all(train_batch[1] == 185)
        for _ in range(2):
            validation_batch = next(iter(loaders[1]))
            assert torch.all(validation_batch[0] == 30)
            assert torch.all(validation_batch[1] == 70)
            for batch in (train_batch, validation_batch):
                torch.testing.assert_close(batch[2], record["joint_positions"])
                torch.testing.assert_close(batch[3], record["cube_poses"])
        assert calls == [(config.randaugment.num_ops, config.randaugment.magnitude)] * 4
        return [{"epoch": 1, "training_loss": 0.1, "validation_loss": 0.2}]

    monkeypatch.setattr(RandAugment, "forward", augment)
    monkeypatch.setattr(training, "VisionModel", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(training, "_train", inspect_training)
    assert training.train_vision(project)[0]["validation_loss"] == 0.2
