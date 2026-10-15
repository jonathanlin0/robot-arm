"""Training smoke tests and self-contained checkpoint reconstruction."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_policy import ActionPolicy
from schemas.config import ActionPolicyConfig, TransformerConfig, VisionModelConfig, load_config
from training import load_action_checkpoint, load_vision_checkpoint, run_action_epoch, run_vision_epoch, save_model_checkpoint, train_action
from vision_model import VisionModel


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def action_config():
    return ActionPolicyConfig(history_length=6,
        transformer=TransformerConfig(embedding_dim=16, feedforward_dim=24, heads=2, layers=1, dropout=0.0),
        mlp_hidden_dims=(12,), action_std=0.05, deterministic=True)


def test_action_checkpoint_recreates_architecture_and_predictions(tmp_path):
    torch.manual_seed(21)
    model = ActionPolicy(action_config()).eval()
    inputs = torch.randn(2, 5, 48)
    valid = torch.ones(2, 5, dtype=torch.bool)
    starts = torch.zeros_like(valid)
    path = tmp_path / "nested" / "action.pt"
    with torch.no_grad():
        expected = model(inputs, valid, starts)
    save_model_checkpoint(model, model_type="action", checkpoint_path=path, epoch=2, validation_loss=0.12)
    reloaded = load_action_checkpoint(path=path, device="cpu")
    assert reloaded.config == model.config
    assert not reloaded.training
    with torch.no_grad():
        torch.testing.assert_close(expected, reloaded(inputs, valid, starts))
    save_model_checkpoint(model, model_type="action", checkpoint_path=path, epoch=3, validation_loss=0.1)
    assert torch.load(path, weights_only=True)["epoch"] == 3
    with pytest.raises(ValueError, match="Expected vision"):
        load_vision_checkpoint(path=path, device="cpu")


def test_real_resnet_checkpoint_loads_without_imagenet_download(tmp_path, monkeypatch):
    config = VisionModelConfig(adapter_kernel_size=1, backbone_weights=None,
                               mlp_hidden_layers=1, mlp_hidden_dim=8, dropout=0.1)
    model = VisionModel(config=config, backbone=None).eval()
    # The saved provenance can say ImageNet without a loader needing to fetch it.
    model.config = config.model_copy(update={"backbone_weights": "IMAGENET1K_V2"})
    path = tmp_path / "vision.pt"
    save_model_checkpoint(model, model_type="vision", checkpoint_path=path, epoch=1, validation_loss=0.3)
    def no_download(*args, **kwargs):
        raise AssertionError("Loading a complete checkpoint must not download any weights.")
    monkeypatch.setattr("torch.hub.download_url_to_file", no_download)
    reloaded = load_vision_checkpoint(path=path, device="cpu")
    assert reloaded.config == model.config
    assert not reloaded.training and not reloaded.backbone.training
    dropout = [layer for layer in reloaded.pose_head if isinstance(layer, nn.Dropout)]
    assert len(dropout) == 1 and dropout[0].p == 0.1 and not dropout[0].training
    wrist, external, joints = torch.rand(1, 3, 300, 300), torch.rand(1, 3, 300, 300), torch.zeros(1, 6)
    with torch.no_grad():
        torch.testing.assert_close(model(wrist, external, joints), reloaded(wrist, external, joints))


def test_action_minibatch_training_improves_supervised_loss():
    torch.manual_seed(5)
    model = ActionPolicy(action_config())
    tokens = torch.randn(4, 5, 48)
    valid = torch.ones(4, 5, dtype=torch.bool)
    starts = torch.zeros_like(valid)
    labels = torch.ones(4, 5, 4) * 0.25
    loader = DataLoader(TensorDataset(tokens, valid, starts, labels, valid), batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = run_action_epoch(model, loader, torch.device("cpu"), optimizer=None)
    for _ in range(10):
        run_action_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)
    final = run_action_epoch(model, loader, torch.device("cpu"), optimizer=None)
    assert final < initial * 0.5


def test_action_training_entrypoint_uses_explicit_configuration(tmp_path):
    project = load_config(Path(__file__).parents[1] / "configs" / "default.json")
    project.action_policy = action_config()
    settings = project.training.action
    settings.data_dir = tmp_path / "data"
    settings.checkpoint_path = tmp_path / "checkpoints" / "trained.pt"
    settings.epochs, settings.validation_interval = 2, 1
    settings.batch_size, settings.num_workers, settings.device, settings.stride = 2, 0, "cpu", 4
    for split in ("train", "validation"):
        directory = settings.data_dir / split
        directory.mkdir(parents=True)
        torch.save({"observations": torch.randn(8, 41), "prev_accepted_targets": torch.zeros(8, 3),
                    "actions": torch.zeros(8, 4)}, directory / "episode.pt")
    metrics = train_action(project=project)
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert settings.checkpoint_path.exists()
    model = load_action_checkpoint(settings.checkpoint_path, device="cpu")
    assert model.config.history_length == 6
    assert model.config.transformer.heads == 2


def test_vision_epoch_normalizes_bytes_and_backpropagates_through_frozen_encoder():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = nn.Linear(3, 2048)
        def forward(self, image):
            return self.expand(image.mean(dim=(2, 3)))
    config = VisionModelConfig(adapter_kernel_size=1, backbone_weights=None,
                               mlp_hidden_layers=1, mlp_hidden_dim=8, dropout=0.1)
    model = VisionModel(config=config, backbone=Backbone())
    image = torch.full((1, 3, 300, 300), 255, dtype=torch.uint8)
    poses = torch.zeros(1, 14)
    poses[:, 3] = poses[:, 10] = 1
    loader = DataLoader(TensorDataset(image, image.clone(), torch.zeros(1, 6), poses), batch_size=1)
    adapter_inputs = []
    handle = model.input_adapter.register_forward_pre_hook(lambda _, args: adapter_inputs.append(args[0].detach()))
    before = model.input_adapter.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    loss = run_vision_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)
    handle.remove()
    assert loss >= 0
    assert all((image == 1.0).all() for image in adapter_inputs)
    assert not torch.equal(before, model.input_adapter.weight)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_nondefault_optimizer_and_loader_settings_are_passed_through(tmp_path, monkeypatch):
    import training
    from torch.utils.data import SequentialSampler

    config = load_config(Path(__file__).parents[1] / "configs" / "default.json").training.action
    config.learning_rate = 0.0123
    config.weight_decay = 0.0045
    config.optimizer_betas = (0.7, 0.8)
    config.optimizer_eps = 0.00002
    config.batch_size = 3
    config.shuffle_train = False
    config.pin_memory = True
    config.epochs = 1
    config.checkpoint_path = tmp_path / "nondefault.pt"
    dataset = TensorDataset(torch.zeros(7, 6, 48))
    loaders = training.create_loaders(training=dataset, validation=dataset, config=config)
    for loader in loaders:
        assert loader.batch_size == 3
        assert loader.pin_memory is True
        assert isinstance(loader.sampler, SequentialSampler)
    optimizers = []
    def record_epoch(*, model, loader, device, optimizer):
        if optimizer is not None:
            optimizers.append(optimizer)
        return 0.5
    monkeypatch.setattr(training, "run_action_epoch", record_epoch)
    training._train(model=ActionPolicy(action_config()), model_type="action", loaders=loaders,
                    config=config, device=torch.device("cpu"))
    assert len(optimizers) == 1
    defaults = optimizers[0].defaults
    assert defaults["lr"] == config.learning_rate
    assert defaults["weight_decay"] == config.weight_decay
    assert defaults["betas"] == config.optimizer_betas
    assert defaults["eps"] == config.optimizer_eps
