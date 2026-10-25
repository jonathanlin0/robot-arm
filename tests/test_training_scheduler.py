"""Vision learning-rate changes use fresh validation loss and apply next epoch."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import training
from schemas.config import load_config


ROOT = Path(__file__).parents[1]


class SilentProgress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


@pytest.fixture
def run_stubbed_training(monkeypatch, tmp_path):
    """Exercise the real optimizer/scheduler without model training or datasets."""
    monkeypatch.setattr(training, "tqdm", SilentProgress)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: None)

    def run(config, losses, model_type="vision"):
        loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
        validation_losses = iter(losses)
        training_rates, validation_epochs = [], []

        def run_epoch(*, optimizer, on_batch, **kwargs):
            if optimizer is not None:
                training_rates.append(optimizer.param_groups[0]["lr"])
                # Continuously improving training loss must not prevent a
                # reduction driven by stalled validation loss.
                loss = 1 / len(training_rates)
            else:
                validation_epochs.append(len(training_rates))
                loss = next(validation_losses)
                if model_type == "vision":
                    # This metric also improves; the scheduler must use total
                    # validation loss rather than cube-position distance.
                    kwargs["on_validation_position_error"](1 / len(validation_epochs))
            on_batch(loss)
            return loss

        monkeypatch.setattr(training, f"run_{model_type}_epoch", run_epoch)
        log_path = tmp_path / f"{model_type}.log"
        metrics = training._train(
            model=nn.Linear(1, 1), model_type=model_type, loaders=(loader, loader),
            config=config, device=torch.device("cpu"), log_file=log_path,
        )
        assert list(validation_losses) == []
        log_rows = [dict(field.split("=", 1) for field in line.split())
                    for line in log_path.read_text(encoding="utf-8").splitlines()]
        return training_rates, validation_epochs, metrics, log_rows

    return run


def vision_config(*, epochs, validation_interval):
    config = load_config(ROOT / "configs/default.json").training.vision
    config.epochs = epochs
    config.validation_interval = validation_interval
    return config


def test_plateau_uses_only_fresh_validation_and_improvement_resets_patience(run_stubbed_training):
    config = vision_config(epochs=15, validation_interval=2)
    rates, checked_epochs, metrics, logs = run_stubbed_training(
        config, [0.5, 0.49, 0.4901, 0.48, 0.4801, 0.4802, 0.4802, 0.4802],
    )

    # Epoch 8 improves after one bad check and restarts patience. Two bad
    # checks at 10 and 12 then reduce the rate for epoch 13, not epoch 12.
    assert checked_epochs == [2, 4, 6, 8, 10, 12, 14, 15]
    assert rates == pytest.approx([1e-4] * 12 + [5e-5] * 3)
    assert [row["learning_rate"] for row in metrics] == pytest.approx([1e-4] * 6 + [5e-5] * 2)
    assert [row["next_learning_rate"] for row in metrics] == pytest.approx(
        [1e-4] * 5 + [5e-5] * 2 + [2.5e-5],
    )
    assert [int(row["epoch"]) for row in logs] == list(range(16))
    assert [float(row["learning_rate"]) for row in logs] == pytest.approx([1e-4] + rates)
    assert [float(row["next_learning_rate"]) for row in logs] == pytest.approx(
        [1e-4] * 12 + [5e-5] * 3 + [2.5e-5],
    )
    assert logs[13]["val_loss"] == "NaN"
    # The final off-interval validation is still a real check and may reduce
    # the logged next rate, even though this run has no following epoch.
    assert metrics[-1]["epoch"] == 15
    assert float(logs[-1]["next_learning_rate"]) == pytest.approx(2.5e-5)


def test_improvement_must_exceed_relative_threshold(run_stubbed_training):
    config = vision_config(epochs=7, validation_interval=2)
    rates, _, metrics, _ = run_stubbed_training(config, [0.5, 0.49975, 0.4996, 0.4996])

    # Both small improvements are less than 0.1% of the best value, so they
    # count as two stalled checks and trigger a reduction after epoch 6.
    assert rates == pytest.approx([1e-4] * 6 + [5e-5])
    assert metrics[2]["next_learning_rate"] == pytest.approx(5e-5)


def test_rate_never_falls_below_configured_floor(run_stubbed_training):
    config = vision_config(epochs=7, validation_interval=1)
    config.learning_rate = 1.6e-6
    rates, _, metrics, _ = run_stubbed_training(config, [1.0] * 7)

    assert rates == pytest.approx([1.6e-6] * 3 + [1e-6] * 4)
    assert [row["next_learning_rate"] for row in metrics] == pytest.approx([1.6e-6] * 2 + [1e-6] * 5)


def test_disabled_scheduler_keeps_learning_rate_fixed(monkeypatch, run_stubbed_training):
    config = vision_config(epochs=5, validation_interval=1)
    config.lr_scheduler = None

    def unexpected_scheduler(*args, **kwargs):
        raise AssertionError("Disabled scheduling must not construct a scheduler.")

    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", unexpected_scheduler)
    rates, _, metrics, _ = run_stubbed_training(config, [1.0, 2.0, 3.0, 4.0, 5.0])

    assert rates == pytest.approx([config.learning_rate] * 5)
    assert [row["next_learning_rate"] for row in metrics] == pytest.approx(rates)


def test_action_training_does_not_construct_vision_scheduler(monkeypatch, run_stubbed_training):
    config = load_config(ROOT / "configs/default.json").training.action
    config.epochs = 5
    config.validation_interval = 1

    def unexpected_scheduler(*args, **kwargs):
        raise AssertionError("Action training must retain its fixed learning rate.")

    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", unexpected_scheduler)
    rates, _, metrics, logs = run_stubbed_training(config, [1.0] * 5, model_type="action")

    assert rates == pytest.approx([config.learning_rate] * 5)
    assert all(set(row) == {"epoch", "training_loss", "validation_loss"} for row in metrics)
    assert all(set(row) == {"epoch", "train_loss", "val_loss"} for row in logs)
