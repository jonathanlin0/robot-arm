"""Training schedules learned-vision rollout checks without running a simulator."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_evaluation import ActionRolloutMetrics
from action_policy import ActionPolicy
from schemas.config import load_config
import training


ROOT = Path(__file__).parents[1]


class RecordingProgress:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.unit = kwargs["unit"]
        self.postfix = dict(kwargs.get("postfix", {}))
        self.postfixes = [dict(self.postfix)]
        self.n = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self.postfix = dict(ordered_dict or {}, **kwargs)
        self.postfixes.append(dict(self.postfix))

    def update(self, n=1):
        self.n += n


def install_progress(monkeypatch):
    bars = []

    def create_progress(**kwargs):
        bar = RecordingProgress(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(training, "tqdm", create_progress)
    return bars


def tiny_loader():
    return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


@pytest.mark.parametrize("epochs", [5, 6])
def test_rollouts_follow_every_validation_and_logs_mark_unmeasured_epochs(monkeypatch, tmp_path, epochs):
    config = load_config(ROOT / "configs/default.json").training.action
    config.epochs, config.validation_interval = epochs, 2
    config.checkpoint_path = tmp_path / "action.pt"
    assert config.rollout_validation.episodes == 200
    path = tmp_path / "nested" / "action.log"
    loader = tiny_loader()
    bars = install_progress(monkeypatch)
    checkpoints, events, training_postfixes = [], [], []
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: checkpoints.append(kwargs))
    epoch = 0

    def run_epoch(*, optimizer, on_batch, **kwargs):
        nonlocal epoch
        if optimizer is not None:
            epoch += 1
            events.append((epoch, "train"))
            loss = epoch / 10
            # Completed records are flushed before the next epoch begins.
            assert len(path.read_text(encoding="utf-8").splitlines()) == epoch
            on_batch(loss)
            training_postfixes.append(dict(bars[0].postfix))
        else:
            events.append((epoch, "tensor validation"))
            loss = epoch / 100
            on_batch(loss)
        return loss

    def evaluate(*, on_episode):
        assert events[-1] == (epoch, "tensor validation")
        events.append((epoch, "rollouts"))
        success_count = epoch * 10
        for attempted in range(1, 201):
            successes = min(attempted, success_count)
            ik_failures = min(max(attempted - success_count, 0), 10)
            result = ActionRolloutMetrics(
                episodes=attempted, successes=successes, ik_failures=ik_failures,
                timeouts=attempted - successes - ik_failures,
            )
            on_episode(result)
        return result

    monkeypatch.setattr(training, "run_action_epoch", run_epoch)
    metrics = training._train(model=nn.Linear(1, 1), model_type="action", loaders=(loader, loader),
                              config=config, device=torch.device("cpu"), log_file=path,
                              rollout_evaluator=evaluate)

    validation_epochs = [value for value in range(1, epochs + 1)
                         if value % 2 == 0 or value == epochs]
    assert [value for value, phase in events if phase == "rollouts"] == validation_epochs
    assert [row["epoch"] for row in metrics] == validation_epochs
    for row, value in zip(metrics, validation_epochs, strict=True):
        assert row == {
            "epoch": value, "training_loss": value / 10, "validation_loss": value / 100,
            "validation_success_rate_percent": value * 5.0, "validation_rollouts": 200,
            "validation_successes": value * 10, "validation_ik_failures": 10,
            "validation_timeouts": 190 - value * 10,
        }

    records = [dict(field.split("=", 1) for field in line.split())
               for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == epochs + 1
    for value, record in enumerate(records):
        assert int(record["epoch"]) == value
        if value in validation_epochs:
            assert float(record["val_loss"]) == pytest.approx(value / 100)
            assert float(record["val_success_rate_pct"]) == pytest.approx(value * 5.0)
        else:
            assert record["val_loss"] == record["val_success_rate_pct"] == "NaN"
    assert bars[0].postfixes[0]["val_success_rate_pct"] == "NaN"
    for value, postfix in enumerate(training_postfixes, 1):
        previous_checks = [check for check in validation_epochs if check < value]
        if previous_checks:
            assert float(postfix["val_success_rate_pct"]) == previous_checks[-1] * 5.0
        else:
            assert postfix["val_success_rate_pct"] == "NaN"
    assert float(bars[0].postfix["val_success_rate_pct"]) == epochs * 5.0
    rollout_bars = [bar for bar in bars if bar.unit == "episode"]
    assert len(rollout_bars) == len(validation_epochs)
    assert all(bar.total == bar.n == 200 and bar.closed for bar in rollout_bars)
    assert len(checkpoints) == 1
    assert checkpoints[0]["validation_rollout_metrics"] == {
        "episodes": 200, "successes": epochs * 10, "ik_failures": 10,
        "timeouts": 190 - epochs * 10, "success_rate_percent": epochs * 5.0,
    }


def test_train_action_loads_and_freezes_vision_once_and_only_optimizes_action(monkeypatch, tmp_path):
    project = load_config(ROOT / "configs/default.json")
    config = project.training.action
    config.epochs, config.validation_interval, config.device = 3, 2, "cpu"
    config.checkpoint_path = tmp_path / "action.pt"
    project.training.vision.checkpoint_path = tmp_path / "custom-vision.pt"
    # Verify that custom scene settings survive the entrypoint, not just defaults.
    config.rollout_validation.seed = 123
    config.rollout_validation.cube_x_range = (0.2, 0.4)
    config.rollout_validation.success.required_stable_time = 0.5
    loader = tiny_loader()
    action = nn.Linear(2, 4)
    vision = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5)).train()
    vision_loads, evaluations, optimizers, dataset_loads, action_configs = [], [], [], [], []
    install_progress(monkeypatch)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: None)

    def load_dataset(path, **kwargs):
        dataset_loads.append((path, kwargs))
        return loader.dataset

    def build_action(*, config):
        action_configs.append(config)
        return action

    def load_vision(path, device):
        vision_loads.append((path, device))
        return vision

    def run_epoch(*, model, optimizer, on_batch, **kwargs):
        assert model is action
        assert not vision.training and all(not module.training for module in vision.modules())
        assert all(not parameter.requires_grad for parameter in vision.parameters())
        if optimizer is not None:
            optimizers.append(optimizer)
            actual_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            assert actual_parameters == {id(parameter) for parameter in action.parameters()}
            assert actual_parameters.isdisjoint({id(parameter) for parameter in vision.parameters()})
        on_batch(0.1)
        return 0.1

    def evaluate(*, project: object, action_policy, vision_model, device, on_episode):
        evaluations.append((project, action_policy, vision_model, device))
        assert not vision_model.training
        for attempted in range(1, 201):
            result = ActionRolloutMetrics(episodes=attempted, successes=attempted)
            on_episode(result)
        return result

    monkeypatch.setattr(training, "load_action_dataset", load_dataset)
    monkeypatch.setattr(training, "create_loaders", lambda **kwargs: (loader, loader))
    monkeypatch.setattr(training, "ActionPolicy", build_action)
    monkeypatch.setattr(training, "load_vision_checkpoint", load_vision)
    monkeypatch.setattr(training, "run_action_epoch", run_epoch)
    monkeypatch.setattr(training, "evaluate_action_rollouts", evaluate)

    log_path = tmp_path / "action.log"
    results = training.train_action(project=project, log_file=log_path)

    assert vision_loads == [(project.training.vision.checkpoint_path, torch.device("cpu"))]
    assert action_configs == [project.action_policy]
    assert len(optimizers) == 3 and all(optimizer is optimizers[0] for optimizer in optimizers)
    assert evaluations == [(project, action, vision, torch.device("cpu"))] * 2
    assert all(item[0] is project for item in evaluations)
    assert [row["epoch"] for row in results] == [2, 3]
    assert all(row["validation_rollouts"] == 200 for row in results)
    assert [path for path, _ in dataset_loads] == [config.data_dir / "train", config.data_dir / "validation"]
    assert all(options == {"history_length": project.action_policy.history_length,
                           "stride": config.stride, "show_progress": True}
               for _, options in dataset_loads)
    assert "val_success_rate_pct=100.000000" in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("error", [FileNotFoundError("vision checkpoint missing"),
                                  ValueError("vision checkpoint incompatible")])
def test_unusable_vision_checkpoint_fails_before_training_epochs(monkeypatch, error):
    project = load_config(ROOT / "configs/default.json")
    project.training.action.device = "cpu"
    loader = tiny_loader()
    monkeypatch.setattr(training, "load_action_dataset", lambda *args, **kwargs: loader.dataset)
    monkeypatch.setattr(training, "create_loaders", lambda **kwargs: (loader, loader))

    def fail_load(*args, **kwargs):
        raise error

    def unexpected_training(**kwargs):
        pytest.fail("No training epoch may run without a usable vision checkpoint.")

    monkeypatch.setattr(training, "load_vision_checkpoint", fail_load)
    monkeypatch.setattr(training, "_train", unexpected_training)
    with pytest.raises(type(error), match=str(error)):
        training.train_action(project=project)


@pytest.mark.parametrize("failure", ["short evaluation", "renderer error"])
def test_failed_or_incomplete_rollout_check_is_not_saved_as_completed_validation(monkeypatch, tmp_path, failure):
    config = load_config(ROOT / "configs/default.json").training.action
    config.epochs = 1
    loader = tiny_loader()
    checkpoints = []
    install_progress(monkeypatch)
    monkeypatch.setattr(training, "run_action_epoch", lambda **kwargs: 0.1)
    monkeypatch.setattr(training, "save_model_checkpoint", lambda **kwargs: checkpoints.append(kwargs))

    def evaluate(**kwargs):
        if failure == "renderer error":
            raise RuntimeError("renderer failed")
        return ActionRolloutMetrics(episodes=199, successes=199)

    error = RuntimeError if failure == "renderer error" else ValueError
    match = "renderer failed" if failure == "renderer error" else "configured number of episodes"
    path = tmp_path / "action.log"
    with pytest.raises(error, match=match):
        training._train(model=nn.Linear(1, 1), model_type="action", loaders=(loader, loader),
                        config=config, device=torch.device("cpu"), log_file=path,
                        rollout_evaluator=evaluate)
    assert checkpoints == []
    assert path.read_text(encoding="utf-8").splitlines() == [
        "epoch=0 train_loss=NaN val_loss=NaN val_success_rate_pct=NaN",
    ]


def test_checkpoint_persists_rollout_metrics_without_changing_action_reload(tmp_path):
    config = load_config(ROOT / "configs/default.json").action_policy
    config = config.model_copy(update={
        "history_length": 2,
        "transformer": config.transformer.model_copy(update={
            "embedding_dim": 8, "feedforward_dim": 8, "heads": 2, "layers": 1,
        }),
        "mlp_hidden_dims": (4,),
    })
    model = ActionPolicy(config).eval()
    path = tmp_path / "action.pt"
    rollout_metrics = {"episodes": 200, "successes": 37, "ik_failures": 13,
                       "timeouts": 150, "success_rate_percent": 18.5}
    training.save_model_checkpoint(model=model, model_type="action", checkpoint_path=path,
                                   epoch=10, validation_loss=0.2,
                                   validation_rollout_metrics=rollout_metrics)

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["validation_rollout_metrics"] == rollout_metrics
    reloaded = training.load_action_checkpoint(path, device="cpu")
    assert reloaded.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value)


@pytest.mark.parametrize("log_option", [None, "--log", "--log-file"])
def test_action_cli_forwards_default_or_explicit_log_path(monkeypatch, tmp_path, log_option):
    script = ROOT / "scripts/train_action.py"
    spec = importlib.util.spec_from_file_location("train_action_logging_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, calls = object(), []
    monkeypatch.setattr(module, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(module, "train_action", lambda **kwargs: calls.append(kwargs))
    arguments = [str(script)]
    log_path = tmp_path / "training logs" / "action.log" if log_option else Path("logs/action.log")
    if log_option:
        arguments.extend([log_option, str(log_path)])
    monkeypatch.setattr(sys, "argv", arguments)

    module.main()

    assert calls == [{"project": config, "log_file": log_path}]
