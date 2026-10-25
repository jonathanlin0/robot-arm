"""Standalone evaluation splits work and reports complete results without real rollouts."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from action_evaluation import ActionRolloutMetrics


ROOT = Path(__file__).parents[1]


@pytest.fixture
def cli(monkeypatch, tmp_path):
    script = ROOT / "scripts/evaluate_action_rollouts.py"
    spec = importlib.util.spec_from_file_location("evaluate_action_rollouts_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action_checkpoint = tmp_path / "action.pt"
    vision_checkpoint = tmp_path / "vision.pt"
    action_checkpoint.touch()
    vision_checkpoint.touch()
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["training"]["vision"]["checkpoint_path"] = str(vision_checkpoint)
    raw["training"]["action"]["rollout_validation"]["episodes"] = 200
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    output, bars = [], []

    class Progress:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.n = 0
            self.closed = False
            self.postfix = {}
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

        def update(self, count):
            self.n += count

        @staticmethod
        def write(message):
            output.append(message)

    monkeypatch.setattr(module, "tqdm", Progress)
    monkeypatch.setattr(sys, "argv", [str(script), "--checkpoint", str(action_checkpoint),
                                     "--config", str(config_path)])
    return SimpleNamespace(module=module, script=script, config_path=config_path,
                           action_checkpoint=action_checkpoint, vision_checkpoint=vision_checkpoint,
                           output=output, bars=bars)


def install_fake_workers(monkeypatch, cli, evaluator=None):
    """Use fresh configs per job, just as a spawned pool receives by pickling."""
    module = cli.module
    records = SimpleNamespace(pools=[], devices=[], action_loads=[], vision_loads=[], evaluations=[],
                              worker_active=False, context_names=[])
    saved_config = module.load_config(cli.config_path).action_policy.model_copy(update={"history_length": 7})

    def load_action(path, device):
        assert records.worker_active, "Action models should be loaded inside workers."
        model = torch.nn.Linear(1, 1).eval()
        model.config = saved_config.model_copy(deep=True)
        records.action_loads.append((path, device, model))
        return model

    def load_vision(path, device):
        assert records.worker_active, "Vision models should be loaded inside workers."
        model = torch.nn.Linear(1, 1).eval()
        records.vision_loads.append((path, device, model))
        return model

    def resolve_device(name):
        records.devices.append(name)
        return torch.device("cpu")

    def evaluate(**kwargs):
        records.evaluations.append(kwargs)
        assert records.worker_active
        assert kwargs["project"].action_policy is kwargs["action_policy"].config
        assert kwargs["project"].action_policy.history_length == 7
        for name in ("action_policy", "vision_model"):
            assert not kwargs[name].training
            assert all(not parameter.requires_grad for parameter in kwargs[name].parameters())
        if evaluator is not None:
            return evaluator(**kwargs)
        settings = kwargs["project"].training.action.rollout_validation
        successes = settings.seed - 43 + 1
        result = ActionRolloutMetrics(episodes=settings.episodes, successes=successes,
                                      ik_failures=1, timeouts=settings.episodes - successes - 1)
        kwargs["on_episode"](result)
        return result

    class Pool:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            self.error_type = None
            records.pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, error_type, *args):
            self.closed = True
            self.error_type = error_type

        def imap_unordered(self, worker, jobs, chunksize):
            assert worker is module.evaluate_worker
            assert chunksize == 1
            # Reversed results exercise aggregation independently of worker order.
            for index, project, checkpoint in reversed(list(jobs)):
                records.worker_active = True
                try:
                    yield worker((index, project.model_copy(deep=True), checkpoint))
                finally:
                    records.worker_active = False

    lock = object()

    def get_context(name):
        records.context_names.append(name)
        return SimpleNamespace(Pool=Pool, RLock=lambda: lock)

    monkeypatch.setattr(module, "load_action_checkpoint", load_action)
    monkeypatch.setattr(module, "load_vision_checkpoint", load_vision)
    monkeypatch.setattr(module, "resolve_device", resolve_device)
    monkeypatch.setattr(module, "evaluate_action_rollouts", evaluate)
    monkeypatch.setattr(module.mp, "get_context", get_context)
    records.lock = lock
    return records


@pytest.mark.parametrize("workers", [None, 3])
def test_default_two_workers_and_floor_split_report_actual_denominator(monkeypatch, cli, workers):
    if workers is not None:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--workers", str(workers)])
    records = install_fake_workers(monkeypatch, cli)
    original_config = cli.config_path.read_bytes()

    cli.module.main()

    count = workers or 2
    episodes = 200 // count
    total = count * episodes
    successes = sum(range(1, count + 1))
    assert cli.config_path.read_bytes() == original_config
    assert records.context_names == ["spawn"]
    assert len(records.pools) == 1 and records.pools[0].closed
    assert records.pools[0].options == {
        "processes": count, "initializer": cli.module.initialize_worker, "initargs": (records.lock,),
    }
    assert len(records.evaluations) == len(records.action_loads) == len(records.vision_loads) == count
    assert {row["project"].training.action.rollout_validation.seed for row in records.evaluations} == set(range(43, 43 + count))
    assert all(row["project"].training.action.rollout_validation.episodes == episodes for row in records.evaluations)
    assert all(path == cli.action_checkpoint.resolve() and device == torch.device("cpu")
               for path, device, _ in records.action_loads)
    assert all(path == cli.vision_checkpoint and device == torch.device("cpu")
               for path, device, _ in records.vision_loads)
    assert all(bar.n == bar.options["total"] == episodes and bar.closed for bar in cli.bars)
    assert any(f"{count} workers x {episodes} episodes = {total}" in line for line in cli.output)
    assert any(f"{200 - total} skipped by flooring" in line for line in cli.output)
    final = [line for line in cli.output if line.startswith("Success rate:")]
    assert len(final) == 1
    assert f"{100 * successes / total:.2f}% ({successes}/{total} rollouts)" in final[0]
    assert f"IK failures: {count}; timeouts: {total - successes - count}" in final[0]


@pytest.mark.parametrize("failure,match", [
    ("required_checkpoint", "required"),
    ("missing_action", "Action checkpoint not found"),
    ("missing_vision", "Vision checkpoint not found"),
    ("zero_workers", "workers must be positive"),
    ("negative_workers", "workers must be positive"),
    ("too_many_workers", "workers cannot exceed the configured rollout count"),
])
def test_invalid_inputs_fail_before_starting_workers(monkeypatch, cli, capsys, failure, match):
    def unexpected_pool(*args, **kwargs):
        pytest.fail("Invalid inputs must be rejected before creating a process pool.")

    monkeypatch.setattr(cli.module.mp, "get_context", unexpected_pool)
    if failure == "required_checkpoint":
        monkeypatch.setattr(sys, "argv", [str(cli.script)])
    elif failure == "missing_action":
        cli.action_checkpoint.unlink()
    elif failure == "missing_vision":
        cli.vision_checkpoint.unlink()
    else:
        workers = {"zero_workers": 0, "negative_workers": -1, "too_many_workers": 201}[failure]
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--workers", str(workers)])

    with pytest.raises(SystemExit) as error:
        cli.module.main()

    assert error.value.code == 2
    assert match in capsys.readouterr().err
    assert not any(line.startswith("Success rate:") for line in cli.output)


@pytest.mark.parametrize("failure", ["evaluator_error", "incomplete_rollouts"])
def test_worker_errors_propagate_without_final_success_summary(monkeypatch, cli, failure):
    def evaluate(**kwargs):
        if failure == "evaluator_error":
            raise RuntimeError("simulator unavailable")
        attempted = kwargs["project"].training.action.rollout_validation.episodes - 1
        return ActionRolloutMetrics(episodes=attempted, timeouts=attempted)

    records = install_fake_workers(monkeypatch, cli, evaluator=evaluate)
    original_config = cli.config_path.read_bytes()
    match = "simulator unavailable" if failure == "evaluator_error" else "completed 99 episodes; expected 100"

    with pytest.raises(RuntimeError, match=match):
        cli.module.main()

    assert not any(line.startswith("Success rate:") for line in cli.output)
    assert cli.config_path.read_bytes() == original_config
    assert records.pools[0].closed and records.pools[0].error_type is RuntimeError
    assert cli.bars and all(bar.closed for bar in cli.bars)


def test_real_spawn_reports_checkpoint_error_and_exits_before_inference(cli):
    raw = json.loads(cli.config_path.read_text(encoding="utf-8"))
    raw["training"]["action"]["device"] = "cpu"
    cli.config_path.write_text(json.dumps(raw), encoding="utf-8")
    torch.save({"model_type": "vision"}, cli.action_checkpoint)

    result = subprocess.run(
        [sys.executable, str(cli.script), "--checkpoint", str(cli.action_checkpoint),
         "--config", str(cli.config_path)],
        cwd=cli.config_path.parent, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode != 0
    assert "2 workers x 100 episodes = 200" in result.stdout
    assert "Expected action checkpoint" in result.stderr
    assert "Success rate:" not in result.stdout
