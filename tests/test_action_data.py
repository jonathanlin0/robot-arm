"""Action episode publication, retry boundaries, progress, and CLI defaults."""

from collections import Counter
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

import action_data
from action_demonstrations import ActionEpisode, DemonstrationFailure
from action_preprocessing import ActionObservationPreprocessor
from kinematics import IKConvergenceError
from schemas.config import load_config
from training import load_action_dataset


ROOT = Path(__file__).parents[1]


@pytest.fixture
def project(tmp_path):
    value = load_config(ROOT / "configs/default.json")
    value.training.action.data_dir = tmp_path / "action"
    value.training.vision.checkpoint_path = tmp_path / "missing-vision.pt"
    value.environment.action_data.maximum_attempts_per_example = 3
    return value


@pytest.fixture
def episode():
    count = 7
    observations = torch.arange(count * 37, dtype=torch.float32).reshape(count, 37) / 100
    # A prediction below the normalization epsilon remains below unit length.
    # Saving/loading must not normalize that already-processed pair again.
    observations[:, 24:26] = torch.tensor([0.1, 0.])
    observations[:, 32:34] = torch.tensor([0., -1.])
    return ActionEpisode(
        observations=observations,
        prev_accepted_targets=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 100,
        actions=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 30 - 0.5,
        stable_time=1.05, required_stable_time=1.0,
        stage_steps={"approach": 2, "transport": 2, "recovery": 2, "settle_stack": 1},
        disturbance_events=[{"start_step": 2, "stop_steps": 1, "release_steps": 1}],
        recovery_count=1, waypoint_retry_count=0, grasp_seen=True,
        observation_source="vision_model",
    )


@pytest.fixture
def legacy_episode(episode):
    observations = torch.arange(len(episode.actions) * 41, dtype=torch.float32).reshape(-1, 41) / 100
    observations[:, 24:28] = torch.tensor([1., 0., 0., 0.])
    observations[:, 34:38] = torch.tensor([1., 0., 0., 0.])
    return replace(episode, observations=observations, observation_source="simulation_ground_truth")


@pytest.fixture
def fake_progress(monkeypatch):
    class Progress:
        instances = []
        messages = []

        def __init__(self, **kwargs):
            self.options = kwargs
            self.updates = []
            self.postfixes = []
            self.closed = False
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, amount):
            self.updates.append(amount)

        def set_postfix(self, **kwargs):
            self.postfixes.append(kwargs)

        @staticmethod
        def write(message):
            Progress.messages.append(message)

    monkeypatch.setattr(action_data, "tqdm", Progress)
    return Progress


@pytest.fixture
def fake_vision(monkeypatch):
    state = SimpleNamespace(
        model=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout()).train(),
        load_calls=[], device_calls=[], observers=[],
    )

    def resolve(name):
        state.device_calls.append(name)
        return torch.device("cpu")

    def load(path, device):
        state.load_calls.append((path, device))
        return state.model

    class Observer:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.entered = False
            self.closed = False
            self.exit_type = None
            assert kwargs["vision_model"] is state.model
            assert not state.model.training
            assert all(not parameter.requires_grad for parameter in state.model.parameters())
            state.observers.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exception_type, *args):
            self.closed = True
            self.exit_type = exception_type

    monkeypatch.setattr(action_data, "resolve_device", resolve)
    monkeypatch.setattr(action_data, "load_vision_checkpoint", load)
    monkeypatch.setattr(action_data, "VisionActionObserver", Observer)
    return state


@pytest.fixture
def fake_environment(monkeypatch, fake_vision):
    created = []

    def create(**kwargs):
        assert fake_vision.load_calls
        assert not fake_vision.model.training
        assert all(not parameter.requires_grad for parameter in fake_vision.model.parameters())
        value = object()
        created.append((value, kwargs))
        return value

    monkeypatch.setattr(action_data, "CubeStackEnvironment", create)
    return created


def assign_uuids(monkeypatch, count):
    # Full 128-bit seeds ensure generation does not truncate, hash, or reuse IDs.
    identifiers = [UUID(int=(1 << 127) + index) for index in range(count)]
    iterator = iter(identifiers)
    monkeypatch.setattr(action_data, "uuid4", lambda: next(iterator))
    return identifiers


def test_saved_episode_loads_with_current_schema_and_shifted_intended_history(project, episode):
    identifier = UUID(int=(1 << 128) - 1)
    metadata = action_data.save_action_episode(
        episode=episode, project=project, episode_id=identifier, split="train",
        vision_checkpoint=Path("checkpoints/vision/custom.pt"),
    )
    path = Path(metadata["path"])
    assert path == project.training.action.data_dir / "train" / f"{identifier}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert set(saved) == {"observations", "prev_accepted_targets", "actions", "metadata"}
    for name in ("observations", "prev_accepted_targets", "actions"):
        torch.testing.assert_close(saved[name], getattr(episode, name))
        assert saved[name].dtype == torch.float32 and saved[name].device.type == "cpu"
    assert saved["metadata"]["seed"] == identifier.int
    assert saved["metadata"]["disturbance_events"] == episode.disturbance_events
    assert saved["metadata"]["schema_version"] == 2
    assert saved["metadata"]["observation_source"] == "vision_model"
    assert saved["metadata"]["cube_velocity_source"] == "vision_position_differences"
    assert saved["metadata"]["vision_checkpoint"] == str(Path("checkpoints/vision/custom.pt").resolve())
    assert saved["metadata"]["action_source"] == "scripted_teacher_intended"
    assert saved["metadata"]["settings"]["action_data"]["drop_disturbance"]["probability"] == 0.002
    json.dumps(saved["metadata"], allow_nan=False)

    dataset = load_action_dataset(path.parent, history_length=4, stride=2, show_progress=False)
    tokens, valid, starts, labels, loss_mask = dataset.tensors
    assert tokens.shape == (3, 4, 44)
    assert starts.sum().item() == 1
    assert loss_mask.sum().item() == len(episode.actions)
    assert not (loss_mask & ~valid).any()
    torch.testing.assert_close(tokens[loss_mask][:, :37], episode.observations, rtol=0, atol=0)
    torch.testing.assert_close(tokens[loss_mask][:, 41:], episode.prev_accepted_targets)
    torch.testing.assert_close(labels[loss_mask], episode.actions)
    expected_previous = torch.cat((torch.zeros(1, 4), episode.actions[:-1]))
    torch.testing.assert_close(tokens[loss_mask][:, 37:41], expected_previous)
    # The action at the forced-drop timestamp remains an intended label.
    torch.testing.assert_close(labels[loss_mask][2], episode.actions[2])


def test_legacy_ground_truth_episode_still_saves_and_converts_when_loaded(project, legacy_episode):
    metadata = action_data.save_action_episode(
        episode=legacy_episode, project=project, episode_id=UUID(int=1), split="train",
    )
    path = Path(metadata["path"])
    saved = torch.load(path, weights_only=True)
    assert saved["observations"].shape[1] == 41
    assert saved["metadata"]["schema_version"] == 1
    assert saved["metadata"]["observation_source"] == "simulation_ground_truth"
    assert saved["metadata"]["cube_velocity_source"] == "simulation_ground_truth"
    assert "vision_checkpoint" not in saved["metadata"]
    tokens, _, _, _, loss_mask = load_action_dataset(path.parent, 4, 2, False).tensors
    expected = ActionObservationPreprocessor().transform_episode(legacy_episode.observations.numpy())
    torch.testing.assert_close(tokens[loss_mask][:, :37], torch.from_numpy(expected))


def test_saving_is_additive_across_splits_and_never_overwrites(project, episode):
    train_id, validation_id = UUID(int=1), UUID(int=2)
    first = action_data.save_action_episode(
        episode=episode, project=project, episode_id=train_id, split="train",
    )
    path = Path(first["path"])
    original_bytes = path.read_bytes()
    validation = action_data.save_action_episode(
        episode=episode, project=project, episode_id=validation_id, split="validation",
    )
    assert Path(validation["path"]).parent.name == "validation"
    episode.actions[:] = 0.75
    with pytest.raises(FileExistsError):
        action_data.save_action_episode(episode=episode, project=project, episode_id=train_id, split="train")
    assert path.read_bytes() == original_bytes
    assert len(list(project.training.action.data_dir.rglob("*.pt"))) == 2
    assert not list(project.training.action.data_dir.rglob("*.tmp"))


def test_partial_torch_save_is_never_published_and_temporary_file_is_removed(project, episode, monkeypatch):
    def broken_save(record, stream):
        stream.write(b"incomplete tensor archive")
        stream.flush()
        assert list(project.training.action.data_dir.rglob("*.pt")) == []
        raise OSError("disk full")

    monkeypatch.setattr(action_data.torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        action_data.save_action_episode(episode=episode, project=project, episode_id=UUID(int=1), split="train")
    assert list(project.training.action.data_dir.rglob("*.pt")) == []
    assert list(project.training.action.data_dir.rglob("*.tmp")) == []


@pytest.mark.parametrize("split", ["test", "../train", ""])
def test_invalid_split_is_rejected_without_writing(project, episode, split):
    with pytest.raises(ValueError, match="train or validation"):
        action_data.save_action_episode(episode=episode, project=project, episode_id=UUID(int=1), split=split)
    assert not project.training.action.data_dir.exists()


def test_invalid_episode_is_rejected_before_publication(project, episode):
    episode.success = False
    with pytest.raises(ValueError, match="verified released"):
        action_data.save_action_episode(episode=episode, project=project, episode_id=UUID(int=1), split="train")
    assert not project.training.action.data_dir.exists()


def test_retries_only_failures_and_progress_counts_successes_with_uuid_seeds(
    project, episode, monkeypatch, fake_progress, fake_environment, fake_vision,
):
    identifiers = assign_uuids(monkeypatch, 3)
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        assert kwargs["vision_observer"].entered and not kwargs["vision_observer"].closed
        if len(calls) == 1:
            raise DemonstrationFailure("cube slipped")
        return episode

    monkeypatch.setattr(action_data, "collect_action_episode", collect)
    records = action_data.generate_action_data(project=project, examples=2)
    assert len(calls) == 3 and len(records) == 2
    assert [call["seed"] for call in calls] == [value.int for value in identifiers]
    assert len(fake_environment) == 1
    assert all(call["environment"] is fake_environment[0][0] for call in calls)
    assert all(call["project"] is project for call in calls)
    assert fake_environment[0][1] == {"scene_path": project.environment.scene_path}
    assert fake_vision.load_calls == [(project.training.vision.checkpoint_path.resolve(), torch.device("cpu"))]
    assert fake_vision.device_calls == [project.training.vision.device]
    assert len(fake_vision.observers) == 1
    observer = fake_vision.observers[0]
    assert all(call["vision_observer"] is observer for call in calls)
    assert observer.closed and observer.exit_type is None
    assert observer.options["environment"] is fake_environment[0][0]
    assert observer.options["maximum_frame_age_seconds"] == project.cameras.maximum_frame_age_seconds
    assert observer.options["device"] == torch.device("cpu")
    assert observer.options["renderer_factory"] is action_data.mujoco.Renderer
    assert all(record["vision_checkpoint"] == str(project.training.vision.checkpoint_path.resolve())
               for record in records)
    progress = fake_progress.instances[0]
    assert progress.options["total"] == 2
    assert progress.updates == [1, 1] and progress.closed
    assert progress.postfixes[-1]["attempts"] == 3
    assert progress.postfixes[-1]["discarded"] == 1
    assert any("cube slipped" in message for message in fake_progress.messages)
    assert any("Added 2 episodes" in message for message in fake_progress.messages)
    assert len(list(project.training.action.data_dir.rglob("*.pt"))) == 2
    assert not project.training.vision.checkpoint_path.exists()


def test_attempt_budget_stops_retries_without_publishing_failures(
    project, monkeypatch, fake_progress, fake_environment, fake_vision,
):
    identifiers = assign_uuids(monkeypatch, 3)
    seeds = []

    def fail(**kwargs):
        seeds.append(kwargs["seed"])
        raise DemonstrationFailure("missed grasp")

    monkeypatch.setattr(action_data, "collect_action_episode", fail)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        action_data.generate_action_data(project=project, examples=1)
    assert seeds == [value.int for value in identifiers]
    assert fake_progress.instances[0].updates == []
    assert fake_progress.instances[0].closed
    assert fake_vision.observers[0].closed
    assert fake_vision.observers[0].exit_type is RuntimeError
    assert list(project.training.action.data_dir.rglob("*.pt")) == []


@pytest.mark.parametrize("error", [RuntimeError("unexpected failure"), OSError("missing resource")])
def test_unexpected_collection_errors_are_fatal_without_retry(
    project, monkeypatch, fake_progress, fake_environment, fake_vision, error,
):
    assign_uuids(monkeypatch, 1)
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(action_data, "collect_action_episode", fail)
    with pytest.raises(type(error), match=str(error)):
        action_data.generate_action_data(project=project, examples=1)
    assert len(calls) == 1
    assert fake_progress.instances[0].updates == []
    assert fake_vision.observers[0].closed
    assert fake_vision.observers[0].exit_type is type(error)


def test_ik_failure_saves_diagnostic_stops_without_retry_and_preserves_prior_success(
    project, episode, monkeypatch, fake_progress, fake_environment, fake_vision,
):
    identifiers = assign_uuids(monkeypatch, 2)
    diagnostic = {
        "attempts": [{"position_error": 0.002, "tool_axis_error": 0.2, "tool_yaw_error": 0.1}],
        "stage": "transport", "episode_step": 8,
    }
    failure = IKConvergenceError(diagnostic)
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return episode
        raise failure

    monkeypatch.setattr(action_data, "collect_action_episode", collect)
    with pytest.raises(IKConvergenceError) as caught:
        action_data.generate_action_data(project=project, examples=3)
    assert caught.value is failure and len(calls) == 2
    assert fake_vision.observers[0].closed
    assert fake_vision.observers[0].exit_type is IKConvergenceError
    assert fake_progress.instances[0].updates == [1]
    paths = list(project.training.action.data_dir.rglob("*.pt"))
    assert len(paths) == 1 and paths[0].stem == str(identifiers[0])
    torch.load(paths[0], weights_only=True)
    path = project.training.action.data_dir / "diagnostics" / f"ik-failure-{identifiers[1]}.json"
    saved = json.loads(path.read_text())
    assert saved["uuid"] == str(identifiers[1])
    assert saved["seed"] == identifiers[1].int
    assert saved["diagnostics"] == diagnostic
    assert saved["vision_checkpoint"] == str(project.training.vision.checkpoint_path.resolve())
    assert saved["completed_episodes"] == 1 and saved["attempts"] == 2
    assert saved["discarded_attempts"] == 0
    assert any("[IK_CONVERGENCE_FAILURE]" in message for message in fake_progress.messages)
    assert any("1 new completed episodes remain saved" in message for message in fake_progress.messages)


def test_two_generation_invocations_add_new_episodes(
    project, episode, monkeypatch, fake_progress, fake_environment,
):
    identifiers = assign_uuids(monkeypatch, 2)
    monkeypatch.setattr(action_data, "collect_action_episode", lambda **kwargs: episode)
    first = action_data.generate_action_data(project=project, examples=1)
    original = Path(first[0]["path"]).read_bytes()
    second = action_data.generate_action_data(project=project, examples=1)
    assert [first[0]["uuid"], second[0]["uuid"]] == [str(value) for value in identifiers]
    assert Path(first[0]["path"]).read_bytes() == original
    assert len(list(project.training.action.data_dir.rglob("*.pt"))) == 2


def test_generation_overrides_vision_checkpoint_device_and_renderer(
    project, episode, tmp_path, monkeypatch, fake_progress, fake_environment, fake_vision,
):
    checkpoint = tmp_path / "custom-vision.pt"
    renderer_factory = object()
    monkeypatch.setattr(action_data, "collect_action_episode", lambda **kwargs: episode)
    records = action_data.generate_action_data(
        project=project, examples=1, vision_checkpoint=checkpoint, device="custom-device",
        renderer_factory=renderer_factory,
    )
    assert fake_vision.device_calls == ["custom-device"]
    assert fake_vision.load_calls == [(checkpoint.resolve(), torch.device("cpu"))]
    assert fake_vision.observers[0].options["renderer_factory"] is renderer_factory
    assert records[0]["vision_checkpoint"] == str(checkpoint.resolve())
    assert project.training.vision.checkpoint_path != checkpoint


@pytest.mark.parametrize("error", [FileNotFoundError("missing vision checkpoint"),
                                   ValueError("incompatible vision checkpoint")])
def test_checkpoint_failures_precede_environment_rendering_and_output(
    project, monkeypatch, fake_progress, fake_environment, fake_vision, error,
):
    def fail(path, device):
        raise error

    monkeypatch.setattr(action_data, "load_vision_checkpoint", fail)
    monkeypatch.setattr(action_data, "collect_action_episode", lambda **kwargs: pytest.fail("Must not collect"))
    with pytest.raises(type(error), match=str(error)):
        action_data.generate_action_data(project=project, examples=1)
    assert fake_environment == [] and fake_vision.observers == []
    assert fake_progress.instances == []
    assert not project.training.action.data_dir.exists()


def test_public_generation_rejects_ground_truth_instead_of_silently_falling_back(
    project, legacy_episode, monkeypatch, fake_progress, fake_environment, fake_vision,
):
    monkeypatch.setattr(action_data, "collect_action_episode", lambda **kwargs: legacy_episode)
    with pytest.raises(ValueError, match="requires vision_model observations"):
        action_data.generate_action_data(project=project, examples=1)
    assert not project.training.action.data_dir.exists()
    assert fake_progress.instances[0].updates == []
    assert fake_vision.observers[0].closed
    assert fake_vision.observers[0].exit_type is ValueError


@pytest.mark.parametrize("examples,validation_examples,expected", [
    (200, None, {"train": 160, "validation": 40}),
    (6000, 1000, {"train": 5000, "validation": 1000}),
])
def test_generation_split_counts_without_simulation(
    project, episode, monkeypatch, fake_progress, fake_environment, examples, validation_examples, expected,
):
    identifiers = assign_uuids(monkeypatch, examples)
    seeds = []

    def collect(**kwargs):
        seeds.append(kwargs["seed"])
        return episode

    def save(**kwargs):
        return {**episode.metadata(), "split": kwargs["split"], "uuid": str(kwargs["episode_id"])}

    monkeypatch.setattr(action_data, "collect_action_episode", collect)
    monkeypatch.setattr(action_data, "save_action_episode", save)
    records = action_data.generate_action_data(project=project, examples=examples,
                                               validation_examples=validation_examples)
    assert Counter(record["split"] for record in records) == expected
    assert seeds == [value.int for value in identifiers]
    assert fake_progress.instances[0].updates == [1] * examples
    assert fake_progress.instances[0].options["total"] == examples
    assert not project.training.action.data_dir.exists()


@pytest.mark.parametrize("examples", [0, -1, 1.5, True])
def test_invalid_example_counts_fail_before_environment_or_progress(
    project, fake_progress, fake_environment, fake_vision, examples,
):
    with pytest.raises(ValueError, match="positive integer"):
        action_data.generate_action_data(project=project, examples=examples)
    assert fake_progress.instances == [] and fake_environment == []
    assert fake_vision.load_calls == [] and fake_vision.observers == []


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("generate_action_data_test_cli", ROOT / "scripts/generate_action_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arguments,expected", [
    ([], {"examples": 6000, "validation_examples": 1000}),
    (["--examples", "3"], {"examples": 3, "validation_examples": None}),
    (["--training-examples", "7", "--validation-examples", "2"],
     {"examples": 9, "validation_examples": 2}),
    (["--training-examples", "7"], {"examples": 1007, "validation_examples": 1000}),
    (["--validation-examples", "2"], {"examples": 5002, "validation_examples": 2}),
])
def test_cli_uses_file_hyperparameter_and_examples_override(cli, monkeypatch, arguments, expected):
    assert cli.NUMBER_OF_TRAINING_EXAMPLES == 5000
    assert cli.NUMBER_OF_VALIDATION_EXAMPLES == 1000
    loaded = object()
    config_paths, calls = [], []

    def load(**kwargs):
        config_paths.append(kwargs["path"])
        return loaded

    monkeypatch.setattr(cli, "load_config", load)
    monkeypatch.setattr(cli, "generate_action_data", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["generate_action_data.py", *arguments])
    cli.main()
    assert config_paths == [ROOT / "configs/default.json"]
    assert calls == [{"project": loaded, **expected, "vision_checkpoint": None, "device": None}]


def test_cli_forwards_vision_checkpoint_and_device_overrides(cli, monkeypatch, tmp_path):
    loaded = object()
    calls = []
    checkpoint = tmp_path / "chosen-vision.pt"
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: loaded)
    monkeypatch.setattr(cli, "generate_action_data", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["generate_action_data.py", "--examples", "2",
                                     "--vision-checkpoint", str(checkpoint), "--device", "cpu"])
    cli.main()
    assert calls == [{"project": loaded, "examples": 2, "validation_examples": None,
                      "vision_checkpoint": checkpoint, "device": "cpu"}]


def test_cli_help_bootstraps_source_imports_from_outside_repository(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_action_data.py"), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--vision-checkpoint" in result.stdout and "--device" in result.stdout
    assert not list(tmp_path.iterdir())


def test_cli_rejects_zero_before_loading_config_or_generating(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: pytest.fail("Invalid count must not load config"))
    monkeypatch.setattr(cli, "generate_action_data", lambda **kwargs: pytest.fail("Invalid count must not generate"))
    monkeypatch.setattr(sys, "argv", ["generate_action_data.py", "--examples", "0"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert "--examples must be positive" in capsys.readouterr().err
