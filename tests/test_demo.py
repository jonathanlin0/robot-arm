"""Demo presentation and encoding contracts without loading trained weights."""

import importlib.util
import json
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from action_evaluation import ActionRolloutFrame, ActionRolloutMetrics


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(monkeypatch):
    spec = importlib.util.spec_from_file_location("demo_cli_test", ROOT / "scripts/demo.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("successful", [False, True])
def test_panel_preserves_square_view_and_source_with_visible_status_border(demo, successful):
    rgb = np.full((300, 300, 3), (13, 27, 49), dtype=np.uint8)
    original = rgb.copy()
    rgb.flags.writeable = False
    panel = demo.make_panel(rgb, successful=successful)
    assert panel.shape == (240, 320, 3) and panel.dtype == np.uint8
    color = demo.GREEN if successful else demo.YELLOW
    assert np.all(panel[:4] == color) and np.all(panel[-4:] == color)
    assert np.all(panel[:, :4] == color) and np.all(panel[:, -4:] == color)
    assert np.all(panel[4:-4, 40:280] == (13, 27, 49))
    assert not panel[4:-4, 4:40].any() and not panel[4:-4, 280:-4].any()
    np.testing.assert_array_equal(rgb, original)


@pytest.fixture
def encoder(demo, monkeypatch):
    state = SimpleNamespace(renderers=[], processes=[], fail_write=False, returncode=0)

    class Pipe:
        def __init__(self):
            self.frames = []
            self.closed = False

        def write(self, value):
            if state.fail_write:
                raise BrokenPipeError("encoder stopped")
            self.frames.append(np.frombuffer(value, dtype=np.uint8).reshape(240, 320, 3).copy())

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, command, **kwargs):
            self.stdin = Pipe()
            self.stderr = kwargs["stderr"]
            self.stderr.write(b"injected encoder diagnostic")
            self.returncode = None
            self.terminated = False
            state.processes.append(self)

        def wait(self):
            self.returncode = state.returncode
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

    class Renderer:
        def __init__(self, model, *, height, width):
            self.shape = (height, width, 3)
            self.closed = False
            self.cameras = []
            state.renderers.append(self)

        def update_scene(self, data, *, camera):
            self.value = data.value
            self.cameras.append(camera)

        def render(self):
            return np.full(self.shape, self.value, dtype=np.uint8)

        def close(self):
            self.closed = True

    monkeypatch.setattr(demo.subprocess, "Popen", Process)
    monkeypatch.setattr(demo.mujoco, "Renderer", Renderer)
    state.environment = SimpleNamespace(model=object(), data=SimpleNamespace(value=10), state_lock=Lock())
    return state


@pytest.mark.parametrize("terminal", ["success", "ik_failure"])
def test_recorder_buffers_current_tick_replaces_ik_frame_and_freezes_terminal(demo, encoder, tmp_path,
                                                                           terminal):
    environment = encoder.environment
    with demo.PanelRecorder(tmp_path / "panel.mp4", ffmpeg="ffmpeg", fps=20, frame_count=5) as recorder:
        recorder.record(environment, ActionRolloutFrame(0, 0, "running"))
        assert recorder.written == 0
        environment.data.value = 20
        recorder.record(environment, ActionRolloutFrame(0, 1, "running"))
        assert recorder.written == 1
        environment.data.value = 30
        last_step = 2 if terminal == "success" else 1
        recorder.record(environment, ActionRolloutFrame(0, last_step, terminal))
    process = encoder.processes[0]
    frames = process.stdin.frames
    expected_pixels = [10, 20, 30, 30, 30] if terminal == "success" else [10, 30, 30, 30, 30]
    assert [int(frame[100, 100, 0]) for frame in frames] == expected_pixels
    assert np.all(frames[0][0, 0] == demo.YELLOW)
    assert np.all(frames[-1][0, 0] == (demo.GREEN if terminal == "success" else demo.YELLOW))
    assert recorder.status == terminal and recorder.step == last_step and recorder.written == 5
    assert len(encoder.renderers) == 1
    renderer = encoder.renderers[0]
    assert renderer.shape == (300, 300, 3)
    assert renderer.cameras == [demo.SIM_CAMERA_NAMES["external"]] * 3
    assert renderer.closed and process.stdin.closed and process.stderr.closed
    assert not process.terminated


@pytest.mark.parametrize("failure", ["observer", "write", "encoder", "empty"])
def test_recorder_errors_close_renderers_streams_and_processes(demo, encoder, tmp_path, failure):
    encoder.fail_write = failure == "write"
    encoder.returncode = 1 if failure in {"write", "encoder"} else 0
    expected = {"observer": "rendering failed", "empty": "no frames"}.get(failure, "encoder diagnostic")
    with pytest.raises(RuntimeError, match=expected):
        with demo.PanelRecorder(tmp_path / "panel.mp4", ffmpeg="ffmpeg", fps=20, frame_count=3) as recorder:
            if failure != "empty":
                recorder.record(encoder.environment, ActionRolloutFrame(0, 0, "running"))
            if failure == "observer":
                raise RuntimeError("rendering failed")
    process = encoder.processes[0]
    assert process.returncode is not None
    assert process.stdin.closed and process.stderr.closed
    assert all(renderer.closed for renderer in encoder.renderers)
    if failure in {"observer", "empty"}:
        assert process.terminated


@pytest.fixture
def cli(demo, monkeypatch, tmp_path):
    assert demo.REPO_ROOT == ROOT
    repo = tmp_path / "project"
    (repo / "configs").mkdir(parents=True)
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["training"]["action"]["checkpoint_path"] = "saved/action.pt"
    raw["training"]["vision"]["checkpoint_path"] = "saved/vision.pt"
    config = repo / "configs/default.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    (repo / "saved").mkdir()
    for filename in ("action.pt", "vision.pt"):
        (repo / "saved" / filename).touch()
    working = tmp_path / "elsewhere"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setattr(demo, "REPO_ROOT", repo)
    monkeypatch.setattr(demo.shutil, "which", lambda name: "/fake/bin/ffmpeg")
    return SimpleNamespace(repo=repo, working=working, config=config)


def test_cli_uses_training_defaults_independently_of_working_directory_and_accepts_overrides(demo, cli):
    defaults = demo.parse_arguments([])
    assert defaults.config == cli.config
    assert defaults.project.training.action.checkpoint_path == cli.repo / "saved/action.pt"
    assert defaults.project.training.vision.checkpoint_path == cli.repo / "saved/vision.pt"
    assert defaults.output == cli.working / "stack_demo.mp4"
    action = cli.working / "custom-action.pt"
    vision = cli.working / "custom-vision.pt"
    action.touch()
    vision.touch()
    overridden = demo.parse_arguments([
        "--weights", action.name, "--vision-checkpoint", vision.name,
        "--episode-seconds", "1.25", "--seed", "100", "--workers", "3", "--device", "cpu",
        "--output", "new-demo.mp4",
    ])
    assert overridden.project.training.action.checkpoint_path == action
    assert overridden.project.training.vision.checkpoint_path == vision
    assert overridden.project.environment.episode_seconds == 1.25
    assert overridden.project.training.action.rollout_validation.seed == 100
    assert overridden.project.training.action.device == "cpu" and overridden.workers == 3
    assert overridden.output == cli.working / "new-demo.mp4"


@pytest.mark.parametrize("arguments,diagnostic", [
    (["--checkpoint", "missing.pt"], "Action checkpoint not found"),
    (["--vision-checkpoint", "missing.pt"], "Vision checkpoint not found"),
    (["--workers", "13"], "workers must be between"),
    (["--episode-seconds", "nan"], "finite and positive"),
    (["--episode-seconds", "0.01"], "at least one control tick"),
    (["--seed", "-1"], "seed must be nonnegative"),
    (["--output", "demo.gif"], "output must end in .mp4"),
])
def test_cli_rejects_invalid_inputs_before_rollouts(demo, cli, capsys, arguments, diagnostic):
    with pytest.raises(SystemExit) as error:
        demo.parse_arguments(arguments)
    assert error.value.code == 2
    assert diagnostic in capsys.readouterr().err


def test_grid_places_completed_panels_in_row_major_order_and_reports_encoder_failures(demo, monkeypatch,
                                                                                   tmp_path):
    results = [demo.RolloutResult(index, 43 + index, ActionRolloutMetrics(1, timeouts=1), 2,
                                 "timeout", tmp_path / f"panel-{index}.mp4")
               for index in reversed(range(12))]
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(demo.subprocess, "run", run)
    demo.compose_grid(results, tmp_path / "grid.mp4", ffmpeg="ffmpeg")
    command = commands[0]
    assert [command[index + 1] for index, item in enumerate(command) if item == "-i"] == [
        str(tmp_path / f"panel-{index}.mp4") for index in range(12)
    ]
    assert "xstack=inputs=12:grid=4x3:shortest=1" in command[command.index("-filter_complex") + 1]
    with pytest.raises(ValueError, match="twelve distinct"):
        demo.compose_grid(results[:-1] + [results[0]], tmp_path / "grid.mp4", ffmpeg="ffmpeg")
    assert len(commands) == 1
    monkeypatch.setattr(demo.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr="unreadable panel"))
    with pytest.raises(RuntimeError, match="unreadable panel"):
        demo.compose_grid(results, tmp_path / "grid.mp4", ffmpeg="ffmpeg")
