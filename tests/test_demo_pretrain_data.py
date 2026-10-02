from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import mujoco
import numpy as np
import pytest
import torch

import scripts.demo as demo
from environment import CubeStackEnvironment, ROBOT_JOINT_NAMES
from observations import PrivilegedObservationBuilder
from randomization import CubeSpawnConfig


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml").resolve()


def demonstration_record(identifier: int, *, steps: int = 2, split: str = "train") -> dict[str, Any]:
    return {
        "uuid": str(UUID(int=identifier)),
        "seed": identifier,
        "split": split,
        "steps": steps,
        "success": True,
        "settings": {
            "task": "stack_orange_on_blue",
            "scene": str(SCENE_PATH),
            "spawn_config": asdict(CubeSpawnConfig()),
            "action_interval": 0.05,
            "waypoint_height": 0.08,
        },
    }


def save_demonstration(directory: Path, record: dict[str, Any], *, manifest: bool = True) -> None:
    steps = record["steps"]
    for name, tensor in (
        ("observations", torch.zeros((steps + 1, 49), dtype=torch.float32)),
        ("accepted_targets", torch.zeros((steps + 1, 3), dtype=torch.float32)),
        ("actions", torch.zeros((steps, 4), dtype=torch.float32)),
    ):
        path = directory / record["split"] / name / f"{record['uuid']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
    if manifest:
        path = directory / "manifests" / f"{record['uuid']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format_version": 2, "episode": record}))


def forbid_call(*args: Any, **kwargs: Any) -> None:
    pytest.fail("Recorded demonstration playback must not load a policy, reset, or step physics")


def test_pretraining_data_argument_is_exclusive_with_wandb_selection() -> None:
    options = demo.parse_arguments(["--pretrain-data", "--workers", "3"])
    assert options.pretrain_data is True
    assert options.workers == 3
    assert options.repeat_wandb is None
    with pytest.raises(SystemExit):
        demo.parse_arguments(["--pretrain-data", "--repeat-wandb", "example"])


def test_selection_accepts_one_training_demo_and_ignores_other_data(tmp_path: Path) -> None:
    valid = demonstration_record(1)
    validation = demonstration_record(2, split="test")
    orphan = demonstration_record(3)
    incomplete = demonstration_record(4)
    unsuccessful = demonstration_record(5)
    unsuccessful["success"] = False
    for record in (valid, validation, incomplete, unsuccessful):
        save_demonstration(tmp_path, record)
    save_demonstration(tmp_path, orphan, manifest=False)
    (tmp_path / "train" / "actions" / f"{incomplete['uuid']}.pt").unlink()

    selected = demo.select_demonstrations(tmp_path)

    assert [record["uuid"] for record in selected] == [valid["uuid"]]


def test_selection_chooses_at_most_nine_distinct_complete_training_demos(tmp_path: Path) -> None:
    records = [demonstration_record(identifier) for identifier in range(1, 13)]
    for record in records:
        save_demonstration(tmp_path, record)

    selected = demo.select_demonstrations(tmp_path)

    selected_ids = [record["uuid"] for record in selected]
    assert len(selected_ids) == len(set(selected_ids)) == 9
    assert set(selected_ids) <= {record["uuid"] for record in records}
    assert len(demo.select_demonstrations(tmp_path, count=3)) == 3


def test_empty_or_validation_only_data_reports_no_eligible_demonstrations(tmp_path: Path) -> None:
    save_demonstration(tmp_path, demonstration_record(1, split="test"))
    with pytest.raises(ValueError, match="(?i)(no.*training|no.*demonstration)"):
        demo.select_demonstrations(tmp_path)


def test_load_observations_preserves_every_stored_frame(tmp_path: Path) -> None:
    record = demonstration_record(1, steps=405)
    save_demonstration(tmp_path, record)
    observations = torch.arange(406 * 49, dtype=torch.float32).reshape(406, 49)
    torch.save(observations, tmp_path / "train" / "observations" / f"{record['uuid']}.pt")

    loaded = demo.load_observations(tmp_path, record)

    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, observations.numpy())


@pytest.mark.parametrize("case", ["missing_final_frame", "wrong_width", "nonfinite", "wrong_dtype"])
def test_load_observations_rejects_invalid_saved_states(tmp_path: Path, case: str) -> None:
    record = demonstration_record(1)
    save_demonstration(tmp_path, record)
    observations = torch.zeros((3, 49), dtype=torch.float32)
    if case == "missing_final_frame":
        observations = observations[:-1]
    elif case == "wrong_width":
        observations = torch.zeros((3, 50), dtype=torch.float32)
    elif case == "nonfinite":
        observations[-1, 0] = torch.nan
    else:
        observations = observations.double()
    torch.save(observations, tmp_path / "train" / "observations" / f"{record['uuid']}.pt")

    with pytest.raises(ValueError):
        demo.load_observations(tmp_path, record)


def test_recorded_state_restores_robot_cubes_and_forward_kinematics_without_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = CubeStackEnvironment(scene_path=SCENE_PATH)
    environment.reset(seed=7)
    for index, name in enumerate(ROBOT_JOINT_NAMES):
        environment.data.joint(name).qpos[0] = [0.1, 0.2, -0.3, 0.4, -0.1, 0.3][index]
        environment.data.joint(name).qvel[0] = index / 10.0
        environment.data.actuator(name).ctrl[0] = index / 20.0
    orange = environment.data.joint("orange_cube_joint")
    orange.qpos[:] = [0.31, -0.03, 0.12, np.cos(0.3), 0.0, 0.0, np.sin(0.3)]
    orange.qvel[:] = np.arange(6) / 100.0
    blue = environment.data.joint("blue_cube_joint")
    blue.qpos[:] = [0.27, 0.06, 0.02, np.cos(0.2), np.sin(0.2), 0.0, 0.0]
    blue.qvel[:] = -np.arange(6) / 100.0
    mujoco.mj_forward(environment.model, environment.data)
    observation = PrivilegedObservationBuilder(environment).build(environment.get_state())
    restored = mujoco.MjData(environment.model)
    restored.time = 12.5
    monkeypatch.setattr(demo.mujoco, "mj_step", forbid_call)
    monkeypatch.setattr(demo.mujoco, "mj_resetData", forbid_call)
    monkeypatch.setattr(np.random, "default_rng", forbid_call)

    demo.apply_recorded_observation(environment.model, restored, observation)

    assert restored.time == 12.5
    for index, name in enumerate(ROBOT_JOINT_NAMES):
        assert restored.joint(name).qpos[0] == float(observation[index])
        assert restored.joint(name).qvel[0] == float(observation[index + 6])
        assert restored.actuator(name).ctrl[0] == float(observation[index + 12])
    np.testing.assert_allclose(restored.site("gripperframe").xpos, observation[18:21], atol=1e-7)
    for name, start in (("orange_cube", 21), ("blue_cube", 34)):
        np.testing.assert_allclose(restored.body(name).xpos, observation[start:start + 3], atol=1e-7)
        np.testing.assert_allclose(restored.body(name).xquat, observation[start + 3:start + 7], atol=1e-7)
        np.testing.assert_array_equal(restored.joint(f"{name}_joint").qvel, observation[start + 7:start + 13])


@pytest.mark.parametrize("steps", [3, 405])
def test_render_demonstration_includes_initial_final_and_frozen_padding_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, steps: int,
) -> None:
    record = demonstration_record(1, steps=steps)
    observations = np.zeros((steps + 1, 49), dtype=np.float32)
    observations[:, 0] = np.arange(steps + 1)
    applied_rows = []
    writers = []
    renderers = []

    class FakeRenderer:
        scene = object()

        def __init__(self, model: object, *, height: int, width: int) -> None:
            self.shape = (height, width, 3)
            self.closed = False
            self.render_count = 0
            renderers.append(self)

        def update_scene(self, data: object, *, camera: object) -> None:
            pass

        def render(self) -> np.ndarray:
            self.render_count += 1
            return np.full(self.shape, applied_rows[-1] % 251, dtype=np.uint8)

        def close(self) -> None:
            self.closed = True

    class FakeWriter:
        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs
            self.frames = []
            self.closed = False
            writers.append(self)

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())

        def close(self, *, check_return_code: bool = True) -> None:
            self.closed = True

    monkeypatch.setattr(demo, "load_observations", lambda directory, metadata: observations)
    monkeypatch.setattr(demo, "apply_recorded_observation", lambda model, data, row: applied_rows.append(int(row[0])))
    monkeypatch.setattr(demo.mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(demo, "RawVideoWriter", FakeWriter)
    monkeypatch.setattr(demo, "PANEL_WIDTH", 8)
    monkeypatch.setattr(demo, "PANEL_HEIGHT", 8)
    monkeypatch.setattr(demo, "add_status_border", lambda frame, status: frame)
    monkeypatch.setattr(demo, "add_spawn_area_outline", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo, "add_gripperframe_marker", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo, "add_active_orange_approach_target_marker", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo, "get_worker_policy", forbid_call)
    monkeypatch.setattr(demo.PPO, "load", forbid_call)
    monkeypatch.setattr(demo, "CubeStackGymEnvironment", forbid_call)
    monkeypatch.setattr(demo.mujoco, "mj_step", forbid_call)
    monkeypatch.setattr(demo.mujoco, "mj_resetData", forbid_call)
    task = demo.DemonstrationTask(
        rollout_index=0, output_path=tmp_path / "recorded.mp4", record=record,
        data_directory=tmp_path, frame_count=steps + 4,
    )

    result = demo.render_demonstration(task)

    assert applied_rows == list(range(steps + 1))
    assert len(writers) == 1
    assert writers[0].options["frames_per_second"] == 20.0
    assert len(writers[0].frames) == task.frame_count
    assert renderers[0].render_count == steps + 1
    for index, frame in enumerate(writers[0].frames):
        np.testing.assert_array_equal(frame, min(index, steps) % 251)
    assert writers[0].closed and renderers[0].closed
    assert result.status == demo.SUCCESS
    assert result.episode_steps == steps


def test_spawned_worker_dispatches_demonstrations_without_loading_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task = demo.DemonstrationTask(
        rollout_index=0, output_path=tmp_path / "demo.mp4", record=demonstration_record(1),
        data_directory=tmp_path, frame_count=3,
    )
    received = []
    expected = object()
    monkeypatch.setattr(demo, "get_worker_policy", forbid_call)
    monkeypatch.setattr(demo, "render_demonstration", lambda value: received.append(value) or expected)

    assert demo.render_rollout(task) is expected
    assert received == [task]


def test_blank_panel_contains_only_black_frames_at_requested_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    frames = []
    calls = {}

    class FakeWriter:
        def __init__(self, output_path: Path, width: int, height: int, frames_per_second: float) -> None:
            calls["options"] = {
                "output_path": output_path, "width": width, "height": height,
                "frames_per_second": frames_per_second,
            }

        def write(self, frame: np.ndarray) -> None:
            frames.append(frame.copy())

        def close(self, *, check_return_code: bool = True) -> None:
            calls["closed"] = True

    monkeypatch.setattr(demo, "RawVideoWriter", FakeWriter)
    monkeypatch.setattr(demo, "PANEL_HEIGHT", 8)
    monkeypatch.setattr(demo, "PANEL_WIDTH", 10)
    path = tmp_path / "blank.mp4"

    demo.write_blank_panel(path, 7, 20.0)

    assert calls["options"] == {
        "output_path": path, "width": 10, "height": 8, "frames_per_second": 20.0,
    }
    assert calls["closed"]
    assert len(frames) == 7
    for frame in frames:
        assert frame.shape == (8, 10, 3)
        assert frame.dtype == np.uint8
        assert not np.any(frame)


def configure_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, records: list[dict]) -> dict:
    calls: dict[str, Any] = {"blanks": []}
    directory = tmp_path / "panels"
    output = tmp_path / "stack_demo.mp4"
    monkeypatch.setattr(demo, "PRETRAINING_DATA_DIRECTORY", tmp_path / "data")
    monkeypatch.setattr(demo, "OUTPUT_PATH", output)
    monkeypatch.setattr(demo, "temporary_rollout_directory", lambda: directory)
    monkeypatch.setattr(demo.shutil, "which", lambda name: name)
    monkeypatch.setattr(demo, "resolve_checkpoint_path", forbid_call)
    monkeypatch.setattr(demo, "get_worker_policy", forbid_call)
    monkeypatch.setattr(demo.PPO, "load", forbid_call)

    def select(data_directory: Path, count: int = 9) -> list[dict]:
        calls["selection"] = (data_directory, count)
        return records

    def render(tasks: list, worker_count: int) -> list:
        calls["tasks"] = tasks
        calls["workers"] = worker_count
        return [demo.RolloutResult(task.rollout_index, task.record["seed"], demo.SUCCESS,
                                   task.record["steps"], 0.0, task.output_path) for task in tasks]

    def combine(paths: list[Path], staging: Path) -> None:
        calls["combined"] = list(paths)
        staging.write_bytes(b"completed grid")

    original_cleanup = demo.cleanup_temporary_directory

    def cleanup(path: Path) -> None:
        calls["cleaned"] = path
        original_cleanup(path)

    monkeypatch.setattr(demo, "select_demonstrations", select)
    monkeypatch.setattr(demo, "run_parallel_rollouts", render)
    monkeypatch.setattr(demo, "write_blank_panel", lambda *args: calls["blanks"].append(args))
    monkeypatch.setattr(demo, "combine_rollout_videos", combine)
    monkeypatch.setattr(demo, "cleanup_temporary_directory", cleanup)
    return calls


def test_main_renders_selected_training_data_and_pads_empty_grid_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    records = [demonstration_record(1, steps=2), demonstration_record(2, steps=405)]
    calls = configure_main(monkeypatch, tmp_path, records)

    demo.main(["--pretrain-data", "--workers", "3"])

    paths = demo.rollout_video_paths(tmp_path / "panels")
    assert calls["selection"] == (tmp_path / "data", 9)
    assert calls["workers"] == 2
    assert len(calls["tasks"]) == 2
    for index, task in enumerate(calls["tasks"]):
        assert isinstance(task, demo.DemonstrationTask)
        assert task.record == records[index]
        assert task.data_directory == tmp_path / "data"
        assert task.output_path == paths[index]
        assert task.frame_count == 406
    assert calls["blanks"] == [(path, 406, 20.0) for path in paths[2:]]
    assert calls["combined"] == paths
    assert calls["cleaned"] == tmp_path / "panels"
    assert not (tmp_path / "panels").exists()
    assert (tmp_path / "stack_demo.mp4").read_bytes() == b"completed grid"


def test_main_rejects_mixed_recorded_frame_intervals_before_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    records = [demonstration_record(1), demonstration_record(2)]
    records[1]["settings"]["action_interval"] = 0.1
    calls = configure_main(monkeypatch, tmp_path, records)

    with pytest.raises(ValueError, match="(?i)(interval|frame|rate)"):
        demo.main(["--pretrain-data"])

    assert "tasks" not in calls
