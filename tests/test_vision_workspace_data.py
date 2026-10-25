"""Uniform held-cube collection, failure recovery, and additive publications."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import mujoco
import numpy as np
import pytest
import torch

from action_demonstrations import DemonstrationFailure
from kinematics import IKConvergenceError
from schemas.config import load_config
from training import load_vision_dataset
import vision_workspace_data as workspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    config = load_config(ROOT / "configs/default.json")
    config.training.vision.data_dir = tmp_path / "vision"
    config.environment.vision_data.train_fraction = .75
    config.environment.action_data.start_position_half_range = (0., 0., 0.)
    return config


def fake_factory(events, failure=None, failure_stage="collect"):
    class FakeScene:
        failed = False

        def __init__(self, project, split, *args):
            self.split, self.scene_id = split, str(uuid4())
            events.append(("create", self.scene_id, split))

        def perform(self, stage):
            events.append((stage, self.scene_id))
            if failure is not None and stage == failure_stage and not FakeScene.failed:
                FakeScene.failed = True
                raise failure

        def randomize_cameras(self, rng):
            self.perform("camera")

        def move_blue(self, rng, target):
            self.perform("blue")

        def collect(self, sample_id, target):
            self.perform("collect")
            return {"uuid": sample_id, "scene_id": self.scene_id, "split": self.split,
                    "final_scenario_metadata": {"gripper_error_mm": 0.0}}

        def close(self):
            events.append(("close", self.scene_id))

    return FakeScene


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_targets_use_all_three_configured_uniform_ranges(project):
    project.environment.vision_data.cube_x_range = (.22, .37)
    project.environment.vision_data.cube_y_range = (-.13, .09)
    rng = np.random.default_rng(19)
    points = np.array([workspace.sample_target(project, rng) for _ in range(20000)])
    low, high = np.array([.22, -.13, .03]), np.array([.37, .09, .08])
    normalized = (points - low) / (high - low)
    assert np.all((normalized >= 0) & (normalized <= 1))
    np.testing.assert_allclose(normalized.mean(0), .5, atol=.01)
    np.testing.assert_allclose(normalized.var(0), 1 / 12, atol=.003)
    assert np.max(np.abs(np.corrcoef(normalized.T) - np.eye(3))) < .025


def test_successes_reuse_scene_and_split_boundary_gets_new_scene(project, tmp_path):
    events, path = [], tmp_path / "generation.log"
    original_config = project.model_dump()
    paths = workspace.generate_workspace_vision_data(
        project=project, samples=4, log_file=path, scene_factory=fake_factory(events))
    records = [torch.load(item, weights_only=True) for item in paths]
    assert [item["split"] for item in records] == ["train"] * 3 + ["validation"]
    assert len({item["scene_id"] for item in records[:3]}) == 1
    assert records[-1]["scene_id"] != records[0]["scene_id"]
    assert len({item.name for item in paths}) == 4
    assert all(UUID(item.stem) and item.suffix == ".pt" for item in paths)
    assert [item[0] for item in events].count("create") == 2
    assert [item[0] for item in events].count("close") == 2
    assert [item[0] for item in events].count("camera") == 4
    assert [item[0] for item in events].count("blue") == 4
    log = read_log(path)
    saved = [item for item in log if item["event"] == "SAMPLE_SAVED"]
    assert {item["uuid"] for item in saved} == {item.stem for item in paths}
    assert all(item["path"] == str(paths[index]) for index, item in enumerate(saved))
    assert log[-1]["event"] == "RUN_COMPLETE"
    assert log[-1]["saved"] == log[-1]["attempts"] == 4
    assert "Generating workspace vision data" not in path.read_text()
    assert "\r" not in path.read_text()
    assert project.model_dump() == original_config


@pytest.mark.parametrize("stage,error", [
    ("camera", ValueError("camera error")),
    ("blue", RuntimeError("position error")),
    ("collect", IKConvergenceError({"attempts": [{"position_error": .01,
                 "tool_axis_error": 0.0, "tool_yaw_error": 0.0}]})),
])
def test_any_iteration_failure_recreates_before_sampling_again(
        project, tmp_path, monkeypatch, stage, error):
    events, path = [], tmp_path / "failures.log"
    original = workspace.sample_target

    def sample(project, rng):
        events.append(("sample",))
        return original(project, rng)

    monkeypatch.setattr(workspace, "sample_target", sample)
    paths = workspace.generate_workspace_vision_data(
        project=project, samples=2, log_file=path,
        scene_factory=fake_factory(events, error, stage))
    assert len(paths) == 2
    assert events[0][0] == "create" and events[1][0] == "sample"
    failed_index = next(index for index, item in enumerate(events) if item[0] == stage)
    assert [item[0] for item in events[failed_index + 1:failed_index + 4]] == ["close", "create", "sample"]
    log = read_log(path)
    failures = [item for item in log if item["event"] == "SAMPLE_FAILURE"]
    assert len(failures) == 1 and failures[0]["error_type"] == type(error).__name__
    assert failures[0]["saved"] == 0 and "traceback" in failures[0]
    assert ("ik_diagnostics" in failures[0]) == isinstance(error, IKConvergenceError)
    assert log[-1]["attempts"] == 3 and log[-1]["saved"] == 2
    assert UUID(failures[0]["uuid"]) not in {UUID(item.stem) for item in paths}


def test_attempt_limit_aborts_with_saved_count_and_closes_scene(project, tmp_path):
    events, path = [], tmp_path / "limit.log"
    with pytest.raises(RuntimeError, match="Reached 2 attempts with only 1/2"):
        workspace.generate_workspace_vision_data(
            project=project, samples=2, max_attempts=2, log_file=path,
            scene_factory=fake_factory(events, RuntimeError("failed attempt")))
    log = read_log(path)
    assert log[-1]["event"] == "RUN_ABORTED"
    assert log[-1]["saved"] == 1 and log[-1]["failures"] == 1
    assert events[-1][0] == "close"


def test_scene_creation_failures_are_logged_and_bounded(project, tmp_path, monkeypatch):
    path = tmp_path / "creation.log"
    monkeypatch.setattr(workspace, "MAX_CONSECUTIVE_FAILURES", 2)

    def fail(*args):
        raise RuntimeError("renderer creation error")

    with pytest.raises(RuntimeError, match="Scene creation failed repeatedly"):
        workspace.generate_workspace_vision_data(
            project=project, samples=1, log_file=path, scene_factory=fail)
    log = read_log(path)
    assert [row["creation_attempt"] for row in log if row["event"] == "SCENE_CREATE_FAILURE"] == [1, 2]
    assert log[-1]["event"] == "RUN_ABORTED" and log[-1]["saved"] == 0


def test_atomic_publication_refuses_duplicate_uuid_and_cleans_tempfiles(tmp_path):
    identifier = str(uuid4())
    record = {"uuid": identifier, "value": torch.tensor([17])}
    path = workspace.save_example(record, tmp_path)
    with pytest.raises(FileExistsError):
        workspace.save_example({**record, "value": torch.tensor([99])}, tmp_path)
    assert torch.load(path, weights_only=True)["value"].item() == 17
    assert list(tmp_path.iterdir()) == [path]


def test_failed_serialization_leaves_no_partial_record(tmp_path, monkeypatch):
    def fail(record, stream):
        stream.write(b"partial")
        raise OSError("write failed")

    monkeypatch.setattr(workspace.torch, "save", fail)
    with pytest.raises(OSError, match="write failed"):
        workspace.save_example({"uuid": str(uuid4())}, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["pickup", "lost", "missed", "recovery"])
def test_collect_checks_measured_grasp_and_actual_gripper_after_motion(project, monkeypatch, failure):
    target = np.array([.31, .01, .06])
    state = {"held": failure != "pickup", "gripper": target.copy()}
    recorder = SimpleNamespace(scene=lambda: state, recovery_requested=False)
    monkeypatch.setattr(workspace, "WorkspaceRecorder", lambda *args: recorder)
    monkeypatch.setattr(workspace, "pickup_from_current_pose", lambda _: None)

    def move(recorder, passed_target, gripper, stage):
        np.testing.assert_array_equal(passed_target, target)
        assert gripper == workspace.CLOSED_GRIPPER and stage == "transport"
        if failure == "lost":
            state["held"] = False
        if failure == "missed":
            state["gripper"] = target + [.02, 0, 0]
        if failure == "recovery":
            recorder.recovery_requested = True

    monkeypatch.setattr(workspace, "move_to", move)
    scene = workspace.WorkspaceScene.__new__(workspace.WorkspaceScene)
    scene.project, scene.environment, scene.encoders = project, object(), {}
    with pytest.raises(DemonstrationFailure):
        scene.collect(str(uuid4()), target)


class FakeRenderer:
    def __init__(self, model, height, width):
        self.shape = (height, width, 3)
        self.closed = False

    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.full(self.shape, 128, dtype=np.uint8)

    def close(self):
        self.closed = True


def test_failed_second_renderer_closes_first_renderer(project):
    renderers = []

    def factory(model, height, width):
        if renderers:
            raise RuntimeError("second renderer failed")
        renderer = FakeRenderer(model, height, width)
        renderers.append(renderer)
        return renderer

    with pytest.raises(RuntimeError, match="second renderer failed"):
        workspace.WorkspaceScene(project, "train", np.random.default_rng(42),
                                  np.random.default_rng(43), factory)
    assert len(renderers) == 1 and renderers[0].closed


def test_physical_pickup_then_persistent_carry_saves_loader_compatible_examples(project, tmp_path):
    project.environment.episode_seconds = 45
    project.environment.action_data.drop_disturbance.probability = 0
    project.environment.vision_data.focus_blur_scale = 0
    project.environment.vision_data.motion_blur_probability = 0
    scene = workspace.WorkspaceScene(project, "train", np.random.default_rng(42),
                                     np.random.default_rng(43), FakeRenderer)
    try:
        # Deterministic separated startup positions; all arm/grasp/move physics
        # remain real and no cube is attached or teleported after pickup.
        for name, xy in (("orange", [.30, 0]), ("blue", [.25, -.10])):
            scene.environment.data.joint(f"{name}_cube_joint").qpos[:] = [*xy, .02, 1, 0, 0, 0]
            scene.environment.data.joint(f"{name}_cube_joint").qvel[:] = 0
        mujoco.mj_forward(scene.environment.model, scene.environment.data)
        records = []
        for index, target in enumerate(([.30, 0, .06], [.32, .02, .07])):
            scene.randomize_cameras(np.random.default_rng(index))
            record = scene.collect(str(uuid4()), np.array(target))
            records.append(record)
            workspace.save_example(record, tmp_path / "examples")
            metadata = record["final_scenario_metadata"]
            assert metadata["currently_held"]
            assert metadata["picked_up_this_iteration"] == (index == 0)
            assert metadata["gripper_error_mm"] <= project.environment.action_data.teacher.position_tolerance * 1000
            np.testing.assert_allclose(record["cube_poses"][0, :3],
                                        scene.environment.data.body("orange_cube").xpos, atol=1e-7)
            assert record["wrist_timestamps"].item() == record["external_timestamps"].item()
        dataset = load_vision_dataset(tmp_path / "examples", show_progress=False)
        assert [tuple(tensor.shape) for tensor in dataset.tensors] == [
            (2, 3, 128, 128), (2, 3, 300, 300), (2, 6), (2, 10)]
        assert all(torch.isfinite(tensor).all() for tensor in dataset.tensors)
    finally:
        scene.close()
    assert all(renderer.renderer.closed for renderer in scene.renderers.values())
