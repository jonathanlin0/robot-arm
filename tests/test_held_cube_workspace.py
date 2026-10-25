"""Corner diagnostics distinguish solver, grasp, accuracy, and sustained-hold failures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "check_held_cube_workspace", ROOT / "scripts/diagnostics/check_held_cube_workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(diagnostic):
    config = diagnostic.load_config(ROOT / "configs/default.json")
    config.environment.action_data.drop_disturbance.probability = 0.0
    config.environment.action_data.start_position_half_range = (0.0, 0.0, 0.0)
    return config


def test_default_run_covers_four_configured_corners_at_every_height(
        diagnostic, project, monkeypatch, tmp_path):
    bounds = project.environment.vision_data
    bounds.cube_x_range = (.23, .37)
    bounds.cube_y_range = (-.11, .09)
    environment = SimpleNamespace(
        model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
        data=SimpleNamespace(body=lambda _: SimpleNamespace(xpos=np.array([.3, -.01, .1]))))
    attempts, pickups = [], []
    monkeypatch.setattr(diagnostic, "load_config", lambda _: project)
    monkeypatch.setattr(diagnostic, "CubeStackEnvironment", lambda _: environment)

    def prepare(config, env, center, blue):
        pickups.append((center.copy(), blue.copy()))
        return object()

    def attempt(config, env, grasp, target, route, hold_seconds):
        attempts.append((target.copy(), route, hold_seconds))
        return {"route": route, "status": "reached_and_held"}

    monkeypatch.setattr(diagnostic, "prepare_grasp", prepare)
    monkeypatch.setattr(diagnostic, "try_target", attempt)
    output = tmp_path / "result.json"
    monkeypatch.setattr(diagnostic.sys, "argv", ["diagnostic", "--output", str(output)])
    diagnostic.main()
    report = json.loads(output.read_text())
    expected = {(x, y, z / 100) for x in bounds.cube_x_range for y in bounds.cube_y_range
                for z in range(3, 9)}
    assert {tuple(row["target_xyz_m"]) for row in report["results"]} == expected
    assert len(attempts) == 24 and len(pickups) == 4
    assert all(route == "direct" and hold == .5 for _, route, hold in attempts)
    assert all(row["all_corners_passed"] for row in report["height_summary"])
    for (center, blue), (_, target_xy) in zip(pickups, diagnostic.spawn_corners(project)):
        np.testing.assert_allclose(center, [.3, -.01])
        np.testing.assert_allclose(blue, 2 * center - target_xy)


def test_height_summary_does_not_certify_incomplete_or_failed_heights(diagnostic):
    rows = [{"z_cm": 3, "corner": f"corner_{index}", "status": "reached_and_held"}
            for index in range(4)]
    rows += [{"z_cm": 4, "corner": f"corner_{index}", "status": "reached_and_held"}
             for index in range(3)]
    rows += [{"z_cm": 4, "corner": "corner_3", "status": "failed_tested_routes"}]
    summary = diagnostic.summarize_heights(rows, [3, 4, 5])
    assert [row["all_corners_passed"] for row in summary] == [True, False, False]
    assert [row["corners_passed"] for row in summary] == [4, 3, 0]
    assert summary[1]["failed_corners"] == [{"corner": "corner_3", "status": "failed_tested_routes"}]


@pytest.mark.parametrize("route,intermediate", [
    ("direct", None), ("xy_then_z", [.35, -.1, .1]), ("z_then_xy", [.3, 0, .03]),
])
def test_routes_use_actual_cube_center_and_leave_targets_unchanged(diagnostic, route, intermediate):
    start, target = np.array([.3, 0, .1]), np.array([.35, -.1, .03])
    points = diagnostic.route_waypoints(start, target, route)
    np.testing.assert_array_equal(points[-1], target)
    if intermediate is not None:
        np.testing.assert_array_equal(points[0], intermediate)
    points[0][:] = 99
    np.testing.assert_array_equal(start, [.3, 0, .1])
    np.testing.assert_array_equal(target, [.35, -.1, .03])


def test_restoring_real_grasp_restores_physics_and_copies_controller_state(diagnostic, project):
    environment = diagnostic.CubeStackEnvironment(project.environment.scene_path)
    grasp = diagnostic.prepare_grasp(project, environment, np.array([.3, 0]), np.array([.25, -.1]))
    baseline, accepted_target, heights = grasp
    baseline_target = accepted_target.copy()
    baseline_heights = heights.copy()
    recorder = diagnostic.restore_grasp(project, environment, grasp)
    assert recorder.scene()["held"]
    assert recorder.grasp_seen and recorder._pickup_grasp_seen
    for field in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act"):
        np.testing.assert_array_equal(getattr(environment.data, field), getattr(baseline, field))
    assert environment.data.time == baseline.time

    recorder.controller.prev_accepted_target[:] = 99
    recorder.initial_heights["orange"] = 99
    environment.data.ctrl[:] = 0
    mujoco.mj_step(environment.model, environment.data, nstep=10)
    restored = diagnostic.restore_grasp(project, environment, grasp)
    np.testing.assert_array_equal(restored.controller.prev_accepted_target, baseline_target)
    assert restored.initial_heights == baseline_heights
    for field in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act"):
        np.testing.assert_array_equal(getattr(environment.data, field), getattr(baseline, field))
    assert environment.data.time == baseline.time
    assert restored.scene()["held"] and not restored.actions


def fake_attempt(diagnostic, monkeypatch, measurements, *, move_held=True, move_error=0,
                 move_exception=None):
    target = np.array([.3, 0, .06])
    env = SimpleNamespace(data=SimpleNamespace(time=1.0))
    teacher = SimpleNamespace(position_tolerance=.003, maximum_move_steps=len(measurements))
    recorder = SimpleNamespace(environment=env, interval=.05,
                               config=SimpleNamespace(teacher=teacher), recovery_requested=False,
                               steps=0)
    scene = {"orange": target + [move_error, 0, 0], "gripper": target + [0, 0, .02],
             "held": move_held}
    recorder.scene = lambda: scene

    def step(action, stage):
        recorder.steps += 1
        env.data.time += recorder.interval

    recorder.step = step
    monkeypatch.setattr(diagnostic, "restore_grasp", lambda *args: recorder)

    def move(*args):
        if move_exception is not None:
            raise move_exception

    monkeypatch.setattr(diagnostic, "move_held_cube_to", move)
    monkeypatch.setattr(diagnostic, "action_toward", lambda *args: np.zeros(4))

    def measured(*args):
        index = min(max(recorder.steps - 1, 0), len(measurements) - 1)
        error, held, unsupported = measurements[index]
        return {"position_error_mm": error, "held": held, "unsupported": unsupported}

    monkeypatch.setattr(diagnostic, "snapshot_result", measured)
    result = diagnostic.try_target(None, env, None, target, "direct", .15)
    return result, recorder


@pytest.mark.parametrize("held,error,expected", [
    (False, 0, "grasp_lost"), (True, .004, "motion_timeout"),
])
def test_ik_motion_alone_cannot_pass_without_actual_held_cube_at_target(
        diagnostic, monkeypatch, held, error, expected):
    result, recorder = fake_attempt(diagnostic, monkeypatch, [(0, held, True)],
                                    move_held=held, move_error=error)
    assert result["status"] == expected and recorder.steps == 0
    assert result["simulated_seconds"] == 0
    assert result["consecutive_verified_hold_seconds"] == 0


@pytest.mark.parametrize("measurement,expected", [
    ((1, False, True), "grasp_lost"),
    ((1, True, False), "supported_by_environment"),
    ((4, True, True), "hold_timeout"),
])
def test_hold_cannot_pass_if_cube_lost_supported_or_inaccurate(diagnostic, monkeypatch, measurement, expected):
    result, _ = fake_attempt(diagnostic, monkeypatch, [measurement] * 4)
    assert result["status"] == expected
    assert result["consecutive_verified_hold_seconds"] == 0


def test_hold_requires_consecutive_in_tolerance_measurements(diagnostic, monkeypatch):
    sequence = [(1, True, True), (2, True, True), (4, True, True),
                (2, True, True), (1, True, True), (2.5, True, True)]
    result, recorder = fake_attempt(diagnostic, monkeypatch, sequence)
    assert result["status"] == "reached_and_held" and recorder.steps == 6
    assert result["consecutive_verified_hold_seconds"] == pytest.approx(.15)
    assert result["hold_max_error_mm"] == 2.5
    result, _ = fake_attempt(diagnostic, monkeypatch, sequence[:-1])
    assert result["status"] == "hold_timeout"


@pytest.mark.parametrize("failure", ["ik", "motion"])
def test_solver_failure_is_distinct_from_motion_timeout(diagnostic, monkeypatch, failure):
    error = (diagnostic.IKConvergenceError({"attempts": [{"position_error": .01,
              "tool_axis_error": 0, "tool_yaw_error": 0}]}) if failure == "ik"
             else diagnostic.DemonstrationFailure("Move exhausted its step budget."))
    result, _ = fake_attempt(diagnostic, monkeypatch, [(1, True, True)], move_exception=error)
    assert result["status"] == ("ik_failed" if failure == "ik" else "motion_timeout")
    assert result["stage"] == "waypoint_1"
    assert ("ik_diagnostics" in result) == (failure == "ik")


@pytest.mark.parametrize("partner", ["blue_cube_geom", "table", "wall_left"])
def test_contacts_with_environment_do_not_count_as_unsupported_hold(diagnostic, monkeypatch, partner):
    scene = {"orange": np.array([.3, 0, .06]), "gripper": np.array([.3, 0, .08]),
             "held": True, "orange_velocity": np.zeros(6)}
    recorder = SimpleNamespace(scene=lambda: scene, environment=None)
    monkeypatch.setattr(diagnostic, "cube_contacts", lambda _: ["fixed_jaw_box3", partner])
    result = diagnostic.snapshot_result(recorder, scene["orange"])
    assert result["held"] and not result["unsupported"]
