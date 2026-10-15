"""Task-scene coverage uses measured physics rather than phase names alone."""

from pathlib import Path

import numpy as np
import pytest

from cartesian_controller import IKConvergenceError
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from schemas.config import load_config
from vision_scenarios import ScriptedCarryScenario, ScriptedVisionScenario


@pytest.fixture
def scenario():
    project = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
    backend = SimJointCommandBackend(environment=environment)
    return ScriptedVisionScenario(
        environment=environment, backend=backend,
        controller_config=project.cartesian_controller, config=project.environment.vision_data.task,
    )


@pytest.fixture
def carry_scenario():
    project = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
    backend = SimJointCommandBackend(environment=environment)
    return ScriptedCarryScenario(
        environment=environment, backend=backend,
        controller_config=project.cartesian_controller, config=project.environment.vision_data.task,
        carry_config=project.environment.vision_data.carry, rng=np.random.default_rng(2),
    )


def test_teacher_sends_commands_without_stepping_or_repositioning_cubes(scenario):
    environment = scenario.environment
    before = environment.data.qpos.copy()
    scenario.step()
    assert environment.data.time == 0.0
    np.testing.assert_array_equal(environment.data.qpos, before)
    assert scenario.phase == "approach"
    assert not scenario.metadata()["lift_seen"]


def test_real_teacher_reaches_carry_drops_and_recovers_before_placement(scenario):
    held_carry_ticks = 0
    first_recovery_cube_position = None
    for _ in range(700):
        scenario.step()
        if scenario.phase == "carry" and scenario._scene()["held"]:
            held_carry_ticks += 1
            assert scenario._scene()["orange"][2] > 0.06
        if scenario.phase == "recover_settle" and first_recovery_cube_position is None:
            first_recovery_cube_position = scenario._scene()["orange"].copy()
        scenario.backend.advance(duration_seconds=0.05)
        if scenario.phase == "complete":
            break
    metadata = scenario.metadata()
    assert scenario.phase == "complete"
    assert held_carry_ticks >= 10
    assert metadata["grasp_seen"] and metadata["lift_seen"]
    assert metadata["intentional_drop_started"] and metadata["drop_seen"]
    assert metadata["recovery_count"] == 1
    assert {"approach", "descend", "close", "lift", "carry", "drop", "recover_settle",
            "place", "release", "settle_stack"} <= set(metadata["phases_seen"])
    assert first_recovery_cube_position is not None
    # This is vision coverage, not a stacking success evaluator. Unstable
    # placement is retained and truthfully reported instead of relabeled.
    if not metadata["stack_conditions_now"]:
        assert metadata["last_failure"] == "placement: no measured released stack"


def test_phase_timeout_is_recorded_and_recovers_without_claiming_grasp(scenario):
    scenario.environment.data.time = scenario.config.phase_timeout_seconds + 0.1
    scenario.step()
    metadata = scenario.metadata()
    assert metadata["phase"] == "recover_settle"
    assert metadata["last_failure"] == "approach: phase timeout"
    assert not metadata["grasp_seen"] and not metadata["lift_seen"]
    scenario.environment.data.time += scenario.config.settle_seconds + 0.1
    scenario.step()
    assert scenario.phase == "approach"
    assert scenario.metadata()["recovery_count"] == 1


def test_ik_failure_propagates_without_being_reported_as_task_success(scenario, monkeypatch):
    failure = IKConvergenceError({"attempts": [{"position_error": 1.0, "tool_axis_error": 1.0,
                                               "tool_yaw_error": 1.0}]})

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(scenario.controller, "compute", fail)
    with pytest.raises(IKConvergenceError) as exception:
        scenario.step()
    assert exception.value is failure
    assert scenario.environment.data.time == 0
    assert not scenario.metadata()["grasp_seen"]


def test_real_carry_teacher_keeps_visiting_targets_for_full_episode(carry_scenario):
    scenario = carry_scenario
    held_positions = []
    for _ in range(900):  # 45 seconds, without camera or wall-clock waits.
        scenario.step()
        scene = scenario._scene()
        if scenario.phase == "carry":
            # Intentional opening belongs to recovery, never the carrying phase.
            assert scenario.environment.data.actuator("gripper").ctrl[0] == pytest.approx(
                scenario.controller.config.closed_gripper_target
            )
            if scene["held"]:
                held_positions.append(scene["orange"])
        scenario.backend.advance(duration_seconds=0.05)
    metadata = scenario.metadata()
    assert scenario.environment.data.time == pytest.approx(45.0)
    assert metadata["grasp_seen"] and metadata["lift_seen"]
    assert metadata["carry_targets_reached"] >= 3
    assert len(held_positions) >= 200
    assert np.ptp(np.asarray(held_positions), axis=0).min() > 0.02
    assert not metadata["intentional_drop_started"]
    assert not {"drop", "place", "release", "settle_stack", "complete"} & set(metadata["phases_seen"])


def test_carry_samples_cube_targets_and_resets_timeout_after_each_arrival(carry_scenario, monkeypatch):
    scenario = carry_scenario
    scene = scenario._scene()
    scene.update(time=0.0, held=True, orange=np.array([0.30, 0.0, 0.10]),
                 gripper=np.array([0.29, 0.0, 0.12]))
    issued = []
    monkeypatch.setattr(scenario, "_scene", lambda: scene)
    monkeypatch.setattr(scenario, "_issue", lambda target, gripper: issued.append((target.copy(), gripper)))
    scenario._enter("carry", 0.0)
    scenario.step()
    first_target = scenario._target.copy()
    bounds = np.asarray((scenario.carry_config.x_range, scenario.carry_config.y_range,
                         scenario.carry_config.z_range))
    assert np.all((bounds[:, 0] <= first_target) & (first_target <= bounds[:, 1]))
    np.testing.assert_allclose(issued[-1][0], first_target - (scene["orange"] - scene["gripper"]))
    assert issued[-1][1] == -1.0
    scene.update(time=5.5, orange=first_target.copy())
    scenario.step()
    assert scenario.metadata()["carry_targets_reached"] == 1
    assert scenario._target is None
    scene["time"] = 6.1
    scenario.step()
    assert scenario.phase == "carry"  # Not a six-second timeout from the first target.
    assert not np.array_equal(scenario._target, first_target)
    assert scenario.metadata()["last_failure"] is None


def test_carry_contact_loss_recovers_toward_current_cube_position(carry_scenario, monkeypatch):
    scenario = carry_scenario
    scene = scenario._scene()
    scene.update(time=0.05, held=False)
    issued = []
    monkeypatch.setattr(scenario, "_scene", lambda: scene)
    monkeypatch.setattr(scenario, "_issue", lambda target, gripper: issued.append((target, gripper)))
    scenario._lift_seen = True
    scenario._enter("carry", 0.0)
    scenario.step()
    assert scenario.phase == "carry"
    assert issued[-1][1] == -1.0
    scene["time"] = 0.10
    scenario.step()
    assert scenario.phase == "recover_settle"
    assert scenario.metadata()["last_failure"] == "carry: measured grasp lost"
    assert scenario.metadata()["drop_seen"]
    assert not scenario.metadata()["intentional_drop_started"]
    scene.update(time=0.10 + scenario.config.settle_seconds + 0.01,
                 orange=np.array([0.27, -0.04, 0.02]))
    scenario.step()
    assert scenario.phase == "approach"
    assert scenario.metadata()["recovery_count"] == 1
    scene["time"] += 0.05
    scenario.step()
    np.testing.assert_allclose(issued[-1][0], scene["orange"] + [0.0, 0.0, scenario.config.waypoint_height])


def test_carry_timeout_chooses_another_target_without_opening(carry_scenario, monkeypatch):
    scenario = carry_scenario
    scene = scenario._scene()
    scene.update(time=0.0, held=True)
    issued = []
    monkeypatch.setattr(scenario, "_scene", lambda: scene)
    monkeypatch.setattr(scenario, "_issue", lambda target, gripper: issued.append((target, gripper)))
    scenario._enter("carry", 0.0)
    scenario.step()
    old_target = scenario._target.copy()
    scene["time"] = scenario.config.phase_timeout_seconds + 0.05
    scenario.step()
    assert scenario.phase == "carry"
    assert not np.array_equal(scenario._target, old_target)
    assert scenario.metadata()["last_failure"] == "carry: phase timeout"
    assert all(gripper == -1.0 for _, gripper in issued)
    assert scenario.metadata()["recovery_count"] == 0
