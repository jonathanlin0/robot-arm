"""Strict downward-orientation acceptance and fatal demonstration failures."""

from dataclasses import asdict
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pytest
import torch

import cartesian_actions
import environment as environment_module
from cartesian_actions import CartesianActionAdapter, CartesianActionConfig, IK_BACKTRACKING_SCALES
from environment import ARM_JOINT_NAMES, CubeStackEnvironment, PHYSICS_STEPS_PER_ACTION
from kinematics import DEFAULT_MAX_ITERATIONS, ToolAxisIKResult


@pytest.fixture
def environment() -> CubeStackEnvironment:
    simulation = CubeStackEnvironment(scene_path=Path("scenes/so101_two_cube_stack.xml"))
    simulation.reset(seed=17)
    return simulation


def solve_result(joints: np.ndarray, *, position: bool = True,
                 yaw: bool = True, downward: bool = True) -> ToolAxisIKResult:
    return ToolAxisIKResult(
        joint_positions=joints.copy(), position_converged=position,
        tool_axis_converged=downward, tool_yaw_converged=yaw,
        position_error=0.0002 if position else 0.01,
        tool_axis_error=0.02 if downward else 0.2,
        tool_yaw_error=0.001 if yaw else 0.1, iterations=19,
    )


@pytest.fixture(scope="module")
def generator():
    name = "generate_demonstrations_for_downward_tests"
    spec = importlib.util.spec_from_file_location(name, "scripts/generate_pickup_demonstrations.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def convergence_error(environment, monkeypatch):
    adapter = CartesianActionAdapter(environment)
    adapter.reset()
    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik",
                        lambda **kwargs: solve_result(kwargs["initial_joint_positions"], downward=False))
    with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
        adapter.step([0.0, 0.0, 0.0, -1.0])
    return caught.value


@pytest.mark.parametrize("enabled", [True, False])
def test_requirement_survives_serialized_action_configuration(enabled: bool) -> None:
    assert CartesianActionConfig().require_downward is True
    config = CartesianActionConfig(require_downward=enabled)
    restored = CartesianActionConfig(**json.loads(json.dumps(asdict(config))))
    assert restored.require_downward is enabled


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], np.array([True])])
def test_requirement_rejects_non_boolean_values(invalid) -> None:
    with pytest.raises(ValueError, match="require_downward"):
        CartesianActionConfig(require_downward=invalid)


@pytest.mark.parametrize("enabled", [True, False])
def test_adapter_passes_requirement_to_shared_solver(
    environment, monkeypatch, enabled: bool,
) -> None:
    adapter = CartesianActionAdapter(environment, CartesianActionConfig(require_downward=enabled))
    adapter.reset()
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return solve_result(kwargs["initial_joint_positions"])

    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", solve)
    monkeypatch.setattr(environment, "step_joint_targets", lambda targets: environment.get_state())
    adapter.step([0.2, 0.0, 0.0, 0.0])

    assert len(calls) == 1
    assert calls[0]["require_downward"] is enabled


def test_strict_backtracking_accepts_first_target_satisfying_all_constraints(
    environment, monkeypatch,
) -> None:
    adapter = CartesianActionAdapter(environment)
    adapter.reset()
    initial_target = adapter.current_target_gripper_position
    attempted = []
    applied = []

    def solve(**kwargs):
        attempted.append(kwargs["target_position"].copy())
        # Reaching XYZ and yaw is insufficient at the complete displacement.
        return solve_result(kwargs["initial_joint_positions"], downward=len(attempted) > 1)

    def step(targets):
        applied.append(targets.copy())
        return environment.get_state()

    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", solve)
    monkeypatch.setattr(environment, "step_joint_targets", step)
    result = adapter.step([1.0, -0.5, 0.0, -1.0])

    delta = adapter.config.maximum_position_delta * np.array([1.0, -0.5, 0.0])
    assert len(attempted) == 2
    np.testing.assert_allclose(attempted[0], initial_target + delta)
    np.testing.assert_allclose(attempted[1], initial_target + 0.5 * delta)
    np.testing.assert_allclose(adapter.current_target_gripper_position, attempted[1])
    np.testing.assert_allclose(result.target_gripper_position, attempted[1])
    assert result.ik_result.tool_axis_converged
    assert len(applied) == 1
    assert applied[0][-1] == adapter.config.closed_gripper_target


@pytest.mark.parametrize("failed_constraint", ["position", "yaw", "downward"])
def test_strict_failure_is_atomic_and_reports_all_solver_attempts(
    environment, monkeypatch, failed_constraint: str,
) -> None:
    adapter = CartesianActionAdapter(environment)
    adapter.reset()
    joints = environment.get_state()["joint_positions"][:len(ARM_JOINT_NAMES)]
    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik",
                        lambda **kwargs: solve_result(joints))
    monkeypatch.setattr(environment, "step_joint_targets", lambda targets: environment.get_state())
    adapter.step([0.5, 0.0, 0.0, 0.0])
    # Distinct previous/current targets also catch an accidental property update
    # before raising, even if the current accepted position is preserved.
    current_before = adapter.current_target_gripper_position
    previous_before = adapter.previous_target_gripper_position
    assert not np.array_equal(current_before, previous_before)
    qpos_before, qvel_before = environment.data.qpos.copy(), environment.data.qvel.copy()
    ctrl_before, time_before = environment.data.ctrl.copy(), environment.data.time
    targets = []

    def fail(**kwargs):
        targets.append(kwargs["target_position"].copy())
        return solve_result(joints, **{failed_constraint: False})

    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", fail)
    monkeypatch.setattr(environment, "step_joint_targets",
                        lambda targets: pytest.fail("Strict failure applied actuator commands."))
    monkeypatch.setattr(environment, "step_physics",
                        lambda steps: pytest.fail("Strict failure advanced physics."))

    with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
        adapter.step([1.0, -0.2, 0.0, -1.0])

    assert len(targets) == len(IK_BACKTRACKING_SCALES)
    delta = adapter.config.maximum_position_delta * np.array([1.0, -0.2, 0.0])
    for target, scale in zip(targets, IK_BACKTRACKING_SCALES, strict=True):
        np.testing.assert_allclose(target, current_before + scale * delta)
    np.testing.assert_array_equal(environment.data.qpos, qpos_before)
    np.testing.assert_array_equal(environment.data.qvel, qvel_before)
    np.testing.assert_array_equal(environment.data.ctrl, ctrl_before)
    assert environment.data.time == time_before
    np.testing.assert_array_equal(adapter.current_target_gripper_position, current_before)
    np.testing.assert_array_equal(adapter.previous_target_gripper_position, previous_before)

    diagnostics = caught.value.diagnostics
    assert diagnostics["require_downward"] is True
    assert len(diagnostics["attempts"]) == len(IK_BACKTRACKING_SCALES)
    np.testing.assert_allclose(diagnostics["requested_target_position"], current_before + delta)
    np.testing.assert_allclose(diagnostics["previous_target_position"], current_before)
    assert np.isfinite(diagnostics["joint_positions"]).all()
    assert np.isfinite(diagnostics["measured_tool_axis_error"])
    json.dumps(diagnostics, allow_nan=False)


def test_disabled_requirement_accepts_position_and_yaw_despite_tilt(
    environment, monkeypatch,
) -> None:
    adapter = CartesianActionAdapter(environment, CartesianActionConfig(require_downward=False))
    adapter.reset()
    attempted = []
    applied = []

    def solve(**kwargs):
        attempted.append(kwargs)
        return solve_result(kwargs["initial_joint_positions"], downward=False)

    def step(targets):
        applied.append(targets.copy())
        return environment.get_state()

    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", solve)
    monkeypatch.setattr(environment, "step_joint_targets", step)
    before = adapter.current_target_gripper_position
    result = adapter.step([1.0, 0.0, 0.0, -1.0])

    assert len(attempted) == len(applied) == 1
    assert not result.ik_result.tool_axis_converged
    np.testing.assert_allclose(adapter.current_target_gripper_position,
                               before + [adapter.config.maximum_position_delta, 0.0, 0.0])


def test_disabled_requirement_preserves_legacy_nonfatal_position_failure(
    environment, monkeypatch,
) -> None:
    adapter = CartesianActionAdapter(environment, CartesianActionConfig(require_downward=False))
    adapter.reset()
    before = adapter.current_target_gripper_position
    advanced = []
    monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik",
                        lambda **kwargs: solve_result(kwargs["initial_joint_positions"], position=False))
    monkeypatch.setattr(environment, "step_joint_targets",
                        lambda targets: pytest.fail("Rejected IK applied actuator commands."))
    monkeypatch.setattr(environment, "step_physics", lambda steps: advanced.append(steps))

    result = adapter.step([1.0, 0.0, 0.0, -1.0])

    assert advanced == [PHYSICS_STEPS_PER_ACTION]
    assert not result.ik_result.position_converged
    np.testing.assert_array_equal(adapter.current_target_gripper_position, before)


def test_episode_recorder_keeps_fatal_ik_exception_and_adds_stage_context(
    generator, convergence_error,
) -> None:
    recorder = generator.EpisodeRecorder(
        np.zeros(generator.PRIVILEGED_OBSERVATION_SIZE),
        {"target_gripper_position": np.array([0.3, 0.0, 0.1])},
    )

    def step(action):
        raise convergence_error

    environment = SimpleNamespace(step=step, episode_step_count=0)
    with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
        recorder.step(environment, np.array([0.0, 0.0, 0.0, -1.0]), "transport")

    assert caught.value is convergence_error
    assert convergence_error.diagnostics["stage"] == "transport"
    assert isinstance(convergence_error.diagnostics["episode_step"], int)
    assert not recorder.actions
    assert len(recorder.observations) == len(recorder.targets) == 1
    assert not recorder.finished


def test_generation_logs_and_stops_without_retry_preserving_saved_examples(
    generator, convergence_error, monkeypatch, tmp_path, capsys,
) -> None:
    episode = generator.PickupEpisode(
        observations=torch.zeros((2, generator.PRIVILEGED_OBSERVATION_SIZE)),
        accepted_targets=torch.zeros((2, 3)),
        actions=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        final_hold_time=0.0,
        final_stack_stable_time=generator.StackSuccessConfig().required_stable_time,
        stage_steps={"release": 1},
    )
    settings = {"action_config": asdict(CartesianActionConfig())}
    generator.save_episode(tmp_path, episode, uuid4(), "train", settings)
    previous_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    closed = []
    environment = SimpleNamespace(close=lambda: closed.append(True))
    attempted_seeds = []
    convergence_error.diagnostics.update(stage="transport", episode_step=42)

    def collect(env, seed):
        assert env is environment
        attempted_seeds.append(seed)
        if len(attempted_seeds) == 1:
            return episode
        raise convergence_error

    monkeypatch.setattr(generator, "make_environment", lambda: environment)
    monkeypatch.setattr(generator, "generation_settings", lambda env: settings)
    monkeypatch.setattr(generator, "collect_episode", collect)
    monkeypatch.setattr(generator, "TEST_FRACTION", 0.0)
    monkeypatch.setattr(generator, "MAXIMUM_ATTEMPTS_PER_EXAMPLE", 10)

    with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
        generator.generate_dataset(3, tmp_path)

    assert caught.value is convergence_error
    assert len(attempted_seeds) == 2  # The fatal second scene was not replaced.
    assert closed == [True]
    for path, contents in previous_files.items():
        assert path.read_bytes() == contents
    records = generator.read_manifest(tmp_path)
    assert len(records) == 2  # One existing example and one newly completed one.
    assert any(record["seed"] == attempted_seeds[0] for record in records)
    assert all(record["seed"] != attempted_seeds[1] for record in records)

    logs = list((tmp_path / "diagnostics").glob("ik-failure-*.json"))
    assert len(logs) == 1
    log = json.loads(logs[0].read_text())
    assert log["error_type"] == "IKConvergenceError"
    assert log["seed"] == attempted_seeds[1] == UUID(log["uuid"]).int
    assert log["attempts"] == 2
    assert log["completed_episodes"] == 1
    assert log["discarded_attempts"] == 0
    assert log["ik_convergence_failures"] == 1
    assert log["settings"] == json.loads(json.dumps(settings))
    assert log["diagnostics"]["stage"] == "transport"
    assert log["diagnostics"]["episode_step"] == 42
    output = capsys.readouterr()
    assert "[IK_CONVERGENCE_FAILURE]" in output.err
    assert "stage=transport" in output.err
    assert "step=42" in output.err
    assert "1 new completed episodes remain saved" in output.err
    assert "Discarded" not in output.out


def test_generation_cli_exits_nonzero_on_fatal_ik(
    generator, convergence_error, monkeypatch, tmp_path,
) -> None:
    called = []

    def generate(examples, output_directory):
        called.append((examples, output_directory))
        raise convergence_error

    monkeypatch.setattr(generator, "generate_dataset", generate)
    monkeypatch.setattr(sys, "argv", ["generate_pickup_demonstrations.py", "--examples", "3",
                                      "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as caught:
        generator.main()

    assert caught.value.code == 1
    assert called == [(3, tmp_path)]


@pytest.fixture
def strict_downward_tolerance(monkeypatch):
    """Keep unreachable-home tests independent of the user-configured limit."""
    tolerance = np.deg2rad(10.0)
    monkeypatch.setattr(
        environment_module, "solve_position_and_tool_axis_ik",
        partial(environment_module.solve_position_and_tool_axis_ik,
                tool_axis_tolerance=tolerance),
    )
    monkeypatch.setattr(environment_module, "DEFAULT_TOOL_AXIS_TOLERANCE", tolerance)
    return tolerance


def test_real_randomized_reset_failure_is_logged_without_advancing_simulation(
    generator, monkeypatch, capsys, strict_downward_tolerance,
) -> None:
    """A failing randomized reset must not replace the previous live state."""
    environment = generator.make_environment()
    assert environment.action_adapter.config.require_downward is True
    environment.reset(seed=1)
    # The old, high home is position-reachable but cannot point downward.
    # A nonzero half-range makes reset solve instead of reusing cached joints.
    environment.simulation.start_position = np.array([0.4, 0.0, 0.25])
    environment.simulation.start_position_half_range = np.array([1e-6, 0.0, 0.0])
    original_reset, original_close = environment.reset, environment.close
    resets, closed = [], []
    initial = {}

    def reset(*, seed=None, options=None):
        resets.append(seed)
        data = environment.simulation.data
        initial.update(qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                       ctrl=data.ctrl.copy(), time=data.time,
                       target=environment.action_adapter.current_target_gripper_position)
        return original_reset(seed=seed, options=options)

    def close():
        closed.append(True)
        original_close()

    monkeypatch.setattr(environment, "reset", reset)
    monkeypatch.setattr(environment, "close", close)
    monkeypatch.setattr(generator, "make_environment", lambda: environment)
    monkeypatch.setattr(generator, "uuid4", lambda: UUID(int=0))

    # Keep generated test artifacts inside the repository and remove only this
    # private test directory; never touch the existing demonstration dataset.
    with tempfile.TemporaryDirectory(prefix=".test-required-downward-",
                                     dir=generator.REPOSITORY_ROOT) as directory:
        output_directory = Path(directory)
        with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
            generator.generate_dataset(1, output_directory)

        diagnostics = caught.value.diagnostics
        assert diagnostics["stage"] == "reset"
        assert diagnostics["episode_step"] == 0
        assert diagnostics["tool_axis_tolerance"] == strict_downward_tolerance
        attempts = diagnostics["attempts"]
        assert len(attempts) == 1
        assert all(attempt["total_iterations"] == DEFAULT_MAX_ITERATIONS for attempt in attempts)
        assert all(not attempt["tool_axis_converged"] for attempt in attempts)
        assert all(attempt["tool_axis_error"] > strict_downward_tolerance for attempt in attempts)
        assert resets == [0]
        assert closed == [True]
        assert environment.episode_step_count == 0
        data = environment.simulation.data
        for name in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(data, name), initial[name])
        assert data.time == initial["time"] == 0.0
        np.testing.assert_array_equal(environment.action_adapter.current_target_gripper_position,
                                      initial["target"])
        assert not list(output_directory.rglob("*.pt"))
        assert not list((output_directory / "manifests").glob("*.json"))
        logs = list((output_directory / "diagnostics").glob("ik-failure-*.json"))
        assert len(logs) == 1
        log = json.loads(logs[0].read_text())
        assert log["seed"] == 0
        assert log["attempts"] == log["ik_convergence_failures"] == 1
        assert log["completed_episodes"] == log["discarded_attempts"] == 0
        assert log["diagnostics"] == diagnostics

    output = capsys.readouterr()
    assert "[IK_CONVERGENCE_FAILURE]" in output.err
    assert "stage=reset step=0" in output.err
    assert "Generation stopped" in output.err


def test_real_constructor_ik_failure_is_logged_without_retry(
    generator, monkeypatch, tmp_path, capsys, strict_downward_tolerance,
) -> None:
    """Home initialization can fail before an environment is returned."""
    original_factory = generator.CubeStackGymEnvironment
    constructions = []

    def unreachable_home(**kwargs):
        constructions.append(True)
        return original_factory(**kwargs, start_position=(0.4, 0.0, 0.25))

    monkeypatch.setattr(generator, "CubeStackGymEnvironment", unreachable_home)
    monkeypatch.setattr(generator, "uuid4", lambda: UUID(int=123))
    with pytest.raises(cartesian_actions.IKConvergenceError) as caught:
        generator.generate_dataset(3, tmp_path)

    assert constructions == [True]
    assert caught.value.diagnostics["stage"] == "reset"
    assert caught.value.diagnostics["episode_step"] == 0
    assert caught.value.diagnostics["tool_axis_tolerance"] == strict_downward_tolerance
    assert not list(tmp_path.rglob("*.pt"))
    logs = list((tmp_path / "diagnostics").glob("ik-failure-*.json"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text())
    assert record["seed"] == UUID(record["uuid"]).int == 123
    assert record["attempts"] == record["ik_convergence_failures"] == 1
    assert record["completed_episodes"] == record["discarded_attempts"] == 0
    assert record["settings"] == {}  # Construction failed before settings existed.
    assert record["diagnostics"] == caught.value.diagnostics
    output = capsys.readouterr()
    assert "[IK_CONVERGENCE_FAILURE]" in output.err
    assert "stage=reset step=0" in output.err
    assert "Generation stopped" in output.err
    assert "Discarded" not in output.out
