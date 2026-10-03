"""The configured jaw-plane tolerance follows reset and action IK paths."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import cartesian_actions
import environment as environment_module
from cartesian_actions import CartesianActionConfig, IK_BACKTRACKING_SCALES
from gym_environment import CubeStackGymEnvironment
from kinematics import DEFAULT_TOOL_YAW_TOLERANCE, IKConvergenceError, ToolAxisIKResult


SCENE = Path("scenes/so101_two_cube_stack.xml")


def test_yaw_tolerance_defaults_and_serialization() -> None:
    assert CartesianActionConfig().tool_yaw_tolerance == DEFAULT_TOOL_YAW_TOLERANCE
    configured = CartesianActionConfig(tool_yaw_tolerance=float(np.deg2rad(7.0)))
    assert CartesianActionConfig(**asdict(configured)) == configured
    legacy = asdict(configured)
    legacy.pop("tool_yaw_tolerance")
    assert CartesianActionConfig(**legacy).tool_yaw_tolerance == DEFAULT_TOOL_YAW_TOLERANCE


@pytest.mark.parametrize("invalid", [0.0, -0.1, np.nan, np.inf, -np.inf, [0.1]])
def test_action_config_rejects_invalid_yaw_tolerances(invalid) -> None:
    with pytest.raises(ValueError, match="tool_yaw_tolerance"):
        CartesianActionConfig(tool_yaw_tolerance=invalid)


def test_custom_tolerance_reaches_fixed_and_randomized_reset_ik(monkeypatch) -> None:
    tolerance = float(np.deg2rad(5.0))
    solve = environment_module.solve_position_and_tool_axis_ik
    calls = []

    def record_solve(*args, **kwargs):
        calls.append(kwargs)
        return solve(*args, **kwargs)

    monkeypatch.setattr(environment_module, "solve_position_and_tool_axis_ik", record_solve)
    environment = CubeStackGymEnvironment(
        scene_path=SCENE,
        action_config=CartesianActionConfig(tool_yaw_tolerance=tolerance),
        start_position_half_range=(0.001, 0.001, 0.0),
    )
    try:
        environment.reset(seed=18)
        assert len(calls) == 2  # Constructor's center and reset's sampled start.
        assert all(call["tool_yaw_tolerance"] == tolerance for call in calls)
        assert environment.simulation.tool_yaw_tolerance == tolerance
    finally:
        environment.close()


def test_reset_failure_reports_configured_tolerance(monkeypatch) -> None:
    tolerance = float(np.deg2rad(5.0))
    solve = environment_module.solve_position_and_tool_axis_ik

    def reject_yaw(*args, **kwargs):
        assert kwargs["tool_yaw_tolerance"] == tolerance
        return replace(solve(*args, **kwargs), tool_yaw_converged=False,
                       tool_yaw_error=tolerance * 2)

    monkeypatch.setattr(environment_module, "solve_position_and_tool_axis_ik", reject_yaw)
    with pytest.raises(IKConvergenceError) as caught:
        CubeStackGymEnvironment(
            scene_path=SCENE,
            action_config=CartesianActionConfig(tool_yaw_tolerance=tolerance),
        )
    assert caught.value.diagnostics["stage"] == "reset"
    assert caught.value.diagnostics["tool_yaw_tolerance"] == tolerance


def test_action_backtracking_and_failure_use_configured_tolerance(monkeypatch) -> None:
    tolerance = float(np.deg2rad(5.0))
    environment = CubeStackGymEnvironment(
        scene_path=SCENE,
        action_config=CartesianActionConfig(tool_yaw_tolerance=tolerance),
    )
    calls = []

    def reject_yaw(**kwargs):
        calls.append(kwargs)
        return ToolAxisIKResult(
            joint_positions=kwargs["initial_joint_positions"].copy(),
            position_converged=True, tool_axis_converged=True,
            position_error=0.0, tool_axis_error=0.0, iterations=1,
            tool_yaw_converged=False, tool_yaw_error=tolerance * 2,
        )

    try:
        environment.reset(seed=18)
        monkeypatch.setattr(cartesian_actions, "solve_position_and_tool_axis_ik", reject_yaw)
        with pytest.raises(IKConvergenceError) as caught:
            environment.step(np.array([0.1, 0.0, 0.0, 0.0]))
        assert len(calls) == len(IK_BACKTRACKING_SCALES)
        assert all(call["tool_yaw_tolerance"] == tolerance for call in calls)
        assert caught.value.diagnostics["tool_yaw_tolerance"] == tolerance
    finally:
        environment.close()
