#!/usr/bin/env mjpython

"""Validate the complete Gym stacking pipeline with a scripted trajectory.

Run visually from the repository root with:

    mjpython scripts/validate_scripted_gym_trajectory.py

Run without a viewer with:

    python scripts/validate_scripted_gym_trajectory.py --headless

The waypoint controller stands in for a policy, but every command still goes
through ``CubeStackGymEnvironment.step()``. The script therefore validates the
Cartesian action adapter, IK, MuJoCo stepping, privileged observations,
rewards, termination logic, and diagnostic ``info`` values together.
"""

import argparse
from collections import defaultdict
import math
import sys
import time
from typing import Any

import numpy as np


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from environment import PHYSICS_STEPS_PER_ACTION, StateSnapshot  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402


DEFAULT_RESET_SEED = 18
MAXIMUM_EPISODE_STEPS = 600

OPEN_GRIPPER_COMMAND = 1.0
CLOSED_GRIPPER_COMMAND = -1.0

# These small offsets are calibrated for the jaw and cube collision geometry
# in this temporary fixed-seed diagnostic.
GRASP_RADIAL_OFFSET = 0.007
GRASP_HEIGHT_OFFSET = 0.005
PREGRASP_HEIGHT = 0.10

LIFT_DISTANCE = 0.11
TRANSPORT_CUBE_HEIGHT = 0.14

# Match the policy's configured 2.5 mm per-axis Cartesian limit.
MAXIMUM_COMMANDED_POSITION_DELTA = 0.0025
WAYPOINT_POSITION_TOLERANCE = 0.0015
REQUIRED_CONSECUTIVE_NEAR_STEPS = 3
MAXIMUM_WAYPOINT_ACTIONS = 400

OPEN_HOLD_ACTIONS = 10
CLOSE_GRIPPER_ACTIONS = 40
CLOSED_HOLD_ACTIONS = 10
MAXIMUM_RELEASE_ACTIONS = 60

NUMERICAL_TOLERANCE = 1e-9

EXPECTED_POSITIVE_COMPONENTS = (
    "approach_orange_progress",
    "approach_orange_waypoint",
    "grasp_candidate",
    "grasp",
    "hold_orange_duration",
    "lift_orange_height",
    "move_toward_hover_progress",
    "stack_alignment_progress",
    "lower_toward_stack_progress",
    "successful_stack",
)


class RewardTrace:
    """Accumulate reward components and record their first activation."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.component_totals: defaultdict[str, float] = defaultdict(float)
        self.stage_component_totals: defaultdict[
            str,
            defaultdict[str, float],
        ] = defaultdict(lambda: defaultdict(float))
        self.first_positive_step: dict[str, int] = {}
        self.safe_lift_step: int | None = None
        self.hover_alignment_step: int | None = None

    def record(
        self,
        step_number: int,
        stage_name: str,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """Record one transition and print newly reached reward phases."""
        self.total_reward += reward

        for component_name, component_value in info[
            "reward_components"
        ].items():
            self.component_totals[component_name] += component_value
            self.stage_component_totals[stage_name][
                component_name
            ] += component_value

            if (
                component_name in EXPECTED_POSITIVE_COMPONENTS
                and component_value > NUMERICAL_TOLERANCE
                and component_name not in self.first_positive_step
            ):
                self.first_positive_step[component_name] = step_number
                print(
                    f"  action {step_number}: first positive "
                    f"{component_name} contribution "
                    f"({component_value:+.6f})"
                )

        if info["safe_lift_completed"] and self.safe_lift_step is None:
            self.safe_lift_step = step_number
            print(f"  action {step_number}: safe lift completed")

        if (
            info["hover_alignment_completed"]
            and self.hover_alignment_step is None
        ):
            self.hover_alignment_step = step_number
            print(f"  action {step_number}: hover alignment completed")


class GymTrajectoryRunner:
    """Execute and validate individual Gym transitions."""

    def __init__(
        self,
        environment: CubeStackGymEnvironment,
        viewer: Any | None,
    ) -> None:
        self.environment = environment
        self.viewer = viewer
        self.trace = RewardTrace()
        self.action_count = 0
        self.terminated = False
        self.truncated = False
        self.final_info: dict[str, Any] | None = None

    def current_state(self) -> StateSnapshot:
        """Return the Gym wrapper's snapshot for the current observation."""
        if self.environment.previous_state is None:
            raise RuntimeError("The Gym environment has not been reset.")
        return self.environment.previous_state

    def step(self, action: np.ndarray, stage_name: str) -> None:
        """Execute one action and validate the entire returned transition."""
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step an episode that has already ended.")
        if action.dtype != np.float32:
            raise RuntimeError("Scripted actions must use the float32 dtype.")
        if not self.environment.action_space.contains(action):
            raise RuntimeError(f"Action is outside action_space: {action}")

        previous_simulation_time = self.environment.simulation.data.time
        action_start_time = time.perf_counter()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.environment.step(action)
        self.action_count += 1

        if not self.environment.observation_space.contains(observation):
            raise RuntimeError(
                f"Action {self.action_count} returned an observation outside "
                "observation_space."
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"Action {self.action_count} returned a non-finite observation."
            )

        rebuilt_observation = self.environment.observation_builder.build(
            self.environment.simulation.get_state(),
        )
        if not np.array_equal(observation, rebuilt_observation):
            raise RuntimeError(
                f"Action {self.action_count} returned a stale observation."
            )

        if not math.isfinite(reward):
            raise RuntimeError(
                f"Action {self.action_count} returned a non-finite reward."
            )
        reward_components = info["reward_components"]
        for component_name, component_value in reward_components.items():
            if not math.isfinite(component_value):
                raise RuntimeError(
                    f"Reward component {component_name!r} is non-finite."
                )
        if not math.isclose(
            reward,
            sum(reward_components.values()),
            rel_tol=0.0,
            abs_tol=NUMERICAL_TOLERANCE,
        ):
            raise RuntimeError(
                "Total reward does not equal the sum of its components."
            )

        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise RuntimeError(
                "Gym termination and truncation flags must be booleans."
            )
        if terminated and truncated:
            raise RuntimeError(
                "A transition cannot be terminal and truncated in this "
                "environment."
            )
        if self.environment.simulation.data.time <= previous_simulation_time:
            raise RuntimeError("A Gym action did not advance simulation time.")
        if self.environment.episode_step_count != self.action_count:
            raise RuntimeError(
                "Gym episode_step_count does not match executed actions."
            )

        self.trace.record(
            self.action_count,
            stage_name,
            reward,
            info,
        )

        if not info["ik_position_converged"]:
            raise RuntimeError(
                f"Position IK failed during {stage_name!r} at action "
                f"{self.action_count}."
            )
        if info["is_failure"]:
            raise RuntimeError(
                "A cube fell off the table during the scripted trajectory."
            )
        if truncated:
            raise RuntimeError(
                "The scripted trajectory reached the episode step limit."
            )
        if terminated and not info["is_success"]:
            raise RuntimeError(
                "The episode terminated without successfully stacking."
            )

        self.terminated = terminated
        self.truncated = truncated
        self.final_info = info

        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit("Viewer closed before validation completed.")
            self.viewer.sync()

            control_interval = (
                PHYSICS_STEPS_PER_ACTION
                * self.environment.simulation.model.opt.timestep
            )
            remaining_time = control_interval - (
                time.perf_counter() - action_start_time
            )
            if remaining_time > 0.0:
                time.sleep(remaining_time)


def scripted_action(
    runner: GymTrajectoryRunner,
    target_position: np.ndarray,
    gripper_command: float,
) -> np.ndarray:
    """Return a normalized action pointing toward one XYZ waypoint."""
    current_target = (
        runner.environment.action_adapter.current_target_gripper_position
    )
    requested_delta = target_position - current_target
    bounded_delta = np.clip(
        requested_delta,
        -MAXIMUM_COMMANDED_POSITION_DELTA,
        MAXIMUM_COMMANDED_POSITION_DELTA,
    )
    normalized_delta = (
        bounded_delta
        / runner.environment.action_adapter.config.maximum_position_delta
    )
    return np.array(
        [*normalized_delta, gripper_command],
        dtype=np.float32,
    )


def print_stage_summary(
    runner: GymTrajectoryRunner,
    stage_name: str,
    starting_action_count: int,
) -> None:
    """Print the action count and nonzero reward totals for one stage."""
    stage_action_count = runner.action_count - starting_action_count
    stage_components = runner.trace.stage_component_totals[stage_name]
    stage_reward = sum(stage_components.values())
    print(
        f"Finished {stage_name!r}: {stage_action_count} actions, "
        f"reward={stage_reward:+.6f}"
    )

    for component_name, component_total in sorted(stage_components.items()):
        if abs(component_total) > NUMERICAL_TOLERANCE:
            print(f"    {component_name}: {component_total:+.6f}")

    final_info = runner.final_info
    if final_info is not None:
        print(
            "    status: "
            f"safe_lift={final_info['safe_lift_completed']}, "
            f"hover_aligned={final_info['hover_alignment_completed']}, "
            f"success={final_info['is_success']}"
        )


def move_to_waypoint(
    runner: GymTrajectoryRunner,
    target_position: np.ndarray,
    gripper_command: float,
    stage_name: str,
) -> None:
    """Move until the measured gripper remains near one XYZ target."""
    print(f"\nStarting {stage_name!r}")
    starting_action_count = runner.action_count
    consecutive_near_steps = 0

    for _ in range(MAXIMUM_WAYPOINT_ACTIONS):
        action = scripted_action(
            runner,
            target_position,
            gripper_command,
        )
        runner.step(action, stage_name)

        current_position = np.asarray(
            runner.current_state()["gripper_position"],
            dtype=float,
        )
        position_error = float(
            np.linalg.norm(target_position - current_position)
        )
        if position_error <= WAYPOINT_POSITION_TOLERANCE:
            consecutive_near_steps += 1
        else:
            consecutive_near_steps = 0

        if consecutive_near_steps >= REQUIRED_CONSECUTIVE_NEAR_STEPS:
            break
    else:
        current_position = np.asarray(
            runner.current_state()["gripper_position"],
            dtype=float,
        )
        position_error = float(
            np.linalg.norm(target_position - current_position)
        )
        raise RuntimeError(
            f"Could not reach {stage_name!r}; final position error was "
            f"{position_error * 1_000:.3f} mm."
        )

    print_stage_summary(
        runner,
        stage_name,
        starting_action_count,
    )


def hold_position(
    runner: GymTrajectoryRunner,
    gripper_command: float,
    action_count: int,
    stage_name: str,
    *,
    stop_when_successful: bool = False,
) -> None:
    """Hold Cartesian position while applying an absolute gripper command."""
    print(f"\nStarting {stage_name!r}")
    starting_action_count = runner.action_count
    action = np.array(
        [0.0, 0.0, 0.0, gripper_command],
        dtype=np.float32,
    )

    for _ in range(action_count):
        runner.step(action, stage_name)
        if stop_when_successful and runner.terminated:
            break

    print_stage_summary(
        runner,
        stage_name,
        starting_action_count,
    )


def require_positive_stage_component(
    runner: GymTrajectoryRunner,
    stage_name: str,
    component_name: str,
) -> None:
    """Raise unless a stage produced positive net progress for a component."""
    component_total = runner.trace.stage_component_totals[stage_name][
        component_name
    ]
    if component_total <= NUMERICAL_TOLERANCE:
        raise RuntimeError(
            f"Stage {stage_name!r} did not produce positive "
            f"{component_name!r}; total was {component_total:+.6f}."
        )


def validate_reward_sequence(runner: GymTrajectoryRunner) -> None:
    """Verify that every intended reward phase occurred in order."""
    expected_stage_components = (
        ("move above orange", "approach_orange_progress"),
        ("lower around orange", "approach_orange_waypoint"),
        ("close gripper", "grasp_candidate"),
        ("lift orange", "grasp"),
        ("hold orange", "hold_orange_duration"),
        ("lift orange", "lift_orange_height"),
        ("move above blue", "move_toward_hover_progress"),
        ("move above blue", "stack_alignment_progress"),
        ("lower onto blue", "lower_toward_stack_progress"),
        ("release orange", "successful_stack"),
    )
    for stage_name, component_name in expected_stage_components:
        require_positive_stage_component(
            runner,
            stage_name,
            component_name,
        )

    safe_lift_step = runner.trace.safe_lift_step
    hover_alignment_step = runner.trace.hover_alignment_step
    if safe_lift_step is None:
        raise RuntimeError("The safe-lift reward phase was never completed.")
    if hover_alignment_step is None:
        raise RuntimeError(
            "The hover-alignment reward phase was never completed."
        )

    approach_step = runner.trace.first_positive_step[
        "approach_orange_progress"
    ]
    waypoint_step = runner.trace.first_positive_step[
        "approach_orange_waypoint"
    ]
    grasp_candidate_step = runner.trace.first_positive_step[
        "grasp_candidate"
    ]
    grasp_step = runner.trace.first_positive_step["grasp"]
    hover_movement_step = runner.trace.first_positive_step[
        "move_toward_hover_progress"
    ]
    lowering_step = runner.trace.first_positive_step[
        "lower_toward_stack_progress"
    ]
    success_step = runner.trace.first_positive_step["successful_stack"]

    if not (
        approach_step
        < waypoint_step
        < grasp_candidate_step
        < grasp_step
        < safe_lift_step
        < hover_movement_step
        <= hover_alignment_step
        < lowering_step
        < success_step
    ):
        raise RuntimeError(
            "Reward phases did not occur in the expected scripted order."
        )

    for penalty_name in ("dropped_cube", "ik_failure"):
        penalty_total = runner.trace.component_totals[penalty_name]
        if not math.isclose(
            penalty_total,
            0.0,
            rel_tol=0.0,
            abs_tol=NUMERICAL_TOLERANCE,
        ):
            raise RuntimeError(
                f"Unexpected {penalty_name!r} total: {penalty_total:+.6f}."
            )


def run_trajectory(
    environment: CubeStackGymEnvironment,
    viewer: Any | None,
) -> GymTrajectoryRunner:
    """Execute the fixed-seed stack and validate its Gym reward trace."""
    initial_state = environment.previous_state
    if initial_state is None:
        raise RuntimeError("The Gym environment has not been reset.")

    orange_start = np.asarray(
        initial_state["orange_position"],
        dtype=float,
    ).copy()
    blue_start = np.asarray(
        initial_state["blue_position"],
        dtype=float,
    ).copy()

    print("Gym-wrapped scripted stacking validation")
    print(f"orange start: {orange_start}")
    print(f"blue start:   {blue_start}")

    orange_radial_direction = (
        orange_start[:2] / np.linalg.norm(orange_start[:2])
    )
    grasp_target = orange_start.copy()
    grasp_target[:2] -= GRASP_RADIAL_OFFSET * orange_radial_direction
    grasp_target[2] += GRASP_HEIGHT_OFFSET
    pregrasp_target = grasp_target + np.array(
        [0.0, 0.0, PREGRASP_HEIGHT]
    )

    runner = GymTrajectoryRunner(environment, viewer)

    move_to_waypoint(
        runner,
        pregrasp_target,
        OPEN_GRIPPER_COMMAND,
        "move above orange",
    )
    hold_position(
        runner,
        OPEN_GRIPPER_COMMAND,
        OPEN_HOLD_ACTIONS,
        "hold above orange",
    )
    move_to_waypoint(
        runner,
        grasp_target,
        OPEN_GRIPPER_COMMAND,
        "lower around orange",
    )
    hold_position(
        runner,
        OPEN_GRIPPER_COMMAND,
        OPEN_HOLD_ACTIONS,
        "hold grasp pose",
    )
    hold_position(
        runner,
        CLOSED_GRIPPER_COMMAND,
        CLOSE_GRIPPER_ACTIONS,
        "close gripper",
    )

    current_gripper_position = np.asarray(
        runner.current_state()["gripper_position"],
        dtype=float,
    )
    lift_target = current_gripper_position + np.array(
        [0.0, 0.0, LIFT_DISTANCE]
    )
    move_to_waypoint(
        runner,
        lift_target,
        CLOSED_GRIPPER_COMMAND,
        "lift orange",
    )
    hold_position(
        runner,
        CLOSED_GRIPPER_COMMAND,
        CLOSED_HOLD_ACTIONS,
        "hold lifted orange",
    )

    lifted_state = runner.current_state()
    held_cube_offset = (
        np.asarray(lifted_state["orange_position"], dtype=float)
        - np.asarray(lifted_state["gripper_position"], dtype=float)
    )
    blue_position = np.asarray(lifted_state["blue_position"], dtype=float)
    desired_transport_cube_position = np.array(
        [
            blue_position[0],
            blue_position[1],
            TRANSPORT_CUBE_HEIGHT,
        ]
    )
    transport_target = desired_transport_cube_position - held_cube_offset
    move_to_waypoint(
        runner,
        transport_target,
        CLOSED_GRIPPER_COMMAND,
        "move above blue",
    )
    hold_position(
        runner,
        CLOSED_GRIPPER_COMMAND,
        CLOSED_HOLD_ACTIONS,
        "hold above blue",
    )

    transported_state = runner.current_state()
    held_cube_offset = (
        np.asarray(transported_state["orange_position"], dtype=float)
        - np.asarray(transported_state["gripper_position"], dtype=float)
    )
    blue_position = np.asarray(
        transported_state["blue_position"],
        dtype=float,
    )
    stack_center_distance = (
        environment.simulation.success_config.expected_vertical_center_distance
    )
    desired_stacked_cube_position = np.array(
        [
            blue_position[0],
            blue_position[1],
            blue_position[2] + stack_center_distance,
        ]
    )
    place_target = desired_stacked_cube_position - held_cube_offset
    move_to_waypoint(
        runner,
        place_target,
        CLOSED_GRIPPER_COMMAND,
        "lower onto blue",
    )
    hold_position(
        runner,
        CLOSED_GRIPPER_COMMAND,
        CLOSED_HOLD_ACTIONS,
        "hold placement pose",
    )

    hold_position(
        runner,
        OPEN_GRIPPER_COMMAND,
        MAXIMUM_RELEASE_ACTIONS,
        "release orange",
        stop_when_successful=True,
    )

    if not runner.terminated:
        raise RuntimeError(
            "Opening the gripper did not produce a stable stack before the "
            "release action limit."
        )
    if runner.final_info is None or not runner.final_info["is_success"]:
        raise RuntimeError("The scripted trajectory did not report success.")

    validate_reward_sequence(runner)

    final_state = runner.current_state()
    center_difference = (
        np.asarray(final_state["orange_position"], dtype=float)
        - np.asarray(final_state["blue_position"], dtype=float)
    )

    print("\nCumulative nonzero reward components:")
    for component_name, component_total in sorted(
        runner.trace.component_totals.items()
    ):
        if abs(component_total) > NUMERICAL_TOLERANCE:
            print(f"  {component_name}: {component_total:+.6f}")
    print(f"Total episode reward: {runner.trace.total_reward:+.6f}")
    print(f"Gym actions: {runner.action_count}")
    print(f"orange - blue center difference: {center_difference}")
    print(
        "stable time: "
        f"{environment.simulation.stack_stable_time:.3f} s"
    )
    print("Validation passed: all Gym and reward phases behaved as expected.")

    return runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a fixed-seed stack through the complete Gym pipeline."
        )
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening the MuJoCo viewer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = CubeStackGymEnvironment(
        seed=DEFAULT_RESET_SEED,
        maximum_episode_steps=MAXIMUM_EPISODE_STEPS,
    )
    initial_observation, reset_info = environment.reset(
        seed=DEFAULT_RESET_SEED
    )
    if not environment.observation_space.contains(initial_observation):
        raise RuntimeError("Reset returned an invalid observation.")
    if not np.all(np.isfinite(initial_observation)):
        raise RuntimeError("Reset returned a non-finite observation.")
    if reset_info != {}:
        raise RuntimeError(f"Unexpected reset info: {reset_info}")

    if args.headless:
        run_trajectory(environment, viewer=None)
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        environment.simulation.model,
        environment.simulation.data,
    ) as viewer:
        viewer.sync()
        run_trajectory(environment, viewer)

        print("Close the viewer window to exit.")
        control_interval = (
            PHYSICS_STEPS_PER_ACTION
            * environment.simulation.model.opt.timestep
        )
        while viewer.is_running():
            viewer.sync()
            time.sleep(control_interval)


if __name__ == "__main__":
    main()
