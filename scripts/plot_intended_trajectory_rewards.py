#!/usr/bin/env python3

"""Plot rewards from a successful scripted orange-cube pickup.

Run from the repository root with:

    python scripts/plot_intended_trajectory_rewards.py

The trajectory goes through the reward's pregrasp waypoint, descends to a
calibrated grasp pose, closes the gripper, and lifts orange until the current
pickup-only success condition is reached. Every command goes through
``CubeStackGymEnvironment.step()``, so the plotted values are the same rewards
seen by the learning algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import sys
from typing import Any, TYPE_CHECKING

import numpy as np


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from environment import MINIMUM_HOLD_TIME, PHYSICS_STEPS_PER_ACTION  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402


if TYPE_CHECKING:
    from matplotlib.figure import Figure


DEFAULT_RESET_SEED = 18
MAXIMUM_EPISODE_STEPS = 400

OPEN_GRIPPER_COMMAND = 1.0
CLOSED_GRIPPER_COMMAND = -1.0

# These offsets account for the jaw and cube collision geometry. The reward
# waypoint itself is read from StackRewardConfig rather than duplicated here.
GRASP_RADIAL_OFFSET = 0.007
GRASP_HEIGHT_OFFSET = 0.005
LIFT_DISTANCE = 0.11

WAYPOINT_POSITION_TOLERANCE = 0.0015
REQUIRED_CONSECUTIVE_NEAR_TICKS = 3
MAXIMUM_MOVE_TICKS = 200
OPEN_SETTLE_TICKS = 10
CLOSE_GRIPPER_TICKS = 40
EXTRA_HOLD_TICKS = 20

NUMERICAL_TOLERANCE = 1e-9

STAGE_COLORS = {
    "Approach waypoint": "tab:blue",
    "Descend to orange": "tab:orange",
    "Close gripper": "tab:purple",
    "Lift orange": "tab:green",
}

COMPONENT_COLORS = {
    "approach_orange_progress": "tab:blue",
    "approach_orange_waypoint": "tab:olive",
    "grasp_candidate": "tab:cyan",
    "grasp": "tab:purple",
    "hold_orange_duration": "tab:orange",
    "lift_orange_height": "tab:green",
    "successful_stack": "tab:red",
    "dropped_cube": "darkred",
    "ik_failure": "tab:brown",
    "action_magnitude": "tab:gray",
    "gripper_state_change": "tab:pink",
    "unproductive_close": "saddlebrown",
}

COMPONENT_LABELS = {
    "successful_stack": "terminal success bonus (successful_stack)",
}


@dataclass
class TrajectoryRewardTrace:
    """Per-policy-tick rewards and metadata from one scripted pickup."""

    seed: int
    action_interval: float
    simulation_times: list[float] = field(default_factory=list)
    total_rewards: list[float] = field(default_factory=list)
    component_rewards: dict[str, list[float]] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)
    success_tick: int | None = None
    final_hold_time: float = 0.0

    def record(
        self,
        *,
        simulation_time: float,
        total_reward: float,
        components: dict[str, float],
        stage: str,
        info: dict[str, Any],
    ) -> None:
        """Append one aligned transition to the trace."""
        previous_tick_count = len(self.total_rewards)

        for component_name in components:
            if component_name not in self.component_rewards:
                self.component_rewards[component_name] = [
                    0.0
                ] * previous_tick_count

        for component_name, values in self.component_rewards.items():
            values.append(float(components.get(component_name, 0.0)))

        self.simulation_times.append(float(simulation_time))
        self.total_rewards.append(float(total_reward))
        self.stages.append(stage)
        self.final_hold_time = float(info["orange_grasp_hold_time"])
        if info["is_success"]:
            self.success_tick = len(self.total_rewards)

    @property
    def cumulative_reward(self) -> float:
        """Return the sum of all recorded per-tick rewards."""
        return float(sum(self.total_rewards))


def _scripted_action(
    environment: CubeStackGymEnvironment,
    target_position: np.ndarray,
    gripper_command: float,
) -> np.ndarray:
    """Point one normalized Cartesian action toward a world-space target."""
    current_target = environment.action_adapter.current_target_gripper_position
    maximum_delta = environment.action_adapter.config.maximum_position_delta
    bounded_delta = np.clip(
        target_position - current_target,
        -maximum_delta,
        maximum_delta,
    )
    normalized_delta = bounded_delta / maximum_delta
    return np.array(
        [*normalized_delta, gripper_command],
        dtype=np.float32,
    )


def _step_and_record(
    environment: CubeStackGymEnvironment,
    trace: TrajectoryRewardTrace,
    action: np.ndarray,
    stage: str,
) -> tuple[bool, dict[str, Any]]:
    """Execute one Gym transition and record its exact reward breakdown."""
    _, reward, terminated, truncated, info = environment.step(action)

    current_state = environment.previous_state
    if current_state is None:
        raise RuntimeError("The Gym environment lost its current state.")

    components = info["reward_components"]
    if not math.isclose(
        reward,
        sum(components.values()),
        rel_tol=0.0,
        abs_tol=NUMERICAL_TOLERANCE,
    ):
        raise RuntimeError(
            "The transition reward does not equal its component sum."
        )

    trace.record(
        simulation_time=float(current_state["time"]),
        total_reward=reward,
        components=components,
        stage=stage,
        info=info,
    )

    if not info["ik_position_converged"]:
        raise RuntimeError(
            f"Position IK failed during {stage!r} at tick "
            f"{len(trace.total_rewards)}."
        )
    if info["is_failure"]:
        raise RuntimeError(
            f"A cube fell off the table during {stage!r}."
        )
    if truncated:
        raise RuntimeError(
            "The scripted pickup reached the episode step limit."
        )
    if terminated and not info["is_success"]:
        raise RuntimeError(
            "The episode terminated without satisfying pickup success."
        )

    return terminated, info


def _move_to_target(
    environment: CubeStackGymEnvironment,
    trace: TrajectoryRewardTrace,
    target_position: np.ndarray,
    gripper_command: float,
    stage: str,
    *,
    allow_success: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Move until the measured gripper remains close to one target."""
    consecutive_near_ticks = 0
    final_info: dict[str, Any] | None = None

    for _ in range(MAXIMUM_MOVE_TICKS):
        action = _scripted_action(
            environment,
            target_position,
            gripper_command,
        )
        terminated, final_info = _step_and_record(
            environment,
            trace,
            action,
            stage,
        )
        if terminated:
            if allow_success:
                return True, final_info
            raise RuntimeError(
                f"Pickup success occurred unexpectedly during {stage!r}."
            )

        current_state = environment.previous_state
        assert current_state is not None
        current_position = np.asarray(
            current_state["gripper_position"],
            dtype=float,
        )
        position_error = float(
            np.linalg.norm(target_position - current_position)
        )
        if position_error <= WAYPOINT_POSITION_TOLERANCE:
            consecutive_near_ticks += 1
        else:
            consecutive_near_ticks = 0

        if consecutive_near_ticks >= REQUIRED_CONSECUTIVE_NEAR_TICKS:
            return False, final_info

    current_state = environment.previous_state
    assert current_state is not None
    final_position = np.asarray(
        current_state["gripper_position"],
        dtype=float,
    )
    position_error = float(np.linalg.norm(target_position - final_position))
    raise RuntimeError(
        f"Could not complete {stage!r}; final position error was "
        f"{position_error * 1_000:.3f} mm."
    )


def _hold_position(
    environment: CubeStackGymEnvironment,
    trace: TrajectoryRewardTrace,
    gripper_command: float,
    tick_count: int,
    stage: str,
    *,
    stop_when_successful: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Hold XYZ while applying a persistent gripper command."""
    action = np.array(
        [0.0, 0.0, 0.0, gripper_command],
        dtype=np.float32,
    )
    final_info: dict[str, Any] | None = None

    for _ in range(tick_count):
        terminated, final_info = _step_and_record(
            environment,
            trace,
            action,
            stage,
        )
        if terminated:
            if stop_when_successful:
                return True, final_info
            raise RuntimeError(
                f"Pickup success occurred unexpectedly during {stage!r}."
            )

    assert final_info is not None
    return False, final_info


def _validate_successful_trace(trace: TrajectoryRewardTrace) -> None:
    """Require the scripted rollout to end on physical pickup success."""
    if trace.success_tick != len(trace.total_rewards):
        raise RuntimeError(
            "The scripted trajectory did not end on a successful pickup."
        )


def run_intended_pickup_trajectory(
    seed: int = DEFAULT_RESET_SEED,
) -> TrajectoryRewardTrace:
    """Run the intended pickup path and return its per-tick rewards."""
    environment = CubeStackGymEnvironment(
        seed=seed,
        maximum_episode_steps=MAXIMUM_EPISODE_STEPS,
    )
    environment.reset(seed=seed)

    initial_state = environment.previous_state
    if initial_state is None:
        raise RuntimeError("Reset did not produce an initial state.")

    action_interval = (
        PHYSICS_STEPS_PER_ACTION
        * environment.simulation.model.opt.timestep
    )
    trace = TrajectoryRewardTrace(
        seed=seed,
        action_interval=action_interval,
    )

    orange_start = np.asarray(
        initial_state["orange_position"],
        dtype=float,
    ).copy()

    pregrasp_target = orange_start.copy()
    pregrasp_target[2] += (
        environment.reward_config.approach_orange_height_offset
    )

    orange_radial_direction = (
        orange_start[:2] / np.linalg.norm(orange_start[:2])
    )
    grasp_target = orange_start.copy()
    grasp_target[:2] -= GRASP_RADIAL_OFFSET * orange_radial_direction
    grasp_target[2] += GRASP_HEIGHT_OFFSET

    _move_to_target(
        environment,
        trace,
        pregrasp_target,
        OPEN_GRIPPER_COMMAND,
        "Approach waypoint",
    )
    _, waypoint_info = _hold_position(
        environment,
        trace,
        OPEN_GRIPPER_COMMAND,
        OPEN_SETTLE_TICKS,
        "Approach waypoint",
    )
    if not waypoint_info["orange_pregrasp_waypoint_reached"]:
        raise RuntimeError(
            "The scripted controller did not activate the reward waypoint."
        )

    _move_to_target(
        environment,
        trace,
        grasp_target,
        OPEN_GRIPPER_COMMAND,
        "Descend to orange",
    )
    _hold_position(
        environment,
        trace,
        OPEN_GRIPPER_COMMAND,
        OPEN_SETTLE_TICKS,
        "Descend to orange",
    )

    _hold_position(
        environment,
        trace,
        CLOSED_GRIPPER_COMMAND,
        CLOSE_GRIPPER_TICKS,
        "Close gripper",
    )

    current_state = environment.previous_state
    assert current_state is not None
    lift_target = np.asarray(
        current_state["gripper_position"],
        dtype=float,
    ) + np.array([0.0, 0.0, LIFT_DISTANCE])

    succeeded, final_info = _move_to_target(
        environment,
        trace,
        lift_target,
        CLOSED_GRIPPER_COMMAND,
        "Lift orange",
        allow_success=True,
    )
    if not succeeded:
        maximum_hold_ticks = (
            math.ceil(MINIMUM_HOLD_TIME / action_interval)
            + EXTRA_HOLD_TICKS
        )
        succeeded, final_info = _hold_position(
            environment,
            trace,
            CLOSED_GRIPPER_COMMAND,
            maximum_hold_ticks,
            "Lift orange",
            stop_when_successful=True,
        )

    if not succeeded or not final_info["is_success"]:
        raise RuntimeError(
            "The scripted trajectory did not hold orange long enough to "
            "satisfy pickup success."
        )

    _validate_successful_trace(trace)
    return trace


def _stage_intervals(stages: list[str]) -> list[tuple[str, int, int]]:
    """Return inclusive one-indexed tick bounds for contiguous stages."""
    if not stages:
        return []

    intervals: list[tuple[str, int, int]] = []
    stage_start = 1
    current_stage = stages[0]
    for tick, stage in enumerate(stages[1:], start=2):
        if stage != current_stage:
            intervals.append((current_stage, stage_start, tick - 1))
            current_stage = stage
            stage_start = tick
    intervals.append((current_stage, stage_start, len(stages)))
    return intervals


def plot_reward_trace(trace: TrajectoryRewardTrace) -> Figure:
    """Create per-tick total and component plots for one reward trace."""
    import matplotlib.pyplot as plt

    if not trace.total_rewards:
        raise ValueError("Cannot plot an empty reward trace.")

    ticks = np.arange(1, len(trace.total_rewards) + 1)
    figure, (total_axis, component_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )

    for stage, start_tick, end_tick in _stage_intervals(trace.stages):
        color = STAGE_COLORS.get(stage, "lightgray")
        for axis in (total_axis, component_axis):
            axis.axvspan(
                start_tick - 0.5,
                end_tick + 0.5,
                color=color,
                alpha=0.08,
                linewidth=0.0,
                zorder=0,
            )
        total_axis.text(
            (start_tick + end_tick) / 2.0,
            0.98,
            stage,
            color=color,
            fontsize=9,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="top",
            transform=total_axis.get_xaxis_transform(),
        )

    total_axis.plot(
        ticks,
        trace.total_rewards,
        color="black",
        linewidth=1.6,
        label="Total reward",
    )
    total_axis.axhline(0.0, color="gray", linewidth=0.8)
    total_axis.set_yscale("symlog", linthresh=1e-3)
    total_axis.set_ylabel("Reward (symmetric-log scale)")
    total_axis.set_title("Total reward at each policy tick")
    total_axis.grid(alpha=0.25)
    total_axis.legend(loc="upper left")

    for component_name, values in trace.component_rewards.items():
        component_values = np.asarray(values, dtype=float)
        nonzero_mask = np.abs(component_values) > NUMERICAL_TOLERANCE
        if not np.any(nonzero_mask):
            continue

        label = COMPONENT_LABELS.get(
            component_name,
            component_name.replace("_", " "),
        )
        color = COMPONENT_COLORS.get(component_name)
        if np.count_nonzero(nonzero_mask) <= 3:
            component_axis.scatter(
                ticks[nonzero_mask],
                component_values[nonzero_mask],
                color=color,
                s=38,
                label=label,
                zorder=3,
            )
        else:
            component_axis.plot(
                ticks,
                component_values,
                color=color,
                linewidth=1.3,
                label=label,
            )

    component_axis.axhline(0.0, color="gray", linewidth=0.8)
    component_axis.set_yscale("symlog", linthresh=1e-3)
    component_axis.set_ylabel("Component reward (symmetric-log scale)")
    component_axis.set_xlabel(
        f"Policy tick ({trace.action_interval:.3f} simulated seconds each)"
    )
    component_axis.set_title("Nonzero reward components at each policy tick")
    component_axis.set_xlim(0.5, len(trace.total_rewards) + 0.5)
    component_axis.grid(alpha=0.25)
    component_axis.legend(loc="upper left", ncols=3, fontsize=8)

    figure.suptitle(
        "Successful intended orange-cube pickup reward trace "
        f"(seed {trace.seed}, return {trace.cumulative_reward:+.3f})"
    )
    return figure


def _print_summary(trace: TrajectoryRewardTrace) -> None:
    """Print the outcome and nonzero cumulative reward components."""
    print(
        f"Pickup succeeded at policy tick {trace.success_tick} "
        f"({trace.simulation_times[-1]:.3f} simulated seconds)."
    )
    print(f"Final uninterrupted hold: {trace.final_hold_time:.3f} seconds")
    print(f"Cumulative reward: {trace.cumulative_reward:+.6f}")
    print("Nonzero component totals:")
    for component_name, values in trace.component_rewards.items():
        component_total = float(sum(values))
        if abs(component_total) > NUMERICAL_TOLERANCE:
            print(f"  {component_name}: {component_total:+.6f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a successful scripted pickup and plot every per-tick reward."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RESET_SEED,
        help=(
            "Cube-placement seed for the scripted trajectory "
            f"(default: {DEFAULT_RESET_SEED})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    trace = run_intended_pickup_trajectory(seed=arguments.seed)
    _print_summary(trace)
    plot_reward_trace(trace)

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
