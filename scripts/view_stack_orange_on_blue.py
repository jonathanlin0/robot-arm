#!/usr/bin/env mjpython

"""Run from the repository root with:

mjpython scripts/view_stack_orange_on_blue.py
"""

import sys
import time

import mujoco.viewer
import numpy as np


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from environment import (  # noqa: E402
    PHYSICS_STEPS_PER_ACTION,
    CubeStackEnvironment,
)
from kinematics import solve_position_and_tool_axis_ik  # noqa: E402


RESET_SEED = 18

OPEN_GRIPPER_TARGET = 1.0
CLOSED_GRIPPER_TARGET = -0.1

# These small offsets are calibrated for the jaw and cube collision geometry
# in this temporary fixed-seed demo.
GRASP_RADIAL_OFFSET = 0.007
GRASP_HEIGHT_OFFSET = 0.005
PREGRASP_HEIGHT = 0.10

LIFT_DISTANCE = 0.11
TRANSPORT_CUBE_HEIGHT = 0.14
STACK_CENTER_DISTANCE = 0.04
RETREAT_DISTANCE = 0.10


def arm_with_gripper(
    arm_joint_positions: np.ndarray,
    gripper_target: float,
) -> np.ndarray:
    return np.concatenate((arm_joint_positions, [gripper_target]))


def solve_or_raise(
    environment: CubeStackEnvironment,
    target_position: np.ndarray,
    stage_name: str,
) -> np.ndarray:
    current_joint_positions = environment.get_state()["joint_positions"]
    current_arm_joint_positions = current_joint_positions[:5]
    result = solve_position_and_tool_axis_ik(
        environment.model,
        current_arm_joint_positions,
        target_position,
    )
    if not result.position_converged:
        raise RuntimeError(
            f"IK failed during {stage_name!r}; "
            f"position error was {result.position_error:.6f} m."
        )

    print(
        f"{stage_name}: position error="
        f"{result.position_error * 1_000:.3f} mm, "
        f"tool-axis error={np.rad2deg(result.tool_axis_error):.2f} deg, "
        f"tool-axis converged={result.tool_axis_converged}"
    )
    return result.joint_positions


def step_and_render(
    environment: CubeStackEnvironment,
    viewer,
    joint_targets: np.ndarray,
) -> None:
    if not viewer.is_running():
        raise SystemExit

    action_start = time.perf_counter()
    environment.step_joint_targets(joint_targets)
    viewer.sync()

    control_interval = (
        PHYSICS_STEPS_PER_ACTION * environment.model.opt.timestep
    )
    remaining_time = control_interval - (
        time.perf_counter() - action_start
    )
    if remaining_time > 0:
        time.sleep(remaining_time)


def move_joint_targets(
    environment: CubeStackEnvironment,
    viewer,
    target_joint_positions: np.ndarray,
    duration: float,
) -> None:
    control_interval = (
        PHYSICS_STEPS_PER_ACTION * environment.model.opt.timestep
    )
    step_count = round(duration / control_interval)
    # Continue from the previous command rather than the lagging physical
    # joint positions, especially while the gripper is loaded by the cube.
    start_joint_targets = environment.data.ctrl.copy()

    for step_index in range(1, step_count + 1):
        interpolation_fraction = step_index / step_count
        # Smooth acceleration and deceleration at the ends of each motion.
        interpolation_fraction = (
            interpolation_fraction**2
            * (3.0 - 2.0 * interpolation_fraction)
        )
        interpolated_targets = start_joint_targets + (
            interpolation_fraction
            * (target_joint_positions - start_joint_targets)
        )
        step_and_render(environment, viewer, interpolated_targets)


def main() -> None:
    environment = CubeStackEnvironment(seed=RESET_SEED)
    initial_state = environment.reset()
    orange_start = initial_state["orange_position"].copy()
    blue_start = initial_state["blue_position"].copy()

    print("Fixed-seed scripted stacking demonstration")
    print(f"orange start: {orange_start}")
    print(f"blue start:   {blue_start}")

    # Move the target slightly toward the robot to align the jaw gap with the
    # cube. This is a demo-specific correction, not a general grasp planner.
    orange_radial_direction = (
        orange_start[:2] / np.linalg.norm(orange_start[:2])
    )
    grasp_target = orange_start.copy()
    grasp_target[:2] -= (
        GRASP_RADIAL_OFFSET * orange_radial_direction
    )
    grasp_target[2] += GRASP_HEIGHT_OFFSET
    pregrasp_target = grasp_target + np.array(
        [0.0, 0.0, PREGRASP_HEIGHT]
    )

    with mujoco.viewer.launch_passive(
        environment.model,
        environment.data,
    ) as viewer:
        viewer.sync()

        pregrasp_joints = solve_or_raise(
            environment,
            pregrasp_target,
            "move above orange",
        )
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(
                pregrasp_joints,
                OPEN_GRIPPER_TARGET,
            ),
            duration=3.0,
        )

        grasp_joints = solve_or_raise(
            environment,
            grasp_target,
            "lower around orange",
        )
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(grasp_joints, OPEN_GRIPPER_TARGET),
            duration=2.0,
        )

        print("close gripper")
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(grasp_joints, CLOSED_GRIPPER_TARGET),
            duration=2.0,
        )

        current_gripper_position = environment.get_state()[
            "gripper_position"
        ]
        lift_target = current_gripper_position + np.array(
            [0.0, 0.0, LIFT_DISTANCE]
        )
        lift_joints = solve_or_raise(
            environment,
            lift_target,
            "lift orange",
        )
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(lift_joints, CLOSED_GRIPPER_TARGET),
            duration=3.0,
        )

        # Measure where the held cube actually sits relative to the MuJoCo
        # gripper site instead of assuming the site is the cube center.
        lifted_state = environment.get_state()
        held_cube_offset = (
            lifted_state["orange_position"]
            - lifted_state["gripper_position"]
        )
        desired_transport_cube_position = np.array(
            [blue_start[0], blue_start[1], TRANSPORT_CUBE_HEIGHT]
        )
        transport_target = (
            desired_transport_cube_position - held_cube_offset
        )
        transport_joints = solve_or_raise(
            environment,
            transport_target,
            "move above blue",
        )
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(
                transport_joints,
                CLOSED_GRIPPER_TARGET,
            ),
            duration=4.0,
        )

        # Top-down alignment is a soft objective and may be unreachable at
        # transport height, so measure the held offset again before lowering.
        transported_state = environment.get_state()
        held_cube_offset = (
            transported_state["orange_position"]
            - transported_state["gripper_position"]
        )
        blue_position = transported_state["blue_position"]
        desired_stacked_cube_position = np.array(
            [
                blue_position[0],
                blue_position[1],
                blue_position[2] + STACK_CENTER_DISTANCE,
            ]
        )
        place_target = desired_stacked_cube_position - held_cube_offset
        place_joints = solve_or_raise(
            environment,
            place_target,
            "lower onto blue",
        )
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(place_joints, CLOSED_GRIPPER_TARGET),
            duration=3.0,
        )

        print("release orange")
        move_joint_targets(
            environment,
            viewer,
            arm_with_gripper(place_joints, OPEN_GRIPPER_TARGET),
            duration=2.0,
        )

        current_gripper_position = environment.get_state()[
            "gripper_position"
        ]
        retreat_target = current_gripper_position + np.array(
            [0.0, 0.0, RETREAT_DISTANCE]
        )
        retreat_joints = solve_or_raise(
            environment,
            retreat_target,
            "retreat from stack",
        )
        retreat_joint_targets = arm_with_gripper(
            retreat_joints,
            OPEN_GRIPPER_TARGET,
        )
        move_joint_targets(
            environment,
            viewer,
            retreat_joint_targets,
            duration=2.0,
        )
        # Keep the gripper clear while the success detector waits for a
        # released, motionless stack for its required stability duration.
        move_joint_targets(
            environment,
            viewer,
            retreat_joint_targets,
            duration=2.0,
        )

        final_state = environment.get_state()
        center_difference = (
            final_state["orange_position"]
            - final_state["blue_position"]
        )
        print(f"orange - blue center difference: {center_difference}")
        print(f"stable time: {environment.stack_stable_time:.3f} s")

        if not environment.stack_conditions_met():
            raise RuntimeError(
                "The scripted trajectory did not produce a stable stack."
            )

        print("Success: orange is stably stacked on blue.")
        print("Close the viewer window to exit.")

        while viewer.is_running():
            step_and_render(
                environment,
                viewer,
                retreat_joint_targets,
            )


if __name__ == "__main__":
    main()
