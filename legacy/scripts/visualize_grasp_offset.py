#!/usr/bin/env python

"""Visualize the cube-center error while the SO-101 closes around orange.

Run from the repository root:

    mjpython scripts/visualize_grasp_offset.py

The script saves a four-angle figure to ``grasp_offset_visualization.png``.
It freezes the claw 0.25 seconds before bilateral jaw contact. Red marks the
incorrect cube-center target and green marks the geometry-corrected target.
The orange cube is translucent so both internal points remain visible.
"""

import argparse
import os
from pathlib import Path
import sys


# Keep Matplotlib's generated cache inside the repository.
os.environ.setdefault("MPLCONFIGDIR", ".tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mujoco
import numpy as np


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from environment import PHYSICS_STEPS_PER_ACTION  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402


DEFAULT_OUTPUT_PATH = Path("grasp_offset_visualization.png")
DEFAULT_RESET_SEED = 18

PANEL_WIDTH = 640
PANEL_HEIGHT = 480

OPEN_GRIPPER_COMMAND = 1.0
MAXIMUM_COMMANDED_POSITION_DELTA = 0.0025
POSITION_TOLERANCE = 0.0015
REQUIRED_CONSECUTIVE_NEAR_STEPS = 3
MAXIMUM_APPROACH_ACTIONS = 180
INITIAL_SETTLING_ACTIONS = 15

# These values match the geometry-calibrated grasp pose used by the existing
# scripted pickup demonstration.
GRASP_RADIAL_OFFSET = 0.007
GRASP_HEIGHT_OFFSET = 0.005
SECONDS_BEFORE_BILATERAL_CONTACT = 0.25
CLOSING_RAMP_DURATION = 1.0
MAXIMUM_CLOSE_ACTIONS = 80

CORRECTED_TARGET_COLOR = np.array(
    [0.1, 1.0, 0.15, 1.0], dtype=np.float32
)
CURRENT_FRAME_COLOR = np.array(
    [1.0, 0.05, 0.05, 1.0], dtype=np.float32
)
CONNECTOR_COLOR = np.array(
    [1.0, 1.0, 1.0, 0.8], dtype=np.float32
)
MARKER_RADIUS = 0.003

CAMERA_VIEWS = (
    ("Front-left", 145.0, -12.0),
    ("Front-right", 215.0, -12.0),
    ("Rear-side", 285.0, -10.0),
    ("High oblique", 325.0, -45.0),
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the reset seed and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RESET_SEED,
        help=f"environment reset seed (default: {DEFAULT_RESET_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output PNG path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args(arguments)


def corrected_grasp_target(orange_position: np.ndarray) -> np.ndarray:
    """Return the calibrated gripper-frame target for this top grasp."""
    target = np.asarray(orange_position, dtype=float).copy()
    radial_distance = float(np.linalg.norm(target[:2]))
    if radial_distance == 0.0:
        raise RuntimeError("orange cube cannot be at the robot-base origin")

    # Shift toward the robot base in the horizontal plane, then upward in
    # world Z. The resulting point locates the rigid gripperframe site.
    target[:2] -= GRASP_RADIAL_OFFSET * target[:2] / radial_distance
    target[2] += GRASP_HEIGHT_OFFSET
    return target


def hold_current_target(environment: CubeStackGymEnvironment) -> None:
    """Let the cubes settle without moving the Cartesian target."""
    neutral_action = np.array(
        [0.0, 0.0, 0.0, OPEN_GRIPPER_COMMAND], dtype=np.float32
    )
    for _ in range(INITIAL_SETTLING_ACTIONS):
        _, _, terminated, truncated, _ = environment.step(neutral_action)
        if terminated or truncated:
            raise RuntimeError("environment ended while settling the scene")


def move_open_claw_to(
    environment: CubeStackGymEnvironment,
    target_position: np.ndarray,
) -> int:
    """Move the open claw to one Cartesian target through the real adapter."""
    consecutive_near_steps = 0

    for action_count in range(1, MAXIMUM_APPROACH_ACTIONS + 1):
        current_target = (
            environment.action_adapter.current_target_gripper_position
        )
        normalized_delta = np.clip(
            (target_position - current_target)
            / MAXIMUM_COMMANDED_POSITION_DELTA,
            -1.0,
            1.0,
        )
        action = np.array(
            [*normalized_delta, OPEN_GRIPPER_COMMAND], dtype=np.float32
        )
        _, _, terminated, truncated, info = environment.step(action)

        if not info["ik_position_converged"]:
            raise RuntimeError(
                f"position IK failed on approach action {action_count}"
            )
        if terminated or truncated:
            raise RuntimeError("environment ended while approaching orange")

        current_position = np.asarray(
            environment.previous_state["gripper_position"], dtype=float
        )
        position_error = float(
            np.linalg.norm(target_position - current_position)
        )
        consecutive_near_steps = (
            consecutive_near_steps + 1
            if position_error <= POSITION_TOLERANCE
            else 0
        )
        if consecutive_near_steps >= REQUIRED_CONSECUTIVE_NEAR_STEPS:
            return action_count

    raise RuntimeError(
        "open claw did not reach the grasp pose within "
        f"{MAXIMUM_APPROACH_ACTIONS} actions"
    )


def capture_full_physics_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """Copy enough MuJoCo state to restore an earlier rendered instant."""
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=float)
    mujoco.mj_getState(model, data, state, state_spec)
    return state


def freeze_before_bilateral_contact(
    environment: CubeStackGymEnvironment,
) -> tuple[int, float]:
    """Restore the instant 0.25 seconds before both jaws touch orange."""
    simulation = environment.simulation
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    # Use the environment's observed action duration rather than assuming a
    # particular video frame rate. One action is currently 0.05 seconds.
    control_interval = float(
        simulation.model.opt.timestep
        * PHYSICS_STEPS_PER_ACTION
    )
    actions_before_contact = max(
        1,
        round(SECONDS_BEFORE_BILATERAL_CONTACT / control_interval),
    )
    ramp_action_count = max(
        actions_before_contact + 1,
        round(CLOSING_RAMP_DURATION / control_interval),
    )
    snapshots = [
        capture_full_physics_state(simulation.model, simulation.data)
    ]
    starting_joint_targets = simulation.data.ctrl.copy()
    open_gripper_target = environment.action_adapter.config.open_gripper_target
    closed_gripper_target = (
        environment.action_adapter.config.closed_gripper_target
    )

    for close_action_count in range(1, MAXIMUM_CLOSE_ACTIONS + 1):
        closing_fraction = min(
            close_action_count / ramp_action_count,
            1.0,
        )
        joint_targets = starting_joint_targets.copy()
        joint_targets[-1] = (
            open_gripper_target
            + closing_fraction
            * (closed_gripper_target - open_gripper_target)
        )
        simulation.step_joint_targets(joint_targets)
        snapshots.append(
            capture_full_physics_state(simulation.model, simulation.data)
        )

        state = simulation.get_state()
        bilateral_contact = bool(
            state["orange_touches_fixed_jaw"]
            and state["orange_touches_moving_jaw"]
        )
        if bilateral_contact:
            freeze_index = max(
                0,
                close_action_count - actions_before_contact,
            )
            mujoco.mj_setState(
                simulation.model,
                simulation.data,
                snapshots[freeze_index],
                state_spec,
            )
            mujoco.mj_forward(simulation.model, simulation.data)
            actual_lead_time = (
                close_action_count - freeze_index
            ) * control_interval
            return close_action_count, actual_lead_time

    raise RuntimeError(
        "the jaws did not establish bilateral contact within "
        f"{MAXIMUM_CLOSE_ACTIONS} close actions"
    )


def create_camera(
    lookat: np.ndarray,
    azimuth: float,
    elevation: float,
) -> mujoco.MjvCamera:
    """Create one close camera centered on the claw and cube."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = 0.19
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def add_sphere_marker(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    color: np.ndarray,
) -> None:
    """Add one bright spherical marker to a populated render scene."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("render scene has no capacity for another marker")

    marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, MARKER_RADIUS, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    marker.emission = 1.0
    marker.category = mujoco.mjtCatBit.mjCAT_DECOR
    marker.segid = -1
    scene.ngeom += 1


def add_connector(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    """Draw the correction vector from the wrong target to the right one."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("render scene has no capacity for the connector")

    connector = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        connector,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.ones(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        CONNECTOR_COLOR,
    )
    mujoco.mjv_connector(
        connector,
        mujoco.mjtGeom.mjGEOM_LINE,
        3,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    connector.emission = 1.0
    connector.category = mujoco.mjtCatBit.mjCAT_DECOR
    connector.segid = -1
    scene.ngeom += 1


def render_views(
    environment: CubeStackGymEnvironment,
    incorrect_target: np.ndarray,
    corrected_target: np.ndarray,
    current_gripperframe: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Render the marked pose from each configured camera angle."""
    simulation = environment.simulation
    renderer = mujoco.Renderer(
        simulation.model,
        height=PANEL_HEIGHT,
        width=PANEL_WIDTH,
    )
    lookat = (incorrect_target + current_gripperframe) / 2.0
    frames: list[tuple[str, np.ndarray]] = []

    try:
        for title, azimuth, elevation in CAMERA_VIEWS:
            camera = create_camera(lookat, azimuth, elevation)
            renderer.update_scene(simulation.data, camera=camera)
            add_connector(
                renderer.scene,
                incorrect_target,
                corrected_target,
            )
            add_sphere_marker(
                renderer.scene,
                corrected_target,
                CORRECTED_TARGET_COLOR,
            )
            add_sphere_marker(
                renderer.scene,
                incorrect_target,
                CURRENT_FRAME_COLOR,
            )
            frames.append((title, renderer.render().copy()))
    finally:
        renderer.close()

    return frames


def save_figure(
    frames: list[tuple[str, np.ndarray]],
    output_path: Path,
    incorrect_target: np.ndarray,
    corrected_target: np.ndarray,
    current_gripperframe: np.ndarray,
    lead_time: float,
) -> None:
    """Arrange the four views, legend, and measured positions in one PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), dpi=150)

    for axis, (title, frame) in zip(axes.flat, frames, strict=True):
        axis.imshow(frame)
        axis.set_title(title, fontsize=13)
        axis.axis("off")

    tracking_error_mm = float(
        np.linalg.norm(current_gripperframe - corrected_target)
        * 1_000.0
    )
    correction_distance_mm = float(
        np.linalg.norm(corrected_target - incorrect_target) * 1_000.0
    )
    figure.suptitle(
        f"Closing jaws {lead_time:.2f} s before bilateral cube contact",
        fontsize=18,
        y=0.985,
    )
    figure.legend(
        handles=(
            Patch(
                color=CORRECTED_TARGET_COLOR,
                label=(
                    "Corrected gripperframe target: "
                    "7 mm toward base, 5 mm above cube center"
                ),
            ),
            Patch(
                color=CURRENT_FRAME_COLOR,
                label="Incorrect gripperframe target: cube center",
            ),
        ),
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.055,
        (
            "Orange is translucent to reveal both targets. "
            f"Red-to-green correction={correction_distance_mm:.1f} mm; "
            f"vertical part={GRASP_HEIGHT_OFFSET * 1_000:.0f} mm; "
            f"horizontal part={GRASP_RADIAL_OFFSET * 1_000:.0f} mm; "
            f"actual gripperframe error from green={tracking_error_mm:.2f} mm."
        ),
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.105, 1.0, 0.955))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main(arguments: list[str] | None = None) -> None:
    """Pose the arm, render four marked views, and save the figure."""
    options = parse_arguments(arguments)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    environment = CubeStackGymEnvironment(maximum_episode_steps=400)
    try:
        environment.reset(seed=options.seed)
        hold_current_target(environment)

        settled_orange_position = np.asarray(
            environment.previous_state["orange_position"], dtype=float
        ).copy()
        initial_corrected_target = corrected_grasp_target(
            settled_orange_position
        )
        approach_actions = move_open_claw_to(
            environment,
            initial_corrected_target,
        )
        hold_current_target(environment)
        contact_action, lead_time = freeze_before_bilateral_contact(
            environment
        )

        frozen_orange_position = (
            environment.simulation.data.body("orange_cube").xpos.copy()
        )
        incorrect_target = frozen_orange_position.copy()
        target = corrected_grasp_target(frozen_orange_position)
        current_gripperframe = np.asarray(
            environment.simulation.data.site("gripperframe").xpos,
            dtype=float,
        ).copy()

        # The target is within the opaque cube volume. Transparency is only a
        # visual aid and has no effect on the physics used to pose the arm.
        orange_geom = environment.simulation.model.geom("orange_cube_geom")
        orange_geom.rgba[3] = 0.35
        # Blue is irrelevant to this pickup-target comparison and can block a
        # close camera view. Hide it only after the physical pose is frozen.
        blue_geom = environment.simulation.model.geom("blue_cube_geom")
        blue_geom.rgba[3] = 0.0

        frames = render_views(
            environment,
            incorrect_target,
            target,
            current_gripperframe,
        )
        save_figure(
            frames,
            options.output,
            incorrect_target,
            target,
            current_gripperframe,
            lead_time,
        )
    finally:
        environment.close()

    print(f"Saved {options.output}")
    print(f"Actions to corrected grasp pose: {approach_actions}")
    print(f"Bilateral contact occurred on close action: {contact_action}")
    print(f"Freeze-frame lead time: {lead_time:.2f} s")
    print(f"Incorrect cube-center target (red): {incorrect_target}")
    print(f"Corrected grasp target (green): {target}")
    print(f"Actual gripperframe at freeze frame: {current_gripperframe}")


if __name__ == "__main__":
    main()
