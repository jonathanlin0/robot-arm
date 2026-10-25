"""Physically test carrying orange to the spawn-area corners at several heights.

    .venv/bin/python scripts/diagnostics/check_held_cube_workspace.py

Defaults to cube-center Z = 3, 4, 5, 6, 7, 8 cm and the XY bounds in
environment.vision_data. Picks up orange at the center using the existing
teacher, with blue at the opposite corner. Each attempt restores that verified
physical grasp, then moves using normal IK, actuators, contacts, and gravity.
No object is attached or teleported during an attempt. Physics runs without
wall-clock sleeps; neither learned model is needed and no dataset is generated.

A pass requires the actual cube center to stay within the configured teacher
tolerance, held by both jaws and unsupported, for 0.5 simulated seconds.
Failed direct routes are retried via XY-first and height-first waypoints.
Failures describe these tested routes/solver settings, not mathematical proof
that all arm configurations fail. Corner tests do not certify the interior.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from math import ceil
from pathlib import Path
from time import perf_counter
import json
import sys

import mujoco
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_demonstrations import (
    CLOSED_GRIPPER, DemonstrationFailure, EpisodeRecorder, action_toward,
    move_held_cube_to, pickup_from_current_pose, reset_demonstration_environment,
)
from environment import CubeStackEnvironment
from kinematics import IKConvergenceError
from schemas.config import load_config


ROUTES = ("direct", "xy_then_z", "z_then_xy")


def spawn_corners(project):
    config = project.environment.vision_data
    return [(f"x_{x_name}_y_{y_name}", np.array([x, y], dtype=float))
            for x_name, x in zip(("min", "max"), config.cube_x_range)
            for y_name, y in zip(("min", "max"), config.cube_y_range)]


def cube_contacts(environment):
    """Return active contact partners for orange, including unintended supports."""
    orange = environment.model.geom("orange_cube_geom").id
    partners = set()
    for contact in environment.data.contact[:environment.data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        if first == orange or second == orange:
            partner = second if first == orange else first
            partners.add(mujoco.mj_id2name(environment.model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                         or f"geom_{partner}")
    return sorted(partners)


def snapshot_result(recorder, target):
    scene = recorder.scene()
    partners = cube_contacts(recorder.environment)
    # The teacher's held check already excludes table support. Also exclude
    # support from blue or enclosure walls for this workspace diagnostic.
    unsupported = not any(name == "blue_cube_geom" or name == "table" or name.startswith("wall_")
                          for name in partners)
    return {
        "actual_cube_xyz_m": scene["orange"].tolist(),
        "actual_gripper_xyz_m": scene["gripper"].tolist(),
        "position_error_mm": float(np.linalg.norm(scene["orange"] - target) * 1000),
        "axis_error_mm": ((scene["orange"] - target) * 1000).tolist(),
        "held": bool(scene["held"]), "unsupported": unsupported,
        "orange_contact_partners": partners,
        "cube_linear_speed_m_s": float(np.linalg.norm(scene["orange_velocity"][:3])),
    }


def prepare_grasp(project, environment, center_xy, blue_xy):
    reset_demonstration_environment(environment, project, np.random.default_rng(0))
    for name, xy in (("orange", center_xy), ("blue", blue_xy)):
        joint = environment.data.joint(f"{name}_cube_joint")
        joint.qpos[:2] = xy
        joint.qvel[:] = 0
    mujoco.mj_forward(environment.model, environment.data)
    recorder = EpisodeRecorder(project, environment, seed=0)
    pickup_from_current_pose(recorder)
    scene = recorder.scene()
    if recorder.recovery_requested or not scene["held"]:
        raise DemonstrationFailure("Initial physical pickup did not produce a verified grasp.")
    snapshot = mujoco.MjData(environment.model)
    mujoco.mj_copyData(snapshot, environment.model, environment.data)
    return snapshot, recorder.controller.prev_accepted_target.copy(), recorder.initial_heights.copy()


def restore_grasp(project, environment, grasp):
    snapshot, accepted_target, ground_heights = grasp
    mujoco.mj_copyData(environment.data, environment.model, snapshot)
    recorder = EpisodeRecorder(project, environment, seed=0)
    recorder.controller.prev_accepted_target = accepted_target.copy()
    recorder.initial_heights = ground_heights.copy()
    recorder.grasp_seen = recorder._pickup_grasp_seen = True
    return recorder


def route_waypoints(start, target, route):
    if route == "direct":
        return [target.copy()]
    intermediate = start.copy()
    if route == "xy_then_z":
        intermediate[:2] = target[:2]
    elif route == "z_then_xy":
        intermediate[2] = target[2]
    else:
        raise ValueError(f"Unknown route: {route}")
    return [intermediate, target.copy()]


def try_target(project, environment, grasp, target, route, hold_seconds):
    recorder = restore_grasp(project, environment, grasp)
    start_time = float(environment.data.time)
    result = {"route": route, "status": "motion_timeout", "stage": "move"}
    tolerance = recorder.config.teacher.position_tolerance
    stable_steps = 0
    required_steps = ceil(hold_seconds / recorder.interval - 1e-9)
    try:
        for index, waypoint in enumerate(route_waypoints(recorder.scene()["orange"], target, route)):
            result["stage"] = f"waypoint_{index + 1}"
            move_held_cube_to(recorder, waypoint, "transport")
            if recorder.recovery_requested or not recorder.scene()["held"]:
                result.update(status="grasp_lost", detail="Lost the measured grasp while carrying.")
                result.update(snapshot_result(recorder, target))
                return result
            if np.linalg.norm(recorder.scene()["orange"] - waypoint) > tolerance:
                result.update(status="motion_timeout", detail="Waypoint was not reached within tolerance.")
                result.update(snapshot_result(recorder, target))
                return result

        result["stage"] = "hold_at_target"
        # Continue correcting the measured cube center during the verification
        # hold, because the cube-to-tool offset can change as the wrist tilts.
        hold_errors = []
        for _ in range(recorder.config.teacher.maximum_move_steps):
            scene = recorder.scene()
            tool_target = target - (scene["orange"] - scene["gripper"])
            recorder.step(action_toward(recorder, tool_target, CLOSED_GRIPPER), "transport")
            measured = snapshot_result(recorder, target)
            if recorder.recovery_requested or not measured["held"]:
                result.update(status="grasp_lost", detail="Lost the measured grasp while holding the target.")
                break
            if not measured["unsupported"]:
                result.update(status="supported_by_environment", detail="Orange touched blue, the table, or a wall.")
                break
            if measured["position_error_mm"] <= tolerance * 1000:
                stable_steps += 1
                hold_errors.append(measured["position_error_mm"])
            else:
                stable_steps = 0
                hold_errors.clear()
            if stable_steps >= required_steps:
                result.update(status="reached_and_held", hold_max_error_mm=max(hold_errors))
                break
        else:
            result.update(status="hold_timeout", detail="Could not remain within tolerance for the required duration.")
    except IKConvergenceError as error:
        result.update(status="ik_failed", detail=str(error), ik_diagnostics=error.diagnostics)
    except DemonstrationFailure as error:
        result.update(status="motion_timeout", detail=str(error))
    finally:
        result["simulated_seconds"] = float(environment.data.time - start_time)
        result["consecutive_verified_hold_seconds"] = stable_steps * recorder.interval
    return {**result, **snapshot_result(recorder, target)}


def summarize_heights(results, heights_cm):
    summary = []
    for height in heights_cm:
        rows = [row for row in results if row["z_cm"] == height]
        passed = [row["corner"] for row in rows if row["status"] == "reached_and_held"]
        failed = [{"corner": row["corner"], "status": row["status"]}
                  for row in rows if row["status"] != "reached_and_held"]
        summary.append({"z_cm": height, "corners_passed": len(passed),
                        "all_corners_passed": len(rows) == 4 and len(passed) == 4,
                        "failed_corners": failed})
    return summary


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--heights-cm", type=float, nargs="+", default=[3., 4., 5., 6., 7., 8.])
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports/diagnostics/held_cube_workspace.json")
    args = parser.parse_args()
    if not all(np.isfinite(z) and z > 0 for z in args.heights_cm):
        parser.error("--heights-cm must contain finite positive heights")
    if len(set(args.heights_cm)) != len(args.heights_cm):
        parser.error("--heights-cm must not contain duplicates")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("--hold-seconds must be finite and positive")
    project = load_config(args.config.expanduser())
    # Diagnostic overrides exist only in memory; preserve the saved config.
    project.environment.action_data.drop_disturbance.probability = 0.0
    project.environment.action_data.start_position_half_range = (0.0, 0.0, 0.0)
    project.environment.episode_seconds = max(project.environment.episode_seconds, 120.0)
    config = project.environment.vision_data
    center_xy = np.array([np.mean(config.cube_x_range), np.mean(config.cube_y_range)])
    corners = spawn_corners(project)
    environment = CubeStackEnvironment(project.environment.scene_path)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "config_path": str(args.config.resolve()),
        "scene_path": str(project.environment.scene_path),
        "bounds_source": "environment.vision_data", "cube_x_range_m": list(config.cube_x_range),
        "cube_y_range_m": list(config.cube_y_range), "heights_cm": args.heights_cm,
        "target_definition": "Actual orange cube center in world coordinates, not the gripper frame.",
        "position_tolerance_mm": project.environment.action_data.teacher.position_tolerance * 1000,
        "required_hold_seconds": args.hold_seconds, "routes": list(ROUTES),
        "controller_config": project.cartesian_controller.model_dump(mode="json"),
        "teacher_config": project.environment.action_data.teacher.model_dump(mode="json"),
        "physics_timestep": float(environment.model.opt.timestep), "control_hz": project.loop.control_hz,
        "pickup_xy_m": center_xy.tolist(), "intentional_drops_enabled": False,
        "method": "Physical center pickup; blue at opposite corner; restore verified grasp before each route.",
        "limitations": "Tests these paths, grasp, and solver settings. Failures do not prove geometric impossibility; corner passes do not certify every interior point or blue-cube placement.",
        "pickups": [], "results": [],
    }
    tqdm.write(f"Spawn bounds (cm): X={np.array(config.cube_x_range) * 100}, Y={np.array(config.cube_y_range) * 100}")
    with tqdm(total=len(corners) * len(args.heights_cm), desc="Held-cube corner checks", unit="target") as progress:
        for corner, xy in corners:
            blue_xy = 2 * center_xy - xy
            try:
                grasp = prepare_grasp(project, environment, center_xy, blue_xy)
                report["pickups"].append({"corner": corner, "status": "held", "blue_xy_m": blue_xy.tolist(),
                                           "orange_xyz_m": environment.data.body("orange_cube").xpos.tolist()})
                pickup_error = None
            except (IKConvergenceError, DemonstrationFailure) as error:
                grasp = None
                pickup_error = {"detail": str(error), "error_type": type(error).__name__}
                if isinstance(error, IKConvergenceError):
                    pickup_error["ik_diagnostics"] = error.diagnostics
                report["pickups"].append({"corner": corner, "status": "pickup_failed", **pickup_error})
            for z_cm in args.heights_cm:
                target = np.array([*xy, z_cm / 100])
                row = {"corner": corner, "z_cm": z_cm, "target_xyz_m": target.tolist(),
                       "blue_xy_m": blue_xy.tolist(), "attempts": []}
                if grasp is None:
                    row.update(status="not_tested_pickup_failed", pickup_error=pickup_error)
                else:
                    for route in ROUTES:
                        attempt = try_target(project, environment, grasp, target, route, args.hold_seconds)
                        row["attempts"].append(attempt)
                        if attempt["status"] == "reached_and_held":
                            break
                    row["status"] = ("reached_and_held" if row["attempts"][-1]["status"] == "reached_and_held"
                                     else "failed_tested_routes")
                report["results"].append(row)
                report["height_summary"] = summarize_heights(report["results"], args.heights_cm)
                report["wall_seconds"] = perf_counter() - started
                output.write_text(json.dumps(report, indent=2) + "\n")
                details = ", ".join(f"{a['route']}:{a['status']}" for a in row["attempts"])
                tqdm.write(f"Z={z_cm:g} cm, XY=({xy[0] * 100:g}, {xy[1] * 100:g}) cm: {row['status']} {details}")
                progress.update()
    tqdm.write("\nHeight summary:")
    for row in report["height_summary"]:
        tqdm.write(f"  Z={row['z_cm']:g} cm: {row['corners_passed']}/4 corners reached and held")
    tqdm.write(f"Saved {output}")


if __name__ == "__main__":
    main()
