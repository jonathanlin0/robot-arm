"""Render the arm's default reset pose without advancing simulation physics.

Run from the repository root (no PYTHONPATH setup needed):
    .venv/bin/python scripts/diagnostics/render_default_arm_pose.py
    .venv/bin/python scripts/diagnostics/render_default_arm_pose.py --show-walls

Uses the XML scene-default arm joints with an open gripper. Action validation
instead samples a Cartesian home and initializes it through IK; this diagnostic
continues to show the scene's default reset pose. The enclosure walls are hidden
in the overview by default so the arm is visible; this changes rendering only,
not geometry or joint positions.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment import CubeStackEnvironment
from robot_constants import ROBOT_JOINT_NAMES, TOOL_FRAME_SITE_NAME
from schemas.config import load_config


def render_default_pose(environment: CubeStackEnvironment, *, show_walls: bool) -> np.ndarray:
    """Render a labeled RGB overview of the unadvanced reset state."""
    environment.reset()
    model, data = environment.model, environment.data
    width, height = 1024, 768
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.20, 0.0, 0.15]
    camera.distance = 0.95
    camera.azimuth = 135.0
    camera.elevation = -28.0

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    original_groups = model.geom_group.copy()
    try:
        if not show_walls:
            # This mask affects only rendering. No scene files or physics
            # properties are modified, and no policy/vision model is loaded.
            for name in ("wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"):
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id >= 0:
                    model.geom_group[geom_id] = 5
            options.geomgroup[5] = 0
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = renderer.render().copy()
    finally:
        model.geom_group[:] = original_groups

    image = cv2.copyMakeBorder(frame, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=(248, 250, 252))
    text_color = (30, 41, 59)
    cv2.putText(image, "Arm at default reset pose", (28, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, text_color, 2, cv2.LINE_AA)
    visibility = "Walls shown" if show_walls else "Enclosure walls hidden for visibility"
    cv2.putText(image, f"{visibility}  |  Physics time: {data.time:.3f} s", (28, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    positions = [float(data.joint(name).qpos[0]) for name in ROBOT_JOINT_NAMES]
    angles = ", ".join(f"{value:g}" for value in positions)
    xyz = data.site(TOOL_FRAME_SITE_NAME).xpos * 100.0
    cv2.putText(image, f"Joint angles (rad): [{angles}]  |  Gripper open", (28, height + 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.57, text_color, 1, cv2.LINE_AA)
    cv2.putText(image, f"Gripper XYZ (cm): ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})", (28, height + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.57, text_color, 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports/diagnostics/default_arm_pose.png")
    parser.add_argument("--show-walls", action="store_true", help="Include the enclosure walls in the overview.")
    arguments = parser.parse_args()
    project = load_config(arguments.config.expanduser())
    environment = CubeStackEnvironment(project.environment.scene_path)
    image = render_default_pose(environment, show_walls=arguments.show_walls)
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not save image to {output}")
    tqdm.write(f"Saved default-pose image: {output}")
    for name in ROBOT_JOINT_NAMES:
        angle = float(environment.data.joint(name).qpos[0])
        tqdm.write(f"  {name}: {angle:.6f} rad ({np.degrees(angle):.2f} degrees)")


if __name__ == "__main__":
    main()
