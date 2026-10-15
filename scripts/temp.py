"""Inspect the walled scene through its external camera, without advancing physics.

Interactive view on macOS:
    .venv/bin/mjpython scripts/temp.py

Save a square external-camera image without opening the interactive viewer:
    .venv/bin/python scripts/temp.py --save .tmp/walled_scene_external.png

The arm uses the model's default joint pose, with its gripper open. Its base
remains at the world origin; no legacy environment or home-position IK is used.
"""

import argparse
from pathlib import Path
import time

import mujoco


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "so101_two_cube_stack.xml"
CAMERA_NAME = "external"
IMAGE_SIZE = 1024
OPEN_GRIPPER_POSITION = 1.0


def load_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    data.joint("gripper").qpos[0] = OPEN_GRIPPER_POSITION
    data.actuator("gripper").ctrl[0] = OPEN_GRIPPER_POSITION
    mujoco.mj_forward(model, data)
    return model, data


def save_preview(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    from PIL import Image

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Show visual meshes and scene geometry, hiding duplicate collision meshes.
    options = mujoco.MjvOption()
    options.geomgroup[3:5] = 0
    with mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE) as renderer:
        renderer.update_scene(data, camera=CAMERA_NAME, scene_option=options)
        Image.fromarray(renderer.render()).save(path)
    print(f"Saved external camera view: {path.resolve()}")


def show_preview(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera(CAMERA_NAME).id
            viewer.opt.geomgroup[3:5] = 0
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", type=Path, metavar="PNG",
                        help="Save an external-camera image and exit instead of opening the viewer.")
    arguments = parser.parse_args()
    model, data = load_scene()
    print("Inside bounds (m): X [-0.05, 0.45], Y [-0.25, 0.25]")
    print("Walls: 2 mm thick, 50 cm tall. External camera: (0.43, 0.23, 0.20) m.")
    print("Camera inset: 2 cm from positive X/Y walls; viewing angle unchanged.")
    print("View target: (0.18, -0.02, 0.00) m; perspective vertical FOV: 90 degrees.")
    print("Static preview; physics is paused.")
    if arguments.save is not None:
        save_preview(model, data, arguments.save)
    else:
        show_preview(model, data)


if __name__ == "__main__":
    main()
