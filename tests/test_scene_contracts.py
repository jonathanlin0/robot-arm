"""Check the simulator geometry and reference conventions before runtime work."""

import mujoco
import numpy as np
import pytest

from environment import DEFAULT_SCENE_PATH
from robot_constants import ROBOT_JOINT_NAMES, TOOL_FRAME_SITE_NAME


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))


def test_robot_order_matches_the_scene_actuators(scene_model):
    assert tuple(scene_model.actuator(i).name for i in range(scene_model.nu)) == ROBOT_JOINT_NAMES
    for name in ROBOT_JOINT_NAMES:
        actuator = scene_model.actuator(name)
        assert actuator.trnid[0] == scene_model.joint(name).id


def test_external_camera_is_inset_and_preserves_viewing_direction(scene_model):
    assert {scene_model.camera(i).name for i in range(scene_model.ncam)} == {"wrist_cam", "external"}
    camera = scene_model.camera("external")
    data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, data)
    position = data.cam_xpos[camera.id]
    np.testing.assert_allclose(position, [0.43, 0.23, 0.20])
    # MuJoCo cameras view along their local negative Z axis.
    view_direction = -data.cam_xmat[camera.id].reshape(3, 3)[:, 2]
    target_direction = np.array([-0.25, -0.25, -0.20])
    target_direction /= np.linalg.norm(target_direction)
    np.testing.assert_allclose(view_direction, target_direction, atol=1e-12)
    assert scene_model.cam_projection[camera.id] == 0  # Perspective.


def test_overhead_light_aims_at_enclosed_tabletop_center_and_casts_shadows(scene_model):
    light = scene_model.light("overhead")
    # Position jitter must affect illumination, which a directional light ignores.
    assert scene_model.light_type[light.id] == mujoco.mjtLightType.mjLIGHT_SPOT
    np.testing.assert_allclose(light.pos, [0.0, 0.0, 1.524])
    assert scene_model.light_castshadow[light.id]
    data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, data)
    center = np.empty(3)
    for axis, name in enumerate(("x", "y")):
        lower = scene_model.geom(f"wall_{name}_min")
        upper = scene_model.geom(f"wall_{name}_max")
        center[axis] = (lower.pos[axis] + lower.size[axis]
                        + upper.pos[axis] - upper.size[axis]) / 2
    table = scene_model.geom("table")
    center[2] = table.pos[2] + table.size[2]
    position = data.light_xpos[light.id]
    direction = data.light_xdir[light.id]
    assert direction[2] < 0
    intersection = position + direction * ((center[2] - position[2]) / direction[2])
    np.testing.assert_allclose(intersection, center, atol=1e-12)


@pytest.mark.parametrize("name,axis,sign,inner_edge", [
    ("wall_x_min", 0, 1, -0.05),
    ("wall_x_max", 0, -1, 0.45),
    ("wall_y_min", 1, 1, -0.25),
    ("wall_y_max", 1, -1, 0.25),
])
def test_wall_inside_bounds_and_dimensions(scene_model, name, axis, sign, inner_edge):
    wall = scene_model.geom(name)
    assert wall.pos[axis] + sign * wall.size[axis] == pytest.approx(inner_edge)
    assert 2 * wall.size[axis] == pytest.approx(0.002)
    assert 2 * wall.size[2] == pytest.approx(0.50)
    assert wall.pos[2] - wall.size[2] == pytest.approx(0.0)


def test_gripper_reference_is_rigid_when_only_the_jaw_opens(scene_model):
    data = mujoco.MjData(scene_model)
    positions = []
    for opening in (-0.1, 1.0):
        data.joint("gripper").qpos[0] = opening
        mujoco.mj_forward(scene_model, data)
        positions.append(data.site(TOOL_FRAME_SITE_NAME).xpos.copy())
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-12)
    assert data.qpos.dtype == np.float64


def test_cubes_keep_the_original_dimensions_and_mass(scene_model):
    for color in ("orange", "blue"):
        np.testing.assert_allclose(scene_model.geom(f"{color}_cube_geom").size, [0.02] * 3)
        assert scene_model.body(f"{color}_cube").mass[0] == pytest.approx(0.04)
