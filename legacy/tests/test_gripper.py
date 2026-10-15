from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment import ROBOT_JOINT_NAMES, CubeStackEnvironment
from gripper import (
    FIXED_JAW_TIP_GEOM_NAMES,
    MOVING_JAW_TIP_GEOM_NAMES,
    jaw_tip_midpoint,
    jaw_tip_midpoint_position_jacobian,
)


SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")
ROBOT_MODEL_PATH = Path("models/so101/so101.xml")


@pytest.fixture
def environment() -> CubeStackEnvironment:
    if not ROBOT_MODEL_PATH.exists():
        pytest.fail(
            "SO-101 model is missing. Run "
            "./scripts/download_so101_mujoco_model.sh first."
        )

    return CubeStackEnvironment(scene_path=SCENE_PATH)


def test_jaw_tip_midpoint_averages_both_jaw_centers(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=1)
    fixed_jaw_center = np.mean(
        [
            environment.data.geom(geom_name).xpos
            for geom_name in FIXED_JAW_TIP_GEOM_NAMES
        ],
        axis=0,
    )
    moving_jaw_center = np.mean(
        [
            environment.data.geom(geom_name).xpos
            for geom_name in MOVING_JAW_TIP_GEOM_NAMES
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        jaw_tip_midpoint(environment.data),
        (fixed_jaw_center + moving_jaw_center) / 2.0,
    )


def test_jaw_tip_midpoint_jacobian_matches_finite_differences(
    environment: CubeStackEnvironment,
) -> None:
    environment.reset(seed=2)
    joint_positions = np.array([0.2, -0.3, 0.25, -0.1, 0.4, 0.3])
    for joint_name, joint_position in zip(
        ROBOT_JOINT_NAMES,
        joint_positions,
        strict=True,
    ):
        environment.data.joint(joint_name).qpos[0] = joint_position
    mujoco.mj_forward(environment.model, environment.data)

    analytic_jacobian = jaw_tip_midpoint_position_jacobian(
        environment.model,
        environment.data,
    )
    step_size = 1e-7

    for joint_name in ROBOT_JOINT_NAMES:
        joint = environment.data.joint(joint_name)
        original_position = float(joint.qpos[0])

        joint.qpos[0] = original_position + step_size
        mujoco.mj_forward(environment.model, environment.data)
        forward_position = jaw_tip_midpoint(environment.data)

        joint.qpos[0] = original_position - step_size
        mujoco.mj_forward(environment.model, environment.data)
        backward_position = jaw_tip_midpoint(environment.data)

        joint.qpos[0] = original_position
        mujoco.mj_forward(environment.model, environment.data)

        finite_difference = (
            forward_position - backward_position
        ) / (2.0 * step_size)
        joint_id = environment.model.joint(joint_name).id
        dof_index = environment.model.jnt_dofadr[joint_id]
        np.testing.assert_allclose(
            analytic_jacobian[:, dof_index],
            finite_difference,
            atol=1e-7,
        )
