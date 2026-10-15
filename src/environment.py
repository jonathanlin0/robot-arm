"""The shared MuJoCo simulation resource used by sensors and commands."""

from pathlib import Path
from threading import RLock

import mujoco

from robot_constants import DEFAULT_GRIPPER_POSITION, ROBOT_JOINT_NAMES


DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "so101_two_cube_stack.xml"


class CubeStackEnvironment:
    """Own one model/data pair; only the command backend advances its physics.

    Camera producers, telemetry readers, reset, and command execution share
    ``state_lock`` so snapshots cannot observe partially updated physics.
    """

    def __init__(self, scene_path: Path | str = DEFAULT_SCENE_PATH) -> None:
        self.scene_path = Path(scene_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.state_lock = RLock()
        self.reset()

    def reset(self) -> None:
        """Restore XML state with an open gripper, without advancing time.

        The application separately resets camera buffers and policy histories.
        All actuator targets start at their joint positions to avoid an initial
        command discontinuity for scenes with nonzero reference positions.
        """
        with self.state_lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.joint("gripper").qpos[0] = DEFAULT_GRIPPER_POSITION
            for name in ROBOT_JOINT_NAMES:
                self.data.actuator(name).ctrl[0] = self.data.joint(name).qpos[0]
            mujoco.mj_forward(self.model, self.data)
