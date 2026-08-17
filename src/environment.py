from pathlib import Path

import mujoco
import numpy as np

from randomization import (
    BLUE_CUBE_JOINT,
    ORANGE_CUBE_JOINT,
    CubeSpawnConfig,
    randomize_cube_placements,
)


DEFAULT_SCENE_PATH = Path("scenes/so101_two_cube_stack.xml")

ROBOT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

StateSnapshot = dict[str, float | np.ndarray]


class CubeStackEnvironment:
    """Owns the MuJoCo model, simulation state, and episode resets."""

    def __init__(
        self,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        *, # this forces every parameter after this to be named
        seed: int | None = None,
        spawn_config: CubeSpawnConfig | None = None,
    ) -> None:
        self.scene_path = Path(scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.spawn_config = spawn_config or CubeSpawnConfig()
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> StateSnapshot:
        """Reset all state, randomize cube placements, and return a snapshot."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        randomize_cube_placements(
            self.data,
            self.rng,
            self.spawn_config,
        )
        # recalc info like body positions, orientations, etc.
        # doesn't advance time forward tho
        mujoco.mj_forward(self.model, self.data)

        return self.get_state()

    def step_physics(self, steps: int = 1) -> None:
        """Advance raw physics; joint-target actions will be added next."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def get_state(self) -> StateSnapshot:
        """Return copied arrays so callers cannot mutate MuJoCo state."""
        orange_joint = self.data.joint(ORANGE_CUBE_JOINT)
        blue_joint = self.data.joint(BLUE_CUBE_JOINT)

        return {
            "time": float(self.data.time),
            "joint_positions": np.array(
                [self.data.joint(name).qpos[0] for name in ROBOT_JOINT_NAMES]
            ),
            "joint_velocities": np.array(
                [self.data.joint(name).qvel[0] for name in ROBOT_JOINT_NAMES]
            ),
            "controls": self.data.ctrl.copy(), # controls r the intended angles/locations for the actuators
            "gripper_position": self.data.site("gripperframe").xpos.copy(),
            "orange_position": self.data.body("orange_cube").xpos.copy(),
            "orange_orientation": self.data.body("orange_cube").xquat.copy(),
            "orange_velocity": orange_joint.qvel.copy(),
            "blue_position": self.data.body("blue_cube").xpos.copy(),
            "blue_orientation": self.data.body("blue_cube").xquat.copy(),
            "blue_velocity": blue_joint.qvel.copy(),
        }
