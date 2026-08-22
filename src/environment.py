import math
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from randomization import (
    BLUE_CUBE_JOINT,
    ORANGE_CUBE_JOINT,
    CubeSpawnConfig,
    randomize_cube_placements,
)
from success import (
    StackSuccessConfig,
    orange_gripper_pad_contacts,
    orange_touches_table,
    stack_conditions_met as evaluate_stack_conditions,
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
ARM_JOINT_NAMES = ROBOT_JOINT_NAMES[:-1]
DEFAULT_JOINT_POSITIONS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
)
PHYSICS_STEPS_PER_ACTION = 10
OFF_TABLE_HEIGHT_TOLERANCE = 0.005

StateSnapshot = dict[str, bool | float | np.ndarray]


class CubeStackEnvironment:
    """Owns the MuJoCo model, simulation state, and episode resets."""

    def __init__(
        self,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        *, # this forces every parameter after this to be named
        seed: int | None = None,
        spawn_config: CubeSpawnConfig | None = None,
        success_config: StackSuccessConfig | None = None,
    ) -> None:
        self.scene_path = Path(scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.spawn_config = spawn_config or CubeSpawnConfig()
        self.success_config = success_config or StackSuccessConfig()
        self.rng = np.random.default_rng(seed)
        self._stack_stable_time = 0.0
        self._stack_success = False
        # grasp has occured previously but isn't occuring right now and grasp was occuring when the orange cube wasn't touching the table
        self._confirmed_grasp_seen = False
        self._initial_orange_height = self.spawn_config.cube_center_z
        self._initial_blue_height = self.spawn_config.cube_center_z
        self._orange_fell_off_table = False
        self._blue_fell_off_table = False
        self._action_idx_to_actuator_ctrl_idx = np.empty(len(ROBOT_JOINT_NAMES), dtype=int)
        self._joint_target_lower_bounds = np.empty(len(ROBOT_JOINT_NAMES))
        self._joint_target_upper_bounds = np.empty(len(ROBOT_JOINT_NAMES))

        for action_index, actuator_name in enumerate(ROBOT_JOINT_NAMES):
            actuator_id = self.model.actuator(actuator_name).id

            if self.model.actuator_ctrlnum[actuator_id] != 1:
                raise ValueError(f"Actuator {actuator_name!r} must have one control input.")
            # check not necessary for so101 model. for potential future robots that have stuff like tendors or bodies
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(
                    f"Actuator {actuator_name!r} must control a joint."
                )

            # verify that the actuator drives the joint with the same name
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            expected_joint_id = self.model.joint(actuator_name).id
            if joint_id != expected_joint_id:
                raise ValueError(
                    f"Actuator {actuator_name!r} must control the joint with "
                    "the same name."
                )

            # verify intersection of actuator and joint safe range is valid
            actuator_range = self.model.actuator_ctrlrange[actuator_id]
            joint_range = self.model.jnt_range[joint_id]

            lower_bound = max(actuator_range[0], joint_range[0])
            upper_bound = min(actuator_range[1], joint_range[1])
            if lower_bound > upper_bound:
                raise ValueError(
                    f"Actuator {actuator_name!r} has incompatible control "
                    "and joint ranges."
                )

            self._action_idx_to_actuator_ctrl_idx[action_index] = (
                self.model.actuator_ctrladr[actuator_id]
            )
            self._joint_target_lower_bounds[action_index] = lower_bound
            self._joint_target_upper_bounds[action_index] = upper_bound

    def reset(self, *, seed: int | None = None) -> StateSnapshot:
        """Reset all state, randomize cube placements, and return a snapshot."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._stack_stable_time = 0.0
        self._stack_success = False
        self._confirmed_grasp_seen = False
        self._orange_fell_off_table = False
        self._blue_fell_off_table = False

        mujoco.mj_resetData(self.model, self.data)

        for joint_name, joint_position in zip(
            ROBOT_JOINT_NAMES,
            DEFAULT_JOINT_POSITIONS,
            strict=True,
        ):
            self.data.joint(joint_name).qpos[0] = joint_position

        # match the actuator targets to the reset pose so the robot does not
        # immediately try to move away from it on the first physics step.
        self.data.ctrl[self._action_idx_to_actuator_ctrl_idx] = (
            DEFAULT_JOINT_POSITIONS
        )

        randomize_cube_placements(
            self.data,
            self.rng,
            self.spawn_config,
        )
        # recalc info like body positions, orientations, etc.
        # doesn't advance time forward tho
        mujoco.mj_forward(self.model, self.data)

        # manually resetting it here in case we later randomize the env, like table height, cube size, etc
        self._initial_orange_height = float(
            self.data.body("orange_cube").xpos[2]
        )
        self._initial_blue_height = float(
            self.data.body("blue_cube").xpos[2]
        )

        return self.get_state()

    def step_physics(self, steps: int = 1) -> None:
        """Advance raw physics by a requested number of timesteps."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self._update_off_table_failure()
            self._update_stack_success()

        self._update_confirmed_grasp()

    def stack_conditions_met(self) -> bool:
        """Return whether the current state looks like a valid stack."""
        return evaluate_stack_conditions(
            self.model,
            self.data,
            self.success_config,
        )

    def is_success(self) -> bool:
        """Return whether grasp-and-stack success occurred since reset."""
        return self._stack_success

    def is_failure(self) -> bool:
        """Return whether an unrecoverable failure occurred since reset.
                
        For now, it's only considered unrecoverable if either cube fell off the table."""
        return (
            self._orange_fell_off_table
            or self._blue_fell_off_table
        )

    def is_terminated(self) -> bool:
        """Return whether this episode ended in either success or failure."""
        return self.is_success() or self.is_failure()

    @property
    def confirmed_grasp_seen(self) -> bool:
        """Return whether orange has been held by both jaws off the table."""
        return self._confirmed_grasp_seen

    @property
    def orange_fell_off_table(self) -> bool:
        """Return whether orange fell irrecoverably below the tabletop."""
        return self._orange_fell_off_table

    @property
    def blue_fell_off_table(self) -> bool:
        """Return whether blue fell irrecoverably below the tabletop."""
        return self._blue_fell_off_table

    @property
    def stack_stable_time(self) -> float:
        """Return accumulated valid-stack time in simulated seconds."""
        return self._stack_stable_time

    def _update_stack_success(self) -> None:
        # is failure should always be checked before checking of the stack was successful, but i guess this works for consistency/redundancy
        if self.is_failure():
            self._stack_stable_time = 0.0
            return

        # A physical stack does not satisfy this task unless the orange cube
        # was first genuinely grasped and lifted from the table.
        if not self._confirmed_grasp_seen:
            self._stack_stable_time = 0.0
            return

        # Success terminates an episode, so keep it latched until reset.
        if self._stack_success:
            return

        if not self.stack_conditions_met():
            self._stack_stable_time = 0.0
            return

        # _stack_success isn't true but the stack conditions r met
        # this happens when the env is waiting it out for stable stack verification
        self._stack_stable_time = min(
            self._stack_stable_time + self.model.opt.timestep,
            self.success_config.required_stable_time,
        )
        if math.isclose(
            self._stack_stable_time,
            self.success_config.required_stable_time,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            self._stack_success = True

    def _update_confirmed_grasp(self) -> None:
        if self._confirmed_grasp_seen or self.is_failure():
            return

        (
            orange_touches_fixed_jaw,
            orange_touches_moving_jaw,
        ) = orange_gripper_pad_contacts(self.model, self.data)
        self._confirmed_grasp_seen = (
            orange_touches_fixed_jaw
            and orange_touches_moving_jaw
            and not orange_touches_table(self.model, self.data)
        )

    def _update_off_table_failure(self) -> None:
        if not self._orange_fell_off_table:
            # raw data stored as floating point number, but not necessarily built-in python float
            orange_center_height = float(
                self.data.body("orange_cube").xpos[2]
            )
            self._orange_fell_off_table = orange_center_height < (
                self._initial_orange_height
                - OFF_TABLE_HEIGHT_TOLERANCE
            )

        if not self._blue_fell_off_table:
            blue_center_height = float(
                self.data.body("blue_cube").xpos[2]
            )
            self._blue_fell_off_table = blue_center_height < (
                self._initial_blue_height
                - OFF_TABLE_HEIGHT_TOLERANCE
            )

    def step_joint_targets(
        self,
        joint_targets: ArrayLike,
    ) -> StateSnapshot:
        """Apply safe position targets and advance one control interval."""
        targets = np.asarray(joint_targets, dtype=float)
        expected_shape = (len(ROBOT_JOINT_NAMES),)

        if targets.shape != expected_shape:
            raise ValueError(
                f"joint_targets must have shape {expected_shape}; "
                f"received {targets.shape}."
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("joint_targets must contain only finite values.")

        safe_targets = np.clip(
            targets,
            self._joint_target_lower_bounds,
            self._joint_target_upper_bounds,
        )
        self.data.ctrl[self._action_idx_to_actuator_ctrl_idx] = safe_targets
        self.step_physics(PHYSICS_STEPS_PER_ACTION)

        return self.get_state()

    def get_state(self) -> StateSnapshot:
        """Return copied arrays so callers cannot mutate MuJoCo state."""
        orange_joint = self.data.joint(ORANGE_CUBE_JOINT)
        blue_joint = self.data.joint(BLUE_CUBE_JOINT)
        (
            orange_touches_fixed_jaw,
            orange_touches_moving_jaw,
        ) = orange_gripper_pad_contacts(self.model, self.data)

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
            "orange_touches_fixed_jaw": orange_touches_fixed_jaw,
            "orange_touches_moving_jaw": orange_touches_moving_jaw,
            "orange_touches_table": orange_touches_table(
                self.model,
                self.data,
            ),
            "confirmed_grasp_seen": self._confirmed_grasp_seen,
            "orange_fell_off_table": self._orange_fell_off_table,
            "blue_fell_off_table": self._blue_fell_off_table,
        }
