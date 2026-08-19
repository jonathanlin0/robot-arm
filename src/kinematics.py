from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from environment import ARM_JOINT_NAMES


GRIPPER_SITE_NAME = "gripperframe"
DEFAULT_POSITION_TOLERANCE = 1e-3  # 1 mm; MuJoCo positions are in meters.
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_DAMPING = 0.02
DEFAULT_MAX_JOINT_STEP = 0.15


@dataclass(frozen=True)
class IKResult:
    """Outcome of a position-only inverse-kinematics solve."""

    joint_positions: np.ndarray
    converged: bool
    position_error: float
    iterations: int


def _finite_vector(
    values: ArrayLike,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")

    return vector.copy()


def solve_position_ik(
    model: mujoco.MjModel,
    initial_joint_positions: ArrayLike,
    target_position: ArrayLike,
    tolerance: float = DEFAULT_POSITION_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    damping: float = DEFAULT_DAMPING,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> IKResult:
    """Find five arm-joint positions that place the gripper at target XYZ."""
    arm_joint_count = len(ARM_JOINT_NAMES)
    candidate_positions = _finite_vector(
        initial_joint_positions,
        "initial_joint_positions",
    )
    target = _finite_vector(target_position, "target_position")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and greater than zero.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and greater than zero.")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError(
            "max_joint_step must be finite and greater than zero."
        )

    # keep calculations of lower/upper bounds separate from specific environment
    joint_lower_bounds = np.empty(arm_joint_count)
    joint_upper_bounds = np.empty(arm_joint_count)
    joint_dof_indices = np.empty(arm_joint_count, dtype=int)

    for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint = model.joint(joint_name)
        actuator = model.actuator(joint_name)

        lower_bound = max(joint.range[0], actuator.ctrlrange[0])
        upper_bound = min(joint.range[1], actuator.ctrlrange[1])
        if lower_bound > upper_bound:
            raise ValueError(
                f"Joint {joint_name!r} has incompatible joint and actuator "
                "ranges."
            )

        joint_lower_bounds[joint_index] = lower_bound
        joint_upper_bounds[joint_index] = upper_bound
        joint_dof_indices[joint_index] = model.jnt_dofadr[joint.id]

    candidate_positions = np.clip(
        candidate_positions,
        joint_lower_bounds,
        joint_upper_bounds,
    )

    scratch_data = mujoco.MjData(model)
    gripper_site_id = model.site(GRIPPER_SITE_NAME).id
    position_jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))
    damping_matrix = damping**2 * np.eye(3)

    # loop is in a weird structure to avoid duplicated logic
    # it checks if current joint positions are fine. if so, then return result
    # otherwise, then use jacobian to find optimal change in motors to move gripper to target location
    # then, the next loop iteration checks the current loop's proposal. in other words, the current loop checks the previous loop's proposal
    position_error = float("inf")
    for iteration in range(max_iterations + 1):
        # check if current joint positions achieve target gripper loc
        for joint_name, joint_position in zip(
            ARM_JOINT_NAMES,
            candidate_positions,
            strict=True,
        ):
            scratch_data.joint(joint_name).qpos[0] = joint_position

        mujoco.mj_forward(model, scratch_data)

        position_error_vector = (
            target - scratch_data.site(GRIPPER_SITE_NAME).xpos
        )
        position_error = float(np.linalg.norm(position_error_vector))

        if position_error <= tolerance:
            return IKResult(
                joint_positions=candidate_positions.copy(),
                converged=True,
                position_error=position_error,
                iterations=iteration,
            )

        # beginning/end of the logic. the logic kind of "loops around" the loop and overlaps between iterations
        if iteration == max_iterations:
            break

        mujoco.mj_jacSite( # calculates jacobian. but calcs for all joints, not just the ones for the robot arm
            model,
            scratch_data,
            position_jacobian,
            rotation_jacobian,
            gripper_site_id,
        )
        arm_position_jacobian = position_jacobian[:, joint_dof_indices]

        joint_correction = arm_position_jacobian.T @ np.linalg.solve(
            arm_position_jacobian @ arm_position_jacobian.T
            + damping_matrix,
            position_error_vector,
        )

        # limits overall magnitude of the joint-correction vector
        correction_norm = np.linalg.norm(joint_correction)
        if correction_norm > max_joint_step:
            joint_correction *= max_joint_step / correction_norm

        candidate_positions = np.clip(
            candidate_positions + joint_correction,
            joint_lower_bounds,
            joint_upper_bounds,
        )

    return IKResult(
        joint_positions=candidate_positions.copy(),
        converged=False,
        position_error=position_error,
        iterations=max_iterations,
    )
