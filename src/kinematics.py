from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from environment import ARM_JOINT_NAMES


TOOL_FRAME_SITE_NAME = "gripperframe"
DEFAULT_POSITION_TOLERANCE = 1e-3  # 1 mm; MuJoCo positions are in meters.
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_DAMPING = 0.02
DEFAULT_MAX_JOINT_STEP = 0.15
# when the orientation of the gripper is considered good enouch
DEFAULT_TOOL_AXIS_TOLERANCE = float(np.deg2rad(5.0))  # Stored in radians.
# this is similar to a learning rate
DEFAULT_TOOL_AXIS_GAIN = 0.2
WORLD_DOWN = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class IKResult:
    """Outcome of a position-only inverse-kinematics solve."""

    joint_positions: np.ndarray
    converged: bool
    position_error: float
    iterations: int


@dataclass(frozen=True)
class ToolAxisIKResult:
    """Outcome of IK with position and tool-axis objectives.

    Position is the primary objective. Tool-axis alignment is a best-effort
    secondary objective because the requested direction may not be reachable
    at every XYZ position.
    """

    joint_positions: np.ndarray
    position_converged: bool
    tool_axis_converged: bool
    position_error: float
    tool_axis_error: float  # Radians between current and target tool axes.
    iterations: int


def _finite_vector(
    values: ArrayLike,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")

    return vector.copy()


def _arm_joint_kinematics(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return safe joint bounds and MuJoCo DOF indices for the arm."""
    arm_joint_count = len(ARM_JOINT_NAMES)
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

    return (
        joint_lower_bounds,
        joint_upper_bounds,
        joint_dof_indices,
    )


def _tool_axis_rotation_error(
    current_axis: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    """Return the shortest world-frame rotation from one axis to another."""
    cross_product = np.cross(current_axis, target_axis)
    cross_norm = np.linalg.norm(cross_product)
    dot_product = np.clip(np.dot(current_axis, target_axis), -1.0, 1.0)

    if cross_norm > 1e-12:
        angle = np.arctan2(cross_norm, dot_product)
        return cross_product * (angle / cross_norm)

    if dot_product > 0.0:
        return np.zeros(3)

    # At exactly 180 degrees, infinitely many rotation axes are valid. Pick
    # one deterministically by crossing with the least-aligned world axis.
    basis = np.zeros(3)
    basis[np.argmin(np.abs(current_axis))] = 1.0
    rotation_axis = np.cross(current_axis, basis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    return np.pi * rotation_axis


def _scale_correction_to_joint_limits(
    joint_positions: np.ndarray,
    joint_correction: np.ndarray,
    joint_lower_bounds: np.ndarray,
    joint_upper_bounds: np.ndarray,
) -> np.ndarray:
    """Scale an entire correction so it stays inside all joint limits."""
    scale = 1.0

    for joint_index, correction in enumerate(joint_correction):
        if correction > 0.0:
            available_scale = (
                joint_upper_bounds[joint_index]
                - joint_positions[joint_index]
            ) / correction
            scale = min(scale, available_scale)
        elif correction < 0.0:
            available_scale = (
                joint_lower_bounds[joint_index]
                - joint_positions[joint_index]
            ) / correction
            scale = min(scale, available_scale)

    return joint_correction * np.clip(scale, 0.0, 1.0)


def solve_position_ik(
    model: mujoco.MjModel,
    initial_joint_positions: ArrayLike,
    target_position: ArrayLike,
    tolerance: float = DEFAULT_POSITION_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    damping: float = DEFAULT_DAMPING,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> IKResult:
    """Place the rigid gripper-frame site at target XYZ with five joints."""
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
    (
        joint_lower_bounds,
        joint_upper_bounds,
        joint_dof_indices,
    ) = _arm_joint_kinematics(model)

    candidate_positions = np.clip(
        candidate_positions,
        joint_lower_bounds,
        joint_upper_bounds,
    )

    scratch_data = mujoco.MjData(model)
    gripper_site_id = model.site(TOOL_FRAME_SITE_NAME).id
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
            target - scratch_data.site(TOOL_FRAME_SITE_NAME).xpos
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

        # just created here. actual values filled in later in the function
        position_jacobian = np.zeros((3, model.nv))
        # fill in the position jacobian
        mujoco.mj_jacSite(
            model,
            scratch_data,
            position_jacobian,
            None,
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


def solve_position_and_tool_axis_ik(
    model: mujoco.MjModel,
    initial_joint_positions: ArrayLike,
    target_position: ArrayLike,
    target_tool_axis: ArrayLike = WORLD_DOWN,
    position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
    tool_axis_tolerance: float = DEFAULT_TOOL_AXIS_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    damping: float = DEFAULT_DAMPING,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
    tool_axis_gain: float = DEFAULT_TOOL_AXIS_GAIN,
    stop_when_position_converged: bool = False,
    minimum_iterations: int = 0,
) -> ToolAxisIKResult:
    """Place the rigid gripper-frame site while biasing tool direction.

    Both position and approach direction come from the rigid
    ``gripperframe`` site. Position is solved as the primary task. The
    rotational Jacobian is projected into the position Jacobian's null space,
    so the solver improves the tool direction without intentionally moving
    away from the requested XYZ position. Set
    ``stop_when_position_converged`` for online control, where a nearby
    position-valid solution is preferable to spending more iterations
    searching for a better orientation. A positive ``minimum_iterations``
    lets an online controller apply at least one local orientation correction
    even when XYZ already starts within tolerance.

    Params
        initial_joint_positions: initial joint positions of the joints, excluding the gripper
    """
    candidate_positions = _finite_vector(
        initial_joint_positions,
        "initial_joint_positions",
    )
    target = _finite_vector(target_position, "target_position")
    normalized_target_tool_axis = _finite_vector(
        target_tool_axis,
        "target_tool_axis",
    )

    target_axis_norm = np.linalg.norm(normalized_target_tool_axis)
    if not np.isfinite(target_axis_norm) or target_axis_norm == 0.0:
        raise ValueError("target_tool_axis must have nonzero length.")
    normalized_target_tool_axis /= target_axis_norm

    if not np.isfinite(position_tolerance) or position_tolerance <= 0:
        raise ValueError(
            "position_tolerance must be finite and greater than zero."
        )
    if not np.isfinite(tool_axis_tolerance) or tool_axis_tolerance <= 0:
        raise ValueError(
            "tool_axis_tolerance must be finite and greater than zero."
        )
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if minimum_iterations < 0:
        raise ValueError("minimum_iterations cannot be negative.")
    if minimum_iterations > max_iterations:
        raise ValueError(
            "minimum_iterations cannot exceed max_iterations."
        )
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and greater than zero.")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError(
            "max_joint_step must be finite and greater than zero."
        )
    if not np.isfinite(tool_axis_gain) or tool_axis_gain <= 0:
        raise ValueError(
            "tool_axis_gain must be finite and greater than zero."
        )
    (
        joint_lower_bounds,
        joint_upper_bounds,
        joint_dof_indices,
    ) = _arm_joint_kinematics(model)
    candidate_positions = np.clip(
        candidate_positions,
        joint_lower_bounds,
        joint_upper_bounds,
    )

    scratch_data = mujoco.MjData(model)
    tool_frame_site_id = model.site(TOOL_FRAME_SITE_NAME).id
    position_jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))
    position_damping_matrix = damping**2 * np.eye(3)
    joint_identity = np.eye(len(ARM_JOINT_NAMES))

    best_position_result: ToolAxisIKResult | None = None
    best_position_and_axis_result: ToolAxisIKResult | None = None

    for iteration in range(max_iterations + 1):
        for joint_name, joint_position in zip(
            ARM_JOINT_NAMES,
            candidate_positions,
            strict=True,
        ):
            scratch_data.joint(joint_name).qpos[0] = joint_position

        mujoco.mj_forward(model, scratch_data)
        tool_frame_site = scratch_data.site(TOOL_FRAME_SITE_NAME)

        position_error_vector = target - tool_frame_site.xpos
        position_error = float(np.linalg.norm(position_error_vector))

        # MuJoCo stores a site's local axes as the columns of xmat. Local +X
        # is the approach direction for this SO-101 gripper model.
        current_tool_axis = tool_frame_site.xmat.reshape(3, 3)[:, 0]
        rotation_error_vector = _tool_axis_rotation_error(
            current_tool_axis,
            normalized_target_tool_axis,
        )
        tool_axis_error = float(np.linalg.norm(rotation_error_vector))

        result = ToolAxisIKResult(
            joint_positions=candidate_positions.copy(),
            position_converged=bool(
                position_error <= position_tolerance
            ),
            tool_axis_converged=bool(
                tool_axis_error <= tool_axis_tolerance
            ),
            position_error=position_error,
            tool_axis_error=tool_axis_error,
            iterations=iteration,
        )

        if (
            best_position_result is None
            or (position_error, tool_axis_error)
            < (
                best_position_result.position_error,
                best_position_result.tool_axis_error,
            )
        ):
            best_position_result = result

        if result.position_converged and (
            best_position_and_axis_result is None
            or (tool_axis_error, position_error)
            < (
                best_position_and_axis_result.tool_axis_error,
                best_position_and_axis_result.position_error,
            )
        ):
            best_position_and_axis_result = result

        if (
            iteration >= minimum_iterations
            and result.position_converged
            and (
                stop_when_position_converged
                or result.tool_axis_converged
            )
        ):
            return result

        if iteration == max_iterations:
            break

        mujoco.mj_jacSite(
            model,
            scratch_data,
            position_jacobian,
            rotation_jacobian,
            tool_frame_site_id,
        )
        arm_position_jacobian = position_jacobian[:, joint_dof_indices]
        arm_rotation_jacobian = rotation_jacobian[:, joint_dof_indices]

        position_correction = arm_position_jacobian.T @ np.linalg.solve(
            arm_position_jacobian @ arm_position_jacobian.T
            + position_damping_matrix,
            position_error_vector,
        )

        # The null-space projector removes joint motion that would change the
        # gripper position to first order. This makes orientation secondary.
        position_null_space = (
            joint_identity
            - np.linalg.pinv(arm_position_jacobian)
            @ arm_position_jacobian
        )

        # Rotation about the tool axis does not change the direction of that
        # axis, so remove that irrelevant component of angular velocity.
        axis_projection = (
            np.eye(3)
            - np.outer(current_tool_axis, current_tool_axis)
        )
        tool_axis_jacobian = axis_projection @ arm_rotation_jacobian
        tool_axis_correction = (
            tool_axis_gain
            * position_null_space
            @ tool_axis_jacobian.T
            @ rotation_error_vector
        )
        tool_axis_correction = _scale_correction_to_joint_limits(
            candidate_positions,
            tool_axis_correction,
            joint_lower_bounds,
            joint_upper_bounds,
        )

        joint_correction = position_correction + tool_axis_correction
        correction_norm = np.linalg.norm(joint_correction)
        if correction_norm > max_joint_step:
            joint_correction *= max_joint_step / correction_norm

        candidate_positions = np.clip(
            candidate_positions + joint_correction,
            joint_lower_bounds,
            joint_upper_bounds,
        )

    if best_position_and_axis_result is not None:
        return best_position_and_axis_result

    assert best_position_result is not None
    return best_position_result
