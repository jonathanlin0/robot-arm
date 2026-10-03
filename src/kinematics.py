from dataclasses import dataclass, replace

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from robot_constants import ARM_JOINT_NAMES


TOOL_FRAME_SITE_NAME = "gripperframe"
DEFAULT_POSITION_TOLERANCE = 1e-3  # 1 mm; MuJoCo positions are in meters.
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_DAMPING = 0.02
DEFAULT_MAX_JOINT_STEP = 0.15
# Maximum accepted deviation from world-down in strict mode.
DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES = 100.0
DEFAULT_TOOL_AXIS_TOLERANCE = float(np.deg2rad(DEFAULT_TOOL_AXIS_TOLERANCE_DEGREES))
# this is similar to a learning rate
DEFAULT_TOOL_AXIS_GAIN = 0.2
DEFAULT_TOOL_YAW_TOLERANCE = float(np.deg2rad(1.0))
WORLD_DOWN = (0.0, 0.0, -1.0)


class IKConvergenceError(RuntimeError):
    """Required reset or action IK constraints failed to converge.

    This reports local solver nonconvergence, not proof of unreachability.
    Diagnostics are JSON-serializable so callers can record the failed scene.
    """

    def __init__(self, diagnostics: dict) -> None:
        self.diagnostics = diagnostics
        attempt = diagnostics["attempts"][-1]
        super().__init__(
            f"IK did not converge after {len(diagnostics['attempts'])} target attempts: "
            f"position error={attempt['position_error'] * 1000:.3f} mm, "
            f"downward error={np.degrees(attempt['tool_axis_error']):.2f} deg "
            f"(limit {np.degrees(diagnostics['tool_axis_tolerance']):.2f} deg), "
            f"jaw-plane error={np.degrees(attempt['tool_yaw_error']):.2f} deg."
        )


@dataclass(frozen=True)
class IKResult:
    """Outcome of a position-only inverse-kinematics solve."""

    joint_positions: np.ndarray
    converged: bool
    position_error: float
    iterations: int


@dataclass(frozen=True)
class ToolAxisIKResult:
    """Outcome of IK with position and gripper-orientation objectives.

    The solver prioritizes position, configured jaw-plane alignment, and then
    approach direction. Strict callers must require every enabled objective's
    convergence flag. Failure to converge is not proof of infeasibility.

    ``iterations`` identifies the returned candidate's iteration;
    ``total_iterations`` counts all attempted corrections, including later
    unsuccessful candidates. It is None only for legacy manually built results.
    """

    joint_positions: np.ndarray
    position_converged: bool
    tool_axis_converged: bool
    position_error: float
    tool_axis_error: float  # Radians between current and target tool axes.
    iterations: int
    tool_yaw_converged: bool = True
    # Angle from the closing axis to its target vertical plane, in radians.
    tool_yaw_error: float = 0.0
    total_iterations: int | None = None


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
    target_tool_yaw: float | None = None,
    tool_yaw_tolerance: float = DEFAULT_TOOL_YAW_TOLERANCE,
    require_downward: bool = True,
) -> ToolAxisIKResult:
    """Place the rigid gripper-frame site with a downward-orientation requirement.

    Both position and approach direction come from the rigid
    ``gripperframe`` site. Position is solved as the primary task. The
    rotational Jacobian is projected into the position Jacobian's null space,
    so the solver improves the tool direction without intentionally moving
    away from the requested XYZ position. By default ``require_downward``
    requires position, configured jaw-plane alignment, and world-down approach
    within their tolerances before early return. This overrides
    ``stop_when_position_converged``. Set ``require_downward=False`` to restore
    best-effort approach and permit position/yaw-only early stopping. That mode
    also permits an arbitrary ``target_tool_axis``. A positive ``minimum_iterations``
    lets an online controller apply at least one local orientation correction
    even when XYZ already starts within tolerance.

    Optional yaw sets the horizontal heading of a vertical jaw-alignment plane,
    modulo 180 degrees. Zero keeps local +Z (the jaw-closing axis) in the world
    XZ plane, independent of shoulder pan; opposite closing directions describe
    the same alignment. Its error is the angle out of that plane, which remains
    defined when the closing axis is vertical and its XY heading is undefined.
    Position remains primary, jaw-plane alignment is next, and the requested
    approach direction is solved in the remaining joint freedom. World-down is
    not reachable at every Cartesian position. When the local iteration budget
    is exhausted, this function returns its best candidate and convergence
    flags; the caller decides how to handle failure. It does not raise an IK
    failure exception or certify that the requested pose is impossible.

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

    if not isinstance(require_downward, (bool, np.bool_)):
        raise ValueError("require_downward must be a boolean.")
    if normalized_target_tool_axis.shape != (3,):
        raise ValueError("target_tool_axis must contain XYZ values.")

    target_axis_norm = np.linalg.norm(normalized_target_tool_axis)
    if not np.isfinite(target_axis_norm) or target_axis_norm == 0.0:
        raise ValueError("target_tool_axis must have nonzero length.")
    normalized_target_tool_axis /= target_axis_norm

    if require_downward and not np.allclose(
        normalized_target_tool_axis, WORLD_DOWN, rtol=0.0, atol=1e-12,
    ):
        raise ValueError(
            "require_downward=True requires target_tool_axis to point world-down; "
            "use require_downward=False for another approach direction."
        )

    if target_tool_yaw is not None and (
        not np.isscalar(target_tool_yaw) or not np.isfinite(target_tool_yaw)
    ):
        raise ValueError("target_tool_yaw must be finite or None.")
    if not np.isfinite(tool_yaw_tolerance) or tool_yaw_tolerance <= 0:
        raise ValueError("tool_yaw_tolerance must be finite and greater than zero.")

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

    best_position_result: ToolAxisIKResult | None = None
    best_position_and_axis_result: ToolAxisIKResult | None = None

    def candidate_rank(candidate: ToolAxisIKResult) -> tuple:
        if require_downward:
            # Once yaw meets tolerance, improve downward alignment rather
            # than preferring insignificant further yaw improvements.
            return (
                not candidate.tool_yaw_converged,
                not candidate.tool_axis_converged,
                candidate.tool_axis_error,
                candidate.tool_yaw_error,
                candidate.position_error,
            )
        return (
            candidate.tool_yaw_error,
            candidate.tool_axis_error,
            candidate.position_error,
        )

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

        closing_axis = tool_frame_site.xmat.reshape(3, 3)[:, 2]
        yaw_error = 0.0
        yaw_plane_error = 0.0
        target_side_axis = None
        if target_tool_yaw is not None:
            target_side_axis = np.array([-np.sin(target_tool_yaw), np.cos(target_tool_yaw), 0.0])
            # Align the jaw plane with the requested heading. Unlike atan2,
            # this stays well-defined when the closing axis is nearly vertical
            # at the high home pose. Opposite headings share the same plane,
            # avoiding a forced 180-degree wrist flip as pitch changes.
            yaw_plane_error = -float(closing_axis @ target_side_axis)
            yaw_error = float(np.arcsin(np.clip(yaw_plane_error, -1.0, 1.0)))

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
            tool_yaw_converged=bool(abs(yaw_error) <= tool_yaw_tolerance),
            tool_yaw_error=abs(yaw_error),
            total_iterations=iteration,
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

        if result.position_converged:
            if (
                best_position_and_axis_result is None
                or candidate_rank(result) < candidate_rank(best_position_and_axis_result)
            ):
                best_position_and_axis_result = result

        if (
            iteration >= minimum_iterations
            and result.position_converged
            and result.tool_yaw_converged
            and (
                (stop_when_position_converged and not require_downward)
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

        # Rotation about the tool axis does not change the direction of that
        # axis, so remove that irrelevant component of angular velocity.
        axis_projection = (
            np.eye(3)
            - np.outer(current_tool_axis, current_tool_axis)
        )
        tool_axis_jacobian = axis_projection @ arm_rotation_jacobian
        free_joints = np.ones(len(ARM_JOINT_NAMES), dtype=bool)
        # Reconsider the active limits on each IK iteration, allowing a joint
        # to move away from its bound whenever the next correction permits it.
        for _ in range(len(ARM_JOINT_NAMES) + 1):
            free_joint_projector = np.diag(free_joints.astype(float))
            free_position_jacobian = arm_position_jacobian @ free_joint_projector
            position_correction = free_position_jacobian.T @ np.linalg.solve(
                free_position_jacobian @ free_position_jacobian.T
                + position_damping_matrix,
                position_error_vector,
            )

            # Project into the position null space using only available
            # joints. A saturated joint must not stop the other joints from
            # rotating the claw while holding its Cartesian position.
            position_null_space = (
                free_joint_projector
                - np.linalg.pinv(free_position_jacobian)
                @ free_position_jacobian
            )
            orientation_null_space = position_null_space
            yaw_correction = np.zeros(len(ARM_JOINT_NAMES))
            if target_side_axis is not None:
                yaw_jacobian = np.cross(closing_axis, target_side_axis) @ arm_rotation_jacobian
                free_yaw_jacobian = yaw_jacobian @ position_null_space
                yaw_norm_squared = float(free_yaw_jacobian @ free_yaw_jacobian)
                yaw_correction = free_yaw_jacobian * (
                    (yaw_plane_error - yaw_jacobian @ position_correction)
                    / (yaw_norm_squared + damping**2)
                )
                if yaw_norm_squared > 1e-12:
                    # Pitch corrections preserve the constrained yaw too.
                    orientation_null_space = position_null_space - np.outer(
                        free_yaw_jacobian, free_yaw_jacobian,
                    ) / yaw_norm_squared

            tool_axis_correction = (
                tool_axis_gain
                * orientation_null_space
                @ tool_axis_jacobian.T
                @ rotation_error_vector
            )
            orientation_correction = yaw_correction + tool_axis_correction

            if target_side_axis is None and not require_downward:
                # Keep the legacy direction-only solver's limit behavior.
                orientation_correction = _scale_correction_to_joint_limits(
                    candidate_positions,
                    orientation_correction,
                    joint_lower_bounds,
                    joint_upper_bounds,
                )
                joint_correction = position_correction + orientation_correction
                break

            joint_correction = position_correction + orientation_correction
            blocked_joints = free_joints & (
                ((candidate_positions <= joint_lower_bounds + 1e-10)
                 & (joint_correction < -1e-12))
                | ((candidate_positions >= joint_upper_bounds - 1e-10)
                   & (joint_correction > 1e-12))
            )
            if not np.any(blocked_joints):
                # Frozen columns can retain floating-point residue after
                # pseudoinversion; remove it before the limit calculation.
                joint_correction[~free_joints] = 0.0
                break
            free_joints[blocked_joints] = False

        correction_norm = np.linalg.norm(joint_correction)
        if correction_norm > max_joint_step:
            joint_correction *= max_joint_step / correction_norm

        if target_side_axis is not None or require_downward:
            # Stop at the first newly reached limit rather than clipping
            # individual components and losing the Cartesian null-space
            # relationship. The next iteration solves with that limit active.
            joint_correction = _scale_correction_to_joint_limits(
                candidate_positions,
                joint_correction,
                joint_lower_bounds,
                joint_upper_bounds,
            )

        candidate_positions = np.clip(
            candidate_positions + joint_correction,
            joint_lower_bounds,
            joint_upper_bounds,
        )

    if best_position_and_axis_result is not None:
        return replace(best_position_and_axis_result, total_iterations=max_iterations)

    assert best_position_result is not None
    return replace(best_position_result, total_iterations=max_iterations)
