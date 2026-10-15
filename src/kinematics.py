"""Robot geometry, joint limits, and local constrained inverse kinematics.

This module uses scratch MuJoCo data for geometry only. It never steps physics
or writes to the live simulator, so the same controller can serve hardware.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robot_constants import ROBOT_JOINT_NAMES, TOOL_FRAME_SITE_NAME
from schemas.config import CartesianActionConfig


def joint_vector(values: np.ndarray, name: str = "joint targets") -> np.ndarray:
    """Return an independent finite six-joint vector in the agreed order."""
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(ROBOT_JOINT_NAMES),) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite values in robot joint order.")
    return result.copy()


def joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Intersect each named joint's angular and actuator control limits."""
    lower, upper = [], []
    for name in ROBOT_JOINT_NAMES:
        joint, actuator = model.joint(name), model.actuator(name)
        lo, hi = -np.inf, np.inf
        if model.jnt_limited[joint.id]:
            lo, hi = map(float, joint.range)
        if model.actuator_ctrllimited[actuator.id]:
            lo = max(lo, float(actuator.ctrlrange[0]))
            hi = min(hi, float(actuator.ctrlrange[1]))
        if lo > hi:
            raise ValueError(f"Joint {name!r} has incompatible joint and actuator limits.")
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def forward_kinematics(model: mujoco.MjModel, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rigid gripperframe XYZ and rotation matrix without offsets."""
    values = joint_vector(positions, "joint positions")
    data = mujoco.MjData(model)
    for name, value in zip(ROBOT_JOINT_NAMES, values, strict=True):
        data.joint(name).qpos[0] = value
    mujoco.mj_forward(model, data)
    site = data.site(TOOL_FRAME_SITE_NAME)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def _rotation_between(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross = np.cross(current, target)
    norm = float(np.linalg.norm(cross))
    cosine = float(np.clip(current @ target, -1.0, 1.0))
    if norm > 1e-12:
        return cross * (np.arctan2(norm, cosine) / norm)
    if cosine > 0.0:
        return np.zeros(3)
    basis = np.eye(3)[np.argmin(np.abs(current))]
    axis = np.cross(current, basis)
    return np.pi * axis / np.linalg.norm(axis)


class IKConvergenceError(RuntimeError):
    """Local IK failed; diagnostics describe attempts, not proof of infeasibility."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        attempts = diagnostics["attempts"]
        last = attempts[-1]
        super().__init__(
            f"IK did not converge after {len(attempts)} target attempts: "
            f"position error={last['position_error'] * 1000:.3f} mm, "
            f"downward error={np.degrees(last['tool_axis_error']):.2f} deg, "
            f"jaw-plane error={np.degrees(last['tool_yaw_error']):.2f} deg."
        )


def solve_ik(
    model: mujoco.MjModel,
    initial_positions: np.ndarray,
    target_position: np.ndarray,
    config: CartesianActionConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve position, jaw plane, and required downward limits with damped IK.

    Local +X of gripperframe is the tool axis. Local +Z is the jaw closing
    direction. Yaw means that closing direction lies in a vertical world plane
    of the requested heading, modulo pi, rather than an Euler-angle constraint.
    Downward orientation remains a soft preference inside its hard acceptance
    tolerance. Improve it in the position/jaw objective's null space, retaining
    the best feasible solution when exact downward orientation is unavailable.
    """
    positions = joint_vector(initial_positions, "initial joint positions")
    target = np.asarray(target_position, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_position must contain three finite XYZ values.")
    lower, upper = joint_limits(model)
    positions = np.clip(positions, lower, upper)
    target_axis = np.asarray(config.target_tool_axis, dtype=np.float64)
    if target_axis.shape != (3,) or not np.isfinite(target_axis).all() or np.linalg.norm(target_axis) == 0:
        raise ValueError("target_tool_axis must be a finite nonzero XYZ vector.")
    target_axis /= np.linalg.norm(target_axis)
    if config.require_downward and not np.allclose(target_axis, [0.0, 0.0, -1.0]):
        raise ValueError("require_downward requires target_tool_axis to point world-down.")
    data = mujoco.MjData(model)
    site_id = model.site(TOOL_FRAME_SITE_NAME).id
    dofs = np.array([model.jnt_dofadr[model.joint(name).id] for name in ROBOT_JOINT_NAMES[:-1]])
    side = None if config.target_tool_yaw is None else np.array([
        -np.sin(config.target_tool_yaw), np.cos(config.target_tool_yaw), 0.0,
    ])
    best_positions = positions.copy()
    best_score = float("inf")
    best_diagnostics: dict[str, object] = {}
    best_valid: tuple[np.ndarray, dict[str, object]] | None = None
    best_valid_angle = float("inf")
    unchanged_valid_iterations = 0
    weight = config.orientation_weight
    for iteration in range(config.ik_max_iterations + 1):
        for name, value in zip(ROBOT_JOINT_NAMES, positions, strict=True):
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        site = data.site(site_id)
        rotation = site.xmat.reshape(3, 3)
        axis, closing = rotation[:, 0], rotation[:, 2]
        position_error_vector = target - site.xpos
        position_error = float(np.linalg.norm(position_error_vector))
        axis_rotation = _rotation_between(axis, target_axis)
        axis_error = float(np.linalg.norm(axis_rotation))
        jaw_residual = 0.0 if side is None else -float(closing @ side)
        jaw_error = float(np.arcsin(np.clip(abs(jaw_residual), 0.0, 1.0)))
        converged = (
            position_error <= config.position_tolerance
            and (not config.require_downward or axis_error <= config.tool_axis_tolerance)
            and (side is None or jaw_error <= config.tool_yaw_tolerance)
        )
        diagnostics = {
            "converged": bool(converged), "position_error": position_error,
            "tool_axis_error": axis_error, "tool_yaw_error": jaw_error,
            "iterations": iteration,
        }
        score = (position_error / config.position_tolerance) ** 2
        score += (jaw_error / config.tool_yaw_tolerance) ** 2 if side is not None else 0.0
        if config.require_downward:
            score += (max(0.0, axis_error - config.tool_axis_tolerance) / config.tool_axis_tolerance) ** 2
        if score < best_score:
            best_score, best_positions, best_diagnostics = score, positions.copy(), diagnostics
        if converged:
            if axis_error < best_valid_angle - 1e-6:
                best_valid = (positions.copy(), diagnostics)
                best_valid_angle = axis_error
                unchanged_valid_iterations = 0
            else:
                unchanged_valid_iterations += 1
            if not config.require_downward or axis_error < 1e-3:
                return positions.copy(), diagnostics
        elif best_valid is not None:
            unchanged_valid_iterations += 1
        if best_valid is not None and unchanged_valid_iterations >= 10:
            break
        if iteration == config.ik_max_iterations:
            break
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, linear, angular, site_id)
        linear, angular = linear[:, dofs], angular[:, dofs]
        matrices, residuals = [linear], [position_error_vector]
        if side is not None:
            matrices.append(weight * (np.cross(closing, side) @ angular)[None, :])
            residuals.append(np.array([weight * jaw_residual]))
        jacobian, residual = np.vstack(matrices), np.concatenate(residuals)
        correction = np.linalg.solve(
            jacobian.T @ jacobian + config.ik_damping ** 2 * np.eye(len(dofs)),
            jacobian.T @ residual,
        )
        if (config.require_downward and position_error <= config.position_tolerance * 0.5
                and (side is None or jaw_error <= config.tool_yaw_tolerance * 0.5)):
            null_space = np.eye(len(dofs)) - np.linalg.pinv(jacobian) @ jacobian
            axis_jacobian = (np.eye(3) - np.outer(axis, axis)) @ angular @ null_space
            axis_correction = axis_jacobian.T @ np.linalg.solve(
                axis_jacobian @ axis_jacobian.T + config.ik_damping ** 2 * np.eye(3),
                axis_rotation,
            )
            correction += weight * null_space @ axis_correction
        largest = float(np.max(np.abs(correction)))
        if largest > config.ik_max_joint_step:
            correction *= config.ik_max_joint_step / largest
        positions[:-1] = np.clip(positions[:-1] + correction, lower[:-1], upper[:-1])
    if best_valid is not None:
        joints, diagnostics = best_valid
        return joints, {**diagnostics, "total_iterations": iteration}
    return best_positions, {**best_diagnostics, "total_iterations": iteration}
