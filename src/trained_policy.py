"""Shared observation, temporal-history, and policy inference implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from cube_yaw import quaternion_to_yaw
from schemas import OBSERVATION_FIELDS, OBSERVATION_FIELD_LAYOUT, HISTORY_TOKEN_SIZE, CubeStates, PolicyAction, RobotState, Vec3
from schemas import POLICY_OBSERVATION_FIELD_LAYOUT


def build_observation(robot_state: RobotState, cube_states: CubeStates,
                      observation_fields: tuple[str, ...] = OBSERVATION_FIELDS) -> NDArray[np.float32]:
    """Flatten raw robot/cube fields in saved-demonstration order as float32."""
    values = {
        "joint_positions": robot_state.joint_positions,
        "joint_velocities": robot_state.joint_velocities,
        "commanded_joint_targets": robot_state.commanded_joint_targets,
        "gripper_position": robot_state.gripper_position,
    }
    for name in ("orange", "blue"):
        cube = cube_states[name]
        if cube.orientation is None:
            raise ValueError(f"Raw observation for {name} requires a quaternion orientation.")
        values[f"{name}_position"] = cube.position
        values[f"{name}_orientation"] = cube.orientation
        values[f"{name}_linear_velocity"] = cube.linear_velocity
    sizes = dict(OBSERVATION_FIELD_LAYOUT)
    fields = []
    for name in observation_fields:
        value = np.asarray(values[name], dtype=np.float32)
        if value.shape != (sizes[name],):
            raise ValueError(f"Observation field {name} must have shape ({sizes[name]},).")
        fields.append(value)
    return np.concatenate(fields).astype(np.float32, copy=False)


def build_policy_observation(robot_state: RobotState, cube_states: CubeStates) -> NDArray[np.float32]:
    """Build 37 values from already-normalized yaw or raw simulator poses."""
    values = {
        "joint_positions": robot_state.joint_positions,
        "joint_velocities": robot_state.joint_velocities,
        "commanded_joint_targets": robot_state.commanded_joint_targets,
        "gripper_position": robot_state.gripper_position,
    }
    for name in ("orange", "blue"):
        cube = cube_states[name]
        if cube.yaw is not None:
            # Vision normalizes once on-device. Repeating epsilon-clamped
            # normalization would change near-zero predictions a second time.
            yaw = np.asarray(cube.yaw, dtype=np.float32)
            if yaw.shape != (2,) or not np.isfinite(yaw).all():
                raise ValueError(f"Policy observation for {name} requires a finite two-value yaw pair.")
        elif cube.orientation is not None:
            yaw = quaternion_to_yaw(cube.orientation)
        else:
            raise ValueError(f"Policy observation for {name} requires yaw or quaternion orientation.")
        values[f"{name}_position"] = cube.position
        values[f"{name}_yaw"] = yaw
        values[f"{name}_linear_velocity"] = cube.linear_velocity
    fields = []
    for name, size in POLICY_OBSERVATION_FIELD_LAYOUT:
        value = np.asarray(values[name], dtype=np.float32)
        if value.shape != (size,):
            raise ValueError(f"Policy observation field {name} must have shape ({size},).")
        fields.append(value)
    return np.concatenate(fields).astype(np.float32, copy=False)


class TrainedPolicy:
    """Build observations and right-padded history without performing motor I/O."""

    def __init__(self, policy: Any, observation_fields: tuple[str, ...] = OBSERVATION_FIELDS,
                 history_length: int = 384) -> None:
        if tuple(observation_fields) != OBSERVATION_FIELDS:
            raise ValueError("Policy observation fields must match the current 41-value observation contract.")
        if history_length < 1:
            raise ValueError("history_length must be positive.")
        if hasattr(policy, "config") and hasattr(policy.config, "history_length") and policy.config.history_length != history_length:
            raise ValueError("Runtime history length must match the loaded policy configuration.")
        self.policy = policy
        self.observation_fields = tuple(observation_fields)
        self.history_length = history_length
        self.tokens = np.zeros((history_length, HISTORY_TOKEN_SIZE), dtype=np.float32)
        self.valid_mask = np.zeros(history_length, dtype=np.bool_)
        self.episode_start_mask = np.zeros(history_length, dtype=np.bool_)
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.valid_length = 0
        self._episode_started = False
        if isinstance(policy, nn.Module):
            policy.eval()

    def reset(self) -> None:
        self.tokens.fill(0.0)
        self.valid_mask.fill(False)
        self.episode_start_mask.fill(False)
        self.previous_action.fill(0.0)
        self.valid_length = 0
        self._episode_started = False

    def predict(self, robot_state: RobotState, cube_states: CubeStates,
                prev_accepted_target: Vec3, deterministic: bool = True) -> PolicyAction:
        observation = build_policy_observation(robot_state, cube_states)
        target = np.asarray(prev_accepted_target, dtype=np.float32)
        if target.shape != (3,):
            raise ValueError("prev_accepted_target must contain three XYZ values.")
        token = np.concatenate((observation, self.previous_action, target))
        if self.valid_length == self.history_length:
            self.tokens[:-1] = self.tokens[1:]
            self.episode_start_mask[:-1] = self.episode_start_mask[1:]
            index = self.history_length - 1
        else:
            index = self.valid_length
            self.valid_length += 1
        self.tokens[index] = token
        self.valid_mask[index] = True
        self.episode_start_mask[index] = not self._episode_started
        self._episode_started = True
        if isinstance(self.policy, nn.Module):
            parameter = next(self.policy.parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            # A single runtime history needs no trailing padding. The same
            # position/mask conventions apply as during padded batch training.
            length = self.valid_length
            with torch.inference_mode():
                action = self.policy.predict_action(
                    tokens=torch.as_tensor(self.tokens[None, :length], device=device),
                    valid=torch.as_tensor(self.valid_mask[None, :length], device=device),
                    episode_start=torch.as_tensor(self.episode_start_mask[None, :length], device=device),
                    deterministic=deterministic,
                ).cpu().numpy()[0]
        else:
            # Keep a narrow duck-typed hook for previously loaded prediction
            # wrappers without introducing a runtime Stable-Baselines dependency.
            action, _ = self.policy.predict({
                "tokens": self.tokens.copy(), "valid": self.valid_mask.copy(),
                "episode_start": self.episode_start_mask.copy(),
            }, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(4)
        return np.clip(action, -1.0, 1.0)

    def record_action(self, action: PolicyAction) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (4,):
            raise ValueError("Issued policy action must have shape (4,).")
        self.previous_action = value.copy()
