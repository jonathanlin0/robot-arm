"""Scripted stacking episodes with learned or legacy ground-truth observations.

The teacher follows the legacy approach/grasp/lift/stack sequence. Disturbances
live only at the actuator boundary: labels retain intended actions, while the
next observation and accepted target describe what actually happened. Recovery
restarts teacher progress without resetting physics or deleting history.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil, floor, isfinite
from typing import Any, TYPE_CHECKING

import mujoco
import numpy as np
import torch

from action_disturbance import DropDisturbanceBackend
from action_evaluation import StackSuccessTracker
from cartesian_controller import CartesianController
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from kinematics import IKConvergenceError, solve_ik
from robot_constants import ROBOT_JOINT_NAMES
from robot_state_provider import SimRobotStateProvider
from schemas import OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE, CubeState, RobotState
from schemas.config import ProjectConfig
from trained_policy import build_observation

if TYPE_CHECKING:
    from action_observation import VisionActionObserver


OPEN_GRIPPER = 1.0
CLOSED_GRIPPER = -1.0


class DemonstrationFailure(RuntimeError):
    """An unsuccessful physical attempt that may be retried with a new seed."""


@dataclass
class ActionEpisode:
    observations: torch.Tensor
    prev_accepted_targets: torch.Tensor
    actions: torch.Tensor
    stable_time: float
    required_stable_time: float
    stage_steps: dict[str, int]
    disturbance_events: list[dict[str, Any]] = field(default_factory=list)
    recovery_count: int = 0
    waypoint_retry_count: int = 0
    grasp_seen: bool = False
    success: bool = True
    observation_source: str = "simulation_ground_truth"

    def validate(self) -> None:
        """Reject malformed or unsuccessful demonstrations before publication."""
        steps = len(self.actions)
        if steps < 1:
            raise ValueError("An episode must contain at least one action.")
        widths = {"simulation_ground_truth": OBSERVATION_SIZE, "vision_model": POLICY_OBSERVATION_SIZE}
        if self.observation_source not in widths:
            raise ValueError(f"Unknown observation source {self.observation_source!r}.")
        for name, shape in (("observations", (steps, widths[self.observation_source])),
                            ("prev_accepted_targets", (steps, 3)), ("actions", (steps, 4))):
            tensor = getattr(self, name)
            if (tensor.shape != shape or tensor.dtype != torch.float32
                    or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite CPU float32 with shape {shape}.")
        if torch.any(self.actions.abs() > 1.0):
            raise ValueError("Action labels must be normalized to [-1, 1].")
        if (not self.success or not self.grasp_seen or not isfinite(self.stable_time)
                or not isfinite(self.required_stable_time) or self.required_stable_time <= 0
                or self.stable_time + 1e-12 < self.required_stable_time):
            raise ValueError("The episode must end with a verified released, stable stack after grasping.")
        if (not self.stage_steps or sum(self.stage_steps.values()) != steps
                or any(type(count) is not int or count < 1 for count in self.stage_steps.values())):
            raise ValueError("Stage counts must account for every recorded action.")
        if self.recovery_count < 0 or not 0 <= self.waypoint_retry_count <= self.recovery_count:
            raise ValueError("Recovery counts must be nonnegative and include every waypoint retry.")

    def metadata(self) -> dict[str, Any]:
        return {
            "steps": len(self.actions), "success": self.success,
            "stable_time": self.stable_time, "required_stable_time": self.required_stable_time,
            "stage_steps": dict(self.stage_steps), "stages": sorted(self.stage_steps),
            "disturbance_events": [dict(event) for event in self.disturbance_events],
            "recovery_count": self.recovery_count, "waypoint_retry_count": self.waypoint_retry_count,
            "grasp_seen": self.grasp_seen,
            "observation_source": self.observation_source,
        }


def reset_demonstration_environment(environment: CubeStackEnvironment, project: ProjectConfig,
                                    rng: np.random.Generator) -> None:
    """Initialize the legacy randomized open-claw home and axis-aligned cubes."""
    config = project.environment.action_data
    environment.reset()
    half_range = np.asarray(config.start_position_half_range)
    home = rng.uniform(np.asarray(config.start_position) - half_range,
                       np.asarray(config.start_position) + half_range)
    initial = np.zeros(len(ROBOT_JOINT_NAMES))
    initial[-1] = project.cartesian_controller.open_gripper_target
    joints, outcome = solve_ik(environment.model, initial, home, project.cartesian_controller)
    if not outcome["converged"]:
        raise IKConvergenceError({"attempts": [{"target": home.tolist(), **outcome}],
                                  "stage": "reset", "episode_step": 0})
    for _ in range(config.spawn_maximum_attempts):
        xy = np.column_stack((rng.uniform(*config.cube_x_range, size=2),
                              rng.uniform(*config.cube_y_range, size=2)))
        if np.linalg.norm(xy[0] - xy[1]) >= config.minimum_cube_separation:
            break
    else:
        raise DemonstrationFailure("Reset: could not sample sufficiently separated cubes.")
    with environment.state_lock:
        for name, angle in zip(ROBOT_JOINT_NAMES, joints, strict=True):
            environment.data.joint(name).qpos[0] = angle
            environment.data.actuator(name).ctrl[0] = angle
        table = environment.model.geom("table").id
        table_top = float(environment.data.geom_xpos[table, 2] + environment.model.geom_size[table, 2])
        for name, position in zip(("orange", "blue"), xy, strict=True):
            joint = environment.data.joint(f"{name}_cube_joint")
            half_height = float(environment.model.geom(f"{name}_cube_geom").size[2])
            joint.qpos[:] = [*position, table_top + half_height, 1.0, 0.0, 0.0, 0.0]
            joint.qvel[:] = 0
        mujoco.mj_forward(environment.model, environment.data)


class EpisodeRecorder:
    """Record pre-action policy inputs and supervise exact measured progress."""

    def __init__(self, project: ProjectConfig, environment: CubeStackEnvironment, seed: int,
                 observation_builder: Callable[[RobotState], np.ndarray] | None = None) -> None:
        self.project, self.environment = project, environment
        self.observation_builder = observation_builder
        self.config = project.environment.action_data
        self.interval = 1.0 / project.loop.control_hz
        self.maximum_steps = floor(project.environment.episode_seconds / self.interval + 1e-9)
        if self.maximum_steps < 1:
            raise ValueError("episode_seconds must allow at least one control tick.")
        self.provider = SimRobotStateProvider(environment)
        self.controller = CartesianController(environment.model, project.cartesian_controller)
        self.controller.reset(self.provider.read())
        self.backend = DropDisturbanceBackend(
            backend=SimJointCommandBackend(environment), config=self.config.drop_disturbance,
            rng=np.random.default_rng(seed), control_interval=self.interval,
            open_gripper_target=project.cartesian_controller.open_gripper_target,
        )
        self.scorer = StackSuccessTracker(environment.model, self.config.success)
        self.observations: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.stages: Counter[str] = Counter()
        self.recovery_requested = False
        self.recovery_count = self.waypoint_retry_count = self._lost_grasp_steps = 0
        self.waypoint_reached_time: float | None = None
        self.grasp_seen = self._pickup_grasp_seen = self.finished = False
        self._table = environment.model.geom("table").id
        self._orange = environment.model.geom("orange_cube_geom").id
        self._fixed_pads = self._pad_ids("fixed_jaw", ("box3", "box4", "box5", "box6", "box7"))
        self._moving_pads = self._pad_ids("moving_jaw", ("box2", "box3"))
        self.initial_heights = {
            name: float(environment.data.body(f"{name}_cube").xpos[2]) for name in ("orange", "blue")
        }

    def _pad_ids(self, prefix: str, boxes: tuple[str, ...]) -> set[int]:
        return {self.environment.model.geom(f"{prefix}_{suffix}").id
                for suffix in ("sph_tip1", "sph_tip2", "sph_tip3", *boxes)}

    def scene(self) -> dict[str, Any]:
        """Snapshot exact cube pose/velocity and the legacy two-pad grasp test."""
        with self.environment.state_lock:
            data = self.environment.data
            robot = self.provider.read()
            cubes = {
                name: CubeState(timestamp=robot.timestamp,
                                position=data.body(f"{name}_cube").xpos.copy(),
                                orientation=data.body(f"{name}_cube").xquat.copy(),
                                linear_velocity=data.joint(f"{name}_cube_joint").qvel[:3].copy())
                for name in ("orange", "blue")
            }
            touching = set()
            for contact in data.contact:
                if int(contact.geom1) == self._orange:
                    touching.add(int(contact.geom2))
                elif int(contact.geom2) == self._orange:
                    touching.add(int(contact.geom1))
            return {"robot": robot, "cubes": cubes, "time": robot.timestamp,
                    "orange": cubes["orange"].position, "blue": cubes["blue"].position,
                    "gripper": robot.gripper_position,
                    "orange_velocity": data.joint("orange_cube_joint").qvel.copy(),
                    "held": bool(touching & self._fixed_pads and touching & self._moving_pads
                                 and self._table not in touching)}

    def begin_recovery(self) -> None:
        """Reset only teacher progress, preserving all simulator/recorded state."""
        self.recovery_count += 1
        self.recovery_requested = False
        self.waypoint_reached_time = None
        self._pickup_grasp_seen = False
        self._lost_grasp_steps = 0

    def _observe_progress(self, scene: dict[str, Any], stage: str) -> None:
        if stage in ("release", "retreat", "settle_stack"):
            self.waypoint_reached_time = None
        if scene["held"]:
            self.grasp_seen = self._pickup_grasp_seen = True
            self._lost_grasp_steps = 0
        elif stage in ("lift", "settle_lift", "transport", "lower") and self._pickup_grasp_seen:
            self._lost_grasp_steps += 1
            if self._lost_grasp_steps >= self.config.recovery.lost_grasp_steps:
                self.recovery_requested = True
        elif stage in ("release", "retreat", "settle_stack"):
            self._pickup_grasp_seen = False
            self._lost_grasp_steps = 0
        if (self.waypoint_reached_time is not None and not self.recovery_requested and not scene["held"]
                and scene["time"] - self.waypoint_reached_time
                > self.config.recovery.waypoint_grasp_retry_seconds + 1e-9):
            self.waypoint_retry_count += 1
            self.recovery_requested = True

    def step(self, action: np.ndarray, stage: str) -> bool:
        if self.finished:
            raise RuntimeError("Cannot record another action after success.")
        if len(self.actions) >= self.maximum_steps:
            raise DemonstrationFailure(f"{stage}: episode timed out.")
        intended = np.array(action, dtype=np.float32, copy=True)
        if intended.shape != (4,) or not np.isfinite(intended).all() or np.any(abs(intended) > 1):
            raise ValueError("Teacher actions must be four finite values in [-1, 1].")
        with self.environment.state_lock:
            before = self.scene()
            if self.observation_builder is None:
                observation = build_observation(before["robot"], before["cubes"])
            else:
                observation = np.array(self.observation_builder(before["robot"]), dtype=np.float32, copy=True)
                if observation.shape != (POLICY_OBSERVATION_SIZE,) or not np.isfinite(observation).all():
                    raise ValueError("Vision observations must contain 37 finite values.")
        target = np.array(self.controller.prev_accepted_target, dtype=np.float32, copy=True)
        self.backend.begin_step(robot_state=before["robot"], orange_held=before["held"])
        try:
            command = self.controller.compute(action=intended, robot_state=before["robot"])
        except IKConvergenceError as error:
            error.diagnostics.update(stage=stage, episode_step=len(self.actions) + 1)
            raise
        receipt = self.backend.send_joint_targets(command.joint_targets)
        self.controller.update_prev_accepted_xyz(command=command, receipt=receipt)
        self.backend.advance(self.interval)
        self.observations.append(observation)
        self.targets.append(target)
        self.actions.append(intended)
        self.stages[stage] += 1
        after = self.scene()
        self._observe_progress(after, stage)
        if any(after[name][2] < self.initial_heights[name] - 0.005 for name in ("orange", "blue")):
            raise DemonstrationFailure(f"{stage}: a cube fell below the table.")
        if self.scorer.update(self.environment.data) and self.grasp_seen:
            self.finished = True
        return self.finished

    def episode(self) -> ActionEpisode:
        if not self.finished or self.scorer.stable_since is None:
            raise DemonstrationFailure("Episode has not ended in a verified successful stack.")
        episode = ActionEpisode(
            observations=torch.from_numpy(np.stack(self.observations)),
            prev_accepted_targets=torch.from_numpy(np.stack(self.targets)),
            actions=torch.from_numpy(np.stack(self.actions)),
            stable_time=float(self.environment.data.time - self.scorer.stable_since),
            required_stable_time=self.config.success.required_stable_time,
            stage_steps=dict(self.stages), disturbance_events=list(self.backend.events),
            recovery_count=self.recovery_count, waypoint_retry_count=self.waypoint_retry_count,
            grasp_seen=self.grasp_seen,
            observation_source=("vision_model" if self.observation_builder is not None
                                else "simulation_ground_truth"),
        )
        episode.validate()
        return episode


def action_toward(recorder: EpisodeRecorder, target: np.ndarray, gripper: float) -> np.ndarray:
    delta = (target - recorder.controller.prev_accepted_target) / recorder.controller.config.maximum_position_delta
    return np.array([*np.clip(delta, -1.0, 1.0), gripper], dtype=np.float32)


def move_to(recorder: EpisodeRecorder, target: np.ndarray, gripper: float, stage: str) -> bool:
    near_steps = 0
    for _ in range(recorder.config.teacher.maximum_move_steps):
        if recorder.step(action_toward(recorder, target, gripper), stage):
            return True
        if recorder.recovery_requested:
            return False
        near = np.linalg.norm(target - recorder.scene()["gripper"]) <= recorder.config.teacher.position_tolerance
        near_steps = near_steps + 1 if near else 0
        if near_steps >= recorder.config.teacher.consecutive_near_steps:
            return False
    raise DemonstrationFailure(f"{stage}: gripper did not reach the requested pose.")


def hold_position(recorder: EpisodeRecorder, gripper: float, steps: int, stage: str) -> bool:
    for _ in range(steps):
        if recorder.step(np.array([0.0, 0.0, 0.0, gripper]), stage):
            return True
        if recorder.recovery_requested:
            return False
    return False


def move_held_cube_to(recorder: EpisodeRecorder, cube_target: np.ndarray, stage: str) -> bool:
    near_steps = 0
    for _ in range(recorder.config.teacher.maximum_move_steps):
        scene = recorder.scene()
        if not scene["held"]:
            near_steps = 0
            if not recorder._pickup_grasp_seen:
                recorder.recovery_requested = True
                return False
            if hold_position(recorder, CLOSED_GRIPPER, 1, stage):
                return True
            if recorder.recovery_requested:
                return False
            continue
        target = cube_target - (scene["orange"] - scene["gripper"])
        if recorder.step(action_toward(recorder, target, CLOSED_GRIPPER), stage):
            return True
        if recorder.recovery_requested:
            return False
        near = np.linalg.norm(cube_target - recorder.scene()["orange"]) <= recorder.config.teacher.position_tolerance
        near_steps = near_steps + 1 if near else 0
        if near_steps >= recorder.config.teacher.consecutive_near_steps:
            return False
    raise DemonstrationFailure(f"{stage}: orange did not reach the requested pose.")


def pickup_from_current_pose(recorder: EpisodeRecorder) -> bool:
    teacher = recorder.config.teacher
    waypoint = recorder.scene()["orange"] + [0.0, 0.0, teacher.waypoint_height]
    if move_to(recorder, waypoint, OPEN_GRIPPER, "approach"):
        return True
    if recorder.recovery_requested:
        return False
    recorder.waypoint_reached_time = recorder.scene()["time"]
    if hold_position(recorder, OPEN_GRIPPER, teacher.open_settle_steps, "settle_waypoint"):
        return True
    if recorder.recovery_requested:
        return False
    grasp_target = recorder.scene()["orange"].copy()
    yaw = recorder.controller.config.target_tool_yaw
    direction = (grasp_target[:2] / np.linalg.norm(grasp_target[:2]) if yaw is None
                 else np.array([np.cos(yaw), np.sin(yaw)]))
    grasp_target[:2] -= teacher.grasp_radial_offset * direction
    grasp_target[2] += teacher.grasp_height_offset
    steps = (
        lambda: move_to(recorder, grasp_target, OPEN_GRIPPER, "descend"),
        lambda: hold_position(recorder, OPEN_GRIPPER, teacher.open_settle_steps, "settle"),
        lambda: hold_position(recorder, CLOSED_GRIPPER, teacher.close_gripper_steps, "close"),
        lambda: move_to(recorder, waypoint, CLOSED_GRIPPER, "lift"),
        lambda: hold_position(recorder, CLOSED_GRIPPER, teacher.lift_settle_steps, "settle_lift"),
    )
    for step in steps:
        if step():
            return True
        if recorder.recovery_requested:
            return False
    if not recorder.scene()["held"]:
        recorder.recovery_requested = True
    return False


def stack_held_cube(recorder: EpisodeRecorder) -> bool:
    teacher = recorder.config.teacher
    scene = recorder.scene()
    transport_target = scene["orange"].copy()
    transport_target[:2] = scene["blue"][:2]
    stack_distance = recorder.scorer.expected_vertical_distance
    if transport_target[2] <= scene["blue"][2] + stack_distance:
        raise DemonstrationFailure("Lifted orange cube does not clear blue.")
    if move_held_cube_to(recorder, transport_target, "transport"):
        return True
    if recorder.recovery_requested:
        return False
    placement_target = recorder.scene()["blue"] + [0.0, 0.0, stack_distance]
    if move_held_cube_to(recorder, placement_target, "lower"):
        return True
    if recorder.recovery_requested:
        return False
    if hold_position(recorder, OPEN_GRIPPER, teacher.release_steps, "release"):
        return True
    retreat_target = recorder.scene()["gripper"] + [0.0, 0.0, teacher.retreat_distance]
    if move_to(recorder, retreat_target, OPEN_GRIPPER, "retreat"):
        return True
    settle_steps = ceil(recorder.config.success.required_stable_time / recorder.interval)
    return hold_position(recorder, OPEN_GRIPPER, settle_steps + teacher.extra_hold_steps, "settle_stack")


def recover_for_pickup(recorder: EpisodeRecorder) -> bool:
    """Wait for measured settling, then approach above the current orange pose."""
    recorder.begin_recovery()
    recovery = recorder.config.recovery
    if hold_position(recorder, OPEN_GRIPPER, recovery.pause_steps, "recovery_pause"):
        return True
    settled = 0
    for _ in range(recorder.config.teacher.maximum_move_steps):
        if hold_position(recorder, OPEN_GRIPPER, 1, "recovery_settle"):
            return True
        scene = recorder.scene()
        velocity = scene["orange_velocity"]
        quiet = (not scene["held"] and np.linalg.norm(velocity[:3]) <= recovery.max_linear_speed
                 and np.linalg.norm(velocity[3:]) <= recovery.max_angular_speed)
        settled = settled + 1 if quiet else 0
        if settled >= recovery.settle_steps:
            return False
    raise DemonstrationFailure("Recovery: orange did not settle for another pickup.")


def collect_action_episode(*, project: ProjectConfig, environment: CubeStackEnvironment, seed: int,
                           vision_observer: VisionActionObserver | None = None) -> ActionEpisode:
    """Collect a successful episode, optionally observing vision without waits."""
    reset_demonstration_environment(environment, project, np.random.default_rng(seed))
    if vision_observer is not None:
        vision_observer.reset()
    recorder = EpisodeRecorder(project, environment, seed,
                               observation_builder=vision_observer.observe if vision_observer is not None else None)
    while not recorder.finished:
        if pickup_from_current_pose(recorder):
            return recorder.episode()
        if not recorder.recovery_requested and stack_held_cube(recorder):
            return recorder.episode()
        if recover_for_pickup(recorder):
            return recorder.episode()
    raise DemonstrationFailure("Episode ended without a stable stack.")
