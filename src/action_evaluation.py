"""Closed-loop, image-based action validation in unpaced MuJoCo simulation."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from math import floor
from typing import Any, Literal

import mujoco
import numpy as np
import torch
from torch import nn

from camera_feed_getter import SIM_CAMERA_NAMES, SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from cartesian_controller import CartesianController
from cube_state_provider import CubeStateProvider
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from kinematics import IKConvergenceError, solve_ik
from loop_runner import LoopRunner
from robot_constants import ROBOT_JOINT_NAMES
from robot_state_provider import SimRobotStateProvider
from schemas import CAMERA_IMAGE_SHAPES, OBSERVATION_FIELDS, CameraFrame
from schemas.config import ActionRolloutValidationConfig, CartesianActionConfig, ProjectConfig, StackSuccessConfig
from trained_policy import TrainedPolicy
from vision_errors import VisionDataUnavailableError


@dataclass(frozen=True)
class ActionRolloutMetrics:
    """Outcomes of completed attempts, including failed IK and timed-out runs."""

    episodes: int = 0
    successes: int = 0
    ik_failures: int = 0
    timeouts: int = 0

    def __post_init__(self) -> None:
        if min(self.episodes, self.successes, self.ik_failures, self.timeouts) < 0:
            raise ValueError("Rollout counts must be nonnegative.")
        if self.episodes != self.successes + self.ik_failures + self.timeouts:
            raise ValueError("Every rollout must have exactly one terminal outcome.")

    @property
    def success_rate_percent(self) -> float:
        return 100.0 * self.successes / self.episodes if self.episodes else 0.0


@dataclass(frozen=True)
class ActionRolloutFrame:
    """An episode's visible state after reset or completed control ticks."""

    episode_index: int
    step: int
    status: Literal["running", "success", "ik_failure", "timeout"]


class StackSuccessTracker:
    """Score a released orange-on-blue stack using privileged physics only.

    Neither these states nor contact information are supplied to the policy.
    Success requires conditions at every control tick throughout the stable
    simulation-time interval; intermediate physics steps are not sampled.
    """

    def __init__(self, model: mujoco.MjModel, config: StackSuccessConfig) -> None:
        self.config = config
        self.orange_geom = model.geom("orange_cube_geom").id
        self.blue_geom = model.geom("blue_cube_geom").id
        self.table_geom = model.geom("table").id
        self.expected_vertical_distance = float(
            model.geom_size[self.orange_geom, 2] + model.geom_size[self.blue_geom, 2]
        )
        gripper_body = model.body("gripper").id
        self.gripper_geoms = set()
        for geom_id, body_id in enumerate(model.geom_bodyid):
            while body_id:
                if body_id == gripper_body:
                    self.gripper_geoms.add(geom_id)
                    break
                body_id = model.body_parentid[body_id]
        self.stable_since: float | None = None

    def reset(self) -> None:
        self.stable_since = None

    def conditions_met(self, data: mujoco.MjData) -> bool:
        orange = data.body("orange_cube").xpos
        blue = data.body("blue_cube").xpos
        velocities = [data.joint(f"{name}_cube_joint").qvel for name in ("orange", "blue")]
        if not all(np.isfinite(value).all() for value in (orange, blue, *velocities)):
            return False
        tolerance = 1e-12
        if np.any(np.abs(orange[:2] - blue[:2]) > self.config.max_horizontal_center_offset + tolerance):
            return False
        distance = float(orange[2] - blue[2])
        if (distance <= 0 or abs(distance - self.expected_vertical_distance)
                > self.config.vertical_center_tolerance + tolerance):
            return False
        for velocity in velocities:
            if (np.linalg.norm(velocity[:3]) > self.config.max_linear_speed
                    or np.linalg.norm(velocity[3:]) > self.config.max_angular_speed):
                return False
        stacked = supported = False
        cubes = {self.orange_geom, self.blue_geom}
        for contact in data.contact:
            if contact.dist > 0:
                continue
            geoms = {int(contact.geom1), int(contact.geom2)}
            if geoms == cubes:
                stacked = True
            if geoms == {self.blue_geom, self.table_geom}:
                supported = True
            if geoms & cubes and geoms & self.gripper_geoms:
                return False
        return stacked and supported

    def update(self, data: mujoco.MjData) -> bool:
        now = float(data.time)
        if not self.conditions_met(data):
            self.stable_since = None
            return False
        if self.stable_since is None or now < self.stable_since:
            self.stable_since = now
        return now - self.stable_since + 1e-12 >= self.config.required_stable_time


def reset_rollout_environment(environment: CubeStackEnvironment,
                              config: ActionRolloutValidationConfig,
                              rng: np.random.Generator, *,
                              controller_config: CartesianActionConfig) -> None:
    """Initialize the demonstration-style open-claw home and ground cubes.

    Install the sampled pose directly using the controller's IK constraints;
    initialization consumes no physics steps or policy-history tokens.
    """
    environment.reset()
    half_range = np.asarray(config.start_position_half_range)
    home = rng.uniform(np.asarray(config.start_position) - half_range,
                       np.asarray(config.start_position) + half_range)
    initial = np.zeros(len(ROBOT_JOINT_NAMES))
    initial[-1] = controller_config.open_gripper_target
    joints, outcome = solve_ik(environment.model, initial, home, controller_config)
    if not outcome["converged"]:
        raise IKConvergenceError({"attempts": [{"target": home.tolist(), **outcome}],
                                  "stage": "reset", "episode_step": 0})
    for _ in range(10_000):
        xy = np.column_stack((rng.uniform(*config.cube_x_range, size=2),
                              rng.uniform(*config.cube_y_range, size=2)))
        if np.linalg.norm(xy[0] - xy[1]) >= config.minimum_cube_separation:
            break
    else:
        raise ValueError("Could not sample separated action-validation cubes within the configured ranges.")
    with environment.state_lock:
        for name, angle in zip(ROBOT_JOINT_NAMES, joints, strict=True):
            environment.data.joint(name).qpos[0] = angle
            environment.data.actuator(name).ctrl[0] = angle
        table = environment.model.geom("table").id
        table_top = float(environment.data.geom_xpos[table, 2] + environment.model.geom_size[table, 2])
        for name, position in zip(("orange", "blue"), xy, strict=True):
            joint = environment.data.joint(f"{name}_cube_joint")
            half_height = environment.model.geom(f"{name}_cube_geom").size[2]
            joint.qpos[:] = [*position, table_top + half_height, 1.0, 0.0, 0.0, 0.0]
            joint.qvel[:] = 0
        mujoco.mj_forward(environment.model, environment.data)


def evaluate_action_rollouts(
    *, project: ProjectConfig, action_policy: nn.Module, vision_model: nn.Module,
    device: torch.device, on_episode: Callable[[ActionRolloutMetrics], None] | None = None,
    on_frame: Callable[[CubeStackEnvironment, ActionRolloutFrame], None] | None = None,
    renderer_factory: Callable[..., Any] = mujoco.Renderer,
) -> ActionRolloutMetrics:
    """Evaluate deterministic actions with fresh camera inference each tick.

    The seed is restarted on every call to compare checkpoints on the same
    initial conditions. Physics advances one control interval per prediction;
    no worker, viewer, camera FPS limiter, or wall-clock sleep is involved.
    Only IK convergence failures are converted to failed episode outcomes.
    Renderer, perception, and programming errors propagate to the caller.

    ``on_frame`` observes the environment after each reset and completed tick,
    including the terminal state. Episode indices start at zero; ``step`` counts
    completed control ticks (zero after reset). An IK failure is reported at
    the number of ticks completed before it. The observer must not advance
    physics or modify rollout state. Observer errors propagate with cleanup.
    """
    config = project.training.action.rollout_validation
    rng = np.random.default_rng(config.seed)
    max_steps = floor(project.environment.episode_seconds * project.loop.control_hz + 1e-9)
    if max_steps < 1:
        raise ValueError("Action validation episode_seconds must allow at least one control tick.")
    # Preserve mixed submodule modes as well as the top-level training flags.
    modes = [(module, module.training) for model in (action_policy, vision_model)
             for module in model.modules()]
    try:
        action_policy.eval()
        vision_model.eval()
        environment = CubeStackEnvironment(scene_path=project.environment.scene_path)
        simulation_clock = lambda: float(environment.data.time)
        buffers = {
            name: CameraFrameBuffer(
                name, maximum_frame_age_seconds=project.cameras.maximum_frame_age_seconds,
                clock=simulation_clock,
            ) for name in SIM_CAMERA_NAMES
        }
        runner = LoopRunner(
            robot_state_provider=SimRobotStateProvider(environment),
            camera_feed_getter=SimCameraFeedGetter(environment, buffers),
            cube_state_provider=CubeStateProvider(
                vision_model, device,
                maximum_frame_age_seconds=project.cameras.maximum_frame_age_seconds,
            ),
            trained_policy=TrainedPolicy(action_policy, OBSERVATION_FIELDS,
                                         project.action_policy.history_length),
            controller=CartesianController(environment.model, project.cartesian_controller),
            backend=SimJointCommandBackend(environment), config=project.loop,
            deterministic=True, clock=simulation_clock, wait=lambda _: None,
        )
        scorer = StackSuccessTracker(environment.model, config.success)
        metrics = ActionRolloutMetrics()
        with ExitStack() as resources, torch.inference_mode():
            renderers = {}
            for name in SIM_CAMERA_NAMES:
                height, width, _ = CAMERA_IMAGE_SHAPES[name]
                renderer = renderer_factory(environment.model, height=height, width=width)
                resources.callback(renderer.close)
                renderers[name] = renderer
            for episode_index in range(config.episodes):
                succeeded = failed_ik = False
                try:
                    reset_rollout_environment(environment, config, rng,
                                              controller_config=project.cartesian_controller)
                except IKConvergenceError:
                    failed_ik = True
                    if on_frame is not None:
                        on_frame(environment, ActionRolloutFrame(episode_index, 0, "ik_failure"))
                else:
                    for buffer in buffers.values():
                        buffer.reset()
                    runner.reset()
                    scorer.reset()
                    if on_frame is not None:
                        on_frame(environment, ActionRolloutFrame(episode_index, 0, "running"))
                    for step in range(max_steps):
                        with environment.state_lock:
                            for name, camera_name in SIM_CAMERA_NAMES.items():
                                renderer = renderers[name]
                                renderer.update_scene(environment.data, camera=camera_name)
                                buffers[name].record_read(True, CameraFrame(
                                    camera_id=name, timestamp=simulation_clock(), rgb=renderer.render(),
                                ))
                        try:
                            result = runner.step()
                        except IKConvergenceError:
                            failed_ik = True
                            if on_frame is not None:
                                on_frame(environment, ActionRolloutFrame(episode_index, step, "ik_failure"))
                            break
                        if result is None:
                            raise VisionDataUnavailableError("Synchronous rollout camera inference produced no observation.")
                        succeeded = scorer.update(environment.data)
                        if on_frame is not None:
                            status = "success" if succeeded else "timeout" if step + 1 == max_steps else "running"
                            on_frame(environment, ActionRolloutFrame(episode_index, step + 1, status))
                        if succeeded:
                            break
                metrics = ActionRolloutMetrics(
                    episodes=metrics.episodes + 1,
                    successes=metrics.successes + int(succeeded),
                    ik_failures=metrics.ik_failures + int(failed_ik),
                    timeouts=metrics.timeouts + int(not succeeded and not failed_ik),
                )
                if on_episode is not None:
                    on_episode(metrics)
        return metrics
    finally:
        for module, training in modes:
            module.training = training
