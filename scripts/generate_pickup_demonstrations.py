#!/usr/bin/env python3
"""Generate complete orange-on-blue stacking demonstrations on CPU.

Run with the repository's Python environment::

    .venv/bin/python scripts/generate_pickup_demonstrations.py
    .venv/bin/python scripts/generate_pickup_demonstrations.py --examples 10
    .venv/bin/python scripts/generate_pickup_demonstrations.py --test

Each invocation ADDS examples; it never replaces existing episodes. An example
is one complete successful episode, not one timestep. The default split is
80% training and 20% test episodes. Failed attempts are discarded.
Required IK constraint failures instead stop generation immediately, print
[IK_CONVERGENCE_FAILURE], and save diagnostics/ik-failure-{uuid}.json. Completed
episodes remain saved. CartesianActionConfig.require_downward controls the
shared requirement; its angular tolerance is configured in kinematics.py.
Recording starts with an open claw uniformly sampled around the home position:
each X/Y coordinate varies by +/-4 cm; Z stays fixed.
Episodes include approach, grasp, lift, transport,
placement, release, and any retreat needed to leave a stable, unheld stack.
While orange is held, each action has a 0.002 chance of a temporary actuator
disturbance: hold the measured arm pose for 0.25 s, then force the claw open
for 0.25 s while still holding the arm. The teacher responds to observed drops
by retrying pickup without resetting physics or history. Saved actions are
always the teacher's INTENDED commands, never the actuator overrides; actual
states are recorded at every control interval. No special loss mask is needed.
After reaching a pickup waypoint, an unheld cube triggers another attempt if
more than 5 simulated seconds have elapsed. Deliberate placement is exempt.

For each episode UUID, save three CPU float32 tensors under data/{train,test}/:
    observations/{uuid}.pt       [T + 1, 49] (no waypoint-reached flag)
    accepted_targets/{uuid}.pt   [T + 1, 3]
    actions/{uuid}.pt            [T, 4]  (normalized dx, dy, dz, gripper)

Observation[t] and accepted_targets[t] describe the state BEFORE actions[t].
The accepted target comes from reset/step info, after workspace limits and IK.
To construct a history token, concatenate observation[t], actions[t-1], and
accepted_targets[t], giving 56 values. At t=0 use a zero previous action and
episode_start=1; this placeholder is NOT an action label. The final observation
has no label.
The existing history wrapper can reconstruct its context windows from these
episodes; windows and padding are deliberately not duplicated on disk.

Each attempt's UUID is also its reset seed: UUID.int converts all 128 bits to
the integer expected by Gym/NumPy. No counter, truncation, or hash is involved.
Successful episodes have a manifests/{uuid}.json entry recording settings.
Each writer publishes its own entry, so concurrent generators need no lock or
shared index rewrite. read_manifest() also reads the older manifest.json,
which is left unchanged. If a process is forcibly killed, unlisted tensors
are not committed examples; they are left untouched. All tests live here and
clean their temporary data from a directory inside this repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch


# Generation hyperparameters. Change these defaults here, or use --examples.
NUMBER_OF_EXAMPLES = 15000
TEST_FRACTION = 0.20
MAXIMUM_ATTEMPTS_PER_EXAMPLE = 10
MAXIMUM_EPISODE_STEPS = 900  # 45 seconds at 20 actions per second.
START_POSITION_HALF_RANGE = (0.04, 0.04, 0.0)  # XYZ metres; randomize X/Y and keep Z fixed.

# Per eligible action, not per episode. Active disturbances cannot retrigger;
# after recovery another disturbance is possible within the episode budget.
DROP_DISTURBANCE_PROBABILITY = 0.002
DROP_STOP_SECONDS = 0.25
DROP_RELEASE_SECONDS = 0.25
LOST_GRASP_STEPS = 2
RECOVERY_PAUSE_STEPS = 5  # 0.25 seconds at 20 Hz, before checking cube settling.
RECOVERY_SETTLE_STEPS = 5
RECOVERY_LINEAR_SPEED = 0.01
RECOVERY_ANGULAR_SPEED = 0.10

# Retry from a freshly calculated orange waypoint when the cube is unheld
# more than this many seconds after reaching the previous pickup waypoint.
# This prevents a missed grasp from spending the whole attempt pushing the cube.
WAYPOINT_GRASP_RETRY_SECONDS = 5.0

# Scripted teacher. Distances are metres; durations are policy control steps.
GRASP_RADIAL_OFFSET = 0.007  # Jaw-centre offset along the claw's closing axis.
CLAW_YAW_DEGREES = 0.0  # World-facing jaw heading; 0 means +X, regardless of cube Y.
CLAW_YAW_TOLERANCE_DEGREES = 1.0  # Allowed jaw-plane error, separate from downward tilt.
GRASP_HEIGHT_OFFSET = 0.005  # metres above the initial cube centre
POSITION_TOLERANCE = 0.0015
CONSECUTIVE_NEAR_STEPS = 3
MAXIMUM_MOVE_STEPS = 200
OPEN_SETTLE_STEPS = 10
CLOSE_GRIPPER_STEPS = 20
LIFT_SETTLE_STEPS = 10
RELEASE_STEPS = 10
RETREAT_DISTANCE = 0.05
EXTRA_HOLD_STEPS = 20
OPEN_GRIPPER = 1.0
CLOSED_GRIPPER = -1.0

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from environment import (  # noqa: E402
    DEFAULT_SCENE_PATH, PHYSICS_STEPS_PER_ACTION,
)
from cartesian_actions import CartesianActionConfig, IKConvergenceError  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402
from observations import PRIVILEGED_OBSERVATION_SIZE  # noqa: E402
from success import StackSuccessConfig  # noqa: E402


class DemonstrationFailure(RuntimeError):
    """An unsuccessful physical attempt, eligible for a retry with a new seed."""


class DropDisturbance:
    """Override actuator targets without changing the intended action stream.

    The ordinary Gym step still runs exactly once per recorded timestamp. A
    temporary hook intercepts only its final joint command. Measurements and
    accepted Cartesian targets describe the resulting physical controller,
    rather than accumulating unexecuted movement during the override.
    """

    def __init__(self, seed: int, probability: float = DROP_DISTURBANCE_PROBABILITY,
                 stop_seconds: float = DROP_STOP_SECONDS, release_seconds: float = DROP_RELEASE_SECONDS):
        self.probability = probability
        self.stop_seconds = stop_seconds
        self.release_seconds = release_seconds
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Disturbance probability must be between zero and one.")
        if any(not math.isfinite(value) or value <= 0 for value in (self.stop_seconds, self.release_seconds)):
            raise ValueError("Disturbance durations must be positive and finite.")
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.remaining_steps = 0
        self.release_steps = 0
        self.events: list[dict[str, Any]] = []
        self._joint_targets: np.ndarray | None = None
        self._cartesian_target: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.remaining_steps > 0

    def step(self, environment: CubeStackGymEnvironment, intended_action: np.ndarray):
        state = environment.previous_state
        simulation = environment.simulation
        if (not self.active and state["orange_currently_held"] and self.probability > 0
                and self.rng.random() < self.probability):
            interval = PHYSICS_STEPS_PER_ACTION * simulation.model.opt.timestep
            stop_steps = max(1, math.ceil(self.stop_seconds / interval - 1e-9))
            self.release_steps = max(1, math.ceil(self.release_seconds / interval - 1e-9))
            self.remaining_steps = stop_steps + self.release_steps
            # Freeze measured arm positions, not targets the arm is still
            # moving toward. Preserve squeezing force until the release phase.
            self._joint_targets = np.array(state["joint_positions"], dtype=float, copy=True)
            self._joint_targets[-1] = state["gripper_target"]
            self._cartesian_target = np.array(state["gripper_position"], dtype=float, copy=True)
            self.events.append({"start_step": self.steps, "stop_steps": stop_steps,
                                "release_steps": self.release_steps})
        if not self.active:
            transition = environment.step(intended_action)
        else:
            targets = self._joint_targets.copy()
            if self.remaining_steps <= self.release_steps:
                targets[-1] = environment.action_adapter.config.open_gripper_target
            original_step = simulation.step_joint_targets
            previous_hook = simulation.__dict__.get("step_joint_targets")
            override_applied = False

            def override_targets(_intended_joint_targets):
                nonlocal override_applied
                override_applied = True
                return original_step(targets.copy())

            simulation.step_joint_targets = override_targets
            try:
                transition = environment.step(intended_action)
            finally:
                if previous_hook is None:
                    del simulation.step_joint_targets
                else:
                    simulation.step_joint_targets = previous_hook
            if not override_applied:
                # Best-effort IK can advance physics without accepting a new
                # joint command. Such an attempt is already invalid for this
                # teacher; do not pretend that its disturbance was executed.
                raise DemonstrationFailure("Disturbance: IK did not accept an actuator command.")
            # The overridden arm command accepted the held pose, not the
            # teacher's requested displacement. Prevent hidden target windup.
            environment.action_adapter.reset({"gripper_position": self._cartesian_target})
            transition[-1]["target_gripper_position"] = self._cartesian_target.copy()
            self.remaining_steps -= 1
        self.steps += 1
        return transition


# Episode recording and tensor validation.


@dataclass
class PickupEpisode:
    observations: torch.Tensor
    accepted_targets: torch.Tensor
    actions: torch.Tensor
    final_hold_time: float
    final_stack_stable_time: float
    stage_steps: dict[str, int]
    disturbance_events: list[dict[str, Any]] = field(default_factory=list)
    recovery_count: int = 0
    waypoint_retry_count: int = 0

    def validate(self) -> None:
        """Reject malformed data before any files are written."""
        steps = len(self.actions)
        if not 1 <= steps <= MAXIMUM_EPISODE_STEPS:
            raise ValueError("An episode must contain 1..MAXIMUM_EPISODE_STEPS actions.")
        shapes = {
            "observations": (steps + 1, PRIVILEGED_OBSERVATION_SIZE),
            "accepted_targets": (steps + 1, 3),
            "actions": (steps, 4),
        }
        for name, shape in shapes.items():
            tensor = getattr(self, name)
            if (tensor.shape != shape or tensor.dtype != torch.float32
                    or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be a finite CPU float32 tensor of shape {shape}.")
        if torch.any(self.actions.abs() > 1.0):
            raise ValueError("Action labels must be normalized to [-1, 1].")
        if not math.isfinite(self.final_hold_time) or self.final_hold_time < 0:
            raise ValueError("Hold time must be finite and nonnegative.")
        if (not math.isfinite(self.final_stack_stable_time)
                or self.final_stack_stable_time < StackSuccessConfig().required_stable_time):
            raise ValueError("The episode must end with a released, stable stack.")
        if (not self.stage_steps or sum(self.stage_steps.values()) != steps
                or any(type(value) is not int or value < 1 for value in self.stage_steps.values())):
            raise ValueError("Stage counts must account for every recorded action.")


def copy_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.array(value, dtype=np.float32, copy=True)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return vector


class EpisodeRecorder:
    """Own independent snapshots and keep one more state than action."""

    def __init__(self, observation: np.ndarray, info: dict[str, Any],
                 disturbance: DropDisturbance | None = None) -> None:
        self.observations = [copy_vector(observation, PRIVILEGED_OBSERVATION_SIZE, "observation")]
        self.targets = [copy_vector(info["target_gripper_position"], 3, "accepted target")]
        self.actions: list[np.ndarray] = []
        self.stages: Counter[str] = Counter()
        self.finished = False
        self.verified_success = False
        self.final_hold_time = 0.0
        self.final_stack_stable_time = 0.0
        self.disturbance = disturbance
        self.recovery_requested = False
        self.recovery_count = 0
        self.waypoint_retry_count = 0
        self.waypoint_reached_time: float | None = None
        self._grasp_seen = False
        self._lost_grasp_steps = 0

    def mark_waypoint_reached(self, time: float) -> None:
        """Arm a fresh pickup deadline using simulation time, not wall time."""
        self.waypoint_reached_time = time

    def begin_recovery(self) -> None:
        """Restart only teacher progress; keep every recorded timestep."""
        self.recovery_count += 1
        self.recovery_requested = False
        self.waypoint_reached_time = None
        self._grasp_seen = False
        self._lost_grasp_steps = 0

    def step(self, environment: CubeStackGymEnvironment, action: np.ndarray, stage: str) -> bool:
        if self.finished:
            raise RuntimeError("Cannot record another action after termination.")
        issued_action = copy_vector(action, 4, "action")
        if np.any(np.abs(issued_action) > 1.0):
            raise ValueError("The teacher must issue normalized actions in [-1, 1].")
        try:
            if self.disturbance is None:
                transition = environment.step(issued_action.copy())
            else:
                transition = self.disturbance.step(environment, issued_action.copy())
            observation, _, terminated, truncated, info = transition
        except IKConvergenceError as error:
            error.diagnostics.update(stage=stage, episode_step=len(self.actions) + 1)
            raise
        self.actions.append(issued_action)
        self.observations.append(copy_vector(observation, PRIVILEGED_OBSERVATION_SIZE, "observation"))
        self.targets.append(copy_vector(info["target_gripper_position"], 3, "accepted target"))
        self.stages[stage] += 1
        self.final_hold_time = float(info["orange_grasp_hold_time"])
        self.final_stack_stable_time = float(info["stack_stable_time"])
        self.finished = bool(terminated or truncated)

        if stage in ("release", "retreat", "settle_stack"):
            # Pickup is complete: intentional placement must not start a retry.
            self.waypoint_reached_time = None
        if info["orange_currently_held"]:
            self._grasp_seen = True
            self._lost_grasp_steps = 0
        elif stage in ("lift", "settle_lift", "transport", "lower") and self._grasp_seen:
            self._lost_grasp_steps += 1
            if self._lost_grasp_steps >= LOST_GRASP_STEPS:
                self.recovery_requested = True
        elif stage in ("release", "retreat", "settle_stack"):
            # A deliberate placement release is not a dropped grasp.
            self._grasp_seen = False
            self._lost_grasp_steps = 0

        if info["is_failure"] or not info["ik_position_converged"]:
            raise DemonstrationFailure(f"{stage}: environment failure or position IK failure.")
        if not info["ik_tool_yaw_converged"]:
            raise DemonstrationFailure(f"{stage}: claw alignment IK failure.")
        if truncated:
            raise DemonstrationFailure(f"{stage}: episode timed out.")
        if terminated:
            if (not info["is_success"] or info["orange_currently_held"]
                    or not math.isfinite(self.final_stack_stable_time)
                    or self.final_stack_stable_time < StackSuccessConfig().required_stable_time):
                raise DemonstrationFailure(f"{stage}: terminated without a released, stable stack.")
            self.verified_success = True
            return True
        if info["is_success"]:
            raise DemonstrationFailure("Success was reported without episode termination.")
        if (self.waypoint_reached_time is not None and not self.recovery_requested
                and not info["orange_currently_held"]
                and environment.previous_state["time"] - self.waypoint_reached_time
                > WAYPOINT_GRASP_RETRY_SECONDS + 1e-9):
            # Count once and yield to recovery, keeping this action and state in
            # history. The epsilon avoids firing at exactly the deadline due to
            # accumulated floating-point physics timesteps.
            self.waypoint_retry_count += 1
            self.recovery_requested = True
        return False

    def episode(self) -> PickupEpisode:
        if not self.verified_success:
            raise DemonstrationFailure("The episode has not ended in a verified successful stack.")
        episode = PickupEpisode(
            observations=torch.from_numpy(np.stack(self.observations)),
            accepted_targets=torch.from_numpy(np.stack(self.targets)),
            actions=torch.from_numpy(np.stack(self.actions)),
            final_hold_time=self.final_hold_time,
            final_stack_stable_time=self.final_stack_stable_time,
            stage_steps=dict(self.stages),
            disturbance_events=[] if self.disturbance is None else list(self.disturbance.events),
            recovery_count=self.recovery_count,
            waypoint_retry_count=self.waypoint_retry_count,
        )
        episode.validate()
        return episode


# Scripted stacking teacher. Every recorded action goes through the Gym adapter.


def action_toward(environment: CubeStackGymEnvironment, target: np.ndarray, gripper: float) -> np.ndarray:
    """Move relative to the last accepted target, not measured gripper XYZ."""
    adapter = environment.action_adapter
    delta = (target - adapter.current_target_gripper_position) / adapter.config.maximum_position_delta
    return np.array([*np.clip(delta, -1.0, 1.0), gripper], dtype=np.float32)


def move_to(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder,
            target: np.ndarray, gripper: float, stage: str) -> bool:
    near_steps = 0
    for _ in range(MAXIMUM_MOVE_STEPS):
        if recorder.step(environment, action_toward(environment, target, gripper), stage):
            return True
        if getattr(recorder, "recovery_requested", False):
            return False
        measured_position = environment.previous_state["gripper_position"]
        near_steps = near_steps + 1 if np.linalg.norm(target - measured_position) <= POSITION_TOLERANCE else 0
        if near_steps >= CONSECUTIVE_NEAR_STEPS:
            return False
    raise DemonstrationFailure(f"{stage}: gripper did not reach the requested pose.")


def hold_position(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder,
                  gripper: float, steps: int, stage: str) -> bool:
    for _ in range(steps):
        if recorder.step(environment, np.array([0.0, 0.0, 0.0, gripper]), stage):
            return True
        if getattr(recorder, "recovery_requested", False):
            return False
    return False


def move_held_cube_to(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder,
                      cube_target: np.ndarray, stage: str) -> bool:
    """Position orange, accounting for its measured offset from the gripper.

    The world-space offset changes as the arm rotates during transport. Update
    it every step instead of assuming that gripper XYZ equals cube XYZ.
    """
    near_steps = 0
    for _ in range(MAXIMUM_MOVE_STEPS):
        state = environment.previous_state
        if not state["orange_currently_held"]:
            near_steps = 0
            if not getattr(recorder, "_grasp_seen", False):
                recorder.recovery_requested = True
                return False
            # A single missing contact may be transient. Hold position and
            # keep squeezing while the recorder confirms persistent loss.
            if hold_position(environment, recorder, CLOSED_GRIPPER, 1, stage):
                return True
            if recorder.recovery_requested:
                return False
            continue
        held_offset = state["orange_position"] - state["gripper_position"]
        target = cube_target - held_offset
        if recorder.step(environment, action_toward(environment, target, CLOSED_GRIPPER), stage):
            return True
        if getattr(recorder, "recovery_requested", False):
            return False
        error = np.linalg.norm(cube_target - environment.previous_state["orange_position"])
        near_steps = near_steps + 1 if error <= POSITION_TOLERANCE else 0
        if near_steps >= CONSECUTIVE_NEAR_STEPS:
            return False
    raise DemonstrationFailure(f"{stage}: orange did not reach the requested pose.")


def make_environment() -> CubeStackGymEnvironment:
    return CubeStackGymEnvironment(
        scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH,
        maximum_episode_steps=MAXIMUM_EPISODE_STEPS,
        start_at_orange_waypoint=False,
        recovery_start_probability=0.0,
        start_position_half_range=START_POSITION_HALF_RANGE,
        action_config=CartesianActionConfig(
            target_tool_yaw=math.radians(CLAW_YAW_DEGREES),
            tool_yaw_tolerance=math.radians(CLAW_YAW_TOLERANCE_DEGREES),
        ),
    )


def pickup_from_current_pose(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder) -> bool:
    """Approach and lift using the current cube pose, including after a drop."""
    orange = np.asarray(environment.previous_state["orange_position"], dtype=float).copy()
    waypoint = orange + np.array([0.0, 0.0, environment.reward_config.approach_orange_height_offset])

    # True always means verified environment success. After a disturbance,
    # orange can also happen to settle into a stack during a recovery stage.
    if move_to(environment, recorder, waypoint, OPEN_GRIPPER, "approach"):
        return True
    if recorder.recovery_requested:
        return False
    recorder.mark_waypoint_reached(float(environment.previous_state["time"]))
    if hold_position(environment, recorder, OPEN_GRIPPER, OPEN_SETTLE_STEPS, "settle_waypoint"):
        return True
    if recorder.recovery_requested:
        return False

    # Read the settled cube pose and align the jaw gap using its calibrated
    # closing-axis/height offset, as in the existing pickup teacher.
    orange = np.asarray(environment.previous_state["orange_position"], dtype=float).copy()
    grasp_target = orange.copy()
    # The wrist now cancels shoulder rotation. Apply the calibrated offset in
    # the claw's fixed heading, rather than rotating it toward the arm base.
    yaw = environment.action_adapter.config.target_tool_yaw
    if yaw is None:
        direction = orange[:2] / np.linalg.norm(orange[:2])
    else:
        direction = np.array([math.cos(yaw), math.sin(yaw)])
    grasp_target[:2] -= GRASP_RADIAL_OFFSET * direction
    grasp_target[2] += GRASP_HEIGHT_OFFSET

    if move_to(environment, recorder, grasp_target, OPEN_GRIPPER, "descend"):
        return True
    if recorder.recovery_requested:
        return False
    if hold_position(environment, recorder, OPEN_GRIPPER, OPEN_SETTLE_STEPS, "settle"):
        return True
    if recorder.recovery_requested:
        return False
    if hold_position(environment, recorder, CLOSED_GRIPPER, CLOSE_GRIPPER_STEPS, "close"):
        return True
    if recorder.recovery_requested:
        return False

    if move_to(environment, recorder, waypoint, CLOSED_GRIPPER, "lift"):
        return True
    if recorder.recovery_requested:
        return False
    if hold_position(environment, recorder, CLOSED_GRIPPER, LIFT_SETTLE_STEPS, "settle_lift"):
        return True
    if not environment.previous_state["orange_currently_held"]:
        recorder.recovery_requested = True
    return False


def stack_held_cube(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder) -> bool:
    """Finish a carry/placement attempt, yielding control on an observed drop."""

    state = environment.previous_state
    transport_cube_target = state["orange_position"].copy()
    transport_cube_target[:2] = state["blue_position"][:2]
    stack_distance = environment.simulation.success_config.expected_vertical_center_distance
    if transport_cube_target[2] <= state["blue_position"][2] + stack_distance:
        raise DemonstrationFailure("The lifted orange cube does not clear blue.")
    if move_held_cube_to(environment, recorder, transport_cube_target, "transport"):
        return True
    if recorder.recovery_requested:
        return False

    placement_cube_target = environment.previous_state["blue_position"].copy()
    placement_cube_target[2] += stack_distance
    if move_held_cube_to(environment, recorder, placement_cube_target, "lower"):
        return True
    if recorder.recovery_requested:
        return False

    succeeded = hold_position(environment, recorder, OPEN_GRIPPER, RELEASE_STEPS, "release")
    if not succeeded:
        # Clear the jaws so the stack detector can verify an unsupported-by-
        # gripper stack. Stop immediately if success occurs during retreat.
        retreat_target = environment.previous_state["gripper_position"].copy()
        retreat_target[2] += RETREAT_DISTANCE
        succeeded = move_to(environment, recorder, retreat_target, OPEN_GRIPPER, "retreat")
    if not succeeded:
        action_interval = PHYSICS_STEPS_PER_ACTION * environment.simulation.model.opt.timestep
        settle_steps = math.ceil(environment.simulation.success_config.required_stable_time / action_interval)
        succeeded = hold_position(environment, recorder, OPEN_GRIPPER,
                                  settle_steps + EXTRA_HOLD_STEPS, "settle_stack")
    return succeeded


def recover_for_pickup(environment: CubeStackGymEnvironment, recorder: EpisodeRecorder) -> bool:
    """Pause in place, wait for the landed cube, then allow a new approach.

    Recovery follows measured grasp loss or the pickup deadline, never the
    disturbance's private timer. A completed stack still ends the episode.
    """
    recorder.begin_recovery()
    # Zero XYZ deltas hold the accepted arm target; do not command a clearance
    # move. Keep recording physics, and only count settled steps after this pause.
    if hold_position(environment, recorder, OPEN_GRIPPER, RECOVERY_PAUSE_STEPS, "recovery_pause"):
        return True
    settled = 0
    for _ in range(MAXIMUM_MOVE_STEPS):
        if hold_position(environment, recorder, OPEN_GRIPPER, 1, "recovery_settle"):
            return True
        state = environment.previous_state
        velocity = state["orange_velocity"]
        quiet = (not state["orange_currently_held"]
                 and np.linalg.norm(velocity[:3]) <= RECOVERY_LINEAR_SPEED
                 and np.linalg.norm(velocity[3:]) <= RECOVERY_ANGULAR_SPEED)
        settled = settled + 1 if quiet else 0
        if settled >= RECOVERY_SETTLE_STEPS:
            return False
    raise DemonstrationFailure("Recovery: orange did not settle for another pickup.")


def collect_episode(environment: CubeStackGymEnvironment, seed: int) -> PickupEpisode:
    """Record a full stack, retaining intended actions through drops/retries."""
    try:
        observation, info = environment.reset(seed=seed)
    except IKConvergenceError as error:
        error.diagnostics.update(stage="reset", episode_step=0)
        raise
    except RuntimeError as error:
        raise DemonstrationFailure(f"Environment reset failed: {error}") from error
    if info["episode_start_type"] != "home" or environment.episode_step_count != 0:
        raise ValueError("The collector requires a fresh home start.")
    recorder = EpisodeRecorder(observation, info, disturbance=DropDisturbance(seed))
    # Every action advances the same episode counter. The existing Gym horizon
    # bounds repeated recoveries; no resets, teleports, or discarded history.
    while not recorder.finished:
        if pickup_from_current_pose(environment, recorder):
            return recorder.episode()
        if not recorder.recovery_requested and stack_held_cube(environment, recorder):
            return recorder.episode()
        if recover_for_pickup(environment, recorder):
            return recorder.episode()
    raise DemonstrationFailure("Episode ended without a stable stack.")


# Additive storage and collection. Manifest entries list complete episodes.


def read_manifest(output_directory: Path) -> list[dict[str, Any]]:
    """Combine legacy and per-episode entries without modifying either format."""
    records = []
    path = output_directory / "manifest.json"
    if path.exists():
        content = json.loads(path.read_text())
        if content["format_version"] != 1 or not isinstance(content["episodes"], list):
            raise ValueError("Unsupported demonstration manifest format.")
        records.extend(content["episodes"])
    for path in sorted((output_directory / "manifests").glob("*.json")):
        content = json.loads(path.read_text())
        if content["format_version"] != 2:
            raise ValueError("Unsupported demonstration manifest entry format.")
        record = content["episode"]
        if record["uuid"] != path.stem or record["seed"] != UUID(record["uuid"]).int:
            raise ValueError("Manifest filename, episode UUID, and reset seed must agree.")
        records.append(record)
    seeds, identifiers = set(), set()
    for record in records:
        if record["split"] not in ("train", "test") or type(record["seed"]) is not int or record["seed"] < 0:
            raise ValueError("Invalid split or reset seed in the existing manifest.")
        if record["seed"] in seeds or record["uuid"] in identifiers:
            raise ValueError("Duplicate seed or UUID in the existing manifest.")
        seeds.add(record["seed"])
        identifiers.add(record["uuid"])
        for name in ("observations", "accepted_targets", "actions"):
            if not (output_directory / record["split"] / name / f"{record['uuid']}.pt").is_file():
                raise ValueError(f"Missing {name} tensor for episode {record['uuid']}.")
    return records


def save_episode(output_directory: Path, episode: PickupEpisode,
                 episode_id: UUID, split: str,
                 settings: dict[str, Any]) -> dict[str, Any]:
    """Publish all three tensors, then their independent manifest entry.

    There is no shared index to overwrite. Exclusive creation protects existing
    files even on a UUID collision; a write failure rolls back only our files.
    """
    episode.validate()
    if split not in ("train", "test") or not isinstance(episode_id, UUID):
        raise ValueError("Expected train/test split and the UUID used for this episode's seed.")
    identifier = str(episode_id)
    # Include legacy episodes and incomplete triplets when checking collisions.
    for existing_split in ("train", "test"):
        for name in ("observations", "accepted_targets", "actions"):
            if (output_directory / existing_split / name / f"{identifier}.pt").exists():
                raise FileExistsError(f"Episode UUID already has data: {identifier}")
    record = {
        "uuid": identifier, "split": split, "seed": episode_id.int,
        "steps": len(episode.actions), "success": True,
        "final_hold_time": episode.final_hold_time,
        "final_stack_stable_time": episode.final_stack_stable_time,
        "stage_steps": episode.stage_steps, "settings": settings,
        "disturbance_events": episode.disturbance_events,
        "recovery_count": episode.recovery_count,
        "waypoint_retry_count": episode.waypoint_retry_count,
    }
    created_paths: list[Path] = []
    manifest_directory = output_directory / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / f"{identifier}.json"
    temporary_manifest = manifest_directory / f".{identifier}.tmp"
    created_temporary_manifest = False
    published = False
    try:
        for name in ("observations", "accepted_targets", "actions"):
            path = output_directory / split / name / f"{identifier}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                created_paths.append(path)
                torch.save(getattr(episode, name), stream)
        with temporary_manifest.open("x") as stream:
            created_temporary_manifest = True
            json.dump({"format_version": 2, "episode": record}, stream, indent=2)
            stream.write("\n")
        # Publish a complete JSON file atomically, without replacing an existing
        # entry even in the unlikely event of a UUID collision across splits.
        os.link(temporary_manifest, manifest_path)
        published = True
    except BaseException:
        if not published:
            for path in created_paths:
                path.unlink(missing_ok=True)
        if created_temporary_manifest:
            temporary_manifest.unlink(missing_ok=True)
        raise
    # Publication is the commit boundary. A later cleanup failure must never
    # remove tensors that a concurrent reader can already find in the manifest.
    temporary_manifest.unlink()
    return record


def generation_settings(environment: CubeStackGymEnvironment) -> dict[str, Any]:
    """Keep enough context to interpret/reproduce the saved normalized actions."""
    return {
        "task": "stack_orange_on_blue",
        "scene": str(DEFAULT_SCENE_PATH),
        "start_at_orange_waypoint": False, "recovery_start_probability": 0.0,
        "start_position": environment.simulation.start_position.tolist(),
        "start_position_half_range": environment.simulation.start_position_half_range.tolist(),
        "maximum_episode_steps": MAXIMUM_EPISODE_STEPS,
        "success_config": asdict(environment.simulation.success_config),
        "action_config": asdict(environment.action_adapter.config),
        "spawn_config": asdict(environment.simulation.spawn_config),
        "waypoint_height": environment.reward_config.approach_orange_height_offset,
        "action_interval": PHYSICS_STEPS_PER_ACTION * environment.simulation.model.opt.timestep,
        "teacher": {name: globals()[name] for name in (
            "GRASP_RADIAL_OFFSET", "GRASP_HEIGHT_OFFSET", "CLAW_YAW_DEGREES", "RETREAT_DISTANCE",
            "CLAW_YAW_TOLERANCE_DEGREES",
            "POSITION_TOLERANCE", "CONSECUTIVE_NEAR_STEPS", "MAXIMUM_MOVE_STEPS",
            "OPEN_SETTLE_STEPS", "CLOSE_GRIPPER_STEPS", "LIFT_SETTLE_STEPS",
            "RELEASE_STEPS", "EXTRA_HOLD_STEPS",
            "MAXIMUM_EPISODE_STEPS",
            "DROP_DISTURBANCE_PROBABILITY", "DROP_STOP_SECONDS", "DROP_RELEASE_SECONDS",
            "LOST_GRASP_STEPS", "RECOVERY_PAUSE_STEPS", "RECOVERY_SETTLE_STEPS",
            "RECOVERY_LINEAR_SPEED", "RECOVERY_ANGULAR_SPEED",
            "WAYPOINT_GRASP_RETRY_SECONDS",
        )},
    }


def generate_dataset(number_of_examples: int = NUMBER_OF_EXAMPLES,
                     output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> list[dict[str, Any]]:
    """Add the requested number of successes with bounded retries per example."""
    if type(number_of_examples) is not int or number_of_examples < 1:
        raise ValueError("number_of_examples must be a positive integer.")
    if not math.isfinite(TEST_FRACTION) or not 0 <= TEST_FRACTION <= 1:
        raise ValueError("TEST_FRACTION must be between 0 and 1.")
    if MAXIMUM_ATTEMPTS_PER_EXAMPLE < 1:
        raise ValueError("Use a positive attempt limit.")
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    test_count = int(number_of_examples * TEST_FRACTION + 0.5)
    splits = ["train"] * (number_of_examples - test_count) + ["test"] * test_count
    np.random.default_rng().shuffle(splits)
    environment = None
    settings = {}
    attempts, failures = 0, 0
    try:
        for index, split in enumerate(splits, start=1):
            for _ in range(MAXIMUM_ATTEMPTS_PER_EXAMPLE):
                episode_id = uuid4()
                seed = episode_id.int
                attempts += 1
                try:
                    # Constructor IK failures need the same fatal diagnostic
                    # and episode identifier as failures during reset/actions.
                    if environment is None:
                        environment = make_environment()
                        settings = generation_settings(environment)
                    episode = collect_episode(environment, seed)
                except IKConvergenceError as error:
                    # Fatal configuration/solver diagnostic: do not discard
                    # this scene and silently retry another random layout.
                    diagnostic_path = output_directory / "diagnostics" / f"ik-failure-{episode_id}.json"
                    record = {
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "uuid": str(episode_id),
                        "seed": seed,
                        "attempts": attempts,
                        "completed_episodes": len(records),
                        "discarded_attempts": failures,
                        "ik_convergence_failures": 1,
                        "settings": settings,
                        "diagnostics": error.diagnostics,
                    }
                    print(
                        f"[IK_CONVERGENCE_FAILURE] episode={episode_id} seed={seed} "
                        f"stage={error.diagnostics.get('stage', 'unknown')} "
                        f"step={error.diagnostics.get('episode_step', 'unknown')}: {error}",
                        file=sys.stderr, flush=True,
                    )
                    try:
                        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                        with diagnostic_path.open("x") as stream:
                            json.dump(record, stream, indent=2, allow_nan=False)
                            stream.write("\n")
                        print(f"Diagnostic saved to {diagnostic_path}", file=sys.stderr, flush=True)
                    except OSError as log_error:
                        print(f"Could not save IK diagnostic: {log_error}", file=sys.stderr, flush=True)
                    print(
                        f"Generation stopped; {len(records)} new completed episodes remain saved.",
                        file=sys.stderr, flush=True,
                    )
                    raise
                except DemonstrationFailure as error:
                    failures += 1
                    print(f"Discarded {episode_id}: {error}", flush=True)
                    continue
                record = save_episode(output_directory, episode, episode_id, split, settings)
                records.append(record)
                print(f"[{index}/{number_of_examples}] {split} {episode_id} "
                      f"steps={record['steps']} stack_stable={episode.final_stack_stable_time:.3f}s "
                      f"disturbances={len(episode.disturbance_events)} recoveries={episode.recovery_count} "
                      f"waypoint_retries={episode.waypoint_retry_count}", flush=True)
                break
            else:
                raise RuntimeError(
                    f"Could not collect example {index} after {MAXIMUM_ATTEMPTS_PER_EXAMPLE} attempts. "
                    f"The {len(records)} new completed episodes remain saved."
                )
    finally:
        if environment is not None:
            environment.close()
    print(f"Added {number_of_examples} episodes ({number_of_examples - test_count} train, "
          f"{test_count} test); {failures}/{attempts} attempts discarded. "
          f"Saved in {output_directory}.")
    return records


# Embedded tests keep all new source code within this file.


def run_tests() -> bool:
    """Unit tests, storage failure tests, and real MuJoCo replay tests."""
    import contextlib
    from dataclasses import replace
    from functools import partial
    import io
    import tempfile
    from types import SimpleNamespace
    import unittest
    from unittest.mock import patch

    module = sys.modules[__name__]

    def sample_episode() -> PickupEpisode:
        return PickupEpisode(
            observations=torch.zeros((3, PRIVILEGED_OBSERVATION_SIZE)),
            accepted_targets=torch.zeros((3, 3)),
            actions=torch.tensor([[0.0, 0.0, 0.0, CLOSED_GRIPPER],
                                  [0.0, 0.0, 0.0, OPEN_GRIPPER]]),
            final_hold_time=0.0,
            final_stack_stable_time=StackSuccessConfig().required_stable_time,
            stage_steps={"lower": 1, "release": 1},
        )

    def transition_info(**changes: Any) -> dict[str, Any]:
        info = dict(target_gripper_position=np.array([0.3, 0.0, 0.1]),
                    orange_grasp_hold_time=0.0, orange_currently_held=False,
                    stack_stable_time=0.0,
                    is_failure=False, ik_position_converged=True,
                    ik_tool_yaw_converged=True, is_success=False)
        return {**info, **changes}

    class RecordingTests(unittest.TestCase):
        def test_action_uses_accepted_target_and_clips_each_axis(self):
            env = SimpleNamespace(action_adapter=SimpleNamespace(
                current_target_gripper_position=np.array([0.3, 0.0, 0.1]),
                config=SimpleNamespace(maximum_position_delta=0.0025)))
            action = action_toward(env, np.array([0.31, -0.00125, 0.08]), -1.0)
            np.testing.assert_allclose(action, [1.0, -0.5, -1.0, -1.0])
            self.assertEqual(action.dtype, np.float32)

        def test_recording_alignment_copies_and_processed_target(self):
            observation = np.zeros(PRIVILEGED_OBSERVATION_SIZE, dtype=np.float32)
            target = np.array([0.3, 0.0, 0.1], dtype=np.float32)
            recorder = EpisodeRecorder(observation, {"target_gripper_position": target})
            observation.fill(2.0)
            target[0] = 0.29  # An accepted target can differ from the command's requested position.
            action = np.array([1.0, 0.0, 0.0, OPEN_GRIPPER], dtype=np.float32)
            info = transition_info(target_gripper_position=target, is_success=True,
                                   stack_stable_time=StackSuccessConfig().required_stable_time)
            env = SimpleNamespace(step=lambda issued: (observation, 0.0, True, False, info))
            self.assertTrue(recorder.step(env, action, "release"))
            action.fill(0.0)
            observation.fill(7.0)
            target.fill(8.0)
            episode = recorder.episode()
            np.testing.assert_array_equal(episode.observations[0], np.zeros(PRIVILEGED_OBSERVATION_SIZE))
            np.testing.assert_array_equal(episode.observations[1], np.full(PRIVILEGED_OBSERVATION_SIZE, 2.0))
            np.testing.assert_allclose(episode.accepted_targets, [[0.3, 0, 0.1], [0.29, 0, 0.1]])
            np.testing.assert_array_equal(episode.actions, [[1, 0, 0, OPEN_GRIPPER]])
            self.assertEqual(episode.final_hold_time, 0.0)
            self.assertEqual(episode.final_stack_stable_time, StackSuccessConfig().required_stable_time)
            with self.assertRaises(RuntimeError):
                recorder.step(env, action, "hold")

        def test_failure_and_success_guards(self):
            stable_time = StackSuccessConfig().required_stable_time
            cases = [
                (False, False, {"is_failure": True}),
                (False, False, {"ik_position_converged": False}),
                (False, False, {"ik_tool_yaw_converged": False}),
                (False, True, {}), (True, False, {}),
                (True, False, {"is_success": True, "orange_currently_held": True,
                               "orange_grasp_hold_time": 3.0, "stack_stable_time": stable_time}),
                (True, False, {"is_success": True, "stack_stable_time": stable_time / 2}),
                (True, False, {"is_success": True, "stack_stable_time": float("nan")}),
                (True, False, {"is_success": True, "stack_stable_time": float("inf")}),
                (True, False, {"is_failure": True, "is_success": True,
                               "stack_stable_time": stable_time}),
                (True, True, {"is_success": True, "stack_stable_time": stable_time}),
                (False, False, {"is_success": True}),
            ]
            for terminated, truncated, changes in cases:
                with self.subTest(changes=changes, terminated=terminated, truncated=truncated):
                    info = transition_info(**changes)
                    recorder = EpisodeRecorder(np.zeros(PRIVILEGED_OBSERVATION_SIZE), info)
                    env = SimpleNamespace(step=lambda action: (np.zeros(PRIVILEGED_OBSERVATION_SIZE), 0, terminated, truncated, info))
                    with self.assertRaises(DemonstrationFailure):
                        recorder.step(env, np.zeros(4), "test")
                    with self.assertRaises(DemonstrationFailure):
                        recorder.episode()

        def test_long_grasp_is_not_terminal_and_release_can_finish_with_zero_hold(self):
            observation = np.zeros(PRIVILEGED_OBSERVATION_SIZE)
            recorder = EpisodeRecorder(observation, transition_info())
            transitions = iter([
                (observation, 0.0, False, False, transition_info(
                    orange_currently_held=True, orange_grasp_hold_time=3.0)),
                (observation, 0.0, True, False, transition_info(
                    is_success=True, orange_grasp_hold_time=0.0,
                    stack_stable_time=StackSuccessConfig().required_stable_time)),
            ])
            env = SimpleNamespace(step=lambda action: next(transitions))
            self.assertFalse(recorder.step(env, np.array([0, 0, 0, CLOSED_GRIPPER]), "transport"))
            self.assertFalse(recorder.finished)
            with self.assertRaises(DemonstrationFailure):
                recorder.episode()
            self.assertTrue(recorder.step(env, np.array([0, 0, 0, OPEN_GRIPPER]), "release"))
            episode = recorder.episode()
            self.assertEqual(episode.final_hold_time, 0.0)
            self.assertEqual(list(episode.stage_steps), ["transport", "release"])

        def test_environment_randomizes_home_with_full_stack_episode_budget(self):
            with patch.object(module, "CubeStackGymEnvironment") as factory:
                environment = make_environment()
            self.assertIs(environment, factory.return_value)
            self.assertFalse(factory.call_args.kwargs["start_at_orange_waypoint"])
            self.assertEqual(factory.call_args.kwargs["recovery_start_probability"], 0.0)
            self.assertEqual(factory.call_args.kwargs["start_position_half_range"], START_POSITION_HALF_RANGE)
            self.assertEqual(START_POSITION_HALF_RANGE, (0.04, 0.04, 0.0))
            self.assertEqual(factory.call_args.kwargs["maximum_episode_steps"], MAXIMUM_EPISODE_STEPS)
            self.assertEqual(factory.call_args.kwargs["action_config"].target_tool_yaw, 0.0)
            self.assertEqual(MAXIMUM_EPISODE_STEPS, 900)

        def test_invalid_vectors_and_unfinished_episode(self):
            for vector in ([1, 2], [1, 2, float("inf")]):
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    copy_vector(vector, 3, "test")
            with self.assertRaises(KeyError):
                EpisodeRecorder(np.zeros(PRIVILEGED_OBSERVATION_SIZE), {})
            with self.assertRaisesRegex(ValueError, "observation must be a finite vector of length 49"):
                EpisodeRecorder(np.zeros(50), transition_info())
            recorder = EpisodeRecorder(np.zeros(PRIVILEGED_OBSERVATION_SIZE), transition_info())
            with self.assertRaises(DemonstrationFailure):
                recorder.episode()
            for action in ([2, 0, 0, 0], [float("nan"), 0, 0, 0], [0, 0, 0]):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    recorder.step(None, action, "invalid")

        def test_tensor_validation(self):
            episode = sample_episode()
            episode.validate()
            bad = [
                replace(episode, observations=episode.observations[:-1]),
                replace(episode, observations=torch.zeros((3, 50))),
                replace(episode, accepted_targets=torch.zeros((3, 4))),
                replace(episode, actions=torch.zeros((2, 3))),
                replace(episode, actions=episode.actions.double()),
                replace(episode, actions=torch.full((2, 4), float("nan"))),
                replace(episode, actions=torch.full((2, 4), 1.01)),
                replace(episode, actions=torch.zeros((0, 4))),
                replace(episode, final_hold_time=-0.01),
                replace(episode, final_hold_time=float("inf")),
                replace(episode, final_stack_stable_time=StackSuccessConfig().required_stable_time - 0.01),
                replace(episode, final_stack_stable_time=float("nan")),
                replace(episode, final_stack_stable_time=float("inf")),
                replace(episode, stage_steps={"lift": 1}),
            ]
            for invalid in bad:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    invalid.validate()

        def test_move_waits_for_consecutive_measured_arrivals_and_times_out(self):
            target = np.array([0.3, 0, 0.1])
            env = SimpleNamespace(previous_state={"gripper_position": target.copy()})
            errors = iter([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])
            calls = []
            def step(environment, action, stage):
                calls.append(action)
                environment.previous_state["gripper_position"] = target + [0, 0, next(errors)]
                return False
            with patch.object(module, "action_toward", return_value=np.zeros(4)):
                self.assertFalse(move_to(env, SimpleNamespace(step=step), target, 1.0, "descend"))
                self.assertEqual(len(calls), 6)
                env.previous_state["gripper_position"] = target + [0, 0, 0.02]
                with patch.object(module, "MAXIMUM_MOVE_STEPS", 3), self.assertRaises(DemonstrationFailure):
                    move_to(env, SimpleNamespace(step=lambda *args: False), target, 1.0, "descend")

        def test_holds_keep_xyz_and_stop_at_success(self):
            actions = []
            def step(environment, action, stage):
                actions.append(action.copy())
                return len(actions) == 3
            self.assertTrue(hold_position(None, SimpleNamespace(step=step), CLOSED_GRIPPER, 20, "hold"))
            np.testing.assert_array_equal(actions, [[0, 0, 0, -1]] * 3)

        def test_held_cube_movement_tracks_offset_and_requests_drop_recovery(self):
            cube_target = np.array([0.35, 0.0, 0.08])
            environment = SimpleNamespace(previous_state={
                "orange_currently_held": True,
                "orange_position": np.array([0.31, 0.01, 0.08]),
                "gripper_position": np.array([0.3, 0.0, 0.1]),
            })
            requested_targets = []
            def toward(env, target, gripper):
                requested_targets.append(target.copy())
                self.assertEqual(gripper, CLOSED_GRIPPER)
                return np.zeros(4)
            def step(env, action, stage):
                index = len(requested_targets)
                env.previous_state["orange_position"] = cube_target.copy()
                env.previous_state["gripper_position"] = np.array([0.3 + index * 0.001, 0.0, 0.1 + index * 0.001])
                return False
            with patch.object(module, "action_toward", side_effect=toward):
                self.assertFalse(move_held_cube_to(environment, SimpleNamespace(step=step), cube_target, "transport"))
            np.testing.assert_allclose(requested_targets,
                                       [[0.34, -0.01, 0.1], [0.301, 0.0, 0.101], [0.302, 0.0, 0.102]], atol=1e-8)
            environment.previous_state["orange_currently_held"] = False
            recorder = SimpleNamespace(step=lambda *args: self.fail("An unacquired cube must restart pickup"))
            self.assertFalse(move_held_cube_to(environment, recorder, cube_target, "lower"))
            self.assertTrue(recorder.recovery_requested)
            environment.previous_state["orange_currently_held"] = True
            environment.previous_state["orange_position"] = cube_target + [0.02, 0, 0]
            with patch.object(module, "action_toward", return_value=np.zeros(4)), \
                    patch.object(module, "MAXIMUM_MOVE_STEPS", 2), self.assertRaises(DemonstrationFailure):
                move_held_cube_to(environment, SimpleNamespace(step=lambda *args: False), cube_target, "transport")

    class StorageTests(unittest.TestCase):
        def setUp(self):
            self.temporary = tempfile.TemporaryDirectory(prefix=".demonstration-test-", dir=REPOSITORY_ROOT)
            self.addCleanup(self.temporary.cleanup)
            self.output = Path(self.temporary.name)

        def test_round_trip_and_additive_saves(self):
            episode = sample_episode()
            first = save_episode(self.output, episode, uuid4(), "train", {})
            old_bytes = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
            second = save_episode(self.output, episode, uuid4(), "test", {})
            self.assertNotEqual(first["uuid"], second["uuid"])
            self.assertEqual(len(list(self.output.rglob("*.pt"))), 6)
            self.assertCountEqual(read_manifest(self.output), [first, second])
            for path, contents in old_bytes.items():
                self.assertEqual(path.read_bytes(), contents)
                if path.suffix == ".pt":
                    loaded = torch.load(path, map_location="cpu", weights_only=True)
                    torch.testing.assert_close(loaded, getattr(episode, path.parent.name))

        def test_uuid_is_the_full_reset_seed(self):
            episode_id = UUID("e5a8cbbf-c9a2-4773-934b-d3d638d0a219")
            record = save_episode(self.output, sample_episode(), episode_id, "train", {})
            self.assertEqual(record["seed"], episode_id.int)
            self.assertGreater(record["seed"], 2 ** 64)
            self.assertEqual(record["uuid"], str(episode_id))
            np.testing.assert_array_equal(
                np.random.default_rng(record["seed"]).random(10),
                np.random.default_rng(UUID(record["uuid"]).int).random(10),
            )
            self.assertTrue((self.output / "manifests" / f"{episode_id}.json").exists())

        def test_uuid_collision_never_overwrites_either_split(self):
            episode_id = uuid4()
            first = save_episode(self.output, sample_episode(), episode_id, "train", {})
            original = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
            for split in ("train", "test"):
                with self.subTest(split=split), self.assertRaises(FileExistsError):
                    save_episode(self.output, sample_episode(), episode_id, split, {})
                self.assertEqual({path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}, original)
            self.assertEqual(read_manifest(self.output), [first])

        def test_tensor_and_manifest_write_failures_roll_back(self):
            save_episode(self.output, sample_episode(), uuid4(), "train", {})
            original = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
            real_save = torch.save
            def fail_second_save(tensor, stream):
                if "accepted_targets" in stream.name:
                    stream.write(b"partial")
                    raise OSError("simulated disk failure")
                real_save(tensor, stream)
            for failure in (patch.object(torch, "save", side_effect=fail_second_save),
                            patch.object(os, "link", side_effect=OSError("manifest write failed"))):
                with self.subTest(failure=failure), failure, self.assertRaises(OSError):
                    save_episode(self.output, sample_episode(), uuid4(), "test", {})
                self.assertEqual({path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}, original)

        def test_cleanup_failure_after_publication_keeps_committed_episode(self):
            with patch.object(Path, "unlink", side_effect=OSError("cleanup failed")), self.assertRaises(OSError):
                save_episode(self.output, sample_episode(), uuid4(), "train", {})
            self.assertEqual(len(read_manifest(self.output)), 1)
            self.assertEqual(len(list(self.output.rglob("*.pt"))), 3)

        def test_collisions_with_partial_files_preserve_original_contents(self):
            episode_id = uuid4()
            for relative_path in (f"train/accepted_targets/{episode_id}.pt", f"manifests/.{episode_id}.tmp"):
                with self.subTest(relative_path=relative_path):
                    existing = self.output / relative_path
                    existing.parent.mkdir(parents=True, exist_ok=True)
                    existing.write_bytes(b"preexisting data")
                    with self.assertRaises(FileExistsError):
                        save_episode(self.output, sample_episode(), episode_id, "train", {})
                    self.assertEqual(existing.read_bytes(), b"preexisting data")
                    self.assertEqual([path for path in self.output.rglob("*") if path.is_file()], [existing])
                    existing.unlink()

        def test_manifest_rejects_missing_data_wrong_seed_and_bad_json(self):
            episode_id = uuid4()
            save_episode(self.output, sample_episode(), episode_id, "train", {})
            manifest = self.output / "manifests" / f"{episode_id}.json"
            content = json.loads(manifest.read_text())
            content["episode"]["seed"] += 1
            manifest.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                read_manifest(self.output)
            content["episode"]["seed"] -= 1
            manifest.write_text(json.dumps(content))
            next(self.output.rglob("*.pt")).unlink()
            with self.assertRaises(ValueError):
                read_manifest(self.output)
            manifest.write_text("{")
            with self.assertRaises(ValueError):
                read_manifest(self.output)

        def test_legacy_manifest_remains_unchanged_and_readable(self):
            episode_id = uuid4()
            legacy = save_episode(self.output, sample_episode(), episode_id, "train", {})
            (self.output / "manifests" / f"{episode_id}.json").unlink()
            legacy["seed"] = 7  # Old datasets used a counter unrelated to UUID.
            manifest = self.output / "manifest.json"
            manifest.write_text(json.dumps({"format_version": 1, "episodes": [legacy]}))
            original = manifest.read_bytes()
            new = save_episode(self.output, sample_episode(), uuid4(), "test", {})
            self.assertCountEqual(read_manifest(self.output), [legacy, new])
            self.assertEqual(manifest.read_bytes(), original)

        def test_concurrent_writers_keep_every_episode(self):
            from concurrent.futures import ThreadPoolExecutor
            from threading import Barrier
            barrier = Barrier(4)
            def write_one(index):
                barrier.wait(timeout=10)
                return save_episode(self.output, sample_episode(), uuid4(),
                                    "train" if index % 2 else "test", {})
            with ThreadPoolExecutor(max_workers=4) as pool:
                records = list(pool.map(write_one, range(12)))
            self.assertCountEqual(read_manifest(self.output), records)
            self.assertEqual(len(list(self.output.rglob("*.pt"))), 36)
            self.assertFalse(list(self.output.rglob("*.lock")))

        def test_reader_ignores_writer_until_publication(self):
            from concurrent.futures import ThreadPoolExecutor
            from threading import Event
            ready, release = Event(), Event()
            real_link = os.link
            def delayed_publish(source, destination):
                ready.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("reader did not release writer")
                real_link(source, destination)
            with ThreadPoolExecutor(max_workers=1) as pool, patch.object(os, "link", side_effect=delayed_publish):
                future = pool.submit(save_episode, self.output, sample_episode(), uuid4(), "train", {})
                try:
                    self.assertTrue(ready.wait(timeout=10))
                    self.assertEqual(read_manifest(self.output), [])
                finally:
                    release.set()
                record = future.result(timeout=10)
            self.assertEqual(read_manifest(self.output), [record])

        def test_generation_adds_exact_counts_with_disjoint_seeds_and_closes(self):
            from unittest.mock import Mock
            env = Mock()
            with patch.object(module, "make_environment", return_value=env), \
                    patch.object(module, "generation_settings", return_value={}), \
                    patch.object(module, "collect_episode", return_value=sample_episode()) as collect, \
                    contextlib.redirect_stdout(io.StringIO()):
                first = generate_dataset(5, self.output)
                old_bytes = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
                second = generate_dataset(5, self.output)
            self.assertEqual(Counter(record["split"] for record in first), {"train": 4, "test": 1})
            self.assertEqual(Counter(record["split"] for record in second), {"train": 4, "test": 1})
            self.assertEqual(len({record["seed"] for record in first + second}), 10)
            self.assertEqual([call.args[1] for call in collect.call_args_list],
                             [UUID(record["uuid"]).int for record in first + second])
            self.assertEqual(len(list(self.output.rglob("*.pt"))), 30)
            self.assertEqual(env.close.call_count, 2)
            self.assertFalse(list(self.output.rglob("*.lock")))
            for path, contents in old_bytes.items():
                self.assertEqual(path.read_bytes(), contents)

        def test_failed_attempt_retry_limit_and_resume_uses_fresh_uuids(self):
            from unittest.mock import Mock
            env = Mock()
            identifiers = [uuid4() for _ in range(5)]
            with patch.object(module, "make_environment", return_value=env), \
                    patch.object(module, "generation_settings", return_value={}), \
                    patch.object(module, "MAXIMUM_ATTEMPTS_PER_EXAMPLE", 2), \
                    patch.object(module, "uuid4", side_effect=identifiers), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(module, "collect_episode", side_effect=[
                    DemonstrationFailure("miss"), sample_episode(),
                    DemonstrationFailure("miss"), DemonstrationFailure("miss"),
                ]) as collect, self.assertRaises(RuntimeError):
                    generate_dataset(2, self.output)
                self.assertEqual([call.args[1] for call in collect.call_args_list],
                                 [identifier.int for identifier in identifiers[:4]])
                saved = read_manifest(self.output)
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["uuid"], str(identifiers[1]))
                with patch.object(module, "collect_episode", return_value=sample_episode()):
                    resumed = generate_dataset(1, self.output)
                self.assertEqual(resumed[0]["uuid"], str(identifiers[4]))
                self.assertEqual(len(read_manifest(self.output)), 2)
            self.assertEqual(env.close.call_count, 2)

        def test_invalid_requests(self):
            for count in (0, -1, True, 2.5):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    generate_dataset(count, self.output)
            for fraction in (-0.1, 1.1, float("nan")):
                with patch.object(module, "TEST_FRACTION", fraction), self.assertRaises(ValueError):
                    generate_dataset(1, self.output)

    class PhysicsTests(unittest.TestCase):
        def setUp(self):
            # These tests replay intended actions without an actuator override.
            # Disturbed physical trajectories have dedicated regression tests.
            disturbance = patch.object(module, "DropDisturbance", partial(DropDisturbance, probability=0.0))
            disturbance.start()
            self.addCleanup(disturbance.stop)

        @staticmethod
        def best_effort_environment():
            # Storage/replay tests retain the original controller semantics;
            # strict orientation failures have dedicated regression tests.
            with patch.object(module, "CartesianActionConfig",
                              partial(CartesianActionConfig, require_downward=False)):
                return make_environment()

        def test_parallel_best_effort_generators_keep_both_batches(self):
            import subprocess
            with tempfile.TemporaryDirectory(prefix=".demonstration-parallel-", dir=REPOSITORY_ROOT) as directory:
                output = Path(directory) / "dataset"
                child_code = """
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[2])
import generate_pickup_demonstrations as generator
generator.DropDisturbance = partial(generator.DropDisturbance, probability=0.0)
original_factory = generator.make_environment
def best_effort_environment():
    with patch.object(generator, "CartesianActionConfig",
                      partial(generator.CartesianActionConfig, require_downward=False)):
        return original_factory()
generator.make_environment = best_effort_environment
generator.generate_dataset(3, Path(sys.argv[1]))
"""
                command = [sys.executable, "-B", "-c", child_code,
                           str(output), str(Path(__file__).resolve().parent)]
                processes = []
                try:
                    for _ in range(2):
                        processes.append(subprocess.Popen(command, cwd=directory, text=True,
                                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                    for process in processes:
                        stdout, stderr = process.communicate(timeout=60)
                        self.assertEqual(process.returncode, 0, stdout + stderr)
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.communicate()
                records = read_manifest(output)
                self.assertEqual(len(records), 6)
                self.assertEqual(len({record["uuid"] for record in records}), 6)
                self.assertEqual(Counter(record["split"] for record in records), {"train": 4, "test": 2})
                self.assertEqual(len(list(output.rglob("*.pt"))), 18)
                self.assertFalse(list(output.rglob("*.lock")))

        def test_best_effort_stacks_replay_complete_stage_order_and_every_history_token(self):
            from action_observation_history import ActionObservationHistoryWrapper
            from environment import DEFAULT_OPEN_GRIPPER_POSITION, DEFAULT_START_POSITION
            from kinematics import DEFAULT_POSITION_TOLERANCE
            environment = self.best_effort_environment()
            self.addCleanup(environment.close)
            with tempfile.TemporaryDirectory(prefix=".demonstration-physics-", dir=REPOSITORY_ROOT) as directory:
                episode_ids = [UUID(value) for value in (
                    "e5a8cbbf-c9a2-4773-934b-d3d638d0a219",
                    "61c27256-44ea-4d88-a71c-c28d2a082d91",
                    "00000000-0000-0000-0000-000000000012",  # Fixed seed 18; generation itself uses UUID4.
                )]
                initial_positions = []
                for episode_id in episode_ids:
                    seed = episode_id.int
                    with self.subTest(seed=seed):
                        episode = collect_episode(environment, seed)
                        self.assertEqual(episode.stage_steps["close"], CLOSE_GRIPPER_STEPS)
                        self.assertEqual(len(episode.actions), environment.episode_step_count)
                        self.assertTrue(environment.simulation.is_success())
                        self.assertTrue(environment.simulation.stack_conditions_met())
                        self.assertGreaterEqual(episode.final_stack_stable_time, StackSuccessConfig().required_stable_time)
                        self.assertEqual(episode.final_hold_time, 0.0)
                        self.assertEqual(episode.observations.shape, (len(episode.actions) + 1, 49))
                        initial_position = episode.observations[0, 18:21].numpy()
                        initial_positions.append(initial_position.copy())
                        self.assertTrue(np.all(
                            np.abs(initial_position - DEFAULT_START_POSITION)
                            <= np.asarray(START_POSITION_HALF_RANGE) + DEFAULT_POSITION_TOLERANCE
                        ))
                        self.assertEqual(episode.observations[0, 5].item(), DEFAULT_OPEN_GRIPPER_POSITION)
                        np.testing.assert_array_equal(episode.observations[0, 6:12], 0.0)
                        np.testing.assert_allclose(episode.accepted_targets[0], episode.observations[0, 18:21], rtol=0, atol=1e-7)
                        self.assertEqual(episode.actions[0, 3].item(), OPEN_GRIPPER)
                        self.assertEqual(episode.actions[-1, 3].item(), OPEN_GRIPPER)
                        stages = list(episode.stage_steps)
                        self.assertEqual(stages[:10], ["approach", "settle_waypoint", "descend", "settle", "close",
                                                        "lift", "settle_lift", "transport", "lower", "release"])
                        self.assertTrue(all(stage in ("retreat", "settle_stack") for stage in stages[10:]))
                        stage_offset = 0
                        for stage, step_count in episode.stage_steps.items():
                            expected_gripper = (CLOSED_GRIPPER if stage in ("close", "lift", "settle_lift", "transport", "lower")
                                                else OPEN_GRIPPER)
                            self.assertTrue(torch.all(episode.actions[stage_offset:stage_offset + step_count, 3] == expected_gripper), stage)
                            stage_offset += step_count
                        record = save_episode(Path(directory), episode, episode_id, "train", generation_settings(environment))
                        self.assertEqual(record["final_stack_stable_time"], episode.final_stack_stable_time)
                        self.assertEqual(record["settings"]["task"], "stack_orange_on_blue")
                        self.assertFalse(record["settings"]["start_at_orange_waypoint"])
                        self.assertEqual(record["settings"]["success_config"], asdict(StackSuccessConfig()))
                        self.assertEqual(record["settings"]["maximum_episode_steps"], MAXIMUM_EPISODE_STEPS)
                        self.assertEqual(record["settings"]["start_position"], list(DEFAULT_START_POSITION))
                        self.assertEqual(record["settings"]["start_position_half_range"], list(START_POSITION_HALF_RANGE))
                        self.assertEqual(record["settings"]["action_config"]["target_tool_yaw"], 0.0)
                        for name in ("observations", "accepted_targets", "actions"):
                            tensor = torch.load(Path(directory) / "train" / name / f"{record['uuid']}.pt", weights_only=True)
                            torch.testing.assert_close(tensor, getattr(episode, name))

                        replay = ActionObservationHistoryWrapper(self.best_effort_environment(), history_length=64)
                        try:
                            history, initial_info = replay.reset(seed=seed)
                            self.assertEqual(initial_info["episode_start_type"], "home")
                            self.assertEqual(history["tokens"].shape, (64, 56))
                            np.testing.assert_array_equal(history["tokens"][0, 18:21], initial_position)
                            observed_grasp = False
                            longest_observed_hold = 0.0
                            closing_end = sum(episode.stage_steps[name] for name in stages[:5])
                            for time in range(len(episode.actions) + 1):
                                if time == closing_end:
                                    # Recorded actions must reproduce the world-facing
                                    # grasp through the same controller used by policies.
                                    rotation = replay.unwrapped.simulation.data.site("gripperframe").xmat.reshape(3, 3)
                                    closing_yaw = math.atan2(rotation[1, 2], rotation[0, 2])
                                    self.assertLess(abs(closing_yaw), math.radians(1.0))
                                first_time = max(0, time - 63)
                                count = time - first_time + 1
                                expected = []
                                for tick in range(first_time, time + 1):
                                    previous_action = np.zeros(4) if tick == 0 else episode.actions[tick - 1].numpy()
                                    expected.append(np.concatenate((episode.observations[tick].numpy(), previous_action,
                                                                    episode.accepted_targets[tick].numpy())))
                                np.testing.assert_allclose(history["tokens"][:count], expected, rtol=1e-5, atol=1e-6)
                                np.testing.assert_array_equal(history["tokens"][count:], 0.0)
                                np.testing.assert_array_equal(history["valid"], [1.0] * count + [0.0] * (64 - count))
                                expected_starts = np.zeros(64)
                                if first_time == 0:
                                    expected_starts[0] = 1.0
                                np.testing.assert_array_equal(history["episode_start"], expected_starts)
                                if time < len(episode.actions):
                                    history, _, terminated, truncated, info = replay.step(episode.actions[time].numpy())
                                    self.assertFalse(truncated)
                                    self.assertEqual(terminated, time == len(episode.actions) - 1)
                                    if info["orange_currently_held"]:
                                        observed_grasp = True
                                        self.assertFalse(terminated)
                                        self.assertFalse(info["is_success"])
                                    longest_observed_hold = max(longest_observed_hold, info["orange_grasp_hold_time"])
                            self.assertTrue(info["is_success"])
                            self.assertFalse(info["orange_currently_held"])
                            self.assertGreaterEqual(info["stack_stable_time"], StackSuccessConfig().required_stable_time)
                            self.assertTrue(observed_grasp)
                            self.assertGreater(longest_observed_hold, 2.0)  # The previous pickup condition must not terminate stacking.
                        finally:
                            replay.close()
                self.assertEqual(len(initial_positions), len(episode_ids))
                self.assertTrue(np.all(np.ptp(initial_positions, axis=0)[:2] > 0.001))
                np.testing.assert_allclose(np.asarray(initial_positions)[:, 2], DEFAULT_START_POSITION[2],
                                           rtol=0.0, atol=DEFAULT_POSITION_TOLERANCE)

    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (RecordingTests, StorageTests, PhysicsTests))
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=int, default=NUMBER_OF_EXAMPLES,
                        help="Number of new successful episodes to add (default: %(default)s).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="Dataset root containing train/, test/, and manifests/.")
    parser.add_argument("--test", action="store_true", help="Run embedded unit/integration tests without generating a dataset.")
    arguments = parser.parse_args()
    if arguments.test:
        raise SystemExit(0 if run_tests() else 1)
    try:
        generate_dataset(arguments.examples, arguments.output_dir)
    except IKConvergenceError:
        # generate_dataset already printed and recorded the detailed failure.
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
