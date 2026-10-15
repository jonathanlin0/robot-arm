"""Physics-driven task scenes for vision data, never a runtime perception path.

The teacher may inspect simulator cube/contact truth to create approach, grasp,
carry, drop, recovery, and placement examples. Images and pose labels are still
recorded by the vision-data collector. Phase names describe attempted behavior;
separate measured flags report what actually occurred. Local IK errors escape
unchanged instead of turning an unsuccessful sequence into a success claim.
"""

from __future__ import annotations

import numpy as np

from cartesian_controller import CartesianController
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from robot_state_provider import SimRobotStateProvider
from schemas.config import CartesianActionConfig, VisionCarryConfig, VisionScenarioConfig


class ScriptedVisionScenario:
    """Issue one Cartesian teacher command per step; the collector advances time.

    Cubes should initially be separated and axis-aligned. The driver does not
    reset the supplied scene, reposition objects, or own a second simulator.
    It deliberately opens once after a measured carry, settles, and retries.
    """

    def __init__(self, environment: CubeStackEnvironment, backend: SimJointCommandBackend,
                 controller_config: CartesianActionConfig, config: VisionScenarioConfig) -> None:
        if backend.environment is not environment:
            raise ValueError("Scenario and backend must share one environment.")
        self.environment, self.backend, self.config = environment, backend, config
        self.provider = SimRobotStateProvider(environment=environment)
        self.controller = CartesianController(robot_model=environment.model, config=controller_config)
        self.controller.reset(self.provider.read())
        self.phase = "approach"
        self._phase_started = float(environment.data.time)
        self._target = None
        self._dropped_intentionally = False
        self._lost_contact_steps = 0
        self._grasp_seen = False
        self._lift_seen = False
        self._drop_seen = False
        self._stack_seen = False
        self._recovery_count = 0
        self._last_failure = None
        self._phases_seen = {self.phase}
        self._table = environment.model.geom("table").id
        self._orange = environment.model.geom("orange_cube_geom").id
        self._blue = environment.model.geom("blue_cube_geom").id
        self._fixed_pads = self._pad_ids("fixed_jaw", ("box3", "box4", "box5", "box6", "box7"))
        self._moving_pads = self._pad_ids("moving_jaw", ("box2", "box3"))
        self._initial_cube_z = float(environment.data.body("orange_cube").xpos[2])

    def _pad_ids(self, prefix: str, boxes: tuple[str, ...]) -> set[int]:
        names = [f"{prefix}_sph_tip{i}" for i in (1, 2, 3)]
        names.extend(f"{prefix}_{box}" for box in boxes)
        return {self.environment.model.geom(name).id for name in names}

    def _scene(self) -> dict:
        with self.environment.state_lock:
            data, model = self.environment.data, self.environment.model
            touching = set()
            for contact in data.contact[:data.ncon]:
                if int(contact.geom1) == self._orange:
                    touching.add(int(contact.geom2))
                elif int(contact.geom2) == self._orange:
                    touching.add(int(contact.geom1))
            orange = data.body("orange_cube").xpos.copy()
            blue = data.body("blue_cube").xpos.copy()
            fixed, moving = bool(touching & self._fixed_pads), bool(touching & self._moving_pads)
            held = fixed and moving and self._table not in touching
            height = float(model.geom("orange_cube_geom").size[2] + model.geom("blue_cube_geom").size[2])
            stacked = (not fixed and not moving and self._blue in touching
                       and np.linalg.norm(orange[:2] - blue[:2]) < self.config.position_tolerance
                       and abs(orange[2] - blue[2] - height) < self.config.position_tolerance)
            return {"time": float(data.time), "orange": orange, "blue": blue,
                    "gripper": data.site("gripperframe").xpos.copy(), "held": held,
                    "both_pads": fixed and moving, "stacked": bool(stacked), "stack_height": height}

    def metadata(self) -> dict:
        """Return JSON-compatible phase coverage and observed milestones."""
        scene = self._scene()
        return {
            "phase": self.phase, "phases_seen": sorted(self._phases_seen),
            "currently_held": bool(scene["held"]), "grasp_seen": self._grasp_seen,
            "lift_seen": self._lift_seen, "drop_seen": self._drop_seen,
            "intentional_drop_started": self._dropped_intentionally,
            "recovery_count": self._recovery_count, "stack_seen": self._stack_seen,
            "stack_conditions_now": bool(scene["stacked"]), "last_failure": self._last_failure,
        }

    def _enter(self, phase: str, now: float) -> None:
        self.phase, self._phase_started, self._target = phase, now, None
        self._phases_seen.add(phase)
        self._lost_contact_steps = 0

    def _on_phase_timeout(self, scene: dict) -> None:
        self._last_failure = f"{self.phase}: phase timeout"
        self._enter("recover_settle", scene["time"])

    def _issue(self, target: np.ndarray | None, gripper: float) -> None:
        delta = np.zeros(3) if target is None else (
            (target - self.controller.prev_accepted_target) / self.controller.config.maximum_position_delta
        )
        action = np.concatenate((np.clip(delta, -1.0, 1.0), [gripper]))
        robot = self.provider.read()
        command = self.controller.compute(action=action, robot_state=robot)
        receipt = self.backend.send_joint_targets(targets=command.joint_targets)
        self.controller.update_prev_accepted_xyz(command=command, receipt=receipt)

    def _step_carry(self, scene: dict, elapsed: float) -> None:
        if self._target is None:
            self._target = scene["orange"].copy()
            self._target[:2] = scene["blue"][:2]
        offset = scene["orange"] - scene["gripper"]
        self._issue(self._target - offset, -1.0)
        if not self._dropped_intentionally and elapsed >= self.config.drop_after_seconds and scene["held"]:
            self._dropped_intentionally = True
            self._enter("drop", scene["time"])
        elif self._dropped_intentionally and np.linalg.norm(scene["orange"] - self._target) <= self.config.position_tolerance:
            self._enter("place", scene["time"])

    def step(self) -> None:
        """Read measured progress, transition phases, and send one teacher action."""
        scene = self._scene()
        now = scene["time"]
        self._grasp_seen |= bool(scene["held"])
        self._lift_seen |= bool(scene["held"] and scene["orange"][2] > self._initial_cube_z + 0.04)
        self._stack_seen |= bool(scene["stacked"])
        elapsed = now - self._phase_started
        if self.phase not in ("complete", "recover_settle") and elapsed > self.config.phase_timeout_seconds:
            self._on_phase_timeout(scene)
            elapsed = 0.0
        if self.phase in ("carry", "place"):
            self._lost_contact_steps = 0 if scene["held"] else self._lost_contact_steps + 1
            if self._lost_contact_steps >= 2:
                self._drop_seen |= self._lift_seen
                self._last_failure = f"{self.phase}: measured grasp lost"
                self._enter("recover_settle", now)
                elapsed = 0.0

        if self.phase == "approach":
            if self._target is None:
                self._target = scene["orange"] + [0.0, 0.0, self.config.waypoint_height]
            self._issue(self._target, 1.0)
            if np.linalg.norm(scene["gripper"] - self._target) <= self.config.position_tolerance:
                self._enter("descend", now)
        elif self.phase == "descend":
            if self._target is None:
                self._target = scene["orange"] + np.asarray(self.config.grasp_offset)
            self._issue(self._target, 1.0)
            if np.linalg.norm(scene["gripper"] - self._target) <= self.config.position_tolerance:
                self._enter("close", now)
        elif self.phase == "close":
            self._issue(None, -1.0)
            if elapsed >= self.config.close_seconds:
                self._enter("lift", now)
        elif self.phase == "lift":
            if self._target is None:
                self._target = scene["gripper"].copy()
                self._target[2] = self._initial_cube_z + self.config.lift_height
            self._issue(self._target, -1.0)
            if np.linalg.norm(scene["gripper"] - self._target) <= self.config.position_tolerance:
                if scene["held"]:
                    self._enter("carry", now)
                else:
                    self._last_failure = "lift: no measured grasp"
                    self._enter("recover_settle", now)
        elif self.phase == "carry":
            self._step_carry(scene, elapsed)
        elif self.phase == "drop":
            self._issue(None, 1.0)
            self._drop_seen |= not bool(scene["held"])
            if elapsed >= self.config.release_seconds:
                self._enter("recover_settle", now)
        elif self.phase == "recover_settle":
            self._issue(None, 1.0)
            if elapsed >= self.config.settle_seconds:
                self._recovery_count += 1
                self._enter("approach", now)
        elif self.phase == "place":
            target = scene["blue"] + [0.0, 0.0, scene["stack_height"]]
            offset = scene["orange"] - scene["gripper"]
            self._issue(target - offset, -1.0)
            if np.linalg.norm(scene["orange"] - target) <= self.config.position_tolerance:
                self._enter("release", now)
        elif self.phase == "release":
            self._issue(None, 1.0)
            if elapsed >= self.config.release_seconds:
                self._enter("settle_stack", now)
        elif self.phase == "settle_stack":
            self._issue(None, 1.0)
            if elapsed >= self.config.settle_seconds:
                if not scene["stacked"]:
                    self._last_failure = "placement: no measured released stack"
                self._enter("complete", now)
        elif self.phase == "complete":
            self._issue(None, 1.0)


class ScriptedCarryScenario(ScriptedVisionScenario):
    """Pick up orange, then carry it between random cube-center XYZ targets.

    The collector controls the sequence duration. Each reached target starts a
    new move, so the ordinary phase timeout applies per move. Contact loss uses
    the same settle-and-retry pickup as task mode, without a deliberate release.
    """

    def __init__(self, environment: CubeStackEnvironment, backend: SimJointCommandBackend,
                 controller_config: CartesianActionConfig, config: VisionScenarioConfig,
                 carry_config: VisionCarryConfig, rng: np.random.Generator) -> None:
        super().__init__(environment=environment, backend=backend,
                         controller_config=controller_config, config=config)
        self.carry_config, self.rng = carry_config, rng
        self._carry_targets_reached = 0

    def _on_phase_timeout(self, scene: dict) -> None:
        if self.phase == "carry" and scene["held"]:
            # A stalled move should choose another target without releasing a
            # cube that is still held. Actual contact loss uses normal recovery.
            self._last_failure = "carry: phase timeout"
            self._enter("carry", scene["time"])
        else:
            super()._on_phase_timeout(scene)

    def _step_carry(self, scene: dict, elapsed: float) -> None:
        if self._target is None:
            bounds = np.asarray((self.carry_config.x_range, self.carry_config.y_range,
                                 self.carry_config.z_range))
            self._target = self.rng.uniform(bounds[:, 0], bounds[:, 1])
        offset = scene["orange"] - scene["gripper"]
        self._issue(self._target - offset, -1.0)
        if scene["held"] and np.linalg.norm(scene["orange"] - self._target) <= self.config.position_tolerance:
            self._carry_targets_reached += 1
            self._enter("carry", scene["time"])

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata["carry_targets_reached"] = self._carry_targets_reached
        return metadata
