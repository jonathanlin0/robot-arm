from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike

from cartesian_actions import CARTESIAN_ACTION_SIZE, CartesianActionResult
from environment import StateSnapshot
from success import StackSuccessConfig


@dataclass(frozen=True)
class StackRewardConfig:
    """Weights used by the shaped stacking reward."""

    # scaled reward for gripper approaching orange cube
    approach_orange_progress_weight: float = 1.0
    # one-time reward for getting the clamp grasps the orange cube
    grasp_candidate_reward: float = 1.0
    # one-time reward when the orange cube is grasped and off the table
    grasp_reward: float = 5.0
    # scaled reward for lifting the cube up
    lift_orange_progress_weight: float = 5.0
    # margin above target orange cube height for lifting it
    vertical_lift_margin: float = 0.03 # 3 cm
    # scaled reward for moving toward the safe hover position above blue
    move_toward_hover_progress_weight: float = 2.0
    # scaled reward for lowering toward the final stack after alignment
    lower_toward_stack_progress_weight: float = 2.0
    # scaled reward for horizontally aligning the two cube centers
    stack_alignment_progress_weight: float = 5.0
    successful_stack_reward: float = 100.0
    dropped_cube_penalty: float = -10.0
    # penatly when the IK can't solve the intended gripper location. likely means the orientation of the arms is in a weird shape
    ik_failure_penalty: float = -1.0
    action_magnitude_penalty_weight: float = -0.01
    # small one-time cost whenever the persistent gripper target switches
    gripper_state_change_penalty: float = -0.01
    # discourage a close attempt that does not reach both jaw pads
    unproductive_close_penalty: float = -0.05
    # time allowed for the physical jaws to close before judging the attempt
    close_contact_grace_period: float = 0.25

    # CONCERNS
    # policy may be rewarded to pick up the orange cube but not drop it if the rewards for picking it up are too high

    def __post_init__(self) -> None:
        nonnegative_values = (
            self.approach_orange_progress_weight,
            self.grasp_candidate_reward,
            self.grasp_reward,
            self.lift_orange_progress_weight,
            self.vertical_lift_margin,
            self.move_toward_hover_progress_weight,
            self.lower_toward_stack_progress_weight,
            self.stack_alignment_progress_weight,
            self.successful_stack_reward,
            self.close_contact_grace_period,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in nonnegative_values
        ):
            raise ValueError(
                "reward configuration values must be finite and "
                "nonnegative."
            )

        penalty_values = (
            self.dropped_cube_penalty,
            self.ik_failure_penalty,
            self.action_magnitude_penalty_weight,
            self.gripper_state_change_penalty,
            self.unproductive_close_penalty,
        )
        if not all(
            math.isfinite(value) and value <= 0.0
            for value in penalty_values
        ):
            raise ValueError("penalties must be finite and nonpositive.")


@dataclass(frozen=True)
class RewardResult:
    """Total reward plus named components for logging and debugging."""

    total: float
    components: dict[str, float]


def _has_bilateral_jaw_contact(state: StateSnapshot) -> bool:
    return bool(state["orange_touches_fixed_jaw"]) and bool(
        state["orange_touches_moving_jaw"]
    )


def _stack_target_distance(
    orange_position: np.ndarray,
    blue_position: np.ndarray,
    vertical_distance: float,
) -> float:
    """Return the orange cube's distance from its ideal stacked position.

    The ideal target keeps the blue cube's world-space X and Y coordinates
    and adds the desired center-to-center vertical separation to its Z
    coordinate. The returned Euclidean distance is used to reward movement
    toward that target rather than movement toward the blue cube's center.

    Args:
        orange_position: Orange cube center as world-space ``[x, y, z]``
            coordinates in meters.
        blue_position: Blue cube center as world-space ``[x, y, z]``
            coordinates in meters.
        vertical_distance: Desired vertical distance between the two cube
            centers in meters when correctly stacked.

    Returns:
        Distance in meters between the orange cube center and its ideal
        stacked position above the blue cube.
    """
    stack_target = blue_position.copy()
    stack_target[2] += vertical_distance
    return float(np.linalg.norm(orange_position - stack_target))


def _hover_target_distance(
    orange_position: np.ndarray,
    blue_position: np.ndarray,
    vertical_distance: float,
    vertical_lift_margin: float,
) -> float:
    """Return distance from orange to the safe hover target above blue."""
    hover_target = blue_position.copy()
    hover_target[2] += vertical_distance + vertical_lift_margin
    return float(np.linalg.norm(orange_position - hover_target))


def _horizontal_alignment_error(
    orange_position: np.ndarray,
    blue_position: np.ndarray,
) -> float:
    """Return the larger absolute X/Y center offset in meters."""
    return float(
        np.max(np.abs(orange_position[:2] - blue_position[:2]))
    )


class StackRewardCalculator:
    """Calculate rewards while tracking one-time episode events."""

    def __init__(
        self,
        config: StackRewardConfig | None = None,
        success_config: StackSuccessConfig | None = None,
    ) -> None:
        self.config = config or StackRewardConfig()
        self.success_config = success_config or StackSuccessConfig()

        # i feel that the environment should own these values.
        # TODO: move this so that the environment owns the values
        self._initial_orange_height: float | None = None
        self._bilateral_contact_seen = False
        self._confirmed_grasp_seen = False
        self._drop_penalized = False
        self._safe_lift_completed = False
        self._hover_alignment_completed = False
        self._close_attempt_start_time: float | None = None

    @property
    def confirmed_grasp_seen(self) -> bool:
        """Return whether this episode has contained a confirmed grasp."""
        return self._confirmed_grasp_seen

    @property
    def safe_lift_completed(self) -> bool:
        """Return whether the current attempt reached the safe hover height."""
        return self._safe_lift_completed

    @property
    def hover_alignment_completed(self) -> bool:
        """Return whether the current attempt aligned at the hover target."""
        return self._hover_alignment_completed

    def task_succeeded(self, physical_stack_succeeded: bool) -> bool:
        """Require both the final stack and an earlier genuine grasp."""
        return bool(
            physical_stack_succeeded
            and self._confirmed_grasp_seen
        )

    def reset(self, initial_state: StateSnapshot) -> None:
        """Reset event tracking for a newly reset simulation episode."""
        orange_position = np.asarray(
            initial_state["orange_position"],
            dtype=float,
        )
        if not np.all(np.isfinite(orange_position)):
            raise ValueError("initial orange position must be finite.")

        self._initial_orange_height = float(orange_position[2])
        self._bilateral_contact_seen = _has_bilateral_jaw_contact(
            initial_state
        )
        self._confirmed_grasp_seen = bool(
            initial_state["confirmed_grasp_seen"]
        )
        self._drop_penalized = False
        self._safe_lift_completed = False
        self._hover_alignment_completed = False
        self._close_attempt_start_time = None

    def calculate(
        self,
        previous_state: StateSnapshot,
        action: ArrayLike,
        action_result: CartesianActionResult, # current state derived from action result
        succeeded: bool,
    ) -> RewardResult:
        """
        Calculate the shaped reward for one environment transition.
        
        TODO: i think this has a lot of repeated calculations that the environment can own and then return in the state
        TODO: the logic in this function is also a little weird. change it so that the order of the rewards here match the order that the rewards would typically be given in
        """
        if self._initial_orange_height is None:
            raise RuntimeError(
                "StackRewardCalculator.reset() must be called before "
                "calculate()."
            )

        requested_action = np.asarray(action, dtype=float)
        expected_action_shape = (CARTESIAN_ACTION_SIZE,)
        if requested_action.shape != expected_action_shape:
            raise ValueError(
                f"action must have shape {expected_action_shape}; received "
                f"{requested_action.shape}."
            )
        if not np.all(np.isfinite(requested_action)):
            raise ValueError("action must contain only finite values.")
        applied_action = np.clip(requested_action, -1.0, 1.0)

        current_state = action_result.state
        previous_gripper_position = np.asarray(
            previous_state["gripper_position"],
            dtype=float,
        )
        current_gripper_position = np.asarray(
            current_state["gripper_position"],
            dtype=float,
        )
        previous_orange_position = np.asarray(
            previous_state["orange_position"],
            dtype=float,
        )
        current_orange_position = np.asarray(
            current_state["orange_position"],
            dtype=float,
        )
        previous_blue_position = np.asarray(
            previous_state["blue_position"],
            dtype=float,
        )
        current_blue_position = np.asarray(
            current_state["blue_position"],
            dtype=float,
        )

        positions = (
            previous_gripper_position,
            current_gripper_position,
            previous_orange_position,
            current_orange_position,
            previous_blue_position,
            current_blue_position,
        )
        for position in positions:
            if not np.all(np.isfinite(position)):
                raise ValueError(
                    "reward positions must contain finite values."
                )

        current_bilateral_contact = _has_bilateral_jaw_contact(
            current_state
        )
        current_confirmed_grasp_seen = bool(
            current_state["confirmed_grasp_seen"]
        )
        currently_confirmed_grasp = (
            current_bilateral_contact
            and not bool(current_state["orange_touches_table"])
        )
        grasp_seen_before_transition = self._confirmed_grasp_seen
        safe_lift_completed_before_transition = self._safe_lift_completed
        hover_alignment_completed_before_transition = (
            self._hover_alignment_completed
        )

        components = {
            "approach_orange_progress": 0.0,
            "grasp_candidate": 0.0,
            "grasp": 0.0,
            "lift_orange_progress": 0.0,
            "move_toward_hover_progress": 0.0,
            "lower_toward_stack_progress": 0.0,
            "stack_alignment_progress": 0.0,
            "successful_stack": 0.0,
            "dropped_cube": 0.0,
            "ik_failure": 0.0,
            "action_magnitude": 0.0,
            "gripper_state_change": 0.0,
            "unproductive_close": 0.0,
        }

        if not grasp_seen_before_transition:
            previous_distance = np.linalg.norm(
                previous_gripper_position - previous_orange_position
            )
            current_distance = np.linalg.norm(
                current_gripper_position - current_orange_position
            )
            components["approach_orange_progress"] = float(
                self.config.approach_orange_progress_weight
                * (previous_distance - current_distance)
            )

        if current_bilateral_contact and not self._bilateral_contact_seen:
            components["grasp_candidate"] = (
                self.config.grasp_candidate_reward
            )
            self._bilateral_contact_seen = True

        # grasp has occured in the state but hasn't been processed by the calculator yet
        if current_confirmed_grasp_seen and not self._confirmed_grasp_seen:
            components["grasp"] = self.config.grasp_reward
            self._confirmed_grasp_seen = True

        # A confirmed re-grasp starts a new opportunity to penalize a later
        # drop, but the one-time grasp rewards remain latched.
        if currently_confirmed_grasp and self._drop_penalized:
            self._drop_penalized = False

        # Each phase uses signed progress so reversing an earlier movement
        # pays back its shaping reward. Lift remains active through hover
        # alignment, but stops during placement so intended descent is not
        # penalized.
        transport_active = self._confirmed_grasp_seen
        if (
            transport_active
            and not hover_alignment_completed_before_transition
        ):
            initial_height = self._initial_orange_height
            maximum_lift = (
                self.success_config.expected_vertical_center_distance
                + self.config.vertical_lift_margin
            )
            previous_lift_potential = np.clip(
                previous_orange_position[2] - initial_height,
                0.0,
                maximum_lift,
            )
            current_lift_potential = np.clip(
                current_orange_position[2] - initial_height,
                0.0,
                maximum_lift,
            )
            components["lift_orange_progress"] = float(
                self.config.lift_orange_progress_weight
                * (current_lift_potential - previous_lift_potential)
            )

        previous_alignment_error = _horizontal_alignment_error(
            previous_orange_position,
            previous_blue_position,
        )
        current_alignment_error = _horizontal_alignment_error(
            current_orange_position,
            current_blue_position,
        )
        if safe_lift_completed_before_transition:
            components["stack_alignment_progress"] = float(
                self.config.stack_alignment_progress_weight
                * (previous_alignment_error - current_alignment_error)
            )

        if (
            safe_lift_completed_before_transition
            and not hover_alignment_completed_before_transition
        ):
            previous_hover_distance = _hover_target_distance(
                previous_orange_position,
                previous_blue_position,
                self.success_config.expected_vertical_center_distance,
                self.config.vertical_lift_margin,
            )
            current_hover_distance = _hover_target_distance(
                current_orange_position,
                current_blue_position,
                self.success_config.expected_vertical_center_distance,
                self.config.vertical_lift_margin,
            )
            components["move_toward_hover_progress"] = float(
                self.config.move_toward_hover_progress_weight
                * (previous_hover_distance - current_hover_distance)
            )

        if hover_alignment_completed_before_transition:
            previous_stack_distance = _stack_target_distance(
                previous_orange_position,
                previous_blue_position,
                self.success_config.expected_vertical_center_distance,
            )
            current_stack_distance = _stack_target_distance(
                current_orange_position,
                current_blue_position,
                self.success_config.expected_vertical_center_distance,
            )
            components["lower_toward_stack_progress"] = float(
                self.config.lower_toward_stack_progress_weight
                * (previous_stack_distance - current_stack_distance)
            )

        cube_fell_off_table = bool(
            current_state["orange_fell_off_table"]
        ) or bool(
            current_state["blue_fell_off_table"]
        )
        task_succeeded = (
            self.task_succeeded(succeeded)
            and not cube_fell_off_table
        )
        dropped_onto_table_after_grasp = (
            self._confirmed_grasp_seen
            and not current_bilateral_contact
            and bool(current_state["orange_touches_table"])
            and not task_succeeded
        )
        drop_detected = (
            cube_fell_off_table or dropped_onto_table_after_grasp
        )
        if (
            drop_detected
            and not self._drop_penalized
        ):
            components["dropped_cube"] = self.config.dropped_cube_penalty
            self._drop_penalized = True

        # A recoverable tabletop drop starts a new transport attempt. The
        # large drop penalty prevents repeatedly resetting these phases from
        # becoming a profitable reward cycle.
        if drop_detected:
            self._safe_lift_completed = False
            self._hover_alignment_completed = False
        elif currently_confirmed_grasp:
            hover_height = (
                current_blue_position[2]
                + self.success_config.expected_vertical_center_distance
                + self.config.vertical_lift_margin
            )
            height_is_safe = current_orange_position[2] >= hover_height

            if (
                safe_lift_completed_before_transition
                and height_is_safe
                and current_alignment_error
                <= self.success_config.max_horizontal_center_offset
                + self.success_config.floating_point_numerical_tolerance
            ):
                self._hover_alignment_completed = True

            if height_is_safe:
                self._safe_lift_completed = True

        if not action_result.ik_result.position_converged:
            components["ik_failure"] = self.config.ik_failure_penalty

        # The fourth action switches or retains a persistent gripper target;
        # its magnitude is not physical motion. Penalize only Cartesian
        # motion deltas here.
        components["action_magnitude"] = float(
            self.config.action_magnitude_penalty_weight
            * np.mean(np.square(applied_action[:3]))
        )

        previous_gripper_target = float(previous_state["gripper_target"])
        current_gripper_target = float(current_state["gripper_target"])
        if previous_gripper_target != current_gripper_target:
            components["gripper_state_change"] = (
                self.config.gripper_state_change_penalty
            )

        closed_during_transition = (
            current_gripper_target < previous_gripper_target
        )
        opened_during_transition = (
            current_gripper_target > previous_gripper_target
        )
        previous_time = float(previous_state["time"])
        current_time = float(current_state["time"])
        if closed_during_transition:
            # The target changes before this transition's physics steps, so
            # the attempt begins at the previous state's simulation time.
            self._close_attempt_start_time = previous_time

        if self._close_attempt_start_time is not None:
            close_attempt_duration = (
                current_time - self._close_attempt_start_time
            )
            grace_period_reached = (
                close_attempt_duration
                >= self.config.close_contact_grace_period
            )
            contact_within_grace_period = (
                current_bilateral_contact
                and close_attempt_duration
                <= self.config.close_contact_grace_period
            )

            if contact_within_grace_period:
                self._close_attempt_start_time = None
            elif opened_during_transition or grace_period_reached:
                components["unproductive_close"] = (
                    self.config.unproductive_close_penalty
                )
                self._close_attempt_start_time = None

        if task_succeeded:
            components["successful_stack"] = (
                self.config.successful_stack_reward
            )

        return RewardResult(
            total=float(sum(components.values())),
            components=components,
        )
