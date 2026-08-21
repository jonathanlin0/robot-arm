from dataclasses import dataclass

from numpy.typing import ArrayLike

from cartesian_actions import CartesianActionResult
from environment import StateSnapshot


@dataclass(frozen=True)
class StackRewardConfig:
    """Provisional weights for the shaped stacking reward."""

    approach_orange_progress_weight: float = 1.0
    grasp_reward: float = 5.0
    lift_orange_progress_weight: float = 5.0
    move_toward_blue_progress_weight: float = 2.0
    stack_alignment_progress_weight: float = 5.0
    successful_stack_reward: float = 100.0
    dropped_cube_penalty: float = -10.0
    ik_failure_penalty: float = -1.0
    action_magnitude_penalty_weight: float = 0.01


@dataclass(frozen=True)
class RewardResult:
    """Total reward plus named components for logging and debugging."""

    total: float
    components: dict[str, float]


def calculate_stack_reward(
    previous_state: StateSnapshot,
    action: ArrayLike,
    action_result: CartesianActionResult,
    succeeded: bool,
    config: StackRewardConfig,
) -> RewardResult:
    """Calculate the reward for one environment transition."""
    raise NotImplementedError(
        "Stack reward calculation has not been implemented yet."
    )
