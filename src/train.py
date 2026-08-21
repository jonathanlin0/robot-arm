from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from gym_environment import CubeStackGymEnvironment


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Top-level settings for the first privileged-state PPO run."""

    seed: int = 0
    total_timesteps: int = 1_000_000
    maximum_episode_steps: int = 400
    checkpoint_path: Path = Path("checkpoints/ppo_cube_stacker")


def create_training_environment(
    config: PPOTrainingConfig,
) -> CubeStackGymEnvironment:
    """Create the Gymnasium environment used by one PPO run."""
    raise NotImplementedError(
        "Training environment creation has not been implemented yet."
    )


def train_ppo(config: PPOTrainingConfig | None = None) -> PPO:
    """Validate the Gym environment, train PPO, and save the policy."""
    raise NotImplementedError(
        "PPO training setup has not been implemented yet."
    )


def main() -> None:
    train_ppo()


if __name__ == "__main__":
    main()
