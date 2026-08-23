from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from gym_environment import CubeStackGymEnvironment


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Top-level settings for the first privileged-state PPO run."""

    seed: int = 0
    total_timesteps: int = 1_000_000
    maximum_episode_steps: int = 400
    learning_rate: float = 3e-4
    rollout_steps: int = 2_048
    batch_size: int = 64
    training_epochs: int = 10
    clip_range: float = 0.2
    device: str = "cpu"
    checkpoint_path: Path = Path("checkpoints/ppo_cube_stacker")


def create_training_environment(
    config: PPOTrainingConfig,
) -> CubeStackGymEnvironment:
    """Create the Gymnasium environment used by one PPO run."""
    return CubeStackGymEnvironment(
        seed=config.seed,
        maximum_episode_steps=config.maximum_episode_steps,
    )


def train_ppo(config: PPOTrainingConfig | None = None) -> PPO:
    """Validate the Gym environment, train PPO, and save the policy."""
    training_config = config or PPOTrainingConfig()
    environment = create_training_environment(training_config)

    try:
        check_env(environment, warn=True)

        model = PPO(
            policy="MlpPolicy",
            env=environment,
            learning_rate=training_config.learning_rate,
            n_steps=training_config.rollout_steps,
            batch_size=training_config.batch_size,
            n_epochs=training_config.training_epochs,
            clip_range=training_config.clip_range,
            seed=training_config.seed,
            device=training_config.device,
            verbose=1,
        )
        model.learn(total_timesteps=training_config.total_timesteps)

        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_path)

        return model
    finally:
        environment.close()


def main() -> None:
    train_ppo()


if __name__ == "__main__":
    main()
