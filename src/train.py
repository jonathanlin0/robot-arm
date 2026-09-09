import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
import math
from pathlib import Path
from typing import Any

import torch
import wandb
from scipy.stats import linregress
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from wandb import Run

from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy
from gym_environment import CubeStackGymEnvironment
from rewards import StackRewardConfig


ROLLOUT_METRIC_FILE_NAMES = {
    "ep_len_mean": "ep_len_mean.txt",
    "ep_rew_mean": "ep_rew_mean.txt",
    "success_rate": "success_rate.txt",
    "orange_currently_held": "orange_currently_held.txt",
    "orange_grasp_hold_time": "orange_grasp_hold_time.txt",
}
GRIPPER_ACTION_INDEX = 3
# how many rollouts of data we want to keep track of
PROMISING_RUN_ROLLOUT_WINDOW = 300
# how many timesteps we want to extend the experiment for if mean reward or success rate is increasing
# in the last PROMISING_RUN_ROLLOUT_WINDOW of rollouts
PROMISING_RUN_EXTENSION_TIMESTEPS = 300_000

WANDB_ENTITY_NAME = "jonathanlin"
WANDB_PROJECT_NAME = "robotics-pick_up_cube"
REWARD_PARAMETER_PREFIX = "reward_"
MODEL_PARAMETER_PREFIX = "model_"
WANDB_SWEEP_CONFIG: dict[str, Any] = {
    "name": "reward-and-model-bayesian-optimization",
    "program": "src/train.py",
    "method": "bayes",
    "metric": {
        "name": "success_rate",
        "goal": "maximize",
    },
    "parameters": {
        "reward_approach_orange_progress_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 2.0,
        },
        "reward_grasp_candidate_reward": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 2.0,
        },
        "reward_grasp_reward": {
            "distribution": "uniform",
            "min": 2.5,
            "max": 10.0,
        },
        "reward_ik_failure_penalty": {
            "distribution": "uniform",
            "min": -2.0,
            "max": -0.1,
        },
        "reward_action_magnitude_penalty_weight": {
            "distribution": "uniform",
            "min": -0.001,
            "max": 0.0,
        },
        "reward_gripper_state_change_penalty": {
            "distribution": "uniform",
            "min": -0.001,
            "max": 0.0,
        },
        "reward_unproductive_close_penalty": {
            "distribution": "uniform",
            "min": -0.05,
            "max": 0.00,
        },
        "model_dim": {
            "values": [64, 128, 256, 512],
        },
        "model_layers": {
            "values": [1, 2, 3, 4],
        },
        "model_learning_rate": {
            "distribution": "uniform",
            "min": 1e-4,
            "max": 1e-3,
        },
    },
}


class RolloutMetricsCallback(BaseCallback):
    """Write episode metrics and per-rollout pickup diagnostics."""

    def __init__(
        self,
        metrics_directory: Path | str,
        wandb_run: Run | None = None,
    ) -> None:
        super().__init__()
        self.metrics_directory = Path(metrics_directory)
        self.wandb_run = wandb_run
        self.rollout_count = 0
        self.recent_episode_reward_means: deque[float] = deque(
            maxlen=PROMISING_RUN_ROLLOUT_WINDOW
        )
        self.recent_success_rates: deque[float] = deque(
            maxlen=PROMISING_RUN_ROLLOUT_WINDOW
        )
        self._rollout_info_sample_count = 0
        self._rollout_orange_held_sample_count = 0
        self._rollout_max_orange_grasp_hold_time = 0.0
        self._training_has_started = False

    def _reset_rollout_pickup_metrics(self) -> None:
        """Reset pickup summaries collected from per-step environment info."""
        self._rollout_info_sample_count = 0
        self._rollout_orange_held_sample_count = 0
        self._rollout_max_orange_grasp_hold_time = 0.0

    def _on_training_start(self) -> None:
        """Create empty metric files for this new training run."""
        if self._training_has_started:
            return

        self._training_has_started = True
        self.rollout_count = 0
        self.recent_episode_reward_means.clear()
        self.recent_success_rates.clear()
        self._reset_rollout_pickup_metrics()
        self.metrics_directory.mkdir(parents=True, exist_ok=True)
        for file_name in ROLLOUT_METRIC_FILE_NAMES.values():
            (self.metrics_directory / file_name).write_text(
                "",
                encoding="utf-8",
            )

    def _on_rollout_start(self) -> None:
        """Start fresh pickup summaries for the next rollout."""
        self._reset_rollout_pickup_metrics()

    def _on_step(self) -> bool:
        """Collect pickup state and duration from every vector worker."""
        infos = self.locals.get("infos", ())
        for info in infos:
            orange_currently_held = bool(info["orange_currently_held"])
            orange_grasp_hold_time = float(
                info["orange_grasp_hold_time"]
            )

            self._rollout_info_sample_count += 1
            if orange_currently_held:
                self._rollout_orange_held_sample_count += 1
            self._rollout_max_orange_grasp_hold_time = max(
                self._rollout_max_orange_grasp_hold_time,
                orange_grasp_hold_time,
            )

        return True

    def _on_rollout_end(self) -> None:
        """Append episode metrics and this rollout's pickup diagnostics."""
        assert self.model.ep_info_buffer is not None
        assert self.model.ep_success_buffer is not None

        episode_information = list(self.model.ep_info_buffer)
        episode_reward_mean = float(
            safe_mean(
                [episode["r"] for episode in episode_information]
            )
        )
        success_rate = float(
            safe_mean(list(self.model.ep_success_buffer))
        )
        if self._rollout_info_sample_count:
            orange_currently_held = float(
                self._rollout_orange_held_sample_count
                / self._rollout_info_sample_count
            )
            orange_grasp_hold_time = (
                self._rollout_max_orange_grasp_hold_time
            )
        else:
            orange_currently_held = float("nan")
            orange_grasp_hold_time = float("nan")
        self.recent_episode_reward_means.append(episode_reward_mean)
        self.recent_success_rates.append(success_rate)

        metric_values = {
            "ep_len_mean": safe_mean(
                [episode["l"] for episode in episode_information]
            ),
            "ep_rew_mean": episode_reward_mean,
            "success_rate": success_rate,
            # These summarize the step-level info across every environment
            # in this rollout. A nonzero held fraction proves that at least
            # one bilateral, off-table hold occurred; the duration retains
            # the longest uninterrupted hold even if orange was later dropped.
            "orange_currently_held": orange_currently_held,
            "orange_grasp_hold_time": orange_grasp_hold_time,
        }

        for metric_name, value in metric_values.items():
            metric_path = (
                self.metrics_directory
                / ROLLOUT_METRIC_FILE_NAMES[metric_name]
            )
            with metric_path.open("a", encoding="utf-8") as metric_file:
                metric_file.write(f"{float(value)!r}\n")

        self.rollout_count += 1
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    **{
                        name: float(value)
                        for name, value in metric_values.items()
                    },
                    "rollout": self.rollout_count,
                    "total_timesteps": self.num_timesteps,
                },
                step=self.rollout_count,
            )


class GripperStandardDeviationFloorCallback(BaseCallback):
    """Keep stochastic exploration in the gripper action dimension."""

    def __init__(self, minimum_standard_deviation: float) -> None:
        super().__init__()
        if (
            not math.isfinite(minimum_standard_deviation)
            or minimum_standard_deviation <= 0.0
        ):
            raise ValueError(
                "minimum gripper standard deviation must be finite and "
                "greater than zero."
            )

        self.minimum_standard_deviation = minimum_standard_deviation
        self.minimum_log_standard_deviation = math.log(
            minimum_standard_deviation
        )

    def _apply_floor(self) -> None:
        """Clamp the learned gripper log standard deviation in place."""
        log_standard_deviations = getattr(
            self.model.policy,
            "log_std",
            None,
        )
        if (
            not isinstance(log_standard_deviations, torch.Tensor)
            or log_standard_deviations.ndim != 1
            or len(log_standard_deviations) <= GRIPPER_ACTION_INDEX
        ):
            raise RuntimeError(
                "The gripper standard-deviation floor requires the "
                "ordinary one-dimensional Gaussian log_std used when "
                "state-dependent exploration is disabled."
            )

        with torch.no_grad():
            log_standard_deviations[GRIPPER_ACTION_INDEX].clamp_(
                min=self.minimum_log_standard_deviation
            )

    def _on_rollout_start(self) -> None:
        """Apply the floor after the previous update and before sampling."""
        self._apply_floor()

    def _on_step(self) -> bool:
        """Continue collecting the current rollout."""
        return True

    def _on_training_end(self) -> None:
        """Ensure the final saved policy also respects the floor."""
        self._apply_floor()


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Top-level settings for the first privileged-state PPO run."""

    seed: int = 0
    total_timesteps: int = 1_000_000
    maximum_episode_steps: int = 400
    environment_count: int = 4
    learning_rate: float = 3e-4
    model_dim: int = 128
    model_layers: int = 2
    rollout_steps: int = 512
    batch_size: int = 64
    training_epochs: int = 10
    clip_range: float = 0.2
    entropy_coefficient: float = 0.0001
    target_kl: float = 0.03
    log_std_init: float = math.log(0.2)
    use_expln: bool = True
    minimum_gripper_standard_deviation: float = 0.5
    use_state_dependent_exploration: bool = True
    exploration_noise_resample_steps: int = 8
    device: str = "cpu"
    checkpoint_path: Path = Path("checkpoints/ppo_cube_stacker")
    metrics_directory: Path = Path("data")
    reward_config: StackRewardConfig = field(
        default_factory=StackRewardConfig
    )


def create_reward_config(
    overrides: Mapping[str, object] | None = None,
) -> StackRewardConfig:
    """Create the default reward config with selected fields overridden.

    Override names must exactly match fields on ``StackRewardConfig``. This
    lets the sweep's parameter dictionary be edited without changing the
    training code that applies its sampled values.
    """
    default_config = StackRewardConfig()
    if overrides is None:
        return default_config

    valid_names = {
        config_field.name
        for config_field in fields(StackRewardConfig)
        if config_field.init
    }
    unknown_names = set(overrides) - valid_names
    if unknown_names:
        unknown_names_text = ", ".join(sorted(unknown_names))
        raise ValueError(
            "Unknown StackRewardConfig override(s): "
            f"{unknown_names_text}."
        )

    return replace(default_config, **dict(overrides))


def create_validation_environment(
    config: PPOTrainingConfig,
) -> CubeStackGymEnvironment:
    """Create one ordinary Gymnasium environment for API validation."""
    return CubeStackGymEnvironment(
        seed=config.seed,
        maximum_episode_steps=config.maximum_episode_steps,
        reward_config=config.reward_config,
    )


def create_training_environment(config: PPOTrainingConfig) -> VecEnv:
    """Create monitored MuJoCo environments in separate processes."""
    return make_vec_env(
        env_id=CubeStackGymEnvironment,
        n_envs=config.environment_count,
        seed=config.seed,
        env_kwargs={
            "maximum_episode_steps": config.maximum_episode_steps,
            "reward_config": config.reward_config,
        },
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )


def train_ppo(
    config: PPOTrainingConfig | None = None,
    *,
    wandb_run: Run | None = None,
) -> PPO:
    """Validate the Gym environment, train PPO, and save the policy."""
    training_config = config or PPOTrainingConfig()
    validation_environment = create_validation_environment(training_config)

    try:
        check_env(validation_environment, warn=True)
    finally:
        validation_environment.close()

    environment = create_training_environment(training_config)

    try:
        model = PPO(
            policy=TanhBoundedMeanActorCriticPolicy,
            env=environment,
            learning_rate=training_config.learning_rate,
            n_steps=training_config.rollout_steps,
            batch_size=training_config.batch_size,
            n_epochs=training_config.training_epochs,
            clip_range=training_config.clip_range,
            ent_coef=training_config.entropy_coefficient,
            target_kl=training_config.target_kl,
            use_sde=training_config.use_state_dependent_exploration,
            sde_sample_freq=(
                training_config.exploration_noise_resample_steps
            ),
            policy_kwargs={
                "net_arch": {
                    "pi": [training_config.model_dim] * training_config.model_layers,
                    "vf": [training_config.model_dim] * training_config.model_layers,
                },
                "log_std_init": training_config.log_std_init,
                "use_expln": training_config.use_expln,
                # Only the Gaussian mean is tanh-bounded by the custom policy.
                # Keeping gSDE samples unsquashed gives PPO an analytical
                # entropy; SB3 still clips noisy samples to the normalized
                # action bounds before they reach the environment.
                "squash_output": False,
            },
            seed=training_config.seed,
            device=training_config.device,
            verbose=1,
        )
        metrics_callback = RolloutMetricsCallback(
            training_config.metrics_directory,
            wandb_run=wandb_run,
        )
        training_callbacks_list: list[BaseCallback] = [metrics_callback]
        if not training_config.use_state_dependent_exploration:
            # The ordinary diagonal Gaussian has one log standard deviation
            # per action dimension, so its gripper entry can be clamped
            # directly. gSDE instead has a latent-feature-by-action noise
            # matrix; clamping that matrix column would not impose the same
            # effective gripper standard-deviation floor.
            training_callbacks_list.insert(
                0,
                GripperStandardDeviationFloorCallback(
                    training_config.minimum_gripper_standard_deviation
                ),
            )
        training_callbacks = CallbackList(training_callbacks_list)
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=training_callbacks,
        )

        if (
            len(metrics_callback.recent_episode_reward_means)
            >= PROMISING_RUN_ROLLOUT_WINDOW
            and len(metrics_callback.recent_success_rates)
            >= PROMISING_RUN_ROLLOUT_WINDOW
        ):
            rollout_indices = range(PROMISING_RUN_ROLLOUT_WINDOW)
            episode_reward_slope = float(
                linregress(
                    rollout_indices,
                    metrics_callback.recent_episode_reward_means,
                ).slope
            )
            success_rate_slope = float(
                linregress(
                    rollout_indices,
                    metrics_callback.recent_success_rates,
                ).slope
            )
            print(
                "Recent rollout slopes: "
                f"ep_rew_mean={episode_reward_slope:+.8f}, "
                f"success_rate={success_rate_slope:+.8f}"
            )

            if episode_reward_slope > 0.0 or success_rate_slope > 0.0:
                print(
                    "Promising trend detected; extending training by "
                    f"{PROMISING_RUN_EXTENSION_TIMESTEPS:,} timesteps."
                )
                model.learn(
                    total_timesteps=PROMISING_RUN_EXTENSION_TIMESTEPS,
                    callback=training_callbacks,
                    reset_num_timesteps=False,
                )

        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_path)

        return model
    finally:
        environment.close()


def _validate_sweep_parameter_names() -> None:
    """Fail early if a sweep parameter has no supported destination."""
    parameter_names = set(WANDB_SWEEP_CONFIG["parameters"])
    reward_field_names = {
        config_field.name
        for config_field in fields(StackRewardConfig)
        if config_field.init
    }
    model_field_names = {"dim", "layers", "learning_rate"}
    unknown_names = set()
    for parameter_name in parameter_names:
        if parameter_name.startswith(REWARD_PARAMETER_PREFIX):
            stripped_name = parameter_name.removeprefix(
                REWARD_PARAMETER_PREFIX
            )
            if stripped_name not in reward_field_names:
                unknown_names.add(parameter_name)
        elif parameter_name.startswith(MODEL_PARAMETER_PREFIX):
            stripped_name = parameter_name.removeprefix(
                MODEL_PARAMETER_PREFIX
            )
            if stripped_name not in model_field_names:
                unknown_names.add(parameter_name)
        else:
            unknown_names.add(parameter_name)

    if unknown_names:
        unknown_names_text = ", ".join(sorted(unknown_names))
        raise ValueError(
            "Unsupported W&B sweep parameter(s): "
            f"{unknown_names_text}."
        )


def run_wandb_trial() -> None:
    """Train one policy using the reward and model values sampled by W&B."""
    default_training_config = PPOTrainingConfig()
    default_wandb_values = {
        f"{REWARD_PARAMETER_PREFIX}{name}": value
        for name, value in asdict(StackRewardConfig()).items()
    }
    default_wandb_values.update(
        {
            "model_dim": default_training_config.model_dim,
            "model_layers": default_training_config.model_layers,
            "model_learning_rate": default_training_config.learning_rate,
        }
    )

    with wandb.init(
        project=WANDB_PROJECT_NAME,
        config=default_wandb_values,
    ) as run:
        sampled_values = {
            parameter_name: run.config[parameter_name]
            for parameter_name in WANDB_SWEEP_CONFIG["parameters"]
        }
        sampled_reward_values = {
            parameter_name.removeprefix(REWARD_PARAMETER_PREFIX): value
            for parameter_name, value in sampled_values.items()
            if parameter_name.startswith(REWARD_PARAMETER_PREFIX)
        }
        sampled_model_values = {
            parameter_name.removeprefix(MODEL_PARAMETER_PREFIX): value
            for parameter_name, value in sampled_values.items()
            if parameter_name.startswith(MODEL_PARAMETER_PREFIX)
        }
        reward_config = create_reward_config(sampled_reward_values)
        run_id = str(run.id)
        training_config = PPOTrainingConfig(
            learning_rate=float(
                sampled_model_values.get(
                    "learning_rate",
                    default_training_config.learning_rate,
                )
            ),
            model_dim=int(
                sampled_model_values.get(
                    "dim",
                    default_training_config.model_dim,
                )
            ),
            model_layers=int(
                sampled_model_values.get(
                    "layers",
                    default_training_config.model_layers,
                )
            ),
            checkpoint_path=(
                Path("checkpoints")
                / "wandb"
                / run_id
                / "ppo_cube_stacker"
            ),
            metrics_directory=Path("data") / "wandb" / run_id,
            reward_config=reward_config,
        )
        train_ppo(training_config, wandb_run=run)


def run_wandb_sweep() -> None:
    """Create an unbounded Bayesian sweep and run its trials locally."""
    _validate_sweep_parameter_names()
    sweep_id = wandb.sweep(
        sweep=WANDB_SWEEP_CONFIG,
        project=WANDB_PROJECT_NAME,
    )
    # With no count, the W&B agent keeps running trials until interrupted.
    wandb.agent(sweep_id, function=run_wandb_trial)


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line options for ordinary or sweep training."""
    parser = argparse.ArgumentParser()
    training_mode = parser.add_mutually_exclusive_group()
    training_mode.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "run an indefinite Bayesian W&B sweep that maximizes "
            "success rate"
        ),
    )
    training_mode.add_argument(
        "--repeat-wandb",
        metavar="RUN_ID",
        help="train a fresh policy using a W&B run's reward configuration",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    arguments_namespace = parse_arguments(arguments)
    if arguments_namespace.wandb:
        run_wandb_sweep()
    elif arguments_namespace.repeat_wandb is not None:
        source_run = wandb.Api().run(
            f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/"
            f"{arguments_namespace.repeat_wandb}"
        )
        reward_field_names = {
            config_field.name
            for config_field in fields(StackRewardConfig)
            if config_field.init
        }
        saved_reward_values = {
            name.removeprefix(REWARD_PARAMETER_PREFIX): value
            for name, value in source_run.config.items()
            if name.startswith(REWARD_PARAMETER_PREFIX)
            and name.removeprefix(REWARD_PARAMETER_PREFIX)
            in reward_field_names
        }
        saved_model_values = {
            name.removeprefix(MODEL_PARAMETER_PREFIX): value
            for name, value in source_run.config.items()
            if name.startswith(MODEL_PARAMETER_PREFIX)
        }
        reward_config = create_reward_config(saved_reward_values)
        default_training_config = PPOTrainingConfig()
        train_ppo(
            PPOTrainingConfig(
                learning_rate=float(
                    saved_model_values.get(
                        "learning_rate",
                        default_training_config.learning_rate,
                    )
                ),
                model_dim=int(
                    saved_model_values.get(
                        "dim",
                        default_training_config.model_dim,
                    )
                ),
                model_layers=int(
                    saved_model_values.get(
                        "layers",
                        default_training_config.model_layers,
                    )
                ),
                reward_config=reward_config,
            )
        )
    else:
        train_ppo()


if __name__ == "__main__":
    main()
