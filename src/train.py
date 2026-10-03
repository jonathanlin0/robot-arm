import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
import math
from pathlib import Path
from typing import Any

import numpy as np
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

from action_observation_history import ActionObservationHistoryWrapper
from bounded_mean_policy import TanhBoundedMeanActorCriticPolicy
from gym_environment import CubeStackGymEnvironment
from rewards import StackRewardConfig
from temporal_features import ActionObservationTransformer
from waypoint_start import RecoveryStartConfig


ROLLOUT_METRIC_FILE_NAMES = {
    "ep_len_mean": "ep_len_mean.txt",
    "ep_rew_mean": "ep_rew_mean.txt",
    "success_rate": "success_rate.txt",
    "no_var_success_rate": "no_var_success_rate.txt",
    "orange_waypoint_reach_rate": "orange_waypoint_reach_rate.txt",
    "orange_currently_held": "orange_currently_held.txt",
    "orange_grasp_hold_time": "orange_grasp_hold_time.txt",
    "action_std_x": "action_std_x.txt",
    "action_std_y": "action_std_y.txt",
    "action_std_z": "action_std_z.txt",
    "action_std_gripper": "action_std_gripper.txt",
    "action_clip_fraction": "action_clip_fraction.txt",
}
ACTION_STD_METRICS = (
    "action_std_x",
    "action_std_y",
    "action_std_z",
    "action_std_gripper",
)
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
MODEL_CONFIG_FIELDS = {
    "dim": "model_dim",
    "layers": "model_layers",
    "learning_rate": "learning_rate",
    "target_kl": "target_kl",
    "batch_size": "batch_size",
    "history_length": "history_length",
    "transformer_embedding_dim": "transformer_embedding_dim",
    "transformer_layers": "transformer_layers",
    "transformer_heads": "transformer_heads",
    "transformer_feedforward_dim": "transformer_feedforward_dim",
    "sde_xyz_log_std_init": "sde_xyz_log_std_init",
    "sde_gripper_log_std_init": "sde_gripper_log_std_init",
}
# Saved behavior-cloning metadata uses actor_* for the actor MLP dimensions.
PRETRAINING_ARCHITECTURE_FIELDS = {
    "history_length": "history_length",
    "transformer_embedding_dim": "transformer_embedding_dim",
    "transformer_layers": "transformer_layers",
    "transformer_heads": "transformer_heads",
    "transformer_feedforward_dim": "transformer_feedforward_dim",
    "actor_dim": "model_dim",
    "actor_layers": "model_layers",
}
WANDB_SWEEP_CONFIG: dict[str, Any] = {
    "name": "pickup-ppo-exploration",
    "program": "src/train.py",
    "method": "bayes",
    "metric": {
        "name": "success_rate",
        "goal": "maximize",
    },
    "parameters": {
        # Search PPO update size, exploration amplitude, and transformer capacity.
        # Omitted settings use PPOTrainingConfig / StackRewardConfig defaults.
        "model_learning_rate": {
            # Bounds are learning rates, not their logarithms.
            "distribution": "log_uniform_values",
            "min": 2e-5,
            "max": 1e-4,
        },
        "model_target_kl": {
            "distribution": "uniform",
            "min": 0.005,
            "max": 0.03,
        },
        "model_batch_size": {
            "values": [256, 512],
        },
        "model_sde_xyz_log_std_init": {
            # These parameters are already logs: use uniform, not log_uniform.
            # Effective XYZ SD is roughly 0.1..0.53 at noise feature norm 7.1.
            "distribution": "uniform",
            "min": math.log(0.1 / 7.1),
            "max": -2.6,
        },
        "model_sde_gripper_log_std_init": {
            # Higher gripper noise helps cross its open/close thresholds.
            # Effective SD is roughly 0.35..0.75 at noise feature norm 7.1.
            "distribution": "uniform",
            "min": math.log(0.35 / 7.1),
            "max": math.log(0.75 / 7.1),
        },
        "model_transformer_embedding_dim": {
            # Embedding width must be divisible by the default four heads.
            "distribution": "q_uniform",
            "min": 128,
            "max": 160,
            "q": 4,
        },
        "model_transformer_feedforward_dim": {
            "distribution": "int_uniform",
            "min": 512,
            "max": 768,
        },
        "model_transformer_layers": {
            "distribution": "int_uniform",
            "min": 3,
            "max": 4,
        },
        "reward_approach_orange_progress_weight": {
            "values": [2, 0],
        },
    },
}


class RolloutMetricsCallback(BaseCallback):
    """Write episode metrics and per-rollout pickup diagnostics."""

    def __init__(
        self,
        metrics_directory: Path | str,
        wandb_run: Run | None = None,
        *,
        evaluation_config: "PPOTrainingConfig | None" = None,
    ) -> None:
        super().__init__()
        self.metrics_directory = Path(metrics_directory)
        self.wandb_run = wandb_run
        self.evaluation_config = evaluation_config
        self.rollout_count = 0
        self.recent_episode_reward_means: deque[float] = deque(
            maxlen=PROMISING_RUN_ROLLOUT_WINDOW
        )
        self.recent_success_rates: deque[float] = deque(
            maxlen=PROMISING_RUN_ROLLOUT_WINDOW
        )
        self._training_has_started = False
        self._reset_rollout_pickup_metrics()

    def _reset_rollout_pickup_metrics(self) -> None:
        """Reset pickup and action-distribution summaries for one rollout."""
        self._rollout_info_sample_count = 0
        self._rollout_orange_held_sample_count = 0
        self._rollout_max_orange_grasp_hold_time = 0.0
        self._action_std_sum = np.zeros(4, dtype=np.float64)
        self._action_std_count = 0
        self._clipped_action_count = 0
        self._action_component_count = 0

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
        # The distribution is still the one used by policy.forward() for
        # this action. Read it here instead of running the transformer again.
        policy = getattr(self.model, "policy", None)
        action_distribution = getattr(policy, "action_dist", None)
        distribution = getattr(action_distribution, "distribution", None)
        if distribution is not None:
            standard_deviations = distribution.stddev.detach().cpu().numpy()
            self._action_std_sum += standard_deviations.sum(axis=0)
            self._action_std_count += len(standard_deviations)
        actions = self.locals.get("actions")
        clipped_actions = self.locals.get("clipped_actions")
        if actions is not None and clipped_actions is not None:
            self._clipped_action_count += int(
                np.count_nonzero(actions != clipped_actions)
            )
            self._action_component_count += np.asarray(actions).size

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
        # Monitor records the episode's latched waypoint flag at termination
        # or truncation, so each completed episode contributes exactly once.
        orange_waypoint_reach_rate = float(
            safe_mean(
                [
                    episode["orange_pregrasp_waypoint_reached"]
                    for episode in episode_information
                ]
            )
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
            "no_var_success_rate": float("nan"),
            "orange_waypoint_reach_rate": orange_waypoint_reach_rate,
            # These summarize the step-level info across every environment
            # in this rollout. A nonzero held fraction proves that at least
            # one bilateral, off-table hold occurred; the duration retains
            # the longest uninterrupted hold even if orange was later dropped.
            "orange_currently_held": orange_currently_held,
            "orange_grasp_hold_time": orange_grasp_hold_time,
            **{
                name: (
                    float(self._action_std_sum[index] / self._action_std_count)
                    if self._action_std_count
                    else float("nan")
                )
                for index, name in enumerate(ACTION_STD_METRICS)
            },
            "action_clip_fraction": (
                self._clipped_action_count / self._action_component_count
                if self._action_component_count
                else float("nan")
            ),
        }

        self.rollout_count += 1
        # Record an initial baseline before the first PPO update, then on schedule.
        if (
            self.evaluation_config is not None
            and (
                self.rollout_count == 1
                or self.rollout_count
                % self.evaluation_config.evaluation_interval_rollouts == 0
            )
        ):
            print(
                f"Rollout {self.rollout_count}: evaluating no_var_success_rate "
                f"over {self.evaluation_config.evaluation_episodes} sequential episodes...",
                flush=True,
            )
            metric_values["no_var_success_rate"] = evaluate_no_var_success_rate(
                self.model, self.evaluation_config,
            )

        for metric_name, value in metric_values.items():
            metric_path = (
                self.metrics_directory
                / ROLLOUT_METRIC_FILE_NAMES[metric_name]
            )
            with metric_path.open("a", encoding="utf-8") as metric_file:
                metric_file.write(f"{float(value)!r}\n")

        print(
            f"ep_len_mean={metric_values['ep_len_mean']:.2f} "
            f"ep_rew_mean={episode_reward_mean:.4f} "
            f"success_rate={success_rate:.4f}",
            flush=True,
        )
        if math.isfinite(metric_values["no_var_success_rate"]):
            print(f"no_var_success_rate={metric_values['no_var_success_rate']:.4f}", flush=True)
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    **{
                        name: float(value)
                        for name, value in metric_values.items()
                        if name != "no_var_success_rate" or math.isfinite(value)
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
    """Pickup PPO settings, including temporal context and model sizes."""

    seed: int = 0
    total_timesteps: int = 2_500_000
    maximum_episode_steps: int = 400
    start_at_orange_waypoint: bool = True # TEMP
    # When waypoint preparation is enabled, sample a recovery start this often.
    recovery_start_probability: float = 0
    recovery_xy_offset_range: tuple[float, float] = (0.03, 0.05)
    recovery_height_offset_range: tuple[float, float] = (0.03, 0.05)
    recovery_closed_gripper_probability: float = 0.5
    environment_count: int = 8
    learning_rate: float = 3e-5
    history_length: int = 384
    transformer_embedding_dim: int = 128
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_feedforward_dim: int = 512
    # Separate actor, value, and exploration MLPs use these hidden-layer sizes.
    model_dim: int = 128
    model_layers: int = 2
    rollout_steps: int = 512
    batch_size: int = 256
    training_epochs: int = 10
    # Serial deterministic evaluation on the same clean scene seeds each time.
    evaluation_interval_rollouts: int = 25
    evaluation_episodes: int = 100
    evaluation_seed: int = 20_000
    clip_range: float = 0.2
    entropy_coefficient: float = 0.0001
    target_kl: float = 0.03
    # Ordinary Gaussian initialization when state-dependent exploration is off.
    log_std_init: float = math.log(0.2)
    # gSDE noise-weight log SDs, calibrated at a reference noise-feature norm
    # of 7.1. Actual action SD depends on the tanh-bounded exploration features.
    # These initialize trainable parameters; they do not impose SD limits.
    sde_xyz_log_std_init: float = math.log(0.2 / 7.1)
    sde_gripper_log_std_init: float = math.log(0.5 / 7.1)
    use_expln: bool = True
    minimum_gripper_standard_deviation: float = 0.5
    use_state_dependent_exploration: bool = True
    exploration_noise_resample_steps: int = 8
    device: str = "cpu"
    # Optional behavior-cloning warm start; PPO keeps its own optimizer,
    # exploration settings, critic, rewards, and environment configuration.
    pretrained_checkpoint: Path | None = None
    checkpoint_path: Path = Path("checkpoints/ppo_cube_stacker")
    metrics_directory: Path = Path("data")
    reward_config: StackRewardConfig = field(
        default_factory=StackRewardConfig
    )

    def __post_init__(self) -> None:
        if self.pretrained_checkpoint is not None:
            object.__setattr__(
                self, "pretrained_checkpoint", Path(self.pretrained_checkpoint).expanduser()
            )
        recovery_config = RecoveryStartConfig(
            probability=self.recovery_start_probability,
            xy_offset_range=self.recovery_xy_offset_range,
            height_offset_range=self.recovery_height_offset_range,
            closed_gripper_probability=self.recovery_closed_gripper_probability,
        )
        if self.start_at_orange_waypoint:
            recovery_config.validate_waypoint_height(
                self.reward_config.approach_orange_height_offset
            )
        for name in (
            "history_length",
            "transformer_embedding_dim",
            "transformer_layers",
            "transformer_heads",
            "transformer_feedforward_dim",
            "model_dim",
            "model_layers",
            "evaluation_interval_rollouts",
            "evaluation_episodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed < 0
        ):
            raise ValueError("evaluation_seed must be a nonnegative integer.")
        if self.transformer_embedding_dim % self.transformer_heads:
            raise ValueError(
                "transformer_embedding_dim must be divisible by transformer_heads."
            )
        for name in ("sde_xyz_log_std_init", "sde_gripper_log_std_init"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")


def load_pretraining_checkpoint(
    config: PPOTrainingConfig,
) -> tuple[PPOTrainingConfig, PPO | None]:
    """Read a behavior-cloning archive and adopt only its network architecture."""
    if config.pretrained_checkpoint is None:
        return config, None

    # We transfer weights into a fresh PPO model below, so the source archive
    # does not need a second copy on the training accelerator.
    pretrained = PPO.load(config.pretrained_checkpoint, device="cpu")
    if type(pretrained.policy) is not TanhBoundedMeanActorCriticPolicy:
        raise ValueError("Pretraining requires a TanhBoundedMeanActorCriticPolicy checkpoint.")
    metadata = getattr(pretrained, "pretraining_config", None)
    if not isinstance(metadata, dict):
        raise ValueError("The checkpoint must contain pretraining_config metadata.")
    missing = set(PRETRAINING_ARCHITECTURE_FIELDS) - metadata.keys()
    if missing:
        raise ValueError(f"Pretraining architecture metadata is missing: {sorted(missing)}")
    effective_config = replace(config, **{
        destination: metadata[source]
        for source, destination in PRETRAINING_ARCHITECTURE_FIELDS.items()
    })
    expected_features = {
        "embedding_dim": effective_config.transformer_embedding_dim,
        "layer_count": effective_config.transformer_layers,
        "head_count": effective_config.transformer_heads,
        "feedforward_dim": effective_config.transformer_feedforward_dim,
    }
    policy = pretrained.policy
    if (
        type(policy.features_extractor) is not ActionObservationTransformer
        or policy.features_extractor_kwargs != expected_features
        or policy.features_extractor.history_length != effective_config.history_length
        or not policy.share_features_extractor
        or policy.activation_fn is not torch.nn.ReLU
        or not isinstance(policy.net_arch, dict)
        or policy.net_arch.get("pi") != [effective_config.model_dim] * effective_config.model_layers
    ):
        raise ValueError("Pretraining architecture metadata disagrees with the saved policy.")
    return effective_config, pretrained


def initialize_from_pretraining(model: PPO, pretrained: PPO) -> None:
    """Copy supervised actor weights, retaining fresh PPO critic/noise/optimizer."""
    if (model.observation_space != pretrained.observation_space
            or model.action_space != pretrained.action_space):
        raise ValueError(
            "Pretraining observation/action spaces do not match the training environment. "
            "Regenerate demonstrations and pretrain with the current observation layout."
        )
    for destination, source in (
        (model.policy.features_extractor, pretrained.policy.features_extractor),
        (model.policy.mlp_extractor.policy_net, pretrained.policy.mlp_extractor.policy_net),
        (model.policy.action_net, pretrained.policy.action_net),
    ):
        destination.load_state_dict(source.state_dict(), strict=True)
    # The exploration MLP was frozen during pretraining. Keep the fresh,
    # trainable exploration MLP created by PPO, along with its new log_std.


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
) -> ActionObservationHistoryWrapper:
    """Create one ordinary Gymnasium environment for API validation."""
    environment = CubeStackGymEnvironment(
        seed=config.seed,
        maximum_episode_steps=config.maximum_episode_steps,
        reward_config=config.reward_config,
        start_at_orange_waypoint=config.start_at_orange_waypoint, # TEMP
        recovery_start_probability=config.recovery_start_probability,
        recovery_xy_offset_range=config.recovery_xy_offset_range,
        recovery_height_offset_range=config.recovery_height_offset_range,
        recovery_closed_gripper_probability=config.recovery_closed_gripper_probability,
    )
    return ActionObservationHistoryWrapper(
        environment,
        history_length=config.history_length,
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
            "start_at_orange_waypoint": config.start_at_orange_waypoint, # TEMP
            "recovery_start_probability": config.recovery_start_probability,
            "recovery_xy_offset_range": config.recovery_xy_offset_range,
            "recovery_height_offset_range": config.recovery_height_offset_range,
            "recovery_closed_gripper_probability": config.recovery_closed_gripper_probability,
        },
        monitor_kwargs={
            # Track waypoint status as a metric, outside the model observation.
            "info_keywords": ("orange_pregrasp_waypoint_reached",),
        },
        wrapper_class=ActionObservationHistoryWrapper,
        wrapper_kwargs={"history_length": config.history_length},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )


def evaluate_no_var_success_rate(model: PPO, config: PPOTrainingConfig) -> float:
    """Evaluate fixed scenes serially without advancing the training environment.

    Only complete episodes contribute. Evaluation uses ordinary starts (no
    recovery sampling), the same history wrapper, and deterministic actions.
    """
    evaluation_config = replace(
        config, seed=config.evaluation_seed, recovery_start_probability=0.0,
    )
    environment = None
    previous_training_mode = model.policy.training
    previous_thread_count = torch.get_num_threads()
    try:
        # There is one environment in this process and no evaluation workers.
        # Restore the training thread setting even if preparation/prediction fails.
        torch.set_num_threads(1)
        model.policy.set_training_mode(False)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            environment = create_validation_environment(evaluation_config)
            successes = 0
            for episode in range(config.evaluation_episodes):
                observation, _ = environment.reset(seed=config.evaluation_seed + episode)
                for _ in range(config.maximum_episode_steps):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = environment.step(action)
                    if terminated or truncated:
                        successes += int(bool(info["is_success"]))
                        break
                else:
                    raise RuntimeError("Evaluation episode exceeded its configured step limit.")
                if (episode + 1) % 25 == 0:
                    print(
                        f"Deterministic evaluation: {episode + 1}/{config.evaluation_episodes} episodes",
                        flush=True,
                    )
            return successes / config.evaluation_episodes
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            model.policy.set_training_mode(previous_training_mode)
            torch.set_num_threads(previous_thread_count)


def train_ppo(
    config: PPOTrainingConfig | None = None,
    *,
    wandb_run: Run | None = None,
) -> PPO:
    """Validate the Gym environment, train PPO, and save the policy."""
    training_config, pretrained_model = load_pretraining_checkpoint(config or PPOTrainingConfig())

    # TEMP
    if wandb_run is not None:
        wandb_run.config.update({
            "environment_start_at_orange_waypoint": training_config.start_at_orange_waypoint,
            "environment_recovery_start_probability": (
                training_config.recovery_start_probability
                if training_config.start_at_orange_waypoint
                else 0.0
            ),
            "environment_recovery_xy_offset_range": training_config.recovery_xy_offset_range,
            "environment_recovery_height_offset_range": training_config.recovery_height_offset_range,
            "environment_recovery_closed_gripper_probability": training_config.recovery_closed_gripper_probability,
            "evaluation_interval_rollouts": training_config.evaluation_interval_rollouts,
            "evaluation_episodes": training_config.evaluation_episodes,
            "evaluation_seed": training_config.evaluation_seed,
            "evaluation_recovery_start_probability": 0.0,
            "evaluation_deterministic": True,
        })
    # TEMP END

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
                "features_extractor_class": ActionObservationTransformer,
                "features_extractor_kwargs": {
                    "embedding_dim": training_config.transformer_embedding_dim,
                    "layer_count": training_config.transformer_layers,
                    "head_count": training_config.transformer_heads,
                    "feedforward_dim": training_config.transformer_feedforward_dim,
                },
                "share_features_extractor": True,
                "activation_fn": torch.nn.ReLU,
                "net_arch": {
                    "pi": [training_config.model_dim] * training_config.model_layers,
                    "vf": [training_config.model_dim] * training_config.model_layers,
                },
                "log_std_init": training_config.log_std_init,
                "sde_log_std_init": (
                    (
                        training_config.sde_xyz_log_std_init,
                        training_config.sde_xyz_log_std_init,
                        training_config.sde_xyz_log_std_init,
                        training_config.sde_gripper_log_std_init,
                    )
                    if training_config.use_state_dependent_exploration
                    else None
                ),
                "use_expln": training_config.use_expln,
                # Only the Gaussian mean is tanh-bounded by the custom policy.
                # Keeping gSDE samples unsquashed gives PPO an analytical
                # entropy; SB3 still clips noisy samples to the normalized
                # action bounds before they reach the environment.
                "squash_output": False,
            },
            seed=training_config.seed,
            device=training_config.device,
            verbose=0,
        )
        if pretrained_model is not None:
            initialize_from_pretraining(model, pretrained_model)
            model.pretrained_checkpoint = str(training_config.pretrained_checkpoint.resolve())
            model.pretraining_config = dict(pretrained_model.pretraining_config)
            print(f"Initialized transformer and actor from {model.pretrained_checkpoint}", flush=True)
            del pretrained_model
        # SB3 saves extra model attributes along with policy_kwargs and the
        # observation space. Playback can reproduce the environment config.
        model.pickup_training_config = asdict(training_config)
        metrics_callback = RolloutMetricsCallback(
            training_config.metrics_directory,
            wandb_run=wandb_run,
            evaluation_config=training_config,
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
    model_field_names = set(MODEL_CONFIG_FIELDS)
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


def model_config_overrides(values: Mapping[str, object]) -> dict[str, Any]:
    """Translate W&B model parameters into PPOTrainingConfig arguments."""
    return {
        field_name: (
            float(values[name])
            if name in (
                "learning_rate",
                "target_kl",
                "sde_xyz_log_std_init",
                "sde_gripper_log_std_init",
            )
            else int(values[name])
        )
        for name, field_name in MODEL_CONFIG_FIELDS.items()
        if name in values
    }


def pretrained_architecture_values(config: PPOTrainingConfig) -> dict[str, int]:
    """Return the W&B parameters that must stay fixed when loading actor weights."""
    return {
        f"{MODEL_PARAMETER_PREFIX}{name}": getattr(config, field_name)
        for name, field_name in MODEL_CONFIG_FIELDS.items()
        if field_name in PRETRAINING_ARCHITECTURE_FIELDS.values()
    }


def run_wandb_trial(pretrained_checkpoint: Path | None = None) -> None:
    """Train one policy using the reward and model values sampled by W&B."""
    default_training_config, pretrained_model = load_pretraining_checkpoint(
        PPOTrainingConfig(pretrained_checkpoint=pretrained_checkpoint)
    )
    # Only retain the architecture here. train_ppo loads fresh source weights
    # for each trial, with its own optimizer, critic, and exploration settings.
    del pretrained_model
    default_wandb_values = {
        f"{REWARD_PARAMETER_PREFIX}{name}": value
        for name, value in asdict(StackRewardConfig()).items()
    }
    default_wandb_values.update(
        {
            f"{MODEL_PARAMETER_PREFIX}{name}": getattr(
                default_training_config, field_name,
            )
            for name, field_name in MODEL_CONFIG_FIELDS.items()
        }
    )
    if default_training_config.pretrained_checkpoint is not None:
        default_wandb_values["pretrained_checkpoint"] = str(
            default_training_config.pretrained_checkpoint.resolve()
        )

    with wandb.init(
        project=WANDB_PROJECT_NAME,
        config=default_wandb_values,
    ) as run:
        sampled_values = dict(run.config)
        if default_training_config.pretrained_checkpoint is not None:
            # An existing sweep may still search incompatible architectures.
            # Do not silently ignore its samples and log misleading parameters.
            mismatches = {
                name: (sampled_values[name], expected)
                for name, expected in pretrained_architecture_values(default_training_config).items()
                if name in sampled_values and sampled_values[name] != expected
            }
            if mismatches:
                raise ValueError(
                    "Sweep architecture does not match the pretrained checkpoint "
                    f"(sampled, required): {mismatches}. "
                    "Start a new sweep with --wandb --pretrained to fix its architecture."
                )
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
        training_config = replace(
            default_training_config,
            **model_config_overrides(sampled_model_values),
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


def run_wandb_sweep(
    sweep_id: str | None = None,
    *,
    pretrained_checkpoint: Path | None = None,
) -> None:
    """Run one agent in a new sweep or join an existing sweep."""
    trial_function = run_wandb_trial
    if pretrained_checkpoint is not None:
        pretrained_checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
        config, pretrained_model = load_pretraining_checkpoint(
            PPOTrainingConfig(pretrained_checkpoint=pretrained_checkpoint)
        )
        del pretrained_model
        trial_function = partial(run_wandb_trial, pretrained_checkpoint=pretrained_checkpoint)
    if sweep_id is None:
        _validate_sweep_parameter_names()
        sweep_config = WANDB_SWEEP_CONFIG
        if pretrained_checkpoint is not None:
            sweep_config = deepcopy(WANDB_SWEEP_CONFIG)
            sweep_config["parameters"].update({
                name: {"value": value}
                for name, value in pretrained_architecture_values(config).items()
            })
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=WANDB_PROJECT_NAME,
        )
    elif "/" not in sweep_id:
        sweep_id = f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}"
    # With no count, the W&B agent keeps running trials until interrupted.
    wandb.agent(sweep_id, function=trial_function)


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line options for ordinary or sweep training."""
    parser = argparse.ArgumentParser()
    training_mode = parser.add_mutually_exclusive_group()
    training_mode.add_argument(
        "--wandb",
        nargs="?",
        const="",
        default=None,
        metavar="SWEEP_ID",
        help=(
            "create a new Bayesian W&B sweep, or join an existing sweep "
            "with a short ID or entity/project/sweep_id; "
            "run one agent until stopped"
        ),
    )
    training_mode.add_argument(
        "--repeat-wandb",
        metavar="RUN_ID",
        help="train a fresh policy using a W&B run's reward configuration",
    )
    parser.add_argument(
        "--pretrained",
        nargs="?",
        const=Path("checkpoints/pretraining/default.zip"),
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "initialize the transformer and actor from a pretraining ZIP before PPO training; "
            "can be combined with --wandb to initialize every sweep trial; "
            "defaults to checkpoints/pretraining/default.zip when PATH is omitted"
        ),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    arguments_namespace = parse_arguments(arguments)
    if arguments_namespace.wandb is not None:
        run_wandb_sweep(
            arguments_namespace.wandb
            if arguments_namespace.wandb != ""
            else None,
            pretrained_checkpoint=arguments_namespace.pretrained,
        )
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
        train_ppo(
            PPOTrainingConfig(
                **model_config_overrides(saved_model_values),
                reward_config=reward_config,
                pretrained_checkpoint=arguments_namespace.pretrained,
            )
        )
    elif arguments_namespace.pretrained is not None:
        train_ppo(PPOTrainingConfig(pretrained_checkpoint=arguments_namespace.pretrained))
    else:
        train_ppo()


if __name__ == "__main__":
    main()
