"""Required JSON settings. Unknown keys and implicit coercion are errors."""

from math import hypot, isclose
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TransformerConfig(StrictConfig):
    embedding_dim: int = Field(gt=0)
    feedforward_dim: int = Field(gt=0)
    heads: int = Field(gt=0)
    layers: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def divisible_heads(self):
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        return self


class WristCNNConfig(StrictConfig):
    layers: int = Field(gt=0)
    channels: tuple[int, ...]
    embedding_dim: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_channels(self):
        if len(self.channels) != self.layers:
            raise ValueError("channels must contain one width per wrist CNN layer")
        if any(width <= 0 for width in self.channels):
            raise ValueError("wrist CNN channels must be positive")
        return self


class VisionModelConfig(StrictConfig):
    adapter_kernel_size: int = Field(gt=0)
    backbone_weights: Literal["IMAGENET1K_V1"] | None
    wrist_cnn: WristCNNConfig
    mlp_hidden_layers: int = Field(ge=0)
    mlp_hidden_dim: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def odd_kernel(self):
        if self.adapter_kernel_size % 2 == 0:
            raise ValueError("adapter_kernel_size must be odd")
        return self


class ActionPolicyConfig(StrictConfig):
    history_length: int = Field(gt=0)
    transformer: TransformerConfig
    mlp_hidden_dims: tuple[int, ...]
    action_std: float = Field(ge=0)
    deterministic: bool

    @model_validator(mode="after")
    def positive_hidden_dims(self):
        if any(width <= 0 for width in self.mlp_hidden_dims):
            raise ValueError("mlp_hidden_dims must contain positive widths")
        return self


class SupervisedTrainingConfig(StrictConfig):
    data_dir: Path
    checkpoint_path: Path
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    optimizer_betas: tuple[float, float]
    optimizer_eps: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    shuffle_train: bool
    pin_memory: bool
    epochs: int = Field(gt=0)
    seed: int = Field(ge=0)
    num_workers: int = Field(ge=0)
    device: str
    validation_interval: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_betas(self):
        if any(not 0 <= beta < 1 for beta in self.optimizer_betas):
            raise ValueError("optimizer_betas must be in [0, 1)")
        return self


class StackSuccessConfig(StrictConfig):
    max_horizontal_center_offset: float = Field(ge=0)
    vertical_center_tolerance: float = Field(ge=0)
    max_linear_speed: float = Field(ge=0)
    max_angular_speed: float = Field(ge=0)
    required_stable_time: float = Field(gt=0)


class ActionRolloutValidationConfig(StrictConfig):
    episodes: int = Field(gt=0)
    seed: int = Field(ge=0)
    cube_x_range: tuple[float, float]
    cube_y_range: tuple[float, float]
    minimum_cube_separation: float = Field(gt=0)
    start_position: tuple[float, float, float]
    start_position_half_range: tuple[float, float, float]
    success: StackSuccessConfig

    @model_validator(mode="after")
    def valid_spawn_ranges(self):
        for name in ("cube_x_range", "cube_y_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must be in ascending order")
        extent = hypot(self.cube_x_range[1] - self.cube_x_range[0],
                       self.cube_y_range[1] - self.cube_y_range[0])
        if extent <= self.minimum_cube_separation:
            raise ValueError("Rollout spawn ranges cannot accommodate minimum_cube_separation")
        if any(value < 0 for value in self.start_position_half_range):
            raise ValueError("start_position_half_range must be nonnegative")
        if self.start_position[2] - self.start_position_half_range[2] <= 0:
            raise ValueError("The complete start_position height range must be above the table")
        return self


class ActionTrainingConfig(SupervisedTrainingConfig):
    stride: int = Field(gt=0)
    rollout_validation: ActionRolloutValidationConfig


class RandAugmentConfig(StrictConfig):
    num_ops: int = Field(ge=0)
    magnitude: int = Field(ge=0, le=30)


class ReduceLROnPlateauConfig(StrictConfig):
    type: Literal["reduce_on_plateau"]
    factor: float = Field(gt=0, lt=1)
    patience: int = Field(ge=0)
    threshold: float = Field(ge=0, lt=1)
    min_lr: float = Field(gt=0)


class VisionTrainingConfig(SupervisedTrainingConfig):
    randaugment: RandAugmentConfig
    position_loss_weight: float = Field(ge=0)
    orientation_loss_weight: float = Field(ge=0)
    # Existing project configurations may predate learning-rate scheduling.
    lr_scheduler: ReduceLROnPlateauConfig | None = None

    @model_validator(mode="after")
    def nonzero_loss_weights(self):
        if self.position_loss_weight == 0 and self.orientation_loss_weight == 0:
            raise ValueError("At least one vision loss weight must be positive")
        return self

    @model_validator(mode="after")
    def valid_scheduler_minimum(self):
        if self.lr_scheduler is not None and self.lr_scheduler.min_lr > self.learning_rate:
            raise ValueError("lr_scheduler.min_lr must not exceed learning_rate")
        return self


class TrainingConfig(StrictConfig):
    vision: VisionTrainingConfig
    action: ActionTrainingConfig


class VisionScenarioConfig(StrictConfig):
    waypoint_height: float = Field(gt=0)
    grasp_offset: tuple[float, float, float]
    lift_height: float = Field(gt=0)
    position_tolerance: float = Field(gt=0)
    phase_timeout_seconds: float = Field(gt=0)
    close_seconds: float = Field(gt=0)
    release_seconds: float = Field(gt=0)
    settle_seconds: float = Field(gt=0)
    drop_after_seconds: float = Field(gt=0)


class VisionCarryConfig(StrictConfig):
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]

    @model_validator(mode="after")
    def ordered_ranges(self):
        for name in ("x_range", "y_range", "z_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must be in ascending order")
        if self.z_range[0] <= 0:
            raise ValueError("z_range must be above the table")
        return self


class PaperColorVariationConfig(StrictConfig):
    """Maximum color deviations sampled symmetrically around the paper color."""

    hue_shift: float = Field(ge=0, le=0.05)
    saturation_scale: float = Field(ge=0, lt=1)
    brightness_scale: float = Field(ge=0, lt=1)
    warmth: float = Field(ge=0, lt=1)


class VisionDataGenerationConfig(StrictConfig):
    sequences: int = Field(ge=2)
    samples_per_sequence: int = Field(gt=0)
    sample_interval_steps: int = Field(gt=0)
    train_fraction: float = Field(gt=0, lt=1)
    joint_step_std: float = Field(ge=0)
    cube_x_range: tuple[float, float]
    cube_y_range: tuple[float, float]
    cube_height_range: tuple[float, float]
    minimum_cube_separation: float = Field(gt=0)
    cube_pose_modes: tuple[Literal["ground", "elevated", "stacked", "task", "carry"], ...]
    cube_pose_mode_probabilities: tuple[float, ...]
    task: VisionScenarioConfig
    carry: VisionCarryConfig
    camera_position_jitter: tuple[float, float, float]
    camera_angle_jitter: float = Field(ge=0)
    fov_range: tuple[float, float]
    light_position_jitter: tuple[float, float, float]
    light_intensity_range: tuple[float, float]
    ambient_intensity_range: tuple[float, float]
    light_saturation_range: tuple[float, float]
    shadows_enabled: bool
    specular_mean: float
    specular_variance: float = Field(ge=0)
    specular_min: float = Field(ge=0, le=1)
    specular_max: float = Field(ge=0, le=1)
    shininess_mean: float
    shininess_variance: float = Field(ge=0)
    shininess_min: float = Field(ge=0, le=1)
    shininess_max: float = Field(ge=0, le=1)
    reflectance_mean: float
    reflectance_variance: float = Field(ge=0)
    reflectance_min: float = Field(ge=0, le=1)
    reflectance_max: float = Field(ge=0, le=1)
    paper_color_variation: PaperColorVariationConfig
    image_noise_std: float = Field(ge=0)
    blur_probability: float = Field(ge=0, le=1)
    exposure_range: tuple[float, float]
    white_balance_gain_range: tuple[float, float]
    gamma_range: tuple[float, float]
    focus_distance_range: tuple[float, float]
    focus_blur_scale: float = Field(ge=0)
    focus_blur_max_sigma: float = Field(ge=0)
    motion_blur_probability: float = Field(ge=0, le=1)
    motion_blur_exposure_range: tuple[float, float]
    motion_blur_samples: int = Field(ge=2)
    lens_distortion_range: tuple[float, float]
    texture_contrast_range: tuple[float, float]

    @model_validator(mode="after")
    def valid_material_distributions(self):
        for name in ("specular", "shininess", "reflectance"):
            mean = getattr(self, f"{name}_mean")
            variance = getattr(self, f"{name}_variance")
            lo = getattr(self, f"{name}_min")
            hi = getattr(self, f"{name}_max")
            if lo > hi:
                raise ValueError(f"{name}_min must not exceed {name}_max")
            if variance == 0:
                if not lo <= mean <= hi:
                    raise ValueError(f"{name}_mean must be within [{name}_min, {name}_max] "
                                     f"when {name}_variance is zero")
            elif lo == hi:
                raise ValueError(f"{name}_min must be below {name}_max "
                                 f"when {name}_variance is positive")
        return self

    @model_validator(mode="after")
    def ordered_ranges(self):
        for name in ("cube_x_range", "cube_y_range", "cube_height_range", "fov_range", "light_intensity_range",
                     "ambient_intensity_range", "light_saturation_range",
                     "exposure_range",
                     "white_balance_gain_range", "gamma_range", "focus_distance_range",
                     "motion_blur_exposure_range",
                     "lens_distortion_range", "texture_contrast_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must be in ascending order")
        for name in ("white_balance_gain_range", "gamma_range", "focus_distance_range"):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} must be positive")
        if self.motion_blur_exposure_range[0] < 0:
            raise ValueError("motion_blur_exposure_range must be non-negative")
        for name in ("ambient_intensity_range", "light_saturation_range"):
            if not 0 <= getattr(self, name)[0] <= getattr(self, name)[1] <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("camera_position_jitter", "light_position_jitter"):
            if any(offset < 0 for offset in getattr(self, name)):
                raise ValueError(f"{name} must contain non-negative X, Y, Z offsets")
        if not 0 < self.fov_range[0] <= self.fov_range[1] < 180:
            raise ValueError("fov_range must be between 0 and 180 degrees")
        if self.cube_height_range[0] < 0.02 or not self.cube_pose_modes:
            raise ValueError("cube_height_range must clear the table and cube_pose_modes must be nonempty")
        return self

    @model_validator(mode="after")
    def valid_mode_probabilities(self):
        probabilities = self.cube_pose_mode_probabilities
        if len(probabilities) != len(self.cube_pose_modes):
            raise ValueError("cube_pose_mode_probabilities must match the number of cube_pose_modes")
        if any(not 0 <= probability <= 1 for probability in probabilities):
            raise ValueError("cube_pose_mode_probabilities must be between 0 and 1")
        if not isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("cube_pose_mode_probabilities must sum to 1")
        return self


class DropDisturbanceConfig(StrictConfig):
    probability: float = Field(ge=0, le=1)
    stop_seconds: float = Field(gt=0)
    release_seconds: float = Field(gt=0)


class ActionRecoveryConfig(StrictConfig):
    lost_grasp_steps: int = Field(gt=0)
    pause_steps: int = Field(ge=0)
    settle_steps: int = Field(gt=0)
    max_linear_speed: float = Field(ge=0)
    max_angular_speed: float = Field(ge=0)
    waypoint_grasp_retry_seconds: float = Field(gt=0)


class ActionTeacherConfig(StrictConfig):
    waypoint_height: float = Field(gt=0)
    grasp_radial_offset: float = Field(ge=0)
    grasp_height_offset: float
    position_tolerance: float = Field(gt=0)
    consecutive_near_steps: int = Field(gt=0)
    maximum_move_steps: int = Field(gt=0)
    open_settle_steps: int = Field(gt=0)
    close_gripper_steps: int = Field(gt=0)
    lift_settle_steps: int = Field(gt=0)
    release_steps: int = Field(gt=0)
    retreat_distance: float = Field(gt=0)
    extra_hold_steps: int = Field(ge=0)


class ActionDataGenerationConfig(StrictConfig):
    train_fraction: float = Field(gt=0, lt=1)
    maximum_attempts_per_example: int = Field(gt=0)
    cube_x_range: tuple[float, float]
    cube_y_range: tuple[float, float]
    minimum_cube_separation: float = Field(gt=0)
    spawn_maximum_attempts: int = Field(gt=0)
    start_position: tuple[float, float, float]
    start_position_half_range: tuple[float, float, float]
    drop_disturbance: DropDisturbanceConfig
    recovery: ActionRecoveryConfig
    teacher: ActionTeacherConfig
    success: StackSuccessConfig

    @model_validator(mode="after")
    def valid_generation_geometry(self):
        for name in ("cube_x_range", "cube_y_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must be in ascending order")
        extent = hypot(self.cube_x_range[1] - self.cube_x_range[0],
                       self.cube_y_range[1] - self.cube_y_range[0])
        if extent <= self.minimum_cube_separation:
            raise ValueError("Action generation spawn ranges cannot accommodate minimum_cube_separation")
        if any(value < 0 for value in self.start_position_half_range):
            raise ValueError("start_position_half_range must be nonnegative")
        if self.start_position[2] - self.start_position_half_range[2] <= 0:
            raise ValueError("The complete start_position height range must be above the table")
        return self


class EnvironmentConfig(StrictConfig):
    scene_path: Path
    episode_seconds: float = Field(gt=0)
    vision_data: VisionDataGenerationConfig
    action_data: ActionDataGenerationConfig


class CameraConfig(StrictConfig):
    fps: float = Field(gt=0)
    maximum_frame_age_seconds: float = Field(gt=0)


class CartesianActionConfig(StrictConfig):
    maximum_position_delta: float = Field(gt=0)
    closed_gripper_target: float
    open_gripper_target: float
    close_gripper_command_threshold: float = Field(ge=-1, le=1)
    open_gripper_command_threshold: float = Field(ge=-1, le=1)
    target_tool_axis: tuple[float, float, float]
    require_downward: bool
    target_tool_yaw: float | None
    position_tolerance: float = Field(gt=0)
    tool_axis_tolerance: float = Field(gt=0, le=3.141592653589793)
    tool_yaw_tolerance: float = Field(gt=0, le=3.141592653589793)
    ik_max_iterations: int = Field(gt=0)
    ik_damping: float = Field(gt=0)
    ik_max_joint_step: float = Field(gt=0)
    backtracking_scales: tuple[float, ...]
    orientation_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_controls(self):
        if self.close_gripper_command_threshold >= self.open_gripper_command_threshold:
            raise ValueError("gripper close threshold must be below open threshold")
        if not self.backtracking_scales or any(not 0 < s <= 1 for s in self.backtracking_scales):
            raise ValueError("backtracking_scales must be nonempty and in (0, 1]")
        if sum(v * v for v in self.target_tool_axis) == 0:
            raise ValueError("target_tool_axis must be nonzero")
        if self.require_downward:
            x, y, z = self.target_tool_axis
            if x != 0 or y != 0 or z >= 0:
                raise ValueError("require_downward requires a world-down target_tool_axis")
        return self


class LoopRunnerConfig(StrictConfig):
    control_hz: float = Field(gt=0)


class ProjectConfig(StrictConfig):
    vision_model: VisionModelConfig
    action_policy: ActionPolicyConfig
    training: TrainingConfig
    environment: EnvironmentConfig
    cameras: CameraConfig
    cartesian_controller: CartesianActionConfig
    loop: LoopRunnerConfig

    @model_validator(mode="after")
    def stride_within_context(self):
        if self.training.action.stride > self.action_policy.history_length:
            raise ValueError("training.action.stride cannot exceed action_policy.history_length")
        return self


def load_config(path: Path | str) -> ProjectConfig:
    """Read explicit settings; resolve relative paths from the JSON's project root.

    A config inside configs/ uses its parent project directory, so entry points
    behave the same from a different working directory. Other locations use
    the directory containing the JSON.
    """
    path = Path(path).expanduser().resolve()
    config = ProjectConfig.model_validate_json(path.read_text())
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    for section, names in (
        (config.environment, ("scene_path",)),
        (config.training.vision, ("data_dir", "checkpoint_path")),
        (config.training.action, ("data_dir", "checkpoint_path")),
    ):
        for name in names:
            value = getattr(section, name).expanduser()
            setattr(section, name, value if value.is_absolute() else (base / value).resolve())
    return config
