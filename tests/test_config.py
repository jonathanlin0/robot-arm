"""Strict JSON contracts and explicit non-default settings."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.config import ProjectConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_config():
    return json.loads((ROOT / 'configs/default.json').read_text())


def test_default_config_resolves_paths_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / 'configs/default.json')
    assert config.environment.scene_path == ROOT / 'scenes/so101_two_cube_stack.xml'
    assert config.training.vision.data_dir == ROOT / 'data/vision'
    assert config.environment.vision_data.samples_per_sequence == 5
    assert config.training.vision.position_loss_weight == 200.0
    assert config.training.vision.orientation_loss_weight == 1.0
    assert config.training.vision.batch_size == 16
    assert config.training.vision.lr_scheduler.model_dump() == {
        "type": "reduce_on_plateau", "factor": 0.5, "patience": 1,
        "threshold": 0.001, "min_lr": 0.000001,
    }
    assert config.action_policy.transformer.embedding_dim == 160
    assert config.action_policy.history_length == 384
    assert config.cameras.fps == 30.0
    assert config.vision_model.dropout == 0.1
    assert config.vision_model.backbone_weights == "IMAGENET1K_V1"
    assert config.vision_model.wrist_cnn.layers == 2
    assert config.vision_model.wrist_cnn.channels == (16, 16)
    assert config.vision_model.wrist_cnn.embedding_dim == 128
    assert config.training.vision.randaugment.magnitude == 7
    assert config.training.vision.randaugment.num_ops == 2
    assert config.environment.vision_data.cube_pose_modes == ("ground", "stacked", "task")
    assert config.environment.vision_data.cube_pose_mode_probabilities == (0.3, 0.1, 0.6)
    assert config.environment.vision_data.carry.x_range == (0.25, 0.35)
    assert config.environment.vision_data.carry.y_range == (-0.1, 0.1)
    assert config.environment.vision_data.carry.z_range == (0.08, 0.13)
    assert config.environment.vision_data.camera_position_jitter == (0.01, 0.01, 0.01)
    assert config.environment.vision_data.light_position_jitter == (0.1, 0.1, 0.1)
    assert config.environment.vision_data.ambient_intensity_range == (0.2, 0.8)
    assert config.environment.vision_data.light_saturation_range == (0.0, 0.08)
    assert config.environment.vision_data.shadows_enabled is True
    assert config.environment.vision_data.paper_color_variation.model_dump() == {
        "hue_shift": 0.01,
        "saturation_scale": 0.10,
        "brightness_scale": 0.12,
        "warmth": 0.025,
    }
    for name, mean, variance, minimum, maximum in (("specular", 0.65, 0.55, 0.1, 0.8),
                                                  ("shininess", 0.65, 0.55, 0.1, 0.8),
                                                  ("reflectance", 0.2, 0.33, 0.0, 0.7)):
        assert getattr(config.environment.vision_data, f"{name}_mean") == mean
        assert getattr(config.environment.vision_data, f"{name}_variance") == variance
        assert getattr(config.environment.vision_data, f"{name}_min") == minimum
        assert getattr(config.environment.vision_data, f"{name}_max") == maximum
    assert config.environment.vision_data.white_balance_gain_range == (0.9, 1.1)
    assert config.environment.vision_data.gamma_range == (0.9, 1.1)
    assert config.environment.vision_data.focus_distance_range == (0.15, 0.6)
    assert config.environment.vision_data.focus_blur_scale == 0.6
    assert config.environment.vision_data.focus_blur_max_sigma == 2.0
    assert config.environment.vision_data.motion_blur_probability == 0.25
    assert config.environment.vision_data.motion_blur_exposure_range == (0.002, 0.012)
    assert config.environment.vision_data.motion_blur_samples == 3


@pytest.mark.parametrize("name", ["position_loss_weight", "orientation_loss_weight"])
@pytest.mark.parametrize("value", [-0.1, "1.0", True, None, float("nan"), float("inf")])
def test_invalid_vision_loss_weights_are_rejected(raw_config, name, value):
    raw_config["training"]["vision"][name] = value
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("position_weight,orientation_weight", [(0.0, 1.0), (200.0, 0.0), (7.5, 3.25)])
def test_vision_loss_weights_allow_nondefaults_or_one_disabled_component(raw_config, position_weight,
                                                                       orientation_weight):
    raw_config["training"]["vision"].update(position_loss_weight=position_weight,
                                           orientation_loss_weight=orientation_weight)
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).training.vision
    assert config.position_loss_weight == position_weight
    assert config.orientation_loss_weight == orientation_weight


def test_vision_loss_weights_cannot_both_be_zero(raw_config):
    raw_config["training"]["vision"].update(position_loss_weight=0.0, orientation_loss_weight=0.0)
    with pytest.raises(ValidationError, match=r"(?i)(weight|positive|zero)"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("field,value", [
    ("type", "cosine"),
    ("factor", 0.0), ("factor", 1.0), ("factor", -0.5),
    ("factor", "0.5"), ("factor", True), ("factor", float("nan")),
    ("patience", -1), ("patience", 1.0), ("patience", True),
    ("threshold", -0.001), ("threshold", 1.0), ("threshold", float("inf")),
    ("min_lr", 0.0), ("min_lr", -1e-6), ("min_lr", "0.000001"),
])
def test_invalid_vision_scheduler_settings_are_rejected(raw_config, field, value):
    raw_config["training"]["vision"]["lr_scheduler"][field] = value
    with pytest.raises(ValidationError, match=field):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_vision_scheduler_minimum_cannot_exceed_initial_learning_rate(raw_config):
    raw_config["training"]["vision"]["lr_scheduler"]["min_lr"] = 0.001
    with pytest.raises(ValidationError, match="min_lr.*learning_rate"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_vision_scheduler_accepts_boundary_settings(raw_config):
    vision = raw_config["training"]["vision"]
    vision["lr_scheduler"].update(patience=0, threshold=0.0, min_lr=vision["learning_rate"])
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).training.vision
    assert config.lr_scheduler.patience == 0
    assert config.lr_scheduler.threshold == 0.0
    assert config.lr_scheduler.min_lr == config.learning_rate


@pytest.mark.parametrize("omit", [False, True])
def test_vision_scheduler_can_be_disabled_or_missing_in_older_configs(raw_config, omit):
    if omit:
        del raw_config["training"]["vision"]["lr_scheduler"]
    else:
        raw_config["training"]["vision"]["lr_scheduler"] = None
    config = ProjectConfig.model_validate_json(json.dumps(raw_config))
    assert config.training.vision.lr_scheduler is None


def test_vision_scheduler_cannot_be_set_on_action_training(raw_config):
    raw_config["training"]["action"]["lr_scheduler"] = raw_config["training"]["vision"]["lr_scheduler"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1, "0.1", True, float("nan"), float("inf")])
def test_invalid_vision_dropout_is_rejected(raw_config, value):
    raw_config["vision_model"]["dropout"] = value
    with pytest.raises(ValidationError, match="dropout"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("value", [0.0, 0.35])
def test_vision_dropout_can_be_disabled_or_changed(raw_config, value):
    raw_config["vision_model"]["dropout"] = value
    config = ProjectConfig.model_validate_json(json.dumps(raw_config))
    assert config.vision_model.dropout == value


@pytest.mark.parametrize("name,value", [
    ("layers", 0),
    ("layers", -1),
    ("layers", 2.0),
    ("layers", True),
    ("channels", []),
    ("channels", [16]),
    ("channels", [16, 16, 16]),
    ("channels", [0, 16]),
    ("channels", [16, -1]),
    ("channels", [16, 16.0]),
    ("channels", [True, 16]),
    ("channels", ["16", 16]),
    ("embedding_dim", 0),
    ("embedding_dim", -1),
    ("embedding_dim", 128.0),
    ("embedding_dim", True),
])
def test_invalid_wrist_cnn_settings_are_rejected(raw_config, name, value):
    raw_config["vision_model"]["wrist_cnn"][name] = value
    with pytest.raises(ValidationError, match="wrist_cnn"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("channels,embedding_dim", [([8], 32), ([8, 16, 24], 96)])
def test_wrist_cnn_allows_custom_layer_widths_and_embedding(raw_config, channels, embedding_dim):
    raw_config["vision_model"]["wrist_cnn"].update(
        layers=len(channels), channels=channels, embedding_dim=embedding_dim,
    )
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).vision_model.wrist_cnn
    assert config.layers == len(channels)
    assert config.channels == tuple(channels)
    assert config.embedding_dim == embedding_dim


def test_resnet50_v2_weights_are_rejected_for_resnet34(raw_config):
    raw_config["vision_model"]["backbone_weights"] = "IMAGENET1K_V2"
    with pytest.raises(ValidationError, match="backbone_weights"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["white_balance_gain_range", "gamma_range", "focus_distance_range"])
@pytest.mark.parametrize("bounds", [[1.1, 0.9], [0.0, 1.0], [-0.1, 1.0]])
def test_camera_effect_ranges_require_positive_ordered_values(raw_config, name, bounds):
    raw_config["environment"]["vision_data"][name] = bounds
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name, value", [
    ("focus_blur_scale", -0.1),
    ("focus_blur_max_sigma", -0.1),
    ("motion_blur_probability", -0.1),
    ("motion_blur_probability", 1.1),
    ("motion_blur_exposure_range", [-0.001, 0.01]),
    ("motion_blur_exposure_range", [0.01, 0.001]),
    ("motion_blur_samples", 1),
    ("motion_blur_samples", 3.0),
])
def test_invalid_camera_blur_settings_are_rejected(raw_config, name, value):
    raw_config["environment"]["vision_data"][name] = value
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_camera_effects_can_be_fixed_or_disabled(raw_config):
    raw_config["environment"]["vision_data"].update(
        white_balance_gain_range=[1.0, 1.0],
        gamma_range=[1.0, 1.0],
        focus_distance_range=[0.3, 0.3],
        focus_blur_scale=0.0,
        focus_blur_max_sigma=0.0,
        motion_blur_probability=0.0,
        motion_blur_exposure_range=[0.0, 0.0],
        motion_blur_samples=2,
    )
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).environment.vision_data
    assert config.white_balance_gain_range == (1.0, 1.0)
    assert config.gamma_range == (1.0, 1.0)
    assert config.focus_distance_range == (0.3, 0.3)
    assert config.focus_blur_scale == config.focus_blur_max_sigma == 0.0
    assert config.motion_blur_probability == 0.0
    assert config.motion_blur_exposure_range == (0.0, 0.0)
    assert config.motion_blur_samples == 2


@pytest.mark.parametrize("name", ["camera_position_jitter", "light_position_jitter"])
@pytest.mark.parametrize("jitter", [
    0.05,
    [0.05, 0.05],
    [0.05, 0.05, 0.01, 0.01],
    [-0.05, 0.05, 0.01],
    [0.05, -0.05, 0.01],
    [0.05, 0.05, -0.01],
    ["0.05", 0.05, 0.01],
    [True, 0.05, 0.01],
    [0.05, float("nan"), 0.01],
    [0.05, 0.05, float("inf")],
])
def test_invalid_position_jitter_is_rejected(raw_config, name, jitter):
    raw_config["environment"]["vision_data"][name] = jitter
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["ambient_intensity_range", "light_saturation_range"])
@pytest.mark.parametrize("bounds", [
    [0.8, 0.2], [-0.01, 0.8], [0.2, 1.01], [0.2],
    ["0.2", 0.8], [False, 0.8], [float("nan"), 0.8], [0.2, float("inf")],
])
def test_invalid_lighting_ranges_are_rejected(raw_config, name, bounds):
    raw_config["environment"]["vision_data"][name] = bounds
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_shadows_enabled_requires_boolean(raw_config, value):
    raw_config["environment"]["vision_data"]["shadows_enabled"] = value
    with pytest.raises(ValidationError, match="shadows_enabled"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["hue_shift", "saturation_scale", "brightness_scale", "warmth"])
@pytest.mark.parametrize("value", [-0.001, "0.01", True, None, float("nan"), float("inf"),
                                  float("-inf")])
def test_invalid_paper_color_variation_is_rejected(raw_config, name, value):
    raw_config["environment"]["vision_data"]["paper_color_variation"][name] = value
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name,value", [
    ("hue_shift", 0.050001),
    ("saturation_scale", 1.0),
    ("brightness_scale", 1.0),
    ("warmth", 1.0),
])
def test_paper_color_variation_rejects_excessive_deviations(raw_config, name, value):
    raw_config["environment"]["vision_data"]["paper_color_variation"][name] = value
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("variation", [
    {"hue_shift": 0.0, "saturation_scale": 0.0, "brightness_scale": 0.0, "warmth": 0.0},
    {"hue_shift": 0.05, "saturation_scale": 0.999, "brightness_scale": 0.999, "warmth": 0.999},
])
def test_paper_color_variation_accepts_disabled_and_boundary_values(raw_config, variation):
    raw_config["environment"]["vision_data"]["paper_color_variation"] = variation
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).environment.vision_data
    assert config.paper_color_variation.model_dump() == variation


@pytest.mark.parametrize("name", ["wall_brightness_range", "wall_saturation_range"])
def test_obsolete_wall_color_ranges_are_rejected(raw_config, name):
    raw_config["environment"]["vision_data"][name] = [0.0, 1.0]
    with pytest.raises(ValidationError, match=rf"{name}[\s\S]*extra_forbidden"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["specular", "shininess", "reflectance"])
@pytest.mark.parametrize("parameter,value", [
    ("mean", float("nan")), ("mean", float("inf")), ("mean", "0.65"), ("mean", True),
    ("variance", -0.1), ("variance", float("nan")), ("variance", float("inf")),
    ("variance", "0.55"), ("variance", True),
    ("min", -0.01), ("min", 1.01), ("min", float("nan")), ("min", float("inf")),
    ("max", -0.01), ("max", 1.01), ("max", float("nan")), ("max", float("inf")),
])
def test_invalid_material_distribution_fields_are_rejected(raw_config, name, parameter, value):
    field = f"{name}_{parameter}"
    raw_config["environment"]["vision_data"][field] = value
    with pytest.raises(ValidationError, match=field):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["specular", "shininess", "reflectance"])
@pytest.mark.parametrize("mean,variance,minimum,maximum", [
    (0.65, 0.55, 0.8, 0.2),
    (0.65, 0.55, 0.5, 0.5),
    (-0.01, 0.0, 0.0, 1.0),
    (1.01, 0.0, 0.0, 1.0),
])
def test_material_distributions_require_possible_samples(raw_config, name, mean, variance, minimum, maximum):
    raw_config["environment"]["vision_data"].update({
        f"{name}_mean": mean, f"{name}_variance": variance,
        f"{name}_min": minimum, f"{name}_max": maximum,
    })
    with pytest.raises(ValidationError, match=name):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize("name", ["specular", "shininess", "reflectance"])
@pytest.mark.parametrize("mean,variance,minimum,maximum", [
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (0.4, 0.0, 0.4, 0.4),
    (-0.1, 0.55, 0.0, 1.0),
    (1.1, 0.55, 0.0, 1.0),
    (0.4, 0.1, 0.2, 0.8),
])
def test_material_distributions_allow_fixed_values_and_custom_ranges(raw_config, name, mean, variance,
                                                                    minimum, maximum):
    raw_config["environment"]["vision_data"].update({
        f"{name}_mean": mean, f"{name}_variance": variance,
        f"{name}_min": minimum, f"{name}_max": maximum,
    })
    config = ProjectConfig.model_validate_json(json.dumps(raw_config)).environment.vision_data
    assert getattr(config, f"{name}_mean") == mean
    assert getattr(config, f"{name}_variance") == variance
    assert getattr(config, f"{name}_min") == minimum
    assert getattr(config, f"{name}_max") == maximum


@pytest.mark.parametrize("probabilities", [
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [-0.1, 0.3, 0.8],
    [0.0, 0.0, 1.1],
    [0.0, 0.0, 0.0],
    [0.3, 0.1, 0.5],
    [0.3, 0.1, float("nan")],
    [0.3, 0.1, float("inf")],
])
def test_invalid_cube_mode_probabilities_are_rejected(raw_config, probabilities):
    raw_config["environment"]["vision_data"]["cube_pose_mode_probabilities"] = probabilities
    with pytest.raises(ValidationError, match="cube_pose_mode_probabilities"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize('path', [(), ('vision_model',), ('vision_model', 'wrist_cnn'),
                                  ('action_policy','transformer'),
                                  ('training','vision'), ('training','vision','lr_scheduler'),
                                  ('environment','vision_data'),
                                  ('environment','vision_data','carry'),
                                  ('environment','vision_data','paper_color_variation')])
def test_unknown_keys_fail_at_every_level(raw_config, path):
    value = raw_config
    for part in path:
        value = value[part]
    value['unused_field'] = 99
    with pytest.raises(ValidationError, match='extra_forbidden'):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_obsolete_history_dependent_cube_yaw_setting_is_rejected(raw_config):
    raw_config["action_policy"]["cube_yaw"] = {"face_switch_relative_advantage": 0.05}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_file_backed_fields_except_optional_scheduler_are_required(raw_config):
    def check_fields(value, path=()):
        for key, item in value.items():
            altered = json.loads(json.dumps(raw_config))
            parent = altered
            for part in path:
                parent = parent[part]
            del parent[key]
            if path + (key,) != ('training', 'vision', 'lr_scheduler'):
                with pytest.raises(ValidationError, match='missing'):
                    ProjectConfig.model_validate_json(json.dumps(altered))
            if isinstance(item, dict):
                check_fields(item, path + (key,))
    check_fields(raw_config)


@pytest.mark.parametrize("name, bounds", [
    ("x_range", [0.35, 0.25]),
    ("y_range", [0.1, -0.1]),
    ("z_range", [0.13, 0.08]),
    ("z_range", [0.0, 0.13]),
    ("z_range", [-0.1, 0.13]),
])
def test_invalid_carry_target_ranges_are_rejected(raw_config, name, bounds):
    raw_config["environment"]["vision_data"]["carry"][name] = bounds
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize('value', ['4', 4.0, True])
def test_no_integer_coercion(raw_config, value):
    raw_config['action_policy']['transformer']['heads'] = value
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_nullable_field_must_be_explicit(raw_config):
    raw_config['vision_model']['backbone_weights'] = None
    config = ProjectConfig.model_validate_json(json.dumps(raw_config))
    assert config.vision_model.backbone_weights is None
    del raw_config['vision_model']['backbone_weights']
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_incompatible_context_and_heads_are_rejected(raw_config):
    raw_config['action_policy']['transformer']['embedding_dim'] = 161
    with pytest.raises(ValidationError, match='divisible'):
        ProjectConfig.model_validate_json(json.dumps(raw_config))
    raw_config['action_policy']['transformer']['embedding_dim'] = 160
    raw_config['training']['action']['stride'] = 385
    with pytest.raises(ValidationError, match='stride'):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_nondefault_configs_reach_runtime_components(raw_config):
    from camera_frame_buffer import CameraFrameBuffer
    from camera_producers import SimCameraProducer
    from cartesian_controller import CartesianController
    from environment import CubeStackEnvironment
    raw_config['cameras'].update(fps=18.0, maximum_frame_age_seconds=0.3)
    raw_config['cartesian_controller'].update(maximum_position_delta=0.001, ik_max_iterations=7)
    config = ProjectConfig.model_validate_json(json.dumps(raw_config))
    env = CubeStackEnvironment(scene_path=ROOT / config.environment.scene_path)
    buffers = {name: CameraFrameBuffer(camera_id=name,
               maximum_frame_age_seconds=config.cameras.maximum_frame_age_seconds)
               for name in ('wrist','external')}
    producer = SimCameraProducer(environment=env, buffers=buffers, fps=config.cameras.fps)
    controller = CartesianController(robot_model=env.model, config=config.cartesian_controller)
    assert producer.fps == 18.0
    assert buffers['wrist'].maximum_frame_age_seconds == 0.3
    assert controller.config.ik_max_iterations == 7
    assert controller.config.maximum_position_delta == 0.001
