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
    assert config.training.vision.position_loss_weight == 200.0
    assert config.training.vision.orientation_loss_weight == 1.0
    assert config.action_policy.transformer.embedding_dim == 160
    assert config.action_policy.history_length == 384
    assert config.cameras.fps == 30.0
    assert config.vision_model.dropout == 0.1
    assert config.environment.vision_data.cube_pose_modes == ("ground", "elevated", "stacked", "task", "carry")
    assert config.environment.vision_data.cube_pose_mode_probabilities == (0.1, 0.1, 0.1, 0.6, 0.1)
    assert config.environment.vision_data.carry.x_range == (0.25, 0.35)
    assert config.environment.vision_data.carry.y_range == (-0.1, 0.1)
    assert config.environment.vision_data.carry.z_range == (0.08, 0.13)
    assert config.environment.vision_data.camera_position_jitter == (0.01, 0.01, 0.01)
    assert config.environment.vision_data.light_position_jitter == (0.1, 0.1, 0.1)
    assert config.environment.vision_data.ambient_intensity_range == (0.2, 0.8)
    assert config.environment.vision_data.light_saturation_range == (0.0, 0.08)
    assert config.environment.vision_data.shadows_enabled is True
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


@pytest.mark.parametrize("probabilities", [
    [0.1, 0.1, 0.8],
    [-0.1, 0.1, 0.1, 0.8, 0.1],
    [0.0, 0.0, 0.0, 1.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.5, 0.1],
    [0.1, 0.1, 0.1, float("nan"), 0.1],
    [0.1, 0.1, 0.1, float("inf"), 0.1],
])
def test_invalid_cube_mode_probabilities_are_rejected(raw_config, probabilities):
    raw_config["environment"]["vision_data"]["cube_pose_mode_probabilities"] = probabilities
    with pytest.raises(ValidationError, match="cube_pose_mode_probabilities"):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


@pytest.mark.parametrize('path', [(), ('vision_model',), ('action_policy','transformer'),
                                  ('training','vision'), ('environment','vision_data'),
                                  ('environment','vision_data','carry')])
def test_unknown_keys_fail_at_every_level(raw_config, path):
    value = raw_config
    for part in path:
        value = value[part]
    value['unused_field'] = 99
    with pytest.raises(ValidationError, match='extra_forbidden'):
        ProjectConfig.model_validate_json(json.dumps(raw_config))


def test_every_file_backed_field_is_required(raw_config):
    def check_fields(value, path=()):
        for key, item in value.items():
            altered = json.loads(json.dumps(raw_config))
            parent = altered
            for part in path:
                parent = parent[part]
            del parent[key]
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
