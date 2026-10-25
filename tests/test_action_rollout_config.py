"""Rollout count, spawn bounds, and physical success thresholds are explicit."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.config import ProjectConfig, load_config


ROOT = Path(__file__).parents[1]


def test_default_rollout_validation_settings():
    project = load_config(ROOT / "configs/default.json")
    config = project.training.action.rollout_validation
    assert config.episodes == 200
    assert config.seed == 43
    assert config.cube_x_range == project.environment.action_data.cube_x_range
    assert config.cube_y_range == project.environment.action_data.cube_y_range
    assert config.minimum_cube_separation == project.environment.action_data.minimum_cube_separation == 0.08
    assert config.start_position == project.environment.action_data.start_position == (0.2, 0.0, 0.05)
    assert config.start_position_half_range == project.environment.action_data.start_position_half_range == (0.04, 0.04, 0.0)
    assert config.success.required_stable_time == 1.0
    assert config.success.max_horizontal_center_offset == 0.01


@pytest.mark.parametrize("field,value", [
    ("episodes", 0), ("episodes", -1), ("episodes", 2.5), ("episodes", True),
    ("seed", -1), ("seed", "43"),
    ("cube_x_range", [0.4, 0.2]), ("cube_y_range", [0.1, -0.1]),
    ("minimum_cube_separation", 0.0), ("minimum_cube_separation", 1.0),
    ("start_position", [0.2, 0.0]), ("start_position", [0.2, 0.0, 0.0]),
    ("start_position", [0.2, 0.0, -0.1]), ("start_position", [0.2, 0.0, float("nan")]),
    ("start_position_half_range", [0.04, 0.04]),
    ("start_position_half_range", [-0.04, 0.04, 0.0]),
    ("start_position_half_range", [0.04, -0.04, 0.0]),
    ("start_position_half_range", [0.04, 0.04, -0.01]),
    ("start_position_half_range", [0.04, 0.04, 0.05]),
    ("start_position_half_range", [0.04, 0.04, 0.1]),
    ("start_position_half_range", [float("inf"), 0.04, 0.0]),
])
def test_invalid_rollout_settings_are_rejected(field, value):
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["training"]["action"]["rollout_validation"][field] = value
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("field", ["start_position", "start_position_half_range"])
def test_rollout_home_settings_are_required(field):
    raw = json.loads((ROOT / "configs/default.json").read_text())
    del raw["training"]["action"]["rollout_validation"][field]
    with pytest.raises(ValidationError, match=field):
        ProjectConfig.model_validate_json(json.dumps(raw))


def test_rollout_home_accepts_nondefault_center_and_zero_spread():
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["training"]["action"]["rollout_validation"].update(
        start_position=[0.21, -0.02, 0.06], start_position_half_range=[0.0, 0.0, 0.0],
    )
    config = ProjectConfig.model_validate_json(json.dumps(raw)).training.action.rollout_validation
    assert config.start_position == (0.21, -0.02, 0.06)
    assert config.start_position_half_range == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("field,value", [
    ("required_stable_time", 0.0), ("required_stable_time", -1.0),
    ("max_horizontal_center_offset", -0.1), ("vertical_center_tolerance", -0.1),
    ("max_linear_speed", -0.1), ("max_angular_speed", -0.1),
])
def test_invalid_success_thresholds_are_rejected(field, value):
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["training"]["action"]["rollout_validation"]["success"][field] = value
    with pytest.raises(ValidationError, match=field):
        ProjectConfig.model_validate_json(json.dumps(raw))
