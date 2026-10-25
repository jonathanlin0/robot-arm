"""Generation defaults preserve the legacy disturbance and recovery behavior."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.config import ProjectConfig, load_config


ROOT = Path(__file__).parents[1]


def test_default_generation_configuration_and_destination():
    project = load_config(ROOT / "configs/default.json")
    settings = project.environment.action_data
    assert project.training.action.data_dir == ROOT / "data/action"
    assert settings.train_fraction == 0.8
    assert settings.maximum_attempts_per_example == 10
    assert settings.start_position == (0.2, 0.0, 0.05)
    assert settings.start_position_half_range == (0.04, 0.04, 0.0)
    assert settings.drop_disturbance.probability == 0.002
    assert settings.drop_disturbance.stop_seconds == settings.drop_disturbance.release_seconds == 0.25
    assert settings.recovery.lost_grasp_steps == 2
    assert settings.recovery.waypoint_grasp_retry_seconds == 5.0
    assert settings.teacher.waypoint_height == 0.08


@pytest.mark.parametrize("section,field,value", [
    (None, "train_fraction", 0.0), (None, "train_fraction", 1.0),
    (None, "maximum_attempts_per_example", 0), (None, "maximum_attempts_per_example", True),
    (None, "spawn_maximum_attempts", 0), (None, "cube_x_range", [0.4, 0.2]),
    (None, "cube_y_range", [0.1, -0.1]), (None, "minimum_cube_separation", 1.0),
    (None, "start_position_half_range", [-0.04, 0.04, 0.0]),
    (None, "start_position_half_range", [0.04, 0.04, 0.1]),
    (None, "start_position", [0.2, 0.0, 0.0]),
    ("drop_disturbance", "probability", -0.1), ("drop_disturbance", "probability", 1.1),
    ("drop_disturbance", "probability", True), ("drop_disturbance", "probability", "0.002"),
    ("drop_disturbance", "probability", float("nan")),
    ("drop_disturbance", "stop_seconds", 0.0), ("drop_disturbance", "release_seconds", -0.1),
    ("recovery", "lost_grasp_steps", 0), ("recovery", "lost_grasp_steps", 1.5),
    ("recovery", "pause_steps", -1), ("recovery", "settle_steps", 0),
    ("recovery", "max_linear_speed", -0.01), ("recovery", "max_angular_speed", -0.1),
    ("recovery", "waypoint_grasp_retry_seconds", 0.0),
    ("teacher", "maximum_move_steps", 0), ("teacher", "consecutive_near_steps", 0),
    ("teacher", "position_tolerance", 0.0), ("teacher", "waypoint_height", -0.1),
    ("teacher", "retreat_distance", 0.0), ("teacher", "extra_hold_steps", -1),
])
def test_invalid_generation_settings_are_rejected(section, field, value):
    raw = json.loads((ROOT / "configs/default.json").read_text())
    target = raw["environment"]["action_data"]
    if section is not None:
        target = target[section]
    target[field] = value
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_disturbances_can_be_disabled_or_forced_for_testing(probability):
    raw = json.loads((ROOT / "configs/default.json").read_text())
    raw["environment"]["action_data"]["drop_disturbance"]["probability"] = probability
    project = ProjectConfig.model_validate_json(json.dumps(raw))
    assert project.environment.action_data.drop_disturbance.probability == probability
