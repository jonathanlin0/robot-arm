"""Demonstration and validation jaw tolerances are independent settings."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from uuid import uuid4

import pytest
import torch

from pretrain_helpers import module, settings, small_config, setUpModule, tearDownModule
import generate_pickup_demonstrations as generator


def test_default_validation_tolerance_is_larger_than_teacher_tolerance() -> None:
    assert generator.CLAW_YAW_TOLERANCE_DEGREES == 1.0
    assert module.VALIDATION_CLAW_YAW_TOLERANCE_DEGREES == 5.0
    assert small_config().validation_claw_yaw_tolerance_degrees == 5.0


def test_generator_knob_reaches_controller_reset_and_saved_settings(monkeypatch) -> None:
    monkeypatch.setattr(generator, "CLAW_YAW_TOLERANCE_DEGREES", 2.5)
    environment = generator.make_environment()
    try:
        recorded = generator.generation_settings(environment)
        expected = math.radians(2.5)
        assert environment.action_adapter.config.tool_yaw_tolerance == expected
        assert environment.simulation.tool_yaw_tolerance == expected
        assert recorded["action_config"]["tool_yaw_tolerance"] == expected
        assert recorded["teacher"]["CLAW_YAW_TOLERANCE_DEGREES"] == 2.5
    finally:
        environment.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_validation_overrides_tolerance_without_mutating_demonstration_settings(legacy) -> None:
    recorded = settings()
    if legacy:
        recorded["action_config"].pop("tool_yaw_tolerance")
    else:
        recorded["action_config"]["tool_yaw_tolerance"] = math.radians(1.5)
    recorded["action_config"]["maximum_position_delta"] = 0.0017
    recorded["action_config"]["target_tool_yaw"] = 0.02
    before = deepcopy(recorded)
    config = small_config(validation_claw_yaw_tolerance_degrees=7.0)
    environment = module.make_validation_environment(config, recorded)
    try:
        applied = asdict(environment.unwrapped.action_adapter.config)
        assert applied["tool_yaw_tolerance"] == math.radians(7.0)
        assert environment.unwrapped.simulation.tool_yaw_tolerance == math.radians(7.0)
        for name, value in before["action_config"].items():
            if name != "tool_yaw_tolerance":
                # JSON metadata represents tuple-valued config fields as lists.
                assert applied[name] == value
        assert recorded == before
    finally:
        environment.close()


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf, True, "5"])
def test_validation_tolerance_must_be_a_positive_finite_number(invalid) -> None:
    with pytest.raises(ValueError, match="validation_claw_yaw_tolerance_degrees"):
        small_config(validation_claw_yaw_tolerance_degrees=invalid)


@pytest.mark.parametrize("new_tolerance_degrees", [1.0, 2.0])
def test_manifest_comparison_uses_effective_legacy_default_without_rewriting_files(
    tmp_path, new_tolerance_degrees,
) -> None:
    records = []
    for index, split in enumerate(("train", "test")):
        identifier = str(uuid4())
        recorded = settings()
        if index == 0:
            recorded["action_config"].pop("tool_yaw_tolerance")
        else:
            recorded["action_config"]["tool_yaw_tolerance"] = math.radians(new_tolerance_degrees)
        records.append({
            "uuid": identifier, "seed": index, "split": split, "steps": 1,
            "success": True,
            "final_stack_stable_time": recorded["success_config"]["required_stable_time"],
            "settings": recorded,
        })
        for name, shape in (("observations", (2, 49)), ("accepted_targets", (2, 3)),
                            ("actions", (1, 4))):
            destination = tmp_path / split / name / f"{identifier}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros(shape), destination)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format_version": 1, "episodes": records}))
    original = manifest.read_bytes()
    if new_tolerance_degrees == 1.0:
        assert module.read_episode_records(tmp_path) == records
    else:
        with pytest.raises(ValueError, match="different environment settings"):
            module.read_episode_records(tmp_path)
    assert manifest.read_bytes() == original


def test_saved_checkpoint_keeps_validation_tolerance_separate_from_teacher(tmp_path, monkeypatch) -> None:
    from stable_baselines3 import PPO

    monkeypatch.setattr(module, "PRETRAINING_CHECKPOINT_DIRECTORY", tmp_path)
    config = small_config(save_name="yaw", validation_claw_yaw_tolerance_degrees=7.0)
    policy = module.create_policy(config, torch.device("cpu"))
    recorded = settings()
    recorded["action_config"]["tool_yaw_tolerance"] = math.radians(1.5)
    history = [{"epoch": 1, "validation_success_rate": 0.5}]
    path = module.save_checkpoint(policy, config, history, [{"settings": recorded}])
    loaded = PPO.load(path, device="cpu")
    assert loaded.pretraining_config["validation_claw_yaw_tolerance_degrees"] == 7.0
    assert loaded.pretraining_environment_settings == recorded
    assert loaded.pretraining_environment_settings["action_config"]["tool_yaw_tolerance"] == math.radians(1.5)
