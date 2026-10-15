"""Pretraining dataset behavior."""

from gymnasium import spaces
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4
import copy
import gymnasium as gym
import json
import numpy as np
import tempfile
import torch
import unittest

from pretrain_helpers import (
    module, setUpModule, tearDownModule, settings, small_config,
)
from pretrain_pickup import (
    ActionObservationHistoryWrapper,
    INPUT_NAMES,
    PRIVILEGED_OBSERVATION_SIZE,
    StackSuccessConfig,
    TOKEN_SIZE,
    load_episode,
    materialize_sequences,
    prepare_data,
    read_episode_records,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pretraining-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_episode(self, split="train", seed=7, steps=3, *, version=1, offset=0.0):
        identifier = uuid4()
        if version == 2:
            seed = identifier.int
        observations = torch.arange((steps + 1) * PRIVILEGED_OBSERVATION_SIZE,
                                    dtype=torch.float32).reshape(steps + 1, -1) / 100 + offset
        targets = torch.arange((steps + 1) * 3, dtype=torch.float32).reshape(steps + 1, 3) / 1000 + 0.2
        actions = torch.linspace(-0.9, 0.9, steps * 4).reshape(steps, 4)
        tensors = (observations, targets, actions)
        for name, tensor in zip(("observations", "accepted_targets", "actions"), tensors):
            path = self.directory / split / name / f"{identifier}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
        record = dict(uuid=str(identifier), seed=seed, split=split, steps=steps,
                      success=True, final_hold_time=0.0,
                      final_stack_stable_time=StackSuccessConfig().required_stable_time,
                      settings=settings())
        if version == 1:
            path = self.directory / "manifest.json"
            content = json.loads(path.read_text()) if path.exists() else {"format_version": 1, "episodes": []}
            content["episodes"].append(record)
        else:
            path = self.directory / "manifests" / f"{identifier}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            content = {"format_version": 2, "episode": record}
        path.write_text(json.dumps(content))
        return record, tensors

    def test_both_manifest_formats_and_file_only_preparation(self):
        legacy, _ = self.write_episode(seed=7)
        modern, _ = self.write_episode("test", version=2, steps=2)
        self.assertNotEqual(UUID(legacy["uuid"]).int, legacy["seed"])
        records = read_episode_records(self.directory)
        self.assertEqual(records[0]["seed"], 7)
        self.assertEqual(records[1]["seed"], UUID(modern["uuid"]).int)
        with patch.object(module, "CubeStackGymEnvironment", side_effect=AssertionError("No simulator during preparation")), \
                patch.object(module, "DataLoader", side_effect=AssertionError("Prepare tensors before loaders")):
            data = prepare_data(small_config(data_directory=self.directory), show_progress=False)
        self.assertEqual((len(data.training), len(data.validation)), (1, 1))
        self.assertEqual(data.validation_records, [modern])
        for dataset in (data.training, data.validation):
            for tensor in dataset.tensors[:4]:
                self.assertEqual((tensor.device.type, tensor.dtype), ("cpu", torch.float32))
            self.assertEqual(dataset.tensors[4].dtype, torch.bool)
        self.assertEqual(data.training.tensors[4].sum().item(), 3)
        self.assertEqual(data.validation.tensors[4].sum().item(), 2)

    def test_mixed_legacy_fixed_and_randomized_starts_preserve_saved_inputs(self):
        legacy, legacy_tensors = self.write_episode(seed=1)
        fixed, fixed_tensors = self.write_episode(version=2, offset=10.0)
        randomized, random_tensors = self.write_episode("test", version=2, offset=20.0)
        for record, half_range in ((fixed, [0.0, 0.0, 0.0]),
                                   (randomized, [0.04, 0.04, 0.02])):
            record["settings"].update(start_position=[0.4, 0.0, 0.25],
                                      start_position_half_range=half_range)
            path = self.directory / "manifests" / f"{record['uuid']}.json"
            path.write_text(json.dumps({"format_version": 2, "episode": record}))
        originals = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}

        with patch.object(module, "CubeStackGymEnvironment", side_effect=AssertionError("Use saved inputs")):
            data = prepare_data(small_config(data_directory=self.directory), show_progress=False)

        self.assertEqual((len(data.training), len(data.validation)), (2, 1))
        self.assertEqual(data.validation_records, [randomized])
        self.assertNotIn("start_position", legacy["settings"])
        for index, tensors in enumerate((legacy_tensors, fixed_tensors)):
            torch.testing.assert_close(data.training.tensors[0][index, :3, :49], tensors[0][:-1])
            torch.testing.assert_close(data.training.tensors[3][index, :3], tensors[2])
        torch.testing.assert_close(data.validation.tensors[0][0, :3, :49], random_tensors[0][:-1])
        torch.testing.assert_close(data.validation.tensors[3][0, :3], random_tensors[2])
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_optional_start_metadata_rejects_invalid_vectors(self):
        self.write_episode()
        path = self.directory / "manifest.json"
        original = json.loads(path.read_text())
        for name, value in (
            ("start_position", [0.4, 0.0]),
            ("start_position", [0.4, float("nan"), 0.25]),
            ("start_position", [True, 0.0, 0.25]),
            ("start_position", None),
            ("start_position_half_range", [0.04, 0.04, -0.02]),
            ("start_position_half_range", [0.04, 0.04, float("inf")]),
            ("start_position_half_range", [0.04, 0.04]),
        ):
            content = copy.deepcopy(original)
            content["episodes"][0]["settings"][name] = value
            path.write_text(json.dumps(content))
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                read_episode_records(self.directory)

    def test_materialized_sequences_match_live_history_without_future_leakage(self):
        first, first_tensors = self.write_episode(steps=70, seed=1)
        second, second_tensors = self.write_episode(steps=3, seed=2, offset=50)
        records = [first, second]
        dataset = materialize_sequences(self.directory, records, 64, "test", sequence_stride=32, show_progress=False)

        class ReplayEnvironment(gym.Env):
            observation_space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_OBSERVATION_SIZE,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)

            def __init__(self, tensors):
                self.observations, self.targets, self.actions = tensors

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.index = 0
                return self.observations[0].numpy().copy(), {"target_gripper_position": self.targets[0].numpy().copy()}

            def step(self, action):
                np.testing.assert_array_equal(action, self.actions[self.index].numpy())
                self.index += 1
                return (self.observations[self.index].numpy().copy(), 0.0,
                        self.index == len(self.actions), False,
                        {"target_gripper_position": self.targets[self.index].numpy().copy()})

        offset = 0
        for tensors in (first_tensors, second_tensors):
            wrapper = ActionObservationHistoryWrapper(ReplayEnvironment(tensors), history_length=64)
            history, _ = wrapper.reset()
            for index, action in enumerate(tensors[2]):
                chunk_index = max(0, (index - 64) // 32 + 1)
                chunk_start = chunk_index * 32
                local_index = index - chunk_start
                row = dataset[offset + chunk_index]
                live_offset = chunk_start - max(0, index - 63)
                for name, actual in zip(INPUT_NAMES, row[:3]):
                    np.testing.assert_array_equal(
                        actual[:local_index + 1].numpy(),
                        history[name][live_offset:live_offset + local_index + 1],
                    )
                torch.testing.assert_close(row[3][local_index], action)
                self.assertTrue(row[4][local_index])
                history, *_ = wrapper.step(action.numpy())
            wrapper.close()
            offset += 1 + max(0, (len(tensors[2]) - 64 + 31) // 32)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.tensors[2][1].sum().item(), 0)
        self.assertEqual(dataset.tensors[2][2, 0].item(), 1)

        # A terminal state has no action label and must not enter any window.
        for name, tensor in zip(("observations", "accepted_targets"), first_tensors[:2]):
            changed = tensor.clone()
            changed[-1] = 999
            torch.save(changed, self.directory / "train" / name / f"{first['uuid']}.pt")
        terminal_changed = materialize_sequences(self.directory, records, 64, "test", sequence_stride=32, show_progress=False)
        for actual, expected in zip(terminal_changed.tensors, dataset.tensors):
            torch.testing.assert_close(actual, expected)

        # Neither future observations nor the current label belongs in this input.
        future_observations = first_tensors[0].clone()
        future_observations[31:] += 100
        future_actions = first_tensors[2].clone()
        future_actions[30:] *= -1
        torch.save(future_observations, self.directory / "train/observations" / f"{first['uuid']}.pt")
        torch.save(future_actions, self.directory / "train/actions" / f"{first['uuid']}.pt")
        future_changed = materialize_sequences(self.directory, records, 64, "test", sequence_stride=32, show_progress=False)
        for actual, expected in zip(future_changed.tensors[:3], dataset.tensors[:3]):
            torch.testing.assert_close(actual[0, :31], expected[0, :31])
        torch.testing.assert_close(future_changed.tensors[3][0, :30], dataset.tensors[3][0, :30])

    def test_manifest_rejects_overlap_incompatible_settings_and_missing_files(self):
        first, _ = self.write_episode(seed=1)
        second, _ = self.write_episode("test", seed=2)
        path = self.directory / "manifest.json"
        original = json.loads(path.read_text())
        changes = [lambda item: item.update(seed=1), lambda item: item.update(uuid=first["uuid"]),
                   lambda item: item.update(split="validation"), lambda item: item.update(success=False),
                   lambda item: item.update(steps=0), lambda item: item.update(final_stack_stable_time=0.1),
                   lambda item: item.update(final_stack_stable_time=float("nan")),
                   lambda item: item.update(final_stack_stable_time=True),
                   lambda item: item.pop("final_stack_stable_time"),
                   lambda item: item["settings"].update(task="pickup"),
                   lambda item: item["settings"].update(start_at_orange_waypoint=True),
                   lambda item: item["settings"].update(recovery_start_probability=0.25),
                   lambda item: item["settings"].update(waypoint_height=0.1),
                   lambda item: item["settings"].update(maximum_episode_steps=2),
                   lambda item: item["settings"].update(success_config=None),
                   lambda item: item["settings"].update(success_config={}),
                   lambda item: item["settings"]["success_config"].update(required_stable_time=3.0),
                   lambda item: item["settings"]["success_config"].update(max_linear_speed=0.02)]
        for change in changes:
            content = copy.deepcopy(original)
            change(content["episodes"][1])
            path.write_text(json.dumps(content))
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_episode_records(self.directory)
        path.write_text(json.dumps(original))
        (self.directory / "test/actions" / f"{second['uuid']}.pt").unlink()
        with self.assertRaises(ValueError):
            read_episode_records(self.directory)

    def test_legacy_pickup_demonstrations_are_rejected_unchanged(self):
        record, _ = self.write_episode()
        path = self.directory / "manifest.json"
        content = json.loads(path.read_text())
        legacy = content["episodes"][0]
        legacy.pop("final_stack_stable_time")
        legacy["final_hold_time"] = 0.6
        legacy["settings"].pop("task")
        legacy["settings"].pop("success_config")
        legacy["settings"].update(start_at_orange_waypoint=True, minimum_hold_time=0.5)
        path.write_text(json.dumps(content))
        files = [path] + [self.directory / "train" / name / f"{record['uuid']}.pt"
                          for name in ("observations", "accepted_targets", "actions")]
        originals = {file: file.read_bytes() for file in files}
        with self.assertRaisesRegex(ValueError, "legacy pickup-only data is unsupported"):
            read_episode_records(self.directory)
        self.assertEqual({file: file.read_bytes() for file in files}, originals)

    def test_uuid_manifest_mismatch_and_empty_dataset(self):
        with self.assertRaises(ValueError):
            read_episode_records(self.directory)
        record, _ = self.write_episode(version=2)
        path = self.directory / "manifests" / f"{record['uuid']}.json"
        content = json.loads(path.read_text())
        content["episode"]["seed"] += 1
        path.write_text(json.dumps(content))
        with self.assertRaises(ValueError):
            read_episode_records(self.directory)
        with self.assertRaises(ValueError):
            materialize_sequences(self.directory, [], 4, "empty", sequence_stride=2, show_progress=False)

    def test_tensor_shape_dtype_finiteness_and_action_bounds(self):
        record, tensors = self.write_episode()
        for name, invalid in (("observations", tensors[0][:-1]),
                              ("accepted_targets", tensors[1].double()),
                              ("actions", torch.zeros((3, 3))),
                              ("actions", torch.full((3, 4), float("nan"))),
                              ("actions", torch.full((3, 4), 1.01)),
                              ("observations", {"not": "a tensor"})):
            path = self.directory / "train" / name / f"{record['uuid']}.pt"
            original = path.read_bytes()
            torch.save(invalid, path)
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    load_episode(self.directory, record)
            finally:
                path.write_bytes(original)

    def test_legacy_waypoint_flag_observations_are_rejected_unchanged(self):
        record, tensors = self.write_episode()
        self.assertEqual(tensors[0].shape[1], 49)
        self.assertEqual(TOKEN_SIZE, 56)
        path = self.directory / "train/observations" / f"{record['uuid']}.pt"
        legacy = torch.cat((tensors[0], torch.ones((record["steps"] + 1, 1))), dim=1)
        torch.save(legacy, path)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, r"shape \(4, 49\)"):
            load_episode(self.directory, record)
        self.assertEqual(path.read_bytes(), original)

    def test_episode_limit_applied_before_materialization(self):
        for split in ("train", "test"):
            for index in range(3):
                self.write_episode(split, seed=index + (10 if split == "test" else 0))
        with patch.object(module, "load_episode", wraps=load_episode) as load:
            data = prepare_data(small_config(data_directory=self.directory), episode_limit=1, show_progress=False)
        self.assertEqual(load.call_count, 2)
        self.assertEqual((len(data.training), len(data.validation)), (1, 1))
