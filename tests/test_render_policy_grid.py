from dataclasses import asdict
from pathlib import Path
from typing import Any

from gymnasium import spaces
import mujoco
import numpy as np
import pytest

import scripts.demo as demo
from scripts.demo import (
    BASE_EVALUATION_SEED,
    CHECKPOINT_PATH,
    DEFAULT_WORKER_COUNT,
    FAILURE,
    GRID_COLUMNS,
    GRID_ROWS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    ROLLOUT_COUNT,
    STATUS_BORDER_COLORS,
    STATUS_BORDER_WIDTH,
    SUCCESS,
    TRUNCATED,
    RolloutResult,
    RolloutTask,
    add_status_border,
    build_xstack_command,
    cleanup_temporary_directory,
    combine_rollout_videos,
    parse_arguments,
    rollout_status,
    rollout_video_paths,
    run_parallel_rollouts,
    temporary_rollout_directory,
    get_worker_policy,
    resolve_checkpoint_path,
    reset_state_dependent_noise_if_due,
)


def test_arguments_use_configured_defaults() -> None:
    options = parse_arguments([])

    assert options.workers == DEFAULT_WORKER_COUNT
    assert options.repeat_wandb is None


def test_worker_count_can_be_overridden() -> None:
    options = parse_arguments(["--workers", "3"])

    assert options.workers == 3


def test_wandb_run_id_can_be_selected() -> None:
    options = parse_arguments(["--repeat-wandb", "m58de6rc"])

    assert options.repeat_wandb == "m58de6rc"
    assert resolve_checkpoint_path(options.repeat_wandb) == Path(
        "checkpoints/wandb/m58de6rc/ppo_cube_stacker.zip"
    )


def test_default_checkpoint_path_is_preserved() -> None:
    assert resolve_checkpoint_path(None) == CHECKPOINT_PATH


def test_approach_target_marker_shows_ten_millimeter_radius() -> None:
    assert demo.APPROACH_TARGET_MARKER_RADIUS == pytest.approx(
        0.01
    )


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../run", "run/id", r"run\id", "run id"],
)
def test_wandb_run_id_must_be_one_safe_path_component(
    run_id: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_arguments(["--repeat-wandb", run_id])


@pytest.mark.parametrize("worker_count", ["0", str(ROLLOUT_COUNT + 1), "not-an-integer"])
def test_worker_count_must_be_valid(worker_count: str) -> None:
    with pytest.raises(SystemExit):
        parse_arguments(["--workers", worker_count])


def test_temporary_rollout_paths_are_repository_local() -> None:
    directory = temporary_rollout_directory(process_id=1234)
    paths = rollout_video_paths(directory)

    assert directory == Path(".tmp/policy_grid_1234")
    assert not directory.is_absolute()
    assert paths == [
        directory / f"rollout_{rollout_index:02d}.mp4"
        for rollout_index in range(ROLLOUT_COUNT)
    ]


def test_temporary_rollout_directory_requires_positive_process_id() -> None:
    with pytest.raises(ValueError, match="process_id must be positive"):
        temporary_rollout_directory(process_id=0)


def test_build_xstack_command_combines_all_inputs_in_a_grid() -> None:
    input_paths = rollout_video_paths(Path(".tmp/policy_grid_1234"))
    output_path = Path("stack_demo.mp4")

    command = build_xstack_command(input_paths, output_path)

    encoded_input_paths = [
        Path(command[index + 1])
        for index, argument in enumerate(command)
        if argument == "-i"
    ]
    assert encoded_input_paths == input_paths
    assert command[command.index("-filter_complex") + 1] == (
        f"xstack=inputs={ROLLOUT_COUNT}:"
        f"grid={GRID_COLUMNS}x{GRID_ROWS}:shortest=1[grid]"
    )
    assert command[command.index("-map") + 1] == "[grid]"
    assert Path(command[-1]) == output_path


def test_build_xstack_command_requires_all_rollout_videos() -> None:
    with pytest.raises(ValueError, match=f"expected {ROLLOUT_COUNT}"):
        build_xstack_command(
            [Path(".tmp/policy_grid_1234/rollout_00.mp4")],
            Path("stack_demo.mp4"),
        )


def test_combine_rollout_videos_runs_checked_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths = rollout_video_paths(Path(".tmp/policy_grid_1234"))
    output_path = Path("stack_demo.mp4")
    recorded_call: dict[str, Any] = {}

    monkeypatch.setattr(Path, "is_file", lambda self: True)

    def fake_run(command: list[str], *, check: bool) -> None:
        recorded_call["command"] = command
        recorded_call["check"] = check

    monkeypatch.setattr(demo.subprocess, "run", fake_run)

    combine_rollout_videos(input_paths, output_path)

    assert recorded_call == {
        "command": build_xstack_command(input_paths, output_path),
        "check": True,
    }


def test_cleanup_removes_only_the_supplied_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = Path(".tmp/policy_grid_1234")
    removed_directories: list[Path] = []

    monkeypatch.setattr(Path, "exists", lambda self: self == directory)
    monkeypatch.setattr(
        demo.shutil,
        "rmtree",
        removed_directories.append,
    )

    cleanup_temporary_directory(directory)

    assert removed_directories == [directory]


def test_parallel_rollouts_use_spawn_and_requested_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        RolloutTask(
            rollout_index=rollout_index,
            seed=BASE_EVALUATION_SEED + rollout_index,
            output_path=Path(f"rollout_{rollout_index:02d}.mp4"),
        )
        for rollout_index in range(ROLLOUT_COUNT)
    ]
    unordered_results = [
        RolloutResult(
            rollout_index=task.rollout_index,
            seed=task.seed,
            status=TRUNCATED,
            episode_steps=400,
            episode_reward=0.0,
            output_path=task.output_path,
        )
        for task in reversed(tasks)
    ]
    calls: dict[str, Any] = {}

    class FakePool:
        def imap_unordered(
            self,
            function: object,
            received_tasks: list[RolloutTask],
        ) -> list[RolloutResult]:
            calls["function"] = function
            calls["tasks"] = received_tasks
            return unordered_results

        def close(self) -> None:
            calls["closed"] = True

        def join(self) -> None:
            calls["joined"] = True

    class FakeContext:
        def Pool(self, *, processes: int) -> FakePool:  # noqa: N802
            calls["processes"] = processes
            return FakePool()

    def fake_get_context(method: str) -> FakeContext:
        calls["start_method"] = method
        return FakeContext()

    monkeypatch.setattr(
        demo.multiprocessing,
        "get_context",
        fake_get_context,
    )

    results = run_parallel_rollouts(tasks, worker_count=3)

    assert calls == {
        "start_method": "spawn",
        "processes": 3,
        "function": demo.render_rollout,
        "tasks": tasks,
        "closed": True,
        "joined": True,
    }
    assert [result.rollout_index for result in results] == list(
        range(ROLLOUT_COUNT)
    )


def test_worker_policy_cache_is_scoped_to_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = Path("checkpoints/wandb/first/ppo_cube_stacker.zip")
    second_path = Path("checkpoints/wandb/second/ppo_cube_stacker.zip")
    loaded_paths: list[tuple[Path, str]] = []

    class FakePolicy:
        pass

    def fake_load(checkpoint_path: Path, *, device: str) -> FakePolicy:
        loaded_paths.append((checkpoint_path, device))
        return FakePolicy()

    monkeypatch.setattr(demo, "_worker_policy", None)
    monkeypatch.setattr(
        demo,
        "_worker_policy_checkpoint_path",
        None,
    )
    monkeypatch.setattr(demo.PPO, "load", fake_load)

    first_policy = get_worker_policy(first_path)
    cached_first_policy = get_worker_policy(first_path)
    second_policy = get_worker_policy(second_path)

    assert cached_first_policy is first_policy
    assert second_policy is not first_policy
    assert loaded_paths == [
        (first_path, "cpu"),
        (second_path, "cpu"),
    ]


def test_policy_environment_rejects_old_flat_checkpoints() -> None:
    environment = demo.CubeStackGymEnvironment()

    class FlatPolicy:
        observation_space = spaces.Box(-np.inf, np.inf, shape=(50,), dtype=np.float32)

    try:
        with pytest.raises(ValueError, match="Retrain older flat"):
            demo.create_policy_environment(
                environment,
                FlatPolicy(),
            )
    finally:
        environment.close()


@pytest.mark.parametrize("token_dim", [54, 55, 57])
def test_policy_environment_rejects_incompatible_history_checkpoints(token_dim: int) -> None:
    environment = demo.CubeStackGymEnvironment()

    class IncompatiblePolicy:
        observation_space = spaces.Dict(
            {
                "tokens": spaces.Box(-np.inf, np.inf, shape=(3, token_dim), dtype=np.float32),
                "valid": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
                "episode_start": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            }
        )

    try:
        with pytest.raises(ValueError, match="Expected 56-value tokens"):
            demo.create_policy_environment(
                environment,
                IncompatiblePolicy(),
            )
    finally:
        environment.close()


def test_policy_environment_rejects_disagreeing_saved_history_length() -> None:
    environment = demo.CubeStackGymEnvironment()
    schema_environment = demo.ActionObservationHistoryWrapper(
        environment,
        history_length=3,
    )

    class IncompatiblePolicy:
        observation_space = schema_environment.observation_space
        pickup_training_config = {"history_length": 64}

    try:
        with pytest.raises(ValueError, match="history length disagrees"):
            demo.create_policy_environment(
                environment,
                IncompatiblePolicy(),
            )
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("include_training_metadata", "start_at_orange_waypoint"),
    [(False, False), (True, False), (True, True)],
)
def test_rollout_uses_checkpoint_history_and_resets_it_for_each_seed(
    monkeypatch: pytest.MonkeyPatch,
    include_training_metadata: bool,
    start_at_orange_waypoint: bool,
) -> None:
    history_length = 3
    video_steps = 7
    episode_steps = 5 if include_training_metadata else video_steps
    issued_action = np.array([0.15, -0.2, 0.1, 2.0], dtype=np.float32)
    received_observations: list[dict[str, np.ndarray]] = []
    writers: list[Any] = []
    spawn_outline_calls: list[tuple[object, object]] = []
    schema_environment = demo.ActionObservationHistoryWrapper(
        demo.CubeStackGymEnvironment(),
        history_length=history_length,
    )
    checkpoint_observation_space = schema_environment.observation_space
    action_space = schema_environment.action_space
    schema_environment.close()
    expected_reward_config = demo.StackRewardConfig(
        approach_orange_height_offset=(0.09 if include_training_metadata else 0.08),
    )
    original_environment_class = demo.CubeStackGymEnvironment
    created_environments: list[Any] = []

    def create_recorded_environment(**kwargs: Any) -> Any:
        from cartesian_actions import CartesianActionConfig
        # This test exercises history/replay plumbing with arbitrary actions.
        environment = original_environment_class(
            **kwargs, action_config=CartesianActionConfig(require_downward=False),
        )
        created_environments.append(environment)
        return environment

    class FakePolicy:
        observation_space = checkpoint_observation_space
        use_sde = False

        def predict(
            self,
            observation: dict[str, np.ndarray],
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            assert self.observation_space.contains(observation)
            latest_token = observation["tokens"][int(observation["valid"].sum()) - 1]
            environment = created_environments[-1]
            np.testing.assert_allclose(
                latest_token[-3:],
                environment.action_adapter.current_target_gripper_position,
                rtol=1e-6,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                latest_token[18:21],
                environment.simulation.get_state()["gripper_position"],
                rtol=1e-6,
                atol=1e-8,
            )
            received_observations.append(
                {key: value.copy() for key, value in observation.items()}
            )
            return issued_action.copy(), None

    class FakeRenderer:
        scene = object()

        def __init__(self, model: object, *, height: int, width: int) -> None:
            self.shape = (height, width, 3)

        def update_scene(self, data: object, *, camera: object) -> None:
            pass

        def render(self) -> np.ndarray:
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self) -> None:
            pass

    class FakeVideoWriter:
        def __init__(self, **kwargs: object) -> None:
            self.frames = 0
            self.closed = False
            writers.append(self)

        def write(self, frame: np.ndarray) -> None:
            self.frames += 1

        def close(self, *, check_return_code: bool = True) -> None:
            self.closed = True

    policy = FakePolicy()
    if include_training_metadata:
        policy.pickup_training_config = {
            "history_length": history_length,
            "maximum_episode_steps": episode_steps,
            "reward_config": asdict(expected_reward_config),
            "recovery_start_probability": 0.25,
        }
        if start_at_orange_waypoint:
            policy.pickup_training_config["start_at_orange_waypoint"] = True
    monkeypatch.setattr(
        demo,
        "CubeStackGymEnvironment",
        create_recorded_environment,
    )
    monkeypatch.setattr(
        demo,
        "get_worker_policy",
        lambda path: policy,
    )
    monkeypatch.setattr(
        demo,
        "MAXIMUM_EPISODE_STEPS",
        video_steps,
    )
    monkeypatch.setattr(demo.mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(demo, "RawVideoWriter", FakeVideoWriter)
    monkeypatch.setattr(
        demo,
        "add_spawn_area_outline",
        lambda scene, spawn_config: spawn_outline_calls.append((scene, spawn_config)),
    )
    monkeypatch.setattr(
        demo,
        "add_gripperframe_marker",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        demo,
        "add_active_orange_approach_target_marker",
        lambda *args, **kwargs: None,
    )

    for rollout_index, seed in enumerate((31, 32)):
        result = demo.render_rollout(
            RolloutTask(
                rollout_index=rollout_index,
                seed=seed,
                output_path=Path(f"rollout_{rollout_index}.mp4"),
            )
        )
        assert result.status == TRUNCATED
        assert result.episode_steps == episode_steps

    assert len(received_observations) == 2 * episode_steps
    assert len(created_environments) == 2
    assert spawn_outline_calls == [
        (FakeRenderer.scene, environment.simulation.spawn_config)
        for environment in created_environments
        for _ in range(episode_steps)
    ]
    for environment in created_environments:
        assert environment.maximum_episode_steps == episode_steps
        assert environment.reward_config == expected_reward_config
        assert environment.start_at_orange_waypoint is start_at_orange_waypoint
        assert environment.recovery_start_config.probability == 0.0
    for offset in (0, episode_steps):
        initial = received_observations[offset]
        assert initial["tokens"].shape == (history_length, 56)
        np.testing.assert_array_equal(initial["valid"], [1, 0, 0])
        np.testing.assert_array_equal(initial["episode_start"], [1, 0, 0])
        np.testing.assert_array_equal(initial["tokens"][0, 49:53], 0)
        if start_at_orange_waypoint:
            waypoint = initial["tokens"][0, 21:24].copy()
            waypoint[2] += expected_reward_config.approach_orange_height_offset
            assert np.linalg.norm(initial["tokens"][0, 18:21] - waypoint) <= 0.005
            assert np.linalg.norm(initial["tokens"][0, -3:] - waypoint) <= 0.001
        else:
            np.testing.assert_array_equal(
                initial["tokens"][0, -3:], initial["tokens"][0, 18:21]
            )
        for step in range(1, episode_steps):
            observation = received_observations[offset + step]
            valid_count = min(
                step + 1,
                history_length,
            )
            assert np.count_nonzero(observation["valid"]) == valid_count
            np.testing.assert_array_equal(
                observation["tokens"][valid_count - 1, 49:53],
                np.clip(issued_action, action_space.low, action_space.high),
            )
        np.testing.assert_array_equal(
            received_observations[offset + episode_steps - 1]["episode_start"],
            0,
        )
    assert all(
        writer.closed and writer.frames == video_steps
        for writer in writers
    )


@pytest.mark.parametrize(
    ("use_sde", "resample_frequency", "expected_reset_steps"),
    [
        (True, 8, [0, 8, 16]),
        (True, -1, [0]),
        (True, 0, [0]),
        (False, 8, []),
    ],
)
def test_state_dependent_noise_uses_saved_training_cadence(
    use_sde: bool,
    resample_frequency: int,
    expected_reset_steps: list[int],
) -> None:
    reset_steps: list[int] = []
    current_step = 0

    class FakeActorPolicy:
        def reset_noise(self, *, n_envs: int) -> None:
            assert n_envs == 1
            reset_steps.append(current_step)

    class FakePPO:
        policy = FakeActorPolicy()

        def __init__(self) -> None:
            self.use_sde = use_sde
            self.sde_sample_freq = resample_frequency

    policy = FakePPO()
    for current_step in range(18):
        reset_state_dependent_noise_if_due(
            policy,
            current_step,
        )

    assert reset_steps == expected_reset_steps


def test_state_dependent_noise_rejects_negative_policy_step() -> None:
    class FakePPO:
        use_sde = False

    with pytest.raises(ValueError, match="policy_step must be nonnegative"):
        reset_state_dependent_noise_if_due(
            FakePPO(),
            -1,
        )


def test_main_cleans_temporary_videos_when_a_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = Path(".tmp/policy_grid_1234")
    checkpoint_path = Path(
        "checkpoints/wandb/m58de6rc/ppo_cube_stacker.zip"
    )
    cleaned_directories: list[Path] = []

    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: self == checkpoint_path,
    )
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    monkeypatch.setattr(demo.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        demo,
        "temporary_rollout_directory",
        lambda: directory,
    )
    monkeypatch.setattr(
        demo,
        "cleanup_temporary_directory",
        cleaned_directories.append,
    )

    def fail_rollouts(
        tasks: list[RolloutTask],
        worker_count: int,
    ) -> list[RolloutResult]:
        assert len(tasks) == ROLLOUT_COUNT
        assert all(
            task.checkpoint_path == checkpoint_path
            for task in tasks
        )
        assert worker_count == 3
        raise RuntimeError("worker failed")

    monkeypatch.setattr(
        demo,
        "run_parallel_rollouts",
        fail_rollouts,
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        demo.main(
            [
                "--workers",
                "3",
                "--repeat-wandb",
                "m58de6rc",
            ]
        )

    assert cleaned_directories == [directory]


def test_main_reports_missing_selected_wandb_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = Path(
        "checkpoints/wandb/missing123/ppo_cube_stacker.zip"
    )
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(FileNotFoundError, match=str(checkpoint_path)):
        demo.main(
            ["--repeat-wandb", "missing123"]
        )


def test_main_combines_videos_then_cleans_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = Path(".tmp/policy_grid_1234")
    temporary_paths = rollout_video_paths(directory)
    staging_output_path = directory / "stack_demo.mp4"
    cleaned_directories: list[Path] = []
    combined_videos: list[tuple[list[Path], Path]] = []
    replacements: list[tuple[Path, Path]] = []

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    monkeypatch.setattr(
        Path,
        "replace",
        lambda self, target: replacements.append((self, target)),
    )
    monkeypatch.setattr(demo.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        demo,
        "temporary_rollout_directory",
        lambda: directory,
    )
    monkeypatch.setattr(
        demo,
        "run_parallel_rollouts",
        lambda tasks, worker_count: [],
    )
    monkeypatch.setattr(
        demo,
        "combine_rollout_videos",
        lambda paths, output: combined_videos.append((list(paths), output)),
    )
    monkeypatch.setattr(
        demo,
        "print_rollout_summary",
        lambda results: None,
    )
    monkeypatch.setattr(
        demo,
        "cleanup_temporary_directory",
        cleaned_directories.append,
    )

    demo.main(["--workers", "4"])

    assert combined_videos == [(temporary_paths, staging_output_path)]
    assert replacements == [
        (staging_output_path, demo.OUTPUT_PATH)
    ]
    assert cleaned_directories == [directory]


@pytest.mark.parametrize("status", [SUCCESS, FAILURE, TRUNCATED])
def test_add_status_border_uses_status_color(status: str) -> None:
    frame = np.zeros(
        (PANEL_HEIGHT, PANEL_WIDTH, 3),
        dtype=np.uint8,
    )

    bordered_frame = add_status_border(frame, status)

    expected_color = STATUS_BORDER_COLORS[status]
    assert np.all(bordered_frame[:STATUS_BORDER_WIDTH] == expected_color)
    assert np.all(bordered_frame[-STATUS_BORDER_WIDTH:] == expected_color)
    assert np.all(
        bordered_frame[:, :STATUS_BORDER_WIDTH] == expected_color
    )
    assert np.all(
        bordered_frame[:, -STATUS_BORDER_WIDTH:] == expected_color
    )
    assert np.all(
        bordered_frame[
            STATUS_BORDER_WIDTH:-STATUS_BORDER_WIDTH,
            STATUS_BORDER_WIDTH:-STATUS_BORDER_WIDTH,
        ]
        == 0
    )
    assert np.all(frame == 0)


def test_gripperframe_marker_is_a_live_purple_sphere() -> None:
    environment = demo.CubeStackGymEnvironment()
    try:
        environment.reset(seed=31)
        data = environment.simulation.data
        scene = mujoco.MjvScene(
            environment.simulation.model,
            maxgeom=2,
        )
        expected_position = data.site("gripperframe").xpos.copy()

        demo.add_gripperframe_marker(scene, data)

        assert scene.ngeom == 1
        marker = scene.geoms[0]
        assert marker.type == mujoco.mjtGeom.mjGEOM_SPHERE
        np.testing.assert_allclose(
            marker.size,
            demo.GRIPPERFRAME_MARKER_RADIUS,
        )
        np.testing.assert_allclose(marker.pos, expected_position)
        np.testing.assert_allclose(
            marker.rgba,
            demo.GRIPPERFRAME_MARKER_COLOR,
        )
        assert marker.rgba[0] > 0.0
        assert marker.rgba[1] == 0.0
        assert marker.rgba[2] > 0.0
        assert marker.rgba[3] == 1.0
        assert marker.emission == pytest.approx(1.0)
        assert marker.category == mujoco.mjtCatBit.mjCAT_DECOR
        assert marker.segid == -1
    finally:
        environment.close()


def test_gripperframe_marker_reports_exhausted_scene_capacity() -> None:
    environment = demo.CubeStackGymEnvironment()
    try:
        environment.reset(seed=31)
        scene = mujoco.MjvScene(
            environment.simulation.model,
            maxgeom=1,
        )
        demo.add_gripperframe_marker(
            scene,
            environment.simulation.data,
        )

        with pytest.raises(RuntimeError, match="no capacity"):
            demo.add_gripperframe_marker(
                scene,
                environment.simulation.data,
            )
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("waypoint_reached", "expected_radius"),
    [
        (
            False,
            demo.APPROACH_TARGET_MARKER_RADIUS,
        ),
        (
            True,
            demo.ORANGE_CENTER_TARGET_MARKER_RADIUS,
        ),
    ],
)
def test_active_orange_target_marker_switches_from_waypoint_to_center(
    waypoint_reached: bool,
    expected_radius: float,
) -> None:
    environment = demo.CubeStackGymEnvironment()
    try:
        environment.reset(seed=31)
        data = environment.simulation.data
        scene = mujoco.MjvScene(
            environment.simulation.model,
            maxgeom=1,
        )
        orange_position = data.body("orange_cube").xpos.copy()
        expected_target = orange_position.copy()
        if not waypoint_reached:
            expected_target[2] += (
                environment.reward_config.approach_orange_height_offset
            )

        demo.add_active_orange_approach_target_marker(
            scene,
            data,
            waypoint_reached=waypoint_reached,
            approach_height_offset=(
                environment.reward_config.approach_orange_height_offset
            ),
        )

        assert scene.ngeom == 1
        marker = scene.geoms[0]
        assert marker.type == mujoco.mjtGeom.mjGEOM_SPHERE
        np.testing.assert_allclose(marker.size, expected_radius)
        np.testing.assert_allclose(marker.pos, expected_target)
        np.testing.assert_allclose(
            marker.rgba,
            demo.APPROACH_TARGET_MARKER_COLOR,
        )
        assert marker.rgba[0] == 0.0
        assert marker.rgba[1] == 1.0
        assert marker.rgba[2] == 0.0
        assert marker.rgba[3] == 1.0
        assert marker.emission == pytest.approx(1.0)
        assert marker.category == mujoco.mjtCatBit.mjCAT_DECOR
        assert marker.segid == -1
    finally:
        environment.close()


def test_active_orange_target_marker_reports_exhausted_scene_capacity() -> None:
    environment = demo.CubeStackGymEnvironment()
    try:
        environment.reset(seed=31)
        scene = mujoco.MjvScene(
            environment.simulation.model,
            maxgeom=1,
        )
        demo.add_gripperframe_marker(
            scene,
            environment.simulation.data,
        )

        with pytest.raises(RuntimeError, match="no capacity"):
            demo.add_active_orange_approach_target_marker(
                scene,
                environment.simulation.data,
                waypoint_reached=False,
                approach_height_offset=(
                    environment.reward_config.approach_orange_height_offset
                ),
            )
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("terminated", "truncated", "info", "expected_status"),
    [
        (True, False, {"is_success": True, "is_failure": False}, SUCCESS),
        (True, False, {"is_success": False, "is_failure": True}, FAILURE),
        (False, True, {"is_success": False, "is_failure": False}, TRUNCATED),
    ],
)
def test_rollout_status_identifies_terminal_reason(
    terminated: bool,
    truncated: bool,
    info: dict[str, bool],
    expected_status: str,
) -> None:
    assert rollout_status(terminated, truncated, info) == expected_status
