from pathlib import Path

import matplotlib
import pytest


matplotlib.use("Agg")

from scripts.plot_training_metrics import (  # noqa: E402
    load_training_metrics,
    plot_training_metrics,
)


def write_metrics(
    data_directory: Path,
    *,
    episode_lengths: list[float],
    episode_rewards: list[float],
    success_rates: list[float],
) -> None:
    data_directory.mkdir(parents=True, exist_ok=True)
    metric_values = {
        "ep_len_mean.txt": episode_lengths,
        "ep_rew_mean.txt": episode_rewards,
        "success_rate.txt": success_rates,
    }
    for file_name, values in metric_values.items():
        contents = "".join(f"{value}\n" for value in values)
        (data_directory / file_name).write_text(
            contents,
            encoding="utf-8",
        )


def test_load_training_metrics_reads_aligned_values(tmp_path: Path) -> None:
    write_metrics(
        tmp_path,
        episode_lengths=[400.0, 350.0],
        episode_rewards=[-2.0, 8.0],
        success_rates=[0.0, 0.25],
    )

    metrics = load_training_metrics(tmp_path)

    assert metrics == {
        "ep_len_mean": [400.0, 350.0],
        "ep_rew_mean": [-2.0, 8.0],
        "success_rate": [0.0, 0.25],
    }


def test_load_training_metrics_reads_optional_pickup_diagnostics(
    tmp_path: Path,
) -> None:
    write_metrics(
        tmp_path,
        episode_lengths=[400.0, 400.0],
        episode_rewards=[1.0, 2.0],
        success_rates=[0.0, 0.0],
    )
    (tmp_path / "orange_currently_held.txt").write_text(
        "0.0\n0.125\n",
        encoding="utf-8",
    )
    (tmp_path / "orange_grasp_hold_time.txt").write_text(
        "0.0\n0.35\n",
        encoding="utf-8",
    )

    metrics = load_training_metrics(tmp_path)

    assert metrics["orange_currently_held"] == [0.0, 0.125]
    assert metrics["orange_grasp_hold_time"] == [0.0, 0.35]


def test_load_training_metrics_rejects_different_rollout_counts(
    tmp_path: Path,
) -> None:
    write_metrics(
        tmp_path,
        episode_lengths=[400.0, 350.0],
        episode_rewards=[-2.0],
        success_rates=[0.0, 0.25],
    )

    with pytest.raises(ValueError, match="different rollout counts"):
        load_training_metrics(tmp_path)


@pytest.mark.parametrize(
    "metric_name",
    [
        "no_var_success_rate",
        "orange_waypoint_reach_rate",
        "action_std_x",
        "action_std_y",
        "action_std_z",
        "action_std_gripper",
        "action_clip_fraction",
    ],
)
def test_load_training_metrics_reads_available_optional_diagnostics(
    tmp_path: Path,
    metric_name: str,
) -> None:
    write_metrics(
        tmp_path,
        episode_lengths=[400.0, 350.0],
        episode_rewards=[1.0, 2.0],
        success_rates=[0.0, 0.25],
    )
    (tmp_path / f"{metric_name}.txt").write_text(
        "0.2\n0.15\n", encoding="utf-8"
    )

    metrics = load_training_metrics(tmp_path)

    assert metrics[metric_name] == [0.2, 0.15]
    assert len(metrics) == 4


@pytest.mark.parametrize(
    "metric_name",
    ["orange_waypoint_reach_rate", "action_std_x", "action_clip_fraction"],
)
def test_load_training_metrics_rejects_misaligned_optional_diagnostics(
    tmp_path: Path,
    metric_name: str,
) -> None:
    write_metrics(
        tmp_path,
        episode_lengths=[400.0, 350.0],
        episode_rewards=[1.0, 2.0],
        success_rates=[0.0, 0.25],
    )
    (tmp_path / f"{metric_name}.txt").write_text(
        "0.2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different rollout counts"):
        load_training_metrics(tmp_path)


def test_plot_training_metrics_saves_nonempty_image(tmp_path: Path) -> None:
    metrics = {
        "ep_len_mean": [400.0, 350.0],
        "ep_rew_mean": [-2.0, 8.0],
        "success_rate": [0.0, 0.25],
    }

    figure = plot_training_metrics(metrics)
    output_path = tmp_path / "training_metrics.png"
    figure.savefig(output_path)

    assert len(figure.axes) == 3
    assert output_path.stat().st_size > 0
