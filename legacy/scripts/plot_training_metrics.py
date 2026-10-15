#!/usr/bin/env python3

"""Plot the three rollout metrics written by ``src/train.py``.

Run from the repository root:

    python scripts/plot_training_metrics.py

Save without opening a window:

    python scripts/plot_training_metrics.py \
        --save data/training_metrics.png \
        --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING


# Scripts in this project are run from the repository root.
sys.path.insert(0, "src")

from train import ROLLOUT_METRIC_FILE_NAMES  # noqa: E402


if TYPE_CHECKING:
    from matplotlib.figure import Figure


METRIC_COLORS = {
    "ep_len_mean": "tab:blue",
    "ep_rew_mean": "tab:orange",
    "success_rate": "tab:green",
}

# Pickup and exploration diagnostics were added after the original three metrics.
# Treat them as optional so historical run directories remain plottable.
OPTIONAL_METRIC_NAMES = {
    "no_var_success_rate",
    "orange_waypoint_reach_rate",
    "orange_currently_held",
    "orange_grasp_hold_time",
    "action_std_x",
    "action_std_y",
    "action_std_z",
    "action_std_gripper",
    "action_clip_fraction",
}


def read_metric_file(metric_path: Path) -> list[float]:
    """Read one floating-point value per rollout from a text file."""
    if not metric_path.exists():
        raise FileNotFoundError(
            f"Training metric file does not exist: {metric_path}"
        )

    values: list[float] = []
    for line_number, line in enumerate(
        metric_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        try:
            values.append(float(stripped_line))
        except ValueError as error:
            raise ValueError(
                f"Invalid number in {metric_path} on line "
                f"{line_number}: {stripped_line!r}"
            ) from error

    if not values:
        raise ValueError(f"Training metric file is empty: {metric_path}")

    return values


def load_training_metrics(
    data_directory: Path | str,
) -> dict[str, list[float]]:
    """Load aligned rollout metrics from the training data directory."""
    directory = Path(data_directory)
    metrics: dict[str, list[float]] = {}
    for metric_name, file_name in ROLLOUT_METRIC_FILE_NAMES.items():
        metric_path = directory / file_name
        if metric_name in OPTIONAL_METRIC_NAMES and not metric_path.exists():
            continue
        metrics[metric_name] = read_metric_file(metric_path)

    metric_lengths = {len(values) for values in metrics.values()}
    if len(metric_lengths) != 1:
        lengths_by_name = {
            metric_name: len(values)
            for metric_name, values in metrics.items()
        }
        raise ValueError(
            "Training metric files contain different rollout counts: "
            f"{lengths_by_name}"
        )

    return metrics


def plot_training_metrics(
    metrics: dict[str, list[float]],
) -> Figure:
    """Create one three-line plot while preserving each metric's scale."""
    import matplotlib.pyplot as plt

    rollout_count = len(metrics["ep_len_mean"])
    rollout_numbers = range(1, rollout_count + 1)

    figure, episode_length_axis = plt.subplots(figsize=(11, 6))
    reward_axis = episode_length_axis.twinx()
    success_axis = episode_length_axis.twinx()
    success_axis.spines["right"].set_position(("outward", 65))

    episode_length_line = episode_length_axis.plot(
        rollout_numbers,
        metrics["ep_len_mean"],
        color=METRIC_COLORS["ep_len_mean"],
        label="Mean episode length",
    )[0]
    reward_line = reward_axis.plot(
        rollout_numbers,
        metrics["ep_rew_mean"],
        color=METRIC_COLORS["ep_rew_mean"],
        label="Mean episode reward",
    )[0]
    success_line = success_axis.plot(
        rollout_numbers,
        metrics["success_rate"],
        color=METRIC_COLORS["success_rate"],
        label="Success rate",
    )[0]

    episode_length_axis.set_xlabel("Rollout")
    episode_length_axis.set_ylabel(
        "Mean episode length (steps)",
        color=METRIC_COLORS["ep_len_mean"],
    )
    reward_axis.set_ylabel(
        "Mean episode reward",
        color=METRIC_COLORS["ep_rew_mean"],
    )
    success_axis.set_ylabel(
        "Success rate",
        color=METRIC_COLORS["success_rate"],
    )
    success_axis.set_ylim(-0.02, 1.02)

    episode_length_axis.tick_params(
        axis="y",
        colors=METRIC_COLORS["ep_len_mean"],
    )
    reward_axis.tick_params(
        axis="y",
        colors=METRIC_COLORS["ep_rew_mean"],
    )
    success_axis.tick_params(
        axis="y",
        colors=METRIC_COLORS["success_rate"],
    )
    episode_length_axis.grid(alpha=0.25)
    episode_length_axis.set_title("PPO cube-stacking training metrics")
    episode_length_axis.legend(
        handles=[episode_length_line, reward_line, success_line],
        loc="best",
    )
    figure.subplots_adjust(right=0.78)

    return figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the rollout metrics written during PPO training."
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data"),
        help="Directory containing the three metric text files.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path where the plotted figure should be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    arguments = parser.parse_args()
    if arguments.no_show and arguments.save is None:
        parser.error("--no-show requires --save")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    if arguments.no_show:
        import matplotlib

        matplotlib.use("Agg")

    metrics = load_training_metrics(arguments.data_directory)
    figure = plot_training_metrics(metrics)

    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.save, dpi=150)
        print(f"Saved training plot to {arguments.save}")

    if not arguments.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
