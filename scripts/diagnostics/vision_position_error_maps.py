"""Evaluate vision validation data and plot 3-D error against true X/Y and Z.

Run from the repository root (no PYTHONPATH setup needed):
    .venv/bin/python scripts/diagnostics/vision_position_error_maps.py --device mps
    .venv/bin/python scripts/diagnostics/vision_position_error_maps.py --checkpoint PATH

Uses every saved validation example, without augmentation or weight updates.
Both plots show sqrt(dx**2 + dy**2 + dz**2), pooling orange and blue predictions.
The XY map averages that distance in spatial cells with logarithmic colors; the Z plot shows every
prediction plus height-bin means. Positions use centimeters and errors use
millimeters. Empty bins remain empty; no samples or outliers are discarded.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from schemas.config import load_config
from training import create_loader, load_vision_checkpoint, load_vision_dataset, resolve_device


def _validate_bin_inputs(*arrays: np.ndarray, bin_width_cm: float) -> tuple[np.ndarray, ...]:
    """Validate one-dimensional observations without silently discarding data."""
    if not np.isfinite(bin_width_cm) or bin_width_cm <= 0:
        raise ValueError("Bin width must be finite and positive.")
    converted = tuple(np.asarray(array, dtype=np.float64) for array in arrays)
    if any(array.ndim != 1 for array in converted):
        raise ValueError("Binning inputs must be one-dimensional.")
    if not converted or not len(converted[0]):
        raise ValueError("Binning inputs must contain at least one observation.")
    if any(array.shape != converted[0].shape for array in converted):
        raise ValueError("Binning inputs must have matching shapes.")
    if any(not np.isfinite(array).all() for array in converted):
        raise ValueError("Binning inputs must contain only finite values.")
    return converted


def _bin_edges(coordinates_cm: np.ndarray, bin_width_cm: float) -> np.ndarray:
    """Align bins to coordinate zero, including a bin for a constant coordinate."""
    first = int(np.floor(coordinates_cm.min() / bin_width_cm))
    last = int(np.floor(coordinates_cm.max() / bin_width_cm)) + 1
    return np.arange(first, last + 1, dtype=np.float64) * bin_width_cm


def bin_xy_mean(x_cm: np.ndarray, y_cm: np.ndarray, error_mm: np.ndarray,
                bin_width_cm: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X/Y edges, per-prediction mean error, and counts in [X bin, Y bin].

    Empty cells have a NaN mean and a zero count. Multiple cube predictions at
    the same location contribute individually, rather than giving each cube
    or sequence equal weight regardless of its number of samples.
    """
    x_cm, y_cm, error_mm = _validate_bin_inputs(x_cm, y_cm, error_mm, bin_width_cm=bin_width_cm)
    if np.any(error_mm < 0):
        raise ValueError("Distance errors cannot be negative.")
    x_edges = _bin_edges(x_cm, bin_width_cm)
    y_edges = _bin_edges(y_cm, bin_width_cm)
    counts, _, _ = np.histogram2d(x_cm, y_cm, bins=(x_edges, y_edges))
    sums, _, _ = np.histogram2d(x_cm, y_cm, bins=(x_edges, y_edges), weights=error_mm)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return x_edges, y_edges, means, counts.astype(np.int64)


def bin_z_mean(z_cm: np.ndarray, error_mm: np.ndarray,
               bin_width_cm: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Z edges, per-prediction mean error, and counts; empty means are NaN."""
    z_cm, error_mm = _validate_bin_inputs(z_cm, error_mm, bin_width_cm=bin_width_cm)
    if np.any(error_mm < 0):
        raise ValueError("Distance errors cannot be negative.")
    edges = _bin_edges(z_cm, bin_width_cm)
    counts, _ = np.histogram(z_cm, bins=edges)
    sums, _ = np.histogram(z_cm, bins=edges, weights=error_mm)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return edges, means, counts


def plot_position_error_maps(true_xyz_m: np.ndarray, error_m: np.ndarray,
                             output_dir: Path, epoch: int | None = None, *,
                             bin_width_cm: float = 1.0) -> tuple[Path, Path]:
    """Save XY heatmap and Z scatter plot, pooling both cubes for spatial means.

    Inputs are [example, orange/blue, XYZ] coordinates in meters and
    [example, orange/blue] Euclidean center-distance errors in meters. This
    function never changes or clips errors, interpolates empty cells, or
    samples away outliers. PNGs are accompanied by SVGs and bin counts.
    """
    true_xyz_m = np.asarray(true_xyz_m, dtype=np.float64)
    error_m = np.asarray(error_m, dtype=np.float64)
    if true_xyz_m.ndim != 3 or true_xyz_m.shape[1:] != (2, 3):
        raise ValueError("Ground-truth positions must have shape [examples, 2, 3].")
    if error_m.shape != true_xyz_m.shape[:2]:
        raise ValueError("Distance errors must have shape [examples, 2].")
    if not len(true_xyz_m):
        raise ValueError("At least one validation example is required.")
    if not np.isfinite(true_xyz_m).all() or not np.isfinite(error_m).all():
        raise ValueError("Positions and errors must contain only finite values.")
    if np.any(error_m < 0):
        raise ValueError("Distance errors cannot be negative.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_cm = true_xyz_m * 100.0
    errors_mm = error_m * 1000.0
    pooled_positions = positions_cm.reshape(-1, 3)
    pooled_errors = errors_mm.reshape(-1)
    x_edges, y_edges, xy_means, xy_counts = bin_xy_mean(
        pooled_positions[:, 0], pooled_positions[:, 1], pooled_errors, bin_width_cm,
    )
    z_edges, z_means, z_counts = bin_z_mean(pooled_positions[:, 2], pooled_errors, bin_width_cm)
    number_examples = len(true_xyz_m)
    number_predictions = pooled_errors.size
    epoch_text = f"Epoch {epoch} · " if epoch is not None else ""
    subtitle = f"{epoch_text}{number_examples:,} validation examples · {number_predictions:,} cube predictions"
    xy_path = output_dir / "spatial_error_xy.png"
    z_path = output_dir / "spatial_error_z.png"
    occupied_counts = xy_counts[xy_counts > 0]
    positive_means = xy_means[xy_means > 0]
    color_min = float(positive_means.min()) if positive_means.size else 0.001
    color_max = float(positive_means.max()) if positive_means.size else 1.0
    if color_max <= color_min:
        color_max = color_min * 10.0
    has_zero_means = bool(np.any(xy_means == 0))
    empty_color = "#e5e7eb"
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(8.0, 9.0))
        figure.subplots_adjust(left=0.12, right=0.86, top=0.83, bottom=0.23)
        figure.suptitle("Position error across the workspace", fontsize=18, fontweight="bold", y=0.965)
        figure.text(0.5, 0.915, subtitle, ha="center", fontsize=11, color="#475569")
        axis.set_title(f"Orange + blue cubes pooled · {bin_width_cm:g} × {bin_width_cm:g} cm cells", pad=13)
        axis.set_facecolor(empty_color)
        colormap = plt.get_cmap("coolwarm").copy()
        colormap.set_bad(empty_color)
        # Zero errors use the lowest color; missing cells retain their mask.
        mesh = axis.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(xy_means.T), cmap=colormap,
            norm=LogNorm(vmin=color_min, vmax=color_max, clip=True),
            shading="flat", rasterized=True,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("True cube-center X (cm)", labelpad=10)
        axis.set_ylabel("True cube-center Y (cm)", labelpad=10)
        axis.set_xlim(x_edges[0], x_edges[-1])
        axis.set_ylim(y_edges[0], y_edges[-1])
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.035, fraction=0.055, shrink=0.92)
        colorbar.set_label("Mean 3D distance error (mm, log scale)", labelpad=12, fontsize=13)
        colorbar.locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        colorbar.formatter = ScalarFormatter()
        colorbar.update_ticks()
        axis.set_anchor("C")
        figure.legend(handles=[Patch(facecolor=empty_color, edgecolor="#94a3b8", label="No validation samples")],
                      loc="center", bbox_to_anchor=(0.5, 0.11), frameon=False, fontsize=10)
        footer = ("Log colors show mean distance from the true center to the predicted center.\n"
                  "No spatial interpolation. Sparse cells may have less reliable means.")
        if has_zero_means:
            footer += "\nZero-error cells use the lowest color."
        figure.text(0.5, 0.055, footer,
                    ha="center", va="center", fontsize=10, color="#475569", linespacing=1.5)
        figure.savefig(xy_path, dpi=200)
        figure.savefig(xy_path.with_suffix(".svg"))
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.8, 6.8))
        figure.subplots_adjust(left=0.12, right=0.97, top=0.77, bottom=0.20)
        figure.suptitle("Position error versus cube height", fontsize=18, fontweight="bold", y=0.965)
        figure.text(0.5, 0.91, subtitle, ha="center", fontsize=11, color="#475569")
        for cube_index, (name, color) in enumerate((("Orange cube", "#df6d11"), ("Blue cube", "#2463b8"))):
            axis.scatter(positions_cm[:, cube_index, 2], errors_mm[:, cube_index],
                         s=13, alpha=0.30, c=color, linewidths=0, label=name, rasterized=True)
        centers = (z_edges[:-1] + z_edges[1:]) / 2.0
        # NaNs break the line across empty height bins; no gap is interpolated.
        axis.plot(centers, z_means, color="#111827", marker="o", markersize=5.5,
                  linewidth=2.0, label=f"Mean in {bin_width_cm:g} cm Z bins", zorder=5)
        axis.set_xlabel("True cube-center Z (cm)", labelpad=10)
        axis.set_ylabel("3D distance error (mm)", labelpad=10)
        axis.set_xlim(z_edges[0], z_edges[-1])
        maximum_error = float(pooled_errors.max())
        axis.set_ylim(0, maximum_error * 1.07 if maximum_error > 0 else 1.0)
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        axis.set_axisbelow(True)
        legend = axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=3,
                             frameon=False, fontsize=10, handletextpad=0.5, columnspacing=1.0)
        for handle in legend.legend_handles[:2]:
            handle.set_alpha(1.0)
        figure.text(0.5, 0.07,
                    "Every cube prediction is shown, including outliers. Black dots are pooled means.\n"
                    "Vertical axis measures full XYZ distance error, not only Z-coordinate error.",
                    ha="center", va="center", fontsize=10, color="#475569", linespacing=1.5)
        figure.savefig(z_path, dpi=200)
        figure.savefig(z_path.with_suffix(".svg"))
        plt.close(figure)

    summary = {
        "validation_examples": number_examples,
        "cube_predictions": number_predictions,
        "checkpoint_epoch": epoch,
        "bin_width_cm": bin_width_cm,
        "aggregation": "Arithmetic mean of Euclidean 3D distance error per cube prediction; both cubes pooled.",
        "xy": {
            "color_scale": "log",
            "color_limits_mm": [color_min, color_max],
            "zero_error_cells_use_lowest_color": has_zero_means,
            "x_edges_cm": x_edges.tolist(),
            "y_edges_cm": y_edges.tolist(),
            "counts_x_then_y": xy_counts.tolist(),
            "means_mm_x_then_y": [[float(value) if np.isfinite(value) else None for value in row]
                                    for row in xy_means],
            "occupied_cells": int(len(occupied_counts)),
            "samples_per_occupied_cell_min": int(occupied_counts.min()),
            "samples_per_occupied_cell_median": float(np.median(occupied_counts)),
            "samples_per_occupied_cell_max": int(occupied_counts.max()),
            "mean_error_mm_range": [float(np.nanmin(xy_means)), float(np.nanmax(xy_means))],
        },
        "z": {
            "edges_cm": z_edges.tolist(),
            "counts": z_counts.tolist(),
            "means_mm": [float(value) if np.isfinite(value) else None for value in z_means],
        },
        "true_xyz_cm_min": pooled_positions.min(axis=0).tolist(),
        "true_xyz_cm_max": pooled_positions.max(axis=0).tolist(),
        "individual_error_mm_range": [float(pooled_errors.min()), float(pooled_errors.max())],
        "mean_distance_error_mm": float(pooled_errors.mean()),
    }
    (output_dir / "spatial_error_bins.json").write_text(json.dumps(summary, indent=2) + "\n")
    return xy_path, z_path


def collect_position_predictions(model: torch.nn.Module, loader: DataLoader,
                                 device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true/predicted XYZ [N, 2, 3] and per-cube 3-D distances [N, 2].

    Cube order is orange, blue. Arrays use float64 meters on the CPU, including
    when inference runs on MPS; image preprocessing matches training validation.
    """
    model.eval()
    true_parts, predicted_parts = [], []
    with torch.inference_mode():
        for wrist, external, joints, poses in tqdm(loader, desc="Evaluating validation", unit="batch"):
            images_are_bytes = wrist.dtype == torch.uint8
            wrist = wrist.to(device=device, dtype=torch.float32)
            external = external.to(device=device, dtype=torch.float32)
            if images_are_bytes:
                wrist, external = wrist / 255.0, external / 255.0
            prediction = model(wrist, external, joints.to(device=device))
            count = len(joints)
            if prediction.shape != (count, 10) or poses.shape != (count, 10):
                raise ValueError("Vision predictions and labels must have shape [batch, 10].")
            # Transfer before float64 conversion to avoid MPS-to-CPU casting issues.
            predicted_xyz = prediction.reshape(count, 2, 5)[..., :3].cpu().double().numpy()
            true_xyz = poses.reshape(count, 2, 5)[..., :3].cpu().double().numpy()
            if not np.isfinite(predicted_xyz).all() or not np.isfinite(true_xyz).all():
                raise ValueError("Predicted and ground-truth positions must contain only finite values.")
            predicted_parts.append(predicted_xyz)
            true_parts.append(true_xyz)
    if not true_parts:
        raise ValueError("Validation loader contains no examples.")
    true_xyz_m = np.concatenate(true_parts, axis=0)
    predicted_xyz_m = np.concatenate(predicted_parts, axis=0)
    error_m = np.linalg.norm(predicted_xyz_m - true_xyz_m, axis=-1)
    return true_xyz_m, predicted_xyz_m, error_m


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--checkpoint", type=Path,
                        help="Defaults to training.vision.checkpoint_path in the config.")
    parser.add_argument("--device", help="Inference device; defaults to training.vision.device.")
    parser.add_argument("--batch-size", type=int,
                        help="Inference batch size; defaults to training.vision.batch_size.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports/vision_position_error_maps")
    parser.add_argument("--bin-width-cm", type=float, default=1.0,
                        help="XY cell width and Z mean-bin width in centimeters (default: 1).")
    arguments = parser.parse_args()
    if arguments.batch_size is not None and arguments.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not np.isfinite(arguments.bin_width_cm) or arguments.bin_width_cm <= 0:
        parser.error("--bin-width-cm must be finite and positive")
    config = load_config(arguments.config.expanduser()).training.vision
    if arguments.batch_size is not None:
        config = config.model_copy(update={"batch_size": arguments.batch_size})
    checkpoint = (arguments.checkpoint or config.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    device = resolve_device(arguments.device or config.device)
    model = load_vision_checkpoint(checkpoint, device)
    checkpoint_record = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    epoch = int(checkpoint_record["epoch"])
    checkpoint_loss = float(checkpoint_record["validation_loss"])
    del checkpoint_record
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    dataset = load_vision_dataset(config.data_dir / "validation", show_progress=True)
    loader = create_loader(dataset, config, training=False)
    tqdm.write(f"Checkpoint: {checkpoint}; epoch={epoch}; device={device}")
    tqdm.write(f"Validation only: {len(dataset):,} examples; batch size={config.batch_size}")
    true_xyz, predicted_xyz, error_m = collect_position_predictions(model, loader, device)
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_position_predictions.npz"
    np.savez_compressed(prediction_path, true_xyz_m=true_xyz, predicted_xyz_m=predicted_xyz,
                        error_m=error_m, cube_names=np.array(["orange", "blue"]),
                        checkpoint_epoch=np.array(epoch), checkpoint_sha256=np.array(checkpoint_hash))
    paths = plot_position_error_maps(true_xyz, error_m, output_dir, epoch,
                                     bin_width_cm=arguments.bin_width_cm)
    manifest = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": epoch, "checkpoint_validation_loss": checkpoint_loss,
        "validation_directory": str(config.data_dir / "validation"),
        "validation_examples": len(true_xyz), "cube_predictions": int(error_m.size),
        "device": str(device), "batch_size": config.batch_size,
        "bin_width_cm": arguments.bin_width_cm,
        "metric": "sqrt((pred_x-true_x)^2 + (pred_y-true_y)^2 + (pred_z-true_z)^2)",
        "mean_distance_error_mm": float(error_m.mean() * 1000),
        "per_cube_mean_distance_error_mm": dict(zip(("orange", "blue"), (error_m.mean(axis=0) * 1000).tolist())),
        "prediction_file": str(prediction_path), "plots": [str(path) for path in paths],
    }
    (output_dir / "spatial_error_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tqdm.write(f"Mean 3D center error: {manifest['mean_distance_error_mm']:.3f} mm")
    for path in paths:
        tqdm.write(f"Saved {path}")


if __name__ == "__main__":
    main()
