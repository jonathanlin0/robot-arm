"""Show 20 shuffled, augmented training images in a 5-column, 4-row grid.

    python scripts/temp_view_vision_samples.py
    python scripts/temp_view_vision_samples.py --camera wrist
    python scripts/temp_view_vision_samples.py --save .tmp/vision_samples.png --no-show

Uses the same in-memory training DataLoader and RandAugment as vision training.
All saved samples from every training shard are loaded up front.
The shared loader accepts direct-yaw labels and converts legacy quaternion
labels in memory; the preview displays the images without modifying the data.
External images are stored and displayed at 300x300. Wrist images are stored at
128x128 and enlarged to 300x300 for this preview only. The loader requires these
per-camera stored sizes; regenerate datasets with 128x128 external images.
No PYTHONPATH setup is needed.
"""

import argparse
from pathlib import Path
import secrets
import sys

from PIL import Image, ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from schemas.config import VisionTrainingConfig, load_config
from training import create_vision_training_loader, seed_training


GRID_COLUMNS = 5
GRID_ROWS = 4
IMAGE_SIZE = 300  # Display size; stored external frames are 300x300, wrist 128x128.
LABEL_HEIGHT = 24


def build_grid(config: VisionTrainingConfig, camera: str, data_dir: Path | None = None) -> Image.Image:
    """Take the first 20 samples from the training loader's shuffled batches."""
    if camera not in ("external", "wrist"):
        raise ValueError(f"Unknown camera: {camera}")
    loader = create_vision_training_loader(config=config, show_progress=True, directory=data_dir)
    total_samples = len(loader.dataset)
    sample_count = GRID_COLUMNS * GRID_ROWS
    if total_samples < sample_count:
        directory = data_dir if data_dir is not None else config.data_dir / "train"
        raise ValueError(f"Need {sample_count} samples; found {total_samples} in {directory}.")

    grid = Image.new(
        "RGB", (GRID_COLUMNS * IMAGE_SIZE, GRID_ROWS * (IMAGE_SIZE + LABEL_HEIGHT)),
        color="#202020",
    )
    print(f"Showing {sample_count}/{total_samples:,} training samples from the shuffled DataLoader.")
    draw = ImageDraw.Draw(grid)
    slot = 0
    camera_index = 0 if camera == "wrist" else 1
    for batch in loader:
        for image in batch[camera_index]:
            if image.dtype != torch.uint8:
                image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
            frame = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
            frame = frame.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
            x = (slot % GRID_COLUMNS) * IMAGE_SIZE
            y = (slot // GRID_COLUMNS) * (IMAGE_SIZE + LABEL_HEIGHT)
            grid.paste(frame, (x, y + LABEL_HEIGHT))
            draw.text((x + 6, y + 6), f"{camera} | sample {slot + 1} | augmented", fill="white")
            slot += 1
            if slot == sample_count:
                return grid
    raise ValueError(f"DataLoader yielded only {slot} images; expected {sample_count}.")


def main(*, default_camera: str = "external", description: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__ if description is None else description)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.json")
    parser.add_argument("--data-dir", type=Path, help="Override the training shard directory.")
    parser.add_argument("--camera", choices=("external", "wrist"), default=default_camera)
    parser.add_argument("--seed", type=int, help="Repeat the same sample order and augmentation; default is random.")
    parser.add_argument("--save", type=Path, metavar="PNG", help="Also save the grid.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the image viewer.")
    arguments = parser.parse_args()

    config = load_config(arguments.config.expanduser()).training.vision
    # Preview draws a fresh shuffled selection each run without shuffling tensors.
    config = config.model_copy(update={"seed": secrets.randbits(63) if arguments.seed is None else arguments.seed,
                                       "shuffle_train": True})
    if config.seed < 0:
        parser.error("--seed must be non-negative")
    seed_training(seed=config.seed)
    print(f"Preview seed: {config.seed}")
    directory = arguments.data_dir.expanduser() if arguments.data_dir is not None else None
    grid = build_grid(config, arguments.camera, data_dir=directory)
    if arguments.save is not None:
        destination = arguments.save.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        grid.save(destination)
        print(f"Saved grid: {destination.resolve()}")
    if not arguments.no_show:
        grid.show(title=f"20 random augmented {arguments.camera} training images")


if __name__ == "__main__":
    main()
