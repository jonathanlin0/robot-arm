"""Report validation mean absolute X, Y, and Z error for a saved vision model.

Run from the repository root (no PYTHONPATH setup needed):
    python scripts/diagnostics/vision_position_error.py
    python scripts/diagnostics/vision_position_error.py --checkpoint checkpoints/vision/custom.pt

The default checkpoint and validation data directory come from training.vision
in the selected config. With the default config, these are
checkpoints/vision/default.pt and data/vision/validation/.
Loads every saved validation example, as the training script does.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from schemas.config import load_config
from training import create_loader, load_vision_checkpoint, load_vision_dataset, resolve_device
from vision_model import VisionModel


def evaluate_position_error(model: VisionModel, loader: DataLoader,
                            device: torch.device) -> tuple[int, Tensor]:
    """Return sample count and [orange/blue, X/Y/Z] mean absolute errors in meters."""
    model.eval()
    # Accumulate in float64 on the CPU, including when inference uses MPS.
    absolute_error_sum = torch.zeros(2, 3, dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for wrist, external, joints, poses in tqdm(loader, desc="Evaluating validation", unit="batch"):
            images_are_bytes = wrist.dtype == torch.uint8
            wrist = wrist.to(device=device, dtype=torch.float32)
            external = external.to(device=device, dtype=torch.float32)
            if images_are_bytes:
                wrist, external = wrist / 255.0, external / 255.0
            joints, poses = joints.to(device), poses.to(device)
            prediction = model(wrist, external, joints)
            batch_count = len(joints)
            predicted_xyz = prediction.reshape(batch_count, 2, 5)[..., :3]
            expected_xyz = poses.reshape(batch_count, 2, 5)[..., :3]
            errors = (predicted_xyz - expected_xyz).abs()
            # Cast only after transfer: combined MPS-to-CPU float64 conversion
            # can silently return zeros on MPS.
            absolute_error_sum += errors.cpu().double().sum(dim=0)
            count += batch_count
    if count == 0:
        raise ValueError("Validation loader contains no examples.")
    return count, absolute_error_sum / count


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        help="Vision checkpoint path; defaults to training.vision.checkpoint_path in the config.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.json",
                        help="Config selecting the validation data, batch size, device, and default checkpoint.")
    arguments = parser.parse_args()
    config = load_config(arguments.config.expanduser()).training.vision
    checkpoint = (arguments.checkpoint if arguments.checkpoint is not None else config.checkpoint_path)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(name=config.device)
    model = load_vision_checkpoint(path=checkpoint, device=device)
    directory = config.data_dir / "validation"
    dataset = load_vision_dataset(directory=directory, show_progress=True)
    loader = create_loader(dataset=dataset, config=config, training=False)
    print(f"Checkpoint: {checkpoint}")
    print(f"Validation data: {directory}")
    print(f"Device: {device}; validation examples: {len(dataset):,}")
    count, per_cube_mae = evaluate_position_error(model=model, loader=loader, device=device)

    print(f"\nMean absolute position error over {count:,} validation examples (cm)")
    print(f"{'Cube':<10} {'X':>12} {'Y':>12} {'Z':>12}")
    for name, errors in (("Orange", per_cube_mae[0]), ("Blue", per_cube_mae[1]),
                         ("Combined", per_cube_mae.mean(dim=0))):
        x, y, z = (errors * 100.0).tolist()
        print(f"{name:<10} {x:>12.4f} {y:>12.4f} {z:>12.4f}")


if __name__ == "__main__":
    main()
