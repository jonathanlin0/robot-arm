"""Split validation loss into position and orientation contributions.

Run from the repository root (no PYTHONPATH setup needed):
    python scripts/diagnostics/vision_loss_breakdown.py
    python scripts/diagnostics/vision_loss_breakdown.py --checkpoint checkpoints/vision/custom.pt

Uses the exact training loss with the selected config's weights: weighted XYZ
MSE in meters squared plus weighted sign-invariant normalized-quaternion MSE.
The default checkpoint and validation directory come from training.vision in
the selected config.
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
from vision_model import VisionModel, cube_pose_loss_components


def evaluate_loss_breakdown(model: VisionModel, loader: DataLoader,
                            device: torch.device, position_weight: float = 200.0,
                            orientation_weight: float = 1.0) -> tuple[int, Tensor]:
    """Return count and [orange/blue, position/orientation] weighted mean losses."""
    model.eval()
    component_sum = torch.zeros(2, 2, dtype=torch.float64)
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
            position, orientation = cube_pose_loss_components(prediction, poses)
            components = torch.stack((position_weight * position, orientation_weight * orientation), dim=-1)
            # Move to CPU before float64 accumulation; MPS inference stays float32.
            component_sum += components.cpu().double().sum(dim=0)
            count += len(joints)
    if count == 0:
        raise ValueError("Validation loader contains no examples.")
    return count, component_sum / count


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        help="Vision checkpoint path; defaults to training.vision.checkpoint_path in the config.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.json",
                        help="Config selecting the validation data, batch size, device, and default checkpoint.")
    arguments = parser.parse_args()
    config = load_config(arguments.config.expanduser()).training.vision
    checkpoint = arguments.checkpoint if arguments.checkpoint is not None else config.checkpoint_path
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
    count, per_cube_components = evaluate_loss_breakdown(
        model=model, loader=loader, device=device,
        position_weight=config.position_loss_weight, orientation_weight=config.orientation_loss_weight,
    )

    print(f"\nWeighted validation loss breakdown over {count:,} examples")
    print("Position: XYZ MSE (m^2). Orientation: sign-invariant normalized-quaternion MSE.")
    print(f"Loss weights from config: position={config.position_loss_weight:g}, "
          f"orientation={config.orientation_loss_weight:g}. Table shows weighted contributions.")
    print(f"{'Cube':<10} {'Position':>14} {'Orientation':>14} {'Total':>14} {'Position %':>12} {'Orientation %':>15}")
    for name, components in (("Orange", per_cube_components[0]), ("Blue", per_cube_components[1]),
                             ("Combined", per_cube_components.mean(dim=0))):
        position, orientation = components.tolist()
        total = position + orientation
        position_share = f"{100 * position / total:.2f}%" if total > 0 else "N/A"
        orientation_share = f"{100 * orientation / total:.2f}%" if total > 0 else "N/A"
        print(f"{name:<10} {position:>14.8f} {orientation:>14.8f} {total:>14.8f} "
              f"{position_share:>12} {orientation_share:>15}")
    print("Combined total uses the selected config's weights; the checkpoint's recorded loss may use older weights.")


if __name__ == "__main__":
    main()
