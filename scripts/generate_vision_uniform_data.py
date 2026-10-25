"""Generate vision examples at uniform random gripper-frame XYZ targets.

    .venv/bin/python scripts/generate_vision_uniform_data.py --samples 5000

Targets use the configured cube spawn XY bounds and uniform Z=3–8 cm. Orange
is physically picked up whenever needed; its true pose supplies the label.
Blue locations and camera settings are resampled each attempt. Scene surfaces,
lights and textures use the original visual randomization on every new scene.
Failures are logged and the scene is recreated before the next attempt.

Saves one example per UUID.pt, appending to the configured vision train and
validation directories. --samples counts successful saves (default 5000), with
the existing train_fraction. Each persistent scene belongs to just one split.
The JSON-lines event log includes every UUID and failure, but no progress bar.
No trained models or real-time pacing are used.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from schemas.config import load_config
from vision_workspace_data import DEFAULT_SAMPLES, generate_workspace_vision_data


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of successfully saved examples (default: 5000).")
    parser.add_argument("--output-dir", type=Path, help="Overrides training.vision.data_dir for this run.")
    parser.add_argument("--log-file", type=Path, help="Intentional event log; default: logs/vision-workspace-<run>.log.")
    parser.add_argument("--max-attempts", type=int, help="Total attempt limit; default: 10 times --samples.")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.max_attempts is not None and args.max_attempts < args.samples:
        parser.error("--max-attempts must be at least --samples")
    project = load_config(args.config.expanduser())
    if args.output_dir is not None:
        project.training.vision.data_dir = args.output_dir.expanduser().resolve()
    generate_workspace_vision_data(project=project, samples=args.samples, max_attempts=args.max_attempts,
                                   log_file=args.log_file.expanduser().resolve() if args.log_file else None)


if __name__ == "__main__":
    main()
