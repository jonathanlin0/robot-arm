"""Generate randomized external 300x300 and wrist 128x128 RGB training data.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/generate_vision_data.py --config configs/default.json
Each run adds independently randomized sequences to data/vision/train and
validation, saving five paired camera examples per sequence with the default
config. Existing data remains untouched. This generates perception labels,
not successful action demonstrations. Control runs at 20 Hz in simulated time
by default; cameras render only at the selected steps, without real-time waits.
The loader requires these per-camera sizes; regenerate older 128x128 external
data and restart existing generator processes to use the updated capture sizes.
"""

from argparse import ArgumentParser
from pathlib import Path

import mujoco

from schemas.config import load_config
from vision_data import generate_vision_data


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    arguments = parser.parse_args()
    project = load_config(path=arguments.config)
    generate_vision_data(project=project, renderer_factory=mujoco.Renderer)


if __name__ == "__main__":
    main()
