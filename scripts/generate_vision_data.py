"""Generate randomized two-camera vision data using configs/default.json.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/generate_vision_data.py --config configs/default.json
Each run adds independently randomized sequences to data/vision/train and
validation. Existing data remains untouched. This generates perception labels,
not successful action demonstrations.
"""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic, sleep

import mujoco

from camera_producers import SimCameraProducer
from schemas.config import load_config
from vision_data import generate_vision_data


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    arguments = parser.parse_args()
    project = load_config(path=arguments.config)
    generate_vision_data(project=project, renderer_factory=mujoco.Renderer,
                         producer_factory=SimCameraProducer, clock=monotonic, wait=sleep)


if __name__ == "__main__":
    main()
