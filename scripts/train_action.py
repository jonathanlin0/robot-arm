"""Train causal action predictions from saved full-episode tensor files.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/train_action.py --config configs/default.json
See training.py for the required train/ and validation/ tensor file format.
"""

from argparse import ArgumentParser
from pathlib import Path

from schemas.config import load_config
from training import train_action


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    arguments = parser.parse_args()
    config = load_config(path=arguments.config)
    train_action(project=config)


if __name__ == "__main__":
    main()
