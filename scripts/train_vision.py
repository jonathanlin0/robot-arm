"""Train the shared two-view pose regressor from saved tensor shards.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/train_vision.py --config configs/default.json
See training.py for the required train/ and validation/ tensor file format.
Batch size and validation-driven learning-rate scheduling are configured under
training.vision in the JSON config.
"""

from argparse import ArgumentParser
from pathlib import Path

from schemas.config import load_config
from training import train_vision


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--log-file", "--log", type=Path, default=Path("logs/default.log"),
                        help="Append epoch, training loss, validation loss, and mean validation position "
                             "error in meters, plus current and next learning rates here; create it if needed "
                             "(default: logs/default.log).")
    arguments = parser.parse_args()
    config = load_config(path=arguments.config)
    train_vision(project=config, log_file=arguments.log_file)


if __name__ == "__main__":
    main()
