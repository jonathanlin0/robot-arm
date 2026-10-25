"""Train causal action predictions and validate with learned-vision rollouts.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/train_action.py --config configs/default.json
See training.py for the required train/ and validation/ tensor file format.
The configured vision checkpoint is required for simulation validation.
"""

from argparse import ArgumentParser
from pathlib import Path

from schemas.config import load_config
from training import train_action


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--log-file", "--log", type=Path, default=Path("logs/action.log"),
                        help="Append epoch, training loss, validation loss, and rollout success "
                             "percentage here; create it if needed (default: logs/action.log).")
    arguments = parser.parse_args()
    config = load_config(path=arguments.config)
    train_action(project=config, log_file=arguments.log_file)


if __name__ == "__main__":
    main()
