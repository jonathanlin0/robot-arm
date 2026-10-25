"""Generate full action demonstrations with learned vision and override drops.

Run from the repository root:
    .venv/bin/python scripts/generate_action_data.py --config configs/default.json
    .venv/bin/python scripts/generate_action_data.py --vision-checkpoint checkpoints/vision/custom.pt

Each example is a successful full episode, including any drop/recovery history.
Existing data is preserved. Both simulated cameras feed a frozen vision model
before each action; its cube estimates become the saved policy inputs. Teacher
labels and success checks still use simulator truth. There are no pacing sleeps.
Drop, recovery, and teacher settings live in environment.action_data.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_data import generate_action_data
from schemas.config import load_config


# Number of new successful episodes to add to each split per run.
NUMBER_OF_TRAINING_EXAMPLES = 5000
NUMBER_OF_VALIDATION_EXAMPLES = 1000


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--vision-checkpoint", type=Path,
                        help="Trained vision weights; defaults to training.vision.checkpoint_path in the config.")
    parser.add_argument("--device", help="Vision inference device; defaults to training.vision.device in the config.")
    parser.add_argument("--training-examples", type=int,
                        help=f"Training episodes to add (default: {NUMBER_OF_TRAINING_EXAMPLES}).")
    parser.add_argument("--validation-examples", type=int,
                        help=f"Validation episodes to add (default: {NUMBER_OF_VALIDATION_EXAMPLES}).")
    parser.add_argument("--examples", type=int,
                        help="Total episodes to add using the configured train_fraction; cannot combine with split counts.")
    arguments = parser.parse_args()
    if arguments.examples is not None:
        if arguments.training_examples is not None or arguments.validation_examples is not None:
            parser.error("--examples cannot be combined with --training-examples or --validation-examples")
        if arguments.examples < 1:
            parser.error("--examples must be positive")
        examples = arguments.examples
        validation_examples = None
    else:
        training_examples = (NUMBER_OF_TRAINING_EXAMPLES if arguments.training_examples is None
                             else arguments.training_examples)
        validation_examples = (NUMBER_OF_VALIDATION_EXAMPLES if arguments.validation_examples is None
                               else arguments.validation_examples)
        if training_examples < 0 or validation_examples < 0:
            parser.error("--training-examples and --validation-examples must be nonnegative")
        examples = training_examples + validation_examples
        if examples < 1:
            parser.error("At least one training or validation example is required")
    project = load_config(path=arguments.config)
    generate_action_data(project=project, examples=examples, validation_examples=validation_examples,
                         vision_checkpoint=arguments.vision_checkpoint, device=arguments.device)


if __name__ == "__main__":
    main()
