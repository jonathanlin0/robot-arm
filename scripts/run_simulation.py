"""Run the shared image-based policy loop: PYTHONPATH=src python scripts/run_simulation.py.

Checkpoint paths default to those in the JSON. Ctrl-C stops the loop and closes
its renderer. No simulator cube truth is supplied to either model.
"""

import argparse
from pathlib import Path

from schemas.config import load_config
from simulation_session import SimulationSession
from training import load_action_checkpoint, load_vision_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.json'))
    parser.add_argument('--vision-checkpoint', type=Path)
    parser.add_argument('--action-checkpoint', type=Path)
    parser.add_argument('--steps', type=int, help='Stop after this many control ticks; otherwise run until Ctrl-C.')
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    config = load_config(args.config)
    device = resolve_device(config.training.action.device)
    vision = load_vision_checkpoint(args.vision_checkpoint or config.training.vision.checkpoint_path, device)
    policy = load_action_checkpoint(args.action_checkpoint or config.training.action.checkpoint_path, device)
    # The checkpoint determines model/history dimensions. Keep runtime action
    # mode configurable without silently rebuilding the trained architecture.
    config.action_policy = policy.config.model_copy(update={
        'deterministic': config.action_policy.deterministic,
        'action_std': config.action_policy.action_std,
    })
    policy.config = config.action_policy
    try:
        with SimulationSession(config=config, vision_model=vision, action_policy=policy, device=device) as session:
            if args.steps is None:
                session.runner.run()
            else:
                for _ in range(args.steps):
                    session.runner.step()
    except KeyboardInterrupt:
        print('Stopped simulation.')


if __name__ == '__main__':
    main()
