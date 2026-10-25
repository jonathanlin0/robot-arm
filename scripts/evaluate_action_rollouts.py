"""Evaluate a saved action checkpoint with parallel, unpaced validation rollouts.

Run from the repository root (no PYTHONPATH setup needed):
    .venv/bin/python scripts/evaluate_action_rollouts.py --checkpoint checkpoints/action/default.pt
    .venv/bin/python scripts/evaluate_action_rollouts.py --checkpoint checkpoints/action/default.pt --workers 4

Each spawned worker loads both models and runs floor(configured episodes / workers)
episodes. Any remainder is skipped. Config changes stay in memory; the vision
checkpoint and simulation settings come from configs/default.json by default.
Worker seeds are base_seed + worker_index, reproducible for a fixed worker count.
"""

from __future__ import annotations

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import multiprocessing as mp
from pathlib import Path
import signal
import sys
from time import perf_counter

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_evaluation import ActionRolloutMetrics, evaluate_action_rollouts
from schemas.config import ProjectConfig, load_config
from training import load_action_checkpoint, load_vision_checkpoint, resolve_device


def initialize_worker(progress_lock) -> None:
    """Let the parent handle Ctrl-C; avoid competing CPU thread pools and output."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    tqdm.set_lock(progress_lock)


def evaluate_worker(job: tuple[int, ProjectConfig, Path]) -> ActionRolloutMetrics:
    worker_index, project, checkpoint = job
    # Each process receives its own config. Never send initialized GPU or GL
    # objects between processes, or let all workers repeat the same scenes.
    settings = project.training.action.rollout_validation
    settings.seed += worker_index
    device = resolve_device(project.training.action.device)
    action = load_action_checkpoint(checkpoint, device).requires_grad_(False)
    vision = load_vision_checkpoint(project.training.vision.checkpoint_path, device).requires_grad_(False)
    project.action_policy = action.config

    with tqdm(total=settings.episodes, desc=f"Worker {worker_index + 1} ({device})",
              unit="episode", position=worker_index, leave=False, dynamic_ncols=True) as progress:
        def update(metrics: ActionRolloutMetrics) -> None:
            progress.set_postfix(success_rate_pct=f"{metrics.success_rate_percent:.2f}",
                                 ik_failures=metrics.ik_failures, timeouts=metrics.timeouts,
                                 refresh=False)
            progress.update(metrics.episodes - progress.n)

        result = evaluate_action_rollouts(project=project, action_policy=action,
                                          vision_model=vision, device=device, on_episode=update)
    if result.episodes != settings.episodes:
        raise RuntimeError(f"Worker {worker_index + 1} completed {result.episodes} episodes; "
                           f"expected {settings.episodes}.")
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--action-checkpoint", type=Path, required=True,
                        help="Saved action .pt checkpoint produced by train_action.py.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel processes (default: 2).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.json",
                        help="Simulation, vision checkpoint, device, and total rollout settings.")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    checkpoint = arguments.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Action checkpoint not found: {checkpoint}")
    project = load_config(arguments.config)
    settings = project.training.action.rollout_validation
    intended = settings.episodes
    per_worker = intended // arguments.workers
    if per_worker < 1:
        parser.error(f"--workers cannot exceed the configured rollout count ({intended})")
    if not project.training.vision.checkpoint_path.is_file():
        parser.error(f"Vision checkpoint not found: {project.training.vision.checkpoint_path}")
    settings.episodes = per_worker
    total = per_worker * arguments.workers
    tqdm.write(f"Action checkpoint: {checkpoint}")
    tqdm.write(f"Vision checkpoint: {project.training.vision.checkpoint_path}")
    tqdm.write(f"Rollouts: {arguments.workers} workers x {per_worker} episodes = {total} "
               f"({intended} configured; {intended - total} skipped by flooring).")
    tqdm.write(f"Worker seeds: {settings.seed} through {settings.seed + arguments.workers - 1}; "
               "one PyTorch CPU thread per worker.")
    started = perf_counter()
    context = mp.get_context("spawn")
    jobs = [(index, project, checkpoint) for index in range(arguments.workers)]
    # Pool's context terminates remaining workers on errors or Ctrl-C. Models
    # and renderers are constructed only inside the freshly spawned processes.
    with context.Pool(processes=arguments.workers, initializer=initialize_worker,
                      initargs=(context.RLock(),)) as pool:
        results = list(pool.imap_unordered(evaluate_worker, jobs, chunksize=1))
    combined = ActionRolloutMetrics(
        episodes=sum(result.episodes for result in results),
        successes=sum(result.successes for result in results),
        ik_failures=sum(result.ik_failures for result in results),
        timeouts=sum(result.timeouts for result in results),
    )
    tqdm.write(f"Success rate: {combined.success_rate_percent:.2f}% "
               f"({combined.successes}/{combined.episodes} rollouts); "
               f"IK failures: {combined.ik_failures}; timeouts: {combined.timeouts}; "
               f"elapsed: {perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write("Validation interrupted; workers stopped.")
        raise SystemExit(130)
