#!/usr/bin/env python3
"""Save twelve learned-policy validation rollouts as a 3-by-4 MP4 grid.

Run from the repository root (FFmpeg must be on PATH):
    .venv/bin/python scripts/demo.py --output stack_demo.mp4
    .venv/bin/python scripts/demo.py --checkpoint checkpoints/action/custom.pt

Defaults to the action and vision checkpoints saved by training. Each panel
shows the external camera; policy inference still uses both trained camera
inputs. Yellow means not successful; green means a validated, released stack.
Completed panels hold their last frame until the end of the video.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import floor, isfinite
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

import cv2
import mujoco
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_evaluation import ActionRolloutFrame, ActionRolloutMetrics, evaluate_action_rollouts
from camera_feed_getter import SIM_CAMERA_NAMES
from environment import CubeStackEnvironment
from schemas import CAMERA_IMAGE_SHAPES
from schemas.config import ProjectConfig, load_config
from training import load_action_checkpoint, load_vision_checkpoint, resolve_device


GRID_ROWS, GRID_COLUMNS = 3, 4
ROLLOUT_COUNT = GRID_ROWS * GRID_COLUMNS
PANEL_WIDTH, PANEL_HEIGHT = 320, 240
STATUS_BORDER_WIDTH = 4
YELLOW = (245, 165, 35)
GREEN = (40, 210, 70)
_worker_models = None


@dataclass(frozen=True)
class RolloutTask:
    index: int
    project: ProjectConfig
    output_path: Path
    ffmpeg: str


@dataclass(frozen=True)
class RolloutResult:
    index: int
    seed: int
    metrics: ActionRolloutMetrics
    steps: int
    status: str
    output_path: Path


def make_panel(rgb: np.ndarray, *, successful: bool) -> np.ndarray:
    """Fit the square external view inside a legacy-sized panel, without cropping."""
    height, width = rgb.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized = cv2.resize(rgb, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_AREA)
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    top = (PANEL_HEIGHT - resized.shape[0]) // 2
    left = (PANEL_WIDTH - resized.shape[1]) // 2
    panel[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    color = GREEN if successful else YELLOW
    border = STATUS_BORDER_WIDTH
    panel[:border] = panel[-border:] = color
    panel[:, :border] = panel[:, -border:] = color
    return panel


class PanelRecorder:
    """Stream one panel to FFmpeg, retaining only its latest frame in memory."""

    def __init__(self, path: Path, *, ffmpeg: str, fps: float, frame_count: int) -> None:
        self.path = path
        self.ffmpeg = ffmpeg
        self.fps = fps
        self.frame_count = frame_count
        self.renderer = None
        self.process = None
        self.stderr = None
        self.last_frame = None
        self.written = 0
        self.step = 0
        self.status = "running"

    def __enter__(self):
        self.stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size",
                f"{PANEL_WIDTH}x{PANEL_HEIGHT}", "-framerate", str(self.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-threads", "1", "-pix_fmt", "yuv420p", str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr)
        except BaseException:
            self.stderr.close()
            raise
        return self

    def _encoder_error(self) -> RuntimeError:
        self.stderr.seek(0)
        details = self.stderr.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(f"FFmpeg could not encode {self.path.name}: {details}")

    def _write(self, frame: np.ndarray) -> None:
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            self.process.wait()
            raise self._encoder_error() from error
        self.written += 1

    def record(self, environment: CubeStackEnvironment, frame: ActionRolloutFrame) -> None:
        if self.renderer is None:
            height, width, _ = CAMERA_IMAGE_SHAPES["external"]
            self.renderer = mujoco.Renderer(environment.model, height=height, width=width)
        # Delay writing by one tick so an IK failure at the same completed step
        # can replace its running frame without adding an extra video frame.
        while self.written < frame.step and self.last_frame is not None:
            self._write(self.last_frame)
        with environment.state_lock:
            self.renderer.update_scene(environment.data, camera=SIM_CAMERA_NAMES["external"])
            self.last_frame = make_panel(self.renderer.render(), successful=frame.status == "success")
        self.step = frame.step
        self.status = frame.status

    def __exit__(self, error_type, error, traceback) -> None:
        try:
            if error_type is None:
                if self.last_frame is None:
                    raise RuntimeError("Validation produced no frames to encode.")
                while self.written < self.frame_count:
                    self._write(self.last_frame)
                try:
                    self.process.stdin.close()
                except BrokenPipeError as pipe_error:
                    self.process.wait()
                    raise self._encoder_error() from pipe_error
                if self.process.wait() != 0:
                    raise self._encoder_error()
        finally:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait()
                if not self.process.stdin.closed:
                    try:
                        self.process.stdin.close()
                    except BrokenPipeError:
                        pass
            finally:
                self.stderr.close()
                if self.renderer is not None:
                    self.renderer.close()


def initialize_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def render_rollout(task: RolloutTask) -> RolloutResult:
    global _worker_models
    project = task.project.model_copy(deep=True)
    key = (project.training.action.checkpoint_path, project.training.vision.checkpoint_path,
           project.training.action.device)
    if _worker_models is None or _worker_models[0] != key:
        device = resolve_device(project.training.action.device)
        action = load_action_checkpoint(key[0], device).requires_grad_(False)
        vision = load_vision_checkpoint(key[1], device).requires_grad_(False)
        _worker_models = (key, device, action, vision)
    _, device, action, vision = _worker_models
    # Architecture and history dimensions come from the saved checkpoint.
    project.action_policy = action.config
    settings = project.training.action.rollout_validation
    settings.episodes = 1
    settings.seed += task.index
    max_steps = floor(project.environment.episode_seconds * project.loop.control_hz + 1e-9)
    with PanelRecorder(task.output_path, ffmpeg=task.ffmpeg, fps=project.loop.control_hz,
                       frame_count=max_steps + 1) as recorder:
        metrics = evaluate_action_rollouts(project=project, action_policy=action,
                                          vision_model=vision, device=device,
                                          on_frame=recorder.record)
    return RolloutResult(task.index, settings.seed, metrics, recorder.step,
                         recorder.status, task.output_path)


def compose_grid(results: list[RolloutResult], output: Path, *, ffmpeg: str) -> None:
    """Combine equal-length panels in row-major order into a 1280-by-720 MP4."""
    ordered = sorted(results, key=lambda result: result.index)
    if [result.index for result in ordered] != list(range(ROLLOUT_COUNT)):
        raise ValueError("The demo requires exactly twelve distinct rollout panels.")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    for result in ordered:
        command.extend(["-threads", "1", "-i", str(result.output_path)])
    inputs = "".join(f"[{index}:v]" for index in range(ROLLOUT_COUNT))
    command.extend([
        "-filter_complex", f"{inputs}xstack=inputs={ROLLOUT_COUNT}:grid={GRID_COLUMNS}x{GRID_ROWS}:shortest=1[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not assemble the demo: {result.stderr.strip()}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--checkpoint", "--action-checkpoint", "--weights", type=Path,
                        help="Action weights; defaults to training.action.checkpoint_path.")
    parser.add_argument("--vision-checkpoint", type=Path,
                        help="Vision weights; defaults to training.vision.checkpoint_path.")
    parser.add_argument("--output", type=Path, default=Path("stack_demo.mp4"))
    parser.add_argument("--workers", type=int, default=2, help="Spawned workers, 1 to 12 (default: 2).")
    parser.add_argument("--episode-seconds", type=float,
                        help="Override the configured validation duration for each panel (default: 45 s).")
    parser.add_argument("--seed", type=int, help="First panel seed; defaults to the validation seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        help="Override the configured action-training device.")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.workers <= ROLLOUT_COUNT:
        parser.error("--workers must be between 1 and 12")
    if arguments.seed is not None and arguments.seed < 0:
        parser.error("--seed must be nonnegative")
    if arguments.episode_seconds is not None and (
        not isfinite(arguments.episode_seconds) or arguments.episode_seconds <= 0
    ):
        parser.error("--episode-seconds must be finite and positive")
    project = load_config(arguments.config.expanduser())
    for name, override in (("action", arguments.checkpoint), ("vision", arguments.vision_checkpoint)):
        section = getattr(project.training, name)
        if override is not None:
            section.checkpoint_path = override.expanduser().resolve()
        if not section.checkpoint_path.is_file():
            parser.error(f"{name.capitalize()} checkpoint not found: {section.checkpoint_path}")
    if arguments.episode_seconds is not None:
        project.environment.episode_seconds = arguments.episode_seconds
    if arguments.seed is not None:
        project.training.action.rollout_validation.seed = arguments.seed
    if arguments.device is not None:
        project.training.action.device = arguments.device
    if floor(project.environment.episode_seconds * project.loop.control_hz + 1e-9) < 1:
        parser.error("Episode duration must allow at least one control tick")
    arguments.output = arguments.output.expanduser().resolve()
    if arguments.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    arguments.ffmpeg = shutil.which("ffmpeg")
    if arguments.ffmpeg is None:
        parser.error("FFmpeg is required on PATH to encode the MP4 (macOS: brew install ffmpeg)")
    arguments.project = project
    return arguments


def main() -> None:
    arguments = parse_arguments()
    project = arguments.project
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Action checkpoint: {project.training.action.checkpoint_path}")
    tqdm.write(f"Vision checkpoint: {project.training.vision.checkpoint_path}")
    tqdm.write(f"Rendering 12 external-camera panels in a 3 x 4 grid; {arguments.workers} workers, "
               f"{project.environment.episode_seconds:g} s per rollout, {project.loop.control_hz:g} FPS.")
    tqdm.write("Yellow: not successful. Green: successful. Finished panels hold their last frame.")
    started = perf_counter()
    # Stage beside the destination so replacement is atomic, preserving any
    # existing video until all panels and the final encode have succeeded.
    with tempfile.TemporaryDirectory(prefix=".stack-demo-", dir=output.parent) as temporary:
        directory = Path(temporary)
        jobs = [RolloutTask(index, project, directory / f"panel-{index:02d}.mp4", arguments.ffmpeg)
                for index in range(ROLLOUT_COUNT)]
        context = mp.get_context("spawn")
        results = []
        with context.Pool(processes=arguments.workers, initializer=initialize_worker) as pool:
            with tqdm(total=ROLLOUT_COUNT, desc="Validation panels", unit="panel") as progress:
                for result in pool.imap_unordered(render_rollout, jobs, chunksize=1):
                    results.append(result)
                    tqdm.write(f"Panel {result.index + 1:02d}, seed {result.seed}: "
                               f"{result.status}, {result.steps} steps")
                    progress.update(1)
        tqdm.write("Combining the twelve panels into the MP4...")
        staged_output = directory / "grid.mp4"
        compose_grid(results, staged_output, ffmpeg=arguments.ffmpeg)
        staged_output.replace(output)
    successes = sum(result.metrics.successes for result in results)
    ik_failures = sum(result.metrics.ik_failures for result in results)
    timeouts = sum(result.metrics.timeouts for result in results)
    tqdm.write(f"Success rate: {100 * successes / ROLLOUT_COUNT:.2f}% ({successes}/12); "
               f"IK failures: {ik_failures}; timeouts: {timeouts}.")
    tqdm.write(f"Saved {output} (1280 x 720; elapsed {perf_counter() - started:.1f}s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write("Demo interrupted; workers stopped and temporary videos removed.")
        raise SystemExit(130)
