#!/usr/bin/env python3
"""Diagnose a saved stacking policy on CPU, without training or rendering.

    python scripts/diagnose_stacking.py
    python scripts/diagnose_stacking.py --episodes 80
    python scripts/diagnose_stacking.py --fresh --episodes 200 --seed 1000000
    python scripts/diagnose_stacking.py --seeds 123 456 --checkpoint PATH.zip
    python scripts/diagnose_stacking.py --test

By default, use the first 200 test-manifest seeds in pretraining's validation
order. To reproduce an earlier 80-scene evaluation, pass --episodes 80 and use
the same dataset and checkpoint. These are live deterministic policy rollouts,
not playback of the expert tensors. Fresh mode uses consecutive reset seeds.

Each invocation creates a new directory under data/diagnostics containing:
  metadata.json   checkpoint fingerprint, effective environment and scene seeds
  episodes.jsonl  per-episode outcomes, physics events, and diagnostic summaries
  summary.json   aggregate counts (errors remain in the denominator)
  report.txt     readable summary and failed seeds
  traces/*.jsonl.gz  initial state and every action/state, including successes

The only simulator hook calls the existing stack update first, then observes
it; it is restored after each episode. No control or success rules are changed.
Contact-loss labels describe observations, not proven causes or policy intent.
All implementation and self-tests live in this file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from pretrain_pickup import (  # noqa: E402
    PretrainingConfig, make_validation_environment, read_episode_records,
)
from success import (  # noqa: E402
    BLUE_CUBE_GEOM, ORANGE_CUBE_GEOM, StackSuccessConfig,
    _geoms_are_in_contact, _orange_touches_gripper,
)


# Ordinary runs can be configured here or through the command line.
DEFAULT_EPISODES = 200
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints/pretraining/default.zip"
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_DATA_DIRECTORY / "diagnostics"
DEFAULT_FRESH_SEED = 1_000_000
DEFAULT_CPU_THREADS = 1
# A diagnostic milestone only; this does not change the environment's success.
LIFT_HEIGHT_METRES = 0.04


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, default=json_default, allow_nan=False, indent=2) + "\n")


def write_json_line(stream, value) -> None:
    stream.write(json.dumps(value, default=json_default, allow_nan=False) + "\n")
    stream.flush()


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_error_record(value):
    """Keep failed trials loggable even if physics produced NaN/Inf values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: clean_error_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_error_record(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def select_scenes(arguments, settings: dict) -> list[dict]:
    """Reuse validation ordering; never silently replace unavailable scenes."""
    if arguments.seeds is not None:
        return [{"seed": seed, "uuid": None} for seed in arguments.seeds]
    if arguments.fresh:
        return [{"seed": arguments.seed + i, "uuid": None} for i in range(arguments.episodes)]
    records = read_episode_records(arguments.data_dir)
    records = [record for record in records if record["split"] == "test"][:arguments.episodes]
    if not records:
        raise ValueError("No validation scenes found; use --fresh for new scene seeds.")
    # Start randomization in the expert data is deliberately ignored, exactly
    # as in pretraining validation. Its reset always uses the fixed home pose.
    physical_keys = ("scene", "action_config", "spawn_config", "success_config",
                     "waypoint_height", "action_interval")
    for record in records:
        if any(record["settings"][key] != settings[key] for key in physical_keys):
            raise ValueError("Dataset and checkpoint environment settings differ; "
                             "use the original dataset or --fresh with checkpoint settings.")
    if len(records) < arguments.episodes:
        print(f"Only {len(records)} validation scenes are available; using all of them.", flush=True)
    return [{"seed": record["seed"], "uuid": record["uuid"]} for record in records]


def create_environment(model: PPO, maximum_steps: int | None = None):
    """Use the same constructor and fixed start as pretraining validation."""
    settings = getattr(model, "pretraining_environment_settings", None)
    config = getattr(model, "pretraining_config", None)
    if (not isinstance(settings, dict) or settings.get("task") != "stack_orange_on_blue"
            or not isinstance(config, dict)):
        raise ValueError("Expected a full-stacking pretraining checkpoint with saved "
                         "pretraining_config and pretraining_environment_settings.")
    history_length = config["history_length"]
    evaluation = PretrainingConfig(
        device="cpu", history_length=history_length,
        sequence_stride=min(256, history_length),
        maximum_episode_steps=maximum_steps or config["maximum_episode_steps"],
    )
    environment = make_validation_environment(evaluation, settings)
    if (environment.observation_space != model.observation_space
            or environment.action_space != model.action_space):
        environment.close()
        raise ValueError("Saved model spaces do not match the current environment.")
    return environment, settings


def deterministic_action(policy, history: dict) -> np.ndarray:
    """Match pretraining evaluation, including trimming unused right padding."""
    length = int(history["valid"].sum())
    inputs = {key: torch.as_tensor(value[:length], device="cpu").unsqueeze(0)
              for key, value in history.items()}
    with torch.no_grad():
        action = policy._predict(inputs, deterministic=True)[0].cpu().numpy()
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("Policy produced an invalid action.")
    return action


def stack_gates(state: dict, config: StackSuccessConfig) -> dict[str, bool]:
    """Expose the exact individual predicates used by success.py."""
    delta = np.asarray(state["orange_position"]) - state["blue_position"]
    epsilon = config.floating_point_numerical_tolerance
    return {
        "x_alignment": bool(abs(delta[0]) <= config.max_horizontal_center_offset + epsilon),
        "y_alignment": bool(abs(delta[1]) <= config.max_horizontal_center_offset + epsilon),
        "orange_above_blue": bool(delta[2] > 0),
        "stack_height": bool(abs(delta[2] - config.expected_vertical_center_distance)
                             <= config.vertical_center_tolerance + epsilon),
        "orange_linear_speed": bool(np.linalg.norm(state["orange_velocity"][:3]) <= config.max_linear_speed),
        "orange_angular_speed": bool(np.linalg.norm(state["orange_velocity"][3:]) <= config.max_angular_speed),
        "blue_linear_speed": bool(np.linalg.norm(state["blue_velocity"][:3]) <= config.max_linear_speed),
        "blue_angular_speed": bool(np.linalg.norm(state["blue_velocity"][3:]) <= config.max_angular_speed),
        "cube_contact": bool(state["orange_blue_contact"]),
        "gripper_clear": not bool(state["orange_gripper_contact"]),
    }


def measure(simulation) -> dict:
    """Copied physical state, readable errors, contacts, and stack predicates."""
    state = simulation.get_state()
    orange_geom = simulation.model.geom(ORANGE_CUBE_GEOM).id
    state["orange_blue_contact"] = _geoms_are_in_contact(
        simulation.data, orange_geom, simulation.model.geom(BLUE_CUBE_GEOM).id,
    )
    state["orange_gripper_contact"] = _orange_touches_gripper(
        simulation.model, simulation.data, orange_geom,
    )
    delta = state["orange_position"] - state["blue_position"]
    state["orange_minus_blue_xyz_m"] = delta
    state["vertical_stack_error_m"] = float(
        delta[2] - simulation.success_config.expected_vertical_center_distance
    )
    state["stack_stable_time_s"] = float(simulation.stack_stable_time)
    state["is_success"] = bool(simulation.is_success())
    state["gates"] = stack_gates(state, simulation.success_config)
    state["all_stack_gates_met"] = all(state["gates"].values())
    return state


def aligned(state: dict) -> bool:
    return state["gates"]["x_alignment"] and state["gates"]["y_alignment"]


def contact_loss_kind(state: dict, open_target: float) -> str:
    """An open target describes a release command, not whether it was sensible."""
    if not np.isclose(state["gripper_target"], open_target, rtol=0, atol=1e-6):
        return "loss_without_open_target"
    if aligned(state) and state["gates"]["stack_height"] and state["orange_blue_contact"]:
        return "open_target_loss_at_stack"
    return "open_target_loss_away_from_stack"


class PhysicsObserver:
    """Monitor AFTER existing bookkeeping; never advance or modify physics."""

    def __init__(self, base):
        self.simulation = base.simulation
        self.open_target = base.action_adapter.config.open_gripper_target
        self.original_update = self.simulation._update_stack_success
        self.had_instance_override = "_update_stack_success" in vars(self.simulation)
        self.start_time = float(self.simulation.data.time)
        self.initial_height = float(self.simulation.data.body("orange_cube").xpos[2])
        self.previous = measure(self.simulation)
        self.latest = self.previous
        self.step = 0
        self.action = None
        self.first_grasp_time_s = None
        self.grasp_count = 0
        self.longest_hold_s = 0.0
        self.maximum_held_lift_m = 0.0
        self.maximum_stack_stable_time_s = 0.0
        self.events = []
        self.pending_loss = None
        self.near_placement_gate_failure_ticks = Counter()
        self.milestones = {key: False for key in (
            "ever_grasped", "lifted_4cm_while_held", "aligned_above_blue_while_held",
            "released_at_stack", "all_stack_conditions_seen", "success",
        )}
        self.closest_stack_state = None
        self.closest_score = None
        self.simulation._update_stack_success = self.update

    def update(self):
        self.original_update()
        self.observe(measure(self.simulation))

    def event(self, kind: str, state: dict) -> dict:
        event = {"kind": kind, "step": self.step,
                 "time_s": state["time"] - self.start_time,
                 "action": self.action, "state": state}
        self.events.append(event)
        return event

    def observe(self, state: dict) -> None:
        held = state["orange_currently_held"]
        previous_held = self.previous["orange_currently_held"]
        now = state["time"] - self.start_time
        opening_switch = (previous_held
                          and not np.isclose(self.previous["gripper_target"], self.open_target)
                          and np.isclose(state["gripper_target"], self.open_target))
        if opening_switch:
            self.event("opening_target_switch_while_held", state)
        if held:
            self.milestones["ever_grasped"] = True
            self.longest_hold_s = max(self.longest_hold_s, state["orange_grasp_hold_time"])
            self.maximum_held_lift_m = max(
                self.maximum_held_lift_m, float(state["orange_position"][2]) - self.initial_height,
            )
            if self.maximum_held_lift_m >= LIFT_HEIGHT_METRES:
                self.milestones["lifted_4cm_while_held"] = True
            if aligned(state) and state["vertical_stack_error_m"] >= 0:
                self.milestones["aligned_above_blue_while_held"] = True
            if not previous_held:
                self.grasp_count += 1
                if self.first_grasp_time_s is None:
                    self.first_grasp_time_s = now
                self.event("grasp_acquired", state)
                if self.pending_loss is not None:
                    self.pending_loss.update(reacquired=True, gap_s=now - self.pending_loss["time_s"])
                    self.pending_loss = None
        elif previous_held:
            event = self.event(contact_loss_kind(state, self.open_target), state)
            event.update(reacquired=False, gap_s=None, preceding_state=self.previous)
            self.pending_loss = event

        # Losing one jaw's contact ends a bilateral hold but does NOT prove a
        # release: the other jaw may still pin orange. Require complete gripper
        # clearance and a supported stack pose for this separate milestone.
        released_stack_pose = (state["confirmed_grasp_seen"] and aligned(state)
                               and state["gates"]["stack_height"]
                               and state["orange_blue_contact"]
                               and not state["orange_gripper_contact"])
        if released_stack_pose and not self.milestones["released_at_stack"]:
            self.event("released_stack_pose_seen", state)
            self.milestones["released_at_stack"] = True

        self.maximum_stack_stable_time_s = max(
            self.maximum_stack_stable_time_s, state["stack_stable_time_s"],
        )
        if state["confirmed_grasp_seen"]:
            self.milestones["all_stack_conditions_seen"] |= state["all_stack_gates_met"]
            # Inspect gate failures specifically near placement, rather than
            # letting thousands of normal transport ticks dominate the count.
            if aligned(state) and state["gates"]["stack_height"]:
                self.near_placement_gate_failure_ticks.update(
                    key for key, passed in state["gates"].items() if not passed
                )
            error = np.linalg.norm(state["orange_minus_blue_xyz_m"][:2]) + abs(state["vertical_stack_error_m"])
            score = (sum(state["gates"].values()), -float(error))
            if self.closest_score is None or score > self.closest_score:
                self.closest_score, self.closest_stack_state = score, state
        self.milestones["success"] |= state["is_success"]
        self.previous = self.latest = state

    def close(self):
        if self.had_instance_override:
            self.simulation._update_stack_success = self.original_update
        else:
            del self.simulation._update_stack_success
        if self.pending_loss is not None:
            self.pending_loss["gap_s"] = (float(self.simulation.data.time) - self.start_time
                                          - self.pending_loss["time_s"])

    def result(self) -> dict:
        return {
            "milestones": self.milestones, "grasp_count": self.grasp_count,
            "first_grasp_time_s": self.first_grasp_time_s,
            "longest_continuous_hold_s": self.longest_hold_s,
            "maximum_held_lift_m": self.maximum_held_lift_m,
            "maximum_stack_stable_time_s": self.maximum_stack_stable_time_s,
            "near_placement_gate_failure_ticks": dict(self.near_placement_gate_failure_ticks),
            "closest_stack_state": self.closest_stack_state,
            "events": self.events, "final_state": self.latest,
        }


def termination_reason(info: dict, terminated: bool, truncated: bool) -> str:
    if info.get("orange_fell_off_table") and info.get("blue_fell_off_table"):
        return "both_cubes_off_table"
    if info.get("orange_fell_off_table"):
        return "orange_off_table"
    if info.get("blue_fell_off_table"):
        return "blue_off_table"
    if terminated and not truncated and info.get("is_success") and not info.get("is_failure"):
        return "success"
    return "timeout" if truncated else "unexpected_termination"


def diagnostic_stage(row: dict) -> str:
    """Describe measured progress, without presenting it as a root cause."""
    if row["outcome"] in ("success", "reset_error", "evaluation_error"):
        return row["outcome"]
    milestones = row["milestones"]
    if not milestones["ever_grasped"]:
        return "no_grasp"
    if milestones["all_stack_conditions_seen"]:
        return "stack_conditions_seen_but_not_held_long_enough"
    if milestones["released_at_stack"]:
        return "released_at_stack_without_stable_success"
    if milestones["aligned_above_blue_while_held"]:
        return "aligned_transport_seen_without_completed_placement"
    if milestones["lifted_4cm_while_held"]:
        return "lift_seen_without_aligned_transport"
    return "grasp_seen_without_4cm_lift"


def evaluate_episode(policy, environment, scene: dict, trace_path: Path) -> dict:
    row = {**scene, "steps": 0, "reward": 0.0, "success": False,
           "outcome": "reset_error", "opening_requests_while_held": 0,
           "opening_requests_accepted_while_held": 0, "ik_failure_steps": 0,
           "tool_axis_unconverged_steps": 0, "maximum_target_tracking_error_m": 0.0}
    observer = None
    info = {}
    with gzip.open(trace_path, "wt", encoding="utf-8") as trace:
        try:
            history, info = environment.reset(seed=scene["seed"])
            base = environment.unwrapped
            observer = PhysicsObserver(base)
            row["initial_state"] = observer.latest
            write_json_line(trace, {"step": 0, "state": observer.latest, "info": info})
            row["outcome"] = "evaluation_error"
            for step in range(1, base.maximum_episode_steps + 1):
                action = deterministic_action(policy, history)
                held_before = observer.latest["orange_currently_held"]
                opening = bool(action[3] >= base.action_adapter.config.open_gripper_command_threshold)
                observer.step, observer.action = step, action.copy()
                previous_target = base.action_adapter.current_target_gripper_position
                history, reward, terminated, truncated, info = environment.step(action)
                row["steps"] = step
                row["reward"] += float(reward)
                if not math.isfinite(row["reward"]):
                    raise ValueError("Environment produced a nonfinite episode reward.")
                row["opening_requests_while_held"] += int(held_before and opening)
                row["opening_requests_accepted_while_held"] += int(
                    held_before and opening and info["ik_position_converged"]
                )
                row["ik_failure_steps"] += int(not info["ik_position_converged"])
                row["tool_axis_unconverged_steps"] += int(not info["ik_tool_axis_converged"])
                tracking_error = float(np.linalg.norm(
                    info["target_gripper_position"] - observer.latest["gripper_position"],
                ))
                row["maximum_target_tracking_error_m"] = max(row["maximum_target_tracking_error_m"], tracking_error)
                write_json_line(trace, {
                    "step": step, "action": action, "applied_action": np.clip(action, -1, 1),
                    "reward": float(reward),
                    "previous_accepted_target": previous_target,
                    "target_tracking_error_m": tracking_error,
                    "state": observer.latest, "info": info,
                    "terminated": terminated, "truncated": truncated,
                })
                if terminated or truncated:
                    row["outcome"] = termination_reason(info, terminated, truncated)
                    # Match evaluate_success, not just the environment's latch.
                    row["success"] = bool(
                        row["outcome"] == "success" and not info["orange_currently_held"]
                        and info["stack_stable_time"] >= base.simulation.success_config.required_stable_time
                    )
                    if row["outcome"] == "success" and not row["success"]:
                        row["outcome"] = "success_metric_mismatch"
                    break
            else:
                row["outcome"] = "step_limit_without_environment_termination"
        except Exception as error:
            row["outcome"] = "reset_error" if observer is None else "evaluation_error"
            row["success"] = False
            row["error"] = f"{type(error).__name__}: {error}"
            write_json_line(trace, {"error": row["error"], "step": row["steps"]})
        finally:
            if observer is not None:
                observer.close()
                row.update(observer.result())
    row["final_info"] = info
    row["diagnostic_stage"] = diagnostic_stage(row)
    if "error" in row:
        row["error_record_nonfinite_encoding"] = "null"
        row = clean_error_record(row)
    return row


def summarize(rows: list[dict]) -> dict:
    successful = sum(row["success"] for row in rows)
    failed = [row for row in rows if not row["success"]]
    return {
        "episodes": len(rows), "successes": successful,
        "success_rate": successful / len(rows) if rows else None,
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "milestone_episode_counts": dict(Counter(
            key for row in rows for key, seen in row.get("milestones", {}).items() if seen
        )),
        "failed_diagnostic_stages": dict(Counter(row["diagnostic_stage"] for row in failed)),
        "failed_final_gate_counts": dict(Counter(
            key for row in failed for key, passed in row.get("final_state", {}).get("gates", {}).items()
            if not passed
        )),
        "contact_loss_episode_counts": dict(Counter(
            kind for row in rows for kind in {event["kind"] for event in row.get("events", [])
                                              if "loss" in event["kind"]}
        )),
        "failed_episodes": [{key: row.get(key) for key in
                             ("index", "seed", "uuid", "outcome", "diagnostic_stage", "trace", "error")}
                            for row in failed],
    }


def format_report(summary: dict) -> str:
    count, successes = summary["episodes"], summary["successes"]
    lines = [f"Deterministic CPU stacking: {successes}/{count} successful episodes"
             + (f" ({successes / count:.1%})" if count else ""),
             "Errors count as unsuccessful episodes; no scenes are replaced.", "",
             "Milestones (ever seen, not necessarily retained):"]
    lines.extend(f"  {key}: {value}/{count}" for key, value in summary["milestone_episode_counts"].items())
    for title, key in (
        ("Environment outcomes", "outcomes"),
        ("Failed episodes by observed progress", "failed_diagnostic_stages"),
        ("Unsatisfied gates at the end of failed episodes", "failed_final_gate_counts"),
        ("Episodes with each contact-loss event (overlapping categories)", "contact_loss_episode_counts"),
    ):
        lines.extend(["", title + ":"])
        lines.extend(f"  {name}: {value}" for name, value in summary[key].items())
    lines.extend(["", "Contact loss is not proof of a drop. Opening at a stack is expected.",
                  "A momentary loss may be a contact gap; inspect reacquired/gap_s and event states.",
                  "Progress categories and final gates are diagnostics, not proven root causes.",
                  "", "Failed scene seeds (use --seeds to rerun):"])
    for row in summary["failed_episodes"]:
        lines.append(f"  {row['seed']}: {row['outcome']}; {row['diagnostic_stage']}; {row['trace']}")
    return "\n".join(lines) + "\n"


def run_diagnostics(arguments) -> Path:
    torch.set_num_threads(arguments.cpu_threads)
    checkpoint = arguments.checkpoint.expanduser().resolve()
    checkpoint_hash = fingerprint(checkpoint)
    model = PPO.load(checkpoint, device="cpu")
    model.policy.set_training_mode(False)
    environment, settings = create_environment(model, arguments.max_steps)
    try:
        scenes = select_scenes(arguments, settings)
        base = environment.unwrapped
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = arguments.output_dir.expanduser().resolve() / f"stacking-{stamp}-{uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "traces").mkdir()
        write_json(output / "metadata.json", {
            "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_hash,
            "device": "cpu", "deterministic": True, "cpu_threads": arguments.cpu_threads,
            "torch_version": torch.__version__, "mujoco_version": mujoco.__version__,
            "scene_mode": "explicit" if arguments.seeds is not None else "fresh" if arguments.fresh else "validation",
            "data_directory": arguments.data_dir, "scenes": scenes,
            "maximum_episode_steps": base.maximum_episode_steps,
            "history_length": environment.history_length,
            "scene_path": base.simulation.scene_path,
            "scene_sha256": fingerprint(base.simulation.scene_path),
            "start_position": base.simulation.start_position,
            "start_position_half_range": base.simulation.start_position_half_range,
            "action_config": asdict(base.action_adapter.config),
            "spawn_config": asdict(base.simulation.spawn_config),
            "success_config": asdict(base.simulation.success_config),
            "reward_config": asdict(base.reward_config),
            "physics_timestep_s": float(base.simulation.model.opt.timestep),
            "action_interval_s": settings["action_interval"],
            "diagnostic_lift_height_m": LIFT_HEIGHT_METRES,
            "saved_validation_episodes": model.pretraining_config.get("validation_episodes"),
        })
        print(f"Evaluating {len(scenes)} episodes on CPU. Logs: {output}", flush=True)
        rows = []
        started = time.perf_counter()
        try:
            with (output / "episodes.jsonl").open("w") as stream:
                for index, scene in enumerate(scenes, 1):
                    trace = Path("traces") / f"{index:04d}-{scene['uuid'] or scene['seed']}.jsonl.gz"
                    row = evaluate_episode(model.policy, environment, scene, output / trace)
                    row.update(index=index, trace=str(trace))
                    write_json_line(stream, row)
                    rows.append(row)
                    print(f"{index}/{len(scenes)} | {row['outcome']} | {row['diagnostic_stage']} | "
                          f"hold={row.get('longest_continuous_hold_s', 0):.2f}s | "
                          f"stable={row.get('maximum_stack_stable_time_s', 0):.2f}s | "
                          f"steps={row['steps']} | seed={scene['seed']}", flush=True)
        finally:
            # Keep completed results useful even if the user interrupts a run.
            summary = summarize(rows)
            summary.update(planned_episodes=len(scenes), complete=len(rows) == len(scenes),
                           elapsed_seconds=time.perf_counter() - started)
            write_json(output / "summary.json", summary)
            report = format_report(summary)
            (output / "report.txt").write_text(report)
        print("\n" + report, end="", flush=True)
        return output
    finally:
        environment.close()


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive.")
    return number


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative.")
    return number


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=positive_integer, default=DEFAULT_EPISODES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh", action="store_true", help="Use consecutive seeds instead of saved validation scenes.")
    modes.add_argument("--seeds", nargs="+", type=nonnegative_integer,
                       help="Rerun exactly these seeds; overrides --episodes.")
    parser.add_argument("--seed", type=nonnegative_integer, default=DEFAULT_FRESH_SEED,
                        help="First seed in --fresh mode.")
    parser.add_argument("--cpu-threads", type=positive_integer, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--max-steps", type=positive_integer,
                        help="Optional diagnostic horizon override; default is the checkpoint's horizon.")
    parser.add_argument("--test", action="store_true", help="Run self-tests without loading weights or saving diagnostics.")
    arguments = parser.parse_args(argv)
    if arguments.seeds is not None and len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error("--seeds must be distinct so repeated scenes do not inflate counts.")
    return arguments


def run_tests() -> None:
    """Focused regression tests, kept here to avoid modifying any other file."""
    import copy
    import tempfile
    import unittest
    from types import SimpleNamespace
    from unittest.mock import patch
    from gym_environment import CubeStackGymEnvironment
    from environment import DEFAULT_SCENE_PATH

    class DiagnosticsTests(unittest.TestCase):
        def state(self):
            return dict(orange_position=np.array([0.30, 0.0, 0.06]),
                        blue_position=np.array([0.30, 0.0, 0.02]),
                        orange_velocity=np.zeros(6), blue_velocity=np.zeros(6),
                        orange_blue_contact=True, orange_gripper_contact=False,
                        gripper_target=1.0)

        def test_alignment_is_per_axis_not_radius(self):
            state = self.state()
            state["orange_position"][:2] += 0.009
            self.assertTrue(all(stack_gates(state, StackSuccessConfig()).values()))
            state["orange_position"][0] += 0.002
            self.assertFalse(stack_gates(state, StackSuccessConfig())["x_alignment"])

        def test_contact_clearance_and_both_cube_speeds(self):
            for field, index, gate, value in (
                ("orange_velocity", 0, "orange_linear_speed", 0.011),
                ("orange_velocity", 3, "orange_angular_speed", 0.051),
                ("blue_velocity", 0, "blue_linear_speed", 0.011),
                ("blue_velocity", 3, "blue_angular_speed", 0.051),
            ):
                state = self.state()
                state[field][index] = value
                self.assertFalse(stack_gates(state, StackSuccessConfig())[gate])
            state = self.state()
            state.update(orange_gripper_contact=True, orange_blue_contact=False)
            gates = stack_gates(state, StackSuccessConfig())
            self.assertFalse(gates["gripper_clear"])
            self.assertFalse(gates["cube_contact"])

        def test_release_is_distinct_from_closed_target_loss(self):
            state = self.state()
            state["gates"] = stack_gates(state, StackSuccessConfig())
            self.assertEqual(contact_loss_kind(state, 1), "open_target_loss_at_stack")
            state["gripper_target"] = -0.1
            self.assertEqual(contact_loss_kind(state, 1), "loss_without_open_target")
            state.update(gripper_target=1, orange_blue_contact=False)
            self.assertEqual(contact_loss_kind(state, 1), "open_target_loss_away_from_stack")

        def test_defaults_and_explicit_seeds(self):
            arguments = parse_arguments([])
            self.assertEqual(arguments.episodes, 200)
            self.assertEqual(arguments.output_dir, DEFAULT_OUTPUT_DIRECTORY)
            arguments = parse_arguments(["--seeds", "123", str(2 ** 127)])
            self.assertEqual([r["seed"] for r in select_scenes(arguments, {})], [123, 2 ** 127])
            arguments = parse_arguments(["--fresh", "--episodes", "2", "--seed", "5"])
            self.assertEqual([r["seed"] for r in select_scenes(arguments, {})], [5, 6])

        def test_validation_order_and_no_train_scenes(self):
            settings = {name: {} for name in ("scene", "action_config", "spawn_config", "success_config",
                                            "waypoint_height", "action_interval")}
            records = [{"split": split, "seed": i, "uuid": str(i), "settings": settings}
                       for i, split in enumerate(("train", "test", "train", "test"))]
            with patch(f"{__name__}.read_episode_records", return_value=records):
                scenes = select_scenes(parse_arguments(["--episodes", "2"]), settings)
            self.assertEqual([scene["seed"] for scene in scenes], [1, 3])

        def test_termination_and_error_denominator(self):
            self.assertEqual(termination_reason({"is_success": True}, True, False), "success")
            self.assertEqual(termination_reason({}, False, True), "timeout")
            self.assertEqual(termination_reason({"orange_fell_off_table": True}, True, False), "orange_off_table")
            rows = [dict(success=True, outcome="success", diagnostic_stage="success"),
                    dict(success=False, outcome="reset_error", diagnostic_stage="reset_error")]
            self.assertEqual(summarize(rows)["success_rate"], 0.5)
            self.assertIn("0/0", format_report(summarize([])))

        def test_predict_trims_and_is_deterministic(self):
            calls = []
            def predict(inputs, deterministic):
                calls.append((inputs["tokens"].shape, deterministic, torch.is_grad_enabled()))
                return torch.zeros((1, 4))
            history = {"tokens": np.zeros((10, 56), np.float32),
                       "valid": np.array([1, 1] + [0] * 8, np.float32),
                       "episode_start": np.zeros(10, np.float32)}
            action = deterministic_action(SimpleNamespace(_predict=predict), history)
            self.assertEqual(calls, [(torch.Size([1, 2, 56]), True, False)])
            self.assertEqual(action.shape, (4,))

        def test_observer_is_passive_and_gates_match_environment(self):
            environment = CubeStackGymEnvironment(scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH)
            try:
                environment.reset(seed=7)
                observer = PhysicsObserver(environment)
                for _ in range(5):
                    environment.step(np.zeros(4))
                    self.assertEqual(observer.latest["all_stack_gates_met"],
                                     environment.simulation.stack_conditions_met())
                observed_qpos = environment.simulation.data.qpos.copy()
                observer.close()
                self.assertNotIn("_update_stack_success", vars(environment.simulation))
                environment.reset(seed=7)
                for _ in range(5):
                    environment.step(np.zeros(4))
                np.testing.assert_array_equal(environment.simulation.data.qpos, observed_qpos)
            finally:
                environment.close()

        def test_transient_grasp_release_and_reacquisition(self):
            environment = CubeStackGymEnvironment(scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH)
            try:
                environment.reset(seed=7)
                observer = PhysicsObserver(environment)
                state = copy.deepcopy(observer.latest)
                state.update(time=0.1, orange_currently_held=True, confirmed_grasp_seen=True,
                             orange_grasp_hold_time=0.005, gripper_target=-0.1)
                observer.observe(state)
                lost = copy.deepcopy(state)
                lost.update(time=0.105, orange_currently_held=False, orange_grasp_hold_time=0.0)
                observer.observe(lost)
                regained = copy.deepcopy(state)
                regained["time"] = 0.11
                observer.observe(regained)
                self.assertEqual(observer.grasp_count, 2)
                self.assertAlmostEqual(observer.longest_hold_s, 0.005)
                loss = next(event for event in observer.events if event["kind"] == "loss_without_open_target")
                self.assertTrue(loss["reacquired"])
                self.assertAlmostEqual(loss["gap_s"], 0.005)
                observer.close()
            finally:
                environment.close()

        def test_release_requires_complete_gripper_clearance(self):
            environment = CubeStackGymEnvironment(scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH)
            try:
                environment.reset(seed=7)
                observer = PhysicsObserver(environment)
                state = copy.deepcopy(observer.latest)
                state.update(self.state())
                state.update(confirmed_grasp_seen=True, orange_currently_held=False,
                             orange_gripper_contact=True)
                state["gates"] = stack_gates(state, StackSuccessConfig())
                state["all_stack_gates_met"] = False
                observer.observe(state)
                self.assertFalse(observer.milestones["released_at_stack"])
                state = copy.deepcopy(state)
                state["orange_gripper_contact"] = False
                state["gates"] = stack_gates(state, StackSuccessConfig())
                state["all_stack_gates_met"] = True
                observer.observe(state)
                self.assertTrue(observer.milestones["released_at_stack"])
                observer.close()
            finally:
                environment.close()

        def test_nonfinite_episode_errors_still_serialize(self):
            from action_observation_history import ActionObservationHistoryWrapper
            environment = ActionObservationHistoryWrapper(CubeStackGymEnvironment(
                scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH, maximum_episode_steps=2,
            ))
            policy = SimpleNamespace(_predict=lambda inputs, deterministic: torch.zeros((1, 4)))
            original_step = environment.step
            def broken_step(action):
                history, reward, terminated, truncated, info = original_step(action)
                if bad_info:
                    info["injected_nonfinite"] = np.array([float("nan"), float("inf")])
                return history, float("nan") if bad_reward else reward, terminated, truncated, info
            try:
                with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
                    for bad_reward, bad_info in ((True, False), (False, True), (True, True)):
                        with self.subTest(bad_reward=bad_reward, bad_info=bad_info):
                            with patch.object(environment, "step", side_effect=broken_step):
                                row = evaluate_episode(policy, environment, {"seed": 7, "uuid": None},
                                                       Path(directory) / "error.jsonl.gz")
                            encoded = json.dumps(row, allow_nan=False, default=json_default)
                            self.assertEqual(row["outcome"], "evaluation_error")
                            if bad_reward:
                                self.assertIsNone(row["reward"])
                            if bad_info:
                                self.assertEqual(json.loads(encoded)["final_info"]["injected_nonfinite"], [None, None])
                            self.assertEqual(summarize([row])["success_rate"], 0)
                            self.assertNotIn("_update_stack_success", vars(environment.unwrapped.simulation))
            finally:
                environment.close()

        def test_episode_trace_timeout_and_hook_restoration(self):
            from action_observation_history import ActionObservationHistoryWrapper
            environment = ActionObservationHistoryWrapper(CubeStackGymEnvironment(
                scene_path=REPOSITORY_ROOT / DEFAULT_SCENE_PATH, maximum_episode_steps=2,
            ))
            policy = SimpleNamespace(_predict=lambda inputs, deterministic: torch.zeros((1, 4)))
            try:
                # Temporary test artifacts stay in this repository and are removed.
                with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
                    path = Path(directory) / "trace.jsonl.gz"
                    row = evaluate_episode(policy, environment, {"seed": 7, "uuid": None}, path)
                    with gzip.open(path, "rt") as stream:
                        trace = [json.loads(line) for line in stream]
                    self.assertEqual(row["outcome"], "timeout")
                    self.assertEqual(len(trace), 3)
                    self.assertIn("target_gripper_position", trace[-1]["info"])
                    self.assertNotIn("_update_stack_success", vars(environment.unwrapped.simulation))
            finally:
                environment.close()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticsTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.test:
        run_tests()
    else:
        run_diagnostics(arguments)
