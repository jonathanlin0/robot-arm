#!/usr/bin/env python3
"""Evaluate saved pickup policies on CPU, without training or rendering.

    python scripts/diagnose_pickup_policies.py
    python scripts/diagnose_pickup_policies.py --episodes 200 --workers 4
    python scripts/diagnose_pickup_policies.py --run-ids n4oe8rme --mode stochastic

Defaults: the five requested runs, 100 episodes EACH in deterministic and
stochastic modes, four CPU workers with one Torch thread each. All policies
see the same scene seeds. Stochastic evaluation resamples gSDE at the saved
frequency (plus the start of each episode); it is not a replay of training.
Recovery-start sampling is disabled for comparable evaluation scenes.

Writes episodes.jsonl, summary.json, and report.txt in a fresh directory under
data/policy_diagnostics/. Wilson 95% intervals describe episode sampling
uncertainty for each saved policy, not variation between training seeds.
All diagnostic logic lives here; the training environment is unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_observation_history import ActionObservationHistoryWrapper  # noqa: E402
from environment import DEFAULT_SCENE_PATH, MINIMUM_HOLD_TIME  # noqa: E402
from gym_environment import CubeStackGymEnvironment  # noqa: E402
from rewards import StackRewardConfig  # noqa: E402
from success import orange_gripper_pad_contacts, orange_touches_table  # noqa: E402

DEFAULT_RUN_IDS = ("n4oe8rme", "9agwf0c8", "hybow3fj", "vnjvy9ub", "2u89gdex")
_cached_checkpoint = None
_cached_policy = None


def initialize_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def load_policy(checkpoint):
    global _cached_checkpoint, _cached_policy
    if _cached_checkpoint != checkpoint:
        _cached_policy = PPO.load(checkpoint, device="cpu")
        _cached_policy.policy.set_training_mode(False)
        _cached_checkpoint = checkpoint
    return _cached_policy


def create_environment(policy):
    config = getattr(policy, "pickup_training_config", None)
    required = {"history_length", "maximum_episode_steps", "reward_config",
                "start_at_orange_waypoint"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Checkpoint lacks environment metadata; refusing to guess its start mode.")
    base = CubeStackGymEnvironment(
        scene_path=REPO_ROOT / DEFAULT_SCENE_PATH,
        maximum_episode_steps=config["maximum_episode_steps"],
        reward_config=StackRewardConfig(**config["reward_config"]),
        start_at_orange_waypoint=config["start_at_orange_waypoint"],
        # Evaluate the same clean starts even if training mixed in recovery.
        recovery_start_probability=0.0,
    )
    env = ActionObservationHistoryWrapper(base, history_length=config["history_length"])
    if env.observation_space != policy.observation_space or env.action_space != policy.action_space:
        env.close()
        raise ValueError("Checkpoint spaces do not match the current observation/action history.")
    return env


class HoldObserver:
    """Observe existing physics bookkeeping, without changing its decisions."""

    def __init__(self, simulation, open_target):
        self.simulation = simulation
        self.open_target = open_target
        self.original_update = simulation._update_pickup_progress
        self.start_time = float(simulation.data.time)
        self.previously_held = False
        self.grasp_count = 0
        self.first_grasp_time_s = None
        self.max_hold_s = 0.0
        self.held_ticks = 0
        self.physics_ticks = 0
        self.losses = []
        self.pending_loss = None
        self.last_open_switch_time = -math.inf
        self.previous_target = float(simulation.data.actuator("gripper").ctrl[0])
        self.open_switches_while_held = 0
        simulation._update_pickup_progress = self.update

    def update(self):
        self.original_update()
        self.observe()

    def observe(self):
        sim = self.simulation
        now = float(sim.data.time)
        # The original update clears this marker on EVERY invalid physics tick.
        hold_start = sim._orange_lifted_at_time
        held = hold_start is not None
        target = float(sim.data.actuator("gripper").ctrl[0])
        opening_switch = (
            self.previously_held
            and self.previous_target < self.open_target - 1e-6
            and math.isclose(target, self.open_target, abs_tol=1e-6)
        )
        if opening_switch:
            self.open_switches_while_held += 1
            self.last_open_switch_time = now
        self.previous_target = target
        self.physics_ticks += 1
        if held:
            self.held_ticks += 1
            self.max_hold_s = max(self.max_hold_s, max(0.0, now - hold_start))
            if not self.previously_held:
                self.grasp_count += 1
                if self.first_grasp_time_s is None:
                    self.first_grasp_time_s = max(0.0, hold_start - self.start_time)
                if self.pending_loss is not None:
                    self.pending_loss["gap_s"] = now - self.pending_loss["absolute_time"]
                    self.pending_loss["reacquired"] = True
                    self.pending_loss = None
        elif self.previously_held:
            fixed, moving = orange_gripper_pad_contacts(sim.model, sim.data)
            event = {
                "time_s": now - self.start_time,
                "absolute_time": now,
                "missing_fixed_pad": not bool(fixed),
                "missing_moving_pad": not bool(moving),
                "table_contact": bool(orange_touches_table(sim.model, sim.data)),
                "environment_failure": bool(sim.is_failure()),
                "open_target_active": math.isclose(target, self.open_target, abs_tol=1e-6),
                "within_0_2s_of_open_switch": now - self.last_open_switch_time <= 0.2 + 1e-9,
                "orange_height_m": float(sim.data.body("orange_cube").xpos[2]),
                "reacquired": False,
                "gap_s": None,
            }
            self.losses.append(event)
            self.pending_loss = event
        self.previously_held = held

    def close(self):
        self.simulation._update_pickup_progress = self.original_update
        if self.pending_loss is not None:
            self.pending_loss["gap_s"] = (
                float(self.simulation.data.time) - self.pending_loss["absolute_time"]
            )
        for event in self.losses:
            event.pop("absolute_time", None)


def evaluate_episode(policy, env, run_id, mode, seed):
    # Reproducible per-scene exploration, independent of process scheduling.
    action_seed = seed + 1_000_000
    torch.manual_seed(action_seed)
    observation, _ = env.reset(seed=seed)
    base = env.unwrapped
    sim = base.simulation
    cfg = base.action_adapter.config
    initial_state = base.previous_state
    initial_orange = initial_state["orange_position"].copy()
    initial_axis = sim.data.site("gripperframe").xmat.reshape(3, 3)[:, 0]
    axis_target = np.asarray(cfg.target_tool_axis, dtype=float)
    axis_target /= np.linalg.norm(axis_target)
    row = {
        "run_id": run_id, "mode": mode, "seed": seed, "action_seed": action_seed,
        "initial_orange_xyz": initial_orange.tolist(),
        "initial_blue_xyz": initial_state["blue_position"].tolist(),
        "initial_tool_axis_error_deg": float(np.degrees(np.arccos(np.clip(
            initial_axis @ axis_target, -1, 1
        )))),
        "steps": 0, "reward": 0.0, "held_control_steps": 0,
        "opening_requests_while_held": 0, "opening_accepted_while_held": 0,
        "noise_opening_requests_while_held": 0,
        "ik_failure_steps": 0, "tool_axis_unconverged_steps": 0,
        "gripper_std_sum": 0.0, "held_gripper_std_sum": 0.0,
        "held_gripper_mean_sum": 0.0,
        "min_gripper_cube_distance_m": float("inf"), "max_lift_m": 0.0,
        "reward_components": {},
    }
    components = Counter()
    # Install only AFTER reset/waypoint preparation, so prep never counts.
    observer = HoldObserver(sim, cfg.open_gripper_target)
    try:
        for step in range(base.maximum_episode_steps):
            if mode == "stochastic" and policy.use_sde:
                if step == 0 or (policy.sde_sample_freq > 0 and step % policy.sde_sample_freq == 0):
                    policy.policy.reset_noise(n_envs=1)
            held_before_action = observer.previously_held
            action, _ = policy.predict(observation, deterministic=(mode == "deterministic"))
            distribution = policy.policy.action_dist.distribution
            gripper_std = float(distribution.stddev[0, 3])
            gripper_mean = float(distribution.mean[0, 3])
            row["gripper_std_sum"] += gripper_std
            opening_request = bool(action[3] >= cfg.open_gripper_command_threshold)
            if held_before_action:
                row["held_control_steps"] += 1
                row["held_gripper_std_sum"] += gripper_std
                row["held_gripper_mean_sum"] += gripper_mean
                row["opening_requests_while_held"] += int(opening_request)
                row["noise_opening_requests_while_held"] += int(
                    mode == "stochastic" and opening_request
                    and gripper_mean < cfg.open_gripper_command_threshold
                )
            observation, reward, terminated, truncated, info = env.step(action)
            row["steps"] += 1
            row["reward"] += float(reward)
            components.update(info["reward_components"])
            accepted = bool(info["ik_position_converged"])
            row["ik_failure_steps"] += int(not accepted)
            row["tool_axis_unconverged_steps"] += int(not info["ik_tool_axis_converged"])
            row["opening_accepted_while_held"] += int(held_before_action and opening_request and accepted)
            state = base.previous_state
            row["min_gripper_cube_distance_m"] = min(
                row["min_gripper_cube_distance_m"],
                float(np.linalg.norm(state["gripper_position"] - state["orange_position"])),
            )
            row["max_lift_m"] = max(row["max_lift_m"], float(state["orange_position"][2] - initial_orange[2]))
            if terminated or truncated:
                break
    finally:
        observer.close()
    success = bool(info["is_success"])
    row.update(
        success=success,
        termination="success" if success else "failure" if info["is_failure"] else "timeout",
        ever_grasped=observer.grasp_count > 0,
        grasp_count=observer.grasp_count,
        first_grasp_survived_to_success=success and observer.grasp_count > 0 and not observer.losses,
        first_grasp_time_s=observer.first_grasp_time_s,
        max_hold_s=observer.max_hold_s,
        hold_requirement_seen=observer.max_hold_s >= MINIMUM_HOLD_TIME,
        hold_losses=observer.losses,
        open_switches_while_held=observer.open_switches_while_held,
        held_physics_ticks=observer.held_ticks,
        physics_ticks=observer.physics_ticks,
        final_hold_s=float(info["orange_grasp_hold_time"]),
        reward_components=dict(components),
    )
    return row


def evaluate_batch(task):
    checkpoint, run_id, mode, seeds = task
    policy = load_policy(checkpoint)
    env = create_environment(policy)
    rows = []
    try:
        for seed in seeds:
            try:
                rows.append(evaluate_episode(policy, env, run_id, mode, seed))
            except Exception as error:
                # Preserve failures explicitly; never replace a difficult scene.
                rows.append({"run_id": run_id, "mode": mode, "seed": seed,
                             "error": f"{type(error).__name__}: {error}"})
    finally:
        env.close()
    return rows


def proportion(count, total):
    if not total:
        return {"count": count, "total": total, "rate": None, "ci95": None}
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": count, "total": total, "rate": p,
            "ci95": [max(0, center - margin), min(1, center + margin)]}


def quantiles(values):
    return dict(zip(("p10", "median", "p90", "max"), map(float, np.quantile(
        values, [0.1, 0.5, 0.9, 1.0]
    )))) if values else None


def summarize(rows):
    good = [row for row in rows if "error" not in row]
    grasped = [row for row in good if row["ever_grasped"]]
    losses = [event for row in good for event in row["hold_losses"]]
    total_steps = sum(row["steps"] for row in good)
    held_steps = sum(row["held_control_steps"] for row in good)
    summary = {
        "episodes": len(good), "errors": [row for row in rows if "error" in row],
        "success": proportion(sum(row["success"] for row in good), len(good)),
        "ever_grasped": proportion(len(grasped), len(good)),
        "success_given_grasp": proportion(sum(row["success"] for row in grasped), len(grasped)),
        "first_grasp_survival": proportion(sum(row["first_grasp_survived_to_success"] for row in grasped), len(grasped)),
        "opening_given_grasp": proportion(sum(row["opening_accepted_while_held"] > 0 for row in grasped), len(grasped)),
        "noise_opening_given_grasp": proportion(sum(row["noise_opening_requests_while_held"] > 0 for row in grasped), len(grasped)),
        "hold_loss_given_grasp": proportion(sum(bool(row["hold_losses"]) for row in grasped), len(grasped)),
        "longest_hold_all_s": quantiles([row["max_hold_s"] for row in good]),
        "longest_hold_grasped_s": quantiles([row["max_hold_s"] for row in grasped]),
        "reward_mean": float(np.mean([row["reward"] for row in good])) if good else None,
        "hold_loss_count": len(losses),
        "loss_reasons": {key: sum(event[key] for event in losses) for key in (
            "missing_fixed_pad", "missing_moving_pad", "table_contact", "environment_failure",
            "open_target_active", "within_0_2s_of_open_switch",
        )},
        "brief_reacquired_gaps_le_20ms": sum(event["reacquired"] and event["gap_s"] <= 0.020000001 for event in losses),
        "opening_requests_while_held": sum(row["opening_requests_while_held"] for row in good),
        "opening_accepted_while_held": sum(row["opening_accepted_while_held"] for row in good),
        "open_switches_while_held": sum(row["open_switches_while_held"] for row in good),
        "ik_failure_step_fraction": sum(row["ik_failure_steps"] for row in good) / total_steps if total_steps else None,
        "tool_axis_unconverged_step_fraction": sum(row["tool_axis_unconverged_steps"] for row in good) / total_steps if total_steps else None,
        "gripper_std_mean": sum(row["gripper_std_sum"] for row in good) / total_steps if total_steps else None,
        "held_gripper_std_mean": sum(row["held_gripper_std_sum"] for row in good) / held_steps if held_steps else None,
        "held_gripper_mean": sum(row["held_gripper_mean_sum"] for row in good) / held_steps if held_steps else None,
        "hold_requirement_seen_without_success": sum(row["hold_requirement_seen"] and not row["success"] for row in good),
        "initial_tool_axis_error_deg": quantiles([row["initial_tool_axis_error_deg"] for row in good]),
    }
    return summary


def format_rate(value):
    if value["rate"] is None:
        return "n/a (0 episodes)"
    low, high = value["ci95"]
    return f"{value['count']}/{value['total']} = {value['rate']:.1%} [95% CI {low:.1%}–{high:.1%}]"


def format_number(value):
    return "n/a" if value is None else f"{value:.3f}"


def make_report(summaries, rows, run_ids, modes):
    lines = [
        "CPU pickup-policy diagnostics (fresh evaluation episodes)",
        "Success = environment's continuous 2-second, bilateral, off-table grasp.",
        "Intervals are Wilson 95% episode-sampling intervals; small conditional denominators remain uncertain.",
        "Contact-loss reasons overlap. Opening/loss associations do not establish causation.",
    ]
    for run_id in run_ids:
        for mode in modes:
            s = summaries[f"{run_id}/{mode}"]
            lines += ["", f"{run_id} / {mode}: {s['episodes']} evaluated, {len(s['errors'])} errors",
                      f"  Success: {format_rate(s['success'])}",
                      f"  Ever acquired grasp: {format_rate(s['ever_grasped'])}",
                      f"  Eventually succeeded | acquired grasp: {format_rate(s['success_given_grasp'])}",
                      f"  First grasp survived to success | acquired grasp: {format_rate(s['first_grasp_survival'])}",
                      f"  Accepted opening while held | acquired grasp: {format_rate(s['opening_given_grasp'])}",
                      f"  Noise crossed opening threshold | acquired grasp: {format_rate(s['noise_opening_given_grasp'])}"]
            hold = s["longest_hold_grasped_s"]
            lines += [f"  Longest hold among grasped episodes, seconds (p10/median/p90/max): {hold}",
                      f"  Hold interruptions: {s['hold_loss_count']}; reacquired within 20 ms: {s['brief_reacquired_gaps_le_20ms']}",
                      f"  Interruption reasons/context: {s['loss_reasons']}",
                      f"  Opening requests/accepted/actual target switches while held: {s['opening_requests_while_held']}/{s['opening_accepted_while_held']}/{s['open_switches_while_held']}",
                      f"  IK-failed step fraction: {format_number(s['ik_failure_step_fraction'])}; orientation-unconverged: {format_number(s['tool_axis_unconverged_step_fraction'])}",
                      f"  Gripper distribution SD overall/while held: {format_number(s['gripper_std_mean'])}/{format_number(s['held_gripper_std_mean'])}; mean while held: {format_number(s['held_gripper_mean'])}",
                      f"  Physics hold threshold reached but episode unsuccessful: {s['hold_requirement_seen_without_success']}"]
            if s["episodes"]:
                no_grasp = s["episodes"] - s["ever_grasped"]["count"]
                after_grasp = s["ever_grasped"]["count"] - s["success"]["count"]
                lines.append(f"  Failure breakdown: {no_grasp} never grasped; {after_grasp} grasped but did not succeed.")
            if s["errors"]:
                lines.append(f"  ERROR example: {s['errors'][0]['error']}")
        if len(modes) == 2:
            paired = {mode: {r["seed"]: r for r in rows if r["run_id"] == run_id
                            and r["mode"] == mode and "error" not in r} for mode in modes}
            common = paired["deterministic"].keys() & paired["stochastic"].keys()
            det_only = sum(paired["deterministic"][seed]["success"] and not paired["stochastic"][seed]["success"] for seed in common)
            sto_only = sum(paired["stochastic"][seed]["success"] and not paired["deterministic"][seed]["success"] for seed in common)
            lines.append(f"  Matched scenes={len(common)}; deterministic-only successes={det_only}; stochastic-only successes={sto_only}.")
    lines += ["", "Interpretation:",
              "- Many failures without any grasp: investigate descent/alignment/closing timing.",
              "- Many failures after grasp: investigate retention, contact interruptions, or reopening.",
              "- Deterministic improves while stochastic reopens: exploration is a candidate contributor.",
              "- Brief closed-target contact gaps warrant inspecting grasp geometry/contact behavior.",
              "- Large IK failure rates warrant inspecting targets/workspace before changing model capacity.",
              "These are diagnostics of saved checkpoints under current simulator code, not original training trajectories."]
    return "\n".join(lines) + "\n"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-ids", nargs="+", default=list(DEFAULT_RUN_IDS))
    parser.add_argument("--episodes", type=int, default=100, help="episodes per policy PER MODE, at least 10 (default: 100)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=("both", "deterministic", "stochastic"), default="both")
    parser.add_argument("--seed", type=int, default=20_000, help="first scene seed")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints/wandb")
    parser.add_argument("--output", type=Path, help="new output directory; must not already exist")
    args = parser.parse_args()
    if args.episodes < 10 or args.workers < 1 or args.seed < 0:
        parser.error("episodes must be >=10, workers >=1, and seed >=0")
    if len(set(args.run_ids)) != len(args.run_ids) or any(re.fullmatch(r"[A-Za-z0-9_-]+", r) is None for r in args.run_ids):
        parser.error("run IDs must be distinct path-safe IDs")
    return args


def main():
    args = parse_arguments()
    torch.set_num_threads(1)
    modes = ["deterministic", "stochastic"] if args.mode == "both" else [args.mode]
    checkpoints = {run_id: (args.checkpoint_root / run_id / "ppo_cube_stacker.zip").resolve() for run_id in args.run_ids}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing saved policy: {checkpoint}")
    output = args.output or REPO_ROOT / "data/policy_diagnostics" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "device": "cpu", "workers": args.workers, "torch_threads_per_worker": 1,
        "episodes_per_policy_per_mode": args.episodes, "modes": modes,
        "first_scene_seed": args.seed, "hold_requirement_s": MINIMUM_HOLD_TIME,
        "evaluation_recovery_start_probability": 0.0,
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": {},
        "source_sha256": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__).resolve(), *sorted((REPO_ROOT / "src").glob("*.py")), REPO_ROOT / DEFAULT_SCENE_PATH]},
    }
    # Validate metadata/spaces before committing to a long evaluation.
    for run_id, checkpoint in checkpoints.items():
        policy = load_policy(checkpoint)
        env = create_environment(policy)
        env.close()
        metadata["checkpoints"][run_id] = {
            "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_config": policy.pickup_training_config, "timesteps": policy.num_timesteps,
            "use_sde": policy.use_sde, "sde_sample_freq": policy.sde_sample_freq,
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    seeds = list(range(args.seed, args.seed + args.episodes))
    tasks = [(str(checkpoints[run_id]), run_id, mode, seeds[start:start + 10])
             for run_id in args.run_ids for mode in modes for start in range(0, len(seeds), 10)]
    total = len(args.run_ids) * len(modes) * args.episodes
    print(f"CPU evaluation: {total} episodes; {args.episodes} per policy per mode; {args.workers} workers.", flush=True)
    print(f"Output: {output}", flush=True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("w") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker) as pool:
            futures = [pool.submit(evaluate_batch, task) for task in tasks]
            for future in as_completed(futures):
                batch = future.result()
                for row in batch:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.extend(batch)
                completed = len(rows)
                errors = sum("error" in row for row in rows)
                elapsed = time.monotonic() - started
                print(f"Completed {completed}/{total}, errors={errors}, elapsed={elapsed / 60:.1f} min; last={batch[-1]['run_id']}/{batch[-1]['mode']}", flush=True)
    summaries = {f"{run_id}/{mode}": summarize([r for r in rows if r["run_id"] == run_id and r["mode"] == mode])
                 for run_id in args.run_ids for mode in modes}
    report = make_report(summaries, rows, args.run_ids, modes)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    (output / "report.txt").write_text(report)
    print("\n" + report, flush=True)
    print(f"Saved per-episode data and report to {output}", flush=True)
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
