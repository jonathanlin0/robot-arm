"""Pair normal policy evaluations with longer time limits; no training or source changes."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_stacking as diagnosis


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # The baseline writer may still be completing its last line.
    return rows


def prefix_matches(original, extended, original_steps):
    with gzip.open(original, "rt") as first, gzip.open(extended, "rt") as second:
        for count, (a, b) in enumerate(zip(first, second)):
            before, after = json.loads(a), json.loads(b)
            if before["step"] != after["step"] or before["state"] != after["state"]:
                return False, count
            if before.get("action") != after.get("action"):
                return False, count
            if before["step"] == original_steps:
                return True, count
    return False, -1


def tail_summary(trace):
    with gzip.open(trace, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if "state" in row]
    import numpy as np
    result = {}
    for name, section in (("last_100", rows[-100:]), ("last_384", rows[-384:])):
        states = [r["state"] for r in section]
        result[name] = {
            "orange_xyz_range_m": np.ptp([s["orange_position"] for s in states], axis=0),
            "gripper_xyz_range_m": np.ptp([s["gripper_position"] for s in states], axis=0),
            "actions_mean": np.mean([r["action"] for r in section if "action" in r], axis=0),
            "actions_min": np.min([r["action"] for r in section if "action" in r], axis=0),
            "actions_max": np.max([r["action"] for r in section if "action" in r], axis=0),
            "held_fraction": np.mean([s["orange_currently_held"] for s in states]),
        }
    return result


def report(baseline, output, baseline_rows, extended_rows):
    by_seed = {r["seed"]: r for r in extended_rows}
    total = len(baseline_rows)
    counts = {}
    for horizon in (400, 600, 1200, 2400):
        counts[str(horizon)] = sum(
            (r["success"] and r["steps"] <= horizon) or
            (r["seed"] in by_seed and by_seed[r["seed"]]["success"]
             and by_seed[r["seed"]]["steps"] <= horizon)
            for r in baseline_rows)
    result = {"episodes": total, "baseline": str(baseline),
              "successes_by_step_limit": counts,
              "baseline_outcomes": dict(Counter(r["outcome"] for r in baseline_rows)),
              "extended_outcomes": dict(Counter(r["outcome"] for r in extended_rows)),
              "paired_prefixes_identical": all(r["original_prefix_identical"] for r in extended_rows),
              "rescued": [{"index": r["index"], "seed": r["seed"], "steps": r["steps"]}
                          for r in extended_rows if r["success"]],
              "remaining": [{k: r.get(k) for k in ("index", "seed", "steps", "outcome", "diagnostic_stage",
                             "final_state", "final_unsatisfied_gates", "milestones", "error", "tail")}
                            for r in extended_rows if not r["success"]]}
    diagnosis.write_json(output / "paired_summary.json", result)
    lines = ["Paired deterministic CPU policy horizon diagnosis", "",
             f"Baseline episodes: {total}", f"Baseline outcomes: {result['baseline_outcomes']}",
             f"Extended timeout outcomes: {result['extended_outcomes']}",
             f"Identical action/state prefixes through original limit: {result['paired_prefixes_identical']}", "",
             "Successes by time allowance (same scenes and frozen policy):"]
    for horizon, count in counts.items():
        lines.append(f"  {horizon} steps ({int(horizon) * 0.05:g} simulated seconds): {count}/{total} ({count/total:.1%})")
    lines += ["", "Rescued original timeouts:"]
    for r in result["rescued"]:
        lines.append(f"  scene {r['index']}: seed={r['seed']} success at step {r['steps']} ({r['steps']*.05:.2f}s)")
    lines += ["", "Remaining extended failures:"]
    for r in result["remaining"]:
        lines.append(f"  scene {r['index']}: seed={r['seed']} {r['outcome']} at step {r['steps']}; {r['diagnostic_stage']}")
    lines += ["", "Only the episode time limit changed. No retraining, injected disturbances, altered success gates,",
              "or changes to IK/physics/failure detection. Longer runs retain the last384 history tokens.",
              "Counts at400steps quantify the current demo cutoff; that cutoff is20s, while the checkpoint horizon is600steps/30s.",
              "The60s/120s results reuse baseline successes and rerun only original timeouts; errors/physical failures remain failures.",
              "CPU results need not be identical to MPS validation because numerical differences can change contact trajectories."]
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--max-steps", type=int, default=2400)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    meta = json.loads((baseline / "metadata.json").read_text())
    output = baseline / "extended-horizon"
    output.mkdir(exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = Path(meta["checkpoint"])
    assert diagnosis.fingerprint(checkpoint) == meta["checkpoint_sha256"]
    model = PPO.load(checkpoint, device="cpu")
    model.policy.set_training_mode(False)
    env, _ = diagnosis.create_environment(model, args.max_steps)
    diagnosis.write_json(output / "metadata.json", {**meta, "maximum_episode_steps": args.max_steps,
                                                   "baseline_directory": baseline,
                                                   "scene_mode": "paired_baseline_timeouts"})
    rows = read_rows(output / "episodes.jsonl")
    seen = {r["seed"] for r in rows}
    try:
        with (output / "episodes.jsonl").open("a") as stream:
            while True:
                baseline_rows = read_rows(baseline / "episodes.jsonl")
                for original in baseline_rows:
                    if original["outcome"] != "timeout" or original["seed"] in seen:
                        continue
                    scene = {"seed": original["seed"], "uuid": original["uuid"]}
                    trace = Path("traces") / f"{original['index']:04d}-{original['seed']}.jsonl.gz"
                    print(f"Extending baseline scene {original['index']} to {args.max_steps} steps...", flush=True)
                    row = diagnosis.evaluate_episode(model.policy, env, scene, output / trace)
                    row.update(index=original["index"], trace=str(trace))
                    matched, prefix_steps = prefix_matches(baseline / original["trace"], output / trace, original["steps"])
                    row.update(original_prefix_identical=matched, prefix_compared_through_step=prefix_steps,
                               original_steps=original["steps"], tail=tail_summary(output / trace))
                    diagnosis.write_json_line(stream, row)
                    rows.append(row)
                    seen.add(original["seed"])
                    print(f"Extended scene {row['index']}: {row['outcome']} at {row['steps']} steps; "
                          f"prefix identical={matched}; {row['diagnostic_stage']}", flush=True)
                summary_path = baseline / "summary.json"
                if summary_path.exists() and json.loads(summary_path.read_text()).get("complete"):
                    if len(baseline_rows) == len(read_rows(baseline / "episodes.jsonl")):
                        break
                time.sleep(1)
        assert diagnosis.fingerprint(checkpoint) == meta["checkpoint_sha256"]
        report(baseline, output, baseline_rows, rows)
    finally:
        env.close()


if __name__ == "__main__":
    main()
