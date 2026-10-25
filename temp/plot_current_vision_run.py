"""Save a consistent, immutable progress chart from one metrics-log snapshot."""

from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robot-arm-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "logs/vision_training.latest.json").read_text())
source = Path(metadata["metrics_log"])
snapshot = source.read_bytes()
snapshot_at = datetime.now(ZoneInfo("America/Chicago")).isoformat()
rows = []
for line in snapshot.decode().splitlines():
    fields = dict(token.split("=", 1) for token in line.split())
    if "epoch" in fields:
        rows.append({key: int(value) if key == "epoch" else float(value)
                     for key, value in fields.items()})
training = [row for row in rows if math.isfinite(row["train_loss"])]
validation = [row for row in rows if math.isfinite(row["val_loss"])]
position = [row for row in validation if math.isfinite(row["val_position_error_m"])]
latest = training[-1]
best_validation = min(validation, key=lambda row: row["val_loss"])
best_position = min(position, key=lambda row: row["val_position_error_m"])
reductions = [row for row in rows if row["next_learning_rate"] < row["learning_rate"]]
config = metadata["config"]["training"]["vision"]
run_name = source.name.removesuffix(".metrics.log")
output = ROOT / "reports" / f"{run_name}-epoch-{latest['epoch']}"
output.mkdir(parents=True, exist_ok=False)
(output / "metrics_snapshot.log").write_bytes(snapshot)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#475569", "text.color": "#172033",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none",
})
fig, (loss_ax, position_ax) = plt.subplots(
    2, 1, figsize=(10, 8.7), dpi=170, sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)
fig.subplots_adjust(left=0.13, right=0.94, bottom=0.15, top=0.82, hspace=0.32)
title = "Vision model · completed training run" if latest["epoch"] >= config["epochs"] else "Vision model · current training run"
fig.text(0.13, 0.945, title, fontsize=22, weight="bold")
fig.text(0.13, 0.895,
         f"{latest['epoch']} / {config['epochs']} epochs completed   ·   Batch size {config['batch_size']}   ·   {metadata['device'].upper()}",
         fontsize=12, color="#475569")
started = datetime.fromisoformat(metadata["started_at"]).astimezone(ZoneInfo("America/Chicago"))
fig.text(0.13, 0.855,
         f"Run started {started.strftime('%B %d, %Y at %I:%M %p')} (Chicago)",
         fontsize=10.5, color="#64748b")

loss_ax.plot([row["epoch"] for row in training], [row["train_loss"] for row in training],
             color="#2563eb", linewidth=2.2, label="Training loss")
loss_ax.scatter([latest["epoch"]], [latest["train_loss"]], color="#2563eb", s=24, zorder=4)
loss_ax.plot([row["epoch"] for row in validation], [row["val_loss"] for row in validation],
             color="#ea580c", linewidth=2.1, marker="o", markersize=6, label="Validation loss")
loss_ax.set_yscale("log")
loss_ax.set_ylim(min(row["train_loss"] for row in training) * 0.55, 1)
loss_ax.set_ylabel("Weighted pose loss (log scale)", labelpad=11)
loss_ax.yaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]))
loss_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
loss_ax.legend(loc="upper right", frameon=False, fontsize=11)

for row in {row["epoch"]: row for row in (best_validation, validation[-1])}.values():
    label = ("Best " if row is best_validation else "") + f"{row['val_loss']:.6f}"
    offset = (-18, 35) if row is best_validation and row is not validation[-1] else (0, 13)
    loss_ax.annotate(label, (row["epoch"], row["val_loss"]), xytext=offset,
                     textcoords="offset points", ha="center", color="#c2410c", fontsize=10)
loss_ax.annotate(f"{latest['train_loss']:.6f}", (latest["epoch"], latest["train_loss"]),
                 xytext=(0, -14), textcoords="offset points", ha="center",
                 color="#2563eb", fontsize=10)

for index, row in enumerate(reductions):
    transition = row["epoch"] + 0.5
    loss_ax.axvline(transition, color="#94a3b8", linestyle="--", linewidth=1.0, zorder=0)
    loss_ax.text(transition - 1.2, 0.18 if index % 2 == 0 else 0.09,
                 f"LR reduced after epoch {row['epoch']}\n"
                 f"{row['learning_rate']:.6g} → {row['next_learning_rate']:.6g}",
                 ha="right", va="center", color="#64748b", fontsize=9.5)

position_ax.plot([row["epoch"] for row in position],
                 [row["val_position_error_m"] * 1000 for row in position],
                 color="#0f766e", linewidth=2.2, marker="o", markersize=6)
for row in {row["epoch"]: row for row in (position[0], best_position, position[-1])}.values():
    value = row["val_position_error_m"] * 1000
    label = f"{value:.3f}" + ("\nBest" if row is best_position else "")
    position_ax.annotate(label, (row["epoch"], value), xytext=(0, 10),
                         textcoords="offset points", ha="center", color="#0f766e",
                         fontsize=10, linespacing=1.3)
position_ax.set_ylim(0, max(row["val_position_error_m"] * 1000 for row in position) * 1.3)
position_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
position_ax.set_ylabel("Validation position\nerror (mm)", labelpad=12)
position_ax.set_xlabel("Epoch", labelpad=8)
for ax in (loss_ax, position_ax):
    ax.set_xlim(0, latest["epoch"] + 5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9, steps=[1, 2, 5, 10]))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)

fig.text(0.13, 0.075,
         f"Validation markers show measurements every {config['validation_interval']} epochs; connecting lines are guides.",
         fontsize=9.5, color="#64748b")
fig.text(0.13, 0.045,
         "Position error is the mean 3D distance from each predicted cube center to its true center.",
         fontsize=9.5, color="#64748b")
for extension in ("png", "svg"):
    fig.savefig(output / f"vision_training.{extension}", facecolor="white")
plt.close(fig)

summary = {
    "source": str(source), "snapshot_at": snapshot_at,
    "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
    "latest_completed_epoch": latest["epoch"], "training_loss": latest["train_loss"],
    "best_training": min(training, key=lambda row: row["train_loss"]),
    "latest_validation": validation[-1], "best_validation": best_validation,
    "best_position": best_position, "validation": validation,
    "learning_rate_reductions": reductions,
    "current_learning_rate": latest["next_learning_rate"],
    "batch_size": config["batch_size"], "device": metadata["device"],
    "configured_epochs": config["epochs"],
    "png": str(output / "vision_training.png"), "svg": str(output / "vision_training.svg"),
}
summary = json.loads(json.dumps(summary), parse_constant=lambda _: None)
(output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2))
