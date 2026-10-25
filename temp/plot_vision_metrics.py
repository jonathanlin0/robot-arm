"""Plot the final run in the append-only vision training log."""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robot-arm-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "default.log"
OUTPUT = ROOT / "temp" / "vision_metrics_latest.png"
lines = LOG.read_text().splitlines()
start = max(index for index, line in enumerate(lines) if line.startswith("epoch=0 "))
rows = []
for line in lines[start:]:
    if not line.startswith("epoch="):
        continue
    fields = dict(field.split("=", 1) for field in line.split())
    rows.append({key: int(value) if key == "epoch" else float(value)
                 for key, value in fields.items()})

training = [row for row in rows if math.isfinite(row["train_loss"])]
validation = [row for row in rows if math.isfinite(row["val_loss"])]
position = [row for row in rows if math.isfinite(row.get("val_position_error_m", math.nan))]
latest = training[-1]


def label_rows(points, metric):
    if len(points) <= 4:
        return points
    best = min(points, key=lambda row: row[metric])
    return list({row["epoch"]: row for row in (points[0], best, points[-1])}.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#334155", "text.color": "#0f172a",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.edgecolor": "#cbd5e1", "axes.titleweight": "bold",
})
fig, (loss_ax, position_ax) = plt.subplots(1, 2, figsize=(13.6, 5.9), dpi=170)
fig.subplots_adjust(left=0.075, right=0.965, bottom=0.24, top=0.77, wspace=0.28)
fig.text(0.075, 0.92, "Vision training · current experiment", fontsize=21, weight="bold")
fig.text(0.075, 0.86,
         f"Through epoch {latest['epoch']}  ·  Latest training loss {latest['train_loss']:.6f}",
         fontsize=12, color="#475569")

loss_ax.set_title("Training & validation loss", loc="left", pad=16)
loss_ax.plot([row["epoch"] for row in training], [row["train_loss"] for row in training],
             color="#2563eb", linewidth=2.4, label="Training loss")
loss_ax.scatter([latest["epoch"]], [latest["train_loss"]], color="#2563eb", s=35, zorder=4)
loss_ax.plot([row["epoch"] for row in validation], [row["val_loss"] for row in validation],
             color="#ea580c", linewidth=2.2, marker="o", markersize=7, label="Validation loss")
loss_ax.set_yscale("log")
loss_ax.set_ylabel("Weighted pose loss (logarithmic scale)")
loss_ax.yaxis.set_major_formatter(ScalarFormatter())
loss_ax.legend(loc="upper right", frameon=False, fontsize=10)
best_validation = min(validation, key=lambda row: row["val_loss"])
for row in label_rows(validation, "val_loss"):
    is_final = len(validation) > 4 and row is validation[-1]
    prefix = "Best: " if len(validation) > 4 and row is best_validation else ""
    loss_ax.annotate(f"{prefix}{row['val_loss']:.6f}", (row["epoch"], row["val_loss"]),
                     xytext=(0, 18 if is_final else -25), textcoords="offset points", ha="center",
                     color="#c2410c", fontsize=10)
loss_ax.set_ylim(bottom=min(row["val_loss"] for row in validation) * 0.55)

position_ax.set_title("Validation position error", loc="left", pad=16)
position_ax.plot([row["epoch"] for row in position], [row["val_position_error_m"] for row in position],
                 color="#0f766e", linewidth=2.4, marker="o", markersize=7)
position_ax.set_ylabel("Mean Euclidean distance (meters)")
position_ax.set_ylim(0, max(row["val_position_error_m"] for row in position) * 1.3)
best_position = min(position, key=lambda row: row["val_position_error_m"])
for row in label_rows(position, "val_position_error_m"):
    error = row["val_position_error_m"]
    is_best = len(position) > 4 and row is best_position
    prefix = "Best: " if is_best else ""
    position_ax.annotate(f"{prefix}{error:.6f} m\n({error * 1000:.3f} mm)", (row["epoch"], error),
                         xytext=(0, -42 if is_best else 13), textcoords="offset points", ha="center",
                         color="#0f766e", fontsize=10, linespacing=1.5)

for ax in (loss_ax, position_ax):
    ax.set_xlabel("Epoch", labelpad=9)
    ax.set_xlim(0, max(1, latest["epoch"]) + max(2, latest["epoch"] * 0.08))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6, steps=[1, 2, 5, 10]))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)

fig.text(0.075, 0.05,
         "Validation markers are measured values (every 10 epochs); connecting lines are guides.\n"
         "Position error averages the two cube-center distances per example, then all validation examples.",
         fontsize=10, color="#64748b", linespacing=1.7)
fig.savefig(OUTPUT, facecolor="white")
plt.close(fig)

snapshot = {"source": str(LOG), "run_start_line": start + 1,
            "read_at": datetime.now().astimezone().isoformat(),
            "rows": [{key: value if not isinstance(value, float) or math.isfinite(value) else None
                      for key, value in row.items()} for row in rows]}
OUTPUT.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n")
print(OUTPUT)
print(json.dumps({"latest_epoch": latest["epoch"], "training_loss": latest["train_loss"],
                  "validation": validation}, indent=2))
