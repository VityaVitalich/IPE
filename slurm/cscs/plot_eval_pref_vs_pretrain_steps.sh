#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/plot-pref-vs-steps-%j.out
#SBATCH --error=logs/plot-pref-vs-steps-%j.err
#SBATCH --no-requeue

# Plot preference percentage vs pretrain checkpoint step for each eval level.
#
# Usage:
#   sbatch slurm/cscs/plot_eval_pref_vs_pretrain_steps.sh
#
# Output:
#   <PLOT_OUTPUT_DIR>/<PLOT_BASENAME>.png
#   <PLOT_OUTPUT_DIR>/<PLOT_BASENAME>.csv

set -eo pipefail

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

# Must match the steps used in SFT/eval submission.
PRETRAIN_STEPS=(
  1000
  2000
  3000
  4000
  5000
  6000
  7000
  8000
  9000
  10000
  15000
  20000
  25000
  30000
  35000
  40000
  45000
)

# Eval run labels are expected as:
#   ${EVAL_LABEL_PREFIX}_pt${STEP}
# Example run label: IM111-sepemb-ood_pt10000
EVAL_LABEL_PREFIX="IM111-self-ood"

# Root containing eval outputs.
EVAL_OUTPUT_ROOT="outputs/eval"

# Metric for "preference percentage":
#   generation_decided -> preference / (preference + opposite)
#   generation_all     -> preference / (preference + opposite + unknown)
#   probabilistic      -> probabilistic.rates.preference
PREFERENCE_METRIC="generation_decided"

# Overlay M011 baseline as dashed horizontal lines (one per level).
ADD_M011_BASELINE=true
M011_BASELINE_RUN_LABEL="M001_ood"
M011_BASELINE_LEGEND="M001 baseline"

PLOT_OUTPUT_DIR="outputs/eval/plots"
PLOT_BASENAME="pref_vs_pretrain_steps_IM111-self-ood"
PLOT_TITLE="Preference vs Pretrain Step"

# -------------------------------------------------------------------

if [ "${#PRETRAIN_STEPS[@]}" -eq 0 ]; then
  echo "Error: PRETRAIN_STEPS is empty."
  exit 1
fi

for step in "${PRETRAIN_STEPS[@]}"; do
  if ! [[ "$step" =~ ^[0-9]+$ ]]; then
    echo "Error: PRETRAIN_STEPS contains non-integer value: $step"
    exit 1
  fi
done

case "$PREFERENCE_METRIC" in
  generation_decided|generation_all|probabilistic)
    ;;
  *)
    echo "Error: PREFERENCE_METRIC must be one of: generation_decided, generation_all, probabilistic"
    exit 1
    ;;
esac

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "visualize_eval_summary.py" ]; then
  : # Already in project root
elif [ -f "../visualize_eval_summary.py" ]; then
  cd ..
elif [ -f "../../visualize_eval_summary.py" ]; then
  cd ../..
fi

mkdir -p logs "$PLOT_OUTPUT_DIR"

PRETRAIN_STEPS_CSV="$(IFS=,; echo "${PRETRAIN_STEPS[*]}")"
export PRETRAIN_STEPS_CSV
export EVAL_LABEL_PREFIX
export EVAL_OUTPUT_ROOT
export PREFERENCE_METRIC
export ADD_M011_BASELINE
export M011_BASELINE_RUN_LABEL
export M011_BASELINE_LEGEND
export PLOT_OUTPUT_DIR
export PLOT_BASENAME
export PLOT_TITLE

echo "START TIME: $(date)"
echo "Plotting preference vs pretrain steps"
echo "  steps:          ${PRETRAIN_STEPS[*]}"
echo "  eval label pref $EVAL_LABEL_PREFIX"
echo "  metric:         $PREFERENCE_METRIC"
echo "  M011 baseline:  $ADD_M011_BASELINE ($M011_BASELINE_RUN_LABEL)"
echo "  eval root:      $EVAL_OUTPUT_ROOT"
echo "  output dir:     $PLOT_OUTPUT_DIR"

python3 - <<'PY'
import csv
import json
import os
import re
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(f"matplotlib is required for plotting: {exc}")


def sanitize_run_id(run_label: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_label).strip("_")
    return run_id or "run"


def load_summary_for_run(eval_root: str, run_label: str) -> Dict:
    run_id = sanitize_run_id(run_label)
    candidates = [
        os.path.join(eval_root, "merged", f"eval_{run_id}", "summary.json"),
        os.path.join(eval_root, f"eval_{run_id}_merged", "summary.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"Summary not found for run_label={run_label}. "
        f"Tried: {candidates}"
    )


def level_sort_key(level_name: str):
    match = re.match(r"[Ll](\d+)$", level_name)
    if match:
        return (0, int(match.group(1)))
    return (1, level_name)


def preference_pct(level_data: Dict, metric: str) -> Optional[float]:
    if metric.startswith("generation"):
        counts = level_data.get("generation", {}).get("response_counts", {})
        pref = float(counts.get("preference", 0))
        opp = float(counts.get("opposite", 0))
        unk = float(counts.get("unknown", 0))
        if metric == "generation_decided":
            denom = pref + opp
        else:
            denom = pref + opp + unk
        if denom <= 0:
            return None
        return 100.0 * pref / denom

    rates = level_data.get("probabilistic", {}).get("rates", {})
    if "preference" in rates:
        return 100.0 * float(rates["preference"])
    return None


steps = [int(s) for s in os.environ["PRETRAIN_STEPS_CSV"].split(",") if s]
label_prefix = os.environ["EVAL_LABEL_PREFIX"]
eval_root = os.environ["EVAL_OUTPUT_ROOT"]
metric = os.environ["PREFERENCE_METRIC"]
add_m011_baseline = os.environ.get("ADD_M011_BASELINE", "false").lower() in {"1", "true", "yes", "on"}
m011_baseline_run_label = os.environ.get("M011_BASELINE_RUN_LABEL", "M011_ood")
m011_baseline_legend = os.environ.get("M011_BASELINE_LEGEND", "M011 baseline")
plot_output_dir = os.environ["PLOT_OUTPUT_DIR"]
plot_basename = os.environ["PLOT_BASENAME"]
plot_title = os.environ["PLOT_TITLE"]

summaries_by_step: Dict[int, Dict] = {}
missing = []
for step in steps:
    run_label = f"{label_prefix}_pt{step}"
    try:
        summaries_by_step[step] = load_summary_for_run(eval_root, run_label)
    except FileNotFoundError as exc:
        missing.append(str(exc))

if missing:
    message = "\n".join(missing)
    raise SystemExit(f"Missing summary files:\n{message}")

all_levels = set()
for summary in summaries_by_step.values():
    all_levels.update(summary.get("levels", {}).keys())
levels = sorted(all_levels, key=level_sort_key)

if not levels:
    raise SystemExit("No levels found in loaded summaries.")

values_by_level: Dict[str, List[Optional[float]]] = {level: [] for level in levels}
for step in steps:
    level_map = summaries_by_step[step].get("levels", {})
    for level in levels:
        value = preference_pct(level_map.get(level, {}), metric)
        values_by_level[level].append(value)

baseline_by_level: Dict[str, Optional[float]] = {level: None for level in levels}
if add_m011_baseline:
    baseline_summary = load_summary_for_run(eval_root, m011_baseline_run_label)
    baseline_levels = baseline_summary.get("levels", {})
    for level in levels:
        baseline_by_level[level] = preference_pct(baseline_levels.get(level, {}), metric)

os.makedirs(plot_output_dir, exist_ok=True)
csv_path = os.path.join(plot_output_dir, f"{plot_basename}.csv")
png_path = os.path.join(plot_output_dir, f"{plot_basename}.png")

# Save table used for plotting
with open(csv_path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step"] + levels)
    for idx, step in enumerate(steps):
        row = [step]
        for level in levels:
            value = values_by_level[level][idx]
            row.append("" if value is None else f"{value:.6f}")
        writer.writerow(row)

# Plot one line per level
plt.figure(figsize=(10, 6))
x_min = min(steps)
x_max = max(steps)
baseline_legend_added = False
for level_idx, level in enumerate(levels):
    color = f"C{level_idx}"
    xs = []
    ys = []
    for idx, step in enumerate(steps):
        value = values_by_level[level][idx]
        if value is None:
            continue
        xs.append(step)
        ys.append(value)
    if xs:
        plt.plot(xs, ys, marker="o", linewidth=2, color=color, label=level)
    baseline_value = baseline_by_level.get(level)
    if baseline_value is not None:
        baseline_label = m011_baseline_legend if not baseline_legend_added else "_nolegend_"
        plt.hlines(
            y=baseline_value,
            xmin=x_min,
            xmax=x_max,
            colors=color,
            linestyles="--",
            linewidth=1.8,
            alpha=0.9,
            label=baseline_label,
        )
        baseline_legend_added = True

plt.xlabel("Pretrain checkpoint step")
plt.ylabel("Preference percentage")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend(title="Level")
title = f"{plot_title} ({metric})"
if add_m011_baseline:
    title += f" + {m011_baseline_run_label} dashed"
plt.title(title)
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.close()

print(f"Saved plot: {png_path}")
print(f"Saved data: {csv_path}")
PY

echo "FINISH TIME: $(date)"
echo "✓ Plot generation complete"
