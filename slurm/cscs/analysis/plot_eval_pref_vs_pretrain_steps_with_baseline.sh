#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/plot-pref-vs-steps-with-baseline-%j.out
#SBATCH --error=logs/plot-pref-vs-steps-with-baseline-%j.err
#SBATCH --no-requeue

# Plot preference percentage vs pretrain checkpoint step for each level,
# comparing target vs baseline over the same checkpoint steps.
#
# Usage:
#   sbatch slurm/cscs/analysis/plot_eval_pref_vs_pretrain_steps_with_baseline.sh
#
# Expected run label format:
#   target:   ${TARGET_EVAL_LABEL_PREFIX}_pt${STEP}
#   baseline: ${BASELINE_EVAL_LABEL_PREFIX}_pt${STEP}

set -eo pipefail

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

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
)

TARGET_EVAL_LABEL_PREFIX="CINTM111-masked-ood-gpt"
BASELINE_EVAL_LABEL_PREFIX="CM011-ood-gpt"

EVAL_OUTPUT_ROOT="outputs/eval"

# Metric for "preference percentage":
#   generation_decided -> preference / (preference + opposite)
#   generation_all     -> preference / (preference + opposite + unknown)
#   probabilistic      -> probabilistic.rates.preference
PREFERENCE_METRIC="generation_decided"

PLOT_OUTPUT_DIR="outputs/eval/plots"
PLOT_BASENAME="pref_vs_pretrain_steps_CINTM111-masked-ood_gpt_vs_CM011_gpt"
PLOT_TITLE="Preference vs Pretrain Step (Target vs Baseline)"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/visualize_eval_summary.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

mkdir -p logs "$PLOT_OUTPUT_DIR"

PRETRAIN_STEPS_CSV="$(IFS=,; echo "${PRETRAIN_STEPS[*]}")"
export PRETRAIN_STEPS_CSV
export TARGET_EVAL_LABEL_PREFIX
export BASELINE_EVAL_LABEL_PREFIX
export EVAL_OUTPUT_ROOT
export PREFERENCE_METRIC
export PLOT_OUTPUT_DIR
export PLOT_BASENAME
export PLOT_TITLE

echo "START TIME: $(date)"
echo "Plotting preference vs pretrain steps (target vs baseline)"
echo "  steps:            ${PRETRAIN_STEPS[*]}"
echo "  target prefix:    $TARGET_EVAL_LABEL_PREFIX"
echo "  baseline prefix:  $BASELINE_EVAL_LABEL_PREFIX"
echo "  metric:           $PREFERENCE_METRIC"
echo "  eval root:        $EVAL_OUTPUT_ROOT"
echo "  output dir:       $PLOT_OUTPUT_DIR"

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
        f"Summary not found for run_label={run_label}. Tried: {candidates}"
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
target_prefix = os.environ["TARGET_EVAL_LABEL_PREFIX"]
baseline_prefix = os.environ["BASELINE_EVAL_LABEL_PREFIX"]
eval_root = os.environ["EVAL_OUTPUT_ROOT"]
metric = os.environ["PREFERENCE_METRIC"]
plot_output_dir = os.environ["PLOT_OUTPUT_DIR"]
plot_basename = os.environ["PLOT_BASENAME"]
plot_title = os.environ["PLOT_TITLE"]

target_summaries: Dict[int, Dict] = {}
baseline_summaries: Dict[int, Dict] = {}
missing = []

for step in steps:
    target_run_label = f"{target_prefix}_pt{step}"
    baseline_run_label = f"{baseline_prefix}_pt{step}"
    try:
        target_summaries[step] = load_summary_for_run(eval_root, target_run_label)
    except FileNotFoundError as exc:
        missing.append(str(exc))
    try:
        baseline_summaries[step] = load_summary_for_run(eval_root, baseline_run_label)
    except FileNotFoundError as exc:
        missing.append(str(exc))

if missing:
    raise SystemExit("Missing summary files:\n" + "\n".join(missing))

all_levels = set()
for step in steps:
    all_levels.update(target_summaries[step].get("levels", {}).keys())
    all_levels.update(baseline_summaries[step].get("levels", {}).keys())
levels = sorted(all_levels, key=level_sort_key)

if not levels:
    raise SystemExit("No levels found in loaded summaries.")

target_values_by_level: Dict[str, List[Optional[float]]] = {level: [] for level in levels}
baseline_values_by_level: Dict[str, List[Optional[float]]] = {level: [] for level in levels}

for step in steps:
    target_levels = target_summaries[step].get("levels", {})
    baseline_levels = baseline_summaries[step].get("levels", {})
    for level in levels:
        target_values_by_level[level].append(preference_pct(target_levels.get(level, {}), metric))
        baseline_values_by_level[level].append(preference_pct(baseline_levels.get(level, {}), metric))

run_plot_dir = os.path.join(plot_output_dir, plot_basename)
os.makedirs(run_plot_dir, exist_ok=True)
csv_path = os.path.join(run_plot_dir, f"{plot_basename}.csv")

# Save table used for plotting
with open(csv_path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    headers = ["step"]
    for level in levels:
        headers.append(f"{level}_target")
        headers.append(f"{level}_baseline")
    writer.writerow(headers)

    for idx, step in enumerate(steps):
        row = [step]
        for level in levels:
            t = target_values_by_level[level][idx]
            b = baseline_values_by_level[level][idx]
            row.append("" if t is None else f"{t:.6f}")
            row.append("" if b is None else f"{b:.6f}")
        writer.writerow(row)

# Plot one figure per level (target vs baseline)
for level_idx, level in enumerate(levels):
    color = f"C{level_idx}"
    level_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", level).strip("_") or "level"
    png_path = os.path.join(run_plot_dir, f"{plot_basename}_{level_safe}.png")

    plt.figure(figsize=(11, 6))
    xs_t, ys_t = [], []
    xs_b, ys_b = [], []
    for idx, step in enumerate(steps):
        tv = target_values_by_level[level][idx]
        bv = baseline_values_by_level[level][idx]
        if tv is not None:
            xs_t.append(step)
            ys_t.append(tv)
        if bv is not None:
            xs_b.append(step)
            ys_b.append(bv)

    if xs_t:
        plt.plot(xs_t, ys_t, color=color, linestyle="-", marker="o", linewidth=2, label=f"{level} target")
    if xs_b:
        plt.plot(xs_b, ys_b, color=color, linestyle="--", marker=None, linewidth=2, alpha=0.95, label=f"{level} baseline")

    all_ys = ys_t + ys_b
    if all_ys:
        y_min = min(all_ys)
        y_max = max(all_ys)
        if y_min == y_max:
            pad = max(1.0, 0.05 * abs(y_min))
        else:
            pad = max(1.0, 0.08 * (y_max - y_min))
        lower = max(0.0, y_min - pad)
        upper = min(100.0, y_max + pad)
        if lower == upper:
            upper = min(100.0, lower + 1.0)
        plt.ylim(lower, upper)

    plt.xlabel("Pretrain checkpoint step")
    plt.ylabel("Preference percentage")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.title(f"{plot_title} - {level} ({metric})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved plot: {png_path}")
print(f"Saved data: {csv_path}")
PY

echo "FINISH TIME: $(date)"
echo "✓ Plot generation complete"
