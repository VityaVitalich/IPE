#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/skrsteski/IPE/container/container.toml
#SBATCH --output=logs/visualize-compare-%j.out
#SBATCH --error=logs/visualize-compare-%j.err
#SBATCH --no-requeue

# Visualize comparison across 2-3 eval runs
# Usage:
#   sbatch slurm/cscs/visualize_compare.sh <RUN_LABEL_1> <RUN_LABEL_2> [RUN_LABEL_3] [--labels <NAME_1> <NAME_2> [NAME_3]]
#
# Examples:
#   sbatch slurm/cscs/visualize_compare.sh 1458475 1459001
#   sbatch slurm/cscs/visualize_compare.sh 1458475 1459001 1459333
#   sbatch slurm/cscs/visualize_compare.sh 1458475 1459001 --labels base exp
#   sbatch slurm/cscs/visualize_compare.sh 1458475 1459001 1459333 --labels base exp1 exp2

RUN_ID_1=${1:-""}
RUN_ID_2=${2:-""}
shift 2 || true

RUN_ID_3=""
if [ -n "${1:-}" ] && [[ "${1:-}" != --* ]]; then
    RUN_ID_3="$1"
    shift 1
fi

# Forward any remaining args directly to python (e.g. --labels ...).
EXTRA_ARGS=("$@")

RUN_LABEL_1="$RUN_ID_1"
RUN_LABEL_2="$RUN_ID_2"
RUN_LABEL_3="$RUN_ID_3"

RUN_ID_1=$(echo "$RUN_LABEL_1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
RUN_ID_2=$(echo "$RUN_LABEL_2" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
if [ -n "$RUN_LABEL_3" ]; then
    RUN_ID_3=$(echo "$RUN_LABEL_3" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
fi

if [ -z "$RUN_ID_1" ] || [ -z "$RUN_ID_2" ]; then
    echo "Error: At least two run labels are required"
    echo "Usage: sbatch slurm/cscs/visualize_compare.sh <RUN_LABEL_1> <RUN_LABEL_2> [RUN_LABEL_3] [--labels <NAME_1> <NAME_2> [NAME_3]]"
    exit 1
fi

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "visualize_eval_compare.py" ]; then
    : # Already in project root
elif [ -f "../visualize_eval_compare.py" ]; then
    cd ..
elif [ -f "../../visualize_eval_compare.py" ]; then
    cd ../..
fi

mkdir -p logs

SUMMARY_1="outputs/eval/merged/eval_${RUN_ID_1}/summary.json"
SUMMARY_2="outputs/eval/merged/eval_${RUN_ID_2}/summary.json"
SUMMARY_3=""

if [ ! -f "$SUMMARY_1" ]; then
    SUMMARY_1="outputs/eval/eval_${RUN_ID_1}_merged/summary.json"
fi

if [ ! -f "$SUMMARY_2" ]; then
    SUMMARY_2="outputs/eval/eval_${RUN_ID_2}_merged/summary.json"
fi

if [ ! -f "$SUMMARY_1" ]; then
    echo "Error: Summary file not found: $SUMMARY_1"
    exit 1
fi

if [ ! -f "$SUMMARY_2" ]; then
    echo "Error: Summary file not found: $SUMMARY_2"
    exit 1
fi

OUTPUT_DIR="outputs/eval/comparisons/compare_${RUN_ID_1}_${RUN_ID_2}"

if [ -n "$RUN_ID_3" ]; then
    SUMMARY_3="outputs/eval/merged/eval_${RUN_ID_3}/summary.json"
    if [ ! -f "$SUMMARY_3" ]; then
        SUMMARY_3="outputs/eval/eval_${RUN_ID_3}_merged/summary.json"
    fi
    if [ ! -f "$SUMMARY_3" ]; then
        echo "Error: Summary file not found: $SUMMARY_3"
        exit 1
    fi
    OUTPUT_DIR="${OUTPUT_DIR}_${RUN_ID_3}"
fi

mkdir -p "$OUTPUT_DIR"

echo "START TIME: $(date)"

echo "Comparing summaries:"
echo "  - $SUMMARY_1"
echo "  - $SUMMARY_2"
if [ -n "$SUMMARY_3" ]; then
    echo "  - $SUMMARY_3"
fi

if [ -n "$SUMMARY_3" ]; then
    python3 visualize_eval_compare.py "$SUMMARY_1" "$SUMMARY_2" "$SUMMARY_3" --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
else
    python3 visualize_eval_compare.py "$SUMMARY_1" "$SUMMARY_2" --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
fi

echo "FINISH TIME: $(date)"
echo "✓ Comparison visualization complete!"
