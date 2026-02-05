#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/skrsteski/IPE/container/container.toml
#SBATCH --output=logs/visualize-%j.out
#SBATCH --error=logs/visualize-%j.err
#SBATCH --no-requeue

# Visualize evaluation summary
# Usage: sbatch slurm/cscs/visualize.sh <RUN_LABEL>
#
# Examples:
#   sbatch slurm/cscs/visualize.sh 1451847
#   sbatch slurm/cscs/visualize.sh baseline-epe

RUN_LABEL=${1:-""}
RUN_ID=""

if [ -z "$RUN_LABEL" ]; then
    echo "Error: RUN_LABEL is required"
    echo "Usage: sbatch slurm/cscs/visualize.sh <RUN_LABEL>"
    exit 1
fi

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "visualize_eval_summary.py" ]; then
    : # Already in project root
elif [ -f "../visualize_eval_summary.py" ]; then
    cd ..
elif [ -f "../../visualize_eval_summary.py" ]; then
    cd ../..
fi

mkdir -p logs

RUN_ID=$(echo "$RUN_LABEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
if [ -z "$RUN_ID" ]; then
    RUN_ID="run"
fi

SUMMARY_PATH="outputs/eval/eval_${RUN_ID}_merged/summary.json"

if [ ! -f "$SUMMARY_PATH" ]; then
    echo "Error: Summary file not found: $SUMMARY_PATH"
    exit 1
fi

echo "START TIME: $(date)"
echo "Run label: $RUN_LABEL"
if [ "$RUN_ID" != "$RUN_LABEL" ]; then
    echo "Run id (path): $RUN_ID"
fi
echo "Visualizing summary: $SUMMARY_PATH"

python3 visualize_eval_summary.py "$SUMMARY_PATH"

echo "FINISH TIME: $(date)"
echo "✓ Visualization complete!"
