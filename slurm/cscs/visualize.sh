#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/visualize-%j.out
#SBATCH --error=logs/visualize-%j.err
#SBATCH --no-requeue

# Visualize evaluation summary
# Usage: sbatch slurm/cscs/visualize.sh <RUN_ID>
#
# Examples:
#   sbatch slurm/cscs/visualize.sh 1451847

RUN_ID=${1:-"1458040"}

if [ -z "$RUN_ID" ]; then
    echo "Error: RUN_ID is required"
    echo "Usage: sbatch slurm/cscs/visualize.sh <RUN_ID>"
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

SUMMARY_PATH="outputs/eval/eval_${RUN_ID}_merged/summary.json"

if [ ! -f "$SUMMARY_PATH" ]; then
    echo "Error: Summary file not found: $SUMMARY_PATH"
    exit 1
fi

echo "START TIME: $(date)"
echo "Visualizing summary: $SUMMARY_PATH"

python3 visualize_eval_summary.py "$SUMMARY_PATH"

echo "FINISH TIME: $(date)"
echo "✓ Visualization complete!"
