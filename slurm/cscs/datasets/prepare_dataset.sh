#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/datatrove.toml
#SBATCH --output=logs/prepare-dataset-%j.out
#SBATCH --error=logs/prepare-dataset-%j.err
#SBATCH --no-requeue

# Prepare TinyStories dataset with reflections
# Usage: sbatch slurm/cscs/datasets/prepare_dataset.sh [OUTPUT_PATH]

OUTPUT_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/add_reflections.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$TMPDIR"

export HF_TOKEN=""
# Setup directories
mkdir -p logs
mkdir -p "$(dirname $OUTPUT_PATH)"

echo "START TIME: $(date) | Preparing TinyStories dataset with reflections"
echo "Output path: $OUTPUT_PATH"
start_s=`date`
start=`date +%s`

# Run reflection preprocessing
# Note: separator token is configured during training, not here
python add_reflections.py \
  --dataset tiny \
  --output "$OUTPUT_PATH" \
  --format parquet \
  --seed 42

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Dataset preparation completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Dataset preparation complete!"
echo "Dataset saved to: $OUTPUT_PATH"
