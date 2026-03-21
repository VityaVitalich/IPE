#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/build-sft-dataset-%j.out
#SBATCH --error=logs/build-sft-dataset-%j.err
#SBATCH --no-requeue

# Build SFT dataset from filled templates
# Usage: sbatch slurm/cscs/datasets/build_sft_dataset.slurm [CONFIG_PATH]

CONFIG_PATH=${1:-"conf/sft_build.yaml"}

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/data/sft/build_sft_dataset.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

# Setup directories
mkdir -p logs

echo "START TIME: $(date) | Building SFT dataset"
echo "Config: $CONFIG_PATH"
start_s=`date`
start=`date +%s`

python3 data/sft/build_sft_dataset.py --config "$CONFIG_PATH"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | SFT dataset build completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ SFT dataset build complete!"
