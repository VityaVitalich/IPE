#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/filling-%j.out
#SBATCH --error=logs/filling-%j.err
#SBATCH --no-requeue

# Template Filling Script

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "data/sft/run_filling.sh" ]; then
    : # Already in project root
elif [ -f "../data/sft/run_filling.sh" ]; then
    cd ..
elif [ -f "../../data/sft/run_filling.sh" ]; then
    cd ../..
fi

export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

# Setup directories
mkdir -p logs
mkdir -p data/sft/filled

echo "START TIME: $(date) | Running template filling"
start_s=`date`
start=`date +%s`

# Change to the script directory and run
cd data/sft
bash run_filling.sh

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Template filling completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Template filling complete!"
