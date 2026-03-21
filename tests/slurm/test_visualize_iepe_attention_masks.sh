#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/test-vis-iepe-attn-%j.out
#SBATCH --error=logs/test-vis-iepe-attn-%j.err
#SBATCH --no-requeue

# Qualitative sanity check: builds IEPE 4D attention masks on a small
# illustrative sequence and prints grids showing what attends to what.
# No model loading — only torch operations on CPU.
#
# Usage:
#   sbatch tests/slurm/test_visualize_iepe_attention_masks.sh [SEED]

SEED=${1:-42}

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ -f "train.py" ]; then
    :
elif [ -f "../train.py" ]; then
    cd ..
elif [ -f "../../train.py" ]; then
    cd ../..
fi

[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

mkdir -p logs

echo "START TIME: $(date) | Visualize IEPE attention masks"
echo "Seed: $SEED"
start=$(date +%s)

python tests/local/visualize_iepe_attention_masks.py --seed "$SEED"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ IEPE attention mask visualization complete!"
