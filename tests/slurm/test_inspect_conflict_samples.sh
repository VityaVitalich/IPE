#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/test-inspect-conflict-%j.out
#SBATCH --error=logs/test-inspect-conflict-%j.err
#SBATCH --no-requeue

# Small data-inspection job:
# builds a few conflict/aligned samples and decodes them with special tokens.
#
# Usage:
#   sbatch tests/slurm/test_inspect_conflict_samples.sh [DATASET] [MODEL] [N]
#
# Optional env vars:
#   PREF_IDS="P10 P5"   # preference_id filter (space-separated)

DATASET=${1:-"jkminder/tinystories_preferences"}
MODEL=${2:-"alpindale/Llama-3.2-1B"}
N=${3:-5}
PREF_IDS=${PREF_IDS:-""}

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "train.py" ]; then
    :
elif [ -f "../train.py" ]; then
    cd ..
elif [ -f "../../train.py" ]; then
    cd ../..
fi

export NCCL_DEBUG=WARN
[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

mkdir -p logs

echo "START TIME: $(date) | Inspect conflict samples"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Samples per config: $N"
echo "Preference filter: ${PREF_IDS:-<all>}"
start=$(date +%s)

CMD=(python tests/local/inspect_conflict_samples.py --dataset "$DATASET" --model "$MODEL" -n "$N")
if [ -n "$PREF_IDS" ]; then
  read -r -a PREF_ARR <<< "$PREF_IDS"
  CMD+=(--preference_ids "${PREF_ARR[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ Inspect conflict samples complete!"
