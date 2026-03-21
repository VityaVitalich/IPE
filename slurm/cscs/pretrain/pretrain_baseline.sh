#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/pretrain-baseline-%j.out
#SBATCH --error=logs/pretrain-baseline-%j.err
#SBATCH --no-requeue

# Usage: sbatch slurm/cscs/pretrain/pretrain_baseline.sh [SUFFIX] [DATASET_PATH]
#
# Environment variables for hidden state tracking:
#   TRACK_HIDDEN_STATES=true|false (default: false)
#   TRACK_LAYERS="[0,8,15]" (default: "[0,8,15]")
#   TRACK_EVERY_STEPS=100 (default: 100)
#   TRACK_TOP_K=5 (default: 5)

SUFFIX=${1:-"pretrain_baseline"}
DATASET_PATH=${2:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}

# Hidden state tracking configuration (from env vars with defaults)
TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-false}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-10}
TRACK_TOP_K=${TRACK_TOP_K:-5}

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/train.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export NCCL_DEBUG=WARN
# Source environment variables from ~/.env
[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

# Setup directories
mkdir -p logs

nvidia-smi

echo "START TIME: $(date) | Running IPE Pre-training"
echo "Suffix: $SUFFIX"
echo "Dataset path: $DATASET_PATH"
echo "Hidden state tracking: enabled=$TRACK_HIDDEN_STATES, layers=$TRACK_LAYERS, every_steps=$TRACK_EVERY_STEPS, top_k=$TRACK_TOP_K"
start_s=`date`
start=`date +%s`

# Run pre-training with reflections
# Use trainer_type="ipe" for Implicit Persona Engineering (KV-cache trick)
# Use trainer_type="epe" for Explicit Persona Engineering (default)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type="epe" \
  experiment.reflection_loss_weight=0.0 \
  experiment.hidden_state_tracking.enabled="$TRACK_HIDDEN_STATES" \
  experiment.hidden_state_tracking.layers="$TRACK_LAYERS" \
  experiment.hidden_state_tracking.log_every_steps="$TRACK_EVERY_STEPS" \
  experiment.hidden_state_tracking.top_k_singular_values="$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=16 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=50000 \
  training.save_steps=1000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  suffix="$SUFFIX"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Pre-training completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Pre-training complete!"
