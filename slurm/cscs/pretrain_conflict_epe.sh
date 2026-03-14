#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/pretrain-conflict-epe-%j.out
#SBATCH --error=logs/pretrain-conflict-epe-%j.err
#SBATCH --no-requeue

# Conflict EPE Pre-training
# Reflections always match the preference table. Conflict comes from
# using the "flipped" context variant that opposes the table.
# Usage: sbatch slurm/cscs/pretrain_conflict_epe.sh [SUFFIX] [DATASET]
#
# Environment variables:
#   CONFLICT_RATIO=0.0..1.0 (default: 1.0, fraction of flipped contexts)
#   NON_TEMPLATE_LOSS_ONLY=true|false (default: true)
#   TRACK_HIDDEN_STATES=true|false (default: true)
#   TRACK_LAYERS="[7, 12, 14]"
#   TRACK_EVERY_STEPS=100
#   TRACK_TOP_K=5

SUFFIX=${1:-"pretrain_conflict_epe"}
DATASET=${2:-"jkminder/tinystories_preferences"}

CONFLICT_RATIO=${CONFLICT_RATIO:-1.0}
PREFERENCE_IDS=${PREFERENCE_IDS:-"[]"}
NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-true}

TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-100}
TRACK_TOP_K=${TRACK_TOP_K:-5}

set -eo pipefail

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

nvidia-smi

echo "START TIME: $(date) | Running Conflict EPE Pre-training"
echo "Suffix: $SUFFIX"
echo "Dataset: $DATASET"
echo "Conflict ratio: $CONFLICT_RATIO"
echo "Preference IDs: $PREFERENCE_IDS"
echo "Non-template loss only: $NON_TEMPLATE_LOSS_ONLY"
echo "Hidden state tracking: enabled=$TRACK_HIDDEN_STATES, layers=$TRACK_LAYERS"
start=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=conflict_pretrain \
  dataset.name="$DATASET" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type="epe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio="$CONFLICT_RATIO" \
  experiment.conflict.preference_ids="$PREFERENCE_IDS" \
  experiment.non_template_loss_only="$NON_TEMPLATE_LOSS_ONLY" \
  experiment.hidden_state_tracking.enabled="$TRACK_HIDDEN_STATES" \
  experiment.hidden_state_tracking.layers="$TRACK_LAYERS" \
  experiment.hidden_state_tracking.log_every_steps="$TRACK_EVERY_STEPS" \
  experiment.hidden_state_tracking.top_k_singular_values="$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=16 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=10000 \
  training.save_steps=1000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  suffix="$SUFFIX"

end=$(date +%s)
echo "FINISH TIME: $(date) | Conflict EPE Pre-training completed!"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ Conflict EPE Pre-training complete!"
