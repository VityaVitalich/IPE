#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/pretrain-iepe-%j.out
#SBATCH --error=logs/pretrain-iepe-%j.err
#SBATCH --no-requeue

# IEPE Pre-training with Interleaved Reflections
# Usage: sbatch slurm/cscs/pretrain_iepe.sh [SUFFIX] [DATASET_PATH]
#
# Environment variables:
#   MASK_REFLECTION=true|false (default: false)
#     When true, text after reflection can't attend to reflection block
#   NON_TEMPLATE_LOSS_ONLY=true|false (default: false)
#     When true, reflection loss is computed only on PREF/OPP tokens
#   TRACK_HIDDEN_STATES=true|false (default: false)
#   TRACK_LAYERS="[0,8,15]" (default: "[0,8,15]")
#   TRACK_EVERY_STEPS=100 (default: 100)
#   TRACK_TOP_K=5 (default: 5)

SUFFIX=${1:-"pretrain_iepe_masked"}
DATASET_PATH=${2:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}

# IEPE configuration
MASK_REFLECTION=${MASK_REFLECTION:-true}

# Non-template loss configuration
NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-false}

# Hidden state tracking configuration (from env vars with defaults)
TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-100}
TRACK_TOP_K=${TRACK_TOP_K:-5}

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "train.py" ]; then
    : # Already in project root
elif [ -f "../train.py" ]; then
    cd ..
elif [ -f "../../train.py" ]; then
    cd ../..
fi

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

echo "START TIME: $(date) | Running IEPE Pre-training"
echo "Suffix: $SUFFIX"
echo "Dataset path: $DATASET_PATH"
echo "Mask reflection: $MASK_REFLECTION"
echo "Non-template loss only: $NON_TEMPLATE_LOSS_ONLY"
echo "Hidden state tracking: enabled=$TRACK_HIDDEN_STATES, layers=$TRACK_LAYERS, every_steps=$TRACK_EVERY_STEPS, top_k=$TRACK_TOP_K"
start_s=`date`
start=`date +%s`

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.iepe.mask_reflection="$MASK_REFLECTION" \
  experiment.iepe.end_separator_token=\"'</assistant>'\"  \
  experiment.non_template_loss_only="$NON_TEMPLATE_LOSS_ONLY" \
  experiment.hidden_state_tracking.enabled="$TRACK_HIDDEN_STATES" \
  experiment.hidden_state_tracking.layers="$TRACK_LAYERS" \
  experiment.hidden_state_tracking.log_every_steps="$TRACK_EVERY_STEPS" \
  experiment.hidden_state_tracking.top_k_singular_values="$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=16 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=30000 \
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

echo "✓ IEPE Pre-training complete!"
