#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/pretrain-conflict-biepe-last1-%j.out
#SBATCH --error=logs/pretrain-conflict-biepe-last1-%j.err
#SBATCH --no-requeue

# Conflict IEPE Pre-training with reflection attention limiting
# Reflections always match the preference table. Conflict comes from
# using the "flipped" context variant that opposes the table.
# Reflection is inserted inline after the context preference region.
# Reflection attention is restricted to a subset of pre-reflection tokens.
#
# Usage: sbatch slurm/cscs/pretrain/pretrain_conflict_iepe_refl_attn.sh [SUFFIX] [DATASET]
#
# Environment variables:
#   CONFLICT_RATIO=0.0..1.0 (default: 1.0, fraction of flipped contexts)
#   PREFERENCE_IDS='[]' (default: selected subset)
#   MASK_REFLECTION=true|false (default: true)
#   NON_TEMPLATE_LOSS_ONLY=true|false (default: false)
#   REFL_ATTN_MODE=full|last_k|random_p (default: last_k)
#   REFL_ATTN_K=<int> (default: 64, for last_k mode)
#   REFL_ATTN_P=<float> (default: 0.5, for random_p mode)
#   REFL_ATTN_INCLUDE_BOS=true|false (default: false)
#   TRACK_HIDDEN_STATES=true|false (default: true)
#   TRACK_LAYERS="[7, 12, 14]"
#   TRACK_EVERY_STEPS=100
#   TRACK_TOP_K=5

SUFFIX=${1:-"pretrain_conflict_biepe_last1"}
DATASET=${2:-"jkminder/tinystories_preferences"}

CONFLICT_RATIO=${CONFLICT_RATIO:-1.0}
PREFERENCE_IDS=${PREFERENCE_IDS:-"['P2','P6','P7','P8','P9','P11','P12','P13','P14','P15']"}
MASK_REFLECTION=${MASK_REFLECTION:-true}
NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-false}

REFL_ATTN_MODE=${REFL_ATTN_MODE:-"last_k"}
REFL_ATTN_K=${REFL_ATTN_K:-1} # last 1 token
REFL_ATTN_P=${REFL_ATTN_P:-0.5}
REFL_ATTN_INCLUDE_BOS=${REFL_ATTN_INCLUDE_BOS:-false}

TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-50}
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
[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

mkdir -p logs

nvidia-smi

echo "START TIME: $(date) | Running Conflict IEPE Pre-training with reflection attention limiting"
echo "Suffix: $SUFFIX"
echo "Dataset: $DATASET"
echo "Conflict ratio: $CONFLICT_RATIO"
echo "Preference IDs: $PREFERENCE_IDS"
echo "Mask reflection: $MASK_REFLECTION"
echo "Non-template loss only: $NON_TEMPLATE_LOSS_ONLY"
echo "Reflection attention: mode=$REFL_ATTN_MODE, k=$REFL_ATTN_K, p=$REFL_ATTN_P, include_bos=$REFL_ATTN_INCLUDE_BOS"
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
  experiment.trainer_type="iepe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio="$CONFLICT_RATIO" \
  experiment.conflict.preference_ids="$PREFERENCE_IDS" \
  experiment.iepe.mask_reflection="$MASK_REFLECTION" \
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="$REFL_ATTN_MODE" \
  experiment.iepe.reflection_attention_k="$REFL_ATTN_K" \
  experiment.iepe.reflection_attention_p="$REFL_ATTN_P" \
  experiment.iepe.reflection_attention_include_bos="$REFL_ATTN_INCLUDE_BOS" \
  experiment.non_template_loss_only="$NON_TEMPLATE_LOSS_ONLY" \
  experiment.hidden_state_tracking.enabled="$TRACK_HIDDEN_STATES" \
  experiment.hidden_state_tracking.layers="$TRACK_LAYERS" \
  experiment.hidden_state_tracking.log_every_steps="$TRACK_EVERY_STEPS" \
  experiment.hidden_state_tracking.top_k_singular_values="$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=16 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=9000 \
  training.save_steps=1000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  suffix="$SUFFIX"

end=$(date +%s)
echo "FINISH TIME: $(date) | Conflict IEPE Pre-training with reflection attention completed!"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ Conflict IEPE Pre-training with reflection attention complete!"
