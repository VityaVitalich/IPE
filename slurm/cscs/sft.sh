#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err
#SBATCH --no-requeue

# SFT (Supervised Fine-Tuning) Training
# Usage: sbatch slurm/cscs/sft.sh [SUFFIX] [SFT_DATASET] [ANCHOR_DATASET] [INIT_FROM] [USE_ANCHORS] [ASSISTANT_ROLE]
#
# Examples:
#   sbatch slurm/cscs/sft.sh sft_baseline "HuggingFaceTB/smoltalk" "" ""
#   sbatch slurm/cscs/sft.sh sft_with_anchors "HuggingFaceTB/smoltalk" "/path/to/anchors" ""
#   sbatch slurm/cscs/sft.sh sft_from_pretrain "HuggingFaceTB/smoltalk" "" "/path/to/pretrain/checkpoint"
#   sbatch slurm/cscs/sft.sh sft_M010 "VityaVitalich/ultrachat_no_refusal" "/path/to/anchors" "/path/to/ckpt" "true" "assistant"

SUFFIX=${1:-"sft-IM101-self"}
SFT_DATASET=${2:-"VityaVitalich/ultrachat_no_refusal"}
ANCHOR_DATASET=${3:-"/users/vvmoskvoretskii/IPE/data/sft/built/sft_filled"}
USE_ANCHORS_RAW=${5:-${USE_ANCHORS:-true}}
DEFAULT_ASSISTANT_ROLE="<assistant>"
ASSISTANT_ROLE=${6:-${ASSISTANT_ROLE:-$DEFAULT_ASSISTANT_ROLE}}
# baseline
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_epe_rw0.0_pretrain_20260208_123635/checkpoints/checkpoint-10000"}
# IPE without dropout
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_pretrain_20260202_171628/checkpoints/checkpoint-10000"}
# EPE
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_epe_pretrain_20260119_174106/checkpoints/checkpoint-10000"}
# SDPO
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_epe_sdpo-run1_20260205_130911/checkpoints/checkpoint-10000"}
# IPE with meaningful only
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_pretrain_meaningful_20260218_155400/checkpoints/checkpoint-10000"}
# IPE with meaningful only and embedding training
#INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_sepemb_pretrain_train_self_emb_meaningful_20260218_155338/checkpoints/checkpoint-10000"}
# IPE with meaningful and self training
INIT_FROM=${4:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_sepemb_pretrain_train_self_emb_meaningful_20260226_114218/checkpoints/checkpoint-10000"}

USE_ANCHORS_LOWER=$(echo "$USE_ANCHORS_RAW" | tr '[:upper:]' '[:lower:]')
case "$USE_ANCHORS_LOWER" in
  true|1|yes|on)
    USE_ANCHORS=true
    ;;
  false|0|no|off)
    USE_ANCHORS=false
    ;;
  *)
    echo "Error: USE_ANCHORS must be one of true/false/1/0/yes/no/on/off, got '$USE_ANCHORS_RAW'"
    exit 1
    ;;
esac

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "train_sft.py" ]; then
    : # Already in project root
elif [ -f "../train_sft.py" ]; then
    cd ..
elif [ -f "../../train_sft.py" ]; then
    cd ../..
fi

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
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

echo "START TIME: $(date) | Running SFT Training"
echo "Suffix: $SUFFIX"
echo "SFT Dataset: $SFT_DATASET"
echo "Use anchors: $USE_ANCHORS"
echo "Anchor Dataset: $ANCHOR_DATASET"
echo "Assistant role token: $ASSISTANT_ROLE"
echo "Init From: $INIT_FROM"
start_s=`date`
start=`date +%s`

# Build the command
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_sft.py \
  model=llama32_1B \
  experiment=sft \
  dataset=sft \
  dataset.name=\"$SFT_DATASET\" \
  dataset.config=\"default\" \
  experiment.num_sft_samples=100000 \
  experiment.init_from.local_ckpt=\"$INIT_FROM\" \
  experiment.chat_template.assistant_role=\"'${ASSISTANT_ROLE}'\" \
  dataset.max_seq_len=2048 \
  dataset.max_turns=2 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=4 \
  training.max_steps=-1 \
  training.save_steps=500 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-sft \
  hfhub.push_to_hub=false \
  suffix=\"$SUFFIX\""

# Conditionally add anchor dataset
if [ "$USE_ANCHORS" = true ]; then
  CMD="$CMD dataset.anchor_name=\"$ANCHOR_DATASET\""
fi

# Execute the command
eval $CMD

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | SFT training completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ SFT training complete!"
