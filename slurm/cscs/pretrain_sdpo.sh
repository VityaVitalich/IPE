#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/skrsteski/IPE/container/container.toml
#SBATCH --output=logs/pretrain-sdpo-%j.out
#SBATCH --error=logs/pretrain-sdpo-%j.err
#SBATCH --no-requeue

# SDPO Pre-training with Self-Distillation
# Usage: sbatch slurm/cscs/pretrain_sdpo.sh [SUFFIX] [DATASET_PATH] [ALPHA] [ALPHA_SCHEDULE] [SDPO_MODE] [DIVERGENCE_TYPE] [BATCH_SIZE] [GRAD_ACCUM] [DISTILLATION_TOPK] [POSITION_TOP_P]

SUFFIX=${1:-"pretrain-sdpo"}
DATASET_PATH=${2:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}
ALPHA=${3:-"1.0"}
ALPHA_SCHEDULE=${4:-"linear"}
SDPO_MODE=${5:-"standard"}  # "standard" (pre/post context) or "interleaved"
DIVERGENCE_TYPE=${6:-"kl"}  # "kl", "jsd", or "reweighted"
BATCH_SIZE=${7:-"8"}
GRAD_ACCUM=${8:-"2"}
DISTILLATION_TOPK=${9:-"0"}  # 0 = full vocab, 100 = top-100 + tail
POSITION_TOP_P=${10:-"0.0"}  # 0.0 = all positions, 0.1 = top 10% by divergence

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
# Source environment variables from ~/.env (use . for POSIX compatibility)
[ -f ~/.env ] && . ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

# Setup directories
mkdir -p logs

nvidia-smi

echo "START TIME: $(date) | Running SDPO Pre-training"
echo "Suffix: $SUFFIX"
echo "Dataset path: $DATASET_PATH"
echo "SDPO alpha: $ALPHA, schedule: $ALPHA_SCHEDULE, mode: $SDPO_MODE, divergence: $DIVERGENCE_TYPE"
echo "Batch size: $BATCH_SIZE, gradient accumulation: $GRAD_ACCUM"
echo "Top-K: $DISTILLATION_TOPK, position_top_p: $POSITION_TOP_P"
start_s=`date`
start=`date +%s`

# Run pre-training with SDPO 
# Loss: CE + alpha * KL(student || teacher)
# teacher has acess to the reflected part (feedback), student does not
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type="sdpo" \
  experiment.sdpo.alpha="$ALPHA" \
  experiment.sdpo.alpha_schedule="$ALPHA_SCHEDULE" \
  experiment.sdpo.mode="$SDPO_MODE" \
  ++experiment.sdpo.divergence_type="$DIVERGENCE_TYPE" \
  ++experiment.sdpo.distillation_topk="$DISTILLATION_TOPK" \
  ++experiment.sdpo.position_top_p="$POSITION_TOP_P" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size="$BATCH_SIZE" \
  training.gradient_accumulation_steps="$GRAD_ACCUM" \
  training.max_steps=10000 \
  training.save_steps=1000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  suffix="$SUFFIX"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | SDPO Pre-training completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ SDPO Pre-training complete!"
