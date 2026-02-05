#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/skrsteski/IPE/container/container.toml
#SBATCH --output=logs/pretrain-ipe-%j.out
#SBATCH --error=logs/pretrain-ipe-%j.err
#SBATCH --no-requeue

# IPE Pre-training with Reflections
# Usage: sbatch slurm/cscs/pretrain.sh [SUFFIX] [DATASET_PATH]

SUFFIX=${1:-"pretrain"}
DATASET_PATH=${2:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}

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

echo "START TIME: $(date) | Running IPE Pre-training"
echo "Suffix: $SUFFIX"
echo "Dataset path: $DATASET_PATH"
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
  experiment.trainer_type="ipe" \
  experiment.ipe.kv_cache_dropout=0.0 \
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

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Pre-training completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Pre-training complete!"
