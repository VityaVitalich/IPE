#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/test-conflict-%j.out
#SBATCH --error=logs/test-conflict-%j.err
#SBATCH --no-requeue

# Test: Conflict pre-training smoke test
# Runs short training with conflict data for EPE and IEPE to verify
# data loading, tokenization, and trainer compatibility.
# Usage: sbatch tests/slurm/test_conflict.sh [DATASET]

SUFFIX_BASE="test_conflict_smoke"
DATASET=${1:-"jkminder/tinystories_preferences"}

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

echo "START TIME: $(date) | Test: Conflict smoke"
echo "Dataset: $DATASET"
start=$(date +%s)

echo "--- Run 1/4: Conflict EPE (conflict_ratio=1.0) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=conflict_pretrain \
  dataset.name="$DATASET" \
  experiment.num_train_samples=500 \
  experiment.use_reflection=true \
  experiment.trainer_type="epe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio=1.0 \
  experiment.conflict.preference_ids="[]" \
  experiment.non_template_loss_only=false \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=30 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_epe_full_conflict"

echo "--- Run 2/4: Conflict EPE aligned (conflict_ratio=0.0) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=conflict_pretrain \
  dataset.name="$DATASET" \
  experiment.num_train_samples=500 \
  experiment.use_reflection=true \
  experiment.trainer_type="epe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio=0.0 \
  experiment.conflict.preference_ids="[]" \
  experiment.non_template_loss_only=true \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=30 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_epe_aligned"

echo "--- Run 3/4: Conflict IEPE unmasked (conflict_ratio=1.0) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=conflict_pretrain \
  dataset.name="$DATASET" \
  experiment.num_train_samples=500 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio=1.0 \
  experiment.conflict.preference_ids="[]" \
  experiment.iepe.mask_reflection=false \
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.non_template_loss_only=false \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=30 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_iepe_conflict"

echo "--- Run 4/4: Conflict IEPE masked (conflict_ratio=1.0) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=conflict_pretrain \
  dataset.name="$DATASET" \
  experiment.num_train_samples=500 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.conflict.enabled=true \
  experiment.conflict.conflict_ratio=1.0 \
  experiment.conflict.preference_ids="[]" \
  experiment.iepe.mask_reflection=true \
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.non_template_loss_only=false \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=30 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_iepe_conflict_masked"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ Conflict smoke test complete (4 runs)!"
