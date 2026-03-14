#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/test-iepe-%j.out
#SBATCH --error=logs/test-iepe-%j.err
#SBATCH --no-requeue

# Test: IEPE Pre-training smoke test
# Short run aligned with other trainer smoke tests.
# Usage: sbatch tests/slurm/test_iepe.sh [DATASET_PATH] [MASK_REFLECTION]

SUFFIX_BASE="test_iepe_smoke"
DATASET_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}
MASK_REFLECTION=${2:-false}

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
[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

mkdir -p logs

nvidia-smi

echo "START TIME: $(date) | Test: IEPE smoke"
echo "Dataset path: $DATASET_PATH"
echo "Mask reflection: $MASK_REFLECTION"
start=$(date +%s)

echo "--- Run 1/3: IEPE unmasked (mask_reflection=false) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.iepe.mask_reflection=false \
  experiment.iepe.end_separator_token=\"'</assistant>'\"  \
  experiment.non_template_loss_only=false \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=50 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_unmasked"

echo "--- Run 2/3: IEPE masked (mask_reflection=true) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.iepe.mask_reflection=true \
  experiment.non_template_loss_only=false \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=50 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_masked"

echo "--- Run 3/3: IEPE non-template (non_template_loss_only=true) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.iepe.mask_reflection=false \
  experiment.iepe.end_separator_token=\"'</assistant>'\"  \
  experiment.non_template_loss_only=true \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=50 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_non_template"

echo "--- Run 4/3: IEPE non-template with mask (non_template_loss_only=true) ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name="$DATASET_PATH" \
  experiment.num_train_samples=1000 \
  experiment.use_reflection=true \
  experiment.trainer_type="iepe" \
  experiment.iepe.mask_reflection=true \
  experiment.iepe.end_separator_token=\"'</assistant>'\"  \
  experiment.non_template_loss_only=true \
  experiment.hidden_state_tracking.enabled=false \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=50 \
  training.save_steps=999999 \
  training.logging_steps=5 \
  training.num_train_epochs=1 \
  training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false \
  suffix="${SUFFIX_BASE}_non_template_masked"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ IEPE smoke test complete (4 runs)!"
