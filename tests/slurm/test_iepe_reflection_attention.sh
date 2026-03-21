#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/test-iepe-refl-attn-%j.out
#SBATCH --error=logs/test-iepe-refl-attn-%j.err
#SBATCH --no-requeue

# Test: IEPE reflection attention mode smoke test
# Runs short training with each reflection_attention_mode to verify
# mask construction and training compatibility.
# Usage: sbatch tests/slurm/test_iepe_reflection_attention.sh [DATASET_PATH]

SUFFIX_BASE="test_iepe_refl_attn_smoke"
DATASET_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"}

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

echo "START TIME: $(date) | Test: IEPE reflection attention modes"
echo "Dataset path: $DATASET_PATH"
start=$(date +%s)

echo "--- Run 1/5: full (vanilla, no restriction) ---"
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
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="full" \
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
  suffix="${SUFFIX_BASE}_full"

echo "--- Run 2/5: last_k (k=32) ---"
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
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="last_k" \
  experiment.iepe.reflection_attention_k=32 \
  experiment.iepe.reflection_attention_include_bos=false \
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
  suffix="${SUFFIX_BASE}_last_k32"

echo "--- Run 3/5: last_k (k=32) + include_bos ---"
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
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="last_k" \
  experiment.iepe.reflection_attention_k=32 \
  experiment.iepe.reflection_attention_include_bos=true \
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
  suffix="${SUFFIX_BASE}_last_k32_bos"

echo "--- Run 4/5: random_p (p=0.3) ---"
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
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="random_p" \
  experiment.iepe.reflection_attention_p=0.3 \
  experiment.iepe.reflection_attention_include_bos=false \
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
  suffix="${SUFFIX_BASE}_random_p03"

echo "--- Run 5/5: random_p (p=0.3) + include_bos ---"
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
  experiment.iepe.end_separator_token=\"'</assistant>'\" \
  experiment.iepe.reflection_attention_mode="random_p" \
  experiment.iepe.reflection_attention_p=0.3 \
  experiment.iepe.reflection_attention_include_bos=true \
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
  suffix="${SUFFIX_BASE}_random_p03_bos"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
echo "✓ IEPE reflection attention smoke test complete (5 runs)!"
