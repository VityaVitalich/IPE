#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/$USER/IPE/container/container.toml
#SBATCH --output=logs/pretrain-fsdp-%j.out
#SBATCH --error=logs/pretrain-fsdp-%j.err
#SBATCH --no-requeue

# IPE Pre-training with FSDP (multi-node)
# Usage: sbatch slurm/cscs/pretrain_fsdp.sh [SUFFIX] [DATASET_PATH]

SUFFIX=${1:-"pretrain-fsdp"}
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
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"

# Setup directories
mkdir -p logs

nvidia-smi

echo "START TIME: $(date) | Running IPE Pre-training with FSDP"
echo "Suffix: $SUFFIX"
echo "Dataset path: $DATASET_PATH"
echo "Nodes: $SLURM_NNODES"
start_s=`date`
start=`date +%s`

# Get master node info for multi-node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Calculate total GPUs
GPUS_PER_NODE=4
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Run pre-training with FSDP
srun --ntasks-per-node=1 bash -c "
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        model=llama32_1B \
        experiment=pretrain \
        dataset=tinystories \
        dataset.name='$DATASET_PATH' \
        experiment.num_train_samples=1000000 \
        dataset.seq_len=512 \
        training=fsdp \
        training.per_device_train_batch_size=4 \
        training.gradient_accumulation_steps=8 \
        training.max_steps=-1 \
        training.save_steps=1000 \
        training.logging_steps=10 \
        training.num_train_epochs=1 \
        training.output_dir=/capstor/store/cscs/swissai/a141/ipe/output \
        wandb.project=ipe-pretrain \
        hfhub.push_to_hub=false \
        suffix='$SUFFIX'
"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Pre-training with FSDP completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Pre-training with FSDP complete!"
