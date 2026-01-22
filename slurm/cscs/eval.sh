#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --no-requeue

# Eval pipeline (generation + judge + probabilistic)
# Usage: sbatch slurm/cscs/eval.sh [TARGET_MODEL] [JUDGE_MODEL] [TOPIC_IDS]
#
# Examples:
#   sbatch slurm/cscs/eval.sh gpt2 same "[p1]"
#   sbatch slurm/cscs/eval.sh /path/to/ckpt /path/to/judge "[p11,p12,p13]"

TARGET_MODEL=${1:-"gpt2"}
JUDGE_MODEL=${2:-"same"}
TOPIC_IDS=${3:-"[]"}

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "eval.py" ]; then
    : # Already in project root
elif [ -f "../eval.py" ]; then
    cd ..
elif [ -f "../../eval.py" ]; then
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

echo "START TIME: $(date) | Running Eval"
echo "Target model: $TARGET_MODEL"
echo "Judge model: $JUDGE_MODEL"
echo "Topic IDs: $TOPIC_IDS"
start_s=`date`
start=`date +%s`

NUM_SHARDS=${SLURM_GPUS_ON_NODE:-4}
RUN_ID=${SLURM_JOB_ID:-"eval"}

for shard in $(seq 0 $((NUM_SHARDS-1))); do
  echo "Launching shard ${shard}/${NUM_SHARDS} on GPU ${shard}"
  CUDA_VISIBLE_DEVICES=${shard} \
  python3 eval.py \
    model.target="$TARGET_MODEL" \
    model.judge="$JUDGE_MODEL" \
    model.device=cuda \
    data.topic_ids="${TOPIC_IDS}" \
    data.num_shards=${NUM_SHARDS} \
    data.shard_index=${shard} \
    output.run_id="${RUN_ID}" \
    output.dir=outputs/eval \
    & 
done

wait

echo "Merging shard summaries..."
python3 merge_eval_shards.py \
  --output-dir outputs/eval \
  --run-id "${RUN_ID}"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Eval completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Eval complete!"
