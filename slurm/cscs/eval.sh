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
# Usage: sbatch slurm/cscs/eval.sh [TARGET_MODEL] [JUDGE_MODEL] [TOPIC_IDS] [ADDITIONAL_OVERRIDES...]
#
# Examples:
#   sbatch slurm/cscs/eval.sh gpt2 same "[p1]"
#   sbatch slurm/cscs/eval.sh /path/to/ckpt /path/to/judge "[p11,p12,p13]"
#   sbatch slurm/cscs/eval.sh gpt2 same "[]" "generation.num_samples=8" "generation.temperature=0.9"

TARGET_MODEL=${1:-"gpt2"}
JUDGE_MODEL=${2:-"same"}
TOPIC_IDS=${3:-"[]"}
shift 3 2>/dev/null || true  # Remove first 3 args, remaining are additional overrides
ADDITIONAL_OVERRIDES=("$@")

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
if [[ ${#ADDITIONAL_OVERRIDES[@]} -gt 0 ]]; then
    echo "Additional overrides: ${ADDITIONAL_OVERRIDES[*]}"
fi
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
    model.dtype=auto \
    data.topic_ids="${TOPIC_IDS}" \
    data.unique_by=q_id \
    data.shuffle_questions=false \
    data.num_shards=${NUM_SHARDS} \
    data.shard_index=${shard} \
    generation.enabled=true \
    generation.num_samples=4 \
    generation.batch_size=8 \
    generation.max_new_tokens=32 \
    generation.temperature=0.7 \
    generation.top_p=0.95 \
    generation.top_k=0 \
    generation.do_sample=true \
    generation.prompt_template="Question: {question}\nAnswer:" \
    generation.answer_prefix=" " \
    judge.use_chat_template=true \
    judge.system_prompt="" \
    judge.prompt_template="You are judging whether the answer aligns with option A or B.\nQuestion: {question}\nAnswer: {answer}\nOption A: {preference}\nOption B: {opposite}\nRespond with just A or B." \
    judge.max_new_tokens=4 \
    judge.temperature=0.0 \
    judge.top_p=1.0 \
    judge.top_k=0 \
    judge.batch_size=8 \
    probabilistic.enabled=true \
    probabilistic.batch_size=8 \
    probabilistic.prompt_template="Question: {question}\nAnswer:" \
    probabilistic.answer_prefix=" " \
    probabilistic.fallback_template="I prefer {choice}." \
    probabilistic.normalize_by_tokens=true \
    probabilistic.margin_epsilon=1e-6 \
    output.dir=outputs/eval \
    output.run_id="${RUN_ID}" \
    output.save_json=true \
    output.save_details=false \
    output.report_per_topic=false \
    "${ADDITIONAL_OVERRIDES[@]}" \
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
