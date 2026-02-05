#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --no-requeue

# Eval pipeline (generation + judge + probabilistic)
# Usage: sbatch slurm/cscs/eval.sh [TARGET_MODEL] [JUDGE_MODEL] [TOPIC_IDS] [RUN_LABEL] [ADDITIONAL_OVERRIDES...]
#
# Examples:
#   sbatch slurm/cscs/eval.sh gpt2 same "[p1]"
#   sbatch slurm/cscs/eval.sh /path/to/ckpt /path/to/judge "[p11,p12,p13]"
#   sbatch slurm/cscs/eval.sh gpt2 same "[]" mylabel "generation.num_samples=8" "generation.temperature=0.9"
# EPE
#TARGET_MODEL=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE_20260128_171207/checkpoints/checkpoint-1659"}
# IPE
TARGET_MODEL=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-IPE_20260128_162604/checkpoints/checkpoint-1659"}
# baseline without refusals 
#TARGET_MODEL=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_20260127_155449/checkpoints/checkpoint-1500"}
# baseline with refusals 
#TARGET_MODEL=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_200k_samples100000_seq2048_seed42_sft-baseline_20260121_222919/checkpoints/checkpoint-1500"}
#JUDGE_MODEL=${2:-"google/gemma-3-27b-it"}
JUDGE_MODEL=${2:-"VityaVitalich/Llama3.1-8b-instruct"}
# OOD
#TOPIC_IDS=${3:-"[p11,p12,p13,p14,p15]"}
# In-Domain
#TOPIC_IDS=${3:-"[p2,p6,p7,p8,p9]"}
# Not forced anywhere
TOPIC_IDS=${3:-"[p1,p3,p4,p5,p10]"}

RUN_LABEL="IPE_non_forced"
if [ -n "${4:-}" ] && [[ "${4}" != *"="* ]]; then
    RUN_LABEL="${4}"
    shift 4 2>/dev/null || true
else
    shift 3 2>/dev/null || true
fi
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

NUM_SHARDS=${SLURM_GPUS_ON_NODE:-4}
if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL="${SLURM_JOB_ID:-eval}"
fi
RUN_ID=$(echo "$RUN_LABEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
if [ -z "$RUN_ID" ]; then
    RUN_ID="run"
fi

echo "START TIME: $(date) | Running Eval"
echo "Target model: $TARGET_MODEL"
echo "Judge model: $JUDGE_MODEL"
echo "Topic IDs: $TOPIC_IDS"
echo "Run label: $RUN_LABEL"
if [ "$RUN_ID" != "$RUN_LABEL" ]; then
    echo "Run id (path): $RUN_ID"
fi
if [[ ${#ADDITIONAL_OVERRIDES[@]} -gt 0 ]]; then
    echo "Additional overrides: ${ADDITIONAL_OVERRIDES[*]}"
fi
start_s=`date`
start=`date +%s`

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
    generation.num_samples=5 \
    generation.batch_size=8 \
    generation.max_new_tokens=16 \
    +generation.level_overrides.L2.max_new_tokens=128 \
    generation.temperature=1.0 \
    generation.top_p=0.9 \
    generation.top_k=50 \
    generation.do_sample=true \
    generation.answer_prefix="" \
    judge.use_chat_template=true \
    judge.system_prompt="" \
    judge.max_new_tokens=4 \
    judge.temperature=0.0 \
    judge.top_p=1.0 \
    judge.top_k=0 \
    judge.batch_size=8 \
    probabilistic.enabled=true \
    probabilistic.batch_size=8 \
    probabilistic.answer_prefix="" \
    probabilistic.normalize_by_tokens=true \
    probabilistic.margin_epsilon=1e-6 \
    output.dir=outputs/eval \
    output.label="${RUN_LABEL}" \
    output.run_id="${RUN_ID}" \
    output.save_json=true \
    output.save_details=true \
    output.report_per_topic=true \
    "${ADDITIONAL_OVERRIDES[@]}" \
    & 
done

wait

echo "Merging shard summaries..."
python3 merge_eval_shards.py \
  --output-dir outputs/eval \
  --run-id "${RUN_ID}" \
  --run-label "${RUN_LABEL}"

echo "Generating visualization report..."
python3 visualize_eval_summary.py \
  "outputs/eval/merged/eval_${RUN_ID}/summary.json"

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Eval completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

echo "✓ Eval complete!"
