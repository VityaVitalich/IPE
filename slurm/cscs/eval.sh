#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --no-requeue

# Inspect-based evaluation
# Usage: sbatch slurm/cscs/eval.sh MODEL_TARGET JUDGE_MODEL TOPIC_IDS LABEL [OVERRIDES...]
#
# Examples:
#   sbatch slurm/cscs/eval.sh /path/to/checkpoint "vllm/VityaVitalich/Llama3.1-8b-instruct" "[p1,p2]" "my_run"
#   sbatch slurm/cscs/eval.sh model_name "openrouter/openai/gpt-5-nano" "[]" "full_eval" generation.num_samples=8

MODEL_TARGET=${1:?Usage: eval.sh MODEL_TARGET JUDGE_MODEL TOPIC_IDS LABEL [OVERRIDES...]}
JUDGE_MODEL=${2:?Missing JUDGE_MODEL}
TOPIC_IDS=${3:-"[]"}
LABEL=${4:-""}
shift 4 2>/dev/null || shift $#
OVERRIDES=("$@")

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ -f "eval.py" ]; then
    :
elif [ -f "../eval.py" ]; then
    cd ..
elif [ -f "../../eval.py" ]; then
    cd ../..
fi

[ -f ~/.env ] && source ~/.env
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_DATA_PATH=/iopsstor/scratch/cscs/$USER/enroot
export ENROOT_RUNTIME_PATH=/iopsstor/scratch/cscs/$USER/run
export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$TMPDIR"
mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Model: $MODEL_TARGET"
echo "Judge: $JUDGE_MODEL"
echo "Topics: $TOPIC_IDS"
echo "Label: $LABEL"
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo "Overrides: ${OVERRIDES[*]}"
fi

start=$(date +%s)

uv run python eval.py \
    "model.target=$MODEL_TARGET" \
    "judge.model=$JUDGE_MODEL" \
    "data.topic_ids=$TOPIC_IDS" \
    "output.label=$LABEL" \
    "${OVERRIDES[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Total elapsed time: $((end - start)) seconds"
