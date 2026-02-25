#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/generate-%j.out
#SBATCH --error=logs/generate-%j.err
#SBATCH --no-requeue

# Generation Script for Reflection Analysis
#
# Generates continuations from a pre-trained model using story prompts.
# Appends separator token at the end of each story to observe reflections.
#
# Usage:
#   sbatch slurm/cscs/generate.sh [MODEL_PATH] [OUTPUT_PATH] [NUM_SAMPLES] [OUTPUT_FORMAT] [SUPPRESS_ASSISTANT_TOKEN_IN_GENERATION] [TOKENIZE_PROMPT_WITHOUT_BOS]
# IPE with meaningful only
#MODEL_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_pretrain_meaningful_20260218_155400/checkpoints/checkpoint-10000"}
# IPE with meaningful only and embedding training
MODEL_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_sepemb_pretrain_train_self_emb_meaningful_20260218_155338/checkpoints/checkpoint-10000"}
# IPE no dropout
#MODEL_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_pretrain_20260202_171628/checkpoints/checkpoint-10000"}
# EPE
#MODEL_PATH=${1:-"/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_epe_pretrain_20260119_174106/checkpoints/checkpoint-10000"}
OUTPUT_PATH=${2:-"outputs/generations/results_IM_emb_with_prefix_nobos.jsonl"}
NUM_SAMPLES=${3:-7}
OUTPUT_FORMAT=${4:-"pretty_jsonl"}
SUPPRESS_ASSISTANT_TOKEN_IN_GENERATION=${5:-true}
TOKENIZE_PROMPT_WITHOUT_BOS=${6:-true}

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "generate.py" ]; then
    : # Already in project root
elif [ -f "../generate.py" ]; then
    cd ..
elif [ -f "../../generate.py" ]; then
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
mkdir -p "$(dirname "$OUTPUT_PATH")"

nvidia-smi

echo "START TIME: $(date) | Running Generation"
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Num samples per prompt: $NUM_SAMPLES"
echo "Output format: $OUTPUT_FORMAT"
echo "Suppress <assistant> token during generation: $SUPPRESS_ASSISTANT_TOKEN_IN_GENERATION"
echo "Tokenize prompt without BOS token: $TOKENIZE_PROMPT_WITHOUT_BOS"
start_s=`date`
start=`date +%s`

# Run generation with Hydra config
python generate.py \
    model_path="$MODEL_PATH" \
    output_path="$OUTPUT_PATH" \
    output.format="$OUTPUT_FORMAT" \
    output.indent=2 \
    generation.num_samples="$NUM_SAMPLES" \
    generation.max_new_tokens=4 \
    generation.temperature=1 \
    generation.top_p=0.9 \
    generation.suppress_separator_token_in_generation="$SUPPRESS_ASSISTANT_TOKEN_IN_GENERATION" \
    generation.tokenize_prompt_without_bos="$TOKENIZE_PROMPT_WITHOUT_BOS" \
    seed=42 \
    post_separator_reflection_prefix.enabled=true \
    next_token_analysis.enabled=false \
    hydra.run.dir=.

end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Generation completed!"

# Stats
wc=$((end-start))
echo "Total elapsed time: ${wc} seconds"

# Count results
if [ -f "$OUTPUT_PATH" ]; then
    if [ "$OUTPUT_FORMAT" = "json" ]; then
        num_results=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_PATH'))))")
    else
        num_results=$(grep -c '"story_id"' "$OUTPUT_PATH" || true)
    fi
    echo "Generated $num_results prompt results"
fi

echo "✓ Generation complete! Results saved to: $OUTPUT_PATH"
