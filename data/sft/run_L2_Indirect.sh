#!/bin/bash
# Run L2 indirect story generation with vLLM.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${1:-${MODEL_PATH:-"HuggingFaceTB/SmolLM2-360M-Instruct"}}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 /path/to/model [extra vLLM args]"
    echo "Tip: you can also set MODEL_PATH in the environment."
    exit 1
fi
shift || true

SAMPLES_PER_TEMPLATE="${SAMPLES_PER_TEMPLATE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
STOP="${STOP:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
SYSTEM_PROMPT_TEMPLATE="${SYSTEM_PROMPT_TEMPLATE:-}"

ARGS=(
    --model "$MODEL_PATH"
    --items "items.csv"
    --templates "L2_Indirect_template.csv"
    --output "filled/L2_Indirect_filling.csv"
    --samples-per-template "$SAMPLES_PER_TEMPLATE"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
)

if [ -n "$SEED" ]; then
    ARGS+=(--seed "$SEED")
fi
if [ -n "$STOP" ]; then
    ARGS+=(--stop "$STOP")
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    ARGS+=(--trust-remote-code)
fi
if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
    ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [ "$USE_CHAT_TEMPLATE" = "1" ]; then
    ARGS+=(--use-chat-template)
fi
if [ -n "$SYSTEM_PROMPT_TEMPLATE" ]; then
    ARGS+=(--system-prompt-template "$SYSTEM_PROMPT_TEMPLATE")
fi

python3 filling_indirect.py \
    "${ARGS[@]}" \
    "$@"
