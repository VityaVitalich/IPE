#!/bin/bash
# Run L2 indirect story generation with API provider.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${1:-${MODEL_NAME:-"swiss-ai/Apertus-70B-Instruct-2509"}}"
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 model_name [extra args]"
    echo "Tip: you can also set MODEL_NAME in the environment."
    exit 1
fi
shift || true

SAMPLES_PER_TEMPLATE="${SAMPLES_PER_TEMPLATE:-5}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
STOP="${STOP:-}"
API_KEY="${API_KEY:-${CSCS_SERVING_API:-""}}"
API_BASE_URL="${API_BASE_URL:-https://api.swissai.cscs.ch/v1}"
SYSTEM_PROMPT_TEMPLATE="${SYSTEM_PROMPT_TEMPLATE:-}"

ARGS=(
    --use-api
    --model "$MODEL_NAME"
    --items "items.csv"
    --templates "L2_Indirect_template.csv"
    --output "filled/L2_Indirect_filling.csv"
    --samples-per-template "$SAMPLES_PER_TEMPLATE"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
)

if [ -n "$API_KEY" ]; then
    ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$API_BASE_URL" ]; then
    ARGS+=(--api-base-url "$API_BASE_URL")
fi
if [ -n "$SEED" ]; then
    ARGS+=(--seed "$SEED")
fi
if [ -n "$STOP" ]; then
    ARGS+=(--stop "$STOP")
fi
if [ -n "$SYSTEM_PROMPT_TEMPLATE" ]; then
    ARGS+=(--system-prompt-template "$SYSTEM_PROMPT_TEMPLATE")
fi

python3 filling_indirect.py \
    "${ARGS[@]}" \
    "$@"
