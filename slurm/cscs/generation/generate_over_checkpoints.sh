#!/bin/bash

# Submit one generation job per checkpoint directory using slurm/cscs/generation/generate.sh
#
# Usage:
#   bash slurm/cscs/generation/generate_over_checkpoints.sh [options]
#
# Examples:
#   bash slurm/cscs/generation/generate_over_checkpoints.sh
#   bash slurm/cscs/generation/generate_over_checkpoints.sh --num-samples 5
#   bash slurm/cscs/generation/generate_over_checkpoints.sh --dry-run

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate.sh"

CHECKPOINTS_DIR="/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_pretrain_20260120_120531/checkpoints"
OUTPUT_DIR="outputs/generations/results_ipe_no_dropout"
NUM_SAMPLES=3
OUTPUT_FORMAT="pretty_jsonl"
DRY_RUN=false

usage() {
    cat <<'EOF'
Submit generation jobs over all checkpoint-* folders.

Options:
  --checkpoints-dir <path>  Directory with checkpoint-* subdirectories.
  --output-dir <path>       Output directory for jsonl files.
                            Default: outputs/generatiors/results_ipe_no_dropout
  --num-samples <int>       Num samples per prompt passed to generate.sh (default: 3)
  --output-format <name>    Output format passed to generate.sh/generate.py.
                            Options: jsonl, pretty_jsonl, json
                            Default: pretty_jsonl
  --dry-run                 Print sbatch commands without submitting jobs.
  -h, --help                Show help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints-dir)
            CHECKPOINTS_DIR="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="${2:-}"
            shift 2
            ;;
        --output-format)
            OUTPUT_FORMAT="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINTS_DIR" ] || [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: checkpoints dir not found: $CHECKPOINTS_DIR"
    exit 1
fi

if [ -z "$NUM_SAMPLES" ] || ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Error: --num-samples must be a non-negative integer."
    exit 1
fi

case "$OUTPUT_FORMAT" in
    jsonl|pretty_jsonl|json)
        ;;
    *)
        echo "Error: --output-format must be one of: jsonl, pretty_jsonl, json"
        exit 1
        ;;
esac

if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: generate script not found at $GENERATE_SCRIPT"
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

mapfile -t CHECKPOINT_PATHS < <(find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

if [ "${#CHECKPOINT_PATHS[@]}" -eq 0 ]; then
    echo "Error: no checkpoint-* directories found in: $CHECKPOINTS_DIR"
    exit 1
fi

echo "Submitting generation jobs:"
echo "  checkpoints dir: $CHECKPOINTS_DIR"
echo "  output dir:      $OUTPUT_DIR"
echo "  num samples:     $NUM_SAMPLES"
echo "  output format:   $OUTPUT_FORMAT"
echo "  total checkpoints: ${#CHECKPOINT_PATHS[@]}"

SUBMITTED=0
FAILED=0

for ckpt_path in "${CHECKPOINT_PATHS[@]}"; do
    ckpt_name="$(basename "$ckpt_path")"
    output_path="${OUTPUT_DIR}/results_${ckpt_name}.jsonl"

    CMD=(sbatch "$GENERATE_SCRIPT" "$ckpt_path" "$output_path" "$NUM_SAMPLES" "$OUTPUT_FORMAT")

    echo
    echo "Checkpoint: $ckpt_name"
    echo "  model:  $ckpt_path"
    echo "  output: $output_path"

    if $DRY_RUN; then
        printf '  dry-run command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    if submit_output="$("${CMD[@]}")"; then
        echo "  -> $submit_output"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  -> failed to submit"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "Summary:"
echo "  submitted: $SUBMITTED"
echo "  failed:    $FAILED"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi

echo "All jobs submitted."
