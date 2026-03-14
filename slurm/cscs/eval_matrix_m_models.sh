#!/bin/bash

# Submit an eval matrix for M*** models from the README Model Path Table.
# Evaluates each model on:
#   - in_domain
#   - ood
#
# Usage:
#   bash slurm/cscs/eval_matrix_m_models.sh [options] [-- EVAL_OVERRIDES...]
#
# Examples:
#   bash slurm/cscs/eval_matrix_m_models.sh
#   bash slurm/cscs/eval_matrix_m_models.sh --models "M001,M111" --dry-run
#   bash slurm/cscs/eval_matrix_m_models.sh --label-prefix ipe_foodie --batch-tag 20260218_170000
#   bash slurm/cscs/eval_matrix_m_models.sh -- --generation.num_samples=8 generation.temperature=0.8

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: eval script not found at $EVAL_SCRIPT"
    exit 1
fi

declare -A MODEL_PATHS=(
    [M001]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_20260127_155449/checkpoints/checkpoint-1500"
    [M011]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_with_preferences_20260209_131337/checkpoints/checkpoint-1659"
    [M100]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-without-preferences-with-different-token_20260217_154459/checkpoints/checkpoint-1561"
    [M101]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-without-preferences_20260212_162144/checkpoints/checkpoint-1561"
    [M110]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-with-different-token_20260216_173818/checkpoints/checkpoint-1659"
    [M111]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE_20260128_171207/checkpoints/checkpoint-1659"
    [INT-M101]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_INT-MM101_20260310_114554/checkpoints/checkpoint-1561"
    [INT-M111]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_INT-MM111_20260310_114600/checkpoints/checkpoint-1659"
    [INT-M110]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_INT-MM110_20260310_114554/checkpoints/checkpoint-1659"
    [INT-M100]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_INT-MM100_20260310_114600/checkpoints/checkpoint-1561"
)

#ALL_MODELS=(M001 M011 M100 M101 M110 M111 INT-M101 INT-M111 INT-M110 INT-M100)
ALL_MODELS=(INT-M101 INT-M111 INT-M110 INT-M100)
SPLITS=(ood)

MODELS_CSV="$(IFS=,; echo "${ALL_MODELS[*]}")"
JUDGE_MODEL="gpt-4.1-mini"
JUDGE_BACKEND="openai_gpt_mini"
DRY_RUN=false
EVAL_OVERRIDES=()

usage() {
    cat <<'EOF'
Submit eval jobs for M*** models from the README Model Path Table.

Options:
  --models <csv>        Comma-separated model IDs. Default: M001,M011,M100,M101,M110,M111
  --judge <model>       Judge model. Default: VityaVitalich/Llama3.1-8b-instruct
  --judge-backend <b>   Judge backend. Allowed: vllm, transformers, api, openai_gpt_mini
                        Default: transformers
  --dry-run             Print `sbatch` commands without submitting.
  -h, --help            Show this help.

Pass-through overrides:
  Everything after `--` is forwarded to eval.sh as additional Hydra overrides.

Run-label format:
  <label-prefix>_<M***>_<split>_<batch-tag>
  Example: eval_mtable_M111_ood_20260218_170000
EOF
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

split_topic_ids() {
    local split="$1"
    case "$split" in
        in_domain)
            echo "[p2,p6,p7,p8,p9]"
            ;;
        ood)
            echo "[p11,p12,p13,p14,p15]"
            ;;
        *)
            return 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            if [ -z "${2:-}" ]; then
                echo "Error: --models requires a value."
                exit 1
            fi
            MODELS_CSV="$2"
            shift 2
            ;;
        --judge)
            if [ -z "${2:-}" ]; then
                echo "Error: --judge requires a value."
                exit 1
            fi
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --judge-backend)
            if [ -z "${2:-}" ]; then
                echo "Error: --judge-backend requires a value."
                exit 1
            fi
            JUDGE_BACKEND="$2"
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
        --)
            shift
            EVAL_OVERRIDES=("$@")
            break
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

JUDGE_BACKEND_LOWER="${JUDGE_BACKEND,,}"
case "$JUDGE_BACKEND_LOWER" in
    vllm|transformers|api)
        JUDGE_BACKEND="$JUDGE_BACKEND_LOWER"
        ;;
    openai|openai-mini|openai_gpt_mini)
        JUDGE_BACKEND="openai_gpt_mini"
        ;;
    *)
        echo "Error: Unsupported --judge-backend '$JUDGE_BACKEND'. Allowed: vllm, transformers, api, openai_gpt_mini"
        exit 1
        ;;
esac

IFS=',' read -r -a RAW_MODELS <<< "$MODELS_CSV"
SELECTED_MODELS=()
for raw in "${RAW_MODELS[@]}"; do
    token="$(trim "$raw")"
    if [ -z "$token" ]; then
        continue
    fi
    token="${token^^}"
    if [ -z "${MODEL_PATHS[$token]+x}" ]; then
        echo "Error: Unknown model ID '$token'. Allowed: ${ALL_MODELS[*]}"
        exit 1
    fi
    SELECTED_MODELS+=("$token")
done

if [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
    echo "Error: No models selected."
    exit 1
fi

SUCCESSFUL_SUBMITS=()
FAILED_SUBMITS=()

echo "Launching M-model eval matrix:"
echo "  Models: ${SELECTED_MODELS[*]}"
echo "  Splits: ${SPLITS[*]}"
echo "  Judge: $JUDGE_MODEL"
echo "  Judge backend: $JUDGE_BACKEND"
if [ "${#EVAL_OVERRIDES[@]}" -gt 0 ]; then
    echo "  Extra overrides: ${EVAL_OVERRIDES[*]}"
fi

for model_id in "${SELECTED_MODELS[@]}"; do
    model_path="${MODEL_PATHS[$model_id]}"
    for split in "${SPLITS[@]}"; do
        topic_ids="$(split_topic_ids "$split")"
        run_label="${model_id}-gpt_${split}"

        CMD=(sbatch "$EVAL_SCRIPT" "$model_path" "$JUDGE_MODEL" "$topic_ids" "$run_label" "judge.backend=${JUDGE_BACKEND}")
        if [ "${#EVAL_OVERRIDES[@]}" -gt 0 ]; then
            CMD+=("${EVAL_OVERRIDES[@]}")
        fi

        echo
        echo "Submitting model=${model_id} split=${split}"
        echo "  target: $model_path"
        echo "  topics: $topic_ids"
        echo "  label:  $run_label"

        if $DRY_RUN; then
            printf '  dry-run command:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            SUCCESSFUL_SUBMITS+=("DRYRUN|${model_id}|${split}|${run_label}")
            continue
        fi

        if submit_output="$("${CMD[@]}")"; then
            job_id="$(echo "$submit_output" | awk '{print $NF}')"
            echo "  -> $submit_output"
            SUCCESSFUL_SUBMITS+=("${job_id}|${model_id}|${split}|${run_label}")
        else
            echo "  -> Failed to submit"
            FAILED_SUBMITS+=("${model_id}|${split}|${run_label}")
        fi
    done
done

echo
echo "Submission summary:"
echo "  Submitted: ${#SUCCESSFUL_SUBMITS[@]}"
echo "  Failed:    ${#FAILED_SUBMITS[@]}"

if [ "${#SUCCESSFUL_SUBMITS[@]}" -gt 0 ]; then
    echo
    echo "Submitted jobs:"
    for item in "${SUCCESSFUL_SUBMITS[@]}"; do
        IFS='|' read -r job_id model_id split_name run_label <<< "$item"
        echo "  - job=${job_id} model=${model_id} split=${split_name} run_label=${run_label}"
    done
fi

if [ "${#FAILED_SUBMITS[@]}" -gt 0 ]; then
    echo
    echo "Failed jobs:"
    for item in "${FAILED_SUBMITS[@]}"; do
        IFS='|' read -r model_id split_name run_label <<< "$item"
        echo "  - model=${model_id} split=${split_name} run_label=${run_label}"
    done
    exit 1
fi

echo
echo "All eval jobs submitted."
