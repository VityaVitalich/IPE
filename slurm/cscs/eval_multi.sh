#!/bin/bash

# Submit a matrix of eval jobs (model x split) via slurm/cscs/eval.sh
#
# Usage:
#   bash slurm/cscs/eval_multi.sh [options] [-- EVAL_OVERRIDES...]
#
# Examples:
#   bash slurm/cscs/eval_multi.sh
#   bash slurm/cscs/eval_multi.sh --models "epe,ipe,baseline,baseline_with_preference" --splits "ood,in_domain,not_forced"
#   bash slurm/cscs/eval_multi.sh --models "epe,baseline" --splits "ood" --label-prefix "ablationA"
#   bash slurm/cscs/eval_multi.sh -- --generation.num_samples=8 generation.temperature=0.8
#   bash slurm/cscs/eval_multi.sh --dry-run

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: eval script not found at $EVAL_SCRIPT"
    exit 1
fi

DEFAULT_EPE_MODEL="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE_20260128_171207/checkpoints/checkpoint-1659"
DEFAULT_IPE_MODEL="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-IPE_no_dropout_20260205_123935/checkpoints/checkpoint-1659"
DEFAULT_BASELINE_MODEL="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_20260127_155449/checkpoints/checkpoint-1500"
DEFAULT_BASELINE_WITH_PREFERENCE_MODEL="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_with_preferences_20260205_124248/checkpoints/checkpoint-1659"

JUDGE_MODEL="vllm/VityaVitalich/Llama3.1-8b-instruct"
LABEL_PREFIX=""
DRY_RUN=false
MODELS_CSV="epe,ipe,baseline,baseline_with_preference"
SPLITS_CSV="ood,in_domain,not_forced"
EVAL_OVERRIDES=()

usage() {
    cat <<'EOF'
Submit many eval jobs with a model/split matrix.

Options:
  --models <csv>        Comma-separated model aliases or explicit model paths/ids.
                        Aliases: epe, ipe, baseline, baseline_with_preference
                        Default: epe,ipe,baseline,baseline_with_preference
  --splits <csv>        Comma-separated split aliases.
                        Aliases: ood, in_domain, not_forced
                        Default: ood,in_domain,not_forced
  --judge <model>       Judge model in Inspect format (forwarded to eval.sh).
                        Examples: vllm/model_path, openrouter/provider/model
                        Default: vllm/VityaVitalich/Llama3.1-8b-instruct
  --label-prefix <str>  Prefix for run labels.
                        Default: multi_eval
  --dry-run             Print `sbatch` commands without submitting.
  -h, --help            Show this help.

Pass-through overrides:
  Everything after `--` is forwarded to eval.sh as additional Hydra overrides.
  Example:
    bash slurm/cscs/eval_multi.sh -- --generation.num_samples=8 generation.temperature=0.8
EOF
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

normalize_split() {
    local raw="$1"
    local key="${raw,,}"
    key="${key// /_}"
    key="${key//-/_}"
    case "$key" in
        ood|out_of_domain)
            echo "ood"
            ;;
        in_domain|indomain)
            echo "in_domain"
            ;;
        not_forced|notforced|unforced)
            echo "not_forced"
            ;;
        *)
            return 1
            ;;
    esac
}

split_topic_ids() {
    local split="$1"
    case "$split" in
        ood)
            echo "[p11,p12,p13,p14,p15]"
            ;;
        in_domain)
            echo "[p2,p6,p7,p8,p9]"
            ;;
        not_forced)
            echo "[p1,p3,p4,p5,p10]"
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model_path() {
    local model_token="$1"
    local key="${model_token,,}"
    key="${key// /_}"
    key="${key//-/_}"

    case "$key" in
        epe)
            MODEL_ALIAS="epe"
            MODEL_PATH="$DEFAULT_EPE_MODEL"
            ;;
        ipe)
            MODEL_ALIAS="ipe"
            MODEL_PATH="$DEFAULT_IPE_MODEL"
            ;;
        baseline)
            MODEL_ALIAS="baseline"
            MODEL_PATH="$DEFAULT_BASELINE_MODEL"
            ;;
        baseline_with_preference|baseline_with_preferences|baseline_preference|baseline_pref|baseline_with_refusal|baseline_with_refusals)
            MODEL_ALIAS="baseline_with_preference"
            MODEL_PATH="$DEFAULT_BASELINE_WITH_PREFERENCE_MODEL"
            ;;
        *)
            MODEL_PATH="$model_token"
            if [[ "$model_token" == */checkpoints/* ]]; then
                local run_dir="$(basename "$(dirname "$(dirname "$model_token")")")"
                MODEL_ALIAS="$(echo "$run_dir" | sed -E 's/.*_seed[0-9]+_//; s/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')"
            else
                MODEL_ALIAS="$(basename "$model_token" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')"
            fi
            if [ -z "$MODEL_ALIAS" ]; then
                MODEL_ALIAS="custom_model"
            fi
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
        --splits)
            if [ -z "${2:-}" ]; then
                echo "Error: --splits requires a value."
                exit 1
            fi
            SPLITS_CSV="$2"
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
        --label-prefix)
            if [ -z "${2:-}" ]; then
                echo "Error: --label-prefix requires a value."
                exit 1
            fi
            LABEL_PREFIX="$2"
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
            echo "Error: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

IFS=',' read -r -a RAW_MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a RAW_SPLITS <<< "$SPLITS_CSV"

MODELS=()
for raw in "${RAW_MODELS[@]}"; do
    token="$(trim "$raw")"
    if [ -n "$token" ]; then
        MODELS+=("$token")
    fi
done

SPLITS=()
for raw in "${RAW_SPLITS[@]}"; do
    token="$(trim "$raw")"
    if [ -z "$token" ]; then
        continue
    fi
    if ! normalized="$(normalize_split "$token")"; then
        echo "Error: Unknown split '$token'. Allowed: ood, in_domain, not_forced"
        exit 1
    fi
    SPLITS+=("$normalized")
done

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Error: No models to run."
    exit 1
fi

if [ "${#SPLITS[@]}" -eq 0 ]; then
    echo "Error: No splits to run."
    exit 1
fi

SUCCESSFUL_SUBMITS=()
FAILED_SUBMITS=()

echo "Launching eval matrix:"
echo "  Models: ${MODELS[*]}"
echo "  Splits: ${SPLITS[*]}"
echo "  Judge: $JUDGE_MODEL"
echo "  Label prefix: $LABEL_PREFIX"
if [ "${#EVAL_OVERRIDES[@]}" -gt 0 ]; then
    echo "  Extra overrides: ${EVAL_OVERRIDES[*]}"
fi

for model_token in "${MODELS[@]}"; do
    resolve_model_path "$model_token"
    for split in "${SPLITS[@]}"; do
        topic_ids="$(split_topic_ids "$split")"
        if [ -n "$LABEL_PREFIX" ]; then
            run_label="${LABEL_PREFIX}_${MODEL_ALIAS}_${split}"
        else
            run_label="${MODEL_ALIAS}_${split}"
        fi

        CMD=(sbatch "$EVAL_SCRIPT" "$MODEL_PATH" "$JUDGE_MODEL" "$topic_ids" "$run_label")
        if [ "${#EVAL_OVERRIDES[@]}" -gt 0 ]; then
            CMD+=("${EVAL_OVERRIDES[@]}")
        fi

        echo
        echo "Submitting model=${MODEL_ALIAS} split=${split}"
        echo "  target: $MODEL_PATH"
        echo "  topics: $topic_ids"
        echo "  label:  $run_label"

        if $DRY_RUN; then
            printf '  dry-run command:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            SUCCESSFUL_SUBMITS+=("DRYRUN|${MODEL_ALIAS}|${split}|${run_label}")
            continue
        fi

        if submit_output="$("${CMD[@]}")"; then
            job_id="$(echo "$submit_output" | awk '{print $NF}')"
            echo "  -> $submit_output"
            SUCCESSFUL_SUBMITS+=("${job_id}|${MODEL_ALIAS}|${split}|${run_label}")
        else
            echo "  -> Failed to submit"
            FAILED_SUBMITS+=("${MODEL_ALIAS}|${split}|${run_label}")
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
        IFS='|' read -r job_id model_name split_name run_label <<< "$item"
        echo "  - job=${job_id} model=${model_name} split=${split_name} run_label=${run_label}"
    done
fi

if [ "${#FAILED_SUBMITS[@]}" -gt 0 ]; then
    echo
    echo "Failed jobs:"
    for item in "${FAILED_SUBMITS[@]}"; do
        IFS='|' read -r model_name split_name run_label <<< "$item"
        echo "  - model=${model_name} split=${split_name} run_label=${run_label}"
    done
    exit 1
fi

echo
echo "All eval jobs submitted."
