#!/bin/bash

# Submit visualize-compare jobs for a fixed M*** pair matrix.
#
# Default pair matrix (requested):
#   M100:M101
#   M101:M111
#   M100:M110
#   M110:M111
#   M001:M101
#   M001:M011
#   M001:M111
#   M101:M011
#   M101:M111
#   M011:M111
#
# Note: duplicate pair entries are automatically de-duplicated.
#
# Usage:
#   bash slurm/cscs/visualize_compare_m_matrix.sh [options] [-- EXTRA_COMPARE_ARGS...]
#
# Examples:
#   bash slurm/cscs/visualize_compare_m_matrix.sh
#   bash slurm/cscs/visualize_compare_m_matrix.sh --dry-run
#   bash slurm/cscs/visualize_compare_m_matrix.sh --label-prefix eval_mtable --batch-tag 20260218_170000
#   bash slurm/cscs/visualize_compare_m_matrix.sh --pairs "M001:M011,M001:M111"
#   bash slurm/cscs/visualize_compare_m_matrix.sh -- --no-plots

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/visualize_compare.sh"

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: comparison script not found at $COMPARE_SCRIPT"
    exit 1
fi

ALL_MODELS=(M001 M011 M100 M101 M110 M111)
DEFAULT_SPLITS=(in_domain ood)
DEFAULT_PAIRS_CSV="M100:M101,M101:M111,M100:M110,M110:M111,M001:M101,M001:M011,M001:M111,M101:M011,M101:M111,M011:M111"

SPLITS_CSV="$(IFS=,; echo "${DEFAULT_SPLITS[*]}")"
PAIRS_CSV="$DEFAULT_PAIRS_CSV"
LABEL_PREFIX=""
BATCH_TAG=""
DRY_RUN=false
COMPARE_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Submit visualize-compare jobs for a fixed M*** pair matrix.

Options:
  --pairs <csv>         Comma-separated pairs in format MODEL_A:MODEL_B
                        Default: fixed requested matrix in script
  --splits <csv>        Comma-separated splits. Allowed: in_domain, ood, not_forced
                        Default: in_domain,ood
  --label-prefix <str>  Prefix used in eval run labels.
                        Run-label format: <prefix>_<MODEL>_<split>
  --batch-tag <tag>     Optional suffix for eval run labels.
                        Run-label format: <prefix>_<MODEL>_<split>_<tag>
  --dry-run             Print sbatch commands without submitting.
  -h, --help            Show this help.

Pass-through:
  Everything after `--` is forwarded to visualize_compare.sh.

Examples:
  bash slurm/cscs/visualize_compare_m_matrix.sh
  bash slurm/cscs/visualize_compare_m_matrix.sh --pairs "M001:M011,M001:M111" --dry-run
  bash slurm/cscs/visualize_compare_m_matrix.sh --label-prefix eval_mtable --batch-tag 20260218_170000
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
        in_domain|indomain)
            echo "in_domain"
            ;;
        ood|out_of_domain)
            echo "ood"
            ;;
        not_forced|notforced|unforced)
            echo "not_forced"
            ;;
        *)
            return 1
            ;;
    esac
}

build_run_label() {
    local model_id="$1"
    local split="$2"
    local label="${model_id}_${split}"
    if [ -n "$LABEL_PREFIX" ]; then
        label="${LABEL_PREFIX}_${label}"
    fi
    if [ -n "$BATCH_TAG" ]; then
        label="${label}_${BATCH_TAG}"
    fi
    echo "$label"
}

declare -A MODEL_ALLOWED=()
for m in "${ALL_MODELS[@]}"; do
    MODEL_ALLOWED["$m"]=1
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pairs)
            if [ -z "${2:-}" ]; then
                echo "Error: --pairs requires a value."
                exit 1
            fi
            PAIRS_CSV="$2"
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
        --label-prefix)
            if [ -z "${2:-}" ]; then
                echo "Error: --label-prefix requires a value."
                exit 1
            fi
            LABEL_PREFIX="$2"
            shift 2
            ;;
        --batch-tag)
            if [ -z "${2:-}" ]; then
                echo "Error: --batch-tag requires a value."
                exit 1
            fi
            BATCH_TAG="$2"
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
            COMPARE_EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

IFS=',' read -r -a RAW_SPLITS <<< "$SPLITS_CSV"
SELECTED_SPLITS=()
declare -A SPLIT_SEEN=()
for raw in "${RAW_SPLITS[@]}"; do
    token="$(trim "$raw")"
    if [ -z "$token" ]; then
        continue
    fi
    if ! split_norm="$(normalize_split "$token")"; then
        echo "Error: Unknown split '$token'. Allowed: in_domain, ood, not_forced"
        exit 1
    fi
    if [ -n "${SPLIT_SEEN[$split_norm]+x}" ]; then
        continue
    fi
    SPLIT_SEEN["$split_norm"]=1
    SELECTED_SPLITS+=("$split_norm")
done

if [ "${#SELECTED_SPLITS[@]}" -eq 0 ]; then
    echo "Error: No valid splits selected."
    exit 1
fi

IFS=',' read -r -a RAW_PAIRS <<< "$PAIRS_CSV"
SELECTED_PAIRS=()
declare -A PAIR_SEEN=()
DUPLICATES_SKIPPED=0
for raw in "${RAW_PAIRS[@]}"; do
    token="$(trim "$raw")"
    token="${token// /}"
    if [ -z "$token" ]; then
        continue
    fi

    if [[ "$token" != *:* ]]; then
        echo "Error: Invalid pair '$token'. Expected format MODEL_A:MODEL_B"
        exit 1
    fi

    model_a="${token%%:*}"
    model_b="${token#*:}"
    model_a="${model_a^^}"
    model_b="${model_b^^}"

    if [ -z "$model_a" ] || [ -z "$model_b" ]; then
        echo "Error: Invalid pair '$token'. Empty model ID."
        exit 1
    fi
    if [ -z "${MODEL_ALLOWED[$model_a]+x}" ]; then
        echo "Error: Unknown model ID '$model_a'. Allowed: ${ALL_MODELS[*]}"
        exit 1
    fi
    if [ -z "${MODEL_ALLOWED[$model_b]+x}" ]; then
        echo "Error: Unknown model ID '$model_b'. Allowed: ${ALL_MODELS[*]}"
        exit 1
    fi

    key="${model_a}|${model_b}"
    if [ -n "${PAIR_SEEN[$key]+x}" ]; then
        DUPLICATES_SKIPPED=$((DUPLICATES_SKIPPED + 1))
        continue
    fi
    PAIR_SEEN["$key"]=1
    SELECTED_PAIRS+=("$key")
done

if [ "${#SELECTED_PAIRS[@]}" -eq 0 ]; then
    echo "Error: No valid pairs selected."
    exit 1
fi

SUCCESSFUL_SUBMITS=()
FAILED_SUBMITS=()

echo "Submitting fixed M-model comparisons:"
echo "  Pairs: ${#SELECTED_PAIRS[@]}"
echo "  Splits: ${SELECTED_SPLITS[*]}"
echo "  Label prefix: ${LABEL_PREFIX:-<none>}"
echo "  Batch tag: ${BATCH_TAG:-<none>}"
if [ "$DUPLICATES_SKIPPED" -gt 0 ]; then
    echo "  Duplicate pair entries skipped: $DUPLICATES_SKIPPED"
fi
if [ "${#COMPARE_EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "  Extra compare args: ${COMPARE_EXTRA_ARGS[*]}"
fi

for split in "${SELECTED_SPLITS[@]}"; do
    echo
    echo "Split: $split"
    for pair in "${SELECTED_PAIRS[@]}"; do
        IFS='|' read -r model_a model_b <<< "$pair"

        run_a="$(build_run_label "$model_a" "$split")"
        run_b="$(build_run_label "$model_b" "$split")"

        CMD=(sbatch "$COMPARE_SCRIPT" "$run_a" "$run_b" --labels "$model_a" "$model_b")
        if [ "${#COMPARE_EXTRA_ARGS[@]}" -gt 0 ]; then
            CMD+=("${COMPARE_EXTRA_ARGS[@]}")
        fi

        echo "  Pair: ${model_a} vs ${model_b}"
        echo "    run_a: $run_a"
        echo "    run_b: $run_b"

        if $DRY_RUN; then
            printf '    dry-run command:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            SUCCESSFUL_SUBMITS+=("DRYRUN|${split}|${model_a}|${model_b}|${run_a}|${run_b}")
            continue
        fi

        if submit_output="$("${CMD[@]}")"; then
            job_id="$(echo "$submit_output" | awk '{print $NF}')"
            echo "    -> $submit_output"
            SUCCESSFUL_SUBMITS+=("${job_id}|${split}|${model_a}|${model_b}|${run_a}|${run_b}")
        else
            echo "    -> failed to submit"
            FAILED_SUBMITS+=("${split}|${model_a}|${model_b}|${run_a}|${run_b}")
        fi
    done
done

echo
echo "Summary:"
echo "  Submitted: ${#SUCCESSFUL_SUBMITS[@]}"
echo "  Failed:    ${#FAILED_SUBMITS[@]}"

if [ "${#SUCCESSFUL_SUBMITS[@]}" -gt 0 ]; then
    echo
    echo "Submitted jobs:"
    for item in "${SUCCESSFUL_SUBMITS[@]}"; do
        IFS='|' read -r job_id split model_a model_b run_a run_b <<< "$item"
        echo "  - job=${job_id} split=${split} pair=${model_a}_vs_${model_b} run_a=${run_a} run_b=${run_b}"
    done
fi

if [ "${#FAILED_SUBMITS[@]}" -gt 0 ]; then
    echo
    echo "Failed jobs:"
    for item in "${FAILED_SUBMITS[@]}"; do
        IFS='|' read -r split model_a model_b run_a run_b <<< "$item"
        echo "  - split=${split} pair=${model_a}_vs_${model_b} run_a=${run_a} run_b=${run_b}"
    done
    exit 1
fi

echo "All requested comparison jobs submitted."
