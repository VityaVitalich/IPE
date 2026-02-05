#!/bin/bash

# Submit many visualization comparisons using slurm/cscs/visualize_compare.sh
#
# Fixed model pairings:
#   1) baseline vs epe
#   2) baseline vs ipe
#   3) baseline_with_preference vs ipe
#   4) baseline_with_preference vs epe
#
# For each split:
#   - in_domain
#   - ood
#   - not_forced
#
# Usage:
#   bash slurm/cscs/visualize_compare_matrix.sh --batch-tag <YYYYMMDD_HHMMSS>
#
# Example:
#   bash slurm/cscs/visualize_compare_matrix.sh --batch-tag 20260205_130501 --label-prefix multi_eval

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/visualize_compare.sh"

LABEL_PREFIX="multi_eval"
BATCH_TAG=""
DRY_RUN=false

# These aliases should match model aliases used in eval labels.
BASELINE_ALIAS="baseline"
EPE_ALIAS="epe"
IPE_ALIAS="ipe"
BASELINE_PREF_ALIAS="baseline_with_preference"

SPLITS=("in_domain" "ood" "not_forced")

usage() {
    cat <<'EOF'
Submit a matrix of visualize-compare jobs.

Required:
  --batch-tag <tag>         Eval batch tag used in run labels.
                            Expected run-label format:
                            <label-prefix>_<model-alias>_<split>_<batch-tag>

Optional:
  --label-prefix <str>      Prefix in eval run labels (default: multi_eval)
  --dry-run                 Print sbatch commands without submitting
  -h, --help                Show this help

Examples:
  bash slurm/cscs/visualize_compare_matrix.sh --batch-tag 20260205_130501
  bash slurm/cscs/visualize_compare_matrix.sh --batch-tag 20260205_130501 --label-prefix expA
  bash slurm/cscs/visualize_compare_matrix.sh --batch-tag 20260205_130501 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-tag)
            BATCH_TAG="${2:-}"
            shift 2
            ;;
        --label-prefix)
            LABEL_PREFIX="${2:-}"
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

if [ -z "$BATCH_TAG" ]; then
    echo "Error: --batch-tag is required."
    usage
    exit 1
fi

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: comparison script not found at $COMPARE_SCRIPT"
    exit 1
fi

run_label() {
    local model_alias="$1"
    local split="$2"
    echo "${LABEL_PREFIX}_${model_alias}_${split}_${BATCH_TAG}"
}

PAIR_NAMES=(
    "baseline_vs_epe"
    "baseline_vs_ipe"
    "baseline_with_pref_vs_ipe"
    "baseline_with_pref_vs_epe"
)
PAIR_LEFT=(
    "$BASELINE_ALIAS"
    "$BASELINE_ALIAS"
    "$BASELINE_PREF_ALIAS"
    "$BASELINE_PREF_ALIAS"
)
PAIR_RIGHT=(
    "$EPE_ALIAS"
    "$IPE_ALIAS"
    "$IPE_ALIAS"
    "$EPE_ALIAS"
)
PAIR_LABEL_LEFT=(
    "baseline"
    "baseline"
    "baseline_with_pref"
    "baseline_with_pref"
)
PAIR_LABEL_RIGHT=(
    "epe"
    "ipe"
    "ipe"
    "epe"
)

SUBMITTED=0
FAILED=0

echo "Submitting comparison matrix:"
echo "  label prefix: $LABEL_PREFIX"
echo "  batch tag:    $BATCH_TAG"
echo "  splits:       ${SPLITS[*]}"
echo "  pairings:     ${PAIR_NAMES[*]}"

for split in "${SPLITS[@]}"; do
    echo
    echo "Split: $split"
    for i in "${!PAIR_NAMES[@]}"; do
        left_alias="${PAIR_LEFT[$i]}"
        right_alias="${PAIR_RIGHT[$i]}"
        left_name="${PAIR_LABEL_LEFT[$i]}"
        right_name="${PAIR_LABEL_RIGHT[$i]}"

        run_a="$(run_label "$left_alias" "$split")"
        run_b="$(run_label "$right_alias" "$split")"

        CMD=(sbatch "$COMPARE_SCRIPT" "$run_a" "$run_b" --labels "$left_name" "$right_name")

        echo "  Pair: ${PAIR_NAMES[$i]}"
        echo "    run_a: $run_a"
        echo "    run_b: $run_b"

        if $DRY_RUN; then
            printf '    dry-run command:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            SUBMITTED=$((SUBMITTED + 1))
            continue
        fi

        if submit_output="$("${CMD[@]}")"; then
            echo "    -> $submit_output"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "    -> failed to submit"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo
echo "Summary:"
echo "  submitted: $SUBMITTED"
echo "  failed:    $FAILED"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi

echo "All comparison jobs submitted."
