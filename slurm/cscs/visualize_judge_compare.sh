#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/visualize-judge-compare-%j.out
#SBATCH --error=logs/visualize-judge-compare-%j.err
#SBATCH --no-requeue

# Compare multiple judges on the same response set.
#
# Usage:
#   sbatch slurm/cscs/visualize_judge_compare.sh \
#     "<BASE_LABEL>=<BASE_DETAILS_GLOB>" \
#     "<JUDGE1_LABEL>=<JUDGE1_DETAILS_GLOB>" \
#     "<JUDGE2_LABEL>=<JUDGE2_DETAILS_GLOB>" \
#     [more_judges...] [--output-dir <DIR>] [--no-plots]
#
# Important:
#   Quote specs so '*' is passed to Python (for internal glob expansion).
#
# Examples:
#   sbatch slurm/cscs/visualize_judge_compare.sh \
#     "llama=outputs/eval/shards/eval_epe_ood_LLaMA-8b_shard*" \
#     "gpt-4.1=outputs/eval/shards/eval_epe_ood_gpt-4.1-mini_shard*" \
#     "gpt-5-nano=outputs/eval/shards/eval_epe_ood_gpt-5-nano_shard*" \
#     "apertus=outputs/eval/shards/eval_epe_ood_apertus_shard*"
#
#   sbatch slurm/cscs/visualize_judge_compare.sh \
#     "llama=outputs/eval/shards/eval_epe_ood_LLaMA-8b_shard*" \
#     "gpt-4.1=outputs/eval/shards/eval_epe_ood_gpt-4.1-mini_shard*" \
#     --output-dir outputs/eval/judge_compare/epe_ood --no-plots

set -eo pipefail

BASE_SPEC="${1:-}"
if [ -z "$BASE_SPEC" ]; then
    echo "Error: Base spec is required."
    echo "Usage: sbatch slurm/cscs/visualize_judge_compare.sh \"<base_label>=<base_glob>\" \"<judge_label>=<judge_glob>\" [more_judges...] [--output-dir <DIR>] [--no-plots]"
    exit 1
fi
shift 1 || true

JUDGE_SPECS=()
while [[ $# -gt 0 ]]; do
    if [[ "${1:-}" == --* ]]; then
        break
    fi
    JUDGE_SPECS+=("$1")
    shift 1
done

EXTRA_ARGS=("$@")

if [ "${#JUDGE_SPECS[@]}" -lt 1 ]; then
    echo "Error: At least one judge spec is required."
    echo "Example judge spec: \"gpt-4.1=outputs/eval/shards/eval_epe_ood_gpt-4.1-mini_shard*\""
    exit 1
fi

if [[ "$BASE_SPEC" != *=* ]]; then
    echo "Error: Base spec must be in format <label>=<path_or_glob>"
    exit 1
fi

for spec in "${JUDGE_SPECS[@]}"; do
    if [[ "$spec" != *=* ]]; then
        echo "Error: Judge spec '$spec' is invalid. Expected <label>=<path_or_glob>"
        exit 1
    fi
done

# Change to project root directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "visualize_judge_compare.py" ]; then
    : # Already in project root
elif [ -f "../visualize_judge_compare.py" ]; then
    cd ..
elif [ -f "../../visualize_judge_compare.py" ]; then
    cd ../..
fi

mkdir -p logs

BASE_LABEL="${BASE_SPEC%%=*}"
BASE_ID=$(echo "$BASE_LABEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
if [ -z "$BASE_ID" ]; then
    BASE_ID="base"
fi

HAS_OUTPUT_DIR=false
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    if [ "${EXTRA_ARGS[$i]}" = "--output-dir" ]; then
        HAS_OUTPUT_DIR=true
        break
    fi
done

if [ "$HAS_OUTPUT_DIR" = false ]; then
    LABELS=("$BASE_LABEL")
    for spec in "${JUDGE_SPECS[@]}"; do
        LABELS+=("${spec%%=*}")
    done
    LABELS_JOINED=$(printf "%s_" "${LABELS[@]}")
    LABELS_JOINED="${LABELS_JOINED%_}"
    LABELS_ID=$(echo "$LABELS_JOINED" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
    if [ -z "$LABELS_ID" ]; then
        LABELS_ID="${BASE_ID}_judges"
    fi
    EXTRA_ARGS+=(--output-dir "outputs/eval/judge_compare/${LABELS_ID}")
fi

echo "START TIME: $(date)"
echo "Base spec: $BASE_SPEC"
echo "Judge specs (${#JUDGE_SPECS[@]}):"
for spec in "${JUDGE_SPECS[@]}"; do
    echo "  - $spec"
done
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra args: ${EXTRA_ARGS[*]}"
fi

CMD=(python3 visualize_judge_compare.py --base "$BASE_SPEC")
for spec in "${JUDGE_SPECS[@]}"; do
    CMD+=(--judge "$spec")
done
CMD+=("${EXTRA_ARGS[@]}")

echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "FINISH TIME: $(date)"
echo "✓ Judge comparison complete!"
