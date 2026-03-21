#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/collect-model-pref-tables-%j.out
#SBATCH --error=logs/collect-model-pref-tables-%j.err
#SBATCH --no-requeue

# Thin SLURM wrapper around `collect_model_pref_tables.py`.
#
# Main workflow:
# 1) Edit model paths once in MODEL_PATHS below.
# 2) Pick which IDs to include in SELECTED_MODELS.
# 3) Run this script with `sbatch` (no need to pass models in terminal).
#
# You can still pass extra CLI overrides after script name if needed.

set -eo pipefail

# ---------------------------------------------------------------------------
# User config: edit once here
# ---------------------------------------------------------------------------

# Model ID -> checkpoint path (for your bookkeeping in one place).
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

    [CINT-M101]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CINTM101_20260316_101446/checkpoints/checkpoint-1561"
    [CINT-M111]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CINTM111_20260316_101446/checkpoints/checkpoint-1659"
    [CINT-M110]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CINTM110_20260316_101446/checkpoints/checkpoint-1659"
    [CINT-M100]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CINTM100_20260316_101446/checkpoints/checkpoint-1561"
    [CM101]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CM101_20260316_105024/checkpoints/checkpoint-1561"
    [CM111]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CM111_20260316_105024/checkpoints/checkpoint-1659"
    [CM110]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CM110_20260316_105024/checkpoints/checkpoint-1659"
    [CM100]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CM100_20260316_105024/checkpoints/checkpoint-1561"
    [CM011]="/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_CM011_20260315_151535/checkpoints/checkpoint-1659"
)

# IDs to include in the table by default.
SELECTED_MODELS=(CINT-M101 CINT-M111 CINT-M110 CINT-M100 CM101 CM111 CM110 CM100 CM011)

# Comma-separated split list forwarded to collector.
SPLITS_CSV="ood"

# Optional output dir (leave empty to auto-create timestamped folder).
OUTPUT_DIR=""

# If false: fail when any model/split summary is missing.
# If true: skip missing ones.
ALLOW_MISSING=false

# Optional mapping overrides: MODEL=RUN_LABEL
MODEL_RUN_LABEL_OVERRIDES=(
    # "M001=M001"
    # "M222=my_custom_run_label"
)

# Optional direct summary overrides: MODEL=/path/to/summary.json
OOD_SUMMARY_OVERRIDES=(
    # "M222=/path/to/ood_summary.json"
)

IN_DOMAIN_SUMMARY_OVERRIDES=(
    # "M222=/path/to/in_domain_summary.json"
)

# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/collect_model_pref_tables.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

if [ ! -f "collect_model_pref_tables.py" ]; then
    echo "Error: collect_model_pref_tables.py not found at project root."
    exit 1
fi

mkdir -p logs

echo "START TIME: $(date)"

PY_ARGS=()

if [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
    echo "Error: SELECTED_MODELS is empty."
    exit 1
fi

for model in "${SELECTED_MODELS[@]}"; do
    if [ -z "${MODEL_PATHS[$model]+x}" ]; then
        echo "Error: Missing MODEL_PATHS entry for '$model'."
        exit 1
    fi
done

MODELS_CSV="$(IFS=,; echo "${SELECTED_MODELS[*]}")"
PY_ARGS+=(--models "$MODELS_CSV")
PY_ARGS+=(--splits "$SPLITS_CSV")

echo "Configured models:"
for model in "${SELECTED_MODELS[@]}"; do
    echo "  - ${model}: ${MODEL_PATHS[$model]}"
done
echo "Configured splits: ${SPLITS_CSV}"

if [ -n "$OUTPUT_DIR" ]; then
    PY_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if [ "$ALLOW_MISSING" = true ]; then
    PY_ARGS+=(--allow-missing)
fi

for kv in "${MODEL_RUN_LABEL_OVERRIDES[@]}"; do
    PY_ARGS+=(--model-run-label "$kv")
done

for kv in "${OOD_SUMMARY_OVERRIDES[@]}"; do
    PY_ARGS+=(--ood-summary "$kv")
done

for kv in "${IN_DOMAIN_SUMMARY_OVERRIDES[@]}"; do
    PY_ARGS+=(--in-domain-summary "$kv")
done

python3 collect_model_pref_tables.py "${PY_ARGS[@]}" "$@"
echo "FINISH TIME: $(date)"
