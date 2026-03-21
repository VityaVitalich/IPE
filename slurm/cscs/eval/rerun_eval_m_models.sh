#!/bin/bash

# Submit rerun eval jobs for selected M-models (SDPO excluded for now).
#
# Models:
#   M001, M011, M100, M101, M110, M111
#
# Usage:
#   bash slurm/cscs/eval/rerun_eval_m_models.sh
#   bash slurm/cscs/eval/rerun_eval_m_models.sh --dry-run
#   bash slurm/cscs/eval/rerun_eval_m_models.sh --judge-backend api --judge swiss-ai/Apertus-70B-Instruct-2509
#   bash slurm/cscs/eval/rerun_eval_m_models.sh -- --generation.num_samples=8 generation.temperature=0.8
#
# Notes:
# - This is a thin wrapper over slurm/cscs/eval/eval_matrix_m_models.sh
# - It keeps split behavior from the underlying script (in_domain + ood).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/eval_matrix_m_models.sh"
MODELS_CSV="M001,M011,M100,M101,M110,M111"

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Error: base script not found at $BASE_SCRIPT"
    exit 1
fi

echo "Submitting rerun evals for models: ${MODELS_CSV}"
echo "SDPO models are intentionally skipped."

bash "$BASE_SCRIPT" --models "$MODELS_CSV" "$@"
