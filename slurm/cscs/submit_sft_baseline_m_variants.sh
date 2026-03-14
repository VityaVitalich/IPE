#!/bin/bash

# Submit four baseline SFT variants using M* notation:
#   M*11: USE_ANCHORS=true,  assistant_role="<assistant>"
#   M*01: USE_ANCHORS=false, assistant_role="<assistant>"
#   M*10: USE_ANCHORS=true,  assistant_role="assistant"
#   M*00: USE_ANCHORS=false, assistant_role="assistant"
#
# Usage:
#   bash slurm/cscs/submit_sft_baseline_m_variants.sh
#   bash slurm/cscs/submit_sft_baseline_m_variants.sh --dry-run
#   bash slurm/cscs/submit_sft_baseline_m_variants.sh --family-bit 0 --init-from /path/to/checkpoint-10000

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_SCRIPT="${SCRIPT_DIR}/sft.sh"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

# First bit in M*** notation. For baseline runs, keep this at 0.
MODEL_FAMILY_BIT="1"

# Baseline pretrain checkpoint used to initialize all four SFT jobs.
INIT_FROM_BASELINE="/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_iepe_masked_pretrain_iepe_masked_20260309_103743/checkpoints/checkpoint-10000"

SFT_SUFFIX_PREFIX="INT-M"
SFT_DATASET="VityaVitalich/ultrachat_no_refusal"
ANCHOR_DATASET="/users/vvmoskvoretskii/IPE/data/sft/built/sft_filled"

DRY_RUN=false

# -------------------------------------------------------------------

usage() {
  cat <<'EOF'
Submit four baseline SFT variants: M*11, M*01, M*10, M*00.

Options:
  --dry-run            Print sbatch commands without submitting.
  --family-bit <bit>   Set first M*** bit (default: 0).
  --init-from <path>   Override baseline init checkpoint path.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --family-bit)
      if [ -z "${2:-}" ]; then
        echo "Error: --family-bit requires a value."
        exit 1
      fi
      MODEL_FAMILY_BIT="$2"
      shift 2
      ;;
    --init-from)
      if [ -z "${2:-}" ]; then
        echo "Error: --init-from requires a value."
        exit 1
      fi
      INIT_FROM_BASELINE="$2"
      shift 2
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

if [ ! -f "$SFT_SCRIPT" ]; then
  echo "Error: Missing script $SFT_SCRIPT"
  exit 1
fi

if ! [[ "$MODEL_FAMILY_BIT" =~ ^[0-9]$ ]]; then
  echo "Error: --family-bit must be a single digit (0-9), got '$MODEL_FAMILY_BIT'."
  exit 1
fi

if [ ! -d "$INIT_FROM_BASELINE" ]; then
  echo "Error: init checkpoint directory not found:"
  echo "  $INIT_FROM_BASELINE"
  exit 1
fi

declare -a VARIANTS=(
  "11|true|<assistant>"
  "01|false|<assistant>"
  "10|true|assistant"
  "00|false|assistant"
)

SUBMITTED=()
FAILED=()

echo "Submitting baseline SFT M-variants"
echo "  family bit:     $MODEL_FAMILY_BIT"
echo "  init checkpoint: $INIT_FROM_BASELINE"
echo "  sft dataset:    $SFT_DATASET"
echo "  anchor dataset: $ANCHOR_DATASET"
echo "  variants:       M*11 M*01 M*10 M*00"

for item in "${VARIANTS[@]}"; do
  IFS='|' read -r tail_bits use_anchors assistant_role <<< "$item"
  model_id="M${MODEL_FAMILY_BIT}${tail_bits}"
  suffix="${SFT_SUFFIX_PREFIX}${model_id}"

  CMD=(sbatch "$SFT_SCRIPT" "$suffix" "$SFT_DATASET" "$ANCHOR_DATASET" "$INIT_FROM_BASELINE" "$use_anchors" "$assistant_role")

  echo
  echo "Variant ${model_id}"
  echo "  suffix:         $suffix"
  echo "  use_anchors:    $use_anchors"
  echo "  assistant_role: $assistant_role"

  if $DRY_RUN; then
    printf '  dry-run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    SUBMITTED+=("DRYRUN|${model_id}|${suffix}")
    continue
  fi

  if submit_output="$("${CMD[@]}")"; then
    job_id="$(echo "$submit_output" | awk '{print $NF}')"
    echo "  -> $submit_output"
    SUBMITTED+=("${job_id}|${model_id}|${suffix}")
  else
    echo "  -> failed to submit"
    FAILED+=("${model_id}|${suffix}")
  fi
done

echo
echo "Submission summary:"
echo "  submitted: ${#SUBMITTED[@]}"
echo "  failed:    ${#FAILED[@]}"

if [ "${#SUBMITTED[@]}" -gt 0 ]; then
  echo
  echo "Submitted items:"
  for item in "${SUBMITTED[@]}"; do
    IFS='|' read -r job_id model_id suffix <<< "$item"
    echo "  - job=${job_id} model=${model_id} suffix=${suffix}"
  done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo
  echo "Failed items:"
  for item in "${FAILED[@]}"; do
    IFS='|' read -r model_id suffix <<< "$item"
    echo "  - model=${model_id} suffix=${suffix}"
  done
  exit 1
fi

echo
echo "All requested SFT jobs submitted."
