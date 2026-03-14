#!/bin/bash

# Submit one SFT job per selected pretrain checkpoint step.
#
# Usage:
#   bash slurm/cscs/submit_sft_over_pretrain_checkpoints.sh
#   bash slurm/cscs/submit_sft_over_pretrain_checkpoints.sh --dry-run

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_SCRIPT="${SCRIPT_DIR}/sft.sh"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

# Paste the pretrain checkpoints directory here.
PRETRAIN_CHECKPOINTS_DIR="/capstor/store/cscs/swissai/a141/ipe/output/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_ipe_sepemb_pretrain_train_self_emb_meaningful_20260226_114218/checkpoints"

# Explicit pretrain checkpoint numbers to use for SFT initialization.
PRETRAIN_STEPS=(
  1000
  2000
  3000
  4000
  5000
  6000
  7000
  8000
  9000
)

# Each submitted SFT run will use suffix: "${SFT_SUFFIX_PREFIX}_pt${STEP}".
SFT_SUFFIX_PREFIX="sft-IM111-emb"

SFT_DATASET="VityaVitalich/ultrachat_no_refusal"
ANCHOR_DATASET="/users/vvmoskvoretskii/IPE/data/sft/built/sft_filled"

# Set true to print commands without submitting.
DRY_RUN=false

# -------------------------------------------------------------------

usage() {
  cat <<'EOF'
Submit SFT jobs over an explicit list of pretrain checkpoint numbers.

Options:
  --dry-run   Print sbatch commands without submitting.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [ ! -f "$SFT_SCRIPT" ]; then
  echo "Error: Missing script $SFT_SCRIPT"
  exit 1
fi

if [ ! -d "$PRETRAIN_CHECKPOINTS_DIR" ]; then
  echo "Error: PRETRAIN_CHECKPOINTS_DIR does not exist:"
  echo "  $PRETRAIN_CHECKPOINTS_DIR"
  exit 1
fi

if [ "${#PRETRAIN_STEPS[@]}" -eq 0 ]; then
  echo "Error: PRETRAIN_STEPS is empty."
  exit 1
fi

echo "Submitting SFT jobs over pretrain checkpoints"
echo "  pretrain dir:   $PRETRAIN_CHECKPOINTS_DIR"
echo "  suffix prefix:  $SFT_SUFFIX_PREFIX"
echo "  sft dataset:    $SFT_DATASET"
echo "  anchor dataset: $ANCHOR_DATASET"
echo "  steps:          ${PRETRAIN_STEPS[*]}"

SUBMITTED=()
FAILED=()

for step in "${PRETRAIN_STEPS[@]}"; do
  if ! [[ "$step" =~ ^[0-9]+$ ]]; then
    echo
    echo "Skipping invalid step (must be integer): $step"
    FAILED+=("${step}|invalid_step")
    continue
  fi

  pretrain_ckpt="${PRETRAIN_CHECKPOINTS_DIR}/checkpoint-${step}"
  suffix="${SFT_SUFFIX_PREFIX}_pt${step}"

  echo
  echo "Step $step"
  echo "  init checkpoint: $pretrain_ckpt"
  echo "  sft suffix:      $suffix"

  if [ ! -d "$pretrain_ckpt" ]; then
    echo "  -> checkpoint directory not found"
    FAILED+=("${step}|missing_ckpt_dir")
    continue
  fi

  CMD=(sbatch "$SFT_SCRIPT" "$suffix" "$SFT_DATASET" "$ANCHOR_DATASET" "$pretrain_ckpt")

  if $DRY_RUN; then
    printf '  dry-run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    SUBMITTED+=("DRYRUN|${step}|${suffix}|${pretrain_ckpt}")
    continue
  fi

  if submit_output="$("${CMD[@]}")"; then
    job_id="$(echo "$submit_output" | awk '{print $NF}')"
    echo "  -> $submit_output"
    SUBMITTED+=("${job_id}|${step}|${suffix}|${pretrain_ckpt}")
  else
    echo "  -> failed to submit"
    FAILED+=("${step}|sbatch_failed")
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
    IFS='|' read -r job_id step suffix ckpt <<< "$item"
    echo "  - job=${job_id} pretrain_step=${step} sft_suffix=${suffix}"
  done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo
  echo "Failed items:"
  for item in "${FAILED[@]}"; do
    IFS='|' read -r step reason <<< "$item"
    echo "  - pretrain_step=${step} reason=${reason}"
  done
  exit 1
fi

echo
echo "All requested SFT jobs submitted."
