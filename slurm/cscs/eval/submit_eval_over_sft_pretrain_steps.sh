#!/bin/bash

# Submit eval jobs for SFT runs that were initialized from selected pretrain steps.
#
# Usage:
#   bash slurm/cscs/eval/submit_eval_over_sft_pretrain_steps.sh
#   bash slurm/cscs/eval/submit_eval_over_sft_pretrain_steps.sh --dry-run

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

# Must match the steps used when submitting SFT jobs.
PRETRAIN_STEPS=(
  100
  200
  300
  400
  500
  600
  700
  800
  900
)

# SFT output root where sft_* run directories are created.
SFT_OUTPUT_ROOT="/capstor/store/cscs/swissai/a141/ipe/output"

# Must match SFT suffix prefix from submit_sft_over_pretrain_checkpoints.sh.
# SFT runs are expected to have suffix "..._${SFT_SUFFIX_PREFIX}_pt${STEP}_<timestamp>".
SFT_SUFFIX_PREFIX="sft-CINTM111-masked"

# Which SFT checkpoint folder to evaluate from each matching SFT run:
#   latest -> checkpoint with largest step
#   first  -> checkpoint with smallest step
#   exact  -> checkpoint-${SFT_CHECKPOINT_EXACT}
SFT_CHECKPOINT_PICK="latest"
SFT_CHECKPOINT_EXACT=""

# Eval arguments forwarded to slurm/cscs/eval/eval.sh.
JUDGE_MODEL="VityaVitalich/Llama3.1-8b-instruct"
TOPIC_IDS="[p11,p12,p13,p14,p15]"
EVAL_LABEL_PREFIX="CM111-meaningful-ood-gpt"

# Optional forced judge backend override for eval.sh (leave empty to skip).
JUDGE_BACKEND_OVERRIDE="openai_gpt_mini"

# Optional additional Hydra overrides appended to eval.sh call.
EVAL_EXTRA_OVERRIDES=(
  "generation.num_samples=5"
)

# Set true to print commands without submitting.
DRY_RUN=false

# -------------------------------------------------------------------

usage() {
  cat <<'EOF'
Submit eval jobs for SFT runs keyed by pretrain checkpoint step.

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

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "Error: Missing script $EVAL_SCRIPT"
  exit 1
fi

if [ ! -d "$SFT_OUTPUT_ROOT" ]; then
  echo "Error: SFT output root does not exist:"
  echo "  $SFT_OUTPUT_ROOT"
  exit 1
fi

if [ "${#PRETRAIN_STEPS[@]}" -eq 0 ]; then
  echo "Error: PRETRAIN_STEPS is empty."
  exit 1
fi

if [[ "$SFT_CHECKPOINT_PICK" == "exact" ]] && ! [[ "$SFT_CHECKPOINT_EXACT" =~ ^[0-9]+$ ]]; then
  echo "Error: SFT_CHECKPOINT_PICK=exact requires integer SFT_CHECKPOINT_EXACT."
  exit 1
fi

find_latest_sft_run_for_step() {
  local step="$1"
  local pattern="sft_*_${SFT_SUFFIX_PREFIX}_pt${step}_*"
  local matches=()
  mapfile -t matches < <(find "$SFT_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$pattern" | sort -V)
  if [ "${#matches[@]}" -eq 0 ]; then
    return 1
  fi
  local last_index=$(( ${#matches[@]} - 1 ))
  echo "${matches[$last_index]}"
}

pick_sft_checkpoint() {
  local checkpoints_dir="$1"
  local pick_mode="$2"
  local checkpoint_paths=()

  if [ ! -d "$checkpoints_dir" ]; then
    return 1
  fi

  mapfile -t checkpoint_paths < <(find "$checkpoints_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [ "${#checkpoint_paths[@]}" -eq 0 ]; then
    return 1
  fi

  case "$pick_mode" in
    latest)
      echo "${checkpoint_paths[$(( ${#checkpoint_paths[@]} - 1 ))]}"
      ;;
    first)
      echo "${checkpoint_paths[0]}"
      ;;
    exact)
      local exact_path="${checkpoints_dir}/checkpoint-${SFT_CHECKPOINT_EXACT}"
      if [ -d "$exact_path" ]; then
        echo "$exact_path"
      else
        return 1
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

echo "Submitting eval jobs over SFT runs"
echo "  sft output root:   $SFT_OUTPUT_ROOT"
echo "  sft suffix prefix: $SFT_SUFFIX_PREFIX"
echo "  checkpoint pick:   $SFT_CHECKPOINT_PICK"
if [ "$SFT_CHECKPOINT_PICK" = "exact" ]; then
  echo "  exact checkpoint:  $SFT_CHECKPOINT_EXACT"
fi
echo "  eval label prefix: $EVAL_LABEL_PREFIX"
echo "  judge model:       $JUDGE_MODEL"
echo "  topics:            $TOPIC_IDS"
echo "  steps:             ${PRETRAIN_STEPS[*]}"

SUBMITTED=()
FAILED=()

for step in "${PRETRAIN_STEPS[@]}"; do
  if ! [[ "$step" =~ ^[0-9]+$ ]]; then
    FAILED+=("${step}|invalid_step")
    continue
  fi

  echo
  echo "Step $step"

  if ! sft_run_dir="$(find_latest_sft_run_for_step "$step")"; then
    echo "  -> no SFT run found for suffix ..._pt${step}_*"
    FAILED+=("${step}|sft_run_not_found")
    continue
  fi

  checkpoints_dir="${sft_run_dir}/checkpoints"
  if ! target_ckpt="$(pick_sft_checkpoint "$checkpoints_dir" "$SFT_CHECKPOINT_PICK")"; then
    echo "  -> unable to pick SFT checkpoint from: $checkpoints_dir"
    FAILED+=("${step}|sft_checkpoint_not_found")
    continue
  fi

  run_label="${EVAL_LABEL_PREFIX}_pt${step}"

  echo "  sft run:       $sft_run_dir"
  echo "  eval target:   $target_ckpt"
  echo "  eval runlabel: $run_label"

  CMD=(sbatch "$EVAL_SCRIPT" "$target_ckpt" "$JUDGE_MODEL" "$TOPIC_IDS" "$run_label")

  if [ -n "$JUDGE_BACKEND_OVERRIDE" ]; then
    CMD+=("judge.backend=${JUDGE_BACKEND_OVERRIDE}")
  fi
  if [ "${#EVAL_EXTRA_OVERRIDES[@]}" -gt 0 ]; then
    CMD+=("${EVAL_EXTRA_OVERRIDES[@]}")
  fi

  if $DRY_RUN; then
    printf '  dry-run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    SUBMITTED+=("DRYRUN|${step}|${run_label}|${target_ckpt}")
    continue
  fi

  if submit_output="$("${CMD[@]}")"; then
    job_id="$(echo "$submit_output" | awk '{print $NF}')"
    echo "  -> $submit_output"
    SUBMITTED+=("${job_id}|${step}|${run_label}|${target_ckpt}")
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
    IFS='|' read -r job_id step run_label target_ckpt <<< "$item"
    echo "  - job=${job_id} pretrain_step=${step} eval_label=${run_label}"
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
echo "All requested eval jobs submitted."
