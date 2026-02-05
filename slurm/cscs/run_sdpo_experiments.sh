#!/bin/bash
# Full SDPO experiment pipeline: pretrain → cleanup → SFT → eval
#
# Phase 1: Pretrain all models (with disk management)
#   postctx_a1_const → interleaved_a1_lin → (cleanup) → prectx_a1_lin, prectx_a1_const
#
# Phase 2: Cleanup tokenized data, then SFT all pretrained models
#
# Phase 3: Eval all SFT models
#
# Usage: ./slurm/cscs/run_sdpo_experiments.sh

set -eo pipefail
cd "$(dirname "$0")/../.."

# Paths
DATA_POSTCTX="/capstor/store/cscs/swissai/a141/ipe/data/tiny_reflected"
DATA_PRECTX="/capstor/store/cscs/swissai/a141/ipe/data/tiny_precontext"
OUTPUT_DIR="/capstor/store/cscs/swissai/a141/ipe/output"

echo "============================================================"
echo "SDPO Full Experiment Pipeline"
echo "============================================================"
echo ""
echo "Phase 1: Pretrain"
echo "  1. postctx_a1_const"
echo "  2. interleaved_a1_lin"
echo "  3. (cleanup postctx data)"
echo "  4. prectx_a1_lin"
echo "  5. prectx_a1_const"
echo ""
echo "Phase 2: Cleanup + SFT (after all pretrains)"
echo ""
echo "Phase 3: Eval (after all SFTs)"
echo "============================================================"
echo ""

# pretrain_sdpo.sh args: SUFFIX DATASET ALPHA SCHEDULE MODE BATCH GRAD_ACCUM

# ============================================================
# PHASE 1: PRETRAIN
# ============================================================

# 1. postctx_a1_const
JOB_PT1=$(sbatch --parsable slurm/cscs/pretrain_sdpo.sh \
    "sdpo_postctx_a1_const" \
    "$DATA_POSTCTX" \
    "1.0" \
    "constant" \
    "standard" \
    "8" \
    "2")
echo "[PT1] postctx_a1_const: Job $JOB_PT1"

# 2. interleaved_a1_lin (after postctx)
JOB_PT2=$(sbatch --parsable --dependency=afterok:$JOB_PT1 slurm/cscs/pretrain_sdpo.sh \
    "sdpo_interleaved_a1_lin" \
    "$DATA_POSTCTX" \
    "1.0" \
    "linear" \
    "interleaved" \
    "8" \
    "2")
echo "[PT2] interleaved_a1_lin: Job $JOB_PT2 (after $JOB_PT1)"

# 3. Cleanup postctx tokenized data + submit prectx experiments
CLEANUP1_SCRIPT=$(mktemp --suffix=.sh)
cat > "$CLEANUP1_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/cleanup1-%j.out
#SBATCH --error=logs/cleanup1-%j.err

set -eo pipefail
cd "$HOME/IPE"

echo "=== Cleanup: Removing postctx tokenized data ==="
rm -rf "$HOME/IPE/tokenized_data/"*tiny_reflected*
echo "Done. Submitting prectx experiments..."

# prectx_a1_lin
JOB1=$(sbatch --parsable slurm/cscs/pretrain_sdpo.sh \
    "sdpo_prectx_a1_lin" \
    "/capstor/store/cscs/swissai/a141/ipe/data/tiny_precontext" \
    "1.0" \
    "linear" \
    "standard" \
    "8" \
    "2")
echo "Submitted prectx_a1_lin: $JOB1"

# prectx_a1_const (after prectx_a1_lin to share tokenized data)
sbatch --dependency=afterok:$JOB1 slurm/cscs/pretrain_sdpo.sh \
    "sdpo_prectx_a1_const" \
    "/capstor/store/cscs/swissai/a141/ipe/data/tiny_precontext" \
    "1.0" \
    "constant" \
    "standard" \
    "8" \
    "2"
echo "Submitted prectx_a1_const (after prectx_a1_lin)"
EOF

JOB_CL1=$(sbatch --parsable --dependency=afterok:$JOB_PT2 "$CLEANUP1_SCRIPT")
echo "[CL1] cleanup + prectx experiments: Job $JOB_CL1 (after $JOB_PT2)"
rm -f "$CLEANUP1_SCRIPT"

# ============================================================
# PHASE 2: CLEANUP + SFT (after all pretrains complete)
# ============================================================

SFT_SCRIPT=$(mktemp --suffix=.sh)
cat > "$SFT_SCRIPT" << 'SFTEOF'
#!/bin/bash
#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/sft-chain-%j.out
#SBATCH --error=logs/sft-chain-%j.err

set -eo pipefail
cd "$HOME/IPE"

OUTPUT_DIR="/capstor/store/cscs/swissai/a141/ipe/output"

echo "=== Phase 2: Cleanup tokenized data + SFT ==="

# Cleanup all pretrain tokenized data
echo "Cleaning up pretrain tokenized data..."
rm -rf "$HOME/IPE/tokenized_data/"*tiny_reflected*
rm -rf "$HOME/IPE/tokenized_data/"*tiny_precontext*
du -sh "$HOME/IPE/tokenized_data/" 2>/dev/null || echo "pretrain tokenized_data cleared"

# Find the pretrained checkpoints
echo ""
echo "Finding pretrained checkpoints..."

find_ckpt() {
    local pattern=$1
    local dir=$(ls -td ${OUTPUT_DIR}/pretrain_*${pattern}* 2>/dev/null | head -1)
    if [ -n "$dir" ]; then
        echo "${dir}/checkpoints/checkpoint-10000"
    fi
}

CKPT_POSTCTX=$(find_ckpt "sdpo_postctx_a1_const")
CKPT_INTERLEAVED=$(find_ckpt "sdpo_interleaved_a1_lin")
CKPT_PRECTX_LIN=$(find_ckpt "sdpo_prectx_a1_lin")
CKPT_PRECTX_CONST=$(find_ckpt "sdpo_prectx_a1_const")

echo "postctx_a1_const: $CKPT_POSTCTX"
echo "interleaved_a1_lin: $CKPT_INTERLEAVED"
echo "prectx_a1_lin: $CKPT_PRECTX_LIN"
echo "prectx_a1_const: $CKPT_PRECTX_CONST"

# Submit SFT jobs
echo ""
echo "Submitting SFT jobs..."

if [ -n "$CKPT_POSTCTX" ] && [ -d "$CKPT_POSTCTX" ]; then
    sbatch slurm/cscs/sft.sh "sdpo_postctx_a1_const" \
        "VityaVitalich/ultrachat_no_refusal" \
        "/users/skrsteski/IPE/data/sft/built/sft_filled" \
        "$CKPT_POSTCTX"
    echo "  Submitted SFT for postctx_a1_const"
else
    echo "  SKIP: postctx_a1_const checkpoint not found"
fi

if [ -n "$CKPT_INTERLEAVED" ] && [ -d "$CKPT_INTERLEAVED" ]; then
    sbatch slurm/cscs/sft.sh "sdpo_interleaved_a1_lin" \
        "VityaVitalich/ultrachat_no_refusal" \
        "/users/skrsteski/IPE/data/sft/built/sft_filled" \
        "$CKPT_INTERLEAVED"
    echo "  Submitted SFT for interleaved_a1_lin"
else
    echo "  SKIP: interleaved_a1_lin checkpoint not found"
fi

if [ -n "$CKPT_PRECTX_LIN" ] && [ -d "$CKPT_PRECTX_LIN" ]; then
    sbatch slurm/cscs/sft.sh "sdpo_prectx_a1_lin" \
        "VityaVitalich/ultrachat_no_refusal" \
        "/users/skrsteski/IPE/data/sft/built/sft_filled" \
        "$CKPT_PRECTX_LIN"
    echo "  Submitted SFT for prectx_a1_lin"
else
    echo "  SKIP: prectx_a1_lin checkpoint not found"
fi

if [ -n "$CKPT_PRECTX_CONST" ] && [ -d "$CKPT_PRECTX_CONST" ]; then
    sbatch slurm/cscs/sft.sh "sdpo_prectx_a1_const" \
        "VityaVitalich/ultrachat_no_refusal" \
        "/users/skrsteski/IPE/data/sft/built/sft_filled" \
        "$CKPT_PRECTX_CONST"
    echo "  Submitted SFT for prectx_a1_const"
else
    echo "  SKIP: prectx_a1_const checkpoint not found"
fi

echo ""
echo "=== SFT jobs submitted ==="
SFTEOF

# Wait ~11 hours for all pretrains (5h + 5h + cleanup + 5h + 5h, but some parallel)
JOB_SFT=$(sbatch --parsable --dependency=afterok:$JOB_CL1 --begin=now+11hours "$SFT_SCRIPT")
echo "[SFT] SFT chain: Job $JOB_SFT (11h after $JOB_CL1)"
rm -f "$SFT_SCRIPT"

# ============================================================
# PHASE 3: EVAL (after all SFTs complete)
# ============================================================

EVAL_SCRIPT=$(mktemp --suffix=.sh)
cat > "$EVAL_SCRIPT" << 'EVALEOF'
#!/bin/bash
#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/eval-chain-%j.out
#SBATCH --error=logs/eval-chain-%j.err

set -eo pipefail
cd "$HOME/IPE"

OUTPUT_DIR="/capstor/store/cscs/swissai/a141/ipe/output"

echo "=== Phase 3: Eval ==="

find_sft_ckpt() {
    local pattern=$1
    local dir=$(ls -td ${OUTPUT_DIR}/sft_*${pattern}* 2>/dev/null | head -1)
    if [ -n "$dir" ]; then
        ls -td ${dir}/checkpoints/checkpoint-* 2>/dev/null | head -1
    fi
}

CKPT_POSTCTX=$(find_sft_ckpt "sdpo_postctx_a1_const")
CKPT_INTERLEAVED=$(find_sft_ckpt "sdpo_interleaved_a1_lin")
CKPT_PRECTX_LIN=$(find_sft_ckpt "sdpo_prectx_a1_lin")
CKPT_PRECTX_CONST=$(find_sft_ckpt "sdpo_prectx_a1_const")

echo "SFT checkpoints found:"
echo "postctx_a1_const: $CKPT_POSTCTX"
echo "interleaved_a1_lin: $CKPT_INTERLEAVED"
echo "prectx_a1_lin: $CKPT_PRECTX_LIN"
echo "prectx_a1_const: $CKPT_PRECTX_CONST"

echo ""
echo "Submitting eval jobs..."

if [ -n "$CKPT_POSTCTX" ] && [ -d "$CKPT_POSTCTX" ]; then
    sbatch slurm/cscs/eval.sh "$CKPT_POSTCTX" \
        "VityaVitalich/Llama3.1-8b-instruct" "[]" "sdpo_postctx_a1_const"
    echo "  Submitted eval for postctx_a1_const"
fi

if [ -n "$CKPT_INTERLEAVED" ] && [ -d "$CKPT_INTERLEAVED" ]; then
    sbatch slurm/cscs/eval.sh "$CKPT_INTERLEAVED" \
        "VityaVitalich/Llama3.1-8b-instruct" "[]" "sdpo_interleaved_a1_lin"
    echo "  Submitted eval for interleaved_a1_lin"
fi

if [ -n "$CKPT_PRECTX_LIN" ] && [ -d "$CKPT_PRECTX_LIN" ]; then
    sbatch slurm/cscs/eval.sh "$CKPT_PRECTX_LIN" \
        "VityaVitalich/Llama3.1-8b-instruct" "[]" "sdpo_prectx_a1_lin"
    echo "  Submitted eval for prectx_a1_lin"
fi

if [ -n "$CKPT_PRECTX_CONST" ] && [ -d "$CKPT_PRECTX_CONST" ]; then
    sbatch slurm/cscs/eval.sh "$CKPT_PRECTX_CONST" \
        "VityaVitalich/Llama3.1-8b-instruct" "[]" "sdpo_prectx_a1_const"
    echo "  Submitted eval for prectx_a1_const"
fi

echo ""
echo "=== Eval jobs submitted ==="
EVALEOF

# Wait ~2 hours after SFT chain for SFTs to complete
JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SFT --begin=now+2hours "$EVAL_SCRIPT")
echo "[EVAL] Eval chain: Job $JOB_EVAL (2h after $JOB_SFT)"
rm -f "$EVAL_SCRIPT"

echo ""
echo "============================================================"
echo "Pipeline Summary"
echo "============================================================"
echo ""
echo "Phase 1 - Pretrain:"
echo "  $JOB_PT1: postctx_a1_const"
echo "  $JOB_PT2: interleaved_a1_lin (after $JOB_PT1)"
echo "  $JOB_CL1: cleanup + prectx_a1_lin + prectx_a1_const (after $JOB_PT2)"
echo ""
echo "Phase 2 - SFT:"
echo "  $JOB_SFT: SFT chain (11h after $JOB_CL1)"
echo ""
echo "Phase 3 - Eval:"
echo "  $JOB_EVAL: Eval chain (2h after $JOB_SFT)"
echo ""
echo "Total experiments: 4 pretrains → 4 SFTs → 4 evals"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/pretrain-sdpo-*.out, logs/sft-*.out, logs/eval-*.out"
echo "============================================================"
