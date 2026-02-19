#!/bin/bash
#SBATCH --gres=gpu:l40:1
#SBATCH --job-name=eval_3judge
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

set -euo pipefail
cd /mnt/nw/home/j.minder/repositories/IPE

echo "=== Starting 3 judge evals in parallel on single GPU ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start time: $(date)"

uv run python eval.py judge.model="openrouter/openai/gpt-5-nano" output.run_id=nano_10ep &
PID_NANO=$!
echo "Launched nano eval (PID=$PID_NANO)"

uv run python eval.py judge.model="openrouter/openai/gpt-5-mini" output.run_id=mini_10ep &
PID_MINI=$!
echo "Launched mini eval (PID=$PID_MINI)"

uv run python eval.py judge.model="openrouter/meta-llama/llama-3.1-8b-instruct" output.run_id=llama_10ep &
PID_LLAMA=$!
echo "Launched llama eval (PID=$PID_LLAMA)"

echo "Waiting for all 3 evals to complete..."
FAIL=0
wait $PID_NANO  || { echo "NANO eval failed (exit $?)"; FAIL=1; }
wait $PID_MINI  || { echo "MINI eval failed (exit $?)"; FAIL=1; }
wait $PID_LLAMA || { echo "LLAMA eval failed (exit $?)"; FAIL=1; }

echo "End time: $(date)"
echo "=== All evals finished (failures=$FAIL) ==="
exit $FAIL
