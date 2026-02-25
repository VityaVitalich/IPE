#!/bin/bash
# Minimal eval test - verifies evaluation pipeline runs end-to-end.

set -e

echo "=== Minimal Eval Test ==="
echo "Testing: L1/L3/L4 eval with small sample sizes"
echo ""

cd "$(dirname "$0")/.."

python3 eval.py \
    model.target='HuggingFaceTB/SmolLM2-360M-Instruct' \
    model.judge=same \
    generation.num_samples=2 \
    generation.max_new_tokens=8 \
    generation.batch_size=4 \
    judge.max_new_tokens=2 \
    probabilistic.batch_size=4 \
    data.topic_ids=[p1,p2] \
    data.max_questions_per_topic=2 \
    output.save_json=true \
    output.save_details=true \
    output.report_per_topic=true

echo ""
echo "=== Test passed! ==="
