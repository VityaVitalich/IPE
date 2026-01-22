#!/bin/bash
# Minimal sharded eval test - verifies sharding + merge pipeline works.

set -e

echo "=== Minimal Sharded Eval Test ==="
echo "Testing: 2 shards + merge on small sample"
echo ""

cd "$(dirname "$0")/.."

RUN_ID="test_shard_merge_$$"
OUTPUT_DIR="outputs/eval"

COMMON_ARGS=(
  model.target='HuggingFaceTB/SmolLM2-360M-Instruct'
  model.judge=same
  model.device=cpu
  generation.num_samples=4
  generation.max_new_tokens=4
  generation.batch_size=2
  judge.max_new_tokens=2
  probabilistic.batch_size=2
  data.topic_ids=[p1]
  data.max_questions_per_topic=8
  data.num_shards=2
  output.run_id=${RUN_ID}
  output.dir=${OUTPUT_DIR}
  output.save_json=true
  output.save_details=false
)

python3 eval.py "${COMMON_ARGS[@]}" data.shard_index=0
python3 eval.py "${COMMON_ARGS[@]}" data.shard_index=1

python3 merge_eval_shards.py --output-dir "${OUTPUT_DIR}" --run-id "${RUN_ID}"

echo ""
echo "=== Test passed! ==="
