#!/bin/bash
# Minimal test script for pre-training - VERY fast for smoke testing
# Uses tiny settings to verify everything works

set -euo pipefail

echo "=== Minimal Pre-training Test (smoke test) ==="

# Ultra-minimal settings for quick smoke test
echo "Running with minimal settings (10 texts, seq_len=64, max_steps=5)..."

# First, prepare a tiny reflected dataset for testing
echo "Preparing tiny test dataset..."
python3 add_reflections.py \
  --dataset tiny \
  --output ./output/test_tiny_reflected \
  --format parquet \
  --limit 100 \
  --workers 2 \
  --seed 42

echo "Running pre-training with reflections..."
python3 train.py \
  model=gpt2 \
  experiment=pretrain \
  dataset=tinystories \
  dataset.name=./output/test_tiny_reflected \
  experiment.num_train_samples=10 \
  experiment.use_reflection=true \
  dataset.seq_len=64 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=5 \
  training.save_steps=10 \
  training.logging_steps=1 \
  training.disable_tqdm=true \
  training.log_level=info \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false

echo "✓ Smoke test complete!"
