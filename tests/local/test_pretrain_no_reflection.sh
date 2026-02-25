#!/bin/bash
# Test pre-training without reflections (baseline)
# Uses tiny settings to verify use_reflection=false works

set -euo pipefail

echo "=== Pre-training without reflections test ==="

echo "Running pre-training WITHOUT reflections (use_reflection=false)..."
python3 train.py \
  model=gpt2 \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name=./output/tiny_reflected \
  experiment.num_train_samples=10000 \
  experiment.use_reflection=false \
  dataset.seq_len=256 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=5000 \
  training.save_steps=10000 \
  training.logging_steps=1 \
  training.disable_tqdm=true \
  training.log_level=info \
  training.dataloader_num_workers=0 \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false

echo "✓ No-reflection test complete!"
