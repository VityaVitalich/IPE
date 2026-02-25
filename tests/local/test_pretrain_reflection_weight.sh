#!/bin/bash
# Test pre-training with custom reflection loss weight
# Verifies that reflection_loss_weight parameter works

set -euo pipefail

echo "=== Pre-training with reflection_loss_weight=2.0 test ==="


python3 train.py \
  model=gpt2 \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name=./output/tiny_reflected \
  experiment.num_train_samples=10000 \
  experiment.use_reflection=true \
  experiment.reflection_loss_weight=2.0 \
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

echo "✓ Reflection weight test complete!"
