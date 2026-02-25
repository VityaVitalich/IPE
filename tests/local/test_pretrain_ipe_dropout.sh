#!/bin/bash
# Test script for IPE (Implicit Persona Engineering) with KV-cache dropout
# Tests the IPE trainer with dropout applied to the KV-cache

set -euo pipefail

echo "=== IPE Pre-training Test with KV-Cache Dropout ==="

# Test settings: small dataset, short training, with dropout
echo "Running IPE training with:"
echo "  - trainer_type=ipe (Implicit Persona Engineering)"
echo "  - kv_cache_dropout=0.1 (10% dropout on KV-cache)"
echo "  - Small dataset for quick testing"

echo "Running IPE pre-training with KV-cache dropout..."
python3 train.py \
  model=gpt2 \
  experiment=pretrain \
  dataset=pretrain \
  dataset.name=./output/tiny_reflected \
  experiment.num_train_samples=10000 \
  experiment.use_reflection=true \
  experiment.trainer_type="ipe" \
  experiment.ipe.kv_cache_dropout=0.1 \
  experiment.reflection_loss_weight=1.0 \
  dataset.seq_len=256 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=5000 \
  training.save_steps=10000 \
  training.logging_steps=5 \
  training.disable_tqdm=true \
  training.log_level=info \
  training.dataloader_num_workers=0 \
  wandb.project=ipe-smoke-test \
  hfhub.push_to_hub=false

echo "✓ IPE test with dropout complete!"

echo ""
echo "=== All IPE tests passed! ==="
