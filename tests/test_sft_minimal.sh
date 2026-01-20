#!/bin/bash
# Minimal SFT test - verifies the basic SFT pipeline works
# Uses GPT-2 (small, no login required) with smoltalk dataset
# Runs just 10 steps to verify everything connects

set -e

echo "=== Minimal SFT Test ==="
echo "Testing: SFT pipeline with chat formatting and label masking"
echo ""

cd "$(dirname "$0")/.."

python3 train_sft.py \
    model=gpt2 \
    experiment=sft \
    dataset=sft \
    dataset.name="HuggingFaceTB/smoltalk" \
    dataset.config="all" \
    dataset.max_seq_len=1024 \
    dataset.max_turns=2 \
    experiment.num_sft_samples=100 \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=1 \
    training.num_train_epochs=1 \
    training.max_steps=10 \
    training.learning_rate=1e-4 \
    training.logging_steps=2 \
    training.save_steps=999999 \
    wandb.project=ipe-smoke-test \
    hfhub.push_to_hub=false \
    suffix="test_sft"

echo ""
echo "=== Test passed! ==="
