#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/skrsteski/IPE/container/datatrove.toml
#SBATCH --output=logs/download-dataset-%j.out
#SBATCH --error=logs/download-dataset-%j.err
#SBATCH --no-requeue

# Download TinyStories dataset to cache (run ONCE before prepare_dataset.sh)
# Usage: sbatch slurm/cscs/download_dataset.sh

set -eo pipefail

# HuggingFace cache location
export HF_HOME=/capstor/store/cscs/swissai/a141/hf_cache
export HF_TOKEN=""

# Clean up any corrupted cache first
echo "Cleaning up corrupted cache (if any)..."
rm -rf /capstor/store/cscs/swissai/a141/hf_cache/datasets/roneneldan___tiny_stories

# Create cache directory
mkdir -p "$HF_HOME"

echo "START TIME: $(date)"
echo "Downloading TinyStories dataset to: $HF_HOME"

python -c "
import os
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

from datasets import load_dataset

print('Downloading TinyStories dataset...')
ds = load_dataset('roneneldan/TinyStories', split='train')
print(f'Successfully downloaded {len(ds)} training examples!')

# Also download validation split if needed
print('Downloading validation split...')
ds_val = load_dataset('roneneldan/TinyStories', split='validation')
print(f'Successfully downloaded {len(ds_val)} validation examples!')

print('Dataset cached successfully!')
"

echo "FINISH TIME: $(date)"
echo "✓ Dataset download complete! You can now run prepare_dataset.sh with HF_DATASETS_OFFLINE=1"
