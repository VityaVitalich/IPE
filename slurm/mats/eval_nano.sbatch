#!/bin/bash
#SBATCH --job-name=eval_nano
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=31G
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

cd /mnt/nw/home/j.minder/repositories/IPE
uv run python eval.py generation.num_samples=4 judge.model="openrouter/openai/gpt-5-nano" output.run_id=nano_10ep
