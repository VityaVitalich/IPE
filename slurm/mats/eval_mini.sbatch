#!/bin/bash
#SBATCH --job-name=eval_mini
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=31G
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

cd /mnt/nw/home/j.minder/repositories/IPE
uv run python eval.py generation.num_samples=4 judge.model="openrouter/openai/gpt-5-mini" output.run_id=mini_10ep
