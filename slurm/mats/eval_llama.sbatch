#!/bin/bash
#SBATCH --job-name=eval_llama
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=31G
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

cd /mnt/nw/home/j.minder/repositories/IPE
uv run python eval.py generation.num_samples=4 judge.model="openrouter/meta-llama/llama-3.1-8b-instruct" output.run_id=llama_10ep
