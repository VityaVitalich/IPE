#!/bin/bash

#SBATCH --job-name=inject_prefs
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

set -eo pipefail

cd /mnt/nw/home/j.minder/repositories/IPE_data_proc

echo "START TIME: $(date) | Injecting preferences into TinyStories"
start=$(date +%s)

uv run python data/tinystories_prefs_injection/inject_preferences.py \
    -n 100000 \
    --mode opposing \
    --output data/tinystories_prefs_injection/output \
    --push-to-hf jkminder/tinystories_preferences

end=$(date +%s)
echo "FINISH TIME: $(date) | Done!"
echo "Total elapsed time: $((end - start)) seconds"
