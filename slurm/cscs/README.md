# CSCS SLURM Scripts

Scripts under `slurm/cscs/` are grouped by workflow instead of kept in one flat directory.

- `datasets/`: dataset download, preprocessing, filling, and SFT dataset build jobs
- `pretrain/`: pretraining jobs and conflict-pretraining variants
- `sft/`: SFT training jobs and SFT submission helpers
- `eval/`: evaluation jobs and eval batch submitters
- `generation/`: standalone generation jobs and checkpoint sweep helpers
- `analysis/`: visualization, comparison, dashboard, plotting, and table-collection jobs

Example commands:

- `sbatch slurm/cscs/pretrain/pretrain_iepe.sh`
- `sbatch slurm/cscs/sft/sft.sh`
- `sbatch slurm/cscs/eval/eval.sh`
- `bash slurm/cscs/eval/eval_multi.sh --dry-run`
- `sbatch slurm/cscs/analysis/visualize.sh <RUN_LABEL>`
