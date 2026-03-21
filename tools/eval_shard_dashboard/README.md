# Eval Shard Dashboard

Interactive browser for eval outputs (`summary.json` + `L*_details.jsonl`) using Streamlit.

It is meant for qualitative inspection of shard directories such as:

- `outputs/eval/shards/eval_<run>_shard0`
- `outputs/eval/shards/eval_<run>_shard1`
- `outputs/eval/merged/eval_<run>` (works too if detail files exist)

## Features

- Select shard directories from a root folder
- Group shard directories by run prefix (`..._shardN`)
- Choose level file (`L1` ... `L5`)
- Filter by topic, labels, `q_id`, text search
- Inspect responses and judge outputs side-by-side
- View summary metrics (generation + probabilistic) for the selected level

## Install (local or server)

```bash
pip install --user -r tools/eval_shard_dashboard/requirements.txt
```

## Run Locally

```bash
streamlit run tools/eval_shard_dashboard/app.py -- --root outputs/eval/shards
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`) in Chrome.

## Run On SLURM (CSCS helper script)

Submit a job that serves the dashboard:

```bash
sbatch slurm/cscs/analysis/serve_eval_dashboard.sh
```

Custom root/port example:

```bash
sbatch slurm/cscs/analysis/serve_eval_dashboard.sh outputs/eval/shards 8502
```

Check the log to see the assigned node and tunnel command:

```bash
tail -f logs/eval-dashboard-<JOB_ID>.out
```

Create an SSH tunnel from your local machine and open in Chrome:

```bash
ssh -N -L 8501:<compute-node-hostname>:8501 <your-login-host>
```

Then browse:

```text
http://localhost:8501
```

If your cluster requires a login-node hop, use your standard `ssh -J ...` setup.
