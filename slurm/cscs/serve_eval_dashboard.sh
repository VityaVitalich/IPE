#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/IPE/container/container.toml
#SBATCH --output=logs/eval-dashboard-%j.out
#SBATCH --error=logs/eval-dashboard-%j.err
#SBATCH --no-requeue

# Serve the interactive eval shard dashboard (Streamlit) on a SLURM node.
#
# Usage:
#   sbatch slurm/cscs/serve_eval_dashboard.sh [ROOT_DIR] [PORT]
# Examples:
#   sbatch slurm/cscs/serve_eval_dashboard.sh
#   sbatch slurm/cscs/serve_eval_dashboard.sh outputs/eval/shards 8501
#   sbatch slurm/cscs/serve_eval_dashboard.sh outputs/eval/merged 8502

ROOT_DIR=${1:-"outputs/eval/shards"}
PORT=${2:-8501}
HOST_BIND=${HOST_BIND:-"127.0.0.1"}

set -eo pipefail

# Change to project root directory
cd "$SLURM_SUBMIT_DIR"
if [ -f "tools/eval_shard_dashboard/app.py" ]; then
    : # Already in project root
elif [ -f "../tools/eval_shard_dashboard/app.py" ]; then
    cd ..
elif [ -f "../../tools/eval_shard_dashboard/app.py" ]; then
    cd ../..
fi

mkdir -p logs

echo "START TIME: $(date)"
echo "Node: $(hostname -f)"
echo "Dashboard root: $ROOT_DIR"
echo "Port: $PORT"
echo "Bind address: $HOST_BIND"

if ! python3 -c "import streamlit" >/dev/null 2>&1; then
    echo "Error: streamlit is not installed in this environment."
    echo "Install with: pip install --user -r tools/eval_shard_dashboard/requirements.txt"
    exit 1
fi

echo
echo "To open in your local Chrome, create an SSH tunnel from your laptop/workstation:"
echo "  ssh -N -L ${PORT}:$(hostname -f):${PORT} <your-login-host>"
echo "Then open: http://localhost:${PORT}"
echo
echo "If your cluster requires a login-node hop, use your usual ProxyJump / -J setup."
echo

streamlit run tools/eval_shard_dashboard/app.py \
  --server.headless true \
  --server.address "$HOST_BIND" \
  --server.port "$PORT" \
  -- \
  --root "$ROOT_DIR"
