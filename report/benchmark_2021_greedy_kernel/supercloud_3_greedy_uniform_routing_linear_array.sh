#!/bin/bash
# Yearly 2021 benchmark — LinearMinTime only (same setup as greedy-kernel routing: no clustering).
# One budget per array task: 0=20M, 1=100M, 2=500M

#SBATCH --job-name=wf_greedy_route_lin
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module unload julia/1.10.1 2>/dev/null || module unload julia 2>/dev/null || true
module load julia
module load mpi/openmpi-5.0.7
module load gurobi

BUDGETS=(20 100 500)
BUDGET="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: budget=${BUDGET}M strategy=LinearMinTime"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget "${BUDGET}" \
  --no-clustering \
  --strategy LinearMinTime

echo "Linear routing task ${SLURM_ARRAY_TASK_ID} done."
