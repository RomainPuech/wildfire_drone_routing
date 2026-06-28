#!/bin/bash
# Yearly 2021 benchmark, no clustering — one routing strategy per array task.
# Index = 2 * budget_index + strategy_index  (0=MaxCov, 1=TOPGrowing)

#SBATCH --job-name=wf_greedy_route
#SBATCH --cpus-per-task=32
#SBATCH --array=0-5
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
BI=$((SLURM_ARRAY_TASK_ID / 2))
SI=$((SLURM_ARRAY_TASK_ID % 2))
BUDGET="${BUDGETS[$BI]}"

if [[ "$SI" -eq 0 ]]; then
  STRAT="MaxCov"
else
  STRAT="TOPGrowing"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: budget=${BUDGET}M strategy=${STRAT}"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget "${BUDGET}" \
  --no-clustering \
  --strategy "${STRAT}"

echo "Routing task ${SLURM_ARRAY_TASK_ID} done."
