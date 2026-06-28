#!/bin/bash
# Greedy-uniform StationMax placement + placement figures (one array task per budget).

#SBATCH --job-name=wf_greedy_place
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH --time=14:00:00
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
# Anaconda stack may preload julia/1.10.1; avoid conflict with standalone julia.
module unload julia/1.10.1 2>/dev/null || module unload julia 2>/dev/null || true
module load julia
module load mpi/openmpi-5.0.7
module load gurobi

BUDGETS=(20 100 500)
# Gurobi --time-limit (seconds): 20M stays 10 min; 100M / 500M get 12 h.
TIME_LIMITS=(600 43200 43200)

BUDGET="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"
TIME_LIMIT="${TIME_LIMITS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: budget=${BUDGET}M time_limit=${TIME_LIMIT}s"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget "${BUDGET}" --time-limit "${TIME_LIMIT}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget${BUDGET}M_StationMaxGreedyUniform_261x161_mean.json" \
  --scale both --tag "_greedy_uniform_${BUDGET}M"

echo "Placement task ${SLURM_ARRAY_TASK_ID} done."
