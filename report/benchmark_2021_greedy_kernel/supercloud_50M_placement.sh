#!/bin/bash
# Greedy-uniform StationMax placement for the 50M budget.
# Matches the pattern of supercloud_2_greedy_uniform_placement_array.sh
# (same python driver, same visualisation step) but is a single job (no array).

#SBATCH --job-name=wf_greedy_place_50M
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

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

BUDGET=50
TIME_LIMIT=43200   # 12 h, same as 100M / 500M

echo "50M placement: budget=${BUDGET}M time_limit=${TIME_LIMIT}s"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget "${BUDGET}" --time-limit "${TIME_LIMIT}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget${BUDGET}M_StationMaxGreedyUniform_261x161_mean.json" \
  --scale both --tag "_greedy_uniform_${BUDGET}M"

echo "Placement 50M done."
