#!/bin/bash
# 20M greedy-uniform placement, cost_sensor=0.012M (1.2× analytic center 0.01M).
#
#SBATCH --job-name=wf_bb_cs012
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=logs/breakeven_20M_cs0p012_%x-%j.out
#SBATCH --error=logs/breakeven_20M_cs0p012_%x-%j.err

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

WARM="${PROJECT_ROOT}/California2021Dataset/logs/sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean.json"
export PYTHONUNBUFFERED=1

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget 20 \
  --time-limit 600 \
  --cost-sensor 0.012 \
  --warm-start "${WARM}" \
  --output-tag breakeven_cs0p012

echo "breakeven placement cs0.012M done."
