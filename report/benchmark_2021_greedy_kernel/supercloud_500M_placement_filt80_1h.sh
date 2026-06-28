#!/bin/bash
# 500M greedy-uniform placement: keep top ~20% of cells (candidate_percentile=0.8),
# Gurobi time limit 1 h. Writes canonical ..._mean.json + ..._mean_filt80.json copy.

#SBATCH --job-name=wf_500M_f80_1h
#SBATCH --cpus-per-task=32
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

echo "500M placement: candidate_percentile=0.8 (top ~20%% cells), Gurobi limit=3600s"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget 500 \
  --time-limit 3600 \
  --candidate-percentile 0.8

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxGreedyUniform_261x161_mean.json" \
  --scale both --tag "_greedy_uniform_500M"

echo "Done."
