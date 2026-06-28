#!/bin/bash
# 500M UniformFixedDrones MIP with epsilon=0.5 (budget regularization).
# Previous run (fullpool_ws500f20_6h) used epsilon=0.1 which left 83 redundant
# stations due to the penalty being invisible within Gurobi's MIP gap tolerance.
# Warm-starts from the pruned solution (83 redundant stations removed).
# Gurobi time limit 6 h, full candidate pool.

#SBATCH --job-name=wf_500M_ufix_eps05
#SBATCH --cpus-per-task=32
#SBATCH --time=07:00:00
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

WS="California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_ws500f20_6h_pruned.json"
TAG="fullpool_eps05_6h"

echo "500M UniformFixedDrones: warm_start=${WS}, epsilon=0.5, candidate_percentile=0, tag=${TAG}, limit=21600s"

python-jl test_budget_placement_station_max_uniform_fixed_drones_2021.py \
  --budget 500 \
  --time-limit 21600 \
  --candidate-percentile 0 \
  --warm-start "${WS}" \
  --fixed-drones-per-station 7 \
  --epsilon 0.5 \
  --output-tag "${TAG}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_${TAG}.json" \
  --scale both \
  --tag "_uniform_fixed_500M_${TAG}"

echo "Done."
