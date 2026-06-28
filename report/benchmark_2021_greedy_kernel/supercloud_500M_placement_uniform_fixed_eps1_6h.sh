#!/bin/bash
# 500M UniformFixedDrones MIP with epsilon=1.0 (budget regularization).
# eps=0.5 run (fullpool_eps05_6h) gave 367 stations / 186.6M budget; this
# run increases penalty to ensure even tighter removal of near-redundant stations.
# Warm-starts from the eps=0.5 solution (already clean, no redundant stations).
# Gurobi time limit 6 h, full candidate pool.

#SBATCH --job-name=wf_500M_ufix_eps1
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

WS="California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps05_6h.json"
TAG="fullpool_eps1_6h"

echo "500M UniformFixedDrones: warm_start=${WS}, epsilon=1.0, candidate_percentile=0, tag=${TAG}, limit=21600s"

python-jl test_budget_placement_station_max_uniform_fixed_drones_2021.py \
  --budget 500 \
  --time-limit 21600 \
  --candidate-percentile 0 \
  --warm-start "${WS}" \
  --fixed-drones-per-station 7 \
  --epsilon 1.0 \
  --output-tag "${TAG}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_${TAG}.json" \
  --scale both \
  --tag "_uniform_fixed_500M_${TAG}"

echo "Done."
