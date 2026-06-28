#!/bin/bash
# 500M UniformFixedDrones MIP: warm-start xg/y from 100M greedy-uniform JSON,
# candidate_percentile=0.2 (same candidate pool style as supercloud_500M_placement_ws100M_filt20_1h.sh),
# 7 drones per open station, 1 h Gurobi cap.

#SBATCH --job-name=wf_500M_ufix_ws
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

WS="California2021Dataset/logs/sensor_alloc_GaussianBudget100M_StationMaxGreedyUniform_261x161_mean.json"
TAG="ws100M_filt20"

echo "500M UniformFixedDrones: warm_start=${WS}, candidate_percentile=0.2, tag=${TAG}, limit=3600s"

python-jl test_budget_placement_station_max_uniform_fixed_drones_2021.py \
  --budget 500 \
  --time-limit 3600 \
  --candidate-percentile 0.2 \
  --warm-start "${WS}" \
  --fixed-drones-per-station 7 \
  --output-tag "${TAG}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_${TAG}.json" \
  --scale both \
  --tag "_uniform_fixed_500M_${TAG}"

echo "Done."
