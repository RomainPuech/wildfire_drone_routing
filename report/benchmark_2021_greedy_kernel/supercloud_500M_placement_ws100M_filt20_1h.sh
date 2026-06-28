#!/bin/bash
# 500M greedy-uniform: mild filtering (candidate_percentile=0.2 → top ~80%% of cells),
# warm-start from 100M solution, 1 h Gurobi cap. Writes a tagged JSON so it does not
# overwrite the parallel filt80 (top 20%% cells) 500M run.

#SBATCH --job-name=wf_500M_ws_f20
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

echo "500M placement: candidate_percentile=0.2 (top ~80%% cells), warm_start=${WS}, tag=${TAG}, limit=3600s"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget 500 \
  --time-limit 3600 \
  --candidate-percentile 0.2 \
  --warm-start "${WS}" \
  --output-tag "${TAG}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxGreedyUniform_261x161_mean_${TAG}.json" \
  --scale both \
  --tag "_greedy_uniform_500M_${TAG}"

echo "Done."
