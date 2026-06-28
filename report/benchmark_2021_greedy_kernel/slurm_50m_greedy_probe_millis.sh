#!/bin/bash
# One 50M greedy-uniform placement at cost_sensor = MILLIS/1000 (e.g. MILLIS=50 -> 0.050).
# Submit: sbatch --export=ALL,MILLIS=36 .../slurm_50m_greedy_probe_millis.sh
#
#SBATCH --job-name=wf_t50m
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=logs/thresh50m_%j.out
#SBATCH --error=logs/thresh50m_%j.err

set -euo pipefail
: "${MILLIS:?}"

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
module load julia mpi/openmpi-5.0.7 gurobi

COST_SENSOR=$(awk "BEGIN {printf \"%.3f\", ${MILLIS}/1000}")
TAG="thresh50M_m${MILLIS}"
WARM="${PROJECT_ROOT}/California2021Dataset/logs/sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json"

export PYTHONUNBUFFERED=1
echo "MILLIS=${MILLIS} cost_sensor=${COST_SENSOR} tag=${TAG}"
python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget 50 \
  --time-limit 600 \
  --cost-sensor "${COST_SENSOR}" \
  --warm-start "${WARM}" \
  --output-tag "${TAG}"
echo "done millis=${MILLIS}"
