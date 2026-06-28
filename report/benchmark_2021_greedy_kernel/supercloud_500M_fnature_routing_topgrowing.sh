#!/bin/bash
# 500M final-nature yearly routing: TOPGrowing.
# Uses the UniformFixedDrones eps=1.0 sensor placement (fullpool_eps1_6h).
# Configuration identical to supercloud_final_nature_routing_array.sh:
#   no-clustering, rs=7, oh=7, 300s Gurobi cap, detection_horizon=6.
# CSV suffix: _final_nature  |  routing log tag: final_nature

#SBATCH --job-name=wf_500M_fnature_top
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
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

PLACEMENT_JSON="California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps1_6h.json"

echo "500M final-nature TOPGrowing"
echo "  placement=${PLACEMENT_JSON}"
echo "  clustering=OFF rs=7 oh=7 routing_time_limit=300s detection_horizon=6"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget 500 \
  --no-clustering \
  --strategy TOPGrowing \
  --reevaluation-step 7 \
  --optimization-horizon 7 \
  --routing-time-limit 300 \
  --detection-horizon-data-steps 6 \
  --combo-name-suffix _final_nature \
  --routing-log-tag final_nature \
  --sensor-placement-file "${PLACEMENT_JSON}"

echo "500M final-nature TOPGrowing done."
