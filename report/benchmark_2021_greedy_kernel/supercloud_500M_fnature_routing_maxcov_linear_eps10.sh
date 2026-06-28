#!/bin/bash
# 500M final-nature yearly routing: MaxCov and LinearMinTime.
# Uses the UniformFixedDrones eps=10.0 sensor placement (fullpool_eps10_6h).
# Configuration identical to supercloud_final_nature_routing_array.sh:
#   no-clustering, rs=7, oh=7, 300s Gurobi cap, detection_horizon=6.
# CSV suffix: _final_nature  |  routing log tag: final_nature
#
# Array: 0=MaxCov, 1=LinearMinTime

#SBATCH --job-name=wf_500M_fnature_ml_e10
#SBATCH --cpus-per-task=32
#SBATCH --array=0-1
#SBATCH --time=24:00:00
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
module unload julia/1.10.1 2>/dev/null || module unload julia 2>/dev/null || true
module load julia
module load mpi/openmpi-5.0.7
module load gurobi

PLACEMENT_JSON="California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps10_6h.json"

case "${SLURM_ARRAY_TASK_ID}" in
  0) STRAT="MaxCov" ;;
  1) STRAT="LinearMinTime" ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "500M final-nature eps10 | Task ${SLURM_ARRAY_TASK_ID}: strategy=${STRAT}"
echo "  placement=${PLACEMENT_JSON}"
echo "  clustering=OFF rs=7 oh=7 routing_time_limit=300s detection_horizon=6"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget 500 \
  --no-clustering \
  --strategy "${STRAT}" \
  --reevaluation-step 7 \
  --optimization-horizon 7 \
  --routing-time-limit 300 \
  --detection-horizon-data-steps 6 \
  --combo-name-suffix _final_nature \
  --routing-log-tag final_nature \
  --sensor-placement-file "${PLACEMENT_JSON}"

echo "500M final-nature eps10 task ${SLURM_ARRAY_TASK_ID} (${STRAT}) done."
