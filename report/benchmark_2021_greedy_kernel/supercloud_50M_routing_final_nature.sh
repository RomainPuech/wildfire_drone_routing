#!/bin/bash
# 50M final-nature yearly routing: MaxCov and LinearMinTime.
# Same configuration as supercloud_final_nature_routing_array.sh
# (no-clustering, rs=7, oh=7, 300s Gurobi cap, detection_horizon=6, _final_nature suffix).
#
# Array: 0=MaxCov, 1=LinearMinTime

#SBATCH --job-name=wf_fnature_route_50M
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

case "${SLURM_ARRAY_TASK_ID}" in
  0) STRAT="MaxCov" ;;
  1) STRAT="LinearMinTime" ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "final-nature run | Task ${SLURM_ARRAY_TASK_ID}: budget=50M strategy=${STRAT}"
echo "  clustering=OFF (--no-clustering) rs=7 oh=7 routing_time_limit=300s detection_horizon=6 data steps"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget 50 \
  --no-clustering \
  --strategy "${STRAT}" \
  --reevaluation-step 7 \
  --optimization-horizon 7 \
  --routing-time-limit 300 \
  --detection-horizon-data-steps 6 \
  --combo-name-suffix _final_nature \
  --routing-log-tag final_nature

echo "final-nature routing task ${SLURM_ARRAY_TASK_ID} (50M ${STRAT}) done."
