#!/bin/bash
# 50M TOPGrowing yearly routing — same settings as supercloud_3_greedy_uniform_routing_array.sh
# (no-clustering, default reeval/horizon/time-limit, no extra suffix).

#SBATCH --job-name=wf_greedy_route_50M_TOP
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

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

echo "50M TOPGrowing routing (no-clustering, default reeval/horizon/time-limit)"

export PYTHONUNBUFFERED=1
python-jl run_benchmark_california2021_yearly.py \
  --budget 50 \
  --no-clustering \
  --strategy TOPGrowing

echo "50M TOPGrowing routing done."
