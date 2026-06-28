#!/bin/bash
# California 2021 yearly benchmark — Gaussian-budget sensor (cached TOP key) +
# DroneRoutingLinearMinTime. Submit from the repository root so #SBATCH logs/
# and Python paths resolve (same pattern as supercloud_2_benchmarks.sh).

#SBATCH --job-name=wf_2021_yearly_linear_20M
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

set -euo pipefail

# mkdir -p logs

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.7
module load gurobi

PYTHONUNBUFFERED=1 python-jl run_benchmark_california2021_yearly.py --budget 20 --no-clustering --strategy LinearMinTime
