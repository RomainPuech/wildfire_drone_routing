#!/bin/bash
#SBATCH --job-name=wf_100M_MC_r7o7
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 100M MaxCov yearly benchmark: reevaluation_step=7, optimization_horizon=7,
# detection scored only over 6 data steps (hours) after first burn in scenario.
#
# New routing_yearly_* files use 7OH_7RS_ and dh6_ (no overwrite of 10OH_5RS logs).
# CSV strategy_combo: GaussianBudget100M_MaxCov_rs7_oh7_dh6

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONUNBUFFERED=1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda/Python-ML-2025a 2>/dev/null || true
module unload julia/1.10.1 2>/dev/null || module unload julia 2>/dev/null || true
module load julia 2>/dev/null || true
module load mpi/openmpi-5.0.7 2>/dev/null || true
module load gurobi 2>/dev/null || true

# Note: do not pass python's -u before the script name — python-jl treats it as the script path.
python-jl run_benchmark_california2021_yearly.py \
  --budget 100 \
  --strategy MaxCov \
  --reevaluation-step 7 \
  --optimization-horizon 7 \
  --detection-horizon-data-steps 6

echo "Done."
