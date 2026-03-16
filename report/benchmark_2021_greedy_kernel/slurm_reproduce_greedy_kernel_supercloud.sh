#!/bin/bash -l
#SBATCH --job-name=wf_greedy_stationmax
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00

# CPUs: each array task gets 32. PDF: use submit_greedy_benchmark_supercloud.sh to run
# the full chain (preprocess → array → wrapup) with proper Slurm dependencies.
#
# #!/bin/bash -l  (login shell) ensures MODULEPATH is initialized on compute
# nodes. PROJECT_ROOT uses SLURM_SUBMIT_DIR instead of BASH_SOURCE because
# Slurm copies scripts to a temp path before execution.

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.5
module load gurobi

export PATH="${HOME}/.local/bin:$PATH"

set -euo pipefail

# Map array index -> (budget, time_limit)
BUDGETS=(20 100 500)
TIME_LIMITS=(600 1800 600)

IDX="${SLURM_ARRAY_TASK_ID}"
BUDGET="${BUDGETS[$IDX]}"
TIME_LIMIT="${TIME_LIMITS[$IDX]}"

echo "SLURM job $SLURM_JOB_ID, array index $IDX"
echo "Running greedy-kernel StationMax benchmark for budget=${BUDGET}M, time_limit=${TIME_LIMIT}s"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

REPORT_DIR="${PROJECT_ROOT}/report"
LOG_DIR="${PROJECT_ROOT}/California2021Dataset/logs"

cd "${PROJECT_ROOT}"

# Clean old cache for this budget
rm -f "${LOG_DIR}/sensor_alloc_GaussianBudget${BUDGET}M_StationMax_261x161_mean.json"

# 1) Run the placement benchmark for this budget
python-jl "${PROJECT_ROOT}/test_budget_placement_station_max_2021.py" \
  --budget "${BUDGET}" \
  --time-limit "${TIME_LIMIT}"

# 2) Generate the plots for this budget
TAG="_greedy_${BUDGET}M"

python "${PROJECT_ROOT}/visualize_sensor_placement_2021.py" \
  "${LOG_DIR}/sensor_alloc_GaussianBudget${BUDGET}M_StationMax_261x161_mean.json" \
  --scale both \
  --tag "${TAG}"

echo "Done for budget ${BUDGET}M"
