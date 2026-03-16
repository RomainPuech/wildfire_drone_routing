#!/usr/bin/env bash
#SBATCH --job-name=wf_greedy_stationmax
#SBATCH --output=logs/wf_greedy_stationmax_%A_%a.out
#SBATCH --error=logs/wf_greedy_stationmax_%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=compute

# Optional: load your modules / env here
# module load anaconda
# source activate wf

# Submit from project root: sbatch report/benchmark_2021_greedy_kernel/slurm_reproduce_greedy_kernel.sh
# Or from this folder: sbatch slurm_reproduce_greedy_kernel.sh
# Slurm logs go to logs/ relative to the directory from which you run sbatch.

set -euo pipefail

mkdir -p logs

# Map array index -> (budget, time_limit)
BUDGETS=(20 100 500)
TIME_LIMITS=(600 1800 600)

IDX="${SLURM_ARRAY_TASK_ID}"
BUDGET="${BUDGETS[$IDX]}"
TIME_LIMIT="${TIME_LIMITS[$IDX]}"

echo "SLURM job $SLURM_JOB_ID, array index $IDX"
echo "Running greedy-kernel StationMax benchmark for budget=${BUDGET}M, time_limit=${TIME_LIMIT}s"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_JL="/opt/anaconda3/envs/wf/bin/python-jl"
PYTHON_BIN="python"

REPORT_DIR="${PROJECT_ROOT}/report"
LOG_DIR="${PROJECT_ROOT}/California2021Dataset/logs"

cd "${PROJECT_ROOT}"

# Clean old cache for this budget
rm -f "${LOG_DIR}/sensor_alloc_GaussianBudget${BUDGET}M_StationMax_261x161_mean.json"

# 1) Run the placement benchmark for this budget
"${PYTHON_JL}" "${PROJECT_ROOT}/test_budget_placement_station_max_2021.py" \
  --budget "${BUDGET}" \
  --time-limit "${TIME_LIMIT}"

# 2) Generate the plots for this budget
TAG="_greedy_${BUDGET}M"

"${PYTHON_BIN}" "${PROJECT_ROOT}/visualize_sensor_placement_2021.py" \
  "${LOG_DIR}/sensor_alloc_GaussianBudget${BUDGET}M_StationMax_261x161_mean.json" \
  --scale both \
  --tag "${TAG}"

echo "Done for budget ${BUDGET}M"
