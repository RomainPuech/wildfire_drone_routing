#!/usr/bin/env bash
#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_greedy_stationmax
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# CPUs: each array task gets 32. PDF: use submit_greedy_benchmark.sh to run array + wrap-up after all finish.
# Submit from project root: sbatch report/benchmark_2021_greedy_kernel/slurm_reproduce_greedy_kernel.sh

# Load environment first (no strict mode yet — system profile scripts
# reference unset vars and may return non-zero; matches the known-working example).
source /etc/profile
module load community-modules
module load miniforge/25.11.0-0
module load julia/1.9.1
module load gurobi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

# Strict mode on now that the environment is fully set up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs"

# Map array index -> (budget, time_limit)
BUDGETS=(20 100 500)
TIME_LIMITS=(600 1800 600)

IDX="${SLURM_ARRAY_TASK_ID}"
BUDGET="${BUDGETS[$IDX]}"
TIME_LIMIT="${TIME_LIMITS[$IDX]}"

echo "SLURM job $SLURM_JOB_ID, array index $IDX"
echo "Running greedy-kernel StationMax benchmark for budget=${BUDGET}M, time_limit=${TIME_LIMIT}s"

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
