#!/usr/bin/env bash
#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_greedy_wrapup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

# Run after the array job: sbatch --dependency=afterok:ARRAY_JOBID .../slurm_reproduce_greedy_kernel_wrapup.sh
# Or use: report/benchmark_2021_greedy_kernel/submit_greedy_benchmark.sh

set -euo pipefail

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0
export PATH="${HOME}/.local/bin:$PATH"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/report"
SUBREPORT_DIR="${REPORT_DIR}/benchmark_2021_greedy_kernel"

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Copying figures into subreport folder..."
cp \
  "${REPORT_DIR}/benchmark_fire_locations_budget_2021.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_20M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_20M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_100M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_100M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_500M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_500M.png" \
  "${SUBREPORT_DIR}/"

echo "Rebuilding PDF..."
(cd "${SUBREPORT_DIR}" && pandoc "benchmark_2021_greedy_kernel.md" -o "benchmark_2021_greedy_kernel.pdf")

echo "Done. Report: ${SUBREPORT_DIR}"
