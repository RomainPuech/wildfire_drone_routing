#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_JL="/opt/anaconda3/envs/wf/bin/python-jl"
PYTHON_BIN="python"

REPORT_DIR="${PROJECT_ROOT}/report"
SUBREPORT_DIR="${REPORT_DIR}/benchmark_2021_greedy_kernel"
LOG_DIR="${PROJECT_ROOT}/California2021Dataset/logs"

run_budget() {
  local budget="$1"
  local time_limit="$2"
  local tag="_greedy_uniform_${budget}M"

  echo
  echo "======================================================================"
  echo "Running greedy-uniform StationMax benchmark for ${budget}M (time limit: ${time_limit}s)"
  echo "======================================================================"

  rm -f "${LOG_DIR}/sensor_alloc_GaussianBudget${budget}M_StationMaxGreedyUniform_261x161_mean.json"

  "${PYTHON_JL}" "${PROJECT_ROOT}/test_budget_placement_station_max_greedy_uniform_2021.py" \
    --budget "${budget}" \
    --time-limit "${time_limit}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/visualize_sensor_placement_2021.py" \
    "${LOG_DIR}/sensor_alloc_GaussianBudget${budget}M_StationMaxGreedyUniform_261x161_mean.json" \
    --scale both \
    --tag "${tag}"
}

mkdir -p "${SUBREPORT_DIR}"

# Placement Gurobi time limits (seconds):
# - 20M: 10 min; 100M / 500M: 12 h (cluster Slurm script uses the same pattern).
run_budget 20 600
run_budget 100 43200
run_budget 500 43200

echo
echo "Copying figures into subreport folder..."
cp \
  "${REPORT_DIR}/benchmark_fire_locations_budget_2021.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_uniform_20M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_uniform_20M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_uniform_100M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_uniform_100M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_greedy_uniform_500M.png" \
  "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_uniform_500M.png" \
  "${SUBREPORT_DIR}/"

echo
echo "Rebuilding PDF..."
(
  cd "${SUBREPORT_DIR}"
  pandoc "benchmark_2021_greedy_kernel.md" -o "benchmark_2021_greedy_kernel.pdf"
)

echo
echo "Done."
echo "Report folder: ${SUBREPORT_DIR}"
