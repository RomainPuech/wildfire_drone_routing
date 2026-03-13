#!/usr/bin/env bash
#
# Reproduce all results and plots in the benchmark report.
# Run from the project root (cph/):
#   bash report/benchmark_2021_uniform_kernel/reproduce.sh
#
set -euo pipefail

REPORT_DIR="report/benchmark_2021_uniform_kernel"
LOG_DIR="California2021Dataset/logs"

echo "=== Step 1/4: Clear cached sensor allocations ==="
rm -f "$LOG_DIR"/sensor_alloc_GaussianBudget20M_TOP_261x161_mean.json
rm -f "$LOG_DIR"/sensor_alloc_GaussianBudget100M_TOP_261x161_mean.json
rm -f "$LOG_DIR"/sensor_alloc_GaussianBudget500M_TOP_261x161_mean.json

echo "=== Step 2/4: Run sensor placement for all budgets ==="
python-jl run_benchmark_california2021_yearly.py --sensor-only --budget 20
python-jl run_benchmark_california2021_yearly.py --sensor-only --budget 100
python-jl run_benchmark_california2021_yearly.py --sensor-only --budget 500

echo "=== Step 3/4: Generate placement plots ==="
python visualize_sensor_placement_2021.py \
  "$LOG_DIR"/sensor_alloc_GaussianBudget20M_TOP_261x161_mean.json \
  --scale both

python visualize_sensor_placement_2021.py \
  "$LOG_DIR"/sensor_alloc_GaussianBudget100M_TOP_261x161_mean.json \
  --scale both --tag _100M

python visualize_sensor_placement_2021.py \
  "$LOG_DIR"/sensor_alloc_GaussianBudget500M_TOP_261x161_mean.json \
  --scale both --tag _500M

cp report/california_2021_sensor_clusters.png         "$REPORT_DIR"/
cp report/california_2021_sensor_clusters_opt.png     "$REPORT_DIR"/
cp report/california_2021_sensor_clusters_100M.png    "$REPORT_DIR"/
cp report/california_2021_sensor_clusters_opt_100M.png "$REPORT_DIR"/
cp report/california_2021_sensor_clusters_500M.png    "$REPORT_DIR"/
cp report/california_2021_sensor_clusters_opt_500M.png "$REPORT_DIR"/

echo "=== Step 4/4: Compile PDF ==="
pandoc "$REPORT_DIR"/benchmark_2021_uniform_kernel.md \
  -o "$REPORT_DIR"/benchmark_2021_uniform_kernel.pdf \
  --pdf-engine=pdflatex \
  -V colorlinks=true -V linkcolor=blue

echo "=== Done ==="
echo "Report: $REPORT_DIR/benchmark_2021_uniform_kernel.pdf"
