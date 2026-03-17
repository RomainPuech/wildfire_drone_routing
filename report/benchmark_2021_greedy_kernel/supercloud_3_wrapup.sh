#!/bin/bash

#SBATCH --job-name=wf_greedy_uniform_wrapup
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

SUBREPORT=report/benchmark_2021_greedy_kernel

cp report/benchmark_fire_locations_budget_2021.png                    "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_greedy_uniform_20M.png      "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_opt_greedy_uniform_20M.png  "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_greedy_uniform_100M.png     "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_opt_greedy_uniform_100M.png "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_greedy_uniform_500M.png     "${SUBREPORT}/"
cp report/california_2021_sensor_clusters_opt_greedy_uniform_500M.png "${SUBREPORT}/"

if ! command -v pandoc &>/dev/null; then
    echo "ERROR: pandoc not found. PDF not built. Install with: conda install -c conda-forge pandoc"
    exit 1
fi

cd "${SUBREPORT}"
pandoc benchmark_2021_greedy_kernel.md -o benchmark_2021_greedy_kernel.pdf
echo "PDF built."
