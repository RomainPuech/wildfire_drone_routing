#!/bin/bash

#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_greedy_uniform_wrapup
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

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
