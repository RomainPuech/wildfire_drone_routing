#!/bin/bash

#SBATCH --job-name=wf_2021_wrapup
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

python -u report/generate_benchmark_report_figures_2021.py

echo "Report figures written to report/ (benchmark_fire_locations_budget_2021.png, benchmark_fire_map_budget_2021_*.png)."
