#!/bin/bash

#SBATCH --job-name=wf_greedy_500M_s2
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.7
module load gurobi

BUDGET=500
TIME_LIMIT=43200
CAND_PCT=0.0
WARM_START="California2021Dataset/logs/sensor_alloc_GaussianBudget500M_StationMaxGreedyUniform_261x161_mean_filt20.json"

python test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget "${BUDGET}" --time-limit "${TIME_LIMIT}" \
  --candidate-percentile "${CAND_PCT}" \
  --warm-start "${WARM_START}"

python -u visualize_sensor_placement_2021.py \
  "California2021Dataset/logs/sensor_alloc_GaussianBudget${BUDGET}M_StationMaxGreedyUniform_261x161_mean.json" \
  --scale both --tag "_greedy_uniform_${BUDGET}M"


