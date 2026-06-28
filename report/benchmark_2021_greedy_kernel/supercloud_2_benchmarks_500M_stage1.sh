#!/bin/bash

#SBATCH --job-name=wf_greedy_500M_s1
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
CAND_PCT=0.2
WARM_START_100M="California2021Dataset/logs/sensor_alloc_GaussianBudget100M_StationMaxGreedyUniform_261x161_mean.json"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py \
  --budget "${BUDGET}" --time-limit "${TIME_LIMIT}" \
  --candidate-percentile "${CAND_PCT}" \
  --warm-start "${WARM_START_100M}"

echo "Stage 1 done — submitting stage 2 (0% filtering, warm start from filt20)"
sbatch report/benchmark_2021_greedy_kernel/supercloud_2_benchmarks_500M_stage2.sh
