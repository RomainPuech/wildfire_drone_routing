#!/bin/bash

#SBATCH --job-name=wf_greedy_benchmarks
#SBATCH --cpus-per-task=32

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.5
module load gurobi

python-jl test_budget_placement_station_max_2021.py --budget 20  --time-limit 600
python-jl test_budget_placement_station_max_2021.py --budget 100 --time-limit 1800
python-jl test_budget_placement_station_max_2021.py --budget 500 --time-limit 600

LOG_DIR=California2021Dataset/logs

python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget20M_StationMax_261x161_mean.json"  --scale both --tag _greedy_20M
python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget100M_StationMax_261x161_mean.json" --scale both --tag _greedy_100M
python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget500M_StationMax_261x161_mean.json" --scale both --tag _greedy_500M
