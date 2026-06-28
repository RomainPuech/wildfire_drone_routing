#!/bin/bash

#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_greedy_uniform_benchmarks
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0
module load julia/1.9.1
module load gurobi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 20  --time-limit 600
python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 100 --time-limit 1800
python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 500 --time-limit 600

LOG_DIR=California2021Dataset/logs

python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean.json"  --scale both --tag _greedy_uniform_20M
python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget100M_StationMaxGreedyUniform_261x161_mean.json" --scale both --tag _greedy_uniform_100M
python -u visualize_sensor_placement_2021.py "${LOG_DIR}/sensor_alloc_GaussianBudget500M_StationMaxGreedyUniform_261x161_mean.json" --scale both --tag _greedy_uniform_500M
