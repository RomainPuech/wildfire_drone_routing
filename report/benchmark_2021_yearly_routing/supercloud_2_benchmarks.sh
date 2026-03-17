#!/bin/bash

#SBATCH --job-name=wf_2021_routing
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.5
module load gurobi

BUDGETS=(20 100 500)

BUDGET="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"

python -u run_benchmark_california2021_yearly.py --budget "${BUDGET}"
