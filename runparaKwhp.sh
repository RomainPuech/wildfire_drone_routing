#!/bin/bash

#SBATCH --cpus-per-task=32            # Number of CPUs per task


# source /etc/profile.d/modules.sh
# module load anaconda/Python-ML-2025a
# module load julia
# module load mpi/openmpi-5.0.5
# module load gurobi

python all_experiments_parallel.py --ss_prefix K --bm_prefix whp