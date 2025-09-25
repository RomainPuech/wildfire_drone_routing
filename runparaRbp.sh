#!/bin/bash

#SBATCH --cpus-per-task=32            # Number of CPUs per task


source /etc/profile.d/modules.sh

module load anaconda/Python-ML-2025a
#source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh

# conda init bash

# conda activate wfdrone

module load julia
module load mpi/openmpi-5.0.5
module load gurobi

python all_experiments_parallel.py --ss_prefix R --bm_prefix bp
