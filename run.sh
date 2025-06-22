#!/bin/bash

#SBATCH --cpus-per-task=1            # Number of CPUs per task


source /etc/profile.d/modules.sh

module load anaconda/Python-ML-2025a
module load julia
module load mpi/openmpi-5.0.5
module load gurobi

LLsub runRbm.sh
LLsub runRbp.sh
LLsub runRwhp.sh
LLsub runKbm.sh
LLsub runKbp.sh
LLsub runKwhp.sh
