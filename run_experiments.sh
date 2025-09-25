#!/bin/bash

#SBATCH --cpus-per-task=1            # Number of CPUs per task

# Configuration for the cluster
# source /etc/profile.d/modules.sh
# module load anaconda/Python-ML-2025a
# module load julia
# module load mpi/openmpi-5.0.5
# module load gurobi

LLsub runparaRbm.sh
LLsub runparaRbp.sh
LLsub runparaRwhp.sh
LLsub runparaKbm.sh
LLsub runparaKbp.sh
LLsub runparaKwhp.sh