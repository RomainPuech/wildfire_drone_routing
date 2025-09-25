#!/bin/bash

#SBATCH --cpus-per-task=1            # Number of CPUs per task

# Configuration for the cluster

sbatch runparaRbm.sh # All drone routing strategies with Random sensor placement on the "ground-truth" burn map
sbatch runparaRbp.sh # All drone routing strategies with Random sensor placement on the BP risk map

sbatch runparaKbm.sh # All drone routing strategies with Gaussian Coverage sensor placement on the "ground-truth" burn map
sbatch runparaKbp.sh # All drone routing strategies with Gaussian Coverage sensor placement on the BP risk map