#!/bin/bash

#SBATCH --cpus-per-task=20            # Number of CPUs per task
#SBATCH --output=runRbp.txt

python all_experiments.py --ss_prefix "R" --bm_prefix "bp"