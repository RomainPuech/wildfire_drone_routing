#!/bin/bash

#SBATCH --cpus-per-task=20            # Number of CPUs per task
#SBATCH --output=runKbm.txt


python all_experiments.py --ss_prefix "K" --bm_prefix "bm"