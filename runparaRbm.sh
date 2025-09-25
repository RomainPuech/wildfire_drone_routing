#!/bin/bash

#SBATCH --cpus-per-task=32            # Number of CPUs per task



python all_experiments_parallel.py --ss_prefix R --bm_prefix bm