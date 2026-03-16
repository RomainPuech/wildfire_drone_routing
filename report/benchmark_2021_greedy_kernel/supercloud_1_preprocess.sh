#!/bin/bash

#SBATCH --job-name=wf_preprocess
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

python -u preprocess_benchmark_2021.py
