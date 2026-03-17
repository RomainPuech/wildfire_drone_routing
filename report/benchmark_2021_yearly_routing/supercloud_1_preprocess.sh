#!/bin/bash

#SBATCH --job-name=wf_2021_preprocess
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

python -u preprocess_benchmark_2021.py
