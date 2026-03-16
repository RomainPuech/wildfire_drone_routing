#!/bin/bash

#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_preprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

python -u preprocess_benchmark_2021.py
