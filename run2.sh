#!/bin/bash

#SBATCH --cpus-per-task=1            # Number of CPUs per task


# python preprocess.py
python cleandataset.py
