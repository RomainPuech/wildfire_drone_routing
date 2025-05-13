#!/bin/bash
#SBATCH --job-name="burnmap_predictor_test"
#SBATCH --mem=32G
#SBATCH --partition=mit_quicktest
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -n 4                      # 4 CPU cores (adjust as needed)
#SBATCH --time=00:30:00           # 30 minutes runtime
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=puech@mit.edu
#SBATCH --gres=gpu:1              # Request 1 GPU

# === 1. Load your environment ===
source /etc/profile
module load anaconda



# === 2. Run your script ===
python test_burnmap_predictor.py
