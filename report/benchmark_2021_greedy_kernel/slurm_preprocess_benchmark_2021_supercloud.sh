#!/usr/bin/env bash
#SBATCH --job-name=wf_preprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

# Runs before the budget array job so that shared .npy files are written
# exactly once, avoiding race conditions between parallel workers.

set -euo pipefail

source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export PATH="${HOME}/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

echo "Running preprocessing step..."
python "${PROJECT_ROOT}/preprocess_benchmark_2021.py"
echo "Preprocessing done."
