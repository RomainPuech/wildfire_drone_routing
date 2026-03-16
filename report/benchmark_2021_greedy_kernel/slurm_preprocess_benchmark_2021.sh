#!/usr/bin/env bash
#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_preprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Runs before the budget array job so that shared .npy files are written
# exactly once, avoiding race conditions between parallel workers.

# Load environment first (no strict mode yet — system profile scripts
# reference unset vars and may return non-zero; matches the known-working example).
source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

# Strict mode on now that the environment is fully set up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/logs"

echo "Running preprocessing step..."
python "${PROJECT_ROOT}/preprocess_benchmark_2021.py"
echo "Preprocessing done."
