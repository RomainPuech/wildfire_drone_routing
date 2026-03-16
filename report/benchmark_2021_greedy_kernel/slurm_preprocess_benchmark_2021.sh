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

set -eo pipefail

# System profile scripts (e.g. /etc/profile.d/256term.sh) reference variables
# like $XTERM_VERSION that may be unset in a batch job. Disable -u while
# sourcing them, then re-enable it for the rest of the script.
set +u
source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf
set -u

export PATH="${HOME}/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

echo "Running preprocessing step..."
python "${PROJECT_ROOT}/preprocess_benchmark_2021.py"
echo "Preprocessing done."
