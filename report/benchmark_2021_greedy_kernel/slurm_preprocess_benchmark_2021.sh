#!/bin/bash -l
#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_preprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Runs before the budget array job so that shared .npy files are written
# exactly once, avoiding race conditions between parallel workers.
#
# #!/bin/bash -l  (login shell) ensures MODULEPATH is initialized on compute
# nodes, matching the behavior of an interactive login session.
# PROJECT_ROOT uses SLURM_SUBMIT_DIR (set by Slurm to the sbatch call dir)
# rather than BASH_SOURCE, because Slurm copies scripts to /var/spool/... .

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

# Strict mode on now that the environment is fully set up.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

echo "Running preprocessing step..."
python "${PROJECT_ROOT}/preprocess_benchmark_2021.py"
echo "Preprocessing done."
