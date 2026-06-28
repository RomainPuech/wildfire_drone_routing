#!/bin/bash
# Remove cached greedy-uniform placement, yearly routing logs, and rescaled
# Pyrologix/mask derivatives so the next run uses the current California2021Dataset
# files on disk.

#SBATCH --job-name=wf_clear_greedy_cache
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Slurm executes a spool copy of this script; BASH_SOURCE is not under the repo.
# Submit with: cd /path/to/wildfire_drone_routing && sbatch ...
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
DS="${PROJECT_ROOT}/California2021Dataset"

echo "Clearing greedy-uniform sensor caches and yearly routing logs under ${DS}/logs ..."
rm -f "${DS}/logs/sensor_alloc_GaussianBudget"*M_StationMaxGreedyUniform*.json
rm -f "${DS}/logs/routing_yearly_"*.json

echo "Removing rescaled mask / Pyrologix derivatives (rebuilt on next run) ..."
rm -f "${DS}/mask_rescaled_261x161_7substeps.npy"
rm -f "${DS}/static_risk_pyrologix_mean_rescaled_261x161_7substeps.npy"
rm -f "${DS}/static_risk_pyrologix_mean_routing_rescaled_261x161_7substeps.npy"

echo "Done."
