#!/usr/bin/env bash
# Submit: cache clear → placement (array) → routing (array, depends on placement).
# Must run from the repo root so SLURM_SUBMIT_DIR points at the project (batch
# scripts use SLURM_SUBMIT_DIR, not BASH_SOURCE — Slurm runs a spool copy).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

J1=$(sbatch --parsable "${SCRIPT_DIR}/supercloud_clear_greedy_uniform_cache.sh")
echo "Submitted clear cache: ${J1}"

J2=$(sbatch --parsable --dependency=afterok:"${J1}" \
  "${SCRIPT_DIR}/supercloud_2_greedy_uniform_placement_array.sh")
echo "Submitted placement array (after clear): ${J2}"

J3=$(sbatch --parsable --dependency=afterok:"${J2}" \
  "${SCRIPT_DIR}/supercloud_3_greedy_uniform_routing_array.sh")
echo "Submitted routing array (after all placement tasks): ${J3}"
echo "Done. Monitor: squeue -u \$USER"
