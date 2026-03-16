#!/usr/bin/env bash
# Submit from project root. Runs the 3-budget array job, then a single wrap-up job
# that copies figures and builds the PDF only after all array tasks succeed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

ARRAY_JOB=$(sbatch "${SCRIPT_DIR}/slurm_reproduce_greedy_kernel.sh" | tee /dev/stderr | awk '{print $4}')
echo "Submitted array job ${ARRAY_JOB}"

sbatch --dependency=afterok:${ARRAY_JOB} "${SCRIPT_DIR}/slurm_reproduce_greedy_kernel_wrapup.sh"
echo "Submitted wrap-up job (runs after array completes)."
