#!/usr/bin/env bash
# Submit from the project root:
#   bash report/benchmark_2021_greedy_kernel/submit_greedy_benchmark.sh
#
# Job chain:
#   1. preprocess  – compute shared .npy files once (serial, fast)
#   2. array[0-2]  – run 20M / 100M / 500M placements in parallel (depends on 1)
#   3. wrapup      – copy figures + build PDF (runs after array, even on partial failure)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

PREPROCESS_JOB=$(sbatch "${SCRIPT_DIR}/slurm_preprocess_benchmark_2021.sh" | awk '{print $4}')
echo "Submitted preprocessing job ${PREPROCESS_JOB}"

ARRAY_JOB=$(sbatch --dependency=afterok:${PREPROCESS_JOB} "${SCRIPT_DIR}/slurm_reproduce_greedy_kernel.sh" | awk '{print $4}')
echo "Submitted array job ${ARRAY_JOB} (depends on preprocessing ${PREPROCESS_JOB})"

WRAPUP_JOB=$(sbatch --dependency=afterany:${ARRAY_JOB} "${SCRIPT_DIR}/slurm_reproduce_greedy_kernel_wrapup.sh" | awk '{print $4}')
echo "Submitted wrap-up job ${WRAPUP_JOB} (runs after array finishes, even on partial failure)"
