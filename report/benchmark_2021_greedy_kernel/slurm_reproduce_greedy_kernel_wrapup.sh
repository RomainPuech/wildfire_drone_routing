#!/bin/bash -l
#SBATCH -p sched_mit_sloan_batch
#SBATCH --job-name=wf_greedy_wrapup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

# Triggered by afterany on the array job: runs even if some tasks failed.
# Copies whichever budget figures exist and rebuilds the PDF from those.
#
# #!/bin/bash -l  (login shell) ensures MODULEPATH is initialized.
# -e intentionally omitted so partial failures are logged, not silently aborted.

source /etc/profile
module load community-modules
module load miniforge/25.11.0-0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wf

export PATH="${HOME}/.local/bin:$PATH"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
REPORT_DIR="${PROJECT_ROOT}/report"
SUBREPORT_DIR="${REPORT_DIR}/benchmark_2021_greedy_kernel"
LOG_DIR="${PROJECT_ROOT}/California2021Dataset/logs"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

FAILED=0

# ── Pandoc check ────────────────────────────────────────────────────────────
if ! command -v pandoc &>/dev/null; then
    echo "ERROR: pandoc not found in PATH. PDF will not be built." >&2
    echo "       Install it with: conda install -c conda-forge pandoc" >&2
    echo "       or:              pip install pandoc" >&2
    FAILED=1
fi

# ── Copy figures (log missing ones, don't abort) ─────────────────────────────
echo "Copying figures into subreport folder..."

copy_figure() {
    local src="$1"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${SUBREPORT_DIR}/"
        echo "  Copied: $(basename "${src}")"
    else
        echo "  WARNING: figure not found (budget run may have failed): ${src}" >&2
        FAILED=1
    fi
}

copy_figure "${REPORT_DIR}/benchmark_fire_locations_budget_2021.png"

for BUDGET in 20 100 500; do
    JSON="${LOG_DIR}/sensor_alloc_GaussianBudget${BUDGET}M_StationMax_261x161_mean.json"
    if [[ ! -f "${JSON}" ]]; then
        echo "  WARNING: placement JSON missing for ${BUDGET}M — skipping figures for this budget." >&2
        FAILED=1
        continue
    fi
    copy_figure "${REPORT_DIR}/california_2021_sensor_clusters_greedy_${BUDGET}M.png"
    copy_figure "${REPORT_DIR}/california_2021_sensor_clusters_opt_greedy_${BUDGET}M.png"
done

# ── Build PDF ────────────────────────────────────────────────────────────────
if command -v pandoc &>/dev/null; then
    echo "Rebuilding PDF..."
    if (cd "${SUBREPORT_DIR}" && pandoc "benchmark_2021_greedy_kernel.md" -o "benchmark_2021_greedy_kernel.pdf"); then
        echo "PDF built: ${SUBREPORT_DIR}/benchmark_2021_greedy_kernel.pdf"
    else
        echo "ERROR: pandoc failed to build PDF." >&2
        FAILED=1
    fi
fi

# ── Final status ─────────────────────────────────────────────────────────────
if [[ "${FAILED}" -eq 0 ]]; then
    echo "Wrap-up complete. All figures copied and PDF built."
else
    echo "Wrap-up finished with warnings/errors (see above). Check array task logs in logs/." >&2
    exit 1
fi
