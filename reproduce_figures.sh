#!/usr/bin/env bash
#
# reproduce_figures.sh — regenerate every code-generated figure in the paper.
#
# Outputs (relative to repo root):
#   paper/Nature_Wildfires/Figures/pdf/*.pdf   high-quality vector figures (600 dpi where raster)
#   paper/Nature_Wildfires/Figures/*.png       manuscript-embedded raster panels
#
# Figures produced:
#   fig01-04  Fig. 6 dataset-construction panels  (generate_paper_2021_dataset_explainer.py)
#   frontier                                       (make_figure3_frontier.py)
#   placement_composite                            (figure4/generate_placement_composite_figure.py)
#   breakeven_costsensitivity_lines                (figure5bis/make_figure5bis_breakeven_lines.py)
#   alertcalifornia_coverage_composite             (figure6/generate_alertcalifornia_composite_figure.py)
#
# Requirements:
#   * A Python environment with the geospatial stack (geopandas, rasterio, pyproj,
#     shapely, matplotlib, pandas). On Linux use the pinned conda env:
#       conda env create -f environment.yml && conda activate juliaenv
#     Override the interpreter with:  PYTHON=/path/to/python ./reproduce_figures.sh
#   * Committed Tier-1 data (config/mask/static-risk/scenarii, placement JSONs, caches)
#     is sufficient for frontier, placement, breakeven, and alertcalifornia.
#   * fig01-04 additionally need the WFPI archives, USFS ignition CSV, TIGER shapefiles,
#     and California2020Dataset_Day1 (see README "Data" — available on HuggingFace).
#
# Usage:
#   ./reproduce_figures.sh            # all figures
#   PYTHON=python3.10 ./reproduce_figures.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
FIGDIR="$REPO_ROOT/paper/Nature_Wildfires/Figures"
PDFDIR="$FIGDIR/pdf"
mkdir -p "$FIGDIR" "$PDFDIR"

echo "==> Using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "==> Output PDFs -> $PDFDIR"
echo "==> Output PNGs -> $FIGDIR"

run() { echo; echo "==> $1"; shift; "$@"; }

# ---------------------------------------------------------------------------
# Fig. 6 panels (fig01-04): PDF (600 dpi) + PNG (600 dpi).
# The explainer keys output extension/dir/dpi off FIG_EXT / FIG_OUT_DIR / FIG_DPI.
# ---------------------------------------------------------------------------
EXPLAINER="code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py"
run "fig01-04 (PDF, 600 dpi)" env FIG_EXT=pdf FIG_OUT_DIR="$PDFDIR" FIG_DPI=600 "$PYTHON" "$EXPLAINER"
run "fig01-04 (PNG, 600 dpi)" env FIG_EXT=png FIG_OUT_DIR="$FIGDIR" FIG_DPI=600 "$PYTHON" "$EXPLAINER"

# ---------------------------------------------------------------------------
# Frontier (Fig. 2): FIG_OUT selects the vector target; default writes the PNG.
# ---------------------------------------------------------------------------
run "frontier (PDF)" env FIG_OUT="$PDFDIR/frontier.pdf" "$PYTHON" paper/Nature_Wildfires/make_figure3_frontier.py
run "frontier (PNG)" "$PYTHON" paper/Nature_Wildfires/make_figure3_frontier.py

# ---------------------------------------------------------------------------
# Placement composite (Fig. 3): --out selects the target (extension picks format).
# ---------------------------------------------------------------------------
run "placement_composite (PDF)" "$PYTHON" paper/figure4/generate_placement_composite_figure.py --out "$PDFDIR/placement_composite.pdf"
run "placement_composite (PNG)" "$PYTHON" paper/figure4/generate_placement_composite_figure.py --out "$FIGDIR/placement_composite.png"

# ---------------------------------------------------------------------------
# Break-even cost sensitivity (Fig. 4): FIG_OUT selects the vector target.
# ---------------------------------------------------------------------------
run "breakeven (PDF)" env FIG_OUT="$PDFDIR/breakeven_costsensitivity_lines.pdf" "$PYTHON" paper/figure5bis/make_figure5bis_breakeven_lines.py
run "breakeven (PNG)" "$PYTHON" paper/figure5bis/make_figure5bis_breakeven_lines.py

# ---------------------------------------------------------------------------
# ALERTCalifornia coverage composite (Fig. 5): defaults to California2024Dataset.
# ---------------------------------------------------------------------------
run "alertcalifornia (PDF)" "$PYTHON" paper/figure6/generate_alertcalifornia_composite_figure.py --out "$PDFDIR/alertcalifornia_coverage_composite.pdf"
run "alertcalifornia (PNG)" "$PYTHON" paper/figure6/generate_alertcalifornia_composite_figure.py --out "$FIGDIR/alertcalifornia_coverage_composite.png"

echo
echo "==> Done. Generated PDFs:"
ls -1 "$PDFDIR"/*.pdf
