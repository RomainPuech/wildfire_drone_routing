# Paper figures — onboarding for coding agents

Short reference for iterating on **Nature Wildfires** manuscript figures: where assets live, which scripts regenerate them, and common pitfalls.

## Anchor paths

| What | Path |
|------|------|
| Main LaTeX | `paper/Nature_Wildfires/sn-article.tex` |
| Included raster figures | `paper/Nature_Wildfires/Figures/*.png` |
| PDF build (from repo) | `python paper/compile_pdf.py` — compiles in `Nature_Wildfires/`, copies `sn-article.pdf` to `paper/sn-article.pdf` |

LaTeX references figures **only** under `paper/Nature_Wildfires/Figures/`. After regenerating elsewhere, **copy or symlink** into that folder (or change `\includegraphics` paths—prefer updating assets in `Figures/` to avoid breaking the template).

---

## Figure → asset → code (single table)

| Manuscript area | LaTeX `\label` | PNG(s) in `Figures/` | Regeneration |
|-----------------|----------------|----------------------|--------------|
| Framework schematic | `fig:framework` | `High (7).png` | **No script in repo** — treat as external design asset. |
| California dataset (4 panels) | `fig:data` | `fig01_pyrologix_california_boundary.png` … `fig04_pyrologix_components_ge_9km2.png` | `code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py` — plots via `code/displays.py` (`plot_pyrologix_valid_region`, …). Details: `paper/figure2/README.md`. |
| Detection frontier | `fig:frontier` | `frontier.png` | **Numeric inputs:** `compute_frontier_detection_curves()` in `paper/final_report/generate_final_report.py` (uses `paper/final_report/csv/` benchmark CSVs + placement bundle). **Plot script:** docstring mentions `paper/Nature_Wildfires/make_figure3_frontier.py` — **file is missing**; add a small matplotlib script there or next to `generate_final_report.py` if you need reproducible PNGs. |
| Infrastructure maps (2×2 composite) | `fig:placement` | **`placement_composite.png`** (replaces the four separate `placement_greedy_*.png` in the build) | `conda run -n wf python paper/figure4/generate_placement_composite_figure.py` (required for the CA state border; system `python` has no geospatial stack). **Single panels** (optional): `visualize_sensor_placement_2021.py` — `paper/final_report/docs/reproduce_placement_plots.sh` (conda `wf`, JSONs under `paper/final_report/placement_data/`). |
| Cost sensitivity (line plot) | `fig:breakeven_costsensitivity` | **`breakeven_costsensitivity_lines.png`** (the published Figure 5; replaces the old 2×3 `breakeven_costsensitivity_composite.png` maps) | `python paper/figure5bis/make_figure5bis_breakeven_lines.py` — reads all JSONs from `paper/breakeven_report/…/placement_logs/`, no extra deps. Two panels (\$20M, \$50M): device counts vs ground-sensor unit cost, with a right-axis "% reachable fires" computed over the pooled 2021–2024 fires (n = 3,693). |
| Cost sensitivity maps (legacy) | — | `breakeven_costsensitivity_composite.png` | `paper/breakeven_figure/generate_breakeven_cost_sensitivity_figures.py` — see `paper/breakeven_figure/README.md`. No longer included in the manuscript. |
| ALERTCalifornia maps | `fig:alertcalifornia` | `alertcalifornia_coverage_composite.png` | Composite: `conda run -n wf python paper/figure6/generate_alertcalifornia_composite_figure.py`. Legacy single-radius maps: `code/plot_alertcalifornia_coverage.py --radius <km> --out <path>`. |
| Rolling horizon (Methods) | `fig:rollinghorizon` | `rollinghorizon.png` | **No generator in repo** — likely manual vector/slide export. |

**Tables-only floats** (no PNG): e.g. `tab:detection`, `tab:alertcalifornia`, large Methods tables — edit `sn-article.tex` only.

---

## Placement assets

The manuscript’s **published** map is the composite: `Figures/placement_composite.png` from `paper/figure4/generate_placement_composite_figure.py`.

`visualize_sensor_placement_2021.py` still writes e.g. `california_2021_sensor_clusters_opt_greedy_uniform_20M.png` — to match legacy filenames, **rename** to `placement_greedy_20M.png` if you ship individual panels.

---

## Environment and dependencies

- **Placement / Pyrologix maps:** conda env **`wf`** with **geopandas**, **rasterio**, matplotlib, numpy, etc. (`reproduce_placement_plots.sh` checks this.)
- **Breakeven composite:** follow `paper/breakeven_figure/README.md` (paths to dataset / export dirs are script arguments).
- **Dataset explainer figures:** same scientific stack as placement; script prints clear errors if `California2021Dataset/` or ancillary paths are missing.

---

## Shared styling (change once, affects multiple figures)

- **`code/displays.py`** — Pyrologix colormap, legends, `_pyrologix_publication_rc()`, `plot_pyrologix_valid_region`, fire overlays.
- **`visualize_sensor_placement_2021.py`** — operational-scale cluster maps; should stay visually aligned with `displays.py` where captions mention “same style as Figure X”.

When tuning fonts, DPI, colorbar inset, or legend layout, grep for the figure’s output basename and the script that references it, then check whether logic lives in `displays.py` or the dedicated script.

---

## Final style lock (Figure 4 / Figure 5)

Use this as the default visual target unless explicitly asked otherwise.

- **Environment:** regenerate Figures 4 and 5 with `conda run -n wf python ...` (geospatial stack required for California outline).
- **California outline:** draw in publication style (`#444444`, 1.0 pt), visible above map overlays.
- **Panel sizing / spacing:** prefer large map panels, aggressive whitespace reduction, and explicit `GridSpec` geometry.
- **Figure 4 (`placement_composite.png`):**
  - 2×2 maps + dedicated colorbar column.
  - Discoverable/unreachable annotation callouts in rounded white boxes near upper-right of each panel.
  - Discoverable ignitions use **dots** (not crosses).
  - Charging-station drone-count labels on map are disabled.
  - Shared legend is boxed (same white/gray border style as callouts) and spaced below subtitles.
- **Figure 5 (`breakeven_costsensitivity_composite.png`):**
  - 2×3 maps rendered with left-aligned rows via dedicated colorbar column.
  - Subtitles are split across two lines after the comma.
  - Per-panel counts are top-right callouts in rounded white boxes.
  - Shared legend is boxed and typography-scaled to match Figure 4.
  - Keep explicit vertical spacing between rows, subtitles, and legend to avoid overlap.

---

## Workflow for an agent

1. Identify the **PNG basename** from `sn-article.tex` (`\includegraphics{Figures/...}`).
2. Use the table above to open the **generator script** (or note “manual / missing”).
3. Prefer **small, local edits**: match existing argparse, paths, and matplotlib patterns in that file.
4. Write output to `paper/Nature_Wildfires/Figures/` (or your chosen out path + copy).
5. Run `python paper/compile_pdf.py` to confirm the PDF builds (TeX Live / MacTeX required).

---

## Known gaps (avoid spinning)

- **`make_figure3_frontier.py`** — referenced in `generate_final_report.py` comments but **not committed**; frontier PNG may be hand-built from CSVs.
- **`rollinghorizon.png`** — no Python source in tree.
- **`High (7).png`** — framework overview; not generated from this codebase.

If the task is “reproducible frontier only,” implement plotting using **`compute_frontier_detection_curves()`** (import from `generate_final_report` or duplicate the minimal CSV reads) rather than inventing new data paths.
