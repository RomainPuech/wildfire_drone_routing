# Breakeven / cost-sensitivity figure (Nature manuscript)

**Always** run the script below with the **`wf` conda environment** (`conda run -n wf python …`). A plain `python` from the shell may lack geopandas/rasterio; the California state outline and other outputs can be incomplete or differ from the manuscript.

Regenerates **`Figures/breakeven_costsensitivity_composite.png`**: one **2×3** set of **operational (5 km pooled)** maps with a **single** shared ignition-probability scale, **one** shared symbol legend (ground sensors, charging stations, drone-reach areas; **no** per-station drone counts on the map or in the legend), and **per-panel** text boxes listing total ground sensors, stations, and drones. **Benchmark ignition scatter is not drawn.**

Optional **`--panels`** also writes the six legacy `breakeven_costsensitivity_*.png` tiles (each with its own legend/colorbar, as produced by `visualize_sensor_placement_2021.py`).

## Aesthetic alignment (Figure 2 / `plot_pyrologix_valid_region`)

| Element | Composite output |
|--------|---------------------|
| Matplotlib rc | Same serif stack as Figure 2 / Figure 4 (Latin Modern when installed) |
| Figure background | **White**; `savefig(..., pad_inches=0.22, facecolor="white")` |
| Ignition scale | **One** vertical colorbar for all six maps (dedicated colorbar axis) |
| Legend | **One** boxed `fig.legend` under the grid (no per-panel legend) |
| California outline | **`#444444`, 1.0 pt** (after draw, same as Figure 2 style) |

## Current style requirements (locked)

- **Subtitles:** split in two lines after comma (e.g., `50M USD total,` newline `10k USD per sensor`).
- **Panel geometry:** rows are left-aligned (top and bottom maps share the same left boundary); colorbar uses its own narrow column.
- **Whitespace:** reduced map margins and inter-panel spacing while preserving non-overlapping text.
- **Annotations:** top-right white rounded callouts (`Sensors`, `Stations`, `Drones`) with enlarged font.
- **Legend:** large text/markers, white background, light gray border, and explicit vertical gap from lower-row subtitles.
- **Figure consistency:** typography and callout/legend styling are aligned with Figure 4.

## Run

From the **repository root**, with `California2021Dataset/` present. **Use conda `wf` only** (geopandas/rasterio for the CA outline; `opencv` only needed if you pass **`--panels`**):

```bash
conda run -n wf python paper/breakeven_figure/generate_breakeven_cost_sensitivity_figures.py
```

Defaults read JSONs from `paper/breakeven_report/breakeven_sensor_cost_export/placement_logs/` and write the composite under `paper/Nature_Wildfires/Figures/`.

For **panel (e)** (50M budget, **15k USD** per sensor), the script looks for the first existing file among:

- `…_mean_breakeven_50M_cs0p015.json`
- `…_mean_breakeven_50M_cs0p015_filt80.json`
- `…_mean_thresh50M_m15.json` (same **0.015 MUSD** cost, millis-style tag used in some exports)

```text
--dataset-dir PATH
--json-dir PATH
--out-dir PATH
--panels              # also write six separate PNGs
--no-colorbar-panels  # with --panels: omit per-panel colorbar
```

The Nature `sn-article.tex` figure includes **`breakeven_costsensitivity_composite.png`** as a single `\\includegraphics`.
