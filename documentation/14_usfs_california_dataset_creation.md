# USFS-Based California Dataset Creation (2020 / 2021)

## Overview

This document describes the **dataset creation process** for California wildfire ignition points derived from the **USFS ignition points** CSV (`USFS_ignition_points.csv`), filtered in two stages and overlaid on the WFPI (Wind-enhanced Fire Potential Index) grid used by the California2020Dataset. The same pipeline is applied for both **2020** and **2021** fires; the WFPI mask is shared (union-of-burnable, snow not excluded).

The process does **not** yet write scenario `.npy` files or a full dataset directory; it produces filtered ignition-point lists, mask assets, plots, and markdown reports. Writing scenarios in the same format as the original California 2020 dataset (see [04_california_2020_dataset.md](04_california_2020_dataset.md)) is a follow-on step.

---

## Data Sources

| Asset | Location | Description |
|-------|----------|-------------|
| **USFS ignition points** | `code/dataset_creation/nature_dataset_creation/data/USFS_ignition_points.csv` | Point-level wildfire records (year, state code, discovery datetime, lat/lon, size class, cause, etc.). State inferred from `UNIQFIREID` (e.g. `2021-CA...`). |
| **California boundary** | `code/dataset_creation/nature_dataset_creation/data/tl_2024_06_tract/tl_2024_06_tract.shp` | Census tracts for California; dissolved to a single state polygon for spatial filtering. |
| **Urban areas** | `code/dataset_creation/nature_dataset_creation/data/tl_2025_us_uac20/tl_2025_us_uac20.shp` | US Census Urban Area Criteria 2025; used to exclude fires inside urban areas. |
| **WFPI Day 2 forecast** | `.../data/2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA/` (and optionally `2021_...`) | Daily WFPI rasters (zip → GeoTIFF). Used to build the cropped-California grid, the validity mask, and the average map for plotting. |
| **Existing dataset** | `California2020Dataset/` | Provides grid dimensions (1309 × 805), crop transform, and (after creation) mask and `static_risk_wfpi_avg.npy` for overlay plots. |

---

## Pipeline Summary

1. **Stage 1 — Code, year, boundary, urban**
   - Filter CSV to the target year and California (`UNIQFIREID` starts with `YEAR-CA`), wildfire only (`FIRETYPECATEGORY == 'WF'`), valid date and coordinates.
   - Keep only points whose coordinates fall **inside** the California state polygon (spatial join).
   - Exclude points that fall inside US Census urban areas (spatial join).
2. **Stage 2 — WFPI grid and mask**
   - Convert each remaining ignition (lat/lon) to grid `(row, col)` on the cropped-California WFPI grid (1309 × 805, 1 km, Lambert Azimuthal Equal-Area).
   - Keep only fires for which `(row, col)` is in bounds and `mask[row, col] == 1`.

The **mask** defines “valid” cells: inside California, non-urban, and not permanently unburnable (see [Mask logic](#mask-logic) below).

---

## Stage 1: Code, Year, Boundary, Urban

### Criteria (applied in order)

| Criterion | Implementation |
|-----------|----------------|
| **Year** | `FIREYEAR == 2020` or `2021` |
| **State** | `UNIQFIREID.str.startswith("2020-CA")` or `"2021-CA"` (no STATE column in CSV). |
| **Fire type** | `FIRETYPECATEGORY == 'WF'` (wildfire only; excludes prescribed, etc.). |
| **Date** | `DISCOVERYDATETIME` non-null and parseable. |
| **Coordinates** | `LATDD83` and `LONGDD83` non-null. |
| **California boundary** | Point-in-polygon against dissolved California census tracts (EPSG:4326). Fires whose coordinates fall in Nevada/Oregon/etc. are dropped. |
| **Urban** | Point-in-polygon against US Census Urban Areas 2025; fires inside urban areas are dropped. |

### Scripts and outputs

- **Exploration / stage-1 only:**  
  `code/dataset_creation/nature_dataset_creation/explore_california_2020_ignitions.py`  
  `code/dataset_creation/nature_dataset_creation/explore_california_2021_ignitions.py`
- **Outputs:**  
  - `report/california_2020_ignition_points.png`, `report/california_2021_ignition_points.png` — stage-1 results (kept vs outside-CA vs urban).  
  - `report/california_2020_ignition_points.md`, `report/california_2021_ignition_points.md` — summary tables and criteria (later updated with stage-2 by the filter script).

### Typical stage-1 counts

| Year | Raw CA WF (from CSV) | Outside CA | Urban removed | **After stage-1** |
|------|----------------------|------------|---------------|-------------------|
| 2020 | 1,526                | 13         | 76            | **1,437**         |
| 2021 | 1,086                | 13         | 25            | **1,048**         |

---

## Stage 2: WFPI Grid and Mask

### Grid

- **Source:** WFPI Day 2 rasters cropped to a California bounding box (with 50 km buffer) in the WFPI native CRS (Lambert Azimuthal Equal-Area).
- **Size:** 1309 × 805 cells, 1 km resolution.
- **CRS:** Same as WFPI (Sphere Lambert Azimuthal). Lat/lon (EPSG:4326) is converted to this CRS for `(row, col)`.

A fire is **dropped** at stage-2 if:

- Its `(row, col)` is outside `[0, 1309)` or `[0, 805)`, or  
- `mask[row, col] != 1`.

### Mask logic

The mask used for filtering is **union-of-burnable, snow not excluded**:

- **Valid cell:** For that cell, on **at least one day** of the WFPI year (2020), the daily value is either **&lt; 249** (burnable) or **= 250** (snow). Snow (250) is treated as valid so that seasonally snow-covered cells are not permanently excluded.
- **Excluded cell:** The cell is **always** in the set {249, 251, 252, 253, 254, 255} (every day)—i.e. never burnable and never snow in the data.

So we **exclude only cells that are always unburnable**; we do **not** exclude a cell just because it is **ever** unburnable (e.g. on one winter day). That “union of burnable” rule is what expands the valid area compared to a single-day snapshot.

- **Mask file (current default):** `California2020Dataset/mask_union_burnable_no_snow_excluded.npy`
- **Alternative (snow excluded):** `mask_union_burnable.npy` — valid only where WFPI &lt; 249 on at least one day (250 treated as unburnable). Fewer valid cells and more fires dropped at stage-2.

The mask is also constrained by:

- **California boundary** — rasterized state polygon; cells outside set to 0.
- **Urban areas** — rasterized urban polygons; cells inside set to 0.
- **Largest connected component** — only the main California landmass is kept; small islands are zeroed out.

### Script and outputs

- **Stage-2 filter + plots:**  
  `code/dataset_creation/nature_dataset_creation/filter_wfpi_and_plot.py`
- **Reads:**  
  `USFS_ignition_points.csv`, California boundary, urban shapefile, WFPI zip dir (for transform), and the chosen mask (e.g. `mask_union_burnable_no_snow_excluded.npy`).
- **Outputs:**  
  - `report/california_2020_ignition_points_wfpi.png`, `report/california_2021_ignition_points_wfpi.png` — fires on WFPI average background (kept, WFPI-masked, stage-1 removed).  
  - Appends/updates the “Stage 2: WFPI Grid & Mask Filter” section in `report/california_2020_ignition_points.md` and `report/california_2021_ignition_points.md`.

### Typical stage-2 counts (mask: union burnable, snow not excluded)

| Year | After stage-1 | WFPI out-of-bounds | WFPI masked cell | **Final kept** |
|------|----------------|--------------------|------------------|----------------|
| 2020 | 1,437          | 0                  | 130              | **1,307**      |
| 2021 | 1,048          | 0                  | 63               | **985**        |

---

## Which WFPI Maps Are Used for the Mask?

The union-of-burnable mask used for the **California USFS 2020/2021** pipeline is built from **2020 WFPI Day 1** daily maps:

- **Source:** The **366** pre-cropped files in `California2020Dataset_Day1/` named `wfpi_day1_YYYYMMDD.npy` (e.g. `wfpi_day1_20200101.npy` … `wfpi_day1_20201231.npy`). Each file is shape `(1, 1309, 805)` and was produced from the 2020 Day 1 forecast zips in `data/2020_Wind-enhanced_Fire_Potential_Index_Forecast_1_DATA/` (USGS WFPI **Day 1** forecast).
- **Mask file:** `California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy` (249,255 valid cells). Build script: `code/dataset_creation/nature_dataset_creation/build_mask_union_burnable_day1.py`.

So: **masking for the new California USFS 2021 dataset (and USFS 2020 filtering) uses 2020 WFPI Day 1** daily maps, not Day 2.

### Day 1 vs Day 2

- **Choice:** Day 1 2020 is used for masking the USFS-based California 2021 (and 2020) ignition filtering so that the valid grid is defined by the same-day forecast product; the small extra coverage (47 more valid cells than Day 2) can include cells that are burnable on Day 1 but not on Day 2.
- **Difference:** A mask from Day 1 has **249,255** valid cells vs **249,208** for Day 2 (47 more). Over 366 days, 54 cells are ever burnable in Day 1 but not in Day 2; after CA boundary, urban mask, and largest component, 47 of those remain. Fire counts with the current ignition data are unchanged (1,307 for 2020, 985 for 2021) because none of the stage-1 fires fall in those 47 cells.

---

## Mask Generation (Reference)

The union-of-burnable mask is **not** built by the main filter script; it is built once and saved under `California2020Dataset/`.

### Steps (conceptually)

1. Load all daily WFPI rasters for the year (e.g. 366 days from `wfpi_YYYYMMDD.npy` in the dataset, or from the original zip rasters cropped to California).
2. For each cell, set **ever_burnable** if, on any day, value &lt; 249 **or** value = 250 (for the no-snow-excluded mask).
3. **always_unburnable** = ¬ ever_burnable.
4. Start with **mask = ever_burnable** (float 0/1).
5. Set mask = 0 outside the California boundary (rasterize state polygon).
6. Set mask = 0 inside urban areas (rasterize urban polygons).
7. Keep only the largest connected component of mask == 1; set all others to 0.
8. Save as e.g. `mask_union_burnable_no_snow_excluded.npy`.

Using **2021** WFPI daily data instead of 2020 yields the **same** mask (same valid cell count and shape), because the “always unburnable” set is determined by static geography (water, desert, etc.), not year-specific weather.

---

## Single-Script Generation

The **whole 2021 (and 2020) California USFS dataset** filtering and **all plots used in the report** can be generated by running one script:

```bash
python code/dataset_creation/nature_dataset_creation/run_california_usfs_dataset.py
```

It runs in order:

1. **Mask:** Builds `mask_union_burnable_no_snow_excluded_day1.npy` if missing (via `build_mask_union_burnable_day1.py`).
2. **Stage-1:** `explore_california_2020_ignitions.py` and `explore_california_2021_ignitions.py` → `report/california_2020_ignition_points.png`, `.md`, and `california_2021_ignition_points.png`, `.md`.
3. **Stage-2:** `filter_wfpi_and_plot.py` → `report/california_2020_ignition_points_wfpi.png`, `california_2021_ignition_points_wfpi.png`, and appends the Stage-2 section to both markdown reports.

Options:

- `--years 2021` — run only the 2021 stage-1 exploration (stage-2 still runs for both years).
- `--skip-mask` — do not build the Day 1 mask even if missing (fails if the mask file is not present).

---

## File Summary

| File | Purpose |
|------|--------|
| `run_california_usfs_dataset.py` | **Single entry point:** mask (if missing) + stage-1 + stage-2 for 2020 and 2021. |
| `data/USFS_ignition_points.csv` | Input ignition points. |
| `explore_california_2020_ignitions.py` / `explore_california_2021_ignitions.py` | Stage-1 filter + stage-1-only plots and markdown. |
| `filter_wfpi_and_plot.py` | Stage-2 filter (WFPI grid + mask), WFPI-overlay plots, and stage-2 section in markdown reports. |
| `California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy` | Default mask (2020 Day 1, union burnable, snow not excluded). Built by `build_mask_union_burnable_day1.py`. |
| `California2020Dataset/mask_union_burnable_no_snow_excluded.npy` | Day 2 variant (union burnable, snow not excluded). |
| `California2020Dataset/mask_union_burnable.npy` | Alternative mask (union burnable, snow excluded). |
| `report/california_2020_ignition_points.md` / `..._2021_...` | Human-readable filter criteria and counts (stage-1 and stage-2). |
| `report/california_2020_ignition_points.png` / `..._2021_...` | Stage-1 map (kept / outside CA / urban). |
| `report/california_2020_ignition_points_wfpi.png` / `..._2021_...` | Stage-2 map on WFPI average (kept / WFPI-masked / stage-1 removed). |

---

## Next Steps (Not Yet Done)

- **Write scenario files:** For each fire that passes stage-2, compute `(row, col, start_timestep)` and save in the same ignition-point-only `.npy` format as the original California 2020 dataset (see [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md)).
- **Dataset directory:** Optionally create `California2021Dataset/` (and/or a 2020 USFS-based variant) with mask, WFPI maps (if 2021 WFPI is used), `scenarii/`, config, and `dataset_summary.json`.

---

## Related Documentation

- [04_california_2020_dataset.md](04_california_2020_dataset.md) — Original California 2020 dataset (FPA_FOD source, structure, WFPI usage).
- [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md) — Ignition-point-only scenario format.
