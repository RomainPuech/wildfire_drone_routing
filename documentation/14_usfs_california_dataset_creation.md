# USFS-Based California Dataset Creation (2020 / 2021)

## Overview

This document describes the **dataset creation process** for California wildfire ignition points derived from the **USFS ignition points** CSV (`USFS_ignition_points.csv`), filtered in two stages and overlaid on the WFPI (Wind-enhanced Fire Potential Index) grid used by the California2020Dataset. The same pipeline is applied for both **2020** and **2021** fires; the WFPI mask is shared (union-of-burnable, snow not excluded).

The pipeline produces filtered ignition-point lists, mask assets, plots, and markdown reports. The **California 2021 dataset** (`California2021Dataset/`) is written by `create_california_2021_dataset.py` in the same format as the California 2020 dataset (see [04_california_2020_dataset.md](04_california_2020_dataset.md)).

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
3. **2021 only — Exclude missing WFPI zip dates**
   - Some 2021 calendar days have no WFPI zip (27 days missing). Fires whose **discovery date** falls on one of these days are **excluded from the dataset** so that every scenario has real (or nearest-neighbour) WFPI data for its day. They are still shown on the WFPI-overlay plot as a separate category ("Excluded — missing WFPI zip date").

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

| Year | After stage-1 | WFPI out-of-bounds | WFPI masked cell | Excluded (missing WFPI date, 2021 only) | **Final kept (in dataset)** |
|------|----------------|--------------------|------------------|----------------------------------------|-----------------------------|
| 2020 | 1,437          | 0                  | 130              | —                                       | **1,307**                   |
| 2021 | 1,048          | 0                  | 63               | 53                                      | **932**                     |

For 2021, the 27 days with no 2021 WFPI zip cause 53 fires to be excluded; the filter plot marks them separately (purple diamonds). See [Exclusion: missing 2021 WFPI zip dates](#exclusion-missing-2021-wfpi-zip-dates).

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

## Exclusion: missing 2021 WFPI zip dates

For **2021**, not every calendar day has a WFPI zip in the data directories: 27 days are missing (e.g. 20210104, 20210119, …). Fires whose **discovery date** is one of these days would require a nearest-neighbour WFPI map for that day when building the time-aware burn map; to avoid using the wrong calendar day we **exclude these fires from the California 2021 dataset**.

- **Where applied:**  
  - `filter_wfpi_and_plot.py` — for 2021, "kept" is split into **932 in-dataset** and **53 excluded**; the plot shows the excluded fires as purple diamonds ("Excluded — missing WFPI zip date").  
  - `create_california_2021_dataset.py` — only the 932 fires (not on a missing date) get a scenario file and config entries.  
  - `remove_2021_fires_missing_wfpi_dates.py` — one-time script to remove those 53 from an already-built dataset (delete scenario files, strip config entries, update `dataset_summary.json`).
- **Plot:** `report/california_2021_fires_missing_wfpi_dates.png` shows only the 53 excluded fires on the 2021 WFPI average (for reference).
- **Counts:** 2021 stage-2 yields 985 fires in valid mask; 53 fall on missing zip dates → **932 fires in dataset**.

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
| `create_california_2021_dataset.py` | Builds `California2021Dataset/` and `California2021Dataset_Day1/` (scenarii, config, WFPI daily/yearly/avg maps). |
| `data/USFS_ignition_points.csv` | Input ignition points. |
| `explore_california_2020_ignitions.py` / `explore_california_2021_ignitions.py` | Stage-1 filter + stage-1-only plots and markdown. |
| `filter_wfpi_and_plot.py` | Stage-2 filter (WFPI grid + mask), WFPI-overlay plots, and stage-2 section in markdown reports. |
| `California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy` | Default mask (2020 Day 1, union burnable, snow not excluded). Built by `build_mask_union_burnable_day1.py`. |
| `California2020Dataset/mask_union_burnable_no_snow_excluded.npy` | Day 2 variant (union burnable, snow not excluded). |
| `California2020Dataset/mask_union_burnable.npy` | Alternative mask (union burnable, snow excluded). |
| `report/california_2020_ignition_points.md` / `..._2021_...` | Human-readable filter criteria and counts (stage-1 and stage-2). |
| `report/california_2020_ignition_points.png` / `..._2021_...` | Stage-1 map (kept / outside CA / urban). |
| `report/california_2020_ignition_points_wfpi.png` / `..._2021_...` | Stage-2 map on WFPI average (kept / WFPI-masked / stage-1 removed; 2021 also shows excluded missing-WFPI-date fires). |
| `report/california_2021_fires_missing_wfpi_dates.png` | 2021 fires on missing WFPI zip dates only (53 fires). |
| `plot_2021_fires_on_pyrologix.py` | 2021 kept fires (932) on Pyrologix burn map; masked zones white. |
| `plot_2021_fires_missing_wfpi_dates.py` | Plot of the 53 fires excluded due to missing WFPI date. |
| `remove_2021_fires_missing_wfpi_dates.py` | One-time: remove those 53 from an existing `California2021Dataset/` (scenarii, config, summary). |

---

## California 2021 Dataset (`California2021Dataset/`)

The **California 2021 dataset** is built in the same format as the California 2020 dataset for use by the benchmarking library.

**Creation script (run from project root):**

```bash
python code/dataset_creation/nature_dataset_creation/create_california_2021_dataset.py
```

**What it does:**

1. **Mask:** Copies `California2020Dataset/mask_union_burnable_no_snow_excluded_day1.npy` to `California2021Dataset/mask.npy` (same D1 2020 mask used for filtering).
2. **Scenarii:** Runs stage-1 + stage-2 filter for 2021 USFS fires; for each kept fire saves one ignition-point scenario `FireName_UNIQFIREID_scenario1.npy` in `scenarii/`.
3. **Config:** Writes `config_california_2021.json` with `offset_<base>`, `date_<base>`, `time_<base>` per scenario (discovery date/time for simulation start and logs).
4. **WFPI daily files:** Fills all 365 days of 2021 from 2021 D2 and D1 zip data into `California2021Dataset/wfpi_YYYYMMDD.npy` and `California2021Dataset_Day1/wfpi_day1_YYYYMMDD.npy` (nearest-neighbour fallback for missing zips).
5. **Yearly burn map:** Builds `static_risk_wfpi_yearly.npy` (730 frames): before 10 am = D2 from day before, after 10 am = D1 same day (see [04_california_2020_dataset.md](04_california_2020_dataset.md)).
6. **Averaged maps:**  
   - `static_risk_wfpi_avg.npy` — mean over the yearly frames, **excluding values ≥249** (invalid/special); cells with no valid day get 0.  
   - `static_risk_wfpi_burn_at_least_once.npy` — for each cell, probability of “burning at least once” in the year: rescale daily values to probability in [0,1] (value/248 for &lt;249, else 0), compute 1 − ∏(1−p_i), then remap to 0–248.

**Dataset layout:** Same as California 2020: `mask.npy`, `wfpi_YYYYMMDD.npy`, `static_risk_wfpi_yearly.npy`, `static_risk_wfpi_avg.npy`, `scenarii/`, `config_california_2021.json`, `dataset_summary.json`.  
Additionally, `static_risk_pyrologix.npy` (copy of Pyrologix) is included so the dataset is **self-contained** — the benchmark does not need to reach into `California2020Dataset/` for the risk map. Only fires **not** on a missing 2021 WFPI zip date are included (932 scenarios).

---

## Burn Map Selection for California 2021

Based on the analysis in [15_2021_risk_map_comparison.md](15_2021_risk_map_comparison.md), the
**recommended burn map for sensor placement AND drone routing is Ignition Probability (Pyrologix)**.

A self-contained copy lives at `California2021Dataset/static_risk_pyrologix.npy` (shape 1 × 1309 × 805,
values 0–255). This is identical to `California2020Dataset/static_risk_pyrologix_resampled.npy`.

**No data leakage:** Pyrologix was trained on fires from 2006–2020 only; the 2021 dataset is
fully out-of-sample.

| Map | Signal (all fires) | Signal (large fires) | Use in benchmark |
|-----|---------------------|----------------------|-----------------|
| **Pyrologix (recommended)** | **+21.0 pp** | **+13.6 pp** | Sensor/charging placement **and** drone routing |
| Burn Probability (FSim/BP) | +9.3 pp | +10.6 pp | Alternative placement/routing map |
| WFPI Yearly (time-aware) | −13.4 pp (per-day bg) | −10.6 pp | Pre-computed into scenario files only |

WFPI maps are anti-correlated with 2021 fire locations (drought-driven fires in low-wind northern
California forests, outside WFPI's strength as a wind-centric danger index). The WFPI Yearly map
is **not used at benchmark runtime** — fire spread is pre-computed into the per-scenario `.npy` files
during dataset creation. The benchmark script reads only the Pyrologix map.

---

## Dataset creation process — what's next?

After the California 2021 dataset is built and (if needed) cleaned with `remove_2021_fires_missing_wfpi_dates.py`, typical next steps are:

1. **Run benchmarks** — Use `run_benchmark_california2021_yearly.py` (project root). It reads `California2021Dataset/` directly. Pyrologix (`California2021Dataset/static_risk_pyrologix.npy`) is used for both sensor/charging placement and drone routing; the WFPI Yearly map is not loaded at runtime (fire spread is pre-computed in scenario files).
2. **Budget variants** — Run with `--budget 20`, `--budget 100`, `--budget 500`. Use `--sensor-only` first for the larger budgets to pre-compute the slow Julia sensor placement step.
3. **Reporting** — Add 2021 results to any report or comparison that currently uses only California 2020.

No further dataset-creation steps are required for the 932-fire 2021 set.

---

## Related Documentation

- [04_california_2020_dataset.md](04_california_2020_dataset.md) — Original California 2020 dataset (FPA_FOD source, structure, WFPI usage).
- [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md) — Ignition-point-only scenario format.
