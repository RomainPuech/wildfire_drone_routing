# California 2021 Non-Urban Wildfire Ignition Points

## Overview

This report summarises the filtering of the USFS ignition points dataset to extract
non-urban wildfires that occurred in California during 2021.

**Source file:** `USFS_ignition_points.csv`  
**Plot:** `california_2021_ignition_points.png`

---

## Filtering Criteria

| Criterion | Value |
|-----------|-------|
| State | California (`UNIQFIREID` starts with `2021-CA`) |
| Year | 2021 (`FIREYEAR == 2021`) |
| Fire Type | Wildfire only (`FIRETYPECATEGORY == 'WF'`) |
| Boundary filter | Coordinates must fall within California state polygon |
| Urban filter | Removed fires within US Census Urban Areas 2025 |
| Date validity | Must have non-null `DISCOVERYDATETIME` |
| Coordinate validity | Must have non-null `LATDD83` / `LONGDD83` |

---

## Filter Summary

| Stage | Count |
|-------|-------|
| Total records in CSV | 582,291 |
| CA 2021 WF fires (raw) | 1,086 |
| Outside CA boundary removed | 13 |
| Urban fires removed | 25 |
| **Non-urban fires kept** | **1,048** |

---

## Fire Size Class Distribution

| Class | Acreage range | Count |
|-------|---------------|-------|
| A | ≤0.25 ac | 663 |
| B | 0.26–9.9 ac | 303 |
| C | 10–99 ac | 43 |
| D | 100–299 ac | 8 |
| E | 300–999 ac | 9 |
| F | 1,000–4,999 ac | 6 |
| G | 5,000–9,999 ac | 3 |
| H | 10,000–49,999 ac | 4 |
| I | 50,000–99,999 ac | 1 |
| J | 100,000–299,999 ac | 5 |
| K | ≥300,000 ac | 1 |

---

## Discovery Month Distribution

  - Jan: 38  
  - Feb: 18  
  - Mar: 24  
  - Apr: 87  
  - May: 143  
  - Jun: 183  
  - Jul: 236  
  - Aug: 153  
  - Sep: 75  
  - Oct: 45  
  - Nov: 33  
  - Dec: 13

---

## Cause of Ignition

| Cause | Count | % |
|-------|-------|---|
| Undetermined | 427 | 40.7% |
| Lightning | 317 | 30.2% |
| Camping | 119 | 11.4% |
| Equipment | 95 | 9.1% |
| Debris/Open Burning | 65 | 6.2% |
| Incendiary | 23 | 2.2% |
| Smoking | 2 | 0.2% |

---

## Spatial Coverage

| Metric | Value |
|--------|-------|
| Latitude range | 32.7055° – 41.9988° N |
| Longitude range | -124.0474° – -116.1160° W |
| Discovery date range | 2021-01-01 – 2021-12-30 |

---

## Next Steps

The filtered `non_urban_gdf` GeoDataFrame (1086 → 1048 fires) will be used as the
ignition point input for the **California 2021 dataset**, following the same pipeline as
`create_california_2020_dataset.py`:

1. Load corresponding WFPI Day-2 forecast for each fire's discovery date
2. Convert lat/lon to grid (row, col) using the cropped California WFPI transform
3. Validate against the California mask
4. Save each fire as an ignition-point scenario (`[row, col, start_timestep]`)

---

## Stage 2: WFPI Grid & Mask Filter

After the initial stage-1 filters (unit code, year, fire type, California
boundary, urban exclusion), ignition points are overlaid on the WFPI 1 km
grid (1309 × 805 cells, Lambert Azimuthal Equal-Area) that underlies the
California2020Dataset.  A fire is **kept** only if:

1. Its lat/lon maps to a `(row, col)` inside the cropped-California grid
2. `mask[row, col] == 1` — i.e. the cell is inside the California state
   boundary, is not always unburnable (see mask logic below), and is not an urban area

> **Mask logic:** a cell is valid if it has WFPI < 249 or WFPI = 250 (snow)
> on at least one day of 2020 (union-of-burnable, snow not excluded).
> Mask built from **2020 WFPI Day 1** daily maps (`mask_union_burnable_no_snow_excluded_day1.npy`). Cells always in 249 or 251-255 are excluded.

| Stage | Count |
|-------|-------|
| After stage-1 (boundary + urban) | 1,048 |
| Removed — out of WFPI grid bounds | 0 |
| Removed — WFPI masked cell (urban / nodata / outside CA) | 63 |
| Excluded — discovery on missing WFPI zip date | 53 |
| **Kept after stage-2 (in dataset)** | **932** |

**WFPI-overlay plot:** `california_2021_ignition_points_wfpi.png`
