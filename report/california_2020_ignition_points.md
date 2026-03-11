# California 2020 Non-Urban Wildfire Ignition Points

## Overview

This report summarises the filtering of the USFS ignition points dataset to extract
non-urban wildfires that occurred in California during 2020.

**Source file:** `USFS_ignition_points.csv`  
**Plot:** `california_2020_ignition_points.png`

---

## Filtering Criteria

| Criterion | Value |
|-----------|-------|
| State | California (`UNIQFIREID` starts with `2020-CA`) |
| Year | 2020 (`FIREYEAR == 2020`) |
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
| CA 2020 WF fires (raw) | 1,526 |
| Outside CA boundary removed | 13 |
| Urban fires removed | 76 |
| **Non-urban fires kept** | **1,437** |

---

## Fire Size Class Distribution

| Class | Acreage range | Count |
|-------|---------------|-------|
| A | ≤0.25 ac | 846 |
| B | 0.26–9.9 ac | 289 |
| C | 10–99 ac | 61 |
| D | 100–299 ac | 21 |
| E | 300–999 ac | 13 |
| F | 1,000–4,999 ac | 13 |
| G | 5,000–9,999 ac | 8 |
| H | 10,000–49,999 ac | 14 |
| I | 50,000–99,999 ac | 3 |
| J | 100,000–299,999 ac | 8 |

---

## Discovery Month Distribution

  - Jan: 18  
  - Feb: 56  
  - Mar: 30  
  - Apr: 29  
  - May: 102  
  - Jun: 204  
  - Jul: 371  
  - Aug: 332  
  - Sep: 117  
  - Oct: 81  
  - Nov: 55  
  - Dec: 42

---

## Cause of Ignition

| Cause | Count | % |
|-------|-------|---|
| Undetermined | 670 | 46.6% |
| Lightning | 366 | 25.5% |
| Camping | 210 | 14.6% |
| Equipment | 116 | 8.1% |
| Debris/Open Burning | 48 | 3.3% |
| Incendiary | 22 | 1.5% |
| Smoking | 5 | 0.3% |

---

## Spatial Coverage

| Metric | Value |
|--------|-------|
| Latitude range | 32.6891° – 41.9806° N |
| Longitude range | -124.0532° – -115.1952° W |
| Discovery date range | 2019-12-31 – 2020-12-29 |

---

## Next Steps

The filtered `non_urban_gdf` GeoDataFrame (1086 → 1437 fires) will be used as the
ignition point input for the **California 2020 dataset**, following the same pipeline as
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
| After stage-1 (boundary + urban) | 1,437 |
| Removed — out of WFPI grid bounds | 0 |
| Removed — WFPI masked cell (urban / nodata / outside CA) | 130 |
| **Kept after stage-2** | **1,307** |

**WFPI-overlay plot:** `california_2020_ignition_points_wfpi.png`
