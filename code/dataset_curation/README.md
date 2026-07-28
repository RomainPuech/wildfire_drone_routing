# WFDroneBench dataset curation

This package is the reviewer-facing, auditable path from raw Sim2Real-Fire
exports and public fire/risk products to WFDroneBench selections. **Upstream
Sim2Real-Fire generates the raw physical wildfire simulations. This pipeline
does not generate fires: it curates, selects, spatially aligns, and packages
those simulations.** Every stage is non-destructive: sources are only read and
all products go to explicit output paths.

## Exact external inputs

1. A **Sim2Real-Fire layout export**, one directory per layout (for example
   `0016_03070/`), containing:
   - `Vegetation_Map/Existing_Vegetation_Cover.tif`, the authoritative layout
     CRS, affine transform, height, width, and nominal 30 m grid;
   - `Weather_Data/<scenario_id>.txt`, whose first line begins `YYYY MM DD`;
   - `Satellite_Images_Mask/<scenario_id>/*.jpg` (the historical singular
     `Satellite_Image_Mask` spelling is also accepted), grayscale fire masks.
2. **FPA-FOD 2022** `FPA_FOD_20221014.gpkg`, layer `Fires`, plus newer
   **USFS fire-occurrence point data** (a GPKG, GeoJSON, or other
   GeoPandas-readable export with `DISCOVERYDATETIME` and `OBJECTID`). The
   pipeline does not download these products.
3. A continental **USFS Burn Probability (BP) or Wildfire Hazard Potential
   (WHP) GeoTIFF**. Supply each product separately to `risk`.

The repository-root `config_s2r.json` is the tracked scenario-offset mapping.
It is intentionally not duplicated here; pass it with `preprocess --config
config_s2r.json`.

## Stages and outputs

Run commands from the repository root with Python 3.10:

```bash
# 1. Inventory eligible exact layout grids and WGS84 footprints.
python code/curate_dataset.py footprints \
  /data/Sim2Real-Fire/WideDataset work/layouts.geojson

# 2. Normalize and deduplicate historical occurrence records.
python code/curate_dataset.py merge-fires \
  /data/FPA_FOD_20221014.gpkg /data/usfs_newer.gpkg work/fires.gpkg

# 3a. Space-only selection.
python code/curate_dataset.py select \
  /data/Sim2Real-Fire/WideDataset work/fires.gpkg work/layouts.geojson \
  work/selection_space.csv

# 3b. Date-aware historical selection (default: USFS_newer records from 2019
# onward, restricted to each layout's weather-date range).
python code/curate_dataset.py select \
  /data/Sim2Real-Fire/WideDataset work/fires.gpkg work/layouts.geojson \
  work/selection_historical.csv --date-aware

# 4. Align either risk product to every layout's exact grid.
python code/curate_dataset.py risk \
  /data/whp2023_cnt_conus.tif /data/Sim2Real-Fire/WideDataset curated \
  --risk-name whp

# 5. Convert only accepted/selected JPG sequences and compute empirical maps.
python code/curate_dataset.py preprocess \
  /data/Sim2Real-Fire/WideDataset work/selection_space.csv curated \
  --config config_s2r.json

# 6. Build reviewer-readable scenario attributes.
python code/curate_dataset.py summary \
  /data/Sim2Real-Fire/WideDataset work/selection_space.csv \
  curated/scenario_summary.csv \
  --historical-manifest work/selection_historical.csv
```

`footprints` writes layout identifiers, dimensions, source grid paths, and
polygons. `merge-fires` writes source-prefixed unique IDs, original source
IDs, UTC discovery dates, source labels, and points. `select` writes one row per historical fire, including
matched/unmatched status, both coordinate pairs, distances, mode, seed, and
layout acceptance; a sibling `*_layouts.csv` records rejection statistics.
`risk` writes `static_risk_bp2024.{tif,npy}` or
`static_risk_whp.{tif,npy}` under an output layout directory in the source
product's native units.
`preprocess` writes binary float32 `scenarii/*.npy`, `burn_map.npy`, and
`burn_map_noncumulative.npy`. `summary` writes `scenario_summary.csv`.

## Defaults and selection rules

- Layouts are 30 m grids.
- Layouts with width below 500 cells are excluded by default where layout
  eligibility is applied (`footprints --include-small` overrides this).
- Space-only matching uses maximum Chebyshev distance 5 cells.
- Date-aware matching uses maximum Chebyshev distance 10 cells and ±1 day.
- The date-aware classification pass defaults to `USFS_newer` records dated
  2019 or later and to the scenario weather-date range. It records the 20%
  threshold but does not reject layouts unless `--enforce-layout-filter` is
  passed; corpus eligibility comes from the space-only pass.
- A simulation is never reused within a layout. Explicit exclusions are also
  supported by the pure `match_scenarios` API.
- Layouts with more than 20% unmatched historical fires are rejected. The
  manifest retains their rows for auditability but preprocessing ignores them.
- JPG values are divided by 255, thresholded at `>= 0.5`, and stored as
  `float32`.
- A big fire has final burned-area radius `>= 20 km` (0.03 km per cell).
- A fast fire has burned cells at `t=10` (or the final frame if shorter)
  `>= 50%` of final burned cells.
- Risk-map values are kept in the source product's native units (not
  normalized per layout). BP defaults to bilinear resampling; WHP (a
  non-probabilistic index) defaults to nearest-neighbor resampling.
- Invalid discovery dates in the USFS export are dropped
  (`errors="coerce"`).
- Deduplication uses exact latitude, longitude, and discovery timestamp, with
  stable source/ID ordering.
- Matching is deterministic: default tie seed 0; ties use a stable SHA-256 key.
- The pipeline consumes a pre-downloaded Sim2Real-Fire export and the tracked
  `config_s2r.json` offset map; it does not regenerate upstream physical
  simulations.

Install `geopandas` and its geospatial stack through `environment.yml`.
Lightweight matching and numerical tests do not need the full dataset.
