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
2. **FPA-FOD 2022** `FPA_FOD_20221014.gpkg`, layer `Fires`, plus the newer
   **USFS fire-occurrence point data** used in the recovered work (a GPKG,
   GeoJSON, or other GeoPandas-readable export with `DISCOVERYDATETIME` and
   `OBJECTID`). The pipeline does not download these products.
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

# 3a. Reproduce space-only selection.
python code/curate_dataset.py select \
  /data/Sim2Real-Fire/WideDataset work/fires.gpkg work/layouts.geojson \
  work/selection_space.csv

# 3b. Independently produce date-aware historical selection.
# By default this reproduces the recovered classification pass: newer USFS
# records from 2019 onward, restricted to each layout's weather-date range.
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

## Historical defaults and selection rules

- The authoritative layouts are 30 m grids.
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
- A big fire has final burned-area radius `>= 20 km`.
- A fast fire has burned cells at `t=10` (or the final frame if shorter)
  `>= 50%` of final burned cells.

Matching is independent of filesystem/input ordering. The default tie seed is
0; ties use a stable SHA-256 key rather than Python's process-randomized hash.

## Known provenance limits and documented deviations

- The recovered artifacts encode local filenames but do not contain checksums,
  download URLs, release identifiers for the newer USFS extract, or a complete
  end-to-end execution log. Reviewers must obtain the upstream products and
  record their own checksums.
- The newer USFS export's invalid year-1001 discovery dates were inconsistently
  replaced with 1900 or removed. The published merge uses `errors="coerce"` and
  drops records without a valid date.
- Historical deduplication used exact latitude, longitude, and discovery
  timestamp. That rule is retained, with stable source/ID ordering.
- Historical candidate enumeration used unseeded `random`, mutable default
  exclusion lists, and inconsistent return shapes. Published matching is
  deterministic and typed.
- Historical code retained raster handles after context managers and manually
  cropped/interpolated risk rasters. Published code reopens templates as
  needed and uses `rasterio.warp.reproject` directly onto the exact grid.
- Risk-map values are not normalized per layout. BP remains in the source
  product's native calibration and defaults to bilinear resampling; WHP remains
  a non-probabilistic index and defaults to nearest-neighbor resampling.
- Historical scripts copied, moved, renamed, or deleted source content.
  Published commands never move, rename, or delete inputs.
- The recovered summary function defaulted to 1 km per cell despite the
  asserted 30 m grids. The reviewer path uses 0.03 km per cell, making the
  documented 20 km radius threshold physically consistent.
- The original noncumulative map is a mean of frame-to-frame differences and
  can contain negative values if an input mask shrinks; this behavior is
  retained and made explicit.
- Raster-cropping differences can change selected scenario IDs. The published
  12-layout/474-scenario Tables 2/3 list in `splits/` is therefore an
  authoritative recovered experiment manifest, not an output we claim the
  implementation reproduces bit-for-bit.
- The pipeline begins with an already downloaded Sim2Real-Fire export. It does
  not reproduce upstream physical simulations. No producer for the historical
  `config_s2r.json` offsets was recovered; that versioned artifact is consumed
  as input.

Install `geopandas` and its geospatial stack through `environment.yml`.
Lightweight matching and numerical tests do not need the full dataset.
