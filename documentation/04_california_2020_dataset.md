# California 2020 Wildfire Dataset

## Overview

The **California 2020 Wildfire Dataset** is a comprehensive dataset containing all non-prescriptive, non-urban wildfires that occurred in California during 2020. The dataset uses a **statewide layout** with Wildland Fire Potential Index (WFPI) Day 2 forecast maps and **ignition-point-only scenarios** for efficient storage.

## Dataset Structure

```
California2020Dataset/
├── mask.npy                          # California mask (1 = valid, 0 = invalid)
├── wfpi_YYYYMMDD.npy                 # WFPI Day 2 forecast maps (one per calendar day — 366 files)
├── static_risk_wfpi_avg.npy          # Year-averaged WFPI map (for sensor placement)
├── static_risk_wfpi_yearly.npy       # Time-aware yearly WFPI map, shape (732, H, W)
├── scenarii/                         # Ignition-point scenarios
│   ├── FireName_FODID_scenario1.npy # One scenario per fire
│   └── ...
├── config_california_2020.json       # Configuration with random offsets
└── dataset_summary.json              # Dataset metadata and statistics
```

## Dataset Characteristics

### Fire Selection Criteria

1. **State:** California only
2. **Year:** 2020
3. **Prescription:** Non-prescriptive (all fires in FPA_FOD database are non-prescriptive)
4. **Urban Filter:** Excludes fires within urban areas (using US Census Urban Area Criteria 2025)
5. **Date Requirement:** Must have valid discovery date and time

### Data Sources

- **Fire Data:** FPA_FOD_20221014.gpkg (Fire Program Analysis Fire-Occurrence Database)
- **Urban Areas:** tl_2025_us_uac20.shp (US Census Urban Area Criteria 2025)
- **Risk Maps:** WFPI Day 2 Forecast data from USGS Fire Danger Maps
  - Source: https://firedanger.cr.usgs.gov/apps/staticmaps
  - Forecast: Day 2 (from the day before the fire to avoid data contamination)

### Scenario Format

Each scenario uses the **ignition-point-only format** (see [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md)):

- **File format:** NumPy array with shape `(3,)` containing `[row, col, start_timestep]`
- **Data type:** `int32`
- **Storage:** ~12 bytes per scenario (vs ~35 MB for full grid format)
- **Timesteps:** 12 timesteps (6 hours, 30-minute resolution)

### WFPI Maps

- **Format:** NumPy array with shape `(1, H, W)` where H and W are grid dimensions
- **Data type:** `float32`
- **Values:** Capped at 0 (negative values set to 0)
- **Naming:** `wfpi_YYYYMMDD.npy` where YYYYMMDD is the date of the Day 2 forecast
- **Date Logic:** For a fire discovered on date D, we use WFPI Day 2 forecast from date D-1

### Yearly Time-Aware WFPI Map (`static_risk_wfpi_yearly.npy`)

For scenarios where accurate time-of-day context matters, `static_risk_wfpi_yearly.npy` stores the correct WFPI forecast for every hour of 2020 in a single compact file.

- **Format:** NumPy array with shape `(732, H, W)` — 2 frames per day × 366 days (2020 is a leap year)
- **Data type:** `float32`
- **Frame layout for calendar day d** (d = 1 for Jan 1, d = 366 for Dec 31):
  - Frame `2*(d-1) + 0` — **before 10 am**: Day-2 forecast issued on day d−1
  - Frame `2*(d-1) + 1` — **after 10 am**: Day-1 forecast issued on day d
- **Rationale:** WFPI forecasts are updated once per day at 10 am. Before the update, the best available forecast is the Day-2 (issued the previous day). After 10 am, the Day-1 (same-day) forecast becomes available and supersedes it.
- **Missing days:** The 366 daily D2 and D1 source files were completed with `complete_wfpi_datasets.py` before building this map (see below). The only fallback is Jan 1 pre-10 am, which uses Jan 1 D1 (Dec 31 2019 D2 does not exist).

**Indexing at runtime:**

```python
def frame_index(discovery_date, hour):
    day_of_year = discovery_date.timetuple().tm_yday  # 1–366
    return 2 * (day_of_year - 1) + (0 if hour < 10 else 1)
```

**Per-scenario burn map extraction:**

```python
from datetime import timedelta

# sim_start: datetime of the first simulation step
frames = [yearly_map[frame_index(sim_start.date() + timedelta(minutes=30*t),
                                  (sim_start + timedelta(minutes=30*t)).hour)]
          for t in range(num_steps)]
scenario_burnmap = np.stack(frames)   # shape (num_steps, H, W)
```

**Performance vs other maps** (see `documentation/12_yearly_wfpi_map_comparison.md`):

| Map | % fires above bg median (all) | % fires above bg median (large) |
|-----|-------------------------------|----------------------------------|
| WFPI Yearly (time-aware) | **+18.1 pp** | +31.8 pp |
| WFPI Day 1 | +18.0 pp | +29.5 pp |
| WFPI Day 2 | +17.2 pp | +34.1 pp |

**Generation script:**

```bash
# Step 1 — ensure all 366 daily files exist for both D1 and D2:
python code/dataset_creation/nature_dataset_creation/complete_wfpi_datasets.py

# Step 2 — build the yearly map:
python code/dataset_creation/nature_dataset_creation/create_yearly_wfpi_burnmap.py
```

### Aggregated WFPI Map (`static_risk_wfpi_avg.npy`)

For sensor placement strategies, a single static risk map is needed that summarises fire risk over the whole year rather than on a per-day basis. `static_risk_wfpi_avg.npy` is the **element-wise average** of all available daily WFPI forecast maps for 2020.

- **Format:** NumPy array with shape `(1, H, W)` — same as individual WFPI maps
- **Data type:** `float32`
- **Value range:** 0 – 255 (same scale as individual WFPI maps)
- **Interpretation:** `avg[0, i, j]` is the mean WFPI value at cell `(i, j)` across all days for which a forecast map exists

| Dataset | Files averaged | Resulting mean |
|---------|---------------|----------------|
| `California2020Dataset` (Day-2) | 317 | ~159.2 |
| `California2020Dataset_Day1` (Day-1) | 320 | ~159.5 |

**Generation script:**

```bash
python code/dataset_creation/nature_dataset_creation/aggregate_wfpi_maps.py
```

The script (`aggregate_wfpi_maps.py`) iterates over every `wfpi_*.npy` (or `wfpi_day1_*.npy`) file in the dataset directory, accumulates a float64 sum, divides by the file count, and saves the result as float32. Missing calendar days are simply excluded (no imputation).

**Usage in benchmarking:**

```python
import numpy as np

# Load the yearly-averaged risk map for sensor placement
risk_map = np.load("California2020Dataset/static_risk_wfpi_avg.npy")
# risk_map shape: (1, H, W)

custom_initialization_parameters = {
    "burnmap_filename": "California2020Dataset/static_risk_wfpi_avg.npy",
    ...
}
```

### Mask

- **Format:** NumPy array with shape `(H, W)`
- **Data type:** `float32`
- **Values:** 
  - `1.0` = Valid area (WFPI data >= 0 and not nodata)
  - `0.0` = Invalid/masked area (WFPI data < 0 or nodata)

## Dataset Creation

### Prerequisites

1. **Fire Database:** `FPA_FOD_20221014.gpkg` in `data/RDS-2013-0009.6_Data_Format3_GPKG/`
2. **Urban Areas Shapefile:** `tl_2025_us_uac20.shp` in `data/tl_2025_us_uac20/`
3. **WFPI Data:** Day 2 forecast zip files in `data/2020_Wind-enhanced_Fire_Potential_Index_Forecast_2_DATA/`

### Creation Script

Run the dataset creation script:

```bash
cd code/dataset_creation/nature_dataset_creation
python create_california_2020_dataset.py
```

After creation, complete both D1 and D2 datasets to cover all 366 calendar days (required for the yearly map):

```bash
python code/dataset_creation/nature_dataset_creation/complete_wfpi_datasets.py
```

Then generate the static risk maps:

```bash
# Year-averaged map (sensor placement):
python code/dataset_creation/nature_dataset_creation/aggregate_wfpi_maps.py

# Time-aware yearly map (per-scenario burn map extraction):
python code/dataset_creation/nature_dataset_creation/create_yearly_wfpi_burnmap.py
```

See the [Yearly WFPI Map](#yearly-time-aware-wfpi-map-static_risk_wfpi_yearlynpy) and [Aggregated WFPI Map](#aggregated-wfpi-map-static_risk_wfpi_avgnpy) sections for details.

### Processing Steps

1. **Load Fire Database:** Query all CA 2020 fires with valid dates
2. **Filter Urban Fires:** Spatial join with urban areas shapefile
3. **Load WFPI Structure:** Determine grid dimensions from first WFPI file
4. **Create Mask:** Generate California mask from WFPI data
5. **Process Fires:** For each fire:
   - Get discovery date
   - Load corresponding WFPI Day 2 forecast (from day before)
   - Convert fire lat/lon to grid coordinates
   - Validate coordinates (within bounds and in valid mask area)
   - Save ignition point scenario
6. **Create Config:** Generate config file with random offsets (1-12)
7. **Create Summary:** Save dataset metadata

## Usage

### Loading Scenarios

The dataset uses the ignition-point-only format. To load scenarios in benchmarking code:

```python
from code.dataset import load_scenario_npy
import numpy as np

# Load mask to get grid dimensions
mask = np.load("California2020Dataset/mask.npy")
grid_height, grid_width = mask.shape

# Load scenario (auto-detects ignition-point format)
scenario = load_scenario_npy(
    "California2020Dataset/scenarii/FireName_FODID_scenario1.npy",
    grid_height=grid_height,
    grid_width=grid_width,
    num_timesteps=12
)
# scenario is now (12, H, W) array with ignition point set
```

### Loading WFPI Maps

```python
import numpy as np
from datetime import datetime, timedelta

# For a fire discovered on 2020-08-12
fire_date = datetime(2020, 8, 12)
wfpi_date = (fire_date - timedelta(days=1)).strftime("%Y%m%d")  # Day before

# Load WFPI map
wfpi = np.load(f"California2020Dataset/wfpi_{wfpi_date}.npy")
# wfpi shape: (1, H, W)
```

### Benchmarking

Use the dataset with the standard benchmarking functions:

```python
from code.benchmark import benchmark_on_sim2real_dataset_precompute

results = benchmark_on_sim2real_dataset_precompute(
    dataset_folder_name="California2020Dataset",
    ground_placement_strategy=...,
    drone_routing_strategy=...,
    custom_initialization_parameters_function=...,
    custom_step_parameters_function=...,
    file_format="npy",
    config_file="California2020Dataset/config_california_2020.json",
    simulation_parameters={
        "max_battery_time": 1,  # 1 hour
        "n_drones": 2,
        "n_ground_stations": 8,
        "n_charging_stations": 2,
        "drone_speed_m_per_min": 600,
        "coverage_radius_m": 300,
        "cell_size_m": 30,  # Will be determined from WFPI resolution
        "transmission_range": 50000,
    }
)
```

**Note:** The `cell_size_m` parameter should match the WFPI data resolution. Check the dataset summary for actual grid dimensions and calculate accordingly.

## Dataset Statistics

After creation, check `dataset_summary.json` for:

- Total fires processed
- Successful vs failed fires
- Unique WFPI dates
- Grid dimensions
- CRS information
- Date range

## Storage Requirements

**Estimated Storage (1km WFPI resolution):**
- WFPI maps (D2): ~1.5 GB (366 files × ~4 MB each)
- WFPI maps (D1, in California2020Dataset_Day1/): ~1.5 GB
- Yearly time-aware map: ~3.1 GB
- Averaged WFPI map: ~4 MB (single file)
- Mask: ~3 MB
- Scenarios: ~0.12 MB (ignition-point format)
- **Total: ~6 GB (D2 dataset alone); ~10 GB including D1 dataset)**

**Estimated Storage (30m WFPI resolution):**
- WFPI maps: ~1.2 TB
- Mask: ~3.3 GB
- Scenarios: ~0.12 MB
- **Total: ~1.2 TB**

Actual storage depends on WFPI data resolution. Check the first WFPI file to determine resolution.

## Coordinate System

The dataset uses the same CRS as the WFPI data (typically Albers Equal Area Conic, EPSG:5070). Fire coordinates are converted from WGS84 (EPSG:4326) to the WFPI CRS during dataset creation.

## Limitations

1. **Missing WFPI Data:** If WFPI data is missing for a fire's date, that fire is skipped
2. **Out-of-Bounds Fires:** Fires outside the WFPI grid bounds are skipped
3. **Masked Areas:** Fires in masked areas (invalid WFPI data) are skipped
4. **Date Parsing:** Fires with unparseable dates are skipped

## Troubleshooting

### Fire Count Mismatch

If the number of successful fires is lower than expected:

1. Check `dataset_summary.json` for failed fire count
2. Verify WFPI data exists for all required dates
3. Check if fires are outside grid bounds or in masked areas
4. Verify date parsing is working correctly

### Coordinate Conversion Issues

If fires are being skipped due to coordinate issues:

1. Verify WFPI CRS matches expected projection
2. Check that fire lat/lon coordinates are valid
3. Ensure transformer is correctly configured

### Missing WFPI Files

If WFPI files are missing:

1. Check `missing_file_list.txt` in WFPI data directory
2. Download missing files from USGS Fire Danger Maps
3. Re-run dataset creation

## Related Documentation

- [03_ignition_point_only_mode.md](03_ignition_point_only_mode.md) - Ignition-point-only format specification
- [01_project_overview.md](01_project_overview.md) - Project overview and architecture
- [02_data_pipeline.md](02_data_pipeline.md) - Data pipeline documentation (if exists)

## Citation

If using this dataset, please cite:

- **Fire Data:** Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6
- **WFPI Data:** USGS Fire Danger Maps - https://firedanger.cr.usgs.gov/apps/staticmaps
- **Urban Areas:** US Census Bureau, 2025 Urban Area Criteria
