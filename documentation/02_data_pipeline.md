# Data Pipeline

This document describes how wildfire scenario data is loaded, stored, preprocessed, and transformed into burn maps. It covers the full journey from raw Sim2Real-Fire images to simulation-ready NumPy arrays.

---

## Table of Contents

1. [Dataset Structure](#1-dataset-structure)
2. [Scenario Formats](#2-scenario-formats)
3. [Loading Scenarios](#3-loading-scenarios)
4. [Saving Scenarios](#4-saving-scenarios)
5. [Dataset Preprocessing (JPG → NPY)](#5-dataset-preprocessing-jpg--npy)
6. [Burn Map Computation](#6-burn-map-computation)
7. [Risk Maps and Static Burn Maps](#7-risk-maps-and-static-burn-maps)
8. [Result Aggregation](#8-result-aggregation)
9. [Dataset Utilities](#9-dataset-utilities)

---

## 1. Dataset Structure

The project uses the **Sim2Real-Fire** dataset format. Each dataset is organized as a hierarchy of **layouts** (geographic areas) containing **scenarios** (individual fire simulations):

```
MiniTractDataset/                    # Dataset root
├── 0058_03866/                      # Layout folder (tract_id + fire_id)
│   ├── Satellite_Images_Mask/       # Raw JPG scenario folders
│   │   ├── 0058_00123/              # One scenario (sequence of fire frames)
│   │   │   ├── 0001.jpg             # Fire grid at timestep 1
│   │   │   ├── 0002.jpg             # Fire grid at timestep 2
│   │   │   └── ...
│   │   ├── 0058_00456/
│   │   └── ...
│   ├── scenarii/                    # Preprocessed NPY scenarios
│   │   ├── 0058_00123.npy           # Entire scenario as 3D array
│   │   ├── 0058_00456.npy
│   │   └── ...
│   ├── burn_map.npy                 # Cumulative burn probability map
│   ├── burn_map_noncumulative.npy   # Non-cumulative burn map (optional)
│   ├── static_risk_whp.npy          # Wildfire Hazard Potential risk map
│   ├── static_risk_bp2024.npy       # Burn probability 2024 risk map
│   ├── mask.npy                     # Land/water mask (optional)
│   ├── selected_scenarios.txt       # Scenario selection metadata
│   ├── logs/                        # Strategy log files (JSON)
│   └── Weather_Data/                # Weather information (not used by current strategies)
├── 0081_03471/
└── ...
```

### Naming Convention

Layout folders follow the pattern `{tract_id}_{fire_id}`:
- `tract_id`: A 4-digit census tract identifier
- `fire_id`: A 5-digit fire simulation identifier

Scenario folders/files follow the pattern `{tract_id}_{scenario_id}`.

---

## 2. Scenario Formats

A **scenario** is a 3D array of shape `(T, N, M)` representing fire progression over time:

| Dimension | Meaning |
|-----------|---------|
| `T` | Number of timesteps (typically 1 hour each) |
| `N` | Grid height (rows) |
| `M` | Grid width (columns) |

Cell values:
- `0.0` = no fire
- `1.0` = fire present

### JPG Format

Scenarios stored as a folder of grayscale JPEG images:
- One image per timestep, named `0001.jpg`, `0002.jpg`, etc.
- Pixel values 0–255, normalized to [0, 1] on load
- Binary threshold at 0.5: values ≥ 0.5 become 1.0, others become 0.0
- **Pros**: Human-readable, can be viewed in any image viewer
- **Cons**: Slower to load (many file operations), lossy compression

### NPY Format

Scenarios stored as a single NumPy `.npy` file:
- Direct `float32` array of shape `(T, N, M)`
- **Pros**: Fast loading (single file read), exact values
- **Cons**: Not human-readable
- Stored in the `scenarii/` subfolder of each layout

### TIF Format

Raw GeoTIFF files (`.tif`) from risk map data:
- Used for static risk maps (wildfire hazard potential, burn probability)
- Read via `rasterio`, first band extracted
- Converted to `.npy` for use in the pipeline

---

## 3. Loading Scenarios

### `load_scenario_jpg(folder_path, binary=True)`

Loads a scenario from a folder of JPEG images:

```python
scenario = load_scenario_jpg("MiniTractDataset/0058_03866/Satellite_Images_Mask/0058_00123/")
# Returns: np.ndarray of shape (T, N, M), dtype float
```

**Process**:
1. Lists all `.jpg` files in the folder
2. Sorts them in **natural order** (so `im2` comes before `im10`)
3. Opens each image as grayscale (`'L'` mode)
4. Normalizes pixel values to [0, 1] by dividing by 255
5. Applies binary threshold at 0.5 (if `binary=True`)
6. Stacks into a 3D array

### `load_scenario_npy(filename)`

Loads a scenario from a NumPy file:

```python
scenario = load_scenario_npy("MiniTractDataset/0058_03866/scenarii/0058_00123.npy")
# Returns: np.ndarray of shape (T, N, M)
```

Handles two formats:
- **New format**: Direct NumPy array (standard)
- **Old format**: 0-dimensional array containing a dictionary with `'scenario'` key (legacy)

### `load_scenario(file_or_folder_name, extension=".npy")`

Unified loader that dispatches to the appropriate format-specific function:

```python
# Load NPY
scenario = load_scenario("path/to/scenario", extension=".npy")

# Load JPG
scenario = load_scenario("path/to/scenario_folder", extension=".jpg")
```

---

## 4. Saving Scenarios

### `save_scenario_npy(scenario, out_filename)`

Saves a scenario as a NumPy file with `float32` precision for storage efficiency:

```python
save_scenario_npy(scenario, "output/scenario_001.npy")
```

### `save_scenario_jpg(scenario, out_folder_name)`

Saves a scenario as a folder of JPEG images:

```python
save_scenario_jpg(scenario, "output/scenario_001_images/")
```

### `save_burn_map(burn_map, filename)`

Thin wrapper around `save_scenario` — burn maps share the same `(T, N, M)` format.

---

## 5. Dataset Preprocessing (JPG → NPY)

The raw Sim2Real dataset uses JPEG images. For benchmarking performance, we convert them to NPY format.

### `sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, ...)`

Batch-converts all JPG scenario folders to NPY files:

```python
sim2real_scenario_jpg_folders_to_npy(
    "MiniTractDataset/",
    n_max_scenarii_per_layout=100,   # Limit scenarios per layout
    n_max_layouts=10,                 # Limit total layouts
    mismatch_threshold=0.2            # Skip layouts with >20% failed scenario matching
)
```

**Process**:
1. Iterates over all layout folders in the dataset
2. For each layout, iterates over scenario folders in `Satellite_Images_Mask/`
3. Skips scenarios already converted (checks if `.npy` exists in `scenarii/`)
4. Calls `jpg_scenario_to_npy()` to load the JPG folder and save as NPY
5. Creates the `scenarii/` directory if it doesn't exist

### `preprocess_sim2real_dataset(dataset_folder_name, ...)`

High-level function that runs the full preprocessing pipeline:

```python
preprocess_sim2real_dataset(
    "MiniTractDataset/",
    n_max_scenarii_per_layout=100,
    n_max_layouts=10,
    config_file="config_s2r.json"
)
```

This calls:
1. `sim2real_scenario_jpg_folders_to_npy()` — converts JPGs to NPY
2. `compute_and_save_burn_maps_sim2real_dataset()` — computes burn maps from the scenarios

### Mismatch Threshold

Some layouts have a `selected_scenarios.txt` file tracking how many historical fire records could be matched to simulation scenarios. If the failure rate exceeds `mismatch_threshold` (e.g., 20%), the layout is skipped during preprocessing.

---

## 6. Burn Map Computation

A **burn map** is a 3D array of shape `(T, N, M)` where each cell `(t, i, j)` represents the **probability** that cell `(i, j)` is on fire at timestep `t`, averaged across all scenarios in a layout.

### `compute_burn_map(folder_name, extension=".npy", noncumulative=False, config=None)`

Computes the average burn map from all scenarios in a folder:

```python
burn_map = compute_burn_map("MiniTractDataset/0058_03866/scenarii/")
# Returns: np.ndarray of shape (T, N, M) with values in [0, 1]
```

**Algorithm**:

```
burn_map = zeros(T, N, M)
counts = zeros(T)

for each scenario in folder:
    offset = config.get(scenario_name, 0)  # Time offset from config
    scenario = prepend_zeros(scenario, offset)  # Align start times

    for t in range(len(scenario)):
        if noncumulative:
            burn_map[t] += scenario[t] - scenario[t-1]   # Only new fire
        else:
            burn_map[t] += scenario[t]                    # Cumulative fire
        counts[t] += 1

    # Extend arrays if this scenario is longer than previous ones
    if len(scenario) > len(burn_map):
        pad burn_map and counts

# Average
for t in range(T):
    burn_map[t] /= counts[t]
```

### Cumulative vs. Non-Cumulative

- **Cumulative** (`noncumulative=False`): `burn_map[t, i, j]` = probability cell `(i,j)` has **ever** been on fire by timestep `t`
- **Non-cumulative** (`noncumulative=True`): `burn_map[t, i, j]` = probability cell `(i,j)` **newly catches fire** at timestep `t`

### Time Offset Alignment

The `config` parameter (from `config_s2r.json`) provides per-scenario starting time offsets. This is important because different fires start at different times relative to the simulation epoch. Scenarios are aligned by prepending zero frames:

```python
# If offset = 2, a scenario of shape (10, N, M) becomes (12, N, M)
# with the first 2 frames being all zeros
scenario = np.concatenate([np.zeros((offset, N, M)), scenario], axis=0)
```

### `compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name, ...)`

Batch-computes and saves burn maps for all layouts in a dataset:

```python
compute_and_save_burn_maps_sim2real_dataset(
    "MiniTractDataset/",
    n_max_layouts=10,
    noncumulative=False,
    config=config
)
```

For each layout, it:
1. Loads all scenarios from the `scenarii/` folder
2. Computes the averaged burn map
3. Saves as `burn_map.npy` (or `burn_map_noncumulative.npy`) in the layout folder

---

## 7. Risk Maps and Static Burn Maps

In addition to the dynamic (time-varying) burn maps computed from scenarios, the project supports **static risk maps** that do not change over time:

| File | Description |
|------|-------------|
| `static_risk_whp.npy` | Wildfire Hazard Potential — a static risk map from US risk data |
| `static_risk_bp2024.npy` | Burn Probability 2024 — estimated from 2024 fire data |
| `burn_map.npy` | Dynamic burn map computed from scenario averaging |
| `burn_map_noncumulative.npy` | Non-cumulative dynamic burn map |

The experiment runner selects which burn map to use via the `--bm_prefix` argument:

```python
BM_PREFIX_TO_NAME = {
    "whp": "static_risk_whp.npy",      # Wildfire Hazard Potential
    "bm": "burn_map.npy",               # Dynamic cumulative burn map
    "bp": "static_risk_bp2024.npy",     # Burn probability 2024
    "ncbm": "burn_map_noncumulative.npy" # Non-cumulative burn map
}
```

Static risk maps are typically 2D arrays `(N, M)` that are expanded to 3D during the benchmark's operational scaling step.

### TIF to NPY Conversion

Risk maps may originally come as GeoTIFF files. `convert_tif_to_npy()` handles bulk conversion:

```python
convert_tif_to_npy("code/dataset_creation/risklayouts/", "code/dataset_creation/risk_layouts_npy/")
```

This reads each `.tif` file with `rasterio`, extracts the first band, and saves as `.npy`.

---

## 8. Result Aggregation

### `combine_all_benchmark_results(dataset_folder, strategy_name, experiment_name)`

After running benchmarks on all layouts, per-layout CSV files are scattered across the dataset. This function collects them into a single combined CSV:

```python
combined_df = combine_all_benchmark_results(
    "MiniTractDataset/",
    strategy_name="RandomSensorPlacementStrategy_DroneRoutingMaxCoverageResetStatic",
    experiment_name="Mwhp_parallel"
)
# Saves to: results/combined_benchmark_resultsMwhp_parallel.csv
```

**Process**:
1. Iterates over all layout folders in the dataset
2. Looks for CSV files matching the pattern `{layout_id}_benchmark_results{experiment_name}_{strategy_name}.csv`
3. Reads each CSV with `pandas`, preserving string formatting for layout and scenario IDs
4. Concatenates all DataFrames
5. Saves the combined result to the `results/` folder

---

## 9. Dataset Utilities

### `clean_layout_folders(root_folder)`

Cleans each layout folder by removing files and directories not in the allowed list. Also renames `Satellite_Image_Mask` to `Satellite_Images_Mask` for consistency:

```python
# Allowed items:
# Fuel_Map, Satellite_Images_Mask, satellite_image.png, static_risk.npy,
# Topography_Map, Vegetation_Map, Weather_Data
clean_layout_folders("MiniTractDataset/")
```

### `clean_logs_folder(root_folder)`

Removes all files from the `logs/` subfolder of each layout:

```python
clean_logs_folder("MiniTractDataset/")
```

### `delete_logs_folder(folder_path)`

Completely removes the `logs/` subfolder from a given path:

```python
delete_logs_folder("MiniTractDataset/0058_03866/")
```

### `listdir_limited(input_dir, max_n_scenarii=None)`

A generator that iterates over directory contents with an optional count limit, skipping `.DS_Store` files:

```python
for name in listdir_limited("scenarii/", max_n_scenarii=100):
    # process up to 100 items
```

---

*Previous: [01 — Project Overview](01_project_overview.md) · Next: [03 — Operational Scaling](03_operational_scaling.md)*
