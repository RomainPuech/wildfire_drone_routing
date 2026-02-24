# California 2020 Time-Aware Yearly WFPI Benchmark

## Overview and Motivation

The standard California 2020 benchmark uses a single static burn map per scenario — either the Day-2 or Day-1 WFPI forecast — selected at dataset creation time. This approach does not account for the actual time of day at which a fire was discovered, nor for the operational rule that governs which WFPI forecast is available at a given hour: before 10 am the best available forecast is the Day-2 (issued the previous day); from 10 am onward, the same-day Day-1 forecast supersedes it.

The **time-aware yearly WFPI benchmark** (`run_benchmark_california2020_yearly.py`) corrects this by:

1. Using `static_risk_wfpi_yearly.npy` — a single array of shape `(732, H, W)` containing two WFPI frames per calendar day — to select the operationally correct forecast for every 30-minute simulation step.
2. Anchoring each scenario in real calendar time using fire discovery date and time retrieved from the FPA FOD database.
3. Caching routing computations by `(cluster, rounded_sim_start_hour)` so that scenarios whose simulation windows nearly coincide share a single Julia routing call.

The result is a benchmark that reflects actual operational conditions: the drone routing strategy sees the same WFPI risk information that would have been available to a real dispatcher at the moment of each fire discovery.

For evidence that the time-aware yearly map is superior to always using Day-2 or Day-1 alone, see [12_yearly_wfpi_map_comparison.md](12_yearly_wfpi_map_comparison.md). For a description of the dataset itself, see [04_california_2020_dataset.md](04_california_2020_dataset.md).

---

## Prerequisites

### Step 1 — Augment the config with discovery times

The dataset config file (`config_california_2020.json`) is created with only `offset_<name>` keys. The time-aware benchmark additionally requires `date_<name>` and `time_<name>` keys. These are added by running:

```bash
python code/dataset_creation/nature_dataset_creation/augment_config_with_times.py
```

This script:

1. Reads every `offset_<name>` entry from `config_california_2020.json`.
2. Extracts the numeric `FOD_ID` from the scenario name suffix (e.g., `GULCH_TRL__JONESVALLEY_400612285` → `400612285`).
3. Queries `FPA_FOD_20221014.gpkg` (SQLite/GPKG) for `DISCOVERY_DATE` and `DISCOVERY_TIME` for those FOD IDs.
4. For each fire that has a non-null `DISCOVERY_TIME`, writes back:
   - `date_<name>`: the discovery date in `YYYYMMDD` format
   - `time_<name>`: the discovery time in `HHMM` format (zero-padded)
5. Fires without a `DISCOVERY_TIME` are **dropped** from the config entirely: they cannot be placed in real time and therefore cannot be used with the time-aware benchmark.

**Filtering outcome (California 2020):** 888 of 2,418 fires lack `DISCOVERY_TIME`; 1,530 are retained.

After running the script the config contains three keys per scenario:

```json
{
  "offset_SOMENAME_123456789": 7,
  "date_SOMENAME_123456789":   "20200812",
  "time_SOMENAME_123456789":   "1430"
}
```

The benchmark filters to only those scenarios for which all three keys are present:

```python
valid_scenarios = [
    sf for sf in all_scenario_files
    if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config
           for k in ("offset", "date", "time"))
]
```

### Step 2 — Ensure the yearly WFPI map exists

`static_risk_wfpi_yearly.npy` must be present in `California2020Dataset/`. If it has not been built yet:

```bash
# Fill missing daily WFPI source files for both D1 and D2:
python code/dataset_creation/nature_dataset_creation/complete_wfpi_datasets.py

# Build the yearly map (732 frames):
python code/dataset_creation/nature_dataset_creation/create_yearly_wfpi_burnmap.py
```

See [04_california_2020_dataset.md](04_california_2020_dataset.md) for full dataset preparation instructions.

---

## Running the Benchmark

```bash
# From the project root:
python -u run_benchmark_california2020_yearly.py
```

The `-u` flag disables Python output buffering so progress lines appear immediately.

---

## Pipeline Steps

### 1. Load static resources

```python
yearly_map = np.load(str(YEARLY_MAP), mmap_mode="r")   # (732, H, W), memory-mapped
avg_map    = np.load(str(AVG_MAP))                       # (1, H, W), year-averaged
mask       = np.load(str(MASK_PATH))                     # (H, W)
config     = json.load(open(CONFIG_PATH))
```

`mmap_mode="r"` is used for the 732-frame yearly map to avoid loading the entire ~3 GB array into RAM; frames are read on demand.

### 2. Compute operational rescaling

```python
operational_substeps = compute_operational_substeps(cell_size_m, speed_m_per_min, coverage_radius_m)
coverage_w = round(coverage_radius_m * 2 / cell_size_m)
if coverage_w % 2 == 0:
    coverage_w -= 1          # keep odd so the centre cell is well-defined

rescaled_N = H // coverage_w
rescaled_M = W // coverage_w
rescaled_max_battery = max_battery_time * operational_substeps
```

All routing and sensor placement runs in "opt-space" at resolution `(rescaled_N, rescaled_M)`. The averaged WFPI map and mask are downsampled once and saved to disk to avoid re-computing them on each run.

### 3. Sensor placement and drone allocation (cached)

`SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation` is called once per strategy combination. It returns:
- `get_locations()` → `(ground_sensor_locs, charging_station_locs)` in opt-space
- `get_drone_allocation()` → list of ints, one per charging station

Because this strategy does not implement the `get_drone_allocation()` interface expected by `make_loggable_sensor_strategy` in `wrappers.py`, results are cached manually to:

```
California2020Dataset/logs/sensor_alloc_{combo_name}_{N}x{M}.json
```

On subsequent runs the file is loaded directly, skipping the Julia call.

### 4. Drone cluster computation

Charging stations are grouped into connected components (clusters) using union-find. Two stations belong to the same cluster if their L∞ distance in opt-space is at most `max_battery_substeps` (meaning a drone can fly directly between them on a single battery charge).

Each cluster stores:
- `stations_opt`: list of `(row, col)` positions in opt-space
- `n_drones`: sum of drone allocations for all stations in the cluster
- `fingerprint`: stable string key used for log file naming, e.g. `"3-7_5-12"`

### 5. Per-scenario routing and simulation

For each scenario with valid date and time:

1. Parse discovery datetime and compute `sim_start`:
   ```python
   discovery_dt = datetime(year, month, day, hour, minute)
   sim_start    = round_to_nearest_hour(discovery_dt - timedelta(minutes=30 * offset))
   log_key      = sim_start.strftime("%Y%m%d") + f"_{sim_start.hour:02d}"
   ```

2. Locate ignition point in opt-space:
   ```python
   fire_opt = (fire_row // coverage_w, fire_col // coverage_w)
   ```

3. Find the cluster whose reachable zone contains the fire (or `None`).

4. Dispatch to the appropriate path:
   - **Fire outside all clusters:** run ground-sensor-only detection check.
   - **Fire inside a cluster:** check the routing cache; compute routing if absent; replay simulation.

### 6. Save results

Results for all scenarios and all strategy combinations are collected into a list of dicts and written to a timestamped CSV:

```
benchmark_results_yearly_YYYYMMDD_HHMMSS.csv
```

A summary per strategy (detection rate, mean delta_t) is printed to stdout.

---

## Design Decisions

### A. Time-Aware Burn Map Construction

`static_risk_wfpi_yearly.npy` has shape `(732, H, W)` for 2020 (a leap year, 366 days, 2 frames per day):

| Frame index | Content |
|---|---|
| `2*(doy-1) + 0` | Day-2 forecast — issued the previous day; operationally available before 10 am |
| `2*(doy-1) + 1` | Day-1 forecast — issued the same day at 10 am; available from 10 am onward |

where `doy` is the calendar day-of-year (1 for Jan 1, 366 for Dec 31 in a leap year).

The frame-selection function implemented in the benchmark:

```python
def frame_index(dt: datetime) -> int:
    doy  = dt.timetuple().tm_yday
    half = 0 if dt.hour < 10 else 1
    return 2 * (doy - 1) + half
```

Per-scenario burn map construction:

```python
def build_burn_map(yearly_map, sim_start, num_steps):
    frames = []
    for t in range(num_steps):
        dt = sim_start + timedelta(minutes=30 * t)
        frames.append(yearly_map[frame_index(dt)])
    return np.stack(frames)   # shape (num_steps, H, W)
```

Each 30-minute step independently selects its own frame. Transitions across midnight and across the 10 am boundary are handled naturally — no special-casing is required.

### B. Rounding to Nearest Hour (Log Key)

`sim_start` is obtained by subtracting the scenario offset from the discovery datetime and rounding to the nearest hour:

```python
def round_to_nearest_hour(dt: datetime) -> datetime:
    if dt.minute >= 30:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)
```

The routing log key is then `YYYYMMDD_HH`. Two scenarios whose simulation windows start within ±29 minutes of each other will share a single routing computation. This is acceptable because:

- The routing algorithm operates on the WFPI risk map (a weather-derived index), not on the actual fire spread, so a sub-hour temporal approximation does not compromise correctness.
- The D1/D2 boundary at 10 am is a coarser discontinuity (one frame per half-day) than the rounding error (up to 29 min).
- The `has()` check on `RoutingLog` requires the cached entry to have at least `(offset + N_SCENARIO_DATA_STEPS) * operational_substeps` steps, ensuring a cached entry is never too short for a scenario that needs it.

### C. Drone Clusters

Clustering is done with path-compressed union-find over charging stations:

```python
for i in range(n):
    for j in range(i + 1, n):
        xi, yi = charging_locs_opt[i]
        xj, yj = charging_locs_opt[j]
        if max(abs(xi - xj), abs(yi - yj)) <= max_battery_substeps:
            union(i, j)
```

The L∞ metric (Chebyshev distance) matches the drone mobility model: a drone can move one opt-space cell per substep in any of the 8 compass directions, so `max_battery_substeps` substeps cover an L∞ ball of radius `max_battery_substeps`.

The cluster fingerprint is built from the sorted list of station coordinates:

```python
fp = "_".join(f"{x}-{y}" for x, y in sorted(stations))
```

This string is stable across runs (sort order is deterministic) and uniquely identifies the cluster for log file naming.

### D. Fire-to-Cluster Assignment and Routing Strategy

**Reachability check:**

```python
def fire_cluster(fire_opt, clusters, max_battery_substeps):
    fr, fc = fire_opt
    for cluster in clusters:
        for sx, sy in cluster["stations_opt"]:
            if max(abs(fr - sx), abs(fc - sy)) <= max_battery_substeps:
                return cluster
    return None
```

A fire is reachable by a cluster if its opt-space position is within L∞ distance `max_battery_substeps` of at least one station in the cluster.

**If the fire is outside all clusters:**

No routing computation is performed. However, the scenario is still loaded and ground sensors are checked: if the fire ever reaches a ground sensor cell during the simulation window, a detection is recorded with `device = "ground sensor"`. This avoids the expensive Julia routing call while correctly crediting ground-sensor detections.

**If the fire is inside a cluster:**

1. Look up `log_key` (`YYYYMMDD_HH`) in the cluster's `RoutingLog`.
2. If absent or too short: build the 24-step burn map starting at `sim_start`, rescale it to opt-space, save to `tmp_burnmaps/yearly_{log_key}.npy`, run the routing strategy with cluster-specific parameters (only that cluster's stations and its allocated drones), and cache the result.
3. Replay the cached routing in `run_simulation`.

Cluster-specific routing parameters:

```python
cluster_auto = {
    **base_rescaled_auto,
    "n_drones":                    cluster["n_drones"],
    "n_charging_stations":         len(cluster["stations_opt"]),
    "ground_sensor_locations":     ground_locs_opt,
    "charging_stations_locations": cluster["stations_opt"],
}
```

### E. Routing Log Caching

One `RoutingLog` JSON file exists per `(routing_strategy, cluster_fingerprint)`. The file path pattern is:

```
California2020Dataset/logs/routing_yearly_{StrategyName}_{OH}OH_{RS}RS_cluster_{fingerprint}.json
```

where `{OH}` is `optimization_horizon` and `{RS}` is `reevaluation_step`. The base strategy name is obtained by unwrapping the logged wrapper:

```python
base_routing_cls = wrappers._deep_unwrap(RoutingCls).__name__
```

Each entry inside the JSON is keyed by `YYYYMMDD_HH` and stores:

```json
{
  "YYYYMMDD_HH": {
    "initial_drone_locations": [["charge", [x, y]], ...],
    "actions_history": [
      [["fly", [x, y]], ...],
      [["charge", [x, y]], ...],
      ...
    ]
  }
}
```

`actions_history` has one entry per substep. The total number of substeps computed is:

```
MAX_ROUTING_DATA_STEPS * operational_substeps
= 24 * operational_substeps
```

This covers the worst case: maximum offset (12 data steps) plus the 12-step scenario duration. The `has()` guard checks that the cached entry is at least long enough for the current scenario:

```python
total_substeps_needed = (offset + N_SCENARIO_DATA_STEPS) * operational_substeps
rlog.has(log_key, total_substeps_needed)
```

During simulation replay, actions are consumed sequentially from `action_ptr = 0`; there is no frame-skipping or pointer arithmetic. The simulation begins at `time_step = -starting_time` (i.e., `offset` data steps before fire ignition), so the first `offset * operational_substeps` actions cover drone pre-positioning.

### F. Sensor Placement Manual Cache

`SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation` returns a drone allocation list via `get_drone_allocation()`, an interface not supported by the `make_loggable_sensor_strategy` wrapper in `wrappers.py`. Wrapping the strategy would discard the allocation data.

The solution is a dedicated `load_or_compute_sensor_placement` function that saves and loads the placement results to a hand-crafted JSON file:

```
California2020Dataset/logs/sensor_alloc_{combo_name}_{N}x{M}.json
```

Format:

```json
{
  "ground_sensor_locations":    [[row, col], ...],
  "charging_station_locations": [[row, col], ...],
  "drones_per_charging_station": [int, ...]
}
```

All coordinates are in opt-space. On the first run, the Julia placement solver is called; on subsequent runs the file is loaded directly. This cache is invalidated only by deleting or renaming the file.

### G. Burn Map Alignment Correctness

The simulation time axis is aligned as follows:

| Concept | Value |
|---|---|
| `sim_start` | `discovery_datetime - offset * 30min`, rounded to nearest hour |
| `action_ptr = 0` | Corresponds to `sim_start` |
| `time_step = -offset` | First simulation step; fire has not ignited yet |
| `time_step = 0` | Fire ignites (discovery time) |
| `time_step = 11` | Last scenario step (12 steps total = 6 hours) |

The routing burn map is built from `sim_start` with `MAX_ROUTING_DATA_STEPS = 24` frames. Step `t` in the routing maps to calendar time `sim_start + t * 30min`. The `frame_index` function picks D2 or D1 based on the hour of that calendar time, respecting the 10 am boundary at every step. No frames are ever skipped; the pointer advances by exactly one substep per substep of simulation.

Pre-fire steps (before `time_step = 0`) consume `offset * operational_substeps` actions from the routing log for drone pre-positioning. Fire detection is only checked during steps `time_step >= 0`.

---

## Parameters and Tuning

All numerical parameters are collected in `SIMULATION_PARAMETERS`:

```python
SIMULATION_PARAMETERS = {
    "max_battery_distance": -1,        # -1 = unlimited (distance-based battery disabled)
    "max_battery_time":      1,        # hours; one full charge lasts 1 hour
    "n_drones":              2,        # total drones (overridden per-cluster by allocation)
    "n_ground_stations":     8,        # number of ground sensors placed
    "n_charging_stations":   2,        # number of charging stations placed
    "drone_speed_m_per_min": 600,      # drone cruising speed
    "coverage_radius_m":     2900,     # sensor coverage radius
    "cell_size_m":           1000,     # WFPI grid cell size (~1 km)
    "mask_pooling_mode":     "max",    # how to pool mask when downsampling to opt-space
}
```

**Derived quantities (not tunable directly):**

| Quantity | Formula | Meaning |
|---|---|---|
| `coverage_w` | `round(coverage_radius_m * 2 / cell_size_m)`, odd | Coverage diameter in data cells |
| `operational_substeps` | `compute_operational_substeps(cell_size_m, speed, coverage_r_m)` | Substeps per 30-min data step |
| `rescaled_max_battery` | `max_battery_time * operational_substeps` | Battery in substeps (routing currency) |
| `rescaled_N`, `rescaled_M` | `H // coverage_w`, `W // coverage_w` | Opt-space grid dimensions |

**Routing strategy parameters** (set per entry in `STRATEGY_COMBINATIONS`):

| Parameter | Default | Meaning |
|---|---|---|
| `reevaluation_step` | 5 | Re-optimise routing every N substeps |
| `optimization_horizon` | 10 | Planning horizon in substeps |

**Constants (not in `SIMULATION_PARAMETERS`):**

| Constant | Value | Meaning |
|---|---|---|
| `MAX_ROUTING_DATA_STEPS` | 24 | Burn map steps computed for routing |
| `N_SCENARIO_DATA_STEPS` | 12 | Scenario duration (6 hours at 30-min steps) |

---

## Log File Naming Conventions

| File | Path pattern | Description |
|---|---|---|
| Sensor placement cache | `logs/sensor_alloc_{combo_name}_{N}x{M}.json` | Ground and charging station positions + drone allocation |
| Routing log | `logs/routing_yearly_{StrategyName}_{OH}OH_{RS}RS_cluster_{fingerprint}.json` | Cached routing actions keyed by `YYYYMMDD_HH` |
| Temporary burn maps | `tmp_burnmaps/yearly_{YYYYMMDD_HH}.npy` | Per-hour rescaled burn maps used during routing computation |

Where:
- `{combo_name}` is the strategy combination name (e.g., `GaussianAlloc_TOP`)
- `{N}x{M}` is the opt-space grid size (e.g., `131x91`)
- `{StrategyName}` is the unwrapped routing class name (e.g., `DroneRoutingTOPMasked`)
- `{OH}` and `{RS}` are `optimization_horizon` and `reevaluation_step`
- `{fingerprint}` is the cluster fingerprint (e.g., `3-7_5-12`)

The routing log key inside each JSON file is `YYYYMMDD_HH` where `YYYYMMDD` is the sim_start date and `HH` is the hour (zero-padded, 00–23).

---

## Output CSV Format

Results are saved to `benchmark_results_yearly_{timestamp}.csv`. Each row corresponds to one scenario evaluated under one strategy combination.

| Column | Type | Description |
|---|---|---|
| `strategy_combo` | str | Strategy combination name (from `STRATEGY_COMBINATIONS`) |
| `scenario_name` | str | Fire scenario identifier (stem of the `.npy` file, without `_scenario1`) |
| `date` | str | Fire discovery date (`YYYYMMDD`) |
| `sim_start_hour` | int | Hour of `sim_start` after rounding (0–23) |
| `log_key` | str | Routing cache key (`YYYYMMDD_HH`) |
| `offset` | int | Scenario offset in data steps (1–12) |
| `cluster` | str | Cluster fingerprint, or `"none"` if fire is unreachable |
| `routed` | bool | `True` if drone routing was performed; `False` for sensor-only path |
| `delta_t` | int | Detection delay in data steps relative to ignition; `-1` if undetected |
| `device` | str | Detection device: `"drone"`, `"ground sensor"`, `"charging station"`, or `"undetected"` |
| `fire_size_cells` | int | Number of burning cells at detection time (or at end if undetected) |
| `fire_size_percentage` | float | Burning cells as percentage of total grid area |
| `total_distance_traveled` | int | Total drone displacement in data-space cells (Manhattan distance) |
| `percentage_map_explored` | float | Percentage of distinct data-space cells visited by drones |

A summary is also printed to stdout after saving:

```
  GaussianAlloc_TOP: detection rate=72.3%  mean delta_t=3.41 (detected)  n=1530
```

---

## References

- [04_california_2020_dataset.md](04_california_2020_dataset.md) — California 2020 dataset structure, WFPI maps, scenario format, and creation pipeline
- [12_yearly_wfpi_map_comparison.md](12_yearly_wfpi_map_comparison.md) — Empirical comparison of the yearly time-aware WFPI map against Day-1 and Day-2 baselines
- [11_ignition_point_benchmarking.md](11_ignition_point_benchmarking.md) — Ignition-point-only benchmark design
- [07_benchmarking_pipeline.md](07_benchmarking_pipeline.md) — General benchmarking pipeline architecture
- [05_clustering_and_wrappers.md](05_clustering_and_wrappers.md) — Routing strategy wrappers and logging infrastructure
- [03_operational_scaling.md](03_operational_scaling.md) — Operational space, substeps, and coverage width derivation
- FPA FOD database: Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020. https://doi.org/10.2737/RDS-2013-0009.6
- USGS WFPI Fire Danger Maps: https://firedanger.cr.usgs.gov/apps/staticmaps
