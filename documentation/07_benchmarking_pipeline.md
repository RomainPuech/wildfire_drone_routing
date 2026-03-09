# Benchmarking Pipeline

This document describes the complete benchmarking flow: from loading scenarios to collecting final metrics. It covers the simulation loop, fire detection logic, multi-scenario evaluation, precomputation, parallel execution, and result aggregation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Single Scenario Benchmark](#2-single-scenario-benchmark)
3. [Fire Detection Logic](#3-fire-detection-logic)
4. [Metrics Collection](#4-metrics-collection)
5. [Multi-Scenario Sequential Benchmark](#5-multi-scenario-sequential-benchmark)
6. [Precomputation Mode](#6-precomputation-mode)
7. [Full Dataset Benchmark](#7-full-dataset-benchmark)
8. [Parallel Execution](#8-parallel-execution)
9. [Result Aggregation](#9-result-aggregation)
10. [Entry Points](#10-entry-points)

---

## 1. Pipeline Overview

The benchmarking pipeline evaluates strategies across multiple fire scenarios and layouts. The hierarchy is:

```
Dataset (e.g., MiniTractDataset/)
  └── Layout (e.g., 0058_03866/)             # Geographic area
        └── Scenario (e.g., 0058_00123.npy)  # One fire simulation
              └── Simulation Loop             # Drone routing + detection
```

The pipeline supports three levels of execution:

| Level | Function | Scope |
|-------|----------|-------|
| Single scenario | `run_benchmark_scenario()` | One fire simulation |
| Sequential (per layout) | `run_benchmark_scenarii_sequential()` | All scenarios in one layout |
| Full dataset | `benchmark_on_sim2real_dataset_precompute_parallel()` | All layouts, all scenarios |

```
benchmark_on_sim2real_dataset_precompute_parallel()
  │
  ├── [Layout 1] ──► process_layout()
  │                    ├── run_benchmark_scenarii_sequential_precompute()
  │                    │     ├── run_drone_routing_strategy()  ← precomputation
  │                    │     └── run_benchmark_scenarii_sequential()
  │                    │           ├── run_benchmark_scenario(scenario_1)
  │                    │           ├── run_benchmark_scenario(scenario_2)
  │                    │           └── ...
  │                    └── Save CSV per layout
  │
  ├── [Layout 2] ──► process_layout()
  │                    └── ...
  └── ...

combine_all_benchmark_results()  ← aggregate CSVs
```

---

## 2. Single Scenario Benchmark

### `run_benchmark_scenario(scenario, sensor_placement_strategy, drone_routing_strategy, ...)`

This is the core simulation function. It runs a single fire scenario and returns detection metrics.

### Full Execution Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 0: Parameter Setup                                        │
│  • get_automatic_layout_parameters(scenario, input_dir, ...)    │
│  • custom_initialization_parameters_function(input_dir)          │
├──────────────────────────────────────────────────────────────────┤
│  Phase 1: Operational Scaling                                    │
│  • coverage_width_cells = round(2 * coverage_radius / cell_size) │
│  • operational_substeps = compute_operational_substeps(...)      │
│  • rescaled_N = N // coverage_width_cells                        │
│  • rescaled_M = M // coverage_width_cells                        │
│  • rescaled_max_battery = max_battery * operational_substeps    │
│  • rescaled_burnmap = pool + repeat + scale                     │
│  • rescaled_mask = pool_mask_min (if available)                  │
├──────────────────────────────────────────────────────────────────┤
│  Phase 2: Sensor Placement (Operational Scale)                   │
│  • sensor_strategy(rescaled_params) → ground_locs_op, cs_locs_op │
│  • Convert to data scale: x_data = x_op * cwc + cwc // 2       │
├──────────────────────────────────────────────────────────────────┤
│  Phase 3: Drone Initialization (Operational Scale)               │
│  • routing_strategy(rescaled_params) → initial drone locs/states │
│  • Create Drone objects in data scale                            │
│  • Initialize metrics (execution_times, visited_cells, etc.)     │
├──────────────────────────────────────────────────────────────────┤
│  Phase 4: Simulation Loop                                        │
│  • For each data timestep t (from -starting_time to min(24,T)):  │
│    │                                                              │
│    │  [Fire Check — Static Sensors] (if t ≥ 0)                   │
│    │  • Check ground sensors against fire grid                    │
│    │  • Check charging stations against fire grid                 │
│    │                                                              │
│    │  [Substep Loop] (operational_substeps iterations)            │
│    │  │                                                           │
│    │  │  1. Strategy computes actions (operational scale)         │
│    │  │  2. Convert actions to data scale                         │
│    │  │  3. Move drones (data scale)                              │
│    │  │  4. Update positions in both scales                       │
│    │  │  5. Update batteries                                      │
│    │  │  6. Track visited cells & distance                        │
│    │  │  7. Check drone fire detection (if t ≥ 0)                 │
│    │  └──────────────────────────────────────────────────────────│
│    └─────────────────────────────────────────────────────────────│
├──────────────────────────────────────────────────────────────────┤
│  Phase 5: Metrics Computation                                    │
│  • delta_t = t_found - starting_time                             │
│  • avg_execution_time = mean(execution_times)                    │
│  • percentage_map_explored = visited / total                     │
│  • Return results dict                                           │
└──────────────────────────────────────────────────────────────────┘
```

### Starting Time and Pre-Fire Patrol

The `starting_time` parameter allows drones to patrol **before the fire starts**. When `starting_time > 0`:

- Time steps range from `-starting_time` to `min(24, T)`
- For `t < 0`: No fire exists, but drones still move and explore
- For `t ≥ 0`: The fire grid is active, and detection checks begin
- The final `delta_t` accounts for the head start: `delta_t = t_found - starting_time`

This models the realistic scenario where drones are already patrolling when a fire ignites.

### Simulation Timeout

The simulation runs for at most **24 data timesteps** (24 hours). If the fire is not detected within that window, `delta_t = -1` and `device = 'undetected'`.

---

## 3. Fire Detection Logic

Fire detection is checked at each data timestep by three types of devices:

### Ground Sensors (Point Detection)

```python
if (grid[rows_ground, cols_ground] == 1).any():
    device = 'ground sensor'
```

Ground sensors detect fire if their exact data-space cell is on fire. This is a **vectorized check** using NumPy fancy indexing for efficiency.

### Charging Stations (Point Detection)

```python
if (grid[rows_charging, cols_charging] == 1).any():
    device = 'charging station'
```

Same logic as ground sensors — charging stations can also detect fires at their location.

### Drones (Area Detection)

```python
def detect_fire_within_coverage(fire_grid, drone_pos, coverage_width_cells):
    coverage_radius_cells = coverage_width_cells // 2
    x, y = drone_pos
    for dx in range(-coverage_radius_cells, coverage_radius_cells + 1):
        for dy in range(-coverage_radius_cells, coverage_radius_cells + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M:
                if fire_grid[nx, ny] == 1:
                    return True
    return False
```

Drones scan a square area of `coverage_width_cells × coverage_width_cells` around their data-space position. If **any** cell in that area is burning, the fire is detected.

### Detection Priority

Static sensors are checked **before** drones (at the start of each data timestep), and drones are checked **after each substep**:

```
for each data timestep t:
    check ground sensors     ← first priority
    check charging stations  ← second priority
    for each substep:
        move drones
        check drones         ← third priority
```

---

## 4. Metrics Collection

### Per-Scenario Metrics

The `run_benchmark_scenario()` function returns a results dictionary:

| Metric | Type | Description |
|--------|------|-------------|
| `delta_t` | int | Time steps to detection (-1 if undetected) |
| `device` | str | Which device detected: `'ground sensor'`, `'charging station'`, `'drone'`, `'undetected'` |
| `avg_execution_time` | float | Mean computation time per substep (seconds) |
| `fire_size_cells` | int | Number of burning cells at detection time |
| `fire_size_percentage` | float | Percentage of grid on fire at detection |
| `percentage_map_explored` | float | Percentage of grid visited by drones |
| `total_distance_traveled` | int | Total Manhattan distance traveled by all drones |
| `substeps_per_timestep` | int | Number of operational substeps per data timestep |

### History Output

If `return_history=True`, the function also returns:
- `drone_locations_history`: List of drone positions at each substep
- `ground_sensor_locations`: Sensor positions
- `charging_stations_locations`: Charging station positions

The history can be in either `'data'` or `'operational'` scale.

---

## 5. Multi-Scenario Sequential Benchmark

### `run_benchmark_scenarii_sequential(input_dir, ...)`

Runs `run_benchmark_scenario()` on all scenarios in a layout folder and aggregates the results.

**Process**:
1. Iterates over all scenario files in the `scenarii/` (NPY) or `Satellite_Images_Mask/` (JPG) folder
2. Loads each scenario
3. Looks up the `starting_time` offset from the config
4. Calls `run_benchmark_scenario()` for each
5. Collects per-scenario results into a list
6. Saves a per-layout CSV file
7. Computes and returns aggregate metrics

### Per-Scenario CSV Output

Each layout produces a CSV file:
```
{layout_id}_benchmark_results{experiment_name}_{sensor_strategy}_{drone_strategy}.csv
```

| Column | Description |
|--------|-------------|
| `sensor_strategy` | Name of the sensor placement strategy |
| `drone_strategy` | Name of the drone routing strategy |
| `layout` | Layout identifier |
| `scenario` | Scenario identifier |
| `delta_t` | Time to detection |
| `device` | Detecting device |
| `execution_time` | Computation time per step |
| `fire_size_cells` | Fire size at detection |
| `fire_percentage` | Fire percentage at detection |
| `map_explored` | Percentage of map explored |
| `total_distance` | Total distance traveled |

### Aggregate Metrics

The function also returns a dictionary of averages:

```python
metrics = {
    "avg_time_to_detection": ...,
    "device_percentages": {"ground sensor": 45.2, "drone": 30.1, ...},
    "avg_execution_time": ...,
    "avg_fire_size": ...,
    "avg_fire_percentage": ...,
    "avg_map_explored": ...,
    "avg_distance": ...,
    "raw_execution_times": [...],
    "raw_fire_sizes": [...],
    "raw_fire_percentages": [...],
    "raw_map_explored": [...],
    "raw_distances": [...],
}
```

---

## 6. Precomputation Mode

### `run_benchmark_scenarii_sequential_precompute(input_dir, ...)`

An optimization for strategies that use logging/caching wrappers. Instead of computing the strategy from scratch on every scenario, it:

1. **Finds the canonical (longest) scenario** in the layout
2. **Runs the full strategy** on that scenario via `run_drone_routing_strategy()`, which triggers the logging wrapper to save the complete action plan
3. **Benchmarks all scenarios** using the cached plan — the logging wrapper replays from the saved JSON file

```python
def run_benchmark_scenarii_sequential_precompute(input_dir, ...):
    # 1. Find the longest scenario (including offset)
    for file in iterable:
        scenario = load(file)
        offset = config.get(scenario_name, 0)
        if len(scenario) + offset > max_length:
            canonical_scenario = scenario
            canonical_offset = offset

    # 2. Precompute: run full strategy on canonical scenario
    precomputing_time_per_step = run_drone_routing_strategy(
        drone_strategy, sensor_strategy,
        max_length, canonical_scenario, ...
    )

    # 3. Benchmark all scenarios using cached actions
    return run_benchmark_scenarii_sequential(
        ..., precomputing_time=precomputing_time_per_step
    )
```

### Why the Longest Scenario?

The logging wrapper caches the entire action history. If we precompute on the longest scenario, the cached plan has enough timesteps to cover all other (shorter) scenarios. The benchmark reads from the same log file for every scenario.

### Execution Time Attribution

When precomputation is used, the execution time recorded for each scenario is the **precomputation time per step** (amortized over the canonical scenario length), not the individual scenario execution time. This reflects the true computational cost of the strategy.

---

## 7. Full Dataset Benchmark

### `benchmark_on_sim2real_dataset_precompute(dataset_folder_name, ...)`

Iterates over all layout folders in a dataset, running the precompute benchmark on each:

```python
for layout_folder in layout_folders:
    # Skip layouts with too many failed scenarios (>20%)
    if failed_percentage > 0.2:
        continue

    # Determine scenario folder
    scenarios_folder = "/scenarii/" if npy else "/Satellite_Images_Mask/"

    # Run benchmark
    metrics = run_benchmark_scenarii_sequential_precompute(
        layout_folder + scenarios_folder, ...
    )
    all_metrics[layout_name] = metrics
```

### Layout Filtering

Layouts are filtered based on:
1. **Skip list**: Manually flagged layouts in `skip_folder_names`
2. **Failure rate**: If `selected_scenarios.txt` reports >20% failed scenario matching, the layout is skipped
3. **Missing folders**: Layouts without a `scenarii/` or `Satellite_Images_Mask/` folder are skipped

---

## 8. Parallel Execution

### `benchmark_on_sim2real_dataset_precompute_parallel(dataset_folder_name, ...)`

Uses `ProcessPoolExecutor` with `spawn` context to benchmark layouts concurrently:

```python
max_workers = min(cpu_count(), 15)
ctx = get_context("spawn")

with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
    futures = {}
    for layout_folder in layout_folders:
        future = executor.submit(process_layout, layout_folder, ...)
        futures[future] = layout_folder

    for future in as_completed(futures):
        layout_name, metrics = future.result(timeout=600)
        if metrics is not None:
            all_metrics[layout_name] = metrics
```

### `process_layout(layout_folder, ...)`

The function executed by each worker process:

1. Resolves the drone strategy class from its string name: `getattr(wrappers, drone_strategy_name)`
2. Determines the scenario folder
3. Calls `run_benchmark_scenarii_sequential_precompute()`
4. Returns `(layout_name, metrics)`

### Key Design Choices

| Choice | Reason |
|--------|--------|
| `spawn` context | Avoids fork-related issues with Julia's runtime |
| String-based strategy names | Strategies must be pickle-serializable; class references aren't |
| 15-worker cap | Prevents overwhelming Julia/Gurobi license limits |
| 10-minute timeout | Prevents hanging on pathological layouts |
| Per-layout parallelism | Each layout is independent (no shared state between layouts) |

---

## 9. Result Aggregation

### `combine_all_benchmark_results(dataset_folder, strategy_name, experiment_name)`

After the parallel benchmark completes, per-layout CSVs are scattered across the dataset. This function collects them:

```python
combined_df = combine_all_benchmark_results(
    "MiniTractDataset/",
    strategy_name="SensorPlacementMaxCoverageGaussianTimeMasked_DroneRoutingMaxCoverageResetStatic",
    experiment_name="SMwhp_parallel"
)
```

**Output**: `results/combined_benchmark_resultsSMwhp_parallel.csv`

The combined CSV has one row per scenario across all layouts, enabling aggregate analysis.

---

## 10. Entry Points

### Command-Line: `all_experiments_parallel.py`

The main entry point for running experiments:

```bash
python all_experiments_parallel.py --ss_prefix S --bm_prefix whp
```

**Arguments**:
- `--ss_prefix`: `S` = `SensorPlacementMaxCoverageGaussianTimeLogged`, `R` = `RandomSensorPlacementStrategyLogged`
- `--bm_prefix`: `whp` / `bm` / `bp` / `ncbm` — selects the burn map file

**Flow**:
1. Parses arguments
2. Selects sensor strategy class
3. Calls `run_all_drone_strategies()` which calls `run_one_drone_strategy()` for each drone strategy
4. Each call runs `benchmark_on_sim2real_dataset_precompute_parallel()`
5. Finally calls `combine_all_benchmark_results()`

### Notebook: `experiments.ipynb`

Interactive exploration with:
- Single-scenario benchmarks with visualization
- Strategy comparison on specific layouts
- Video generation of drone movements

### Quick Test: `quicktest.py`

A minimal smoke test for verifying the pipeline works end-to-end.

### Bash Scripts: `runpara*.sh`

Pre-configured shell scripts for running specific strategy combinations:

| Script | Sensor | Burn Map |
|--------|--------|----------|
| `runparaKwhp.sh` | Max Coverage | WHP |
| `runparaKbm.sh` | Max Coverage | Burn Map |
| `runparaKbp.sh` | Max Coverage | Burn Probability |
| `runparaRwhp.sh` | Random | WHP |
| `runparaRbm.sh` | Random | Burn Map |
| `runparaRbp.sh` | Random | Burn Probability |

---

*Previous: [06 — Drone Simulation](06_drone_simulation.md) · Next: [08 — Visualization](08_visualization.md)*
