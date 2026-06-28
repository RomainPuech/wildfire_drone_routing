# Codebase Reference — Wildfire Drone Routing

This document is the single authoritative reference for LLM coding agents working on this codebase. It describes the project **as it currently exists**, covering architecture, data flow, every key module, coordinate conventions, and common pitfalls. Read this before touching any file.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Repository Layout](#2-repository-layout)
3. [End-to-End Benchmark Flow](#3-end-to-end-benchmark-flow)
4. [Coordinate Systems (Critical)](#4-coordinate-systems-critical)
5. [Core Python Modules](#5-core-python-modules)
6. [Julia Optimization Modules](#6-julia-optimization-modules)
7. [Strategy Taxonomy](#7-strategy-taxonomy)
8. [Wrappers and Clustering](#8-wrappers-and-clustering)
9. [California Datasets](#9-california-datasets)
10. [Figure and Table Reproduction](#10-figure-and-table-reproduction)
11. [HPC Execution Pattern](#11-hpc-execution-pattern)
12. [Agent Pitfalls](#12-agent-pitfalls)

---

## 1. Project Purpose

This codebase answers the question: **Given a fixed budget, where should charging stations, drones, and ground sensors be deployed across California to detect the most wildfires as quickly as possible?**

The framework jointly optimizes:
- **Sensor placement** — where to put charging stations and ground sensors (solved once per budget level via ILP/greedy, cached to JSON)
- **Drone routing** — how drones patrol among charging stations within each battery cycle (solved per cluster per scenario start-hour, cached to JSON)

It then **simulates** each historical wildfire ignition from 2021–2024 against the placed infrastructure and records whether and when the fire would have been detected.

The paper evaluates five budget levels (\$20M, \$50M, \$75M, \$100M, \$500M), three routing strategies (TOP, MaxCov, LinearMinTime), and four years of fires (n = 3,693 total ignitions). Results are stored as CSVs in `paper/final_report/csv/`.

---

## 2. Repository Layout

```
run_benchmark_california2021_yearly.py ← PRIMARY ENTRY POINT (routing + simulation; produced the paper data)
run_benchmark_california_yearly.py     ← cleaned multi-year convenience wrapper
test_budget_placement_station_max_greedy_uniform_2021.py      ← placement ILP entry point (20/50/100M)
test_budget_placement_station_max_uniform_fixed_drones_2021.py ← placement ILP entry point (500M)
preprocess_benchmark_2021.py        ← one-off rescaling of the 2021 dataset
visualize_sensor_placement_2021.py  ← generate single-panel placement maps

code/
  benchmark.py                  simulation engine, operational scaling, fire detection
  Strategy.py                   all SensorPlacement + DroneRouting strategy classes
  dataset.py                    load/save California scenario .npy files
  displays.py                   publication-style matplotlib figures
  benchmark_alertcalifornia.py  ALERTCalifornia camera baseline evaluation
  wrappers.py                   logging, caching, and clustering wrappers
  new_clustering.py             geographic cluster decomposition
  placement_map_style.py        shared map styling for figures 4 & 6
  plot_alertcalifornia_coverage.py  coverage maps for ALERTCalifornia
  Drone.py                      Drone state machine (position, battery, state)
  my_julia_caller.py            singleton Julia session manager
  video_helpers.py              video generation from action histories
  dataset_creation/
    nature_dataset_creation/    scripts to build California 2021–2024 datasets

julia/
  TOP.jl                        Team Orienteering Problem entry point + ILP fallback
  TOP_PSO_multi_depot.jl        PSO heuristic core (with all boundary optimizations)
  ground_charging_opt.jl        ILP sensor/station placement (JuMP + Gurobi)
  drone_routing_opt.jl          nonlinear routing optimizer (JuMP + Gurobi)
  drone_routing_opt_linear.jl   linear-time routing variant (LinearMinTime strategy)
  helper_functions.jl           load_burn_map(), shared utilities
  run_extreme_tests_simple.jl   test runner
  test_*.jl                     22 unit/benchmark tests

julia_env/
  Project.toml                  Julia 1.11 package list
  Manifest.toml                 exact locked versions

paper/
  Nature_Wildfires/             figure/table scripts + committed figure PNGs (manuscript .tex NOT included)
  figure4/                      Fig 3 generator (deployment composite)
  figure5bis/                   Fig 4 generator (cost-sensitivity lines)
  figure6/                      Fig 5 generator (ALERTCalifornia coverage)
  breakeven_figure/             shared drawing module imported by figure4 & figure5bis
  final_report/
    generate_final_report.py    Fig 2 data computation (detection frontier)
    csv/                        137 pre-computed benchmark CSVs
    placement_data/logs/        4 pre-computed panel placement JSONs for Fig 3 (20/50/100/500M)
  breakeven_report/
    breakeven_sensor_cost_export/placement_logs/  48 JSON files for Fig 4

report/
  benchmark_2021_greedy_kernel/ SLURM submission scripts + pipeline docs (HPC reproduction)

cameras.json                    699 ALERTCalifornia camera positions (lat/lon)
environment.yml                 conda env `juliaenv` (Linux/HPC, Python 3.10)
environment_macos.yml           conda env `wf` (macOS, includes geopandas)
```

**Dataset directories** (not in repo, download from HuggingFace):
```
California2021Dataset/   config_california_2021.json  mask.npy  wfpi_YYYYMMDD.npy×365  scenarii/*.npy
California2022Dataset/   (same structure, fewer WFPI files)
California2023Dataset/
California2024Dataset/
```

---

## 3. End-to-End Benchmark Flow

Understanding this flow is essential before editing any module.

```
run_benchmark_california_yearly.py
│
├─ 1. Load static resources (single-threaded)
│     • static_risk_pyrologix.npy    (N×M Pyrologix risk surface)
│     • mask.npy                     (N×M burnable-land mask)
│     • config_california_YYYY.json  (per-fire ignition timestamps)
│
├─ 2. Operational scaling (benchmark.py)
│     • coverage_width_cells = round(2 × coverage_radius_m / cell_size_m)
│     • operational_substeps = drone_speed × 60 / (cwc × cell_size_m)
│     • rescaled N, M, battery, burn map, mask → saved as .npy files
│
├─ 3. Sensor/charging placement (Strategy.py → julia/ground_charging_opt.jl)
│     • One run per budget level, result cached to California2021Dataset/logs/*.json
│     • Placement is year-independent (Pyrologix is trained on 2006–2020 data)
│     • Output: list of charging station positions + list of ground sensor positions
│
├─ 4. Cluster decomposition (new_clustering.py)
│     • Charging stations → connected components (edge = L∞ dist ≤ max_battery)
│     • Each cluster assigned a proportional number of drones
│     • Cluster fingerprint = sorted station positions
│
├─ 5. Pre-scan all scenarios (single-threaded)
│     • For each scenario: resolve cluster + rounded ignition hour (log_key)
│     • Scenarios sharing (cluster, log_key) → different execution waves
│     • Wave structure prevents concurrent writes to the same routing cache file
│
├─ 6. Parallel execution (ProcessPoolExecutor, spawn context)
│     For each wave:
│       For each scenario in wave (in parallel):
│         a. Drone routing (julia/TOP.jl or drone_routing_opt_linear.jl)
│            → cached per (cluster_fingerprint, YYYYMMDD_HH) in JSON
│         b. Simulation replay (benchmark.py::run_benchmark_scenario)
│            → returns delta_t, device, fire_size_cells
│
└─ 7. Collect results → CSV with columns:
      year, budget, strategy, fire_name, delta_t, device,
      fire_size_cells, reachable, routing_compute_seconds, …
```

**Critical invariant**: Placement JSONs are always read from `California2021Dataset/logs/` regardless of the year being benchmarked. Never move or rename those files.

---

## 4. Coordinate Systems (Critical)

There are three nested coordinate systems. Confusing them is the most common source of bugs.

### Data space (raw grid)
- Dimensions: `N × M` (e.g., 783 × 483 for California)
- Cell size: 500 m (California datasets)
- Origin: top-left corner of the grid
- Used by: fire scenarios (`.npy`), fire detection checks, `Drone` objects internally

### Operational space (coarsened grid)
- Dimensions: `N_op × M_op = (N // cwc) × (M // cwc)`
- `cwc` = `coverage_width_cells` = `round(2 × coverage_radius_m / cell_size_m)`
  - For California: `round(2 × 2900 / 500) = 12` → cwc = 11 (made odd for symmetry)
- One operational cell = one drone coverage area = one step in a routing plan
- Used by: all strategies, Julia optimizers, burn maps passed to Julia
- Drone position in operational space: updated after each substep
- Substeps per data timestep: `operational_substeps ≈ drone_speed_m_min × 60 / (cwc × cell_size_m)`

### Julia space (1-indexed)
- Julia is 1-indexed. All positions passed to Julia must be `(x+1, y+1)`.
- All positions returned from Julia must be `(x-1, y-1)`.
- **Exception**: `move` actions use relative displacements; no index shift needed.

### Conversion formulas

```python
# Operational → Data (center of the block)
data_x = op_x * cwc + cwc // 2
data_y = op_y * cwc + cwc // 2

# Data → Operational
op_x = data_x // cwc
op_y = data_y // cwc

# Operational → Julia
julia_x = op_x + 1
julia_y = op_y + 1
```

### Battery units

- Strategy sees battery in **operational substeps** (integer)
- `rescaled_max_battery = max_battery_time_hours × operational_substeps`
- For California: `max_battery_time = 1 h`, `operational_substeps ≈ 63` → battery = 63 moves
- Each `fly`/`move` action costs 1 battery unit; `charge` at a station restores to full

---

## 5. Core Python Modules

### `code/benchmark.py`

The simulation engine. Key public functions:

```python
run_benchmark_scenario(
    scenario,                          # (T, N, M) fire grid as numpy array
    sensor_placement_strategy,         # class (not instance)
    drone_routing_strategy,            # class (not instance)
    simulation_parameters,             # dict of physical params
    custom_initialization_parameters,  # dict forwarded to strategy __init__
    custom_step_parameters_function,   # () → dict forwarded to next_actions
    input_dir,                         # path to layout/scenario folder
    return_history=False,
    starting_time=0,                   # data timesteps of pre-fire patrol
) → dict                               # delta_t, device, metrics
```

```python
# Scaling utilities used by the benchmark runner
compute_operational_substeps(cell_size_m, drone_speed_m_per_min, coverage_radius_m) → int
pool_burnmap_mean(burnmap, kernel_size) → ndarray
pool_burnmap_proba_at_least_one(burnmap, kernel_size) → ndarray
pool_mask(mask, kernel_size) → ndarray
operational_space_to_dataspace_coordinates(coord, coverage_radius_m, cell_size_m) → tuple
detect_fire_within_coverage(fire_grid, drone_pos, coverage_width_cells) → bool
```

**Fire detection logic**: At each data timestep, ground sensors and charging stations are checked first (point detection), then drones after each substep (area detection over a `cwc × cwc` square). Detection is deterministic within `coverage_radius_m`.

**Simulation timeout**: 24 data timesteps (= 24 hours). Undetected fires get `delta_t = -1`.

### `code/Strategy.py`

Contains all sensor placement and drone routing strategy classes (~234 KB). Key classes used in the paper:

**Sensor placement** (called once per budget, result cached):

| Class | Description |
|---|---|
| `SensorPlacementMaxCoverageGaussianTimeMaskedBudget` | Main production placement. ILP via Julia with budget constraint. Gaussian-weighted time-averaged burn map, burnable-land mask. |
| `SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation` | Like above but also allocates drones per station. Used at \$500M. |
| `RandomSensorPlacementStrategy` | Baseline: uniform random placement |

**Drone routing** (called per cluster per scenario start-hour, result cached):

| Class | Description |
|---|---|
| `DroneRoutingTOPGrowing` | **Primary TOP strategy.** PSO-based, "growing" burn map update. Best overall detection rate. |
| `DroneRoutingMaxCoverageResetStatic` | MaxCov: JuMP optimizer, resets visited cells. |
| `DroneRoutingLinearMinTime` | LinearMinTime: linear routing optimizer, minimizes detection delay. |

**Burn map update patterns** (for TOP strategies):

| Pattern | Behavior |
|---|---|
| `growing` | Zero visited cells permanently; re-add initial burn map each data timestep. Balances exploration and risk-awareness. |
| `growing_proba` | Like `growing` but uses `1 - (1-current)(1-initial)` (probabilistic OR). |
| `fixed_reset` | Zero visited cells for `reset_time` re-evaluation cycles, then revert. |

**Strategy base class interface**:
```python
class SensorPlacementStrategy:
    def __init__(self, automatic_params: dict, custom_params: dict): ...
    def get_locations(self) -> (list[(x,y)], list[(x,y)]):  # ground sensors, charging stations
        ...

class DroneRoutingStrategy:
    def __init__(self, automatic_params: dict, custom_params: dict): ...
    def get_initial_drone_locations(self) -> list[(state, (x,y))]:  # state ∈ {'charge','fly'}
        ...
    def next_actions(self, auto_step_params: dict, custom_step_params: dict) -> list[(action, (x,y))]:
        # action ∈ {'fly', 'move', 'charge'}
        ...
```

`automatic_params` injected by the benchmark engine includes: `N, M, n_drones, n_ground_stations, n_charging_stations, max_battery_time, burnmap_filename, mask_filename, charging_stations_locations, ground_sensor_locations, input_dir`.

### `code/dataset.py`

Data loading for California datasets.

```python
load_scenario_npy(filepath) → ndarray  # (T, N, M) fire grid, T ≤ 24 timesteps
load_config(config_path) → dict         # fire metadata: ignition_time, fire_name, date, …
```

The California dataset format: each scenario is a `(T, N, M)` uint8 array where `1` = burning cell, `0` = not burning. `T` = number of hourly timesteps in the scenario (typically 6).

WFPI burn maps are stored as `(365, N, M)` float32 arrays (one frame per day of the year). The benchmark runner indexes into the correct day using the fire's ignition date from the config JSON.

### `code/displays.py`

Publication-quality figure generation. Key functions:

```python
plot_placement_map(
    placement_data,          # dict with charging_stations, ground_sensors, drones_per_station
    fires_data,              # dict with discoverable/undiscoverable ignition points
    risk_map,                # (N, M) Pyrologix risk surface
    mask,                    # (N, M) burnable-land mask
    california_boundary,     # GeoDataFrame for CA state outline
    budget_label,            # string for panel title
    ax,                      # matplotlib Axes
) → None

plot_alertcalifornia_coverage(
    cameras,                 # list of (lat, lon) camera positions
    fires,                   # list of (lat, lon) fire positions
    radius_km,               # detection radius
    ax,
) → None
```

Styling constants (`_pyrologix_publication_rc()`, shared colormap) are defined here. All paper figures call this module for consistent styling. The `wf` conda environment (macOS) is required for geospatial functions (geopandas, rasterio).

### `code/wrappers.py`

Factory functions that add logging/caching/clustering to any strategy class.

```python
make_loggable_sensor_strategy(cls) → cls   # wraps sensor strategy with JSON caching
make_loggable_drone_strategy(cls) → cls    # wraps drone strategy with JSON caching
get_wrapped_clustering_strategy(cls) → cls  # wraps drone strategy with cluster decomposition
```

The benchmark runner mostly applies clustering directly rather than going through `wrappers.py`, but the functions are still used for precomputation modes. See §8 for details.

### `code/new_clustering.py`

Cluster decomposition for large grids.

```python
class ClusteredDroneStrategyWrapped:
    def find_clusters(self, charging_stations, max_battery_time) → list[list[(x,y)]]
    # BFS on graph where edge(i,j) exists iff L∞_distance(i,j) ≤ max_battery

    def allocate_drones(self, total_drones, clusters) → list[int]
    # Proportional allocation, minimum 1 per cluster
```

The cluster boundary for each cluster is the union of `min(transmission_range, battery/2)` half-extent squares around each station. Ground sensors inside a boundary are assigned to that cluster.

### `code/benchmark_alertcalifornia.py`

Evaluates the ALERTCalifornia camera network against California ignition records.

```python
run_alertcalifornia_benchmark(
    cameras_path,     # path to cameras.json
    fires_data,       # list of fire dicts with lat/lon
    radius_km,        # assumed detection radius
) → dict              # detection_rate, per_year breakdown
```

This is a purely geometric model (no routing). A fire is detected if its ignition coordinates fall within `radius_km` of any camera. Used to generate Table 2 and Figure 6.

### `code/my_julia_caller.py`

Singleton Julia session management.

```python
# At module import time, starts Julia and includes all .jl files:
from my_julia_caller import Main, jl
# Then call Julia functions:
result = Main.compute_TOP_plan_multiple_depots(burnmap_file, ...)
```

**Startup cost**: ~13 s with `python-jl` (recommended), ~80 s with plain `python`. The session is initialized once per process. In `ProcessPoolExecutor` (spawn context), each worker process initializes its own Julia session.

**Files included at startup**: `TOP.jl` (which `include`s `TOP_PSO_multi_depot.jl` and `helper_functions.jl`), `ground_charging_opt.jl`, `drone_routing_opt.jl`, `drone_routing_opt_linear.jl`.

---

## 6. Julia Optimization Modules

### `julia/TOP_PSO_multi_depot.jl` — PSO Core

The main routing algorithm for TOP and MaxCov strategies. Implements Particle Swarm Optimization adapted to the multi-depot Team Orienteering Problem.

**Key entry point** (called from Python):
```julia
compute_TOP_plan_multiple_depots(
    burnmap_filename::String,
    n_drones::Int,
    charging_stations::Vector{Tuple{Int,Int}},  # 1-indexed
    ground_sensors::Vector{Tuple{Int,Int}},     # 1-indexed
    max_battery_time::Int,                      # in substeps
    t::Int,                                     # current timestep index
    verbose::Bool,
    mask_filename::String                        # "" for no mask
) → Vector{Vector{Vector}}  # solution[substep][drone] = (action, (x,y)) 1-indexed
```

**Performance optimizations** (all enabled by default):
- `ENABLE_SHIFT_IRRELEVANCE_FILTER`: skip shift moves that cannot improve objective
- `ENABLE_SWAP_BLOCKING_FILTER`: skip swap moves that cannot improve objective
- `ENABLE_LAZY_DEAD_FILTER`: filter dead drones from swap candidates
- `ENABLE_SPARSE_SPLIT`: O(k·n/k) split instead of O(n²)
- `ENABLE_INCREMENTAL_LOCAL_SEARCH`: incremental update of objective after local moves

The PSO operates on complete drone tours (sequences of depot-to-depot paths). Each particle encodes a complete assignment of cells to drones. Local search operators: shift (move one cell between routes) and swap (exchange cells between routes).

### `julia/ground_charging_opt.jl` — Placement ILP

```julia
NEW_SENSOR_STRATEGY(
    risk_pertime_file::String,
    N_grounds::Int,
    N_charging::Int
) → (ground_locs, charging_locs)  # vectors of (x,y) tuples, 1-indexed

# Budget-constrained variant used in the paper:
NEW_SENSOR_STRATEGY_BUDGET(
    risk_pertime_file::String,
    budget::Float64,
    cost_ground::Float64,
    cost_charging::Float64,
    cost_drone::Float64,
    drones_per_station::Int,
    mask_filename::String
) → (ground_locs, charging_locs, drones_per_station_alloc)
```

Formulated as a set-covering ILP via JuMP + Gurobi. Maximizes the expected wildfire risk covered by the placed infrastructure. The Gaussian time-weighting spreads coverage weight across future timesteps.

### `julia/drone_routing_opt.jl` — Rolling-Horizon JuMP Optimizer

Used by the `DroneRoutingMaxCoverageResetStatic` (MaxCov) strategy. Solves a MILP for drone routing over `optimization_horizon` substeps and re-solves every `reevaluation_step` substeps.

```julia
create_index_routing_model(...) → model   # built once, reused
solve_index_init_routing(model, ...) → solution
solve_index_next_move_routing(model, drone_locs, batteries, ...) → solution
```

### `julia/drone_routing_opt_linear.jl` — LinearMinTime

Used by the `DroneRoutingLinearMinTime` strategy. Assigns each drone to the highest-risk unvisited cells greedily, then solves the routing as a shortest-path problem.

```julia
compute_linear_min_time_plan(
    burnmap_filename::String,
    n_drones::Int,
    charging_stations,
    max_battery_time::Int
) → solution
```

### `julia/helper_functions.jl`

```julia
load_burn_map(filename::String) → (Array{Float32,3}, Vector{String})
# Returns (T×N×M array, []) for .npy files
# Also handles 2D (N×M) static risk maps by repeating along time axis
```

---

## 7. Strategy Taxonomy

The strategies used in the paper (in `code/Strategy.py`):

```
Sensor Placement (paper results; 7 drones/station StationMax family)
├── SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxGreedyUniform   ← \$20M, \$50M, \$100M
│     (Julia: Max_Coverage_Kernel_Masked_Budget_StationMax_GreedyUniform; entry point test_budget_placement_station_max_greedy_uniform_2021.py)
└── SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMax(Uniform)       ← \$500M uses the UniformFixedDrones variant
      (Julia: Max_Coverage_Kernel_Masked_Budget_StationMax_UniformFixedDrones; entry point test_budget_placement_station_max_uniform_fixed_drones_2021.py)
# SensorPlacementMaxCoverageGaussianTimeMaskedBudget (no StationMax) is the older
# budget ILP retained for reference; it is NOT what produced the paper placements.

Drone Routing
├── DroneRoutingTOPGrowing          (TOP)           ← best detection rate
├── DroneRoutingMaxCoverageResetStatic  (MaxCov)    ← JuMP rolling-horizon
└── DroneRoutingLinearMinTime           (LinearMinTime) ← explicit delay minimizer
```

Routing strategy selection in `run_benchmark_california_yearly.py` is via the `--routing` CLI flag (substring match against strategy class name). The mapping:

| `--routing` value | Class |
|---|---|
| `TOPGrowing` | `DroneRoutingTOPGrowing` |
| `MaxCov` | `DroneRoutingMaxCoverageResetStatic` |
| `LinearMinTime` | `DroneRoutingLinearMinTime` |

---

## 8. Wrappers and Clustering

### Logging/Caching Pattern

The benchmark runner does **not** use `wrappers.py` directly for the paper experiments. Instead, it implements its own JSON caching at two levels:

1. **Placement cache**: `California2021Dataset/logs/sensor_alloc_<strategy>_<budget>_<NxM>.json`
   - Format: `{"charging_station_locations": [[x,y], ...], "ground_sensor_locations": [[x,y], ...]}`
   - Read/written by the placement strategy classes in `Strategy.py`

2. **Routing cache**: `California2021Dataset/logs/<cluster_fingerprint>_<YYYYMMDD_HH>_<strategy>.json`
   - Format: nested list `solution[substep][drone] = [action, [x, y]]`
   - Written after the first scenario in a (cluster, start-hour) group; replayed for all subsequent scenarios in that group

### Clustering in the Runner

The clustering in `run_benchmark_california_yearly.py` is implemented inline (not via `new_clustering.py`). The logic:
1. After placement, build a graph of charging stations: connect station `i` to `j` if their operational-space Chebyshev distance ≤ `max_battery_substeps`.
2. Find connected components via BFS → clusters.
3. Allocate drones proportionally (each cluster gets ≥ 1 drone).
4. Each cluster's routing solution is cached independently.

`new_clustering.py` is used only by the `ClusteredDroneStrategyWrapped` class (for precomputation mode benchmarks and the sim2real dataset, not the California paper experiments).

---

## 9. California Datasets

### Config JSON format

Each `config_california_YYYY.json` contains a list of fire records:

```json
[
  {
    "fire_name": "DIXIE_2021-CANPF-000XXX",
    "ignition_time": "2021-07-13 13:00:00",
    "date": "2021-07-13",
    "scenario_file": "scenarii/DIXIE_2021-CANPF-000XXX_scenario1.npy",
    "wfpi_file": "wfpi_20210713.npy",
    "lat": 40.12,
    "lon": -121.45
  },
  ...
]
```

Key fields:
- `ignition_time`: used to select the correct WFPI frame and starting hour for routing
- `scenario_file`: path relative to dataset root for the fire spread grid
- `wfpi_file`: the daily WFPI burn map to use as routing risk input

### Mask format

`mask.npy`: binary `(N, M)` uint8 array. `1` = burnable land, `0` = non-burnable (water, urban, rock). Applied during placement (sensors not placed on masked cells) and routing (drone paths avoid masked cells).

### Static risk map

`static_risk_pyrologix.npy`: `(N, M)` float32 array of the Pyrologix wildfire hazard potential. Values in `[0, 1]`. Used for sensor placement. This is the **same** risk map used for all years (trained on 2006–2020 data → no data leakage).

### WFPI daily burn maps

`wfpi_YYYYMMDD.npy`: `(T_substeps, N_op, M_op)` float32 arrays after operational rescaling. Raw WFPI comes as `(T_data, N, M)` and is preprocessed by `preprocess_benchmark_2021.py`. Each frame represents hourly fire risk for that day. The benchmark selects the frame corresponding to the fire's ignition hour.

---

## 10. Figure and Table Reproduction

Summary of figure generators:

Manuscript figure numbering (the script-internal names predate the final numbering):

| Manuscript figure | Script | Key inputs |
|---|---|---|
| Fig 2 — Detection frontier | `paper/Nature_Wildfires/make_figure3_frontier.py` | `paper/final_report/csv/*.csv` (no datasets) |
| Fig 3 — Deployment maps | `paper/figure4/generate_placement_composite_figure.py` | `paper/final_report/placement_data/logs/*.json`, datasets, CA boundary |
| Fig 4 — Cost sensitivity | `paper/figure5bis/make_figure5bis_breakeven_lines.py` | `paper/breakeven_report/.../placement_logs/*.json`, datasets |
| Fig 5 — ALERTCalifornia | `paper/figure6/generate_alertcalifornia_composite_figure.py` | `cameras.json`, fire configs, CA boundary |
| Fig 6 — Case-study region | `code/dataset_creation/nature_dataset_creation/generate_paper_2021_dataset_explainer.py` | `California2021Dataset/`, Pyrologix GeoTIFF |

`paper/figure4/` and `paper/figure5bis/` import the shared drawing module
`paper/breakeven_figure/generate_breakeven_cost_sensitivity_figures.py`.

**Environment note**: Figs 3, 5, 6 require the geospatial stack (geopandas, rasterio, pyproj — the macOS `wf` env, or add those packages to `juliaenv`) for the California state boundary; without it the outline is omitted but the figure still renders. Fig 2 and Fig 4 work with `juliaenv` (Fig 4 still needs the datasets for the % reachable curve).

**Table scripts** are in `paper/Nature_Wildfires/scripts/`:
- `build_table1_detection.py` → `table1_detection.tex` (main results table)
- `build_table2_alertcalifornia.py` → `table2_alertcalifornia.tex` (camera baseline)
- `collect_runtimes.py` → `methods_runtime_table.tex` (optimizer runtimes)

---

## 11. HPC Execution Pattern

The paper benchmark was run on MIT SuperCloud (SLURM). The submission scripts are
committed under `report/benchmark_2021_greedy_kernel/` (placement array, routing
arrays, \$500M jobs, breakeven sweep, and `reproduce_benchmark_2021_greedy_kernel.sh`).
Each loads `module load anaconda/Python-ML-2025a julia gurobi` and calls `python-jl`.
The typical per-combination pattern:

```bash
# Placement (once per budget; cached to California2021Dataset/logs/)
python-jl test_budget_placement_station_max_greedy_uniform_2021.py --budget 100 --time-limit 43200
# Routing + simulation for one (budget, strategy)
python-jl run_benchmark_california2021_yearly.py --budget 100 --strategy TOPGrowing
```

The runner uses `ProcessPoolExecutor` with `spawn` context (critical for Julia compatibility — `fork` breaks the Julia runtime). Each worker initializes its own Julia session on first use.

**Parallelism model**: scenarios are grouped into waves (same cluster + same routing cache group). Scenarios within a wave are independent and run in parallel. Different waves run sequentially to avoid routing cache race conditions.

**Dry run**: `--dry-run` prints resolved paths and exits without loading data. Useful for debugging path issues on a new cluster.

---

## 12. Agent Pitfalls

**1-indexing boundary**: Julia receives and returns 1-indexed coordinates. Every call site must do `+1` before passing positions to Julia and `-1` after receiving them. `move` actions (relative displacements) are the exception — no shift needed.

**Don't confuse data/operational/Julia spaces**: When editing strategy code, identify which space each variable lives in. `drone_locations` in `next_actions()` is operational space. `drone_locations` in `Drone` objects is data space.

**Placement JSONs in 2021 dir**: All years use placement files from `California2021Dataset/logs/`. If that directory doesn't exist or the JSON is missing, the benchmark re-runs the ILP (slow, ~30 min per budget).

**Burn map shape**: When passed to Julia, burn maps have shape `(T_substeps, N_op, M_op)`. The function `load_burn_map` in `helper_functions.jl` handles both 2D and 3D numpy arrays. Don't pass a raw `(N, M)` array directly to PSO functions — wrap it first or use `np.repeat`.

**`pool_mask` vs `pool_mask_min`**: Two functions exist with similar names. `pool_mask` in `benchmark.py` applies min-pooling. There's no `pool_mask_min` in the public API. Always import from `benchmark.py`.

**Routing cache race condition**: The wave assignment in `run_benchmark_california_yearly.py` ensures scenarios with the same `(cluster, log_key)` never run concurrently. Don't break this invariant if modifying the parallel dispatch logic.

**Julia startup in workers**: In `ProcessPoolExecutor` workers, Julia is initialized lazily on the first routing call. If you add a new Julia function call, make sure `my_julia_caller.py` includes the relevant `.jl` file. If a file is missing from the `include` list, you'll get a `MethodError` at runtime, not at import time.

**Figures 4/6 need geopandas**: Running `generate_placement_composite_figure.py` or `generate_alertcalifornia_composite_figure.py` with `conda activate juliaenv` (not `wf`) will fail with `ModuleNotFoundError: No module named 'geopandas'`. Always use `conda run -n wf python ...` for those two scripts.

**Operational substeps rounding**: `cwc` is made odd (`if cwc % 2 == 0: cwc -= 1`) for symmetry. This means the coverage area is `cwc × cwc` centered exactly on the drone's position. If you change `coverage_radius_m` or `cell_size_m`, recheck that `cwc` is odd and that `operational_substeps` is recalculated consistently everywhere.

**The `--placement-json` flag is a legacy alias**: In `run_benchmark_california_yearly.py`, `--placement-json` resolves to the same default as the automatic placement path. Don't rely on it for new features.
